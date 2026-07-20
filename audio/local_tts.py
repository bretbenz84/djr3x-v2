"""
On-device TTS via mlx-audio Qwen3-TTS voice cloning.

This module owns ONLY the model lifecycle and raw synthesis — loading the
Qwen3-TTS weights and turning (text, voice reference) into float32 audio chunks
at 24 kHz. It deliberately does NOT own playback parity (output gate, AEC
suppression, mouth LEDs, servo speech motion, barge-in). That lives in
``audio/tts.py`` so BOTH the ElevenLabs and local backends share one, identical
playback implementation. See ``audio.tts._speak_local``.

Three callers use this engine:
  1. ``--local-tts`` runtime mode (config.LOCAL_TTS_MODE) — Rex's whole voice.
  2. Automatic ElevenLabs→local fallback (config.LOCAL_TTS_FALLBACK_ENABLED).
  3. Impersonation (features/impersonation.py) — an arbitrary cloned voice.

Design notes (verified against the model source, 2026-07-19):
  * Offline: the model is loaded from a project-controlled directory by ABSOLUTE
    path. mlx-audio's get_model_path returns an existing local dir directly and
    never calls the network; we additionally scope HF_HUB_OFFLINE /
    TRANSFORMERS_OFFLINE to the load call (the model's post_load_hook builds a
    transformers AutoTokenizer, whose only theoretical network reach is closed
    off this way) without leaking those flags to the rest of the process.
  * Reference audio is passed as a FILE PATH, never an in-memory array: on the
    path route the model auto-downmixes to mono and resamples to 24 kHz, so a
    16 kHz mono mic capture works unmodified. The array route does NO resampling
    and would produce a chipmunk voice — so we never use it.
  * Model load and generation are serialized behind locks (mlx generation is not
    assumed re-entrant; the speech queue is single-worker anyway).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


class VoiceRef(NamedTuple):
    """A voice the local engine can clone: a reference clip + its transcript."""

    wav_path: str        # absolute path to a short reference WAV (any sample rate)
    ref_text: str        # exact transcript of the reference clip (whitespace-normalized)
    label: str           # "rex" | "person:<id>" | "famous:<slug>" — for logs/telemetry


# ── Model lifecycle ───────────────────────────────────────────────────────────

_model = None
_load_lock = threading.Lock()
_generate_lock = threading.Lock()
_load_failed = False


def _project_root() -> Path:
    return Path(config.__file__).resolve().parent


def _model_dir() -> Path:
    """Absolute dir the Qwen3-TTS weights live in (per active variant)."""
    return (
        _project_root()
        / getattr(config, "QWEN_TTS_MODEL_DIR", "assets/models/qwen_tts")
        / getattr(config, "LOCAL_TTS_MODEL_VARIANT", "1.7B-Base-8bit")
    )


def sample_rate() -> int:
    return int(getattr(config, "LOCAL_TTS_SAMPLE_RATE", 24000))


def unavailable_reason() -> Optional[str]:
    """None when the engine is fully usable; otherwise a short human-readable reason.

    Kept cheap enough to call per-turn: a find_spec (no import) plus two stat()s.
    Startup logs this so a failed --local-tts run explains itself instead of a
    silent fall-through. The sentinel checks BOTH weight files — the top-level
    talker and the nested speech_tokenizer vocoder — since the model loads but
    makes no audio if the vocoder is missing.
    """
    try:
        spec = importlib.util.find_spec("mlx_audio")
    except Exception as exc:
        # Don't swallow this to a bare "unavailable" — a raising find_spec is
        # exactly the kind of thing we need to SEE, not guess at.
        return f"mlx-audio import check errored: {exc!r}"
    if spec is None:
        return "mlx-audio is not installed (run: pip install -r requirements.txt)"
    d = _model_dir()
    if not (d / "model.safetensors").exists():
        return f"model weights not found at {d} (run: python setup_assets.py)"
    if not (d / "speech_tokenizer" / "model.safetensors").exists():
        return (
            f"vocoder weights not found at {d / 'speech_tokenizer'} "
            "(re-run: python setup_assets.py)"
        )
    return None


def is_available() -> bool:
    """True when mlx-audio is importable AND the model weights are fully present."""
    return unavailable_reason() is None


def is_loaded() -> bool:
    return _model is not None


def _load_model():
    """Load the model from the local dir, offline. Scoped HF offline flags so the
    tokenizer build in post_load_hook can't reach the network, without leaking
    the flags to whisper / the rest of the process."""
    model_path = str(_model_dir())
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from mlx_audio.tts.utils import load_model
        return load_model(model_path)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def preload(blocking: bool = True) -> bool:
    """Load the model once (idempotent, thread-safe). Returns True on success.

    Non-blocking mode spawns a daemon loader and returns True immediately (the
    caller only learns it kicked off); use is_loaded() to observe completion.
    """
    global _model, _load_failed
    if _model is not None:
        return True
    if not is_available():
        logger.warning("[local_tts] model not available at %s", _model_dir())
        return False

    if not blocking:
        threading.Thread(
            target=preload, kwargs={"blocking": True},
            daemon=True, name="local-tts-preload",
        ).start()
        return True

    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        t0 = time.monotonic()
        try:
            logger.info(
                "[local_tts] loading %s from %s ...",
                getattr(config, "LOCAL_TTS_MODEL_ID", "?"), _model_dir(),
            )
            _model = _load_model()
            logger.info("[local_tts] model loaded in %.1fs", time.monotonic() - t0)
        except Exception as exc:
            _load_failed = True
            logger.error("[local_tts] model load failed: %s", exc)
            return False

    # Warm the Metal kernels with one tiny throwaway generation so the FIRST real
    # line doesn't pay one-time kernel compilation (~4-5s observed cold). Outside
    # the load lock; the generator MUST be closed to release the generation lock.
    if bool(getattr(config, "LOCAL_TTS_WARMUP_ON_LOAD", True)):
        ref = rex_voice_ref()
        if ref is not None:
            t0 = time.monotonic()
            gen = generate_stream("Ready.", ref)
            try:
                next(gen, None)   # first chunk is enough to compile the kernels
                logger.info("[local_tts] warmed in %.1fs", time.monotonic() - t0)
            except Exception as exc:
                logger.debug("[local_tts] warmup generation skipped: %s", exc)
            finally:
                try:
                    gen.close()
                except Exception:
                    pass
    return True


def _ensure_model():
    if _model is None and not preload(blocking=True):
        raise RuntimeError("local TTS model unavailable")
    return _model


# ── Voice references ──────────────────────────────────────────────────────────

def _read_ref_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def voice_ref_from_files(
    wav_path: str | Path, txt_path: str | Path, label: str
) -> Optional[VoiceRef]:
    """Build a VoiceRef from a wav + its transcript file, or None if either is
    missing / the transcript is empty."""
    wav = Path(wav_path)
    txt = Path(txt_path)
    if not wav.exists() or not txt.exists():
        return None
    try:
        ref_text = _read_ref_text(txt)
    except Exception as exc:
        logger.warning("[local_tts] could not read ref text %s: %s", txt, exc)
        return None
    if not ref_text:
        logger.warning("[local_tts] empty ref text for %s", wav.name)
        return None
    return VoiceRef(str(wav.resolve()), ref_text, label)


def rex_voice_ref() -> Optional[VoiceRef]:
    """Rex's own reference voice (VOICES_DIR/rex/<LOCAL_TTS_VOICE>.{wav,txt})."""
    base = _project_root() / getattr(config, "VOICES_DIR", "assets/voices") / "rex"
    voice = getattr(config, "LOCAL_TTS_VOICE", "RX24-pure")
    return voice_ref_from_files(base / f"{voice}.wav", base / f"{voice}.txt", "rex")


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _split_line(text: str) -> list[str]:
    """Split a long line at sentence boundaries so each segment generates in a
    bounded time (carried over from the POC). Short lines pass through whole."""
    threshold = int(getattr(config, "LOCAL_TTS_SPLIT_THRESHOLD", 120))
    text = " ".join((text or "").split())
    if len(text) <= threshold:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?—])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def generate_stream(text: str, voice_ref: VoiceRef) -> Iterator[np.ndarray]:
    """Yield float32 mono audio chunks at ``sample_rate()`` Hz for ``text`` in
    the reference voice. Segments a long line and streams each segment. The
    generation lock is held for the whole iteration (one synthesis at a time)."""
    if not text or not text.strip():
        return
    model = _ensure_model()
    interval = float(getattr(config, "LOCAL_TTS_STREAMING_INTERVAL", 0.32))
    with _generate_lock:
        for seg in _split_line(text):
            for result in model.generate(
                text=seg,
                ref_audio=voice_ref.wav_path,
                ref_text=voice_ref.ref_text,
                stream=True,
                streaming_interval=interval,
            ):
                chunk = np.ascontiguousarray(
                    np.asarray(result.audio, dtype=np.float32).reshape(-1)
                )
                if chunk.size:
                    yield chunk


def synthesize(text: str, voice_ref: VoiceRef) -> tuple[Optional[np.ndarray], int]:
    """Buffered synthesis: concatenate the whole stream into one array. Used for
    cache prefill. Returns (audio, sample_rate) or (None, sr) on empty/failure."""
    sr = sample_rate()
    chunks = list(generate_stream(text, voice_ref))
    if not chunks:
        return None, sr
    return np.concatenate(chunks), sr
