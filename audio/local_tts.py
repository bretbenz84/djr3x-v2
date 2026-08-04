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
import queue
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


def unavailable_reason(require_rex_ref: bool = False) -> Optional[str]:
    """None when the engine is fully usable; otherwise a short human-readable reason.

    Kept cheap enough to call per-turn: a find_spec (no import) plus a few stat()s.
    Startup logs this so a failed --local-tts run explains itself instead of a
    silent fall-through. The sentinel checks BOTH weight files — the top-level
    talker and the nested speech_tokenizer vocoder — since the model loads but
    makes no audio if the vocoder is missing.

    require_rex_ref=True additionally checks Rex's OWN reference clip — required
    to speak in Rex's local voice (--local-tts / fallback), but NOT for
    impersonation, which brings its own VoiceRef; hence opt-in.
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
    if require_rex_ref and rex_voice_ref() is None:
        voice = getattr(config, "LOCAL_TTS_VOICE", "RX24-pure")
        base = _project_root() / getattr(config, "VOICES_DIR", "assets/voices") / "rex"
        return (
            f"Rex voice reference missing: expected {base / voice}.wav + .txt "
            "(tracked in git — git pull, or restore assets/voices/rex/)"
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
    the flags to whisper / the rest of the process. Holds MLX_LOCK: the load is
    itself MLX/Metal compute and must not overlap a whisper transcription."""
    from utils.mlx_lock import MLX_LOCK

    model_path = str(_model_dir())
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from mlx_audio.tts.utils import load_model
        with MLX_LOCK:
            model = load_model(model_path)
            try:
                import mlx.core as mx
                mx.synchronize()   # drain load-time Metal work before releasing
            except Exception:
                pass
            return model
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


def _segment_chunks(
    model, seg: str, voice_ref: VoiceRef, interval: float
) -> Iterator[np.ndarray]:
    """Yield the streamed audio chunks for ONE segment. Caller must already hold
    ``_generate_lock``.

    Locking discipline (this is what prevents the fatal MLX GIL crash observed
    live 2026-07-19): the process-wide MLX_LOCK is held around EACH compute step
    (one streamed chunk, ~0.3 s) — and released between chunks, where the caller
    is busy writing audio to the device — so an mlx_whisper transcription on
    another thread interleaves between chunks instead of colliding with a
    concurrent MLX evaluation (native crash) or stalling for a whole utterance.
    """
    from utils.mlx_lock import MLX_LOCK

    with MLX_LOCK:   # generator construction may already run MLX setup
        gen = model.generate(
            text=seg,
            ref_audio=voice_ref.wav_path,
            ref_text=voice_ref.ref_text,
            stream=True,
            streaming_interval=interval,
        )
    try:
        while True:
            with MLX_LOCK:   # one compute step per acquisition
                try:
                    result = next(gen)
                except StopIteration:
                    break
                # The numpy conversion EVALUATES the lazy mx.array — that's MLX
                # compute; keep it under the lock.
                chunk = np.ascontiguousarray(
                    np.asarray(result.audio, dtype=np.float32).reshape(-1)
                )
            if chunk.size:
                yield chunk
    finally:
        # Close under the lock too — teardown of a half-consumed MLX generator
        # (barge-in mid-line) is also MLX work. Then DRAIN all pending Metal work
        # (mx.synchronize) before releasing: the second dev-mac crash
        # (2026-07-19) fired right AFTER a generation finished, from a Metal-side
        # thread with no Python thread state — async work must not outlive the
        # generation.
        with MLX_LOCK:
            try:
                gen.close()
            except Exception:
                pass
            try:
                import mlx.core as mx
                mx.synchronize()
            except Exception:
                pass


def generate_stream(text: str, voice_ref: VoiceRef) -> Iterator[np.ndarray]:
    """Yield float32 mono audio chunks at ``sample_rate()`` Hz for ``text`` in
    the reference voice. Segments a long line and streams each segment.

    ``_generate_lock`` is held for the WHOLE iteration, serializing synthesis
    against synthesis (see _segment_chunks for the MLX_LOCK discipline inside).
    """
    if not text or not text.strip():
        return
    model = _ensure_model()
    interval = float(getattr(config, "LOCAL_TTS_STREAMING_INTERVAL", 0.32))
    with _generate_lock:
        for seg in _split_line(text):
            yield from _segment_chunks(model, seg, voice_ref, interval)


def synthesize(text: str, voice_ref: VoiceRef) -> tuple[Optional[np.ndarray], int]:
    """Buffered synthesis: concatenate the whole stream into one array. Used for
    cache prefill. Returns (audio, sample_rate) or (None, sr) on empty/failure."""
    sr = sample_rate()
    chunks = list(generate_stream(text, voice_ref))
    if not chunks:
        return None, sr
    return np.concatenate(chunks), sr


# ── Pipelined takes (cloned-voice playback) ──────────────────────────────────
# A cloned parody line is the LONGEST text the local engine ever gets, and
# chunk-level streaming stuttered whenever generation ran slower than real time
# (field 2026-08-01: 0.25 s of preroll is no match for a 12 s line). Rendering
# the WHOLE take first killed the stutter but paid for it in dead time — the
# room waited on every sentence before hearing any of them.
#
# A Take splits the line into SENTENCES and pipelines them: sentence 1 starts
# playing the moment it exists while sentence 2 is already rendering behind it,
# so the wait is one sentence of synthesis instead of the whole take, and each
# unit is fully buffered before it plays (no underrun inside a sentence).
#
# Takes are never cached. A Take is a live one-shot object — the player pops it
# and closes it when playback ends, so the same request always renders fresh
# audio. (config.LOCAL_TTS_CACHE_ENABLED only ever covered Rex's OWN voice.)


def _split_take(text: str) -> list[str]:
    """Split a take into sentence units for the pipeline. Units shorter than
    LOCAL_TTS_TAKE_MIN_CHARS are merged into the sentence that FOLLOWS them —
    a two-word fragment ("Six!") is not worth its own generation, and the model
    reads a bare exclamation better with its follow-on attached."""
    text = " ".join((text or "").split())
    if not text:
        return []
    floor = int(getattr(config, "LOCAL_TTS_TAKE_MIN_CHARS", 24))
    parts = [p.strip() for p in re.split(r"(?<=[.!?…—])\s+", text) if p.strip()]
    units: list[str] = []
    for part in parts:
        if units and len(units[-1]) < floor:
            units[-1] = f"{units[-1]} {part}"
        else:
            units.append(part)
    return units or [text]


def _synthesize_unit(text: str, voice_ref: VoiceRef) -> Optional[np.ndarray]:
    """Render ONE pipeline unit to a contiguous array. Acquires _generate_lock
    for this unit only — never for the whole take — so a take in flight can't
    lock another speaker out for its full duration."""
    model = _ensure_model()
    interval = float(getattr(config, "LOCAL_TTS_STREAMING_INTERVAL", 0.32))
    with _generate_lock:
        chunks = list(_segment_chunks(model, text, voice_ref, interval))
    if not chunks:
        return None
    return np.concatenate(chunks)


class Take:
    """A sentence-pipelined clone take, rendering on a background thread.

    ``first_ready`` fires once the first unit is playable (or the take has given
    up entirely — check ``failed``). ``stream()`` yields the finished units in
    order, padding with short silences while the next one renders so the
    caller's output stream never underruns at a seam.
    """

    def __init__(self, text: str, voice_ref: VoiceRef, *, lookahead: int = 1):
        self.text = " ".join((text or "").split())
        self.voice_ref = voice_ref
        self.first_ready = threading.Event()
        self._units = _split_take(self.text)
        self._queue: "queue.Queue" = queue.Queue(maxsize=max(1, int(lookahead)))
        self._stop = threading.Event()
        self._done = threading.Event()
        self._failed = False
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._produce, daemon=True, name="local-tts-take"
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        """True when the take produced nothing at all (only meaningful once
        first_ready is set — which the producer also sets on giving up)."""
        return self._failed

    def _produce(self) -> None:
        rendered = 0
        try:
            for unit in self._units:
                if self._stop.is_set():
                    break
                try:
                    audio = _synthesize_unit(unit, self.voice_ref)
                except Exception as exc:
                    logger.warning("[local_tts] take unit failed: %s", exc)
                    audio = None
                if audio is None or not len(audio):
                    continue
                rendered += 1
                if rendered == 1:
                    logger.info(
                        "[local_tts] take unit 1/%d ready in %.1fs (voice=%s)",
                        len(self._units), time.monotonic() - self._started_at,
                        getattr(self.voice_ref, "label", "?"),
                    )
                # Bounded queue = the lookahead. Poll rather than block forever
                # so close() during playback tears the producer down promptly,
                # and give up entirely if nobody is draining — a take whose line
                # was dropped before playback (shutdown, a busy output gate)
                # must not leave a thread spinning for the rest of the session.
                abandon_at = time.monotonic() + float(
                    getattr(config, "LOCAL_TTS_TAKE_ABANDON_SECS", 120.0)
                )
                while not self._stop.is_set():
                    try:
                        self._queue.put(audio, timeout=0.25)
                        break
                    except queue.Full:
                        if time.monotonic() >= abandon_at:
                            logger.warning(
                                "[local_tts] take abandoned — nothing consumed it (voice=%s)",
                                getattr(self.voice_ref, "label", "?"),
                            )
                            self._stop.set()
                        continue
                self.first_ready.set()
        finally:
            self._failed = rendered == 0
            if self._failed:
                logger.warning("[local_tts] take produced no audio (voice=%s)",
                               getattr(self.voice_ref, "label", "?"))
            self._done.set()
            self.first_ready.set()

    def stream(self) -> Iterator[np.ndarray]:
        """Yield each finished unit in order. While the producer is still
        working on the next one, yield short silences instead: the caller writes
        them straight to the device, so a slow sentence costs a clean gap rather
        than an underrun. Nothing is emitted before the FIRST real unit — the
        caller's preroll must fill with audio, not silence."""
        fill_ms = float(getattr(config, "LOCAL_TTS_TAKE_FILL_MS", 120.0))
        fill_ms = max(20.0, fill_ms)
        fill = np.zeros(int(sample_rate() * fill_ms / 1000.0), dtype=np.float32)
        played_any = False
        try:
            while True:
                try:
                    item = self._queue.get(timeout=fill_ms / 1000.0)
                except queue.Empty:
                    if self._done.is_set() and self._queue.empty():
                        break
                    if played_any:
                        yield fill
                    continue
                played_any = True
                yield item
        finally:
            self.close()

    def close(self) -> None:
        """Stop rendering and release the producer. Idempotent."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# One-shot parking slot. The impersonation flow starts a take behind Rex's intro
# line and the player picks it up by (text, voice); anything left unclaimed is
# closed rather than kept, so a stale take can never be played back later.
_pending_takes: "dict[tuple[str, str], Take]" = {}
_take_lock = threading.Lock()


def _take_key(text: str, voice_ref: VoiceRef) -> "tuple[str, str]":
    return (" ".join((text or "").split()), getattr(voice_ref, "label", "") or "")


def start_take(text: str, voice_ref: VoiceRef, *, lookahead: int = 1) -> Take:
    """Begin rendering a take NOW and park it for the player to claim.

    Only one take is ever parked — starting a new one closes whatever was left
    behind (an abandoned bit, a barge-in). NOTE: units share _generate_lock with
    live synthesis, so in --local-tts mode start this only when nothing else
    needs the engine (features/impersonation.py orders around that).
    """
    take = Take(text, voice_ref, lookahead=lookahead)
    with _take_lock:
        stale = list(_pending_takes.values())
        _pending_takes.clear()
        _pending_takes[_take_key(text, voice_ref)] = take
    for old in stale:
        old.close()
    return take


def pop_take(text: str, voice_ref: VoiceRef) -> Optional[Take]:
    """One-shot claim of a parked take, or None. Never returns the same take
    twice — a repeated line is rendered fresh."""
    with _take_lock:
        return _pending_takes.pop(_take_key(text, voice_ref), None)


def discard_takes() -> None:
    """Close and drop every parked take (abandoned bit, shutdown)."""
    with _take_lock:
        takes = list(_pending_takes.values())
        _pending_takes.clear()
    for take in takes:
        take.close()
