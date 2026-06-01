#!/usr/bin/env python3
"""
rex_supervisor.py — the always-on "wake up Rex" launcher.

This is a deliberately tiny, dependency-light process meant to run for the whole
login session (via a macOS LaunchAgent). It does ONE thing: listen for the single
wake word "wake up rex" (wakeuprex.onnx) and, when it hears it, launch the full
DJ-R3X controller (main.py in the project venv).

Why a separate process instead of just running main.py at login:
  - The robot stays "off" (no servos waking, no camera, no LLM) until you summon
    it by voice, but the Mac is always ready to listen.
  - "shut down" / "shut down rex" cleanly exits main.py and hands control back
    here, so you can power the droid down without killing this listener.

The coordination that prevents a DOUBLE launch (the tricky case):
  main.py holds a single-instance flock for its entire lifetime — including while
  it is merely ASLEEP (the "go to sleep" state, which only wakes on its own
  internal "wake up rex" detector). This supervisor checks that lock and stays
  DORMANT whenever a controller is alive. So:
    - main.py awake  → lock held → supervisor dormant (main.py owns the mic)
    - main.py asleep → lock held → supervisor dormant (main.py's own wake word
                                    handles waking; we must NOT spawn a 2nd one)
    - no main.py     → lock free → supervisor listens for "wake up rex"
  The flock auto-frees if main.py crashes, so the supervisor resumes on its own.

Only one process listens to the mic at a time, so there is no contention.

Run directly for debugging:
    venv/bin/python rex_supervisor.py
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / "venv" / "bin" / "python"
_WAKE_MODEL = _PROJECT_ROOT / "assets" / "models" / "wake_word" / "wakeuprex.onnx"
_WHISPER_DIR = _PROJECT_ROOT / "assets" / "models" / "whisper"

# 80 ms at 16 kHz — openWakeWord's preferred sequential frame size.
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280
_CHUNK_SECS = _CHUNK_SAMPLES / _SAMPLE_RATE

# How the supervisor decides it heard "wake up rex":
#   transcribe — VAD + local Whisper, match the phrase (same proven path the main
#                app uses to wake from SLEEP; reliable, the default)
#   onnx       — only the wakeuprex.onnx confidence score (lighter, but the model
#                is finicky and was the reason "nothing happened")
#   both       — either one fires
_WAKE_MODE = (os.environ.get("REX_SUPERVISOR_WAKE_MODE", "both").strip().lower())
_DEBUG = os.environ.get("REX_SUPERVISOR_DEBUG", "").strip() in ("1", "true", "True")

# Accumulate up to this many seconds of speech before transcribing a phrase.
_MAX_PHRASE_SECS = 3.0
# Stop accumulating after this much trailing silence.
_SILENCE_TAIL_SECS = 0.5

# Make utils.single_instance importable without importing the heavy project config.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG if _DEBUG else logging.INFO,
    format="%(asctime)s | rex_supervisor | %(levelname)s | %(message)s",
)
log = logging.getLogger("rex_supervisor")

_stop = threading.Event()


# ── Minimal .env reading (no project config import) ────────────────────────────

def _read_env_file() -> dict[str, str]:
    """Parse KEY=VALUE lines from .env without importing the project config.

    The supervisor must start even when apikeys.py / full config would fail, so
    it reads only what it needs (the mic device) straight from .env.
    """
    env: dict[str, str] = {}
    path = _PROJECT_ROOT / ".env"
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    except OSError:
        pass
    return env


def _resolve_input_device(env: dict[str, str]):
    """Resolve a sounddevice input device from .env, else the system default."""
    import sounddevice as sd

    name = (os.environ.get("AUDIO_DEVICE_NAME") or env.get("AUDIO_DEVICE_NAME") or "").strip()
    index_raw = (os.environ.get("AUDIO_DEVICE_INDEX") or env.get("AUDIO_DEVICE_INDEX") or "").strip()

    if name:
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0 and dev.get("name") == name:
                    return idx
        except Exception as exc:
            log.warning("Could not match AUDIO_DEVICE_NAME=%r: %s", name, exc)
    if index_raw:
        try:
            return int(index_raw)
        except ValueError:
            log.warning("AUDIO_DEVICE_INDEX=%r is not an integer; using default.", index_raw)
    return None  # sounddevice picks the default input


# ── Controller liveness ────────────────────────────────────────────────────────

def _controller_running(child: Optional[subprocess.Popen]) -> bool:
    """True if a DJ-R3X controller is alive (our child, or any lock holder)."""
    if child is not None and child.poll() is None:
        return True
    try:
        from utils import single_instance
        return single_instance.is_held_by_other()
    except Exception as exc:
        log.debug("single_instance check failed: %s", exc)
        return False


def _launch_controller() -> Optional[subprocess.Popen]:
    """Start main.py in the project venv as a detached child."""
    if not _VENV_PYTHON.exists():
        log.error("venv python not found at %s — cannot launch controller.", _VENV_PYTHON)
        return None
    log.info("Wake word heard — launching DJ-R3X controller.")
    try:
        return subprocess.Popen(
            [str(_VENV_PYTHON), str(_PROJECT_ROOT / "main.py")],
            cwd=str(_PROJECT_ROOT),
        )
    except Exception as exc:
        log.error("Failed to launch controller: %s", exc)
        return None


# ── Wake-word model ────────────────────────────────────────────────────────────

def _feature_model_kwargs() -> dict:
    """Point openWakeWord at the repo's bundled feature models when the pip
    package is missing its own (same self-heal as audio/wake_word.py)."""
    try:
        import openwakeword as oww
        melspec_default = Path(
            oww.FEATURE_MODELS["melspectrogram"]["model_path"]
        ).with_suffix(".onnx")
        embedding_default = Path(
            oww.FEATURE_MODELS["embedding"]["model_path"]
        ).with_suffix(".onnx")
        if melspec_default.exists() and embedding_default.exists():
            return {}
    except Exception:
        pass

    res = _PROJECT_ROOT / "assets" / "models" / "wake_word" / "_openwakeword_resources"
    melspec = res / "melspectrogram.onnx"
    embedding = res / "embedding_model.onnx"
    if melspec.exists() and embedding.exists():
        return {
            "melspec_model_path": str(melspec),
            "embedding_model_path": str(embedding),
        }
    return {}


def _load_model():
    """Load ONLY the wakeuprex model via openWakeWord."""
    try:
        from openwakeword.model import Model
    except ImportError:
        log.error("openwakeword not installed in venv — supervisor cannot listen.")
        return None
    if not _WAKE_MODEL.exists():
        log.error("Wake model missing: %s", _WAKE_MODEL)
        return None
    try:
        return Model(
            wakeword_models=[str(_WAKE_MODEL)],
            inference_framework="onnx",
            **_feature_model_kwargs(),
        )
    except Exception as exc:
        log.error("Failed to initialise wakeuprex model: %s", exc)
        return None


def _wake_threshold() -> float:
    try:
        return float(os.environ.get("REX_SUPERVISOR_WAKE_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


# ── VAD + transcription wake path (mirrors the main app's SLEEP wake) ───────────
# The custom wakeuprex.onnx model is unreliable on its own (it's why "nothing
# happened"). The full robot doesn't trust it either: from SLEEP it wakes by
# running VAD + local Whisper and matching the phrase. We reuse that approach
# here, self-contained (no project `config` import), behind _WAKE_MODE.

import re as _re

# "wake up rex" / "rex wake up" and close variants — same shapes the main app
# accepts (intelligence.interaction._is_sleep_wake_transcript).
_REX_NAME = r"(?:d\s*j\s+)?(?:rex|r\s*3\s*x|r3x|rx)"
_WAKE_RE_A = _re.compile(rf"^(?:(?:hey|yo)\s+)?(?:please\s+)?wake\s+up\s+{_REX_NAME}(?:\s+please)?$")
_WAKE_RE_B = _re.compile(rf"^(?:(?:hey|yo)\s+)?(?:please\s+)?{_REX_NAME}\s+wake\s+up(?:\s+please)?$")
_WAKE_COMPACT = {"wakeuprex", "wakeupr3x", "wakeuprx", "rexwakeup", "r3xwakeup", "rxwakeup"}


def _transcript_is_wake_phrase(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False
    compact = _re.sub(r"[^a-z0-9]", "", raw)
    if compact in _WAKE_COMPACT:
        return True
    cleaned = _re.sub(r"[^a-z0-9]+", " ", raw).strip()
    return bool(_WAKE_RE_A.fullmatch(cleaned) or _WAKE_RE_B.fullmatch(cleaned))


_vad = None  # cached silero VAD callable


def _load_vad():
    """Load Silero VAD (lazily). Returns a get_speech_timestamps-style callable or None."""
    global _vad
    if _vad is not None:
        return _vad
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
        model = load_silero_vad()
        _vad = (get_speech_timestamps, model)
        return _vad
    except Exception as exc:
        log.warning("Silero VAD unavailable (%s) — transcription wake disabled.", exc)
        return None


def _chunk_has_speech(samples) -> bool:
    vad = _load_vad()
    if vad is None:
        return False
    get_speech_timestamps, model = vad
    try:
        import numpy as np
        segments = get_speech_timestamps(
            samples.astype(np.float32),
            model,
            threshold=0.5,
            sampling_rate=_SAMPLE_RATE,
        )
        return bool(segments)
    except Exception as exc:
        log.debug("VAD check failed: %s", exc)
        return False


def _transcribe(samples) -> str:
    """Transcribe a mono float32 [-1,1] array with the local mlx-whisper model."""
    try:
        import mlx_whisper
    except Exception as exc:
        log.warning("mlx_whisper unavailable (%s) — transcription wake disabled.", exc)
        return ""
    if not (_WHISPER_DIR / "config.json").exists():
        log.warning("Local Whisper model missing at %s — run setup_assets.py.", _WHISPER_DIR)
        return ""
    try:
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=str(_WHISPER_DIR),
            language="en",
            fp16=True,
        )
        return str(result.get("text", "") if isinstance(result, dict) else "").strip()
    except Exception as exc:
        log.warning("Transcription failed: %s", exc)
        return ""


# ── Main loop ──────────────────────────────────────────────────────────────────

def run() -> int:
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    model = _load_model()
    if model is None:
        return 1

    env = _read_env_file()
    threshold = _wake_threshold()

    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        log.error("Audio stack unavailable (%s) — supervisor cannot run.", exc)
        return 1

    device = _resolve_input_device(env)
    log.info(
        "Supervisor online. Listening for 'wake up rex' "
        "(device=%s, mode=%s, onnx_threshold=%.2f, debug=%s).",
        device if device is not None else "default", _WAKE_MODE, threshold, _DEBUG,
    )
    use_onnx = _WAKE_MODE in ("onnx", "both")
    use_transcribe = _WAKE_MODE in ("transcribe", "both")
    if use_transcribe:
        _load_vad()  # warm VAD now so the first phrase isn't slow

    child: Optional[subprocess.Popen] = None
    stream = None
    listening = False

    # Diagnostics + transcription accumulation state.
    peak_score = 0.0
    last_diag = 0.0
    speech_frames: list = []          # accumulated chunks during a spoken phrase
    silence_run = 0.0                 # trailing silence while accumulating

    def _open_stream():
        s = sd.InputStream(
            device=device,
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_CHUNK_SAMPLES,
        )
        s.start()
        return s

    def _fire(reason: str):
        nonlocal stream, listening, child
        log.info("Wake detected (%s) — launching controller.", reason)
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
            stream = None
        listening = False
        child = _launch_controller()
        _stop.wait(3.0)  # let main.py take the lock so we don't double-fire

    try:
        while not _stop.is_set():
            running = _controller_running(child)

            # Reap a finished child so the lock check is the single source of truth.
            if child is not None and child.poll() is not None:
                log.info("Controller exited (code=%s). Resuming wake-word listening.", child.returncode)
                child = None
                running = _controller_running(None)

            if running:
                # Dormant: release the mic so the controller owns it, and poll.
                if stream is not None:
                    try:
                        stream.stop(); stream.close()
                    except Exception:
                        pass
                    stream = None
                if listening:
                    log.info("Controller is running — supervisor dormant (mic released).")
                    listening = False
                speech_frames.clear(); silence_run = 0.0
                _stop.wait(1.0)
                continue

            # Active: ensure the mic stream is open and scan for the wake word.
            if stream is None:
                try:
                    stream = _open_stream()
                    model.reset()
                except Exception as exc:
                    log.error("Could not open mic (%s) — retrying in 2s.", exc)
                    _stop.wait(2.0)
                    continue
            if not listening:
                log.info("No controller running — listening for 'wake up rex'.")
                listening = True

            try:
                audio, _ = stream.read(_CHUNK_SAMPLES)
            except Exception as exc:
                log.warning("Mic read failed (%s) — reopening.", exc)
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
                stream = None
                continue

            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0

            # ── Path 1: openWakeWord confidence score ──────────────────────────
            if use_onnx:
                try:
                    scores = model.predict(samples)
                    score = max(scores.values()) if scores else 0.0
                except Exception as exc:
                    log.warning("Wake prediction error: %s", exc)
                    score = 0.0
                peak_score = max(peak_score, score)
                if score >= threshold:
                    _fire(f"onnx score={score:.3f}")
                    peak_score = 0.0
                    speech_frames.clear(); silence_run = 0.0
                    continue

            # ── Path 2: VAD-gated local transcription ──────────────────────────
            if use_transcribe:
                voiced = _chunk_has_speech(samples)
                if voiced:
                    speech_frames.append(samples)
                    silence_run = 0.0
                elif speech_frames:
                    silence_run += _CHUNK_SECS
                # Cap the phrase length so we don't accumulate forever.
                phrase_secs = len(speech_frames) * _CHUNK_SECS
                phrase_done = speech_frames and (
                    silence_run >= _SILENCE_TAIL_SECS or phrase_secs >= _MAX_PHRASE_SECS
                )
                if phrase_done:
                    phrase = np.concatenate(speech_frames)
                    speech_frames.clear(); silence_run = 0.0
                    text = _transcribe(phrase)
                    if text:
                        match = _transcript_is_wake_phrase(text)
                        log.info("Heard %.1fs of speech → %r (wake=%s)",
                                 phrase_secs, text, match)
                        if match:
                            _fire(f"transcript={text!r}")
                            peak_score = 0.0
                            continue

            # ── Periodic diagnostics ───────────────────────────────────────────
            now = time.monotonic()
            if now - last_diag >= 5.0:
                last_diag = now
                if use_onnx:
                    log.info("[diag] listening… peak onnx score (last 5s)=%.3f, mic rms=%.4f",
                             peak_score, rms)
                else:
                    log.info("[diag] listening… mic rms=%.4f", rms)
                peak_score = 0.0
    finally:
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
        log.info("Supervisor stopping (controller left running: %s).",
                 child is not None and child.poll() is None)

    return 0


if __name__ == "__main__":
    sys.exit(run())
