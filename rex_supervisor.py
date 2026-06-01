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

# 80 ms at 16 kHz — openWakeWord's preferred sequential frame size.
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280

# Make utils.single_instance importable without importing the heavy project config.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
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
        "Supervisor online. Listening for 'wake up rex' (device=%s, threshold=%.2f).",
        device if device is not None else "default", threshold,
    )

    child: Optional[subprocess.Popen] = None
    stream = None
    listening = False

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
            try:
                scores = model.predict(samples)
            except Exception as exc:
                log.warning("Wake prediction error: %s", exc)
                continue

            score = max(scores.values()) if scores else 0.0
            if score >= threshold:
                log.info("Detected 'wake up rex' (confidence=%.3f).", score)
                # Drop the mic before launching so the controller gets it cleanly.
                if stream is not None:
                    try:
                        stream.stop(); stream.close()
                    except Exception:
                        pass
                    stream = None
                listening = False
                child = _launch_controller()
                # Give main.py a moment to take the lock so we don't double-fire.
                _stop.wait(3.0)
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
