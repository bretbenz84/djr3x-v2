"""
Wake word detection using OpenWakeWord ONNX models.

Five models are loaded from assets/models/wake_word/:
  - Dee-Jay_Rex, Hey_DJ_Rex, Hey_rex, Yo_robot  — active in IDLE, QUIET, ACTIVE
  - wakeuprex                                     — active in SLEEP only

Detection runs in a background daemon thread. Missing model files are skipped
with a warning rather than causing startup failures.

Usage:
    from audio import wake_word
    wake_word.start(lambda model_name: print(f"Wake word: {model_name}"))
    # ...
    wake_word.stop()
"""

import logging
import os
import threading
from typing import Callable, Optional

import numpy as np

import config
import state as state_module
from state import State
from audio import stream

_log = logging.getLogger(__name__)

# 80 ms at 16 kHz — OpenWakeWord's preferred sequential frame size.
_CHUNK_SAMPLES = 1280
_CHUNK_SECS = _CHUNK_SAMPLES / config.AUDIO_SAMPLE_RATE

_GENERAL_MODELS = frozenset({"Dee-Jay_Rex", "Hey_DJ_Rex", "Hey_rex", "Yo_robot"})
_SLEEP_MODELS = frozenset({"wakeuprex"})

_oww_model = None
_loaded_models: frozenset[str] = frozenset()

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    global _oww_model, _loaded_models

    try:
        from openwakeword.model import Model
    except ImportError:
        _log.error("openwakeword package not installed — wake word detection disabled.")
        return

    paths = []
    loaded: set[str] = set()
    for name, path in config.WAKE_WORD_MODELS.items():
        if os.path.exists(path):
            paths.append(path)
            loaded.add(name)
        else:
            _log.warning("Wake word model missing, skipping: %s (%s)", name, path)

    if not paths:
        _log.error("No wake word model files found — wake word detection disabled.")
        return

    try:
        _oww_model = Model(wakeword_models=paths, inference_framework="onnx")
        _loaded_models = frozenset(loaded)
        _log.info(
            "Loaded %d wake word model(s): %s",
            len(_loaded_models),
            sorted(_loaded_models),
        )
    except Exception as exc:
        _log.error("Failed to initialise wake word models: %s", exc)
        _oww_model = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_for_state(current_state: State) -> frozenset[str]:
    if current_state in (State.IDLE, State.QUIET, State.ACTIVE):
        return _GENERAL_MODELS & _loaded_models
    if current_state is State.SLEEP:
        return _SLEEP_MODELS & _loaded_models
    return frozenset()  # SHUTDOWN — nothing should fire


def _threshold(model_name: str) -> float:
    return config.WAKE_WORD_THRESHOLDS.get(model_name, config.WAKE_WORD_THRESHOLD)


# ── Detection loop ────────────────────────────────────────────────────────────

def _detection_loop(callback: Callable[[str], None]) -> None:
    _log.info("Wake word detection loop started.")

    while not _stop_event.is_set():
        # Sleep one chunk duration; returns early if stop is requested.
        _stop_event.wait(timeout=_CHUNK_SECS)

        if _oww_model is None:
            continue

        audio = stream.get_audio_chunk(_CHUNK_SECS)
        if len(audio) < _CHUNK_SAMPLES:
            continue  # stream not yet warmed up

        chunk = audio[-_CHUNK_SAMPLES:].astype(np.float32)

        current_state = state_module.get_state()
        active = _active_for_state(current_state)
        if not active:
            continue

        try:
            predictions = _oww_model.predict(chunk)
        except Exception as exc:
            _log.error("Wake word prediction error: %s", exc)
            continue

        for model_name, score in predictions.items():
            if model_name not in active:
                continue
            if score >= _threshold(model_name):
                _log.info("Wake word detected: %s (confidence=%.3f)", model_name, score)
                try:
                    callback(model_name)
                except Exception as exc:
                    _log.error("Wake word callback raised: %s", exc)

    _log.info("Wake word detection loop stopped.")


# ── Public API ────────────────────────────────────────────────────────────────

def start(callback: Callable[[str], None]) -> None:
    """Load models (first call only) and start detection in a background daemon thread.

    callback(model_name) is called on every detection above threshold. Firing
    during speech or audio playback is intentional — the caller decides the response.
    """
    global _thread

    with _lock:
        if _thread is not None and _thread.is_alive():
            _log.warning("Wake word detection is already running.")
            return

        if _oww_model is None:
            _load_models()

        _stop_event.clear()
        _thread = threading.Thread(
            target=_detection_loop,
            args=(callback,),
            daemon=True,
            name="wake-word-detector",
        )
        _thread.start()
        _log.info("Wake word detector started.")


def stop() -> None:
    """Signal the detection thread to stop and wait for it to exit (up to 2 s)."""
    global _thread

    with _lock:
        if _thread is None or not _thread.is_alive():
            return
        _stop_event.set()
        t = _thread

    t.join(timeout=2.0)
    if t.is_alive():
        _log.warning("Wake word detector thread did not stop cleanly.")
    else:
        _log.info("Wake word detector stopped.")

    with _lock:
        _thread = None
