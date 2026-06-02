"""
Wake word detection using OpenWakeWord ONNX models.

Five models are loaded from assets/models/wake_word/:
  - Dee-Jay_Rex, Hey_DJ_Rex, Hey_rex, Yo_robot  — active in IDLE, QUIET, ACTIVE
  - wakeuprex                                     — active in SLEEP

While asleep, all loaded Rex wake models are active so a missed dedicated
"wake up Rex" model cannot strand the droid in sleep mode.

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
import time
import urllib.request
from pathlib import Path
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

# Fallback directory for openWakeWord feature models when the pip package is
# installed without bundled resources (missing resources/models/*.onnx).
_OWW_RESOURCE_DIR = Path(config.WAKE_WORD_MODELS_DIR) / "_openwakeword_resources"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    global _oww_model, _loaded_models

    try:
        import openwakeword as oww_pkg
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

    feature_kwargs = _resolve_feature_model_paths(oww_pkg)

    try:
        _oww_model = Model(
            wakeword_models=paths,
            inference_framework="onnx",
            **feature_kwargs,
        )
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

def _download_oww_file(url: str, dest: Path) -> bool:
    """Download one openWakeWord resource file to dest, atomically."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        return True
    except Exception as exc:
        _log.error("Failed downloading %s: %s", url, exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _resolve_feature_model_paths(openwakeword_pkg) -> dict[str, str]:
    """Return kwargs for Model(...) when package resources are missing.

    openWakeWord expects melspectrogram.onnx and embedding_model.onnx inside the
    package resources directory. Some installs are missing those files; in that
    case we self-heal by downloading them once into assets and passing explicit
    paths to the Model constructor.
    """
    try:
        melspec_default = Path(
            openwakeword_pkg.FEATURE_MODELS["melspectrogram"]["model_path"]
        ).with_suffix(".onnx")
        embedding_default = Path(
            openwakeword_pkg.FEATURE_MODELS["embedding"]["model_path"]
        ).with_suffix(".onnx")
    except Exception:
        return {}

    if melspec_default.exists() and embedding_default.exists():
        return {}

    _OWW_RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    melspec_path = _OWW_RESOURCE_DIR / "melspectrogram.onnx"
    embedding_path = _OWW_RESOURCE_DIR / "embedding_model.onnx"

    targets = [
        (
            melspec_path,
            openwakeword_pkg.FEATURE_MODELS["melspectrogram"]["download_url"].replace(
                ".tflite", ".onnx"
            ),
        ),
        (
            embedding_path,
            openwakeword_pkg.FEATURE_MODELS["embedding"]["download_url"].replace(
                ".tflite", ".onnx"
            ),
        ),
    ]

    for path, url in targets:
        if path.exists():
            continue
        _log.warning(
            "openwakeword package resources missing (%s). Downloading fallback resource to %s",
            path.name,
            path,
        )
        if not _download_oww_file(url, path):
            return {}

    return {
        "melspec_model_path": str(melspec_path),
        "embedding_model_path": str(embedding_path),
    }

def _active_for_state(current_state: State) -> frozenset[str]:
    if current_state in (State.IDLE, State.QUIET, State.ACTIVE):
        return _GENERAL_MODELS & _loaded_models
    if current_state is State.SLEEP:
        return (_SLEEP_MODELS | _GENERAL_MODELS) & _loaded_models
    return frozenset()  # SHUTDOWN — nothing should fire


def _dj_playback_active() -> bool:
    """True when DJ/radio music is playing (lazy import to avoid an import cycle)."""
    try:
        from features import dj as dj_mod
        return bool(dj_mod.is_playing())
    except Exception:
        return False


def _tts_playback_active() -> bool:
    """True while Rex's OWN spoken-audio playback is active (TTS / speech-queue).

    Rex's playback through the speakers bleeds into the mic and acoustically MASKS
    a spoken wake word, so a "hey rex" said to interrupt him scores far lower while
    he's talking than after he stops. We use this to drop the wake threshold during
    his speech so a mid-sentence interrupt can still fire. DJ music is handled
    separately by ``_dj_playback_active()``; ``echo_cancel.is_suppressed()`` is True
    for both, so callers check dj first.
    """
    try:
        from audio import echo_cancel
        return bool(echo_cancel.is_suppressed())
    except Exception:
        return False


def _threshold(model_name: str, *, dj_playing: bool = False, tts_playing: bool = False) -> float:
    base = config.WAKE_WORD_THRESHOLDS.get(model_name, config.WAKE_WORD_THRESHOLD)
    floor = float(getattr(config, "WAKE_WORD_MIN_THRESHOLD", 0.30))
    if dj_playing:
        delta = max(0.0, float(getattr(config, "WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA", 0.15)))
        return max(floor, base - delta)
    if tts_playing:
        # Rex's own voice masks a spoken wake word in the mic — drop the bar so the
        # user can still bark a wake word to interrupt him mid-sentence. Floored at
        # WAKE_WORD_MIN_THRESHOLD; if the masked score never reaches the floor, only
        # real AEC / mic placement / lower playback volume will let it through.
        delta = max(0.0, float(getattr(config, "WAKE_WORD_TTS_PLAYBACK_THRESHOLD_DELTA", 0.15)))
        return max(floor, base - delta)
    return base


# Diagnostic: surface the loudest below-threshold wake score while Rex is playing,
# so masking is measurable (peaks near threshold → raise the playback delta; peaks
# near zero → the voice is fully masked, needs AEC / mic placement / lower volume).
_NEAR_MISS_SCORE_FLOOR = 0.2
_NEAR_MISS_LOG_INTERVAL_SECS = 1.5
_last_near_miss_log_at = 0.0


def _maybe_log_masked_near_miss(model_name: str, score: float, threshold: float, dj_playing: bool) -> None:
    global _last_near_miss_log_at
    if score < _NEAR_MISS_SCORE_FLOOR:
        return
    now = time.monotonic()
    if now - _last_near_miss_log_at < _NEAR_MISS_LOG_INTERVAL_SECS:
        return
    _last_near_miss_log_at = now
    source = "dj" if dj_playing else "tts"
    knob = "DJ" if dj_playing else "TTS"
    _log.info(
        "[wake_diag] masked wake near-miss during %s playback: %s peak=%.3f < threshold=%.3f "
        "— raise WAKE_WORD_%s_PLAYBACK_THRESHOLD_DELTA if peaks sit near threshold",
        source, model_name, score, threshold, knob,
    )


def _to_oww_input(mono: "np.ndarray") -> "np.ndarray":
    """Scale a mono float32 [-1, 1] frame to int16-range PCM for openWakeWord.

    LOAD-BEARING (same fix as ``rex_supervisor._to_oww_input``): openWakeWord's
    melspectrogram front-end is trained on 16-bit PCM (±32767). ``audio.stream``
    returns float32 in [-1, 1]; feeding that raw makes the model see near-silence,
    so every score pins at ~0.001 and the wake word never fires — the real cause of
    "I said the wake word and nothing happened." Scaling to int16 makes a clear wake
    phrase score ~0.9+, which the ``WAKE_WORD_THRESHOLDS`` (0.5) are written for.
    """
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)


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

        chunk = _to_oww_input(audio[-_CHUNK_SAMPLES:])

        current_state = state_module.get_state()
        active = _active_for_state(current_state)
        if not active:
            continue

        try:
            predictions = _oww_model.predict(chunk)
        except Exception as exc:
            _log.error("Wake word prediction error: %s", exc)
            continue

        dj_playing = _dj_playback_active()
        tts_playing = (not dj_playing) and _tts_playback_active()
        best_model: Optional[str] = None
        best_score = 0.0
        best_threshold = 1.0
        fired = False
        for model_name, score in predictions.items():
            if model_name not in active:
                continue
            threshold = _threshold(model_name, dj_playing=dj_playing, tts_playing=tts_playing)
            if score > best_score:
                best_model, best_score, best_threshold = model_name, float(score), threshold
            if score >= threshold:
                fired = True
                _log.info(
                    "Wake word detected: %s (confidence=%.3f%s)",
                    model_name,
                    score,
                    " during-dj" if dj_playing else (" during-tts" if tts_playing else ""),
                )
                try:
                    callback(model_name)
                except Exception as exc:
                    _log.error("Wake word callback raised: %s", exc)
        # While Rex is playing audio, log how close a (masked) wake word got, so the
        # mid-speech-interrupt threshold can be tuned to the room.
        if not fired and (dj_playing or tts_playing) and best_model is not None:
            _maybe_log_masked_near_miss(best_model, best_score, best_threshold, dj_playing)

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


def is_ready() -> bool:
    """True when wake-word models are successfully loaded and usable."""
    return _oww_model is not None and bool(_loaded_models)
