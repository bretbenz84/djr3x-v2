"""
Sound-event awareness — a real local classifier for what Rex HEARS beyond speech.

YAMNet (Google's AudioSet classifier, Apache-2.0, 3.7M params, ~16MB ONNX) runs on
the same onnxruntime the face stack already uses, classifying the scene-analysis
window into 521 audio classes in ~3-4ms/s of audio on CPU. This module maps those
classes onto a small set of behavior-relevant FAMILIES (dog_bark, doorbell, knock,
laughter, scream, baby_cry, glass_break, bang, siren, alarm, cat) and applies
per-family thresholds + cooldowns, so the scene loop can publish discrete "I just
heard X" events instead of raw class soup.

Design rules:
  • Fail-safe everywhere: a missing/broken model disables the feature and the
    legacy energy heuristics in audio/scene.py carry on unchanged. Never raise
    into the scene loop.
  • The scene loop's self-noise gate (audio/scene._should_skip_cycle) is the ONLY
    thing keeping Rex's own TTS/music out of this classifier — this module never
    runs unless that gate said the window is room audio.
  • Model + class map live in assets/models/yamnet/ (gitignored), downloaded by
    setup_assets.py. Input contract: mono float32 16 kHz waveform, arbitrary
    length; output 0 is [n_frames, 521] scores (one frame per ~0.48s hop).

Toggle with SOUND_AWARENESS_ENABLED. Tunables in config.py (families, thresholds,
cooldowns, priority).
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_session = None            # onnxruntime.InferenceSession | None
_input_name: str = ""
_class_index: dict[str, int] = {}
_load_failed = False
_last_family_at: dict[str, float] = {}


def _model_dir() -> Path:
    configured = getattr(config, "SOUND_EVENT_MODEL_DIR", None)
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "assets" / "models" / "yamnet"


def _enabled() -> bool:
    return bool(getattr(config, "SOUND_AWARENESS_ENABLED", True))


def _load() -> bool:
    """Load the ONNX session + class map once. False (and remembered) on any failure
    so a broken install degrades to the legacy heuristics instead of log-spamming."""
    global _session, _input_name, _class_index, _load_failed
    if _session is not None:
        return True
    if _load_failed:
        return False
    with _lock:
        if _session is not None:
            return True
        if _load_failed:
            return False
        model_path = _model_dir() / "yamnet.onnx"
        map_path = _model_dir() / "yamnet_class_map.csv"
        try:
            if not model_path.exists() or not map_path.exists():
                raise FileNotFoundError(f"missing {model_path.name} or {map_path.name}")
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            with open(map_path, newline="") as fh:
                class_index = {
                    row["display_name"]: int(row["index"])
                    for row in csv.DictReader(fh)
                }
            if not class_index:
                raise ValueError("empty class map")
            _input_name = session.get_inputs()[0].name
            _class_index = class_index
            _session = session
            logger.info(
                "Sound-event classifier loaded (%d classes, %s).",
                len(class_index), model_path,
            )
            return True
        except Exception as exc:
            _load_failed = True
            logger.warning(
                "Sound-event classifier unavailable (%s) — falling back to "
                "energy heuristics only. Run setup_assets.py to fetch the model.",
                exc,
            )
            return False


def available() -> bool:
    """True when enabled and the model is (or can be) loaded."""
    return _enabled() and _load()


def preload() -> None:
    """Optional startup warm: load the session off the first live window."""
    if _enabled():
        _load()


def reset_cooldowns() -> None:
    """Test hook: forget per-family cooldown state."""
    _last_family_at.clear()


def _families() -> dict:
    return getattr(config, "SOUND_EVENT_FAMILY_CLASSES", {}) or {}


def _threshold(family: str) -> float:
    overrides = getattr(config, "SOUND_EVENT_FAMILY_THRESHOLDS", {}) or {}
    default = float(getattr(config, "SOUND_EVENT_DEFAULT_THRESHOLD", 0.45))
    try:
        return float(overrides.get(family, default))
    except Exception:
        return default


def _cooldown_secs() -> float:
    return float(getattr(config, "SOUND_EVENT_FAMILY_COOLDOWN_SECS", 30.0))


def _class_scores(audio: np.ndarray) -> Optional[np.ndarray]:
    """Per-class MAX over the window's frames — a 0.5s bang must not be diluted
    by the quiet rest of a 2s window the way a mean would."""
    if _session is None:
        return None
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))
    min_len = int(sr * float(getattr(config, "SOUND_EVENT_MIN_WINDOW_SECS", 0.975)))
    if wave.size < min_len:
        return None
    scores = _session.run(None, {_input_name: wave})[0]
    if scores is None or len(scores) == 0:
        return None
    return np.max(scores, axis=0)


def classify_events(audio: np.ndarray, *, now: Optional[float] = None) -> list[dict]:
    """Classify one scene window into fresh family events.

    Returns [{"family", "score", "top_class"}] for every configured family whose
    best class score clears its threshold AND whose per-family cooldown has
    elapsed — highest-priority first (config.SOUND_EVENT_PRIORITY order). Firing
    starts the family's cooldown, so a barking dog yields ONE dog_bark event per
    cooldown window, not one per bark. [] when disabled, unavailable, or quiet.
    """
    if not _enabled() or not _load():
        return []
    try:
        maxes = _class_scores(audio)
    except Exception as exc:
        logger.debug("sound-event inference failed: %s", exc)
        return []
    if maxes is None:
        return []
    now = time.monotonic() if now is None else now
    cooldown = _cooldown_secs()
    hits: list[dict] = []
    for family, class_names in _families().items():
        best_score, best_class = 0.0, ""
        for name in class_names:
            idx = _class_index.get(name)
            if idx is None:
                logger.debug("sound-event class %r not in class map — skipped", name)
                continue
            score = float(maxes[idx])
            if score > best_score:
                best_score, best_class = score, name
        if best_score < _threshold(family):
            continue
        last = _last_family_at.get(family, 0.0)
        if (now - last) < cooldown:
            continue
        _last_family_at[family] = now
        hits.append({"family": family, "score": round(best_score, 3), "top_class": best_class})
        logger.info(
            "[sound_event] %s (%s %.2f)", family, best_class, best_score
        )
    priority = list(getattr(config, "SOUND_EVENT_PRIORITY", ()) or ())
    order = {fam: i for i, fam in enumerate(priority)}
    hits.sort(key=lambda h: order.get(h["family"], len(order)))
    return hits
