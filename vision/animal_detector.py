"""
vision/animal_detector.py — Local pet/animal detection via MediaPipe.

This is the no-OpenAI-credits path for live "a dog wandered into frame" style
awareness. It consumes the same camera frame buffer as face expression telemetry,
but uses MediaPipe Object Detector instead of Face Landmarker.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config
from vision.image_utils import bgr_to_rgb

_log = logging.getLogger(__name__)

_detector = None
_mp = None
_load_attempted = False
_load_ok = False
_model_lock = threading.Lock()

_SOURCE = "mediapipe_object_detector"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_path() -> Path:
    configured = Path(
        getattr(
            config,
            "LOCAL_ANIMAL_DETECTION_MODEL",
            getattr(config, "MEDIAPIPE_OBJECT_DETECTOR_MODEL", ""),
        )
    )
    return configured if configured.is_absolute() else _project_root() / configured


def _load_model() -> bool:
    global _detector, _mp, _load_attempted, _load_ok

    if not bool(getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)):
        return False
    if _load_attempted:
        return _load_ok
    _load_attempted = True

    model_path = _model_path()
    if not model_path.exists():
        _log.warning(
            "MediaPipe Object Detector model missing: %s — run setup_assets.py",
            model_path,
        )
        return False

    try:
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            max_results=int(getattr(config, "LOCAL_ANIMAL_DETECTION_MAX_RESULTS", 8) or 8),
            # Use the low MODEL_FLOOR here, NOT the acceptance threshold, so the
            # detector still returns sub-threshold animal candidates; acceptance is
            # applied (and sub-threshold sightings logged) in _records_from_detections.
            score_threshold=float(
                getattr(config, "LOCAL_ANIMAL_DETECTION_MODEL_FLOOR", 0.15)
            ),
        )
        _detector = ObjectDetector.create_from_options(options)
        _mp = mp
        _load_ok = True
        _log.info("MediaPipe Object Detector loaded for local animal detection: %s", model_path)
    except ImportError:
        _log.warning("mediapipe not installed — local animal detection disabled")
    except Exception as exc:
        _log.error("Failed to init MediaPipe Object Detector: %s", exc)

    return _load_ok


def preload() -> bool:
    """Warm the local object detector and keep it open for the process lifetime."""
    return _load_model()


def close() -> None:
    """Close the MediaPipe object detector before interpreter teardown."""
    global _detector, _mp, _load_attempted, _load_ok

    with _model_lock:
        detector = _detector
        _detector = None
        _mp = None
        _load_attempted = False
        _load_ok = False

    if detector is None:
        return

    try:
        detector.close()
    except Exception as exc:
        _log.debug("MediaPipe Object Detector close failed during shutdown: %s", exc)
        # MediaPipe's ObjectDetector.__del__ calls close() again. If close()
        # failed mid-shutdown, clear the native handle so a late destructor does
        # not print an ignored exception while Python is tearing modules down.
        try:
            setattr(detector, "_handle", None)
        except Exception:
            pass


def stop() -> None:
    """Alias used by main.py service shutdown."""
    close()


def _animal_species() -> set[str]:
    configured = getattr(config, "LOCAL_ANIMAL_DETECTION_SPECIES", None)
    if not configured:
        return set()
    return {str(item).strip().lower() for item in configured if str(item).strip()}


def _category_text(category) -> str:
    for attr in ("category_name", "display_name"):
        value = getattr(category, attr, None)
        if value:
            return str(value).strip().lower()
    return ""


def _best_animal_category(detection) -> Optional[tuple[str, float]]:
    animal_species = _animal_species()
    best: Optional[tuple[str, float]] = None
    for category in getattr(detection, "categories", []) or []:
        name = _category_text(category)
        if not name or name not in animal_species:
            continue
        score = float(getattr(category, "score", 0.0) or 0.0)
        if best is None or score > best[1]:
            best = (name, score)
    return best


def _position_from_bbox(bbox, frame_shape) -> str:
    height, width = frame_shape[:2]
    origin_x = float(getattr(bbox, "origin_x", 0.0) or 0.0)
    origin_y = float(getattr(bbox, "origin_y", 0.0) or 0.0)
    box_w = max(0.0, float(getattr(bbox, "width", 0.0) or 0.0))
    box_h = max(0.0, float(getattr(bbox, "height", 0.0) or 0.0))

    center_x = origin_x + (box_w / 2.0)
    center_y = origin_y + (box_h / 2.0)
    x_ratio = center_x / max(1.0, float(width))
    y_ratio = center_y / max(1.0, float(height))
    area_ratio = (box_w * box_h) / max(1.0, float(width * height))

    if x_ratio < 0.33:
        horizontal = "left"
    elif x_ratio > 0.67:
        horizontal = "right"
    else:
        horizontal = "center"

    if y_ratio < 0.35:
        depth = "upper"
    elif area_ratio >= 0.16:
        depth = "foreground"
    elif y_ratio > 0.70:
        depth = "lower"
    else:
        depth = "midframe"

    return f"{depth} {horizontal}"


def _box_tuple(bbox) -> Optional[tuple[float, float, float, float]]:
    if bbox is None:
        return None
    try:
        x = float(getattr(bbox, "origin_x", 0.0) or 0.0)
        y = float(getattr(bbox, "origin_y", 0.0) or 0.0)
        w = float(getattr(bbox, "width", 0.0) or 0.0)
        h = float(getattr(bbox, "height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _is_furred(species: str) -> bool:
    furry = getattr(config, "FURRY_COMPANION_ANIMAL_SPECIES", set()) or set()
    return species.strip().lower() in {str(item).strip().lower() for item in furry}


def _accept_threshold_for(species: str) -> float:
    """Per-species acceptance bar. Likely indoor companions (dog/cat) keep the lenient
    base threshold; every other species must clear a higher 'exotic' bar, because
    indoors those are almost always object misclassifications (a lamp read as a 'bird'),
    not real animals."""
    base = float(getattr(config, "LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD", 0.30))
    companions = {
        str(s).strip().lower()
        for s in (getattr(config, "LOCAL_ANIMAL_COMPANION_SPECIES", {"dog", "cat"}) or set())
    }
    if species.strip().lower() in companions:
        return base
    exotic = float(getattr(config, "LOCAL_ANIMAL_EXOTIC_SCORE_THRESHOLD", base))
    return max(base, exotic)


def _records_from_detections(detections, frame_shape, *, now: Optional[float] = None) -> list[dict]:
    timestamp = time.time() if now is None else now
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for detection in detections or []:
        best = _best_animal_category(detection)
        if best is None:
            continue
        species, score = best
        accept = _accept_threshold_for(species)
        if score < accept:
            # Below acceptance but above the model floor — log it so a near-miss
            # (e.g. a dog held close that scores low) is visible for tuning instead
            # of silently dropped. Lower LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD (or
            # LOCAL_ANIMAL_EXOTIC_SCORE_THRESHOLD) if these are real animals you want
            # Rex to react to.
            _log.info(
                "animal candidate below accept threshold: %s score=%.3f (accept=%.2f)",
                species, score, accept,
            )
            continue
        bbox = getattr(detection, "bounding_box", None)
        position = _position_from_bbox(bbox, frame_shape) if bbox is not None else "unknown"
        box = _box_tuple(bbox)
        key = (species, position)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "id": f"animal_{len(records) + 1}",
            "species": species,
            "position": position,
            "last_seen": timestamp,
            "confidence": round(max(0.0, min(1.0, score)), 3),
            "furred": _is_furred(species),
            "source": _SOURCE,
        })
        if box is not None:
            records[-1]["box"] = box

    return records


def detect_animals(frame) -> Optional[list[dict]]:
    """
    Return visible animals from a BGR frame using a local MediaPipe model.

    Returns None when the detector is disabled/unavailable, and [] when the
    detector ran successfully but no configured animal species were visible.
    """
    if frame is None:
        return None
    if not _load_model():
        return None

    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        with _model_lock:
            result = _detector.detect(image)
    except Exception as exc:
        _log.error("local animal detection failed: %s", exc)
        return None

    return _records_from_detections(
        getattr(result, "detections", []) or [],
        frame.shape,
    )


atexit.register(close)
