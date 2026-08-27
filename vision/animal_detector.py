"""
vision/animal_detector.py — Local animal + room-object detection.

This is the no-OpenAI-credits path for live "a dog wandered into frame" style
awareness plus the room-object stream. It consumes the same camera frame buffer
as face expression telemetry.

Backend selection (config.OBJECT_DETECTOR_BACKEND)
────────────────────────────────────────────────────
"rfdetr" (default): RF-DETR nano (Apache 2.0, real-time DETR) via torch —
measured ~40ms/frame CPU warm with dramatically better recall/precision than
the 2019 EfficientDet-Lite0 it replaced (live: a 6-person group photo scored
0.87-0.94 per person vs EfficientDet's sub-threshold noise). Weights live in
config.RFDETR_MODEL_DIR (downloaded by setup_assets.py, ~350MB).

"mediapipe": legacy EfficientDet-Lite0 via MediaPipe Tasks. Also the automatic
runtime fallback if RF-DETR fails to load.

Both backends feed the SAME record builders (`_records_from_detections` /
`_object_records_from_detections`) — RF-DETR outputs are adapted to the
MediaPipe detection duck-type, so species lists, per-species thresholds, the
no-screens exclusion rule, and position phrasing are backend-independent.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
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

# RF-DETR backend state (resolved at first load; falls back to mediapipe).
_rf_model = None
_rf_classes: dict = {}
_active_backend: Optional[str] = None
_preload_done = threading.Event()
_preload_thread: Optional[threading.Thread] = None

_SOURCE = "mediapipe_object_detector"
_RF_SOURCE = "rfdetr_object_detector"


def _source() -> str:
    return _RF_SOURCE if _active_backend == "rfdetr" else _SOURCE


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


def _model_floor() -> float:
    """The low model-level score floor shared by animals + objects; acceptance
    thresholds are applied downstream in the record builders."""
    return float(getattr(config, "LOCAL_ANIMAL_DETECTION_MODEL_FLOOR", 0.15))


def _load_rfdetr() -> bool:
    """RF-DETR nano — the primary backend. Returns True when ready."""
    global _rf_model, _rf_classes
    model_dir = str(getattr(config, "RFDETR_MODEL_DIR", "assets/models/rfdetr"))
    weights = Path(model_dir)
    if not weights.is_absolute():
        weights = _project_root() / weights
    if not (weights / "rf-detr-nano.pth").exists():
        _log.warning("RF-DETR weights missing at %s — run setup_assets.py", weights)
        return False
    try:
        # RF_HOME steers the package's weight cache to the project assets dir
        # (must be set before the model class instantiates).
        os.environ.setdefault("RF_HOME", str(weights))
        from rfdetr import RFDETRNano
        from rfdetr.util.coco_classes import COCO_CLASSES
        _rf_model = RFDETRNano()
        _rf_classes = dict(COCO_CLASSES) if isinstance(COCO_CLASSES, dict) else {
            i: name for i, name in enumerate(COCO_CLASSES)
        }
        _log.info("RF-DETR nano loaded for local animal/object detection: %s", weights)
        return True
    except ImportError as exc:
        _log.warning("rfdetr not installed (%s)", exc)
        return False
    except Exception as exc:
        _log.error("Failed to init RF-DETR: %s", exc)
        return False


def _rf_detections_to_mp(result) -> list:
    """Adapt a supervision.Detections result to the MediaPipe detection duck-type
    the record builders consume: .categories[].category_name/.score and
    .bounding_box.origin_x/origin_y/width/height."""
    adapted = []
    try:
        xyxy = getattr(result, "xyxy", None)
        confidence = getattr(result, "confidence", None)
        class_id = getattr(result, "class_id", None)
        if xyxy is None or confidence is None or class_id is None:
            return adapted
        for (x1, y1, x2, y2), score, cid in zip(xyxy, confidence, class_id):
            name = str(_rf_classes.get(int(cid), "")).strip().lower()
            if not name:
                continue
            adapted.append(SimpleNamespace(
                categories=[SimpleNamespace(category_name=name, score=float(score))],
                bounding_box=SimpleNamespace(
                    origin_x=float(x1), origin_y=float(y1),
                    width=float(x2) - float(x1), height=float(y2) - float(y1),
                ),
            ))
    except Exception as exc:
        _log.debug("RF-DETR detection adaptation failed: %s", exc)
    return adapted


def _rf_detect(frame) -> Optional[list]:
    """Run RF-DETR on a BGR frame; returns adapted detections or None on failure."""
    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        with _model_lock:
            result = _rf_model.predict(rgb, threshold=_model_floor())
        return _rf_detections_to_mp(result)
    except Exception as exc:
        _log.error("RF-DETR detection failed: %s", exc)
        return None


def _load_model() -> bool:
    global _detector, _mp, _load_attempted, _load_ok, _active_backend

    # The same detector backs BOTH animal and object detection — load it when EITHER
    # is enabled (objects reuse this instance; no second model).
    if not (
        bool(getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True))
        or bool(getattr(config, "OBJECT_DETECTION_ENABLED", True))
    ):
        return False
    if _load_attempted:
        return _load_ok
    _load_attempted = True

    backend = str(getattr(config, "OBJECT_DETECTOR_BACKEND", "rfdetr") or "rfdetr").lower()
    if backend == "rfdetr":
        if _load_rfdetr():
            _active_backend = "rfdetr"
            _load_ok = True
            return True
        _log.warning("RF-DETR unavailable — falling back to MediaPipe object detector")

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
            # Shared by animals + objects, so return enough for the larger of the two.
            max_results=max(
                int(getattr(config, "LOCAL_ANIMAL_DETECTION_MAX_RESULTS", 8) or 8),
                int(getattr(config, "OBJECT_DETECTION_MAX_RESULTS", 12) or 12),
            ),
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
        _active_backend = "mediapipe"
        _log.info("MediaPipe Object Detector loaded for local animal detection: %s", model_path)
    except ImportError:
        _log.warning("mediapipe not installed — local animal detection disabled")
    except Exception as exc:
        _log.error("Failed to init MediaPipe Object Detector: %s", exc)

    return _load_ok


def active_backend() -> Optional[str]:
    """The object-detector backend in use ("rfdetr"/"mediapipe"), or None if unloaded."""
    return _active_backend


def preload() -> bool:
    """Warm the local object detector and keep it open for the process lifetime.

    Runs in a background thread: RF-DETR pays ~4s of model build plus ~8s of
    first-inference torch warmup — far too much for the synchronous boot path
    (MediaPipe loaded in ~0.2s). Until the thread finishes, detect_* calls
    simply report unavailable and the scene scan skips a beat; the warmup
    dummy predict absorbs the torch graph cost so the FIRST real frame is fast.
    """
    global _preload_thread
    if _preload_thread is not None and _preload_thread.is_alive():
        return True

    _preload_done.clear()

    def _warm() -> None:
        try:
            if not _load_model():
                _log.warning("local object detector preload failed; detection will stay off")
                return
            if _active_backend == "rfdetr":
                try:
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    _rf_detect(dummy)
                    _log.info("RF-DETR warmed (first-inference torch warmup absorbed at boot)")
                except Exception as exc:
                    _log.debug("RF-DETR warmup failed: %s", exc)
        finally:
            _preload_done.set()

    _preload_thread = threading.Thread(
        target=_warm, daemon=True, name="object-detector-preload"
    )
    _preload_thread.start()
    return True


def wait_for_preload(timeout: Optional[float] = None) -> bool:
    """Wait until the background model build/first inference is finished."""
    thread = _preload_thread
    if thread is None:
        return True
    if not _preload_done.wait(timeout):
        return False
    return bool(_load_ok)


def close() -> None:
    """Close the active object detector before interpreter teardown."""
    global _detector, _mp, _load_attempted, _load_ok, _rf_model, _active_backend

    with _model_lock:
        detector = _detector
        _detector = None
        _mp = None
        _rf_model = None      # torch model needs no explicit close; drop the ref
        _active_backend = None
        _load_attempted = False
        _load_ok = False
        _preload_done.clear()

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


# Every animal label the model has floated recently — accepted OR near-miss —
# stamped with when it last appeared. Field 2026-08-27 13:35:45: the near-miss
# stream was already the loudest evidence in the log (dog 0.166/0.178, cat
# 0.153-0.267, horse up to 0.354, bird 0.165, all at the same furry shape inside
# forty seconds) and it was logged and thrown away, so the arrival gate could not
# tell "the model is sure" from "the model is shopping".
_species_candidate_seen: dict[str, float] = {}


def _note_species_candidate(species: str) -> None:
    """Stamp one animal label the model floated this scan. Never raises."""
    try:
        key = str(species or "").strip().lower()
        if key:
            _species_candidate_seen[key] = time.monotonic()
    except Exception:
        pass


def contested_by(species) -> Optional[str]:
    """The other likely-companion label the model floated within
    ANIMAL_SPECIES_CONTEST_WINDOW_SECS, or None.

    A dog read as a cat is the one confusion that renames the household pet, so
    only the companion pair counts as a rival. The stray "horse" this room throws
    every minute is not a contest, it is clutter the exotic threshold already
    turns away. Never raises: an unknown contest reads as uncontested."""
    try:
        window = float(getattr(config, "ANIMAL_SPECIES_CONTEST_WINDOW_SECS", 60.0))
        if window <= 0:
            return None
        key = str(species or "").strip().lower()
        companions = {
            str(s).strip().lower()
            for s in (getattr(config, "LOCAL_ANIMAL_COMPANION_SPECIES", {"dog", "cat"}) or set())
        }
        if key not in companions:
            return None
        now = time.monotonic()
        rivals = [
            (seen_at, other)
            for other, seen_at in _species_candidate_seen.items()
            if other != key and other in companions and (now - seen_at) <= window
        ]
        if not rivals:
            return None
        return max(rivals)[1]
    except Exception:
        return None


def _records_from_detections(detections, frame_shape, *, now: Optional[float] = None) -> list[dict]:
    timestamp = time.time() if now is None else now
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for detection in detections or []:
        best = _best_animal_category(detection)
        if best is None:
            continue
        species, score = best
        _note_species_candidate(species)
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
            "source": _source(),
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

    if _active_backend == "rfdetr":
        detections = _rf_detect(frame)
        if detections is None:
            return None
        return _records_from_detections(detections, frame.shape)

    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        # Hold the lock across BOTH the _mp.Image construction and detect() so a
        # concurrent close() (atexit) can't null _mp/_detector mid-call.
        with _model_lock:
            image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            result = _detector.detect(image)
    except Exception as exc:
        _log.error("local animal detection failed: %s", exc)
        return None

    return _records_from_detections(
        getattr(result, "detections", []) or [],
        frame.shape,
    )


# ── Object detection (the rest of the COCO 80 classes the animal path discards) ──────

def _object_excluded_classes() -> set[str]:
    """COCO labels NOT published as room objects: screens/devices (the no-screens
    rule), plus people and animals — those are tracked in world_state.people /
    world_state.animals, not the object stream."""
    banned = {
        str(x).strip().lower()
        for x in (getattr(config, "OBJECT_DETECTION_BANNED_CLASSES", set()) or set())
    }
    return banned | _animal_species() | {"person"}


def _best_object_category(detection, excluded: set[str]) -> Optional[tuple[str, float]]:
    """Highest-scoring category for a detection that is NOT excluded, or None."""
    best: Optional[tuple[str, float]] = None
    for category in getattr(detection, "categories", []) or []:
        name = _category_text(category)
        if not name or name in excluded:
            continue
        score = float(getattr(category, "score", 0.0) or 0.0)
        if best is None or score > best[1]:
            best = (name, score)
    return best


def _self_occlusion_fraction(box, frame_shape) -> float:
    """Fraction of a pixel box (x, y, w, h) lying inside any configured
    self-occlusion zone (normalized rects; config.CAMERA_SELF_OCCLUSION_ZONES) —
    the regions of the frame permanently blocked by Rex's own eye stalks."""
    zones = getattr(config, "CAMERA_SELF_OCCLUSION_ZONES", None) or []
    if not zones or box is None:
        return 0.0
    fh, fw = float(frame_shape[0]), float(frame_shape[1])
    if fh <= 0 or fw <= 0:
        return 0.0
    x, y, w, h = (float(v) for v in box)
    area = max(1.0, w * h)
    worst = 0.0
    for zx0, zy0, zx1, zy1 in zones:
        ix0, iy0 = max(x, zx0 * fw), max(y, zy0 * fh)
        ix1, iy1 = min(x + w, zx1 * fw), min(y + h, zy1 * fh)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        worst = max(worst, inter / area)
    return worst


def _object_records_from_detections(
    detections, frame_shape, *, now: Optional[float] = None
) -> list[dict]:
    timestamp = time.time() if now is None else now
    excluded = _object_excluded_classes()
    accept = float(getattr(config, "OBJECT_DETECTION_SCORE_THRESHOLD", 0.35))
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for detection in detections or []:
        best = _best_object_category(detection, excluded)
        if best is None:
            continue
        label, score = best
        if score < accept:
            continue
        bbox = getattr(detection, "bounding_box", None)
        position = _position_from_bbox(bbox, frame_shape) if bbox is not None else "unknown"
        box = _box_tuple(bbox)
        # Self-occlusion mask: a detection sitting mostly on Rex's own eye stalks is
        # his face, not furniture (field bug: the stalks kept publishing as "chairs").
        occl = _self_occlusion_fraction(box, frame_shape)
        if occl > float(getattr(config, "CAMERA_SELF_OCCLUSION_MAX_OVERLAP", 0.55)):
            _log.debug(
                "object %r suppressed: %.0f%% inside a self-occlusion zone",
                label, occl * 100.0,
            )
            continue
        key = (label, position)
        if key in seen:
            continue
        seen.add(key)
        record = {
            "id": f"object_{len(records) + 1}",
            "label": label,
            "position": position,
            "last_seen": timestamp,
            "confidence": round(max(0.0, min(1.0, score)), 3),
            "source": _source(),
        }
        if box is not None:
            record["box"] = box
        records.append(record)

    return records


def detect_objects(frame) -> Optional[list[dict]]:
    """
    Return visible non-animal room OBJECTS from a BGR frame — the COCO 80-class stream
    MINUS screens/devices (no-screens rule), people, and animals (tracked elsewhere).

    Returns None when object detection is disabled/unavailable, and [] when the detector
    ran but nothing publishable was visible. Reuses the SAME loaded detector as
    detect_animals (one model; a separate inference pass).
    """
    if frame is None:
        return None
    if not bool(getattr(config, "OBJECT_DETECTION_ENABLED", True)):
        return None
    if not _load_model():
        return None

    if _active_backend == "rfdetr":
        detections = _rf_detect(frame)
        if detections is None:
            return None
        return _object_records_from_detections(detections, frame.shape)

    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        # Hold the lock across BOTH the _mp.Image construction and detect() so a
        # concurrent close() (atexit) can't null _mp/_detector mid-call.
        with _model_lock:
            image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            result = _detector.detect(image)
    except Exception as exc:
        _log.error("local object detection failed: %s", exc)
        return None

    return _object_records_from_detections(
        getattr(result, "detections", []) or [],
        frame.shape,
    )


atexit.register(close)
