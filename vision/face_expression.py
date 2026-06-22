"""
vision/face_expression.py — Local face-expression telemetry via MediaPipe.

This module does not replace dlib identity. It samples the shared camera frame,
uses MediaPipe Face Landmarker blendshapes to infer visible expression, then
annotates the existing world_state.people slots that dlib already owns.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config
from vision.image_utils import bgr_to_rgb
from world_state import world_state

_log = logging.getLogger(__name__)

_landmarker = None
_mp = None
_load_attempted = False
_load_ok = False
_last_timestamp_ms = 0
_model_lock = threading.Lock()
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None

_SOURCE = "mediapipe_face_landmarker"
_BLENDSHAPE_KEYS = (
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_path() -> Path:
    configured = Path(getattr(config, "MEDIAPIPE_FACE_LANDMARKER_MODEL", ""))
    return configured if configured.is_absolute() else _project_root() / configured


def _load_model() -> bool:
    global _landmarker, _mp, _load_attempted, _load_ok

    if not bool(getattr(config, "FACE_EXPRESSION_LOCAL_ENABLED", True)):
        return False
    if _load_attempted:
        return _load_ok
    _load_attempted = True

    model_path = _model_path()
    if not model_path.exists():
        _log.warning(
            "MediaPipe Face Landmarker model missing: %s — run setup_assets.py",
            model_path,
        )
        return False

    try:
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=int(getattr(config, "FACE_EXPRESSION_MAX_FACES", 2) or 2),
            min_face_detection_confidence=float(
                getattr(config, "FACE_EXPRESSION_MIN_DETECTION_CONFIDENCE", 0.5)
            ),
            min_face_presence_confidence=float(
                getattr(config, "FACE_EXPRESSION_MIN_PRESENCE_CONFIDENCE", 0.5)
            ),
            min_tracking_confidence=float(
                getattr(config, "FACE_EXPRESSION_MIN_TRACKING_CONFIDENCE", 0.5)
            ),
            output_face_blendshapes=True,
            # Free per-face 4x4 head-pose matrix (already computed internally) —
            # vision/active_speaker.py reads yaw from it for its facing gate.
            output_facial_transformation_matrixes=True,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
        _mp = mp
        _load_ok = True
        _log.info(
            "MediaPipe Face Landmarker loaded for local expression telemetry: %s",
            model_path,
        )
    except ImportError:
        _log.warning("mediapipe not installed — local face-expression telemetry disabled")
    except Exception as exc:
        _log.error("Failed to init MediaPipe Face Landmarker: %s", exc)

    return _load_ok


def _next_timestamp_ms() -> int:
    global _last_timestamp_ms
    ts = int(time.monotonic() * 1000)
    if ts <= _last_timestamp_ms:
        ts = _last_timestamp_ms + 1
    _last_timestamp_ms = ts
    return ts


def _score(scores: dict[str, float], key: str) -> float:
    return max(0.0, min(1.0, float(scores.get(key, 0.0) or 0.0)))


def _mean(scores: dict[str, float], *keys: str) -> float:
    if not keys:
        return 0.0
    return sum(_score(scores, key) for key in keys) / float(len(keys))


# Per-face adaptive brow-furrow baseline (see config.FACE_EXPRESSION_BROW_*). Tracks each
# visible face's RESTING browDown so a high-neutral face (MediaPipe over-reads browDown for
# some face/camera geometries) isn't tagged "furrowing" every frame. Keyed by IoU-matched
# face box; pruned by TTL when a face leaves the frame.
_brow_tracks: list[dict] = []
_brow_lock = threading.Lock()


def reset_brow_baselines() -> None:
    """Clear the adaptive brow baselines (device changes / tests)."""
    with _brow_lock:
        _brow_tracks.clear()


def _brow_furrow_baseline(face_box, brow_down: float, now: float) -> Optional[float]:
    """Update and return the resting-browDown baseline for the face at ``face_box``.

    Returns None — meaning "use the absolute threshold" — when adaptive baselining is
    off, the face box is unknown, or the track has fewer than WARMUP_SAMPLES observations.
    """
    if not bool(getattr(config, "FACE_EXPRESSION_BROW_ADAPTIVE_BASELINE_ENABLED", True)):
        return None
    if not face_box:
        return None
    ttl = float(getattr(config, "FACE_EXPRESSION_BROW_BASELINE_TTL_SECS", 8.0))
    a_down = float(getattr(config, "FACE_EXPRESSION_BROW_BASELINE_ALPHA_DOWN", 0.20))
    a_up = float(getattr(config, "FACE_EXPRESSION_BROW_BASELINE_ALPHA_UP", 0.02))
    warmup = max(1, int(getattr(config, "FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES", 15)))
    with _brow_lock:
        # Drop tracks for faces that have left the frame so boxes can't accumulate.
        _brow_tracks[:] = [t for t in _brow_tracks if (now - t["last_ts"]) <= ttl]
        track = None
        best = 0.0
        for candidate in _brow_tracks:
            score = _iou(face_box, candidate["box"])
            if score > best:
                best, track = score, candidate
        if track is None or best < 0.10:
            track = {"box": face_box, "baseline": float(brow_down), "samples": 0, "last_ts": now}
            _brow_tracks.append(track)
        base = float(track["baseline"])
        # Asymmetric: follow a more-relaxed reading quickly, a tenser one slowly, so the
        # baseline tracks the resting floor and a real furrow stays a spike above it.
        alpha = a_down if brow_down < base else a_up
        track["baseline"] = base + alpha * (float(brow_down) - base)
        track["box"] = face_box
        track["last_ts"] = now
        track["samples"] += 1
        if track["samples"] < warmup:
            return None
        return float(track["baseline"])


def _classify_expression(
    scores: dict[str, float], brow_baseline: Optional[float] = None
) -> dict:
    smile = _mean(scores, "mouthSmileLeft", "mouthSmileRight")
    frown = _mean(scores, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _mean(scores, "browDownLeft", "browDownRight")
    eye_wide = _mean(scores, "eyeWideLeft", "eyeWideRight")
    jaw_open = _score(scores, "jawOpen")
    brow_inner = _score(scores, "browInnerUp")
    surprise = (eye_wide + jaw_open + brow_inner) / 3.0

    # Brow-furrow fires on a rise ABOVE the person's resting brow, never below the
    # absolute floor — so a high-neutral face stops false-triggering while a
    # low-neutral face keeps its original sensitivity unchanged.
    brow_threshold = float(getattr(config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45))
    if brow_baseline is not None:
        delta = float(getattr(config, "FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA", 0.18))
        brow_threshold = max(brow_threshold, float(brow_baseline) + delta)

    candidates = [
        (
            "happy",
            "smile",
            "smiling",
            smile,
            float(getattr(config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.35)),
        ),
        (
            "surprised",
            "surprise",
            "wide eyes/open mouth",
            surprise,
            float(getattr(config, "FACE_EXPRESSION_SURPRISE_THRESHOLD", 0.40)),
        ),
        (
            "sad",
            "frown",
            "downturned mouth",
            frown,
            float(getattr(config, "FACE_EXPRESSION_FROWN_THRESHOLD", 0.35)),
        ),
        (
            "focused",
            "brow_furrow",
            "furrowed brow",
            brow_down,
            brow_threshold,
        ),
    ]
    mood, expression, notes, confidence, threshold = max(
        candidates,
        key=lambda item: item[3] - item[4],
    )
    if confidence < threshold:
        return {
            "mood": "neutral",
            "expression": "neutral",
            "confidence": max(0.0, min(1.0, 1.0 - max(smile, frown, surprise, brow_down))),
            "notes": "",
        }
    return {
        "mood": mood,
        "expression": expression,
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": notes,
    }


def _blendshape_scores(categories) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in categories or []:
        name = getattr(item, "category_name", "") or getattr(item, "display_name", "")
        if not name:
            continue
        try:
            scores[str(name)] = float(getattr(item, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            scores[str(name)] = 0.0
    return scores


def _face_box_from_landmarks(landmarks, frame_shape) -> Optional[tuple[int, int, int, int]]:
    if not landmarks or frame_shape is None or len(frame_shape) < 2:
        return None
    height, width = int(frame_shape[0] or 0), int(frame_shape[1] or 0)
    if width <= 0 or height <= 0:
        return None

    xs = [max(0.0, min(1.0, float(getattr(point, "x", 0.0) or 0.0))) for point in landmarks]
    ys = [max(0.0, min(1.0, float(getattr(point, "y", 0.0) or 0.0))) for point in landmarks]
    if not xs or not ys:
        return None
    x0 = int(min(xs) * width)
    y0 = int(min(ys) * height)
    x1 = int(max(xs) * width)
    y1 = int(max(ys) * height)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _result_to_expressions(result, frame_shape) -> list[dict]:
    blendshapes = list(getattr(result, "face_blendshapes", None) or [])
    landmarks = list(getattr(result, "face_landmarks", None) or [])
    matrices = list(getattr(result, "facial_transformation_matrixes", None) or [])
    expressions: list[dict] = []
    now = time.time()
    for idx, categories in enumerate(blendshapes):
        scores = _blendshape_scores(categories)
        face_box = (
            _face_box_from_landmarks(landmarks[idx], frame_shape)
            if idx < len(landmarks)
            else None
        )
        brow_down = _mean(scores, "browDownLeft", "browDownRight")
        brow_baseline = _brow_furrow_baseline(face_box, brow_down, now)
        classified = _classify_expression(scores, brow_baseline=brow_baseline)
        expressions.append({
            **classified,
            "source": _SOURCE,
            "face_box": face_box,
            "blendshapes": {
                key: round(_score(scores, key), 4)
                for key in _BLENDSHAPE_KEYS
                if key in scores
            },
            # Active-speaker inputs (vision/active_speaker.py): RAW jawOpen (not the
            # 4-decimal-rounded blendshapes map above — its quantum swamps the
            # variance signal) + the per-face head-pose matrix and landmarks.
            "jaw_open": _score(scores, "jawOpen"),
            "transform_matrix": (matrices[idx] if idx < len(matrices) else None),
            "landmarks": (landmarks[idx] if idx < len(landmarks) else None),
        })
    return expressions


def detect_expressions(frame) -> list[dict]:
    """Return local expression readings for the visible faces in a BGR frame."""
    if frame is None:
        return []
    if not _load_model():
        return []

    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        with _model_lock:
            result = _landmarker.detect_for_video(mp_image, _next_timestamp_ms())
        return _result_to_expressions(result, getattr(frame, "shape", None))
    except Exception as exc:
        _log.debug("MediaPipe face-expression detection failed: %s", exc)
        return []


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    try:
        ax, ay, aw, ah = [float(v) for v in a[:4]]
        bx, by, bw, bh = [float(v) for v in b[:4]]
    except Exception:
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    return inter / ((aw * ah) + (bw * bh) - inter)


def _visible_person_indices(people: list[dict]) -> list[int]:
    return [
        idx
        for idx, person in enumerate(people)
        if isinstance(person, dict)
        and person.get("face_visible") is not False
        and not person.get("face_missing")
    ]


def _match_expression_to_people(
    expression: dict,
    people: list[dict],
    visible_indices: list[int],
    used_indices: set[int],
) -> Optional[int]:
    available = [idx for idx in visible_indices if idx not in used_indices]
    if not available:
        return None
    if len(available) == 1:
        return available[0]

    expr_box = expression.get("face_box")
    best_idx = None
    best_iou = 0.0
    for idx in available:
        score = _iou(expr_box, people[idx].get("face_box"))
        if score > best_iou:
            best_iou = score
            best_idx = idx
    return best_idx if best_idx is not None and best_iou >= 0.05 else available[0]


def _expression_payload(expression: dict) -> tuple[dict, dict, str]:
    now = time.time()
    mood = str(expression.get("mood") or "neutral")
    apparent = str(expression.get("expression") or mood)
    confidence = max(0.0, min(1.0, float(expression.get("confidence") or 0.0)))
    notes = str(expression.get("notes") or "")
    face_mood = {
        "mood": mood,
        "confidence": confidence,
        "notes": notes,
        "source": _SOURCE,
        "updated_at": now,
    }
    face_expression = {
        "expression": apparent,
        "mood": mood,
        "confidence": confidence,
        "notes": notes,
        "source": _SOURCE,
        "updated_at": now,
        "blendshapes": dict(expression.get("blendshapes") or {}),
    }
    return face_mood, face_expression, mood


def merge_expressions_into_world_state(
    expressions: list[dict],
    collect_matches: Optional[list] = None,
) -> int:
    """Attach expression readings to existing visible people slots.

    If ``collect_matches`` is provided, it is filled with ``(expr_index, slot_idx,
    person_db_id)`` tuples for each matched expression — reusing the SAME IoU
    association so the active-speaker hook never re-associates faces (spec §2).
    """
    if not expressions:
        return 0

    result = {"changed": 0}

    def _merge_expressions(people):
        if not isinstance(people, list) or not people:
            return None

        updated = list(people)
        visible_indices = _visible_person_indices(updated)
        used_indices: set[int] = set()
        for expr_i, expression in enumerate(expressions):
            idx = _match_expression_to_people(expression, updated, visible_indices, used_indices)
            if idx is None:
                continue
            used_indices.add(idx)
            if collect_matches is not None:
                collect_matches.append((expr_i, idx, updated[idx].get("person_db_id")))
            face_mood, face_expression, label = _expression_payload(expression)
            person = dict(updated[idx])
            person["face_mood"] = face_mood
            person["face_expression"] = face_expression
            person["facial_expression"] = face_expression
            person["expression"] = label
            updated[idx] = person
            result["changed"] += 1
        return updated if result["changed"] else None

    # Read-modify-write under the world_state lock so concurrent identity/pose
    # writes aren't reverted by a stale snapshot.
    world_state.mutate("people", _merge_expressions)
    return result["changed"]


def _run_active_speaker(expressions: list[dict], matches: list) -> None:
    """Feed per-face lip/pose signals to the active-speaker detector, reusing the
    expression→slot matches just computed by the IoU association (no re-detect, no
    re-associate). Best-effort: never let it disturb the expression pipeline."""
    try:
        from vision import active_speaker
        from awareness.situation import assessor
    except Exception:
        return
    if not active_speaker.enabled():
        return
    try:
        now = time.time()
        by_expr = {expr_i: (slot_idx, pid) for (expr_i, slot_idx, pid) in matches}
        signals = []
        for expr_i, expr in enumerate(expressions):
            if expr_i not in by_expr:
                continue
            slot_idx, pid = by_expr[expr_i]
            signals.append({
                "slot_idx": slot_idx,
                "person_db_id": pid,
                "jaw_open": float(expr.get("jaw_open") or 0.0),
                "yaw": active_speaker.yaw_from_transform_matrix(expr.get("transform_matrix")),
                "ts": now,
            })
        active_speaker.update(signals, vad_active=assessor.is_user_speaking())
    except Exception as exc:
        _log.debug("active-speaker update failed: %s", exc)


def process_frame(frame) -> list[dict]:
    expressions = detect_expressions(frame)
    matches: list = []
    merge_expressions_into_world_state(expressions, collect_matches=matches)
    if expressions and bool(getattr(config, "ACTIVE_SPEAKER_ENABLED", True)):
        _run_active_speaker(expressions, matches)
    return expressions


def _loop() -> None:
    from vision import camera

    interval = max(
        0.05,
        float(getattr(config, "FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS", 0.25) or 0.25),
    )
    while not _stop_event.is_set():
        try:
            frame = camera.get_frame()
            if frame is not None:
                process_frame(frame)
        except Exception as exc:
            _log.debug("face-expression telemetry loop error: %s", exc)
        _stop_event.wait(interval)


def start() -> None:
    global _thread
    if not bool(getattr(config, "FACE_EXPRESSION_LOCAL_ENABLED", True)):
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(
        target=_loop,
        daemon=True,
        name="face-expression-telemetry",
    )
    _thread.start()
    _log.info(
        "MediaPipe face-expression telemetry started (interval=%.2fs)",
        float(getattr(config, "FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS", 0.25) or 0.25),
    )


def stop() -> None:
    global _thread, _landmarker, _mp, _load_attempted, _load_ok
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=2.0)
    _thread = None
    try:
        from vision import active_speaker
        active_speaker.reset()
    except Exception:
        pass
    with _model_lock:
        if _landmarker is not None:
            try:
                _landmarker.close()
            except Exception:
                pass
        _landmarker = None
        _mp = None
        _load_attempted = False
        _load_ok = False


def _reset_for_tests() -> None:
    global _last_timestamp_ms
    stop()
    _last_timestamp_ms = 0
