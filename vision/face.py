"""
vision/face.py — dlib face detection, recognition, and GPT-4o appearance profiling.

Models are loaded lazily on first call to detect_faces(). If any model file is
missing, the affected functions degrade gracefully and return empty results.

Detector selection
──────────────────
mmod (CNN) detector is used by default for accuracy. If it runs above
_SLOW_THRESHOLD_SECS for _SLOW_COUNT_TO_SWITCH consecutive frames the module
permanently switches to the HOG detector for the remainder of the session and
logs a warning. HOG is faster but misses non-frontal faces.
"""

import json
import logging
import time
from typing import Optional

import numpy as np

import config
from memory import facts, people
from vision.image_utils import bgr_to_rgb, encode_jpeg_base64

_log = logging.getLogger(__name__)

# ── Model handles (populated once on first use) ───────────────────────────────

_cnn_detector    = None   # dlib.cnn_face_detection_model_v1
_hog_detector    = None   # dlib.get_frontal_face_detector()
_shape_predictor = None   # dlib.shape_predictor
_face_recognizer = None   # dlib.face_recognition_model_v1
_models_ok       = False
_models_attempted = False

# ── mmod performance tracking for automatic HOG fallback ─────────────────────

_use_hog              = False
_SLOW_THRESHOLD_SECS  = 0.4   # single-frame mmod budget
_SLOW_COUNT_TO_SWITCH = 3     # consecutive slow frames before switching
_slow_count           = 0

# Keys whose stored values are never echoed in log output.
_SILENT_KEYS = frozenset({"skin_color"})


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> bool:
    global _cnn_detector, _hog_detector, _shape_predictor, _face_recognizer
    global _models_ok, _models_attempted

    if _models_attempted:
        return _models_ok
    _models_attempted = True

    try:
        import dlib
    except ImportError:
        _log.error("dlib not installed — face detection unavailable")
        return False

    ok = True

    try:
        _cnn_detector = dlib.cnn_face_detection_model_v1(config.FACE_DETECTOR_MODEL)
        _log.info("Loaded mmod face detector: %s", config.FACE_DETECTOR_MODEL)
    except Exception as exc:
        _log.error("Failed to load mmod detector %s: %s", config.FACE_DETECTOR_MODEL, exc)
        ok = False

    try:
        _hog_detector = dlib.get_frontal_face_detector()
    except Exception as exc:
        _log.error("Failed to init HOG detector: %s", exc)
        ok = False

    try:
        _shape_predictor = dlib.shape_predictor(config.FACE_LANDMARK_MODEL)
        _log.info("Loaded shape predictor: %s", config.FACE_LANDMARK_MODEL)
    except Exception as exc:
        _log.error("Failed to load shape predictor %s: %s", config.FACE_LANDMARK_MODEL, exc)
        ok = False

    try:
        _face_recognizer = dlib.face_recognition_model_v1(config.FACE_RECOGNITION_MODEL)
        _log.info("Loaded face recognizer: %s", config.FACE_RECOGNITION_MODEL)
    except Exception as exc:
        _log.error("Failed to load face recognizer %s: %s", config.FACE_RECOGNITION_MODEL, exc)
        ok = False

    _models_ok = ok
    return ok


# ── Internal detection helpers ────────────────────────────────────────────────

def _detect_rects(rgb: np.ndarray) -> list:
    """Run the active detector and return a list of dlib rectangles."""
    global _use_hog, _slow_count

    if _use_hog:
        return list(_hog_detector(rgb, 1))

    t0 = time.monotonic()
    cnn_dets = _cnn_detector(rgb, 1)
    elapsed = time.monotonic() - t0
    rects = [d.rect for d in cnn_dets]

    if elapsed > _SLOW_THRESHOLD_SECS:
        _slow_count += 1
        if _slow_count >= _SLOW_COUNT_TO_SWITCH:
            _use_hog = True
            _log.warning(
                "mmod averaging >%.2fs per frame — switching to HOG detector for this session",
                _SLOW_THRESHOLD_SECS,
            )
    else:
        _slow_count = 0

    return rects


def _largest_face(faces: list[dict]) -> Optional[dict]:
    if not faces:
        return None
    return max(faces, key=lambda f: f["bounding_box"][2] * f["bounding_box"][3])


# ── Public API ────────────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Detect all faces in an OpenCV BGR frame.

    Returns a list of dicts, one per face detected:
        bounding_box  (x, y, w, h) in pixels
        encoding      128-dim float32 numpy array (dlib face descriptor)
        landmarks     (68, 2) int32 numpy array of (x, y) landmark coordinates
    Returns an empty list if no faces found, frame is None, or models unavailable.
    """
    if frame is None:
        return []
    if not _load_models():
        return []

    rgb = bgr_to_rgb(frame)
    rects = _detect_rects(rgb)
    results = []

    for rect in rects:
        try:
            shape    = _shape_predictor(rgb, rect)
            encoding = np.array(
                _face_recognizer.compute_face_descriptor(rgb, shape),
                dtype=np.float32,
            )
            landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
            x = max(0, rect.left())
            y = max(0, rect.top())
            results.append({
                "bounding_box": (x, y, rect.width(), rect.height()),
                "encoding":     encoding,
                "landmarks":    landmarks,
            })
        except Exception as exc:
            _log.warning("Error processing face rect (%d,%d): %s", rect.left(), rect.top(), exc)

    return results


def identify_face(encoding: np.ndarray) -> Optional[dict]:
    """
    Look up a 128-dim dlib face encoding in the people database.
    Returns the matching person dict, or None if no match clears the threshold.
    """
    return people.find_by_face(encoding)


def enroll_face(person_id: int, frame: np.ndarray) -> bool:
    """
    Detect the largest face in frame and store its encoding for person_id.
    Returns True if the encoding was written to the database, False otherwise.
    """
    faces = detect_faces(frame)
    if not faces:
        _log.warning("enroll_face: no face detected in frame (person_id=%d)", person_id)
        return False

    face   = _largest_face(faces)
    result = people.add_biometric(person_id, "face", face["encoding"])
    if result is not None:
        _log.info("enroll_face: biometric stored for person_id=%d", person_id)
        return True

    _log.error("enroll_face: database write failed for person_id=%d", person_id)
    return False


def get_face_position(frame: np.ndarray) -> Optional[tuple[int, int]]:
    """
    Return the (x, y) pixel center of the largest detected face, or None.
    Called by the face-tracking loop to compute neck servo corrections.
    """
    face = _largest_face(detect_faces(frame))
    if face is None:
        return None
    x, y, w, h = face["bounding_box"]
    return (x + w // 2, y + h // 2)


def update_appearance(person_id: int, frame: np.ndarray) -> None:
    """
    Extract appearance attributes from frame using GPT-4o vision and persist them.

    Attributes stored under category="appearance":
        height_estimate, build, hair_color, hair_style, notable_features,
        age_range, age_category.

    Any field returned by GPT-4o outside the explicit prompt schema is stored
    silently without log output.
    """
    try:
        import apikeys
        from openai import OpenAI
    except ImportError as exc:
        _log.error("update_appearance: missing dependency — %s", exc)
        return

    b64 = encode_jpeg_base64(frame, quality=90)
    if b64 is None:
        _log.error("update_appearance: failed to JPEG-encode frame for person_id=%d", person_id)
        return

    detail = config.VISION_DETAIL.get("face_enrollment", "high")

    prompt = (
        "Observe the person in this image and return a JSON object with exactly "
        "these keys:\n"
        '  "height_estimate": rough height description (e.g. "tall", "average", "5ft 6in"),\n'
        '  "build": body build (e.g. "slim", "athletic", "stocky", "heavy-set"),\n'
        '  "hair_color": hair color (e.g. "black", "brown", "blonde", "gray", "none visible"),\n'
        '  "hair_style": brief style description (e.g. "short curly", "long straight ponytail"),\n'
        '  "notable_features": JSON array of strings for any distinctive features '
        "(glasses, beard, tattoo, hat, etc.) — empty array if none,\n"
        '  "age_range": estimated age range as a string (e.g. "25-35"),\n'
        '  "age_category": one of child|teen|young_adult|adult|middle_aged|senior\n'
        "Return only the JSON object, no other text."
    )

    client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": detail,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=300,
        )
    except Exception as exc:
        _log.error("update_appearance: GPT-4o error for person_id=%d: %s", person_id, exc)
        return

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            _log.error("update_appearance: non-JSON response: %.120s", raw)
            return
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            _log.error("update_appearance: JSON parse error: %s", exc)
            return

    for key, value in data.items():
        if value is None:
            continue
        str_value = json.dumps(value) if isinstance(value, list) else str(value)
        if not str_value or str_value == "[]":
            continue
        facts.add_fact(
            person_id=person_id,
            category="appearance",
            key=key,
            value=str_value,
            source="vision_gpt4o",
            confidence=0.8,
        )
        if key not in _SILENT_KEYS:
            _log.debug("update_appearance: stored %s for person_id=%d", key, person_id)

    _log.info("update_appearance: appearance profile updated for person_id=%d", person_id)
