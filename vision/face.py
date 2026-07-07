"""
vision/face.py — face detection, recognition, and GPT-4o appearance profiling.

Models are loaded lazily on first call to detect_faces(). If any model file is
missing, the affected functions degrade gracefully and return empty results.

Backend selection (config.FACE_BACKEND)
────────────────────────────────────────
"insightface" (default): SCRFD detector + ArcFace recognizer (512-dim L2-normalized
embeddings) via ONNX Runtime. Handles the robot's upward camera angle, small/
distant faces, and non-frontal views far better than dlib, at ~70ms/frame CPU.
If the InsightFace models fail to load, the module falls back to dlib for the
session and logs a warning.

"dlib": legacy HOG/mmod detector + 128-dim ResNet descriptor. When
config.FACE_DETECTOR_FORCE_HOG is True, the HOG detector is used from the
start and mmod is never loaded. When False, mmod (CNN) is used by default; if it
runs above _SLOW_THRESHOLD_SECS for _SLOW_COUNT_TO_SWITCH consecutive frames the
module permanently switches to HOG for the session and logs a warning.

The two backends' embeddings are INCOMPATIBLE (512 vs 128 dim) — the matcher in
memory/people.find_by_face silently skips stored encodings whose dimension does
not match the query, so faces enrolled under one backend must be re-enrolled
after switching.
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

_insight_app     = None   # insightface.app.FaceAnalysis (detection + recognition)
_cnn_detector    = None   # dlib.cnn_face_detection_model_v1
_hog_detector    = None   # dlib.get_frontal_face_detector()
_shape_predictor = None   # dlib.shape_predictor
_face_recognizer = None   # dlib.face_recognition_model_v1
_models_ok       = False
_models_attempted = False

# Resolved at first load: "insightface" or "dlib" (after any fallback).
_active_backend: Optional[str] = None

# ── mmod performance tracking for automatic HOG fallback ─────────────────────

_use_hog              = config.FACE_DETECTOR_FORCE_HOG
_SLOW_THRESHOLD_SECS  = 0.4   # single-frame mmod budget
_SLOW_COUNT_TO_SWITCH = 3     # consecutive slow frames before switching
_slow_count           = 0

# Keys whose stored values are never echoed in log output.
_SILENT_KEYS = frozenset({"skin_color"})


# ── Recognized-identity helper ────────────────────────────────────────────────

def visible_known_people(snapshot=None) -> list[tuple[int, str]]:
    """``(person_db_id, name)`` for each recognized, currently-visible person.

    The id-bearing companion to :func:`visible_known_names`. Use this when a memory
    needs to be ATTRIBUTED to the person (stored against their ``person_id``), not
    just name them in prose — e.g. an episodic scene "Bret was at his desk" that can
    later be recalled as part of Rex's history WITH Bret.

    Order-preserving, de-duplicated by id. Unknown faces (no ``person_db_id``) are
    skipped. Never raises — returns [] on any failure.
    """
    try:
        if snapshot is not None:
            entries = (snapshot or {}).get("people") or []
        else:
            from world_state import world_state
            entries = world_state.get("people") or []
    except Exception:
        return []

    out: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for person in entries:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        pid = person.get("person_db_id")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid is None or pid in seen_ids:
            continue
        seen_ids.add(pid)
        try:
            row = people.get_person(pid)
        except Exception:
            row = None
        name = (row or {}).get("name")
        if name:
            out.append((pid, name))
    return out


def visible_known_names(snapshot=None) -> list[str]:
    """Names of recognized people currently visible to the camera.

    The bridge that folds dlib face-recognition identity into the GPT-4o vision
    descriptions, so Rex can say "Bret is at his desk" instead of "a man is at a
    desk". Order-preserving, de-duplicated; unknown faces are skipped. Never raises.
    """
    names: list[str] = []
    for _pid, name in visible_known_people(snapshot):
        if name and name not in names:
            names.append(name)
    return names


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> bool:
    """Load the configured backend; on InsightFace failure fall back to dlib."""
    global _models_ok, _models_attempted, _active_backend

    if _models_attempted:
        return _models_ok
    _models_attempted = True

    backend = str(getattr(config, "FACE_BACKEND", "insightface") or "insightface").lower()

    if backend == "insightface":
        if _load_insightface():
            _active_backend = "insightface"
            _models_ok = True
            return True
        _log.warning("InsightFace unavailable — falling back to dlib backend")

    _models_ok = _load_dlib()
    _active_backend = "dlib" if _models_ok else None
    return _models_ok


def active_backend() -> Optional[str]:
    """The face backend actually in use ("insightface"/"dlib"), or None if unloaded."""
    return _active_backend


def _load_insightface() -> bool:
    """Load SCRFD detection + ArcFace recognition from the local model root.

    insightface auto-downloads the model pack on first use if the directory is
    missing (setup_assets.py pre-downloads it so the robot never needs to).
    """
    global _insight_app

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        _log.error("insightface not installed — pip install insightface onnxruntime")
        return False

    try:
        app = FaceAnalysis(
            name=getattr(config, "INSIGHTFACE_MODEL_PACK", "buffalo_l"),
            root=getattr(config, "INSIGHTFACE_MODEL_ROOT", "assets/models/insightface"),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        det = int(getattr(config, "INSIGHTFACE_DET_SIZE", 640) or 640)
        app.prepare(ctx_id=0, det_size=(det, det))
    except Exception as exc:
        _log.error("Failed to load InsightFace models: %s", exc)
        return False

    # FaceAnalysis silently drops modules whose .onnx is missing — verify both.
    loaded = set(getattr(app, "models", {}) or {})
    if not {"detection", "recognition"} <= loaded:
        _log.error("InsightFace pack incomplete (loaded=%s) — need detection+recognition", sorted(loaded))
        return False

    _insight_app = app
    _log.info("Loaded InsightFace %s (det_size=%d): SCRFD + ArcFace 512-dim",
              getattr(config, "INSIGHTFACE_MODEL_PACK", "buffalo_l"), det)
    return True


def _load_dlib() -> bool:
    global _cnn_detector, _hog_detector, _shape_predictor, _face_recognizer

    try:
        import dlib
    except ImportError:
        _log.error("dlib not installed — face detection unavailable")
        return False

    ok = True

    if not config.FACE_DETECTOR_FORCE_HOG:
        try:
            _cnn_detector = dlib.cnn_face_detection_model_v1(config.FACE_DETECTOR_MODEL)
            _log.info("Loaded mmod face detector: %s", config.FACE_DETECTOR_MODEL)
        except Exception as exc:
            _log.error("Failed to load mmod detector %s: %s", config.FACE_DETECTOR_MODEL, exc)
            ok = False
    else:
        _log.info("FACE_DETECTOR_FORCE_HOG=True — skipping mmod, using HOG only")

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

    return ok


# ── Internal detection helpers ────────────────────────────────────────────────

def _detect_rects(rgb: np.ndarray) -> list:
    """Run the active detector and return a list of dlib rectangles.

    Detections below config.FACE_DETECTOR_MIN_CONFIDENCE are dropped. Background
    clutter the HOG detector reports comes back as LOW-confidence detections, so a
    confidence gate removes phantom faces (which were spawning bogus "unknown
    person" prompts in a messy room) without discarding small/distant real faces —
    a min-SIZE gate would instead drop exactly the distant face we want to keep.
    """
    global _use_hog, _slow_count

    upsample = max(0, int(getattr(config, "FACE_DETECTOR_UPSAMPLE", 1) or 0))
    min_conf = float(getattr(config, "FACE_DETECTOR_MIN_CONFIDENCE", 0.0) or 0.0)

    if _use_hog:
        # Scored overload: run() returns (rects, scores, sub_detector_idx). The
        # plain __call__ overload returns rects only, with no score to gate on.
        rects, scores, _ = _hog_detector.run(rgb, upsample, 0.0)
        return [r for r, s in zip(rects, scores) if s >= min_conf]

    t0 = time.monotonic()
    cnn_dets = _cnn_detector(rgb, upsample)
    elapsed = time.monotonic() - t0
    # mmod exposes per-detection .confidence (a different scale than HOG; the
    # default threshold is HOG-tuned, so this mainly affects the HOG path actually
    # in use). Keep detections at or above the gate.
    rects = [d.rect for d in cnn_dets if float(getattr(d, "confidence", 1.0)) >= min_conf]

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
        encoding      float32 numpy array face embedding — 512-dim L2-normalized
                      ArcFace (insightface backend) or 128-dim dlib descriptor
        landmarks     int32 numpy array of (x, y) points — (5, 2) keypoints
                      (insightface) or (68, 2) shape-predictor points (dlib)
        confidence    detector score (insightface only)
    Returns an empty list if no faces found, frame is None, or models unavailable.
    """
    if frame is None:
        return []
    if not _load_models():
        return []

    if _active_backend == "insightface":
        return _detect_faces_insightface(frame)
    return _detect_faces_dlib(frame)


def _detect_faces_insightface(frame: np.ndarray) -> list[dict]:
    """SCRFD detect + ArcFace embed. FaceAnalysis.get expects BGR (cv2-native)."""
    min_conf = float(getattr(config, "INSIGHTFACE_MIN_CONFIDENCE", 0.5) or 0.0)
    fh, fw = frame.shape[:2]

    try:
        faces = _insight_app.get(frame)
    except Exception as exc:
        _log.warning("InsightFace detection failed: %s", exc)
        return []

    results = []
    for f in faces:
        score = float(getattr(f, "det_score", 1.0))
        if score < min_conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        x = max(0, int(round(x1)))
        y = max(0, int(round(y1)))
        w = min(fw, int(round(x2))) - x
        h = min(fh, int(round(y2))) - y
        if w <= 0 or h <= 0:
            continue
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            continue
        kps = getattr(f, "kps", None)
        landmarks = (np.asarray(kps, dtype=np.int32)
                     if kps is not None else np.zeros((0, 2), dtype=np.int32))
        results.append({
            "bounding_box": (x, y, w, h),
            "encoding":     np.asarray(emb, dtype=np.float32),
            "landmarks":    landmarks,
            "confidence":   score,
        })
    return results


def _detect_faces_dlib(frame: np.ndarray) -> list[dict]:
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
    Look up a face embedding (512-dim ArcFace or 128-dim dlib) in the people
    database. Returns the matching person dict, or None if no match clears the
    backend-appropriate threshold.
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


def enroll_unknown_face(
    person_id: int,
    frame: np.ndarray,
    *,
    allow_largest_fallback: bool = False,
) -> bool:
    """
    Like enroll_face, but when multiple faces are present, pick the LARGEST face
    that does NOT already match an existing person in the DB. Used when Rex
    enrolls a newcomer while still seeing the known conversational partner.
    """
    faces = detect_faces(frame)
    if not faces:
        _log.warning("enroll_unknown_face: no face detected (person_id=%d)", person_id)
        return False

    # Filter to faces that DON'T identify as any existing known person.
    unknown_faces = []
    for f in faces:
        match = identify_face(f["encoding"])
        if match is None:
            unknown_faces.append(f)

    if not unknown_faces:
        if not allow_largest_fallback:
            _log.warning(
                "enroll_unknown_face: no unknown face found in frame; refusing largest-face fallback (person_id=%d)",
                person_id,
            )
            return False
        _log.warning(
            "enroll_unknown_face: no unknown face found in frame, falling back to largest (person_id=%d)",
            person_id,
        )
        unknown_faces = faces

    target = _largest_face(unknown_faces)
    result = people.add_biometric(person_id, "face", target["encoding"])
    if result is not None:
        _log.info("enroll_unknown_face: biometric stored for person_id=%d", person_id)
        return True
    _log.error("enroll_unknown_face: database write failed for person_id=%d", person_id)
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

    # Tier 2: don't build a physical-appearance dossier on a child/teen. Keep only the
    # coarse age markers (so family-safe mode + child question-suppression engage); skip
    # build / hair / height / notable_features for a minor.
    is_minor = str(data.get("age_category") or "").strip().lower() in {"child", "teen"}
    _MINOR_ALLOWED_KEYS = {"age_category", "age_range"}

    for key, value in data.items():
        if value is None:
            continue
        if is_minor and key not in _MINOR_ALLOWED_KEYS:
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

    _log.info(
        "update_appearance: appearance profile updated for person_id=%d%s",
        person_id, " (minor — age only)" if is_minor else "",
    )


def detect_mood(frame: np.ndarray) -> Optional[dict]:
    """
    Send a frame to GPT-4o and read the mood of the most prominent face.

    Returns dict {mood, confidence, notes} or None on failure.
        mood        one of "happy", "sad", "tired", "angry", "surprised",
                    "anxious", "neutral"
        confidence  0.0–1.0
        notes       short phrase describing the expression, or "" if unclear
    """
    try:
        import apikeys
        from openai import OpenAI
    except ImportError as exc:
        _log.error("detect_mood: missing dependency — %s", exc)
        return None

    b64 = encode_jpeg_base64(frame, quality=85)
    if b64 is None:
        _log.error("detect_mood: failed to JPEG-encode frame")
        return None

    detail = config.VISION_DETAIL.get("mood_analysis", "low")

    prompt = (
        "Look at the most prominent person's face in this image and assess "
        "their apparent mood from their facial expression. Return ONLY a JSON "
        "object with exactly these keys:\n"
        '  "mood": one of "happy"|"sad"|"tired"|"angry"|"surprised"|"anxious"|"neutral",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "notes": one short phrase describing the expression '
        '(e.g. "soft smile", "furrowed brow", "tight jaw"), or "" if unclear.\n'
        "If no face is clearly visible, return mood=\"neutral\" with "
        "confidence=0. Return only the JSON object, no other text."
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
            max_tokens=120,
        )
    except Exception as exc:
        _log.error("detect_mood: GPT-4o error: %s", exc)
        return None

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            _log.error("detect_mood: non-JSON response: %.120s", raw)
            return None
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            _log.error("detect_mood: JSON parse error: %s", exc)
            return None

    mood = str(data.get("mood", "") or "").strip().lower()
    if not mood:
        return None
    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    notes = str(data.get("notes", "") or "").strip()

    _log.info("detect_mood: %s (confidence=%.2f) — %s", mood, confidence, notes or "—")
    return {"mood": mood, "confidence": confidence, "notes": notes}


def detect_group_moods(frame: np.ndarray, max_people: int = 2) -> list[dict]:
    """
    Send a frame to GPT-4o and read apparent moods for up to max_people visible faces.

    Intended for small startup groups, not crowd analysis. Returns a list of
    dicts with {mood, confidence, notes}. Empty list on failure.
    """
    try:
        import apikeys
        from openai import OpenAI
    except ImportError as exc:
        _log.error("detect_group_moods: missing dependency — %s", exc)
        return []

    b64 = encode_jpeg_base64(frame, quality=85)
    if b64 is None:
        _log.error("detect_group_moods: failed to JPEG-encode frame")
        return []

    max_people = max(1, int(max_people or 1))
    detail = config.VISION_DETAIL.get("mood_analysis", "low")
    prompt = (
        f"Look at the people in this image and assess apparent facial mood for "
        f"up to {max_people} clearly visible faces, left to right. Return ONLY a "
        "JSON array. Each item must have exactly these keys:\n"
        '  "mood": one of "happy"|"sad"|"tired"|"angry"|"surprised"|"anxious"|"neutral",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "notes": one short phrase describing the expression.\n'
        "If a face is unclear, use mood=\"neutral\" and low confidence. "
        "No markdown, no explanation."
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
            max_tokens=220,
        )
    except Exception as exc:
        _log.error("detect_group_moods: GPT-4o error: %s", exc)
        return []

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            _log.error("detect_group_moods: non-JSON response: %.120s", raw)
            return []
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            _log.error("detect_group_moods: JSON parse error: %s", exc)
            return []
    if not isinstance(data, list):
        _log.error("detect_group_moods: expected list, got: %.120s", raw)
        return []

    moods: list[dict] = []
    for item in data[:max_people]:
        if not isinstance(item, dict):
            continue
        mood = str(item.get("mood", "") or "").strip().lower() or "neutral"
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        notes = str(item.get("notes", "") or "").strip()
        moods.append({"mood": mood, "confidence": confidence, "notes": notes})

    _log.info("detect_group_moods: %s", moods)
    return moods
