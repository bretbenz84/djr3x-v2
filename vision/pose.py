"""
vision/pose.py — MediaPipe Pose estimation, gesture, pose, and engagement detection.

Uses the modern MediaPipe Tasks API (``mp.tasks.vision.PoseLandmarker``). The legacy
``mp.solutions.pose`` Python solution was REMOVED in mediapipe 0.10.x, so the old
loader silently failed and no gesture (including "waving") was ever published — which
is why wave-back never fired on-device. This mirrors the same Tasks-API pattern already
used by ``vision/face_expression.py`` and ``vision/animal_detector.py`` and loads a
downloaded ``.task`` model file.

MediaPipe Pose processes one person per call. Multi-person support would require
running a person detector first and cropping individual bounding boxes before passing
each crop through this pipeline.

All landmark coordinates from MediaPipe are normalized (0.0–1.0, origin top-left,
y increases downward). Visibility values are in [0.0, 1.0].

Gesture, pose, and engagement are classified from keypoint geometry using simple
geometric heuristics. All thresholds are documented inline — tune these constants
after real-world testing to adjust sensitivity.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

import config
from vision.image_utils import bgr_to_rgb
from world_state import world_state

_log = logging.getLogger(__name__)

# ── Model handles (MediaPipe Tasks API) ───────────────────────────────────────

_landmarker   = None   # mp.tasks.vision.PoseLandmarker instance
_mp           = None   # mediapipe module reference (for mp.Image)
_landmark_names: list[str] = []   # ordered PoseLandmark names (33-point skeleton)
_load_ok      = False
_load_attempted = False
_model_lock   = threading.Lock()
_last_timestamp_ms = 0

# Dedicated sampling loop (mirrors vision/face_expression.py). Pose runs in its own
# thread, NOT pull-based off the ~1 Hz consciousness tick, so the GUI skeleton overlay
# and wave-back stay live instead of jumping once a second.
_stop_event   = threading.Event()
_thread: Optional[threading.Thread] = None

# Recent raised-wrist (timestamp, normalized_x) samples, so wave-back can mirror how fast
# the user is waving. Written by the pose loop, read from the consciousness thread.
_wave_motion: "deque[tuple[float, float]]" = deque(maxlen=16)
_wave_motion_lock = threading.Lock()

# ── Visibility threshold — landmarks below this are treated as not-detected ──
_VIS_MIN = 0.4

# ── Landmark index aliases (MediaPipe Pose 33-point skeleton) ─────────────────
# Full reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_NOSE          = 0
_LEFT_EYE      = 2
_RIGHT_EYE     = 5
_LEFT_EAR      = 7
_RIGHT_EAR     = 8
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW    = 13
_RIGHT_ELBOW   = 14
_LEFT_WRIST    = 15
_RIGHT_WRIST   = 16
_LEFT_HIP      = 23
_RIGHT_HIP     = 24
_LEFT_KNEE     = 25
_RIGHT_KNEE    = 26
_LEFT_ANKLE    = 27
_RIGHT_ANKLE   = 28


# ── Gesture classification thresholds ────────────────────────────────────────
# All values are in normalized frame coordinates (0.0–1.0) unless stated.
# y=0 is top of frame, y=1 is bottom — "above" means smaller y value.

# raised hand (waving / raising_hand): wrist this much above the shoulder (in y)
_RAISE_Y_MARGIN = 0.05

# waving: a raised wrist must also sit this far to the SIDE of its shoulder
# (abs(wrist.x - shoulder.x) as a fraction of frame width) to count as a wave
# rather than a hand raised straight up. Kept low so a greeting wave near the head
# is caught at any swing phase; a straight-up hand stays "raising_hand".
_WAVE_LATERAL_MIN = 0.07

# crossed_arms: both wrists within this fraction of shoulder_width from shoulder midpoint
# Shoulder width = abs(left_shoulder.x - right_shoulder.x)
_CROSS_CENTER_FRACTION = 0.25

# pointing: wrist y within this margin of shoulder y (arm roughly horizontal)
# AND wrist is far from body center laterally (>= _POINT_LATERAL_MIN of frame width)
_POINT_Y_MARGIN    = 0.12
_POINT_LATERAL_MIN = 0.28

# leaning_in: nose x offset from shoulder midpoint > this fraction of shoulder width
_LEAN_NOSE_FRACTION = 0.35


# ── Pose classification thresholds ───────────────────────────────────────────
# facing_forward: shoulder y difference below this → shoulders roughly level
_FACING_SHOULDER_Y_DIFF = 0.08

# facing_forward: ear distance asymmetry must be below this fraction
# asymmetry = abs(|left_ear.x - nose.x| - |right_ear.x - nose.x|)
#             / max(|left_ear.x - nose.x|, |right_ear.x - nose.x|)
_EAR_ASYMMETRY_MAX = 0.4

# side_on: shoulder_x_distance < this fraction of frame width suggests profile view
# (shoulders appear close together when person is turned sideways)
_SIDE_ON_SHOULDER_X_MAX = 0.15


# ── Engagement thresholds ─────────────────────────────────────────────────────
# These are derived from gesture + pose:
# high:   facing_forward AND not crossed_arms
# medium: side_on OR facing_forward with crossed_arms
# low:    facing_away


# ── Model loading ─────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_path() -> Path:
    configured = Path(getattr(config, "MEDIAPIPE_POSE_LANDMARKER_MODEL", ""))
    return configured if configured.is_absolute() else _project_root() / configured


def _load_model() -> bool:
    global _landmarker, _mp, _landmark_names, _load_ok, _load_attempted

    if not bool(getattr(config, "POSE_DETECTION_ENABLED", True)):
        return False
    if _load_attempted:
        return _load_ok
    _load_attempted = True

    model_path = _model_path()
    if not model_path.exists():
        _log.warning(
            "MediaPipe Pose Landmarker model missing: %s — run setup_assets.py. "
            "Body pose/gesture cues (including wave-back) disabled.",
            model_path,
        )
        return False

    try:
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        max_people = max(1, int(getattr(config, "POSE_MAX_PEOPLE", 3)))
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=float(
                getattr(config, "POSE_MIN_DETECTION_CONFIDENCE", 0.6)),
            min_pose_presence_confidence=float(
                getattr(config, "POSE_MIN_PRESENCE_CONFIDENCE", 0.5)),
            min_tracking_confidence=float(
                getattr(config, "POSE_MIN_TRACKING_CONFIDENCE", 0.5)),
        )
        _landmarker = PoseLandmarker.create_from_options(options)
        _mp = mp
        _landmark_names = [lm.name for lm in mp.tasks.vision.PoseLandmark]
        _load_ok = True
        _log.info("MediaPipe Pose Landmarker loaded (Tasks API): %s", model_path)
    except ImportError:
        _log.warning("mediapipe not installed — pose detection unavailable")
    except Exception as exc:
        _log.error("Failed to init MediaPipe Pose Landmarker: %s", exc)

    return _load_ok


def _next_timestamp_ms() -> int:
    """Monotonic, strictly-increasing timestamps for detect_for_video (VIDEO mode)."""
    global _last_timestamp_ms
    ts = int(time.monotonic() * 1000)
    if ts <= _last_timestamp_ms:
        ts = _last_timestamp_ms + 1
    _last_timestamp_ms = ts
    return ts


# ── Landmark extraction ───────────────────────────────────────────────────────

def _lm_dict(result, index: int = 0) -> dict[str, tuple[float, float, float]]:
    """
    Convert a MediaPipe Tasks PoseLandmarkerResult to a dict keyed by landmark name.
    Values are (x, y, visibility) all in [0.0, 1.0].

    The Tasks API exposes ``result.pose_landmarks`` as a LIST of per-pose landmark
    lists (one entry per detected person), unlike the legacy single-pose
    ``results.pose_landmarks.landmark``.
    """
    poses = getattr(result, "pose_landmarks", None) or []
    if index >= len(poses):
        return {}

    landmarks = poses[index]
    out: dict[str, tuple[float, float, float]] = {}
    for i, lm in enumerate(landmarks):
        if i >= len(_landmark_names):
            break
        vis = getattr(lm, "visibility", None)
        # pose_landmarker populates visibility; default a missing value to "visible"
        # so a model that omits it doesn't blank out every landmark via _VIS_MIN.
        vis = 1.0 if vis is None else float(vis)
        out[_landmark_names[i]] = (float(lm.x), float(lm.y), vis)
    return out


def _get(kp: dict, name: str) -> Optional[tuple[float, float, float]]:
    """Return (x, y, vis) for a landmark, or None if visibility is below threshold."""
    entry = kp.get(name)
    if entry is None or entry[2] < _VIS_MIN:
        return None
    return entry


def _midpoint(a: Optional[tuple], b: Optional[tuple]) -> Optional[tuple]:
    """Return midpoint (x, y) of two (x, y, vis) tuples, or None if either is missing."""
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ── Gesture classification ────────────────────────────────────────────────────

def _classify_gesture(kp: dict) -> str:
    """
    Classify the dominant gesture from keypoints. Rules applied in priority order:
    raising_hand → waving → crossed_arms → pointing → leaning_in → neutral.

    All rules use normalized coordinates (0=top-left, 1=bottom-right, y down).
    """
    ls = _get(kp, "LEFT_SHOULDER")
    rs = _get(kp, "RIGHT_SHOULDER")
    lw = _get(kp, "LEFT_WRIST")
    rw = _get(kp, "RIGHT_WRIST")
    le = _get(kp, "LEFT_ELBOW")
    re = _get(kp, "RIGHT_ELBOW")
    lh = _get(kp, "LEFT_HIP")
    rh = _get(kp, "RIGHT_HIP")
    nose = _get(kp, "NOSE")

    # ── waving ────────────────────────────────────────────────────────────────
    # A wave is a hand RAISED above the shoulder (covering a hand up by the face,
    # the head, or above it) and held OUT to the side of the body (a lateral offset
    # of the wrist from its shoulder ≥ _WAVE_LATERAL_MIN of frame width).
    #
    # This is checked BEFORE raising_hand on purpose: an enthusiastic wave puts the
    # wrist well above the shoulder, so the old order swallowed it as "raising_hand"
    # and the wave-back never fired. Detection is single-frame posture — the pose
    # pipeline samples far too slowly to resolve the side-to-side motion, and a
    # raised open hand pointed at Rex reads as a greeting regardless of phase.
    for shoulder, wrist in ((ls, lw), (rs, rw)):
        if shoulder and wrist and wrist[1] <= shoulder[1] - _RAISE_Y_MARGIN:
            if abs(wrist[0] - shoulder[0]) >= _WAVE_LATERAL_MIN:
                return "waving"

    # ── raising_hand ──────────────────────────────────────────────────────────
    # Wrist above the shoulder but NOT out to the side (e.g. raised straight up to
    # answer / get attention). y decreases upward, so wrist_y < shoulder_y is raised.
    for shoulder, wrist in ((ls, lw), (rs, rw)):
        if shoulder and wrist and wrist[1] <= shoulder[1] - _RAISE_Y_MARGIN:
            return "raising_hand"

    # ── crossed_arms ──────────────────────────────────────────────────────────
    # Rule: both wrists are near the torso centerline (within _CROSS_CENTER_FRACTION
    # of shoulder width from shoulder midpoint), and both wrists are between shoulder
    # y and hip y (at chest/belly level, not at sides or raised).
    # Crossed wrists pull toward the opposite shoulder, reducing lateral extent.
    if ls and rs and lw and rw:
        shoulder_mid_x  = (ls[0] + rs[0]) / 2
        shoulder_width  = abs(ls[0] - rs[0])
        center_threshold = shoulder_width * _CROSS_CENTER_FRACTION
        lw_near_center  = abs(lw[0] - shoulder_mid_x) < center_threshold + shoulder_width * 0.5
        rw_near_center  = abs(rw[0] - shoulder_mid_x) < center_threshold + shoulder_width * 0.5
        shoulder_mid_y  = (ls[1] + rs[1]) / 2
        hip_mid_y       = ((lh[1] + rh[1]) / 2) if (lh and rh) else (shoulder_mid_y + 0.25)
        lw_at_torso     = shoulder_mid_y <= lw[1] <= hip_mid_y
        rw_at_torso     = shoulder_mid_y <= rw[1] <= hip_mid_y
        if lw_near_center and rw_near_center and lw_at_torso and rw_at_torso:
            return "crossed_arms"

    # ── pointing ──────────────────────────────────────────────────────────────
    # Rule: one wrist is roughly at shoulder height (wrist.y within _POINT_Y_MARGIN
    # of shoulder.y) and is far from the body laterally
    # (abs(wrist.x - shoulder_midpoint.x) > _POINT_LATERAL_MIN of frame width).
    # Elbow should also be roughly extended (elbow x between shoulder x and wrist x).
    if ls and rs:
        shoulder_mid_x = (ls[0] + rs[0]) / 2
    else:
        shoulder_mid_x = None

    if shoulder_mid_x is not None:
        if ls and lw and le:
            wrist_at_shoulder_height = abs(lw[1] - ls[1]) < _POINT_Y_MARGIN
            wrist_far_lateral        = abs(lw[0] - shoulder_mid_x) > _POINT_LATERAL_MIN
            elbow_between            = min(ls[0], lw[0]) <= le[0] <= max(ls[0], lw[0])
            if wrist_at_shoulder_height and wrist_far_lateral and elbow_between:
                return "pointing"
        if rs and rw and re:
            wrist_at_shoulder_height = abs(rw[1] - rs[1]) < _POINT_Y_MARGIN
            wrist_far_lateral        = abs(rw[0] - shoulder_mid_x) > _POINT_LATERAL_MIN
            elbow_between            = min(rs[0], rw[0]) <= re[0] <= max(rs[0], rw[0])
            if wrist_at_shoulder_height and wrist_far_lateral and elbow_between:
                return "pointing"

    # ── leaning_in ────────────────────────────────────────────────────────────
    # Rule: nose x is offset from shoulder midpoint by more than _LEAN_NOSE_FRACTION
    # of shoulder width. This catches a lateral upper-body lean toward Rex.
    # Forward lean (toward camera) is better captured by bounding-box size in proxemics.
    if nose and ls and rs:
        shoulder_mid_x = (ls[0] + rs[0]) / 2
        shoulder_width = abs(ls[0] - rs[0]) or 0.1
        nose_offset    = abs(nose[0] - shoulder_mid_x)
        if nose_offset > _LEAN_NOSE_FRACTION * shoulder_width:
            return "leaning_in"

    return "neutral"


# ── Pose (body orientation) classification ───────────────────────────────────

def _classify_pose(kp: dict) -> str:
    """
    Classify body orientation relative to the camera.

    facing_forward: shoulders level, both ears visible with symmetric ear-nose distances.
    facing_away:    nose has low visibility or face landmarks absent.
    side_on:        shoulders appear foreshortened (close x-positions) or strong ear asymmetry.
    """
    ls   = _get(kp, "LEFT_SHOULDER")
    rs   = _get(kp, "RIGHT_SHOULDER")
    le   = _get(kp, "LEFT_EAR")
    re   = _get(kp, "RIGHT_EAR")
    nose = _get(kp, "NOSE")

    # facing_away: nose is not detected (low visibility means face turned away)
    if nose is None:
        return "facing_away"

    # side_on check 1: shoulders are close together in x — person turned sideways,
    # causing shoulders to foreshorten and appear at nearly the same x position.
    if ls and rs:
        shoulder_x_dist = abs(ls[0] - rs[0])
        if shoulder_x_dist < _SIDE_ON_SHOULDER_X_MAX:
            return "side_on"

    # side_on check 2: strong ear visibility asymmetry.
    # When facing sideways, one ear is behind the head and becomes invisible.
    # Asymmetry = normalized difference between left and right ear-nose distances.
    if le and re and nose:
        dist_le = abs(le[0] - nose[0])
        dist_re = abs(re[0] - nose[0])
        max_dist = max(dist_le, dist_re, 0.001)
        ear_asymmetry = abs(dist_le - dist_re) / max_dist
        if ear_asymmetry > _EAR_ASYMMETRY_MAX:
            return "side_on"

    # facing_forward: shoulders roughly level (small y difference)
    if ls and rs:
        shoulder_y_diff = abs(ls[1] - rs[1])
        if shoulder_y_diff < _FACING_SHOULDER_Y_DIFF:
            return "facing_forward"

    # Default: if we have a visible nose and level-ish body, assume side_on as a
    # conservative middle-ground (avoids false "facing_forward" with partial data).
    return "side_on"


# ── Engagement classification ─────────────────────────────────────────────────

def _classify_engagement(pose: str, gesture: str) -> str:
    """
    Estimate engagement level from pose and gesture.

    high:   facing_forward AND no closed body language (not crossed_arms)
    medium: side_on, or facing_forward with crossed_arms or neutral gesture
    low:    facing_away, or crossed_arms while side_on
    """
    if pose == "facing_away":
        return "low"
    if pose == "side_on" and gesture == "crossed_arms":
        return "low"
    if pose == "facing_forward" and gesture not in ("crossed_arms",):
        return "high"
    return "medium"


# ── Age estimation ────────────────────────────────────────────────────────────

def get_age_category(keypoints: dict) -> str:
    """
    Estimate age category from skeletal proportions.

    Returns one of: "child", "teen", "adult".

    Method: head-width-to-shoulder-width ratio.
    Children have proportionally larger heads relative to shoulder width.
    Ear separation is used as a proxy for head width; shoulder landmark
    separation is used for shoulder width.

    Thresholds (tunable — these are starting estimates):
      head/shoulder > 0.60  → child
      head/shoulder > 0.48  → teen
      otherwise             → adult

    Fallback: if ankle landmarks are visible, limb-to-torso ratio is used as
    a second signal. Children have proportionally shorter limbs (ratio < 1.4).
    """
    le   = keypoints.get("LEFT_EAR")
    re   = keypoints.get("RIGHT_EAR")
    ls   = keypoints.get("LEFT_SHOULDER")
    rs   = keypoints.get("RIGHT_SHOULDER")
    lh   = keypoints.get("LEFT_HIP")
    rh   = keypoints.get("RIGHT_HIP")
    la   = keypoints.get("LEFT_ANKLE")
    ra   = keypoints.get("RIGHT_ANKLE")
    le_elbow = keypoints.get("LEFT_ELBOW")
    re_elbow = keypoints.get("RIGHT_ELBOW")
    lw   = keypoints.get("LEFT_WRIST")
    rw   = keypoints.get("RIGHT_WRIST")

    # ── Primary: head-to-shoulder width ratio ─────────────────────────────────
    # head_width ≈ ear separation
    # shoulder_width = shoulder landmark separation
    if le and re and ls and rs and le[2] >= _VIS_MIN and re[2] >= _VIS_MIN:
        head_width     = abs(le[0] - re[0])
        shoulder_width = abs(ls[0] - rs[0])
        if shoulder_width > 0.01:
            ratio = head_width / shoulder_width
            # Children (~0-12): head is large relative to narrow shoulders
            if ratio > 0.60:
                return "child"
            # Teens (~13-17): intermediate
            if ratio > 0.48:
                return "teen"
            return "adult"

    # ── Fallback: limb-to-torso ratio using arm length / torso height ─────────
    # Torso height ≈ shoulder midpoint y to hip midpoint y.
    # Upper arm ≈ distance(shoulder, elbow), lower arm ≈ distance(elbow, wrist).
    # Children have limb_ratio < ~1.4 (shorter arms relative to torso).
    if ls and rs and lh and rh and le_elbow and lw:
        torso_h = abs(((ls[1] + rs[1]) / 2) - ((lh[1] + rh[1]) / 2))
        upper_arm = np.hypot(ls[0] - le_elbow[0], ls[1] - le_elbow[1])
        lower_arm = np.hypot(le_elbow[0] - lw[0], le_elbow[1] - lw[1])
        if torso_h > 0.01:
            limb_ratio = (upper_arm + lower_arm) / torso_h
            if limb_ratio < 1.1:
                return "child"
            if limb_ratio < 1.4:
                return "teen"
            return "adult"

    return "adult"  # conservative default when landmarks are insufficient


# ── Phantom-pose rejection ──────────────────────────────────────────────────────

def _is_plausible_pose(kp: dict) -> bool:
    """True if ``kp`` looks like a real human body rather than a MediaPipe phantom.

    At ``num_poses>1`` the detector will hallucinate weak skeletons onto bright blobs —
    ceiling lights, reflections, monitors. The reliable tell is the SHOULDER GIRDLE: a
    real upper body shows two confidently-visible shoulders separated by a plausible
    width (or, side-on, one strong shoulder plus a hip forming a torso column). A phantom
    collapses to low-visibility and/or near-zero width. Tunable via POSE_* config; the
    whole filter is bypassable with POSE_PHANTOM_FILTER_ENABLED=False."""
    if not kp:
        return False
    if not bool(getattr(config, "POSE_PHANTOM_FILTER_ENABLED", True)):
        return True

    vmin = float(getattr(config, "POSE_MIN_TORSO_VISIBILITY", 0.6))

    def _vis(name: str) -> float:
        e = kp.get(name)
        return float(e[2]) if e else 0.0

    ls, rs = kp.get("LEFT_SHOULDER"), kp.get("RIGHT_SHOULDER")
    ls_v, rs_v = _vis("LEFT_SHOULDER"), _vis("RIGHT_SHOULDER")
    lh_v, rh_v = _vis("LEFT_HIP"), _vis("RIGHT_HIP")

    # Primary: both shoulders confidently visible AND plausibly separated (rejects a
    # collapsed blob whose two "shoulders" sit on top of each other).
    if ls_v >= vmin and rs_v >= vmin and ls and rs:
        width = abs(float(ls[0]) - float(rs[0]))
        if width >= float(getattr(config, "POSE_MIN_SHOULDER_WIDTH", 0.04)):
            return True

    # Side-on fallback: one strong shoulder + one strong hip = a real vertical torso.
    if max(ls_v, rs_v) >= vmin and max(lh_v, rh_v) >= vmin:
        return True

    return False


# ── Public API ────────────────────────────────────────────────────────────────

def detect_pose(frame) -> list[dict]:
    """
    Detect pose from an OpenCV BGR frame.

    Returns a list of person dicts. Currently always 0 or 1 elements because
    MediaPipe Pose processes a single person. Each dict contains:
        keypoints  dict of landmark name → (x, y, visibility)  — normalized coords
        gesture    str: neutral | raising_hand | waving | crossed_arms | pointing | leaning_in
        pose       str: facing_forward | facing_away | side_on
        engagement str: high | medium | low

    Clears world_state.people pose fields and re-populates from detected results.
    Returns [] if frame is None or MediaPipe is unavailable.
    """
    if frame is None:
        return []
    if not _load_model():
        return []

    try:
        rgb = np.ascontiguousarray(bgr_to_rgb(frame))
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        with _model_lock:
            results = _landmarker.detect_for_video(mp_image, _next_timestamp_ms())
    except Exception as exc:
        _log.warning("MediaPipe Pose processing error: %s", exc)
        return []

    frame_h = int(getattr(frame, "shape", [0, 0])[0] or 0)
    frame_w = int(getattr(frame, "shape", [0, 0, 0])[1] or 0)

    poses = getattr(results, "pose_landmarks", None) or []
    people: list[dict] = []
    wave_kp: Optional[dict] = None
    dropped_phantoms = 0
    for idx in range(len(poses)):
        kp = _lm_dict(results, idx)
        if not kp:
            continue
        # Reject phantom skeletons (ceiling lights, reflections) before they become a
        # "person" and get drawn / tracked. Real bodies have a visible shoulder girdle.
        if not _is_plausible_pose(kp):
            dropped_phantoms += 1
            continue

        gesture    = _classify_gesture(kp)
        pose_label = _classify_pose(kp)
        engagement = _classify_engagement(pose_label, gesture)
        age        = get_age_category(kp)

        # Compute position from nose or shoulder midpoint for world_state matching.
        nose = _get(kp, "NOSE")
        ls   = _get(kp, "LEFT_SHOULDER")
        rs   = _get(kp, "RIGHT_SHOULDER")
        if nose:
            position = (nose[0], nose[1])
        elif ls and rs:
            position = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        else:
            position = (0.5, 0.5)

        people.append({
            "keypoints":  kp,
            "gesture":    gesture,
            "pose":       pose_label,
            "engagement": engagement,
            "age_estimate": age,
            "position":   position,
        })
        # Wave-speed mirroring keeps ONE wave history — prefer the person actually
        # waving, else fall back to the first detected body.
        if wave_kp is None and gesture == "waving":
            wave_kp = kp

    # Track the raised wrist's lateral position over time so wave-back can mirror the
    # speed of the user's wave (see recent_wave_speed()).
    _record_wave_motion(wave_kp or (people[0]["keypoints"] if people else {}))

    if not people:
        if dropped_phantoms:
            _log.debug("detect_pose: 0 real poses (dropped %d phantom)", dropped_phantoms)
        _update_world_state([], frame_w, frame_h)
        return []

    _log.debug("detect_pose: %d pose(s) gestures=%s (dropped %d phantom)",
               len(people), [p["gesture"] for p in people], dropped_phantoms)

    _update_world_state(people, frame_w, frame_h)
    return people


def _update_world_state(detected: list[dict], frame_w: int = 0, frame_h: int = 0) -> None:
    """
    Merge multi-person pose detection results into world_state.people.

    Each detected pose is bound to the person slot whose FACE BOX is closest to that
    pose's head (proximity match in normalized coords), so a body skeleton / gesture
    lands on the RIGHT person in a group instead of always slot 0. Slots that get no
    pose this tick have their pose fields cleared. When no slot has a face box yet
    (face pipeline hasn't populated boxes), falls back to index order so an early /
    face-less pose still shows a skeleton. ``frame_w/h`` convert the pixel face boxes
    to normalized space for the match.
    """
    def _pose_fields(person_data):
        return {
            "pose":         person_data["pose"],
            "gesture":      person_data["gesture"],
            "engagement":   person_data["engagement"],
            "age_estimate": person_data["age_estimate"],
            "position":     person_data["position"],
            # Normalized landmark dict (name -> (x, y, visibility)) so the GUI can draw a
            # live skeleton overlay. NOTE: consciousness._step_person_recognition rebuilds
            # people on its tick and only carries forward an allowlist of decoration fields
            # — "pose_keypoints" MUST be in that `decor` tuple or the skeleton flickers off.
            "pose_keypoints": person_data.get("keypoints"),
        }

    def _slot_face_center(slot):
        box = slot.get("face_box") if isinstance(slot, dict) else None
        if (isinstance(box, (list, tuple)) and len(box) >= 4
                and frame_w > 0 and frame_h > 0):
            x, y, w, h = [float(v) for v in box[:4]]
            return ((x + w / 2.0) / float(frame_w), (y + h / 2.0) / float(frame_h))
        return None

    def _merge_pose(current):
        updated = list(current)
        centers = [_slot_face_center(s) for s in updated]
        have_anchors = any(c is not None for c in centers)
        used: set[int] = set()  # slot indices that received a pose this tick

        if have_anchors:
            # Greedy nearest-match: bind each pose to the closest face slot within range.
            thresh = float(getattr(config, "POSE_FACE_MATCH_MAX_DIST", 0.22))
            pairs = []
            for pi, pd in enumerate(detected):
                px, py = pd.get("position") or (0.5, 0.5)
                for si, c in enumerate(centers):
                    if c is None:
                        continue
                    d = ((px - c[0]) ** 2 + (py - c[1]) ** 2) ** 0.5
                    pairs.append((d, pi, si))
            pairs.sort(key=lambda t: t[0])
            matched_pose: set[int] = set()
            for d, pi, si in pairs:
                if pi in matched_pose or si in used or d > thresh:
                    continue
                updated[si] = {**updated[si], **_pose_fields(detected[pi])}
                matched_pose.add(pi)
                used.add(si)
            # A pose with no nearby face slot → attach to a face-less slot, else append.
            for pi, pd in enumerate(detected):
                if pi in matched_pose:
                    continue
                si = next((k for k in range(len(updated))
                           if k not in used and centers[k] is None), None)
                if si is not None:
                    updated[si] = {**updated[si], **_pose_fields(pd)}
                    used.add(si)
                else:
                    updated.append({
                        "id": f"person_{len(updated) + 1}", "face_id": None,
                        "voice_id": None, "distance_zone": None, **_pose_fields(pd),
                    })
                    used.add(len(updated) - 1)
        else:
            # No face anchors yet — legacy index order keeps a single/early pose working.
            for pi, pd in enumerate(detected):
                if pi < len(updated):
                    updated[pi] = {**updated[pi], **_pose_fields(pd)}
                else:
                    updated.append({
                        "id": f"person_{pi + 1}", "face_id": None,
                        "voice_id": None, "distance_zone": None, **_pose_fields(pd),
                    })
                used.add(pi)

        # Clear pose fields on any slot that received no pose this tick.
        for si in range(len(updated)):
            if si not in used:
                updated[si] = {
                    **updated[si],
                    "pose":           None,
                    "gesture":        None,
                    "engagement":     None,
                    "pose_keypoints": None,
                }

        return updated

    # Read-modify-write under the world_state lock so a concurrent face/identity
    # write (which sets person_db_id) isn't reverted by a stale snapshot.
    world_state.mutate("people", _merge_pose)


def _raised_wrist_x(kp: dict) -> Optional[float]:
    """Normalized x of the most-raised wrist (above its shoulder), or None. Tracks the
    waving hand's lateral position across frames for speed measurement."""
    ls, rs = _get(kp, "LEFT_SHOULDER"), _get(kp, "RIGHT_SHOULDER")
    lw, rw = _get(kp, "LEFT_WRIST"), _get(kp, "RIGHT_WRIST")
    best = None  # (height_above_shoulder, wrist_x)
    for shoulder, wrist in ((ls, lw), (rs, rw)):
        if shoulder and wrist and wrist[1] <= shoulder[1] - _RAISE_Y_MARGIN:
            height = shoulder[1] - wrist[1]
            if best is None or height > best[0]:
                best = (height, wrist[0])
    return best[1] if best else None


def _record_wave_motion(kp: dict) -> None:
    """Append the raised wrist's lateral position (with a timestamp) to the motion history.
    No raised wrist → nothing recorded (old samples age out of recent_wave_speed's window)."""
    x = _raised_wrist_x(kp)
    if x is None:
        return
    with _wave_motion_lock:
        _wave_motion.append((time.monotonic(), float(x)))


def recent_wave_speed() -> Optional[float]:
    """Mean absolute lateral wrist speed (normalized-x units / second) over the recent
    raised-wrist samples, or None if there aren't enough fresh ones. Higher = a faster wave.

    Velocity-based (path length / elapsed time) rather than frequency-based, so it stays
    meaningful at the ~5 Hz pose rate without needing full back-and-forth cycles.
    """
    window = float(getattr(config, "WAVE_SPEED_WINDOW_SECS", 1.2))
    now = time.monotonic()
    with _wave_motion_lock:
        samples = [(t, x) for (t, x) in _wave_motion if (now - t) <= window]
    if len(samples) < 3:
        return None
    path = sum(abs(samples[i + 1][1] - samples[i][1]) for i in range(len(samples) - 1))
    dt = samples[-1][0] - samples[0][0]
    if dt <= 0:
        return None
    return path / dt


def _anchor_from_kp(kp, frame_w: int, frame_h: int):
    """``(nose_x, nose_y, head_width)`` in PIXELS for one pose's normalized keypoints,
    or None. ``head_width`` is a per-person scale (ear span, else eye span, else a
    fraction of shoulder width) so the tolerance adapts to how close the person is."""
    if not isinstance(kp, dict) or not kp:
        return None

    def _pt(name):
        v = kp.get(name)
        if isinstance(v, (list, tuple)) and len(v) >= 3 and float(v[2]) >= _VIS_MIN:
            return (float(v[0]), float(v[1]))
        return None

    nose = _pt("NOSE") or _pt("LEFT_EYE") or _pt("RIGHT_EYE")
    if nose is None:
        return None

    le, re = _pt("LEFT_EAR"), _pt("RIGHT_EAR")
    leye, reye = _pt("LEFT_EYE"), _pt("RIGHT_EYE")
    ls, rs = _pt("LEFT_SHOULDER"), _pt("RIGHT_SHOULDER")
    if le and re:
        head_w = abs(le[0] - re[0])
    elif leye and reye:
        head_w = abs(leye[0] - reye[0]) * 2.2
    elif ls and rs:
        head_w = abs(ls[0] - rs[0]) * 0.45
    else:
        head_w = 0.08  # normalized fallback (~8% of frame width)
    head_w = max(0.05, head_w)
    return (nose[0] * frame_w, nose[1] * frame_h, head_w * frame_w)


def head_anchors_px(frame_w: int, frame_h: int) -> list:
    """ALL detected pose heads in PIXELS — a list of ``(nose_x, nose_y, head_width)``,
    one per posed body currently in world_state.

    The pose's own face landmarks (nose / eyes / ears) track each head reliably even
    when dlib throws a phantom face elsewhere, so the face guard can keep a real face
    near ANY tracked body and reject only faces far from EVERY body (so a second real
    person is no longer dropped as a phantom)."""
    if frame_w <= 0 or frame_h <= 0:
        return []
    try:
        people = world_state.get("people") or []
    except Exception:
        return []
    anchors = []
    for person in people:
        kp = person.get("pose_keypoints") if isinstance(person, dict) else None
        a = _anchor_from_kp(kp, frame_w, frame_h)
        if a is not None:
            anchors.append(a)
    return anchors


def head_anchor_px(frame_w: int, frame_h: int):
    """The FIRST detected pose head in PIXELS (``(nose_x, nose_y, head_width)``), or
    None. Kept for back-compat; prefer head_anchors_px for the multi-person guard."""
    anchors = head_anchors_px(frame_w, frame_h)
    return anchors[0] if anchors else None


def process_frame(frame) -> list[dict]:
    """Detect pose on one frame and publish it to world_state. Loop body / test seam."""
    return detect_pose(frame)


def _loop() -> None:
    from vision import camera

    interval = max(0.05, float(getattr(config, "POSE_ANALYSIS_INTERVAL_SECS", 0.2) or 0.2))
    while not _stop_event.is_set():
        try:
            frame = camera.get_frame()
            if frame is not None:
                process_frame(frame)
        except Exception as exc:
            _log.debug("pose detection loop error: %s", exc)
        _stop_event.wait(interval)


def start() -> None:
    """Start the background pose-sampling loop (idempotent)."""
    global _thread
    if not bool(getattr(config, "POSE_DETECTION_ENABLED", True)):
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, daemon=True, name="pose-detection")
    _thread.start()
    _log.info(
        "MediaPipe pose detection started (interval=%.2fs)",
        float(getattr(config, "POSE_ANALYSIS_INTERVAL_SECS", 0.2) or 0.2),
    )


def stop() -> None:
    """Stop the sampling loop and release the MediaPipe Pose Landmarker (idempotent)."""
    global _thread, _landmarker, _mp, _load_attempted, _load_ok
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=2.0)
    _thread = None
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
    with _wave_motion_lock:
        _wave_motion.clear()


def _reset_for_tests() -> None:
    global _last_timestamp_ms
    stop()
    _stop_event.clear()
    _last_timestamp_ms = 0
