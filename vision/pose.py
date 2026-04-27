"""
vision/pose.py — MediaPipe Pose estimation, gesture, pose, and engagement detection.

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
from typing import Optional

import numpy as np

import config
from vision.image_utils import bgr_to_rgb
from world_state import world_state

_log = logging.getLogger(__name__)

# ── Model handles ─────────────────────────────────────────────────────────────

_pose        = None   # mediapipe Pose solution instance
_mp_pose     = None   # mediapipe.solutions.pose module reference
_mp_drawing  = None   # mediapipe.solutions.drawing_utils (optional)
_mp_ok       = False
_mp_attempted = False

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

# raising_hand: wrist must be this much above the shoulder (in y, pointing upward)
_RAISE_Y_MARGIN = 0.05

# waving: wrist y between nose y and shoulder y, and arm extended laterally
# Lateral extension threshold: abs(wrist.x - shoulder.x) as fraction of frame width
_WAVE_LATERAL_MIN = 0.10

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

def _load_model() -> bool:
    global _pose, _mp_pose, _mp_ok, _mp_attempted

    if _mp_attempted:
        return _mp_ok
    _mp_attempted = True

    try:
        import mediapipe as mp
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "pose"):
            _log.info(
                "MediaPipe body pose unavailable: installed mediapipe %s does not "
                "expose the legacy mp.solutions.pose API. Body pose/gesture cues "
                "disabled; face-based proxemics remain active.",
                getattr(mp, "__version__", "unknown"),
            )
            return False
        _mp_pose = solutions.pose
        _pose    = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # 0=lite, 1=full, 2=heavy — balance speed/accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _mp_ok = True
        _log.info("MediaPipe Pose loaded (model_complexity=1)")
    except ImportError:
        _log.warning("mediapipe not installed — pose detection unavailable")
    except Exception as exc:
        _log.error("Failed to init MediaPipe Pose: %s", exc)

    return _mp_ok


# ── Landmark extraction ───────────────────────────────────────────────────────

def _lm_dict(results) -> dict[str, tuple[float, float, float]]:
    """
    Convert MediaPipe PoseLandmarks to a dict keyed by landmark name.
    Values are (x, y, visibility) all in [0.0, 1.0].
    """
    if not results or not results.pose_landmarks:
        return {}

    names = [lm.name for lm in _mp_pose.PoseLandmark]
    lms   = results.pose_landmarks.landmark
    return {
        name: (lms[i].x, lms[i].y, lms[i].visibility)
        for i, name in enumerate(names)
    }


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

    # ── raising_hand ──────────────────────────────────────────────────────────
    # Rule: either wrist.y < shoulder.y - _RAISE_Y_MARGIN (wrist above shoulder).
    # y decreases upward in image coordinates, so wrist_y < shoulder_y means raised.
    if ls and lw and lw[1] < ls[1] - _RAISE_Y_MARGIN:
        return "raising_hand"
    if rs and rw and rw[1] < rs[1] - _RAISE_Y_MARGIN:
        return "raising_hand"

    # ── waving ────────────────────────────────────────────────────────────────
    # Rule: wrist is at approximately face/ear height (wrist.y between nose.y and
    # shoulder.y) AND the arm is extended laterally away from the shoulder.
    # Lateral threshold: abs(wrist.x - shoulder.x) > _WAVE_LATERAL_MIN of frame width.
    # This catches a hand held up near the face with an open lateral arm extension.
    if nose and ls and lw:
        wrist_at_face_height = nose[1] <= lw[1] <= ls[1]
        arm_extended = abs(lw[0] - ls[0]) > _WAVE_LATERAL_MIN
        if wrist_at_face_height and arm_extended:
            return "waving"
    if nose and rs and rw:
        wrist_at_face_height = nose[1] <= rw[1] <= rs[1]
        arm_extended = abs(rw[0] - rs[0]) > _WAVE_LATERAL_MIN
        if wrist_at_face_height and arm_extended:
            return "waving"

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
        rgb = bgr_to_rgb(frame)
        results = _pose.process(rgb)
    except Exception as exc:
        _log.warning("MediaPipe Pose processing error: %s", exc)
        return []

    kp = _lm_dict(results)
    if not kp:
        _update_world_state([])
        return []

    gesture    = _classify_gesture(kp)
    pose_label = _classify_pose(kp)
    engagement = _classify_engagement(pose_label, gesture)
    age        = get_age_category(kp)

    # Compute position from nose or shoulder midpoint for world_state
    nose = _get(kp, "NOSE")
    ls   = _get(kp, "LEFT_SHOULDER")
    rs   = _get(kp, "RIGHT_SHOULDER")
    if nose:
        position = (nose[0], nose[1])
    elif ls and rs:
        position = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    else:
        position = (0.5, 0.5)

    person = {
        "keypoints":  kp,
        "gesture":    gesture,
        "pose":       pose_label,
        "engagement": engagement,
        "age_estimate": age,
        "position":   position,
    }

    _log.debug("detect_pose: gesture=%s pose=%s engagement=%s age=%s",
               gesture, pose_label, engagement, age)

    people = [person]
    _update_world_state(people)
    return people


def _update_world_state(detected: list[dict]) -> None:
    """
    Merge pose detection results into world_state.people.

    Strategy: update existing entries by index (person 0 gets the first detected
    person's pose data, etc.). Entries beyond the detected count have pose fields
    cleared. This keeps face_id / voice_id assigned by other pipeline stages intact.
    """
    current = world_state.get("people")

    updated = list(current)

    for i, person_data in enumerate(detected):
        pose_fields = {
            "pose":         person_data["pose"],
            "gesture":      person_data["gesture"],
            "engagement":   person_data["engagement"],
            "age_estimate": person_data["age_estimate"],
            "position":     person_data["position"],
        }
        if i < len(updated):
            updated[i] = {**updated[i], **pose_fields}
        else:
            updated.append({
                "id":            f"person_{i+1}",
                "face_id":       None,
                "voice_id":      None,
                "distance_zone": None,
                **pose_fields,
            })

    # Clear pose fields on any existing entries that no longer have a detected person
    for i in range(len(detected), len(updated)):
        updated[i] = {
            **updated[i],
            "pose":       None,
            "gesture":    None,
            "engagement": None,
        }

    world_state.update("people", updated)
