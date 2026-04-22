"""
vision/proxemics.py — Proxemic zone and approach-vector estimation.

Distance zones are derived from the ratio of a face bounding box's width to the
total frame width. A larger face means the person is closer to the camera.

Thresholds are configurable in config.py:
    PROXEMICS_INTIMATE_MIN_FRACTION  — bbox_width / frame_width above this → intimate
    PROXEMICS_SOCIAL_MIN_FRACTION    — bbox_width / frame_width above this → social
    Below PROXEMICS_SOCIAL_MIN_FRACTION → public

Approach detection compares bounding box area between frames to determine whether
a person is moving toward or away from the camera.
"""

import logging

import config

_log = logging.getLogger(__name__)

# Approach detection: minimum fractional change in bounding box area to register
# as movement. Below this threshold in either direction → stationary.
# (area_current - area_prev) / area_prev > +_APPROACH_THRESHOLD → approaching
# (area_current - area_prev) / area_prev < -_APPROACH_THRESHOLD → retreating
_APPROACH_THRESHOLD = 0.05


def get_distance_zone(bounding_box: tuple, frame_width: int) -> str:
    """
    Classify a person's distance from the camera into a proxemic zone.

    Args:
        bounding_box: (x, y, w, h) face bounding box in pixels.
        frame_width:  total width of the camera frame in pixels.

    Returns one of: "intimate", "social", "public".

    Classification rules (all ratios are bbox_w / frame_width):
        intimate: ratio >= PROXEMICS_INTIMATE_MIN_FRACTION  (default 0.65)
                  Person's face fills most of the frame — very close.
        social:   ratio >= PROXEMICS_SOCIAL_MIN_FRACTION    (default 0.30)
                  Normal conversation distance, face clearly visible.
        public:   ratio < PROXEMICS_SOCIAL_MIN_FRACTION
                  Person is far away, face is small in frame.
    """
    if frame_width <= 0:
        _log.warning("get_distance_zone: frame_width must be > 0, got %d", frame_width)
        return "public"

    _x, _y, bbox_w, _h = bounding_box
    ratio = bbox_w / frame_width

    if ratio >= config.PROXEMICS_INTIMATE_MIN_FRACTION:
        return "intimate"
    if ratio >= config.PROXEMICS_SOCIAL_MIN_FRACTION:
        return "social"
    return "public"


def get_approach_vector(
    current_box: tuple,
    previous_box: tuple,
) -> str:
    """
    Determine whether a person is approaching, retreating, or stationary.

    Args:
        current_box:  (x, y, w, h) bounding box in the current frame.
        previous_box: (x, y, w, h) bounding box in the previous frame.

    Returns one of: "approaching", "retreating", "stationary".

    Classification rule:
        area_change = (current_area - previous_area) / previous_area
        approaching: area_change  >  +_APPROACH_THRESHOLD (default +0.05)
        retreating:  area_change  <  -_APPROACH_THRESHOLD (default -0.05)
        stationary:  abs(area_change) <= _APPROACH_THRESHOLD

    Bounding box area (w * h) grows as a person moves closer to the camera and
    shrinks as they move away, making it a reliable proxy for approach direction.
    """
    _cx, _cy, cw, ch = current_box
    _px, _py, pw, ph = previous_box

    current_area  = cw * ch
    previous_area = pw * ph

    if previous_area <= 0:
        _log.debug("get_approach_vector: previous_area is zero, returning stationary")
        return "stationary"

    change = (current_area - previous_area) / previous_area

    if change > _APPROACH_THRESHOLD:
        return "approaching"
    if change < -_APPROACH_THRESHOLD:
        return "retreating"
    return "stationary"
