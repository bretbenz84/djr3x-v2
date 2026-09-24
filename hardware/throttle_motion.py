"""Pure throttle clearance/profile rules shared by runtime and supervised tours.

Positions are Maestro quarter-microseconds, not measured shaft angles. The boxes
encode the owner's empirical clearance observations, including independent joint
progress; they are not a Cartesian model of the hand.
"""
import json
import math
from pathlib import Path

CHANNELS = (8, 9, 10)
PARK = {8: 2272 * 4, 9: 2496 * 4, 10: 512 * 4}
TUCK = {8: 1636 * 4, 9: 2100 * 4, 10: 512 * 4}
RAISED = 544 * 4
FULL_DOWN = {8: 2272 * 4, 9: 5589, 10: 9511}
HIGH_FIVE = {8: 544 * 4, 9: 9361, 10: 6302}
FULL_DOWN_RAISED = {**FULL_DOWN, 8: RAISED}
RETRACT_RAISED = {**TUCK, 8: RAISED}


def full_down_corridor(pose):
    """Owner-verified recorded pose 3, with shoulder-only entry/exit."""
    return (RAISED - 2 <= pose[8] <= FULL_DOWN[8] + 2
            and all(abs(pose[ch] - FULL_DOWN[ch]) <= 2 for ch in (9, 10)))


STATE_FILE = Path(__file__).resolve().parents[1] / 'data' / 'throttle_arm_parked.json'


def clearance_box(start, end):
    # Owner explicitly verified simultaneous full-down -> high-five movement.
    # Keep this directed exception to the recorded targets; it does not change
    # the general low-shoulder elbow limit or approve the reverse transition.
    if full_down_corridor(start) and end == HIGH_FIVE:
        return True
    # Exact downstream configuration only; do not relax the general elbow or
    # wrist envelope. Also permits resuming a partially completed shoulder move.
    if full_down_corridor(start) and full_down_corridor(end):
        return True
    shoulder = max(start[8], end[8])
    elbow = min(start[9], end[9])
    wrist = max(start[10], end[10])
    minimum = 1546 if shoulder > 1702 * 4 else 636 if shoulder > 1636 * 4 else 500
    if elbow < minimum * 4:
        return False
    if shoulder <= RAISED:
        return True
    if shoulder <= 1384.5 * 4 and elbow >= 512 * 4 and wrist <= 2254.25 * 4:
        return True
    if shoulder <= 1636 * 4:
        return wrist <= 1500 * 4 or elbow >= 900 * 4
    return elbow >= 1546 * 4 and wrist <= 512 * 4


def validate_pose(pose, limits):
    if set(pose) != set(CHANNELS):
        raise ValueError('Throttle poses must contain exactly channels 8, 9, 10')
    # Intersect configured limits with the actual stored Maestro endpoints.
    board = {8: (544 * 4, 2272 * 4), 9: (512 * 4, 2496 * 4), 10: (512 * 4, 2496 * 4)}
    for ch, value in pose.items():
        lo, hi = board[ch]
        if (not isinstance(value, int) or
                not max(lo, limits[ch]['min']) <= value <= min(hi, limits[ch]['max'])):
            raise ValueError(f'Throttle target outside configured/board limits: {ch}={value}')


def profiles(start, end, limits, speed_caps, accel_caps, duration, *, pace=None):
    """Proportional travel profiles with optional independent joint pace.

    Duration sets the unpaced travel time; each joint's pace divides that time.
    Integer Maestro units and acceleration can lengthen arrival; the caller waits
    for pulse readback instead of assuming duration is a physical timing guarantee.
    """
    distances = {ch: abs(end[ch] - start[ch]) for ch in CHANNELS if end[ch] != start[ch]}
    if not distances:
        return {}
    pace = {ch: 1.0 if pace is None else pace[ch] for ch in distances}
    if any(not math.isfinite(value) or value <= 0 for value in pace.values()):
        raise ValueError('Throttle pace must be finite and positive')
    speeds = {ch: min(limits[ch]['speed'], speed_caps[ch]) for ch in distances}
    accels = {ch: min(limits[ch]['acceleration'], accel_caps[ch]) for ch in distances}
    if not math.isfinite(duration) or duration <= 0 or min(*speeds.values(), *accels.values()) < 1:
        raise ValueError('Throttle profiles must have positive bounded speed/acceleration')
    # Recover the unpaced coordinated profile, then accelerate each joint
    # separately. Scaling caps alone leaves duration-limited moves unchanged.
    speed_scale = max(duration * 100, max(distances[ch] * pace[ch] / speeds[ch] for ch in distances))
    accel_scale = max(distances[ch] * pace[ch] / accels[ch] for ch in distances)
    return {ch: (max(1, min(speeds[ch], round(distances[ch] * pace[ch] / speed_scale))),
                 max(1, min(accels[ch], round(distances[ch] * pace[ch] / accel_scale))))
            for ch in distances}


def invalidate_park():
    """Before any throttle target: a cold restart must not assume a clean park."""
    STATE_FILE.unlink(missing_ok=True)


def remember_park():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps({'park_us': [PARK[ch] / 4 for ch in CHANNELS]}))
    temporary.replace(STATE_FILE)


def cold_start_park_known():
    try:
        return json.loads(STATE_FILE.read_text()).get('park_us') == [PARK[ch] / 4 for ch in CHANNELS]
    except (OSError, ValueError, AttributeError):
        return False
