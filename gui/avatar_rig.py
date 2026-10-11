"""Read-only servo-to-rig mapping, shared with Blender export and GUI.

Coordinates retain the source assembly's Z-up metres. No hardware imports or
commands: input values are normalized using the runtime's calibrated channels.

The head is not a servo norm any more (October 10 2026 rework: two servos on one
post). The rig takes ``head_pitch`` (norm 0 = fully down, 1 = fully up, level at
:data:`LEVEL_PITCH_NORM`) and ``head_tilt`` (norm 0 = cocked to Rex's left, .5
straight, 1 = right); :func:`pitch_norm` / :func:`tilt_norm` turn the
``hardware.head_pose`` fractions into those norms. The visor norm keeps its
meaning (0 closed, 1 open) even though the servo is now inverted — callers feed
``head_pose.visor_frac``, never the raw pulse fraction.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'rex'
SPEC = json.loads((ASSET_DIR / 'rig.json').read_text())

def translation(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m

def rotation(axis, degrees):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    m = np.eye(4)
    m[:3, :3] = c * np.eye(3) + (1-c) * np.outer(a, a) + s*k
    return m

def around(p, axis, angle):
    return translation(p) @ rotation(axis, angle) @ translation(-np.asarray(p))

def point(m, p):
    return (m @ np.r_[p, 1])[:3]

def align(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    dot = float(np.clip(a @ b, -1, 1))
    if np.linalg.norm(axis) < 1e-9:
        if dot > 0:
            return np.eye(4)
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0])
    return rotation(axis, math.degrees(math.acos(dot)))

def level_pitch_norm(spec=SPEC):
    """The head_pitch norm at which the rig's head is level (0 degrees)."""
    lo, hi = spec['limits']['head_pitch']
    return (0 - lo) / (hi - lo) if hi != lo else .5

LEVEL_PITCH_NORM = level_pitch_norm()

def pitch_norm(pitch, spec=SPEC):
    """A head_pose pitch fraction (-1 down .. +1 up) as the rig's head_pitch norm.

    Piecewise so that pitch 0 lands exactly on the rig's level angle even though
    the drawn travel is asymmetric (15 down, 30 up)."""
    try:
        pitch = float(pitch)
    except (TypeError, ValueError):
        pitch = 0.
    if not math.isfinite(pitch):
        pitch = 0.
    pitch = max(-1., min(1., pitch))
    level = level_pitch_norm(spec)
    return level + pitch * (level if pitch < 0 else 1 - level)

def tilt_norm(tilt):
    """A head_pose tilt fraction (-1 Rex's left .. +1 right) as the rig's head_tilt norm."""
    try:
        tilt = float(tilt)
    except (TypeError, ValueError):
        tilt = 0.
    if not math.isfinite(tilt):
        tilt = 0.
    return .5 + .5 * max(-1., min(1., tilt))

def servo_values(norms, spec=SPEC):
    values = {}
    for name, (lo, hi) in spec['limits'].items():
        n = float(norms.get(name, .5))
        if not math.isfinite(n):
            n = .5
        values[name] = lo + (hi-lo) * max(0, min(1, n))
    values['headlift'] -= spec['lift_reference'] * (spec['limits']['headlift'][1] - spec['limits']['headlift'][0])
    return values

def pose_matrices(norms, spec=SPEC):
    v = servo_values(norms, spec)
    m = {'fixed': np.eye(4)}
    m['poker'] = rotation([0, 0, 1], v['pokerarm'])
    m['hero'] = rotation([0, 0, 1], v['heroarm'])
    m['elbow'] = m['hero'] @ around(spec['elbow_pivot'], spec['elbow_axis'], v['elbow'])
    m['wrist'] = m['elbow'] @ around(spec['wrist_pivot'], spec['wrist_axis'], v['hand'])
    m['neck'] = translation([0, 0, v['headlift']]) @ rotation([0, 0, 1], v['neck'])
    # Pitch about the left-right head axis (positive degrees = up), then a side roll
    # about the forward (camera) axis: positive cocks the head to Rex's own right.
    m['head'] = (m['neck'] @ around(spec['head_pivot'], spec['head_axis'], -v['head_pitch'])
                 @ around(spec['head_pivot'], spec['head_forward'], v['head_tilt']))
    m['visor'] = m['head'] @ around(spec['visor_pivot'], spec['visor_axis'], v['visor'])
    lower, upper = np.array(spec['piston_lower']), np.array(spec['piston_upper'])
    a, b = point(m['hero'], lower), point(m['elbow'], upper)
    r = m['hero'] @ align(upper-lower, point(np.linalg.inv(m['hero']), b)-lower)
    m['piston_body'] = translation(b) @ r @ translation(-upper)
    m['piston_rod'] = translation(a) @ r @ translation(-lower)
    return m

def spring_points(matrices, spec=SPEC, count=181):
    a = np.array(spec['spring_bottom'])
    b = point(matrices['head'], spec['spring_top'])
    t = np.linspace(0, 1, count)
    p = a[None, :] + t[:, None] * (b-a)[None, :]
    p[:, 0] += .045 * np.cos(t * 5 * math.pi)
    p[:, 1] += .045 * np.sin(t * 5 * math.pi)
    return p
