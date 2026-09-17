"""Read-only servo-to-rig mapping, shared with Blender export and GUI.

Coordinates retain the source assembly's Z-up metres. No hardware imports or
commands: input values are normalized using the runtime's calibrated channels.
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
    m['head'] = m['neck'] @ around(spec['head_pivot'], spec['head_axis'], -v['headtilt'])
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
