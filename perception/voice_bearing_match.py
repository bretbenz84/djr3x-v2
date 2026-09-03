"""
Tie a voice bearing to a face on camera — perception/voice_bearing_match.py

Two frames meet here. A FACE has a bearing off the body's nose made of the neck
yaw plus the face's offset across the camera frame (both + = Rex's RIGHT — the
face-tracking convention, see motion_agency._come_bearing_deg). A VOICE has a
bearing from the Flex XVF3800's direction of arrival in the BASE frame
(+ = LEFT/CCW, the turn convention, hardware/flex_doa.py). This module converts
the face into the base frame (one negation, here and nowhere else) and asks,
for a given voice, which visible face — if any — the voice came from.

Pure functions, no world_state access, so the identity code and the bench tool
(tools/voice_face_test.py) share one piece of math and the unit tests pin it.

Outputs of ``match_faces_to_voice``:
  confirm_pid   the nearest face's person id when it lies within TOLERANCE of the
                voice — "the voice came from where this face is"
  selected      that face's person dict when it is also UNAMBIGUOUS (the next
                face is at least MARGIN farther from the voice) — safe to treat
                as THE visible speaker among several faces
  contradicts   True when every visible face is farther than CONTRADICTION from
                the voice — the talker is not any face on camera
"""

from __future__ import annotations

from typing import Optional


def wrap180(deg: float) -> float:
    d = (float(deg) + 180.0) % 360.0
    return d - 180.0 if d != 0.0 else 180.0


def face_offset_fraction(person: dict, frame_width: float) -> Optional[float]:
    """Signed horizontal offset of the face box centre from frame centre as a
    fraction of the half-width, + = toward Rex's RIGHT (larger x)."""
    if not person:
        return None
    box = person.get("face_box") or person.get("bounding_box") or person.get("bbox")
    if not box or len(box) < 4:
        return None
    try:
        x, _y, w, _h = (float(v) for v in box[:4])
        width = float(frame_width)
    except (TypeError, ValueError):
        return None
    if width <= 0.0:
        return None
    centre = x + w / 2.0
    return max(-1.0, min(1.0, (centre - width / 2.0) / (width / 2.0)))


def face_bearing_deg(person: dict, neck_yaw_right_deg: float, *,
                     frame_width: float, half_fov_deg: float,
                     px_per_deg: Optional[float] = None,
                     yaw_offset_deg: float = 0.0) -> Optional[float]:
    """A visible face's bearing in the BASE frame (+ = left), or None without a box.

    neck_yaw_right_deg: head yaw off the body's nose, + = right.
    px_per_deg: the lens model — pixels of horizontal offset per degree off the
      optical axis, treated as linear (the robot camera is a fisheye, whose
      equidistant projection IS close to linear). Calibrated against the voice
      bearing with `tools/voice_face_test.py --fit` (2026-09-02: ~16 px/deg).
      When None/0 the legacy fraction × half_fov_deg model is used.
    yaw_offset_deg: constant added to the camera's yaw (+ = right) — a mount
      offset between the camera axis and the body's nose the neck readback does
      not know about. Also fitted by --fit (it cannot be told apart from an
      unread neck yaw by a static test; see the doc).
    """
    frac = face_offset_fraction(person, frame_width)
    if frac is None:
        return None
    try:
        k = float(px_per_deg or 0.0)
    except (TypeError, ValueError):
        k = 0.0
    if k > 0.0:
        cam_deg = frac * (float(frame_width) / 2.0) / k
    else:
        cam_deg = frac * float(half_fov_deg)
    right_deg = float(neck_yaw_right_deg or 0.0) + float(yaw_offset_deg or 0.0) + cam_deg
    return wrap180(-right_deg)


def match_faces_to_voice(people: list, voice_bearing_deg: float, neck_yaw_right_deg: float, *,
                         frame_width: float, half_fov_deg: float,
                         tolerance_deg: float, margin_deg: float,
                         contradiction_deg: float,
                         px_per_deg: Optional[float] = None,
                         yaw_offset_deg: float = 0.0) -> dict:
    """Rank visible faces by angular distance from the voice bearing.

    Faces without a usable box are skipped. Returns
    {"voice_deg", "faces": [{"pid", "bearing_deg", "delta_deg", "person"}, ... nearest first],
     "confirm_pid", "selected", "contradicts", "nearest_delta_deg"}.
    """
    voice = wrap180(voice_bearing_deg)
    rows = []
    for person in people or []:
        if not isinstance(person, dict):
            continue
        bearing = face_bearing_deg(person, neck_yaw_right_deg,
                                   frame_width=frame_width, half_fov_deg=half_fov_deg,
                                   px_per_deg=px_per_deg, yaw_offset_deg=yaw_offset_deg)
        if bearing is None:
            continue
        pid = person.get("person_db_id")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        rows.append({"pid": pid, "bearing_deg": bearing,
                     "delta_deg": abs(wrap180(bearing - voice)), "person": person})
    rows.sort(key=lambda r: r["delta_deg"])
    out = {"voice_deg": voice, "faces": rows, "confirm_pid": None, "selected": None,
           "contradicts": False, "nearest_delta_deg": None}
    if not rows:
        return out
    nearest = rows[0]
    out["nearest_delta_deg"] = nearest["delta_deg"]
    if nearest["delta_deg"] <= float(tolerance_deg):
        out["confirm_pid"] = nearest["pid"]
        unambiguous = (len(rows) == 1
                       or (rows[1]["delta_deg"] - nearest["delta_deg"]) >= float(margin_deg))
        if unambiguous and nearest["pid"] is not None:
            out["selected"] = nearest["person"]
    elif nearest["delta_deg"] > float(contradiction_deg):
        out["contradicts"] = True
    return out


def describe(result: Optional[dict], names: Optional[dict] = None) -> str:
    """One-line log rendering: 'voice -18° vs faces Bret -20° (Δ2) | PJ +31° (Δ49) → Bret'."""
    if not result:
        return "no bearing match"
    names = names or {}
    parts = []
    for r in result.get("faces") or []:
        label = names.get(r["pid"]) or (f"person {r['pid']}" if r["pid"] is not None else "unknown")
        parts.append(f"{label} {r['bearing_deg']:+.0f}° (Δ{r['delta_deg']:.0f})")
    sel = result.get("selected")
    if sel is not None:
        verdict = f"→ {names.get(result.get('confirm_pid')) or result.get('confirm_pid')}"
    elif result.get("confirm_pid") is not None:
        verdict = f"→ nearest {names.get(result.get('confirm_pid')) or result.get('confirm_pid')} (ambiguous)"
    elif result.get("contradicts"):
        verdict = "→ no face there (off-camera talker)"
    else:
        verdict = "→ inconclusive"
    return f"voice {result.get('voice_deg', 0.0):+.0f}° vs faces {' | '.join(parts) or 'none'} {verdict}"


def fit_camera_model(samples: list) -> Optional[dict]:
    """Calibrate the lens against the voice: least squares for px_per_deg and
    yaw_offset_deg from takes where ONE face was on camera while its owner talked.

    samples: [(px_offset, voice_bearing_deg, neck_yaw_right_deg), ...] with
      px_offset = face centre − frame centre in pixels (+ = right).
    Model: voice = −(neck + offset + px / k)  ⇒  −voice − neck = offset + px · (1/k).
    Returns {"px_per_deg", "yaw_offset_deg", "rms_deg", "n", "residuals"} or
    None with fewer than two usable samples. Two samples give an exact fit;
    the rms only means something from three up, and a spread of positions
    (centre, both edges) is what makes the two unknowns separable.
    """
    rows = []
    for px, voice, neck in samples or []:
        try:
            rows.append((float(px), -float(voice) - float(neck or 0.0)))
        except (TypeError, ValueError):
            continue
    n = len(rows)
    if n < 2:
        return None
    sx = sum(px for px, _ in rows); sy = sum(y for _, y in rows)
    sxx = sum(px * px for px, _ in rows); sxy = sum(px * y for px, y in rows)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None                       # every face at the same x — no scale information
    slope = (n * sxy - sx * sy) / denom   # degrees per pixel
    offset = (sy - slope * sx) / n
    if slope <= 0.0:
        return None                       # a face moving right must read more right
    residuals = [y - (offset + slope * px) for px, y in rows]
    rms = (sum(r * r for r in residuals) / n) ** 0.5
    return {"px_per_deg": 1.0 / slope, "yaw_offset_deg": offset, "rms_deg": rms,
            "n": n, "residuals": residuals}
