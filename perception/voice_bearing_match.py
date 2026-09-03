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
                     frame_width: float, half_fov_deg: float) -> Optional[float]:
    """A visible face's bearing in the BASE frame (+ = left), or None without a box.

    neck_yaw_right_deg: head yaw off the body's nose, + = right.
    half_fov_deg: degrees from frame centre to the frame edge (~25° on the
    robot camera — a −33° base turn moved a face 1290 px across 1920).
    """
    frac = face_offset_fraction(person, frame_width)
    if frac is None:
        return None
    right_deg = float(neck_yaw_right_deg or 0.0) + frac * float(half_fov_deg)
    return wrap180(-right_deg)


def match_faces_to_voice(people: list, voice_bearing_deg: float, neck_yaw_right_deg: float, *,
                         frame_width: float, half_fov_deg: float,
                         tolerance_deg: float, margin_deg: float,
                         contradiction_deg: float) -> dict:
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
                                   frame_width=frame_width, half_fov_deg=half_fov_deg)
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
