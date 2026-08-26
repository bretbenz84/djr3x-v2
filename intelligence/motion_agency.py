"""
intelligence/motion_agency.py — autonomous base motion (owner spec 2026-07-06).

Four behaviors, evaluated once per consciousness tick (~1 Hz), highest priority
first:

REQUESTED COME — after an explicit "come here" command, find the REQUESTER (the
voice-identified speaker, when known — other people's faces are skipped; an
anonymous requester accepts any known face), square the base to them off the
camera, then issue the firmware `come` command with a social stop distance. The
search is RADAR-FIRST (owner spec 2026-08-15): the LD2450 ring on the base
(hardware/radar.py) reports where bodies are, so with no face on camera he turns
straight to the best radar body instead of sweeping blind, dwells for the camera
to find the requester's face, and if that body is not them (no face, or someone
else's) marks the spot rejected and turns to the next. Camera evidence always
outranks radar: a visible/locked requester face goes straight to alignment, and a
fresh sighting turns back toward the sighting before radar is consulted. Radar
bearings smear while the base rotates, so radar decisions are made only from ring
frames received after a turn's `done` plus a settle, and a body must persist over
several frames. The blind sweep survives as the fallback when the ring is down,
quiet, or has no unvisited body. Every leg is followed by a settled-camera dwell
(keyed to the firmware `done`, not command issue) so the detect→identify pipeline
gets still frames to work with; after any turn, alignment measurements wait out a
short settle so a mid-slew neck can't produce oscillating corrections (field
2026-08-11: he circled the room twice, swept past the owner repeatedly, and timed
out while looking straight at him). The forward ToF target may be the person or an
intervening obstacle, so furniture and walls stop the approach just as safely as
the intended person does.

FLINCH — a reflexive back-off when someone crowds Rex from the front, the way an
animal edges back when you get in its face. Each front matrix ToF half (fl/fr,
floor-rejected) is watched on its OWN adaptive open-distance baseline, so a real
approach — fast or slow, from either side — is caught while static clutter on one
sensor can't mask it. When a side is inside MOTION_FLINCH_TRIGGER_M AND has closed
by MOTION_FLINCH_APPROACH_DROP_M off its (frozen) baseline for
MOTION_FLINCH_CONFIRM_TICKS consecutive ticks (so one noisy frame never lurches
him), he retreats a short step. A firmware BLOCKED-on-the-front state — a very close
or fast crowder the ~1 Hz sampler would otherwise skip — triggers the same back-off
immediately. He backs up ONLY to a point: the retreat is capped by the rear ToF
(rl/rr) so he leaves MOTION_FLINCH_REAR_MARGIN_M of clearance and stops SHORT of the
wall; cornered — or BLIND behind (rear sensors dead, where the firmware stop also
fails open) — he holds his ground rather than reverse into it. The firmware's
always-on rear-ToF stop is the hard backstop when the rear sensors report. Unlike
the two behaviors below, FLINCH needs no tracked/known person (someone can walk up
while Rex looks elsewhere) and — being a reflex — may fire even mid-sentence
(MOTION_FLINCH_ALLOW_MID_SENTENCE).

REALIGN — turn the base to face the person the head is tracking, but ONLY as the
last resort after the neck has done all it can (owner spec 2026-07-31: the neck
servo, not the wheels, is the primary way to keep the camera on someone — the base
was turning far too often). Two conditions must BOTH hold, each for
MOTION_FACE_CONFIRM_TICKS consecutive ticks:
  1. the neck sweep is EXHAUSTED — the neck sits past MOTION_FACE_NECK_FRACTION
     (near its travel limit) of its half-span, so it cannot pan further; and
  2. the face still sits meaningfully off-centre — past MOTION_FACE_EDGE_FRACTION
     of the half-width, on the same side the neck is pointing. With the neck at its
     limit, ANY sustained same-side offset means face-tracking can no longer
     re-center them, so this threshold only needs to clear tracking jitter/deadband
     (~0.06), not reach the physical frame edge (field 2026-07-31 20:14: neck pinned
     at its minimum, face 38% off-centre, and a 0.70 "extreme edge" bar meant the
     base never turned).
Then the base turns by a proportional chunk and face-tracking naturally re-centers
the neck as it comes around. Iterative small corrections + a cooldown, never one
exact spin (no oscillation).

APPROACH — when the tracked person stays at "public" distance (vision/proxemics:
face width < 30% of frame) for MOTION_APPROACH_CONFIRM_TICKS ticks AND the base is
already roughly facing them AND the front ToF confirms genuinely open floor ahead
(nothing within MOTION_APPROACH_MIN_START_M), issue `come`: the firmware turns to
heading 0 and advances until the nearest FORWARD ToF obstacle is
MOTION_APPROACH_STOP_AT_M away — the person's own body is the stop target, and
anything in between (furniture, wall) stops the base the same way. The ToF gate
exists because face width lies on a wide-angle lens: a face 3-4 ft away reads under
the "public" fraction, and Rex drove up on someone already well inside conversation
range (field 2026-07-31). No cliff sensing needed or used (owner: never upstairs).

Safety layering (all independent of this module):
  - firmware reflex: Z_STOP zone forces ST_BLOCKED regardless of host commands
  - drive deadman + comms watchdog on the ESP32
  - motion_controller._autonomous_allowed(): manual gamepad owner wins, paused
    interaction blocks, disconnected blocks
This module only DECIDES; it never streams velocities (turn/come are closed-loop
firmware commands), acts only from motion state "idle", one action per tick.

Kill switches: AUTONOMOUS_MOTION_ENABLED master; MOTION_FLINCH_ENABLED /
MOTION_FACE_PERSON_ENABLED / MOTION_APPROACH_ENABLED per behavior.
"""

import logging
import math
import random
import threading
import time
from typing import Optional

import config
import state as state_module
from hardware import motion
from intelligence import motion_controller
from state import State

_log = logging.getLogger(__name__)

# Per-behavior confirmation counters + cooldown stamps (reset by _reset()).
_state = {
    "neck_hits": 0,
    "far_hits": 0,
    "orient_hits": 0,        # consecutive nobody-on-camera ticks with a radar body
    "last_turn_at": 0.0,
    "last_approach_at": 0.0,
    "last_flinch_at": 0.0,
    "orient_last_at": 0.0,   # radar-orient cooldown stamp
    "orient_visited": [],    # (world_bearing_deg, at) — bodies already looked at
    "wander_pending": None,  # in-flight weight-shift pair (out leg + inverse)
    "wander_next_at": 0.0,   # randomized idle-wander cooldown stamp
    "edge_hits": 0,          # consecutive edge-in-eligible conversation ticks
    "edge_last_at": 0.0,     # edge-in cooldown stamp
    "object_step": None,     # armed step toward an asked-about object
    "object_step_at": 0.0,   # object-step cooldown stamp
    "first_step_at": 0.0,    # first LIVE autonomy tick (startup-approach window)
    "startup_approach_done": False,  # once per session
    "startup_hits": 0,       # startup-approach confirm counter
    "neck_strain_since": 0.0,  # comfort-realign timer: neck past the comfort
                               # fraction since this stamp (0 = relaxed)
    "user_motion_at": 0.0,   # last explicit voice motion command (stand-down window)
    "realign_pending_seq": None,   # realign turn awaiting its firmware verdict
    "traction_fails": 0,     # consecutive realigns that produced no actual rotation
    "no_traction_until": 0.0,
    "no_drive_log_at": 0.0,  # throttle for the room-rule log line
    "hold_at": None,         # "don't move" — latched until told to move again
                             # (None = not held; a stamp of 0.0 is a real hold, so
                             # the falsy-means-released shorthand is wrong here)
}


def _emit_traction_notice() -> None:
    """Tell the human ONCE why he stopped trying — silence would read as a freeze."""
    if not _flag("MOTION_TRACTION_ANNOUNCE_ENABLED", True):
        return
    try:
        from audio import speech_queue
        speech_queue.enqueue(
            str(getattr(config, "MOTION_TRACTION_NOTICE_LINE",
                        "My wheels can't get a grip on this floor — I'll stay put.")),
            emotion="neutral", priority=1, tag="no_traction",
        )
    except Exception as exc:
        _log.debug("traction notice failed: %s", exc)


def _traction_lost(now: float) -> bool:
    return now < float(_state.get("no_traction_until") or 0.0)


def note_traction_recovered(reason: str = "") -> None:
    """Clear the no-traction latch (a human command, or a turn that actually worked)."""
    if _state.get("traction_fails") or _state.get("no_traction_until"):
        _log.info("[motion_agency] traction latch cleared (%s)", reason or "recovered")
    _state["traction_fails"] = 0
    _state["no_traction_until"] = 0.0
    _state["realign_pending_seq"] = None


def note_user_hold(reason: str = "user said stop") -> None:
    """The human told him to STOP / not move. Unlike a steering command this is a
    standing instruction, so it LATCHES: realign and approach stay down until he is
    explicitly told to move again (or MOTION_STOP_STANDDOWN_SECS elapses, if set).

    Field 2026-07-25, the second carpet run: "Don't move." -> "Stopping." -> and 49 s
    later he was turning again, because a stop only armed the 45 s steering window.
    The owner's plain meaning is "stay put", not "pause briefly".

    It also must NOT clear the no-traction latch. Being told to stop is the opposite
    of evidence that the wheels found grip — that clear reset the abort streak to
    zero mid-count in the same run, so the carpet detector never reached its
    threshold. Only a real drive command clears it."""
    _state["hold_at"] = time.monotonic()
    _clear_idle_wander("user hold")   # never fire a pending inverse after "don't move"
    _log.info("[motion_agency] autonomous motion held (%s)", reason)


def release_user_hold(reason: str = "user commanded motion") -> None:
    if _state.get("hold_at") is not None:
        _log.info("[motion_agency] hold released (%s)", reason)
    _state["hold_at"] = None


def no_drive_room() -> "tuple[str | None, str | None] | None":
    """(room name, reason) when the room he currently believes he's in is flagged
    no-drive, else None. The flag rides on the belief itself (place_recognition
    publishes it), so this is a dict lookup, not a DB hit — safe on every tick."""
    if not _flag("MOTION_ROOM_NO_DRIVE_ENABLED", True):
        return None
    try:
        from perception import place_service
        belief = place_service.current_place()
    except Exception:
        return None
    if not belief or not belief.get("no_drive"):
        return None
    return str(belief.get("name") or "this room"), belief.get("no_drive_reason")


def _user_hold_active(now: float) -> bool:
    at = _state.get("hold_at")
    if at is None:
        return False
    window = _num("MOTION_STOP_STANDDOWN_SECS", 0.0)
    return True if window <= 0.0 else (now - float(at)) < window


def note_user_motion() -> None:
    """Record an explicit voice motion command. Also releases a stop-hold and clears
    the no-traction latch — the human is asking for movement, and may have carried
    him onto a floor he can actually turn on. The social realign/approach
    behaviors stand down for MOTION_USER_MOTION_STANDDOWN_SECS afterwards — the
    human deliberately pointed the body, and realign was rotating it right back
    (field 2026-07-23: "turn right a little" -> -45, then realign +30 toward the
    face 13 s later, reading as "I tell it to turn right, it turns left"). The
    flinch reflex and an explicit come-here request are unaffected."""
    _state["user_motion_at"] = time.monotonic()
    release_user_hold()
    note_traction_recovered("user commanded motion")


def _user_motion_standdown(now: float) -> bool:
    at = float(_state.get("user_motion_at") or 0.0)
    if at <= 0.0:
        return False
    window = _num("MOTION_USER_MOTION_STANDDOWN_SECS", 45.0)
    return (now - at) < window

_requested_come = {
    "active": False,
    "started_at": 0.0,
    "requester_id": None,   # person_db_id of the voice that asked, when known —
                            # the search then targets THEM and skips other faces
                            # (owner spec 2026-08-11: JT on the couch must not
                            # satisfy Bret's "come here")
    "search_turns": 0,
    "last_turn_at": 0.0,    # when the LAST chassis turn (align or scan) was issued
    "pending_turn_seq": None,  # a turn we issued whose firmware `done` hasn't landed —
                               # no search/align decision is made while the camera swings
    "turn_done_at": 0.0,    # when that `done` landed; dwell/settle windows key off this
    "scan_sign": 1.0,       # which side the person was last known on (sweep starts there)
    "last_seen_at": 0.0,    # last time face tracking held the person — sampled EVERY
                            # tick, including while the base is mid-turn (see step()):
                            # a sighting during a scan turn must not be thrown away
    "seen_sign": 0.0,       # which way to turn to re-center that sighting (+ = left)
    "seen_deg": 0.0,        # fused bearing (real degrees) AT the sighting moment
                            # (neck + face read together, so it is synchronized) —
                            # the resight turn uses this actual angle, not a fixed step
    "front_near_hits": 0,   # consecutive completed-drive ticks with the radial
                            # front reading NEAR — one speckled frame must not
                            # end the errand as "arrived" (field 2026-08-11
                            # 20:37: arrived at 0.62m nowhere near the requester)
    "approach_at": 0.0,     # when the last `come` was issued (retry pacing)
    "approaches": 0,        # how many times we've launched at them this errand
    "align_turns": 0,       # consecutive align turns without reaching "centered" —
                            # after MOTION_COME_ALIGN_MAX_TRIES a good-enough residual
                            # goes to the firmware as the `come` heading instead of
                            # another base turn (field 2026-08-11: ±12-45 deg align
                            # oscillation never settled and the approach never launched)
    "skip_log_at": 0.0,     # throttle for the "seeing X, waiting for requester" log
    # ── radar-first search (owner spec 2026-08-15) ─────────────────────────
    "radar_since": 0.0,     # only ring frames received at/after this monotonic stamp
                            # may drive a turn: a turn's `done` + settle, or the
                            # errand start when the base was already still
    "radar_turns": 0,       # radar-directed turns this errand (share the search budget)
    "radar_pending_world": None,  # world bearing of the body we are turning to /
                                  # dwelling on; becomes "visited" if the dwell finds
                                  # no requester face
    "radar_pending_since": 0.0,   # when that radar turn was issued (a sighting
                                  # AFTER it means the spot is not empty — don't reject)
    "radar_visited": [],    # world bearings of bodies already looked at and rejected
    "heading_mode": "cmd",  # "imu": world = imu.yaw + bearing (the base publishes
                            # a gyro heading); "cmd": world = sum of commanded turns
    "cmd_heading": 0.0,     # running sum of turns THIS module issued (cmd mode)
}

# Flinch detector state, sampled every idle tick and reset whenever the base is
# busy/exploring/gone (see _reset_flinch). Per FRONT SIDE (fl/fr, tracked separately
# so static clutter on one side can't mask a real approach on the other) we keep an
# adaptive "open-distance" baseline: it drifts toward the reading (capped per tick, so
# a single spurious far frame can't manufacture a big drop) while the front is CLEAR,
# and FREEZES the instant something enters personal space — so the "where they came
# from" reference survives even a long gated stretch (mid-sentence / paused) instead of
# decaying out of a fixed window. `hits` counts consecutive intruding ticks; a flinch
# needs MOTION_FLINCH_CONFIRM_TICKS of them, so one noisy frame never lurches him.
_flinch_state = {
    "baseline": {"fl": None, "fr": None},  # adaptive open distance per side (m) or None
    "clear_run": {"fl": 0, "fr": 0},       # consecutive CLEAR ticks per side (gates baseline rises)
    "hits": 0,                              # consecutive intruding ticks
    "last_corner_log_at": 0.0,             # throttle for the "cornered/blind, holding" log
    "last_veto_log_at": 0.0,               # throttle for the "held, uncorroborated" log
}


def _flag(name: str, default: bool = True) -> bool:
    return bool(getattr(config, name, default))


def _num(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def _reset(*counters: str) -> None:
    for key in counters:
        _state[key] = 0


def requested_come_active() -> bool:
    """Whether an explicit person-seeking come-here sequence owns the base."""
    return bool(_requested_come["active"])


def request_come_here(person_id: "int | None" = None, *,
                      behind: bool = False,
                      side_deg: "float | None" = None) -> bool:
    """Arm a bounded search/align/approach sequence for an explicit voice request.

    ``person_id`` is the voice-identified requester (person_db_id). When known, the
    search goes to THAT face and skips everyone else until it finds them — with two
    people in the room, "the first known face wins" meant Rex could deliver himself
    to whoever happened to be on camera, not to whoever called him (owner spec
    2026-08-11). An anonymous requester keeps the old any-known-face behavior.

    ``behind=True`` ("I'm behind you, come here") seeds the search with an
    immediate about-face instead of sweeping the wrong hemisphere first (owner
    spec 2026-08-11). ``side_deg`` ("I'm to your left, come here") is the sideways
    version: a signed opening swing toward the stated side (+ = left/CCW, the
    turn() convention); the follow-up sweep also starts on that side."""
    if not _flag("AUTONOMOUS_MOTION_ENABLED", True) or not motion_controller.available():
        return False
    # "Come here" asks for movement, so it lifts an earlier "don't move" outright
    # rather than merely bypassing it — otherwise realign would still be silently
    # held after he had plainly been invited to drive across the room.
    # A room rule is the one thing come-here does NOT override: the owner set it for
    # this room deliberately, and "come here" is almost always said from inside it.
    if no_drive_room() is not None:
        _log.info("[motion_agency] come-here refused — room is flagged no-drive")
        return False
    release_user_hold("come-here request")
    note_traction_recovered("come-here request")
    # An explicit "come here" outranks the autonomous explorer — stop it and take
    # the base (field 2026-07-23: come requests died with "room exploration owns
    # the base" and Rex kept wandering instead of coming).
    try:
        from intelligence import exploration
        if exploration.active():
            exploration.stop("come-here request takes the base")
    except Exception:
        pass
    now = time.monotonic()
    # Radar frames from BEFORE the request are usable only if the base was
    # already still (they are in the current frame); a base mid-motion means
    # wait for a settled sample instead.
    try:
        base_still = motion.state() == "idle"
    except Exception:
        base_still = False
    settle = _num("MOTION_COME_RADAR_SETTLE_SECS", 1.5)
    sample = _num("MOTION_COME_RADAR_SAMPLE_SECS", 1.0)
    _requested_come.update(
        active=True,
        started_at=now,
        requester_id=person_id,
        search_turns=0,
        last_turn_at=0.0,
        pending_turn_seq=None,
        turn_done_at=0.0,
        scan_sign=1.0,
        last_seen_at=0.0,
        seen_sign=0.0,
        seen_deg=0.0,
        front_near_hits=0,
        approach_at=0.0,
        approaches=0,
        align_turns=0,
        skip_log_at=0.0,
        radar_since=(now - sample) if base_still else (now + settle),
        radar_turns=0,
        radar_pending_world=None,
        radar_pending_since=0.0,
        radar_visited=[],
        heading_mode="imu" if _base_yaw_deg() is not None else "cmd",
        cmd_heading=0.0,
    )
    _reset("neck_hits", "far_hits")
    if person_id is not None:
        _log.info("[motion_agency] requested come: searching for requester "
                  "person %s (radar-first, heading via %s)", person_id,
                  _requested_come["heading_mode"])
    else:
        _log.info("[motion_agency] requested come: searching for a visible person "
                  "(radar-first, heading via %s)", _requested_come["heading_mode"])
    if behind:
        seq = _issue_come_turn(180.0, now, rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
        if seq is not None:
            _log.info("[motion_agency] requested come: speaker says they're "
                      "behind — leading with an about-face")
    elif side_deg:
        seq = _issue_come_turn(float(side_deg), now,
                               rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
        if seq is not None:
            # If the swing doesn't find them, keep sweeping on THEIR side rather
            # than snapping back to the default left-first pattern.
            _requested_come["scan_sign"] = 1.0 if float(side_deg) > 0 else -1.0
            _log.info("[motion_agency] requested come: speaker says they're to "
                      "the %s — leading with a %.0f° swing",
                      "left" if float(side_deg) > 0 else "right", abs(float(side_deg)))
    return True


def note_behind_turn(seq: "int | None") -> None:
    """A standalone "I'm behind you" about-face was issued through the normal
    motion.turn lane while a come-here search is RUNNING: adopt that turn as the
    search's own leg, so the dwell + neck sweep run at the new heading instead
    of the search blindly resuming its old sweep pattern mid-rotation."""
    _adopt_voice_bearing_turn(seq, "behind")


def note_side_turn(seq: "int | None", side: str) -> None:
    """The sideways sibling of note_behind_turn: a standalone "I'm to your left/
    right" swing issued through the motion.turn lane mid-come-search becomes the
    search's own leg, and the sweep continues on the speaker's side."""
    _adopt_voice_bearing_turn(seq, "to the %s" % ("right" if side == "right" else "left"))
    if requested_come_active() and seq is not None:
        _requested_come["scan_sign"] = -1.0 if side == "right" else 1.0


def _adopt_voice_bearing_turn(seq: "int | None", where: str) -> None:
    if seq is None or not requested_come_active():
        return
    _stop_come_dwell_gaze()
    _requested_come.update(
        pending_turn_seq=int(seq),
        last_turn_at=time.monotonic(),
        search_turns=0,          # fresh sweep budget at the new heading
        radar_turns=0,
        align_turns=0,
        # Their voice IS a localization: keep the give-up clock fresh, but drop
        # any stored visual bearing — it predates this turn.
        last_seen_at=time.monotonic(),
        seen_deg=0.0,
        seen_sign=0.0,
        # The human just said where they are; radar bodies rejected so far are
        # moot, and any body we were about to check is superseded. In cmd
        # heading mode the size of this externally-issued turn is unknown, so
        # the visited list could not be kept in frame anyway.
        radar_pending_world=None,
        radar_pending_since=0.0,
        radar_visited=[],
    )
    _log.info("[motion_agency] requested come: speaker says they're %s — "
              "adopting the turn as a search leg", where)


def cancel_requested_come(reason: str = "cancelled") -> None:
    if _requested_come["active"]:
        _log.info("[motion_agency] requested come: %s", reason)
        try:
            from intelligence import consciousness
            consciousness.resume_face_tracking()
        except Exception:
            pass
    _stop_come_dwell_gaze()
    _stop_come_drive_gaze()
    _requested_come.update(active=False, started_at=0.0, requester_id=None,
                           search_turns=0, last_turn_at=0.0,
                           pending_turn_seq=None, turn_done_at=0.0,
                           scan_sign=1.0, last_seen_at=0.0, seen_sign=0.0,
                           seen_deg=0.0, front_near_hits=0, approach_at=0.0,
                           approaches=0, align_turns=0, skip_log_at=0.0,
                           radar_since=0.0, radar_turns=0,
                           radar_pending_world=None, radar_pending_since=0.0,
                           radar_visited=[], heading_mode="cmd", cmd_heading=0.0)


# ── Radar-first search helpers ─────────────────────────────────────────────────
# The LD2450 ring reports bodies as (bearing, range, confidence) in the BASE
# frame, + = left/CCW — the very convention motion_controller.turn() takes, so a
# radar bearing IS the turn command, no neck involved. The ring is a hint
# source, never a detector: it says "a body is at 137°"; the camera dwell says
# whether that body is the requester (docs/radar-bearing-prior-spec.md).

def _wrap180(deg: float) -> float:
    d = (float(deg) + 180.0) % 360.0
    return d - 180.0 if d != 0.0 else 180.0


def _base_yaw_deg() -> Optional[float]:
    """The drive base's gyro heading (imu.yaw, deg, + = left/CCW, relative to
    boot — drifts slowly, fine across a 45 s errand), or None when the base does
    not publish a healthy IMU."""
    try:
        tele = motion.telemetry()
        imu = tele.get("imu") if isinstance(tele, dict) else None
        if isinstance(imu, dict) and imu.get("ok") and imu.get("yaw") is not None:
            return float(imu["yaw"])
    except Exception:
        pass
    return None


def _come_heading_deg() -> Optional[float]:
    """Absolute heading of the base for radar bookkeeping. The errand picks its
    mode once at request time so a flickering IMU can't mix two frames: "imu"
    reads the base's gyro yaw, "cmd" sums the turns this module issued. None
    when the chosen source is currently unavailable — callers then skip the
    visited filter rather than trust a bearing in the wrong frame."""
    if _requested_come.get("heading_mode") == "imu":
        return _base_yaw_deg()
    return float(_requested_come.get("cmd_heading") or 0.0)


def _issue_come_turn(deg: float, now: float, *, rate: Optional[float] = None) -> Optional[int]:
    """Every base turn the errand issues goes through here so the cmd-mode
    heading stays in step (pending seq + issue stamp bookkeeping too)."""
    seq = (motion_controller.turn(deg, rate=rate) if rate is not None
           else motion_controller.turn(deg))
    if seq is not None:
        _requested_come["pending_turn_seq"] = seq
        _requested_come["last_turn_at"] = now
        _requested_come["cmd_heading"] = _wrap180(
            float(_requested_come.get("cmd_heading") or 0.0) + float(deg))
    return seq


def _radar_visited_now(heading: Optional[float]) -> "list[float]":
    """Rejected bodies as bearings in the CURRENT base frame (from their stored
    world bearings), or [] when the heading is unavailable."""
    if heading is None:
        return []
    return [_wrap180(w - heading) for w in (_requested_come.get("radar_visited") or [])]


def _radar_mark_pending_visited(now: float) -> None:
    """The dwell after a radar-directed turn found no requester face: that body
    is not them (or shows no face) — remember the spot so the next radar read
    goes elsewhere. NOT applied if the requester was sighted at any point since
    the turn was issued: the camera saw them there, the spot is not empty."""
    world = _requested_come.get("radar_pending_world")
    if world is None:
        return
    _requested_come["radar_pending_world"] = None
    since = float(_requested_come.get("radar_pending_since") or 0.0)
    if float(_requested_come.get("last_seen_at") or 0.0) >= since > 0.0:
        return
    visited = list(_requested_come.get("radar_visited") or [])
    visited.append(float(world))
    _requested_come["radar_visited"] = visited
    _log.info("[motion_agency] requested come: radar body rejected — no requester "
              "face after the dwell (%d spot%s ruled out)", len(visited),
              "" if len(visited) == 1 else "s")


def _radar_bodies(now: float, since: "float | None" = None,
                  window: "float | None" = None) -> "tuple[list[dict], bool]":
    """Cluster the ring's post-settle frames into bodies in the current base
    frame. Returns (bodies, ready): ``ready`` False means the ring is delivering
    but the sample window since radar_since isn't full yet (caller may wait);
    an unavailable ring returns ([], True) so the caller falls straight through.
    Each body: {bearing_deg, range_m, confidence, hits, frames}, best first —
    most persistent, then most confident, then least turning.

    Defaults serve the come-here search (post-turn settle stamp). ``since`` /
    ``window`` let other callers (radar orient) sample their own window."""
    try:
        from hardware import radar
        if not (radar.connected() and radar.radar_ok()):
            return [], True
        if since is None:
            since = float(_requested_come.get("radar_since") or 0.0)
        sample = _num("MOTION_COME_RADAR_SAMPLE_SECS", 1.0)
        if window is None:
            window = sample + _num("MOTION_COME_RADAR_WAIT_SECS", 3.0) + 2.0
        frames = radar.recent_targets(window_secs=window, since=since)
    except Exception as exc:
        _log.debug("radar read failed: %s", exc)
        return [], True
    if not frames or (frames[-1][0] - frames[0][0]) < max(0.0, sample - 0.25):
        return [], False                       # sample not full yet
    min_conf = _num("MOTION_COME_RADAR_MIN_CONFIDENCE", 0.15)
    cluster_deg = _num("MOTION_COME_RADAR_CLUSTER_DEG", 15.0)
    clusters: "list[dict]" = []               # {sx, sy, w, range, conf_max, frame_ids}
    for fidx, (_stamp, targets) in enumerate(frames):
        for t in targets:
            try:
                b = float(t["bearing_deg"]); r = float(t["range_m"]); c = float(t["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if c < min_conf:
                continue
            home = None
            for cl in clusters:
                if abs(_wrap180(b - cl["bearing"])) <= cluster_deg:
                    home = cl
                    break
            if home is None:
                home = {"sx": 0.0, "sy": 0.0, "w": 0.0, "range": 0.0,
                        "conf_max": 0.0, "frame_ids": set(), "bearing": b}
                clusters.append(home)
            w = max(c, 0.05)
            home["sx"] += w * math.cos(math.radians(b))
            home["sy"] += w * math.sin(math.radians(b))
            home["w"] += w
            home["range"] += w * r
            home["conf_max"] = max(home["conf_max"], c)
            home["frame_ids"].add(fidx)
            home["bearing"] = math.degrees(math.atan2(home["sy"], home["sx"]))
    min_frames = max(1, int(_num("MOTION_COME_RADAR_MIN_FRAMES", 3)))
    bodies = []
    for cl in clusters:
        hits = len(cl["frame_ids"])
        if hits < min_frames or cl["w"] <= 0.0:
            continue
        bodies.append({
            "bearing_deg": _wrap180(cl["bearing"]),
            "range_m": cl["range"] / cl["w"],
            "confidence": cl["conf_max"],
            "hits": hits,
            "frames": len(frames),
        })
    bodies.sort(key=lambda b: (-b["hits"], -b["confidence"], abs(b["bearing_deg"])))
    return bodies, True


def _step_come_radar(now: float) -> "bool | None":
    """Radar-directed leg. Returns True when it consumed the tick (turned, or is
    waiting for a settled sample), None when the sweep should take over (radar
    off/unavailable, or no unvisited body)."""
    if not _flag("MOTION_COME_RADAR_ENABLED", True):
        return None
    bodies, ready = _radar_bodies(now)
    if not ready:
        since = float(_requested_come.get("radar_since") or 0.0)
        if (now - since) < _num("MOTION_COME_RADAR_WAIT_SECS", 3.0):
            return True                        # let the ring settle after the turn
        return None                            # ring is quiet — sweep instead
    if not bodies:
        return None
    heading = _come_heading_deg()
    visited = _radar_visited_now(heading)
    visited_deg = _num("MOTION_COME_RADAR_VISITED_DEG", 25.0)
    facing_deg = _num("MOTION_COME_RADAR_FACING_DEG", 12.0)
    # Has the camera seen the requester since the current look began (the last
    # turn we issued, or the hold on a body already ahead)? Then whatever is
    # dead ahead is NOT an empty spot, however the dwell ended.
    look_began = max(float(_requested_come.get("last_turn_at") or 0.0),
                     float(_requested_come.get("radar_pending_since") or 0.0))
    seen_since_look = (look_began > 0.0
                       and float(_requested_come.get("last_seen_at") or 0.0) >= look_began)
    fresh = []
    for body in bodies:
        b = body["bearing_deg"]
        if any(abs(_wrap180(b - v)) <= visited_deg for v in visited):
            continue
        if (abs(b) <= facing_deg
                and float(_requested_come.get("turn_done_at") or 0.0) > 0.0
                and not seen_since_look):
            # Already facing this body and the dwell just looked at it without
            # finding the requester — it's ruled out, not a place to turn to.
            if heading is not None:
                vis = list(_requested_come.get("radar_visited") or [])
                vis.append(_wrap180(heading + b))
                _requested_come["radar_visited"] = vis
            continue
        fresh.append(body)
    if not fresh:
        _log.info("[motion_agency] requested come: radar shows %d bod%s, all already "
                  "checked — falling back to the sweep", len(bodies),
                  "y" if len(bodies) == 1 else "ies")
        return None
    best = fresh[0]
    turn_deg = float(best["bearing_deg"])
    if abs(turn_deg) <= facing_deg:
        # First decision of the errand with a body already dead ahead: no turn
        # needed — dwell on it (the settled camera gets its still frames) and let
        # the next pass either find the face or rule the spot out.
        _requested_come["turn_done_at"] = now
        _requested_come["radar_since"] = now + _num("MOTION_COME_RADAR_SETTLE_SECS", 1.5)
        if heading is not None:
            _requested_come["radar_pending_world"] = _wrap180(heading + turn_deg)
            _requested_come["radar_pending_since"] = now
        _log.info("[motion_agency] requested come: radar body already ahead "
                  "(%+.0f°, %.1fm) — holding for the camera", turn_deg, best["range_m"])
        return True
    seq = _issue_come_turn(turn_deg, now, rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
    if seq is None:
        return None
    _requested_come["radar_turns"] = int(_requested_come.get("radar_turns") or 0) + 1
    _requested_come["scan_sign"] = 1.0 if turn_deg >= 0 else -1.0
    _requested_come["search_turns"] = 0      # a fresh sweep budget from this heading
    if heading is not None:
        _requested_come["radar_pending_world"] = _wrap180(heading + turn_deg)
        _requested_come["radar_pending_since"] = now
    _log.info(
        "[motion_agency] requested come: radar shows %d bod%s (%s) — turning %+.0f° "
        "to the best (%.1fm, c=%.2f, %d/%d frames)%s",
        len(bodies), "y" if len(bodies) == 1 else "ies",
        ", ".join(f"{b['bearing_deg']:+.0f}°/{b['range_m']:.1f}m" for b in bodies),
        turn_deg, best["range_m"], best["confidence"], best["hits"], best["frames"],
        "" if len(fresh) == len(bodies) else f", {len(bodies) - len(fresh)} already checked",
    )
    return True


# ── Come-search dwell gaze ────────────────────────────────────────────────────
# While the base holds still at a scan stop, the NECK sweeps left-right so each
# stop covers roughly ±(neck half-span + camera half-FOV) instead of the camera's
# straight-ahead cone alone (owner spec 2026-08-11: "would it help if the neck did
# a sweep each time he stopped turning"). The every-tick sighting sampler catches
# a face at any neck angle, and the fused alignment bearing (neck + face offset)
# then turns the BODY by exactly the spotted angle while face tracking re-centres
# the head — spotted frame-left at neck-left means "turn left while the neck
# straightens", with no extra machinery. Wider per-stop coverage is also what lets
# the scan legs grow to 90°, so facing directly away no longer costs a ~7-leg tour.

_come_gaze: dict = {"stop": None, "thread": None, "done_key": 0.0,
                    "recenter": True}


def _come_gaze_busy() -> bool:
    """True while the sweep worker (including its recentre glide) is running —
    alignment/scan decisions must wait it out, or they sample the neck mid-glide
    (field 2026-08-11 19:37: aligns pinned at the ±60° clamp because the fused
    bearing was read while the sweep had the neck at full throw)."""
    thread = _come_gaze.get("thread")
    return bool(thread is not None and thread.is_alive())


def _come_dwell_gaze_running_or_ran(done_key: float) -> bool:
    thread = _come_gaze.get("thread")
    if float(_come_gaze.get("done_key") or 0.0) == done_key:
        return True          # already ran (or is running) for this scan stop
    return bool(thread is not None and thread.is_alive())


def _maybe_start_come_dwell_gaze(done_key: float, now: float) -> bool:
    """Start (at most once per scan stop) the dwell neck sweep. Returns True when
    a sweep is running or already ran for this stop, so the caller can extend the
    dwell window to cover it."""
    if _come_dwell_gaze_running_or_ran(done_key):
        return True
    if not _flag("MOTION_COME_NECK_SWEEP_ENABLED", True):
        return False
    try:
        from hardware import servos
        if not servos.connected():
            return False
    except Exception:
        return False
    stop_event = threading.Event()
    first_side = "left" if float(_requested_come["scan_sign"]) >= 0 else "right"
    worker = threading.Thread(
        target=_come_dwell_gaze_loop,
        args=(stop_event, first_side),
        name="come-dwell-gaze",
        daemon=True,
    )
    _come_gaze.update(stop=stop_event, thread=worker, done_key=done_key,
                      recenter=True)
    worker.start()
    return True


def _come_dwell_gaze_loop(stop_event: threading.Event, first_side: str) -> None:
    """One left-and-right neck sweep, first toward the person's last-known side.
    Face tracking still runs — a face spotted mid-sweep locks the head and the
    sampler stops this worker WITHOUT recentring (the head is ON them; gliding
    it back to centre would be throwing the sighting away). Only a sweep that
    ends empty (dwell over, next leg coming) recentres, and never once the
    system is shutting down — the recentre glide racing the power-down droop is
    what stood the servos back up after the rest pose (field 2026-08-11 19:39)."""
    hold = _num("MOTION_COME_NECK_SWEEP_HOLD_SECS", 1.2)
    try:
        from sequences import animations
    except Exception:
        return
    try:
        from intelligence import consciousness
    except Exception:
        consciousness = None

    def _shutting_down() -> bool:
        try:
            return state_module.get_state() in (State.SLEEP, State.SHUTDOWN)
        except Exception:
            return False

    try:
        sides = (first_side, "right" if first_side == "left" else "left")
        for side in sides:
            if stop_event.is_set() or _shutting_down():
                return
            if consciousness is not None:
                try:
                    # Pin the pose against the idle wander / speaker room-scan for
                    # the duration of the hold; face tracking still overrides.
                    consciousness.hold_directed_gaze(side, secs=hold + 2.0)
                except Exception:
                    pass
            # Sweep the side in fractional CHUNKS with stop checks between: one
            # full-throw glide blocks for seconds and cannot be aborted, so on a
            # mid-glide sighting it kept dragging the camera off the person
            # while face tracking fought it — the neck ricocheted thousands of
            # qus in seconds and alignment sampled the chaos (field 2026-08-11
            # 19:57: neck 6872→4065→8904 in 6 s, aligns pinned at the clamp).
            for fraction in (0.45, 0.75, 1.0):
                if stop_event.is_set() or _shutting_down():
                    return
                try:
                    animations.travel_glance_pose(side, "level", fraction=fraction)
                except Exception:
                    return
                if stop_event.wait(0.15):
                    return
            if stop_event.wait(max(0.1, hold)):
                return
    finally:
        if consciousness is not None:
            try:
                consciousness.hold_directed_gaze("center")   # clears the hold
            except Exception:
                pass
        if bool(_come_gaze.get("recenter", True)) and not _shutting_down():
            try:
                animations.travel_glance_pose("center", "level")
            except Exception:
                pass


def _stop_come_dwell_gaze(recenter: bool = False) -> None:
    """Stop the dwell sweep. ``recenter=False`` (sighting / cancel / shutdown)
    leaves the neck where it is; ``recenter=True`` (dwell over, empty view)
    glides it back before the next scan leg."""
    _come_gaze["recenter"] = bool(recenter)
    stop_event = _come_gaze.get("stop")
    if stop_event is not None:
        try:
            stop_event.set()
        except Exception:
            pass


# ── Come-approach drive gaze ──────────────────────────────────────────────────
# While the firmware drives a `come` approach, its steering assist may ARC the
# chassis around lateral obstacles — but the head used to ride the body, so his
# gaze swung off the person he was walking toward with every dodge. This worker
# counter-pans the neck by the base's yaw deviation from the travel heading
# (IMU gyro, + = left/CCW), so the gaze stays pinned on where he is GOING while
# the wheels find their way around the clutter (owner spec 2026-08-19). It also
# dips the camera slightly (down-slight) so floor obstacles directly ahead are
# in frame during the drive. The errand already owns the head (face-tracking
# steering suspended), so there is exactly one neck writer while this runs; it
# self-terminates when the drive's `done` lands and glides back to the
# canonical centre-level pose the alignment measurement expects.

_come_drive_gaze: dict = {"stop": None, "thread": None}


def _neck_qus_for_yaw(bearing_deg: float) -> "int | None":
    """Neck servo target for a head yaw of ``bearing_deg`` off the body's nose
    (+ = Rex's RIGHT — the neck_offset_fraction convention: qus above neutral)."""
    try:
        cfg = config.SERVO_CHANNELS["neck"]
        neutral = float(cfg["neutral"])
        half_span = max(1.0, min(float(cfg["max"]) - neutral,
                                 neutral - float(cfg["min"])))
    except Exception:
        return None
    span_deg = _num("MOTION_COME_NECK_HALF_SPAN_DEG", 45.0)
    if span_deg <= 0:
        return None
    frac = max(-1.0, min(1.0, float(bearing_deg) / span_deg))
    return int(round(neutral + frac * half_span))


def _start_come_drive_gaze(seq: "int | None", heading_deg: float = 0.0) -> None:
    if seq is None or not _flag("MOTION_COME_GAZE_COMP_ENABLED", True):
        return
    _stop_come_drive_gaze()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_come_drive_gaze_loop, args=(stop_event, int(seq), float(heading_deg)),
        name="come-drive-gaze", daemon=True,
    )
    _come_drive_gaze.update(stop=stop_event, thread=worker)
    worker.start()


def _stop_come_drive_gaze() -> None:
    stop_event = _come_drive_gaze.get("stop")
    worker = _come_drive_gaze.get("thread")
    if stop_event is not None:
        stop_event.set()
    if worker is not None and worker.is_alive():
        worker.join(timeout=1.5)
    _come_drive_gaze.update(stop=None, thread=None)


def _come_drive_gaze_loop(stop_event: threading.Event, seq: int,
                          heading_deg: float) -> None:
    try:
        from hardware import servos
        from sequences import animations
    except Exception:
        return
    # Drive pose: camera dips a touch so floor clutter ahead is visible.
    pitch = str(getattr(config, "MOTION_COME_DRIVE_PITCH", "down-slight"))
    try:
        animations.travel_glance_pose("center", pitch)
    except Exception:
        pass
    anchor = _base_yaw_deg()
    max_comp = _num("MOTION_COME_GAZE_COMP_MAX_DEG", 35.0)
    deadband = _num("MOTION_COME_GAZE_COMP_DEADBAND_QUS", 40.0)
    deadline = time.monotonic() + _num("MOTION_COME_GAZE_COMP_MAX_SECS", 30.0)
    try:
        neck_ch = int(config.SERVO_CHANNELS["neck"]["ch"])
    except Exception:
        return
    last_qus = None
    profile_set = False
    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            if motion.done_result(int(seq)) is not None:
                break                    # drive ended — the errand decides what's next
        except Exception:
            break
        yaw = _base_yaw_deg()
        if anchor is None:
            anchor = yaw                 # IMU came up late — anchor on first reading
        if yaw is not None and anchor is not None:
            # Travel heading = anchor + the come command's own alignment turn.
            # + deviation = base swung LEFT of the travel line, so the head pans
            # RIGHT by the same amount and the gaze holds.
            dev = _wrap180(yaw - anchor - heading_deg)
            dev = max(-max_comp, min(max_comp, dev))
            qus = _neck_qus_for_yaw(dev)
            if qus is not None and (last_qus is None
                                    or abs(qus - last_qus) >= deadband):
                try:
                    if not profile_set:
                        servos.set_motion_profile(
                            [neck_ch],
                            speed=int(_num("MOTION_COME_GAZE_COMP_SERVO_SPEED", 60)),
                            acceleration=int(_num("MOTION_COME_GAZE_COMP_SERVO_ACCEL", 10)),
                        )
                        profile_set = True
                    servos.set_servos({neck_ch: int(qus)})
                    servos.set_face_tracking_baseline(neck=int(qus))
                    last_qus = int(qus)
                except Exception:
                    break
        stop_event.wait(0.15)
    # Self-exit (drive done / deadline): glide back to the canonical pose the
    # alignment measurement expects. An explicit stop leaves the head alone —
    # the errand teardown / face tracking owns what happens next.
    if not stop_event.is_set():
        try:
            animations.travel_glance_pose("center", "level")
        except Exception:
            pass


def _step_requested_come(snapshot: dict, now: float, base_idle: bool = True) -> bool:
    """Run one settled-state step. True means this mode consumed the autonomy tick.

    base_idle=False means the firmware is in BLOCKED: scanning/aligning turns are
    still safe (turning away from a block is always allowed), but the forward
    approach must wait for a clear front.
    """
    if not requested_come_active():
        return False
    # The errand OWNS the head for its whole duration: face tracking's neck
    # steering is suspended (rolling re-up, so it resumes shortly after the
    # errand ends however it ends). Five field runs on 2026-08-11 all failed
    # the same way — SOMETHING (face tracking, the sweep, a stale pose) was
    # slewing the neck at the instant alignment sampled it. Detection and
    # identity keep running; only the neck steering stands down.
    try:
        from intelligence import consciousness
        consciousness.suspend_face_tracking(10.0)
    except Exception:
        pass
    timeout = _num("MOTION_COME_SEARCH_TIMEOUT_SECS", 45.0)
    max_turns = max(1, int(_num("MOTION_COME_SEARCH_MAX_TURNS", 8)))
    # The give-up clock restarts on every sighting of the target: measured from the
    # errand start alone, the align phase burned it down and the errand died with
    # "no person found" five seconds after aligning on the requester (field
    # 2026-08-11). max_turns already resets on sightings, bounding each sweep.
    seen_at = float(_requested_come["last_seen_at"])
    anchor = max(float(_requested_come["started_at"]), seen_at)
    turns_used = (int(_requested_come["search_turns"])
                  + int(_requested_come.get("radar_turns") or 0))
    if (now - anchor) >= timeout or turns_used >= max_turns:
        cancel_requested_come("lost them again after sighting — giving up"
                              if seen_at > 0.0
                              else "no person found before search limit")
        return True

    # A turn WE issued is still executing (or its `done` hasn't landed): the camera
    # is swinging, so any "person gone" or alignment judgment now is garbage. Wait.
    pending = _requested_come["pending_turn_seq"]
    if pending is not None:
        verdict = None
        try:
            verdict = motion.done_result(int(pending))
        except Exception:
            verdict = "completed"       # can't ask — don't wedge the errand
        if verdict is None:
            if (now - float(_requested_come["last_turn_at"])) < _num(
                "MOTION_COME_TURN_RESOLVE_TIMEOUT_SECS", 8.0
            ):
                return True
            verdict = "completed"       # `done` lost (comms hiccup) — assume settled
        _requested_come["pending_turn_seq"] = None
        _requested_come["turn_done_at"] = now
        # Radar tracks smear while the base rotates: only ring frames received
        # after this `done` plus a settle may drive the next radar decision.
        _requested_come["radar_since"] = now + _num("MOTION_COME_RADAR_SETTLE_SECS", 1.5)
        if verdict != "completed":
            # The turn was cut short (blocked swing side, no traction): the
            # camera never faced the radar body, so the coming dwell says
            # nothing about that spot — don't let it be rejected. The next
            # radar read sees the body at its new bearing and tries again.
            _requested_come["radar_pending_world"] = None

    # The head LOCK is a head-behavior signal, not a visibility one — it drops
    # whenever anything else steers the head and flickers on a small far-away face.
    # Seeing the target's face at all is enough to go to them (field 2026-07-24).
    requester = _requested_come["requester_id"]
    locked_person = _tracked_person(snapshot, requester)
    person = locked_person or _visible_known_person(snapshot, requester)
    if person is None:
        _requested_come["align_turns"] = 0   # sighting lost — alignment starts over
        # DWELL: the camera settled only turn_done_at ago, and the detect→identify
        # pipeline needs a couple of seconds of STILL camera to find a face across
        # the room. Concluding "nobody this way" any sooner is how two full sweeps
        # crossed the owner without seeing him (field 2026-08-11). This also covers
        # the old re-acquire grace (field 2026-07-21 bookshelf spiral).
        done_at = float(_requested_come["turn_done_at"])
        if done_at > 0.0:
            dwell = _num("MOTION_COME_SCAN_DWELL_SECS", 3.0)
            # The dwell doubles as the neck-sweep window: the head scans left and
            # right from the stationary base, widening what this stop can see.
            if _maybe_start_come_dwell_gaze(done_at, now):
                dwell = max(dwell, _num("MOTION_COME_SCAN_SWEEP_DWELL_SECS", 4.5))
            if (now - done_at) < dwell:
                return True
        _stop_come_dwell_gaze(recenter=True)   # dwell over — recentre for the next leg
        if _come_gaze_busy():
            return True              # let the recentre glide finish first
        # RESIGHT: face tracking held the person moments ago — typically mid-scan,
        # when the sweeping camera swept PAST them and a follow-up micro-turn (e.g.
        # compass correction) lost the lock again (field 2026-07-23: lock on Bret at
        # scan turn 3, sweep continued to -180 and Rex pirouetted instead of coming).
        # Turn a small step back toward that sighting and restart the sweep budget
        # centered there, instead of taking the next ever-bigger sweep leg away.
        fresh = _num("MOTION_COME_SIGHT_FRESH_SECS", 6.0)
        seen_sign = float(_requested_come["seen_sign"])
        if seen_at > 0.0 and (now - seen_at) < fresh and seen_sign != 0.0:
            # Turn by the ACTUAL bearing measured at the sighting (fused
            # neck+face, synchronized) when we have one — a fixed 30° step
            # chronically under-turned toward a face spotted at full neck throw
            # (~60°+ off-body) and the sweep swung right past him again (field
            # 2026-08-11 19:37). Fixed step is the fallback for a signless read.
            seen_deg = float(_requested_come["seen_deg"])
            if seen_deg != 0.0:
                turn_deg = _come_turn_for_bearing(seen_deg)
            else:
                turn_deg = seen_sign * abs(_num("MOTION_COME_RESIGHT_TURN_DEG", 30.0))
            seq = _issue_come_turn(turn_deg, now,
                                   rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
            if seq is not None:
                _requested_come["search_turns"] = 0
                _requested_come["scan_sign"] = seen_sign
                _requested_come["seen_deg"] = 0.0   # spent — don't re-turn on it
                _log.info(
                    "[motion_agency] requested come: recent sighting %.1fs ago — "
                    "turning back %+.0f deg toward it",
                    now - seen_at, turn_deg,
                )
            return True
        # ── RADAR FIRST (owner spec 2026-08-15) ─────────────────────────────
        # No face, no fresh sighting. The dwell that just ended was the camera's
        # look at whatever body the last radar turn pointed at — if the requester
        # wasn't found there, that spot is ruled out. Then ask the ring, from
        # frames received only after the base settled, where the bodies ARE and
        # turn straight to the best unvisited one; the camera dwell after that
        # turn is what decides whether it is them. The blind sweep below is the
        # fallback for a ring that is down, quiet, or has no unvisited body.
        _radar_mark_pending_visited(now)
        radar_step = _step_come_radar(now)
        if radar_step:
            return True
        # Sweep AROUND the last-known side instead of spiraling one direction: net
        # offsets +45, -45, +90, -90, ... (x scan_sign), so the search stays centered
        # on where the person actually was. Relative command i (1-based):
        #   sign = scan_sign * (-1)^(i+1),  magnitude = deg * i.
        # Legs sweep at the slower scan rate so the every-tick sighting sampler can
        # catch a face the camera crosses mid-turn (field 2026-08-11: 75 deg/s legs
        # blew past the owner repeatedly).
        deg = abs(_num("MOTION_COME_SEARCH_TURN_DEG", 90.0))
        i = int(_requested_come["search_turns"]) + 1
        sign = float(_requested_come["scan_sign"]) * (1.0 if i % 2 == 1 else -1.0)
        rel = sign * deg * i
        # Always rotate the SHORT way to the target offset: the raw relative command
        # grows to +/-225, +/-270 in the later sweep steps, which the chassis executed
        # as multi-second pirouettes ("he just spins"). Same net heading, shorter arc.
        rel = ((rel + 180.0) % 360.0) - 180.0
        # 90° legs make some steps degenerate (i=4: rel 360 → 0): a no-op turn
        # would burn a leg staring at the same view — take a plain leg instead.
        if abs(rel) < 1.0:
            rel = sign * deg
        seq = _issue_come_turn(rel, now, rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
        if seq is not None:
            _requested_come["search_turns"] = i
            _log.info(
                "[motion_agency] requested come: scan turn %d/%d (%+.0f deg, sweep)",
                i, max_turns, rel,
            )
        return True

    # The requester's face is on camera: whatever radar body we were checking is
    # NOT a rejected spot (they may well be it), and from here on the camera loop
    # owns the errand — radar is only consulted again if the face is lost.
    _requested_come["radar_pending_world"] = None

    # SETTLE before trusting any alignment measurement: after one of our turns the
    # base and the neck are re-centring the SAME error at once, so sampling the neck
    # (or the face box) mid-slew flips signs and over-corrects — the +15/-37/+37
    # align oscillation that never read "centered" (field 2026-08-11). The sighting
    # sampler keeps last_seen_at fresh through this hold, so the errand can't time
    # out while he is simply letting the picture stabilize. Same for the dwell
    # sweep worker: while it is still gliding the neck, any read is mid-slew.
    if _come_gaze_busy():
        return True
    done_at = float(_requested_come["turn_done_at"])
    if done_at > 0.0 and (now - done_at) < _num("MOTION_COME_ALIGN_SETTLE_SECS", 1.2):
        return True

    # ── CAMERA-LOOP ALIGNMENT (owner spec 2026-08-11, final form: "if I was
    # frame left he would know to turn left while straightening out his neck
    # servo... once he's got me in the center and his head is pointed straight
    # ahead he could reasonably move forward") ───────────────────────────────
    # The neck PARKS dead centre with the head level, so the camera points
    # exactly where the body points — and then the face's position in the
    # frame ALONE is the body bearing. One sensor, one number, and nothing
    # else is allowed to move the neck (face-tracking steering is suspended
    # for the whole errand). Every neck-sampling variant failed in the field
    # because something was always slewing the neck at the moment alignment
    # read it — five runs on 2026-08-11 ended in sign-flipping or clamped
    # align turns from exactly that race.
    if not _neck_parked_centre():
        _park_head_for_alignment(now)
        return True                # settle: measure off a fresh, still frame
    centered_deg = _num("MOTION_COME_CENTERED_DEG", 11.0)
    approach_heading = 0.0
    face_frac = _face_offset_fraction(person)
    bearing = (None if face_frac is None
               else face_frac * _num("MOTION_COME_CAM_HALF_FOV_DEG", 25.0))
    if bearing is not None and abs(bearing) >= centered_deg:
        # GOOD-ENOUGH ESCAPE: repeated align turns that keep missing "centered"
        # must not starve the approach forever — field 2026-08-11: ±12-45 deg
        # oscillation for four minutes, the drive never re-launched, and the
        # errand died to a user "stop". After MOTION_COME_ALIGN_MAX_TRIES
        # attempts, a moderate residual is handed to the firmware as the `come`
        # heading (its closed-loop turn owns the last few degrees) instead of
        # burning another host-side base turn.
        tries = int(_requested_come["align_turns"])
        good_enough = _num("MOTION_COME_ALIGN_GOOD_ENOUGH_DEG", 24.0)
        if (tries >= int(_num("MOTION_COME_ALIGN_MAX_TRIES", 3))
                and abs(bearing) <= good_enough):
            approach_heading = _come_turn_for_bearing(bearing, floor=False)
            _log.info(
                "[motion_agency] requested come: alignment not settling after "
                "%d tries — approaching with residual heading %+.0f deg",
                tries, approach_heading,
            )
        else:
            deg = _come_turn_for_bearing(bearing)
            seq = _issue_come_turn(deg, now)
            if seq is not None:
                _requested_come["align_turns"] = tries + 1
                # Remember which side they were on: if the align turn loses them,
                # the sweep starts back toward that side, not away from it.
                _requested_come["scan_sign"] = 1.0 if deg >= 0 else -1.0
                _requested_come["search_turns"] = 0  # fresh sweep budget after a sighting
                _log.info(
                    "[motion_agency] requested come: acquired %s %s, aligning %+.0f deg",
                    "requester" if requester is not None else "person",
                    person.get("person_db_id") or person.get("id"), deg,
                )
            return True

    if not base_idle:
        # Person found and centered but the front is momentarily blocked — hold
        # this tick; the approach starts once the zone clears (firmware final say).
        return True
    # ── APPROACH ────────────────────────────────────────────────────────────
    # The errand stays ALIVE across the drive. It used to end the instant `come`
    # was sent, so anything that stopped him short ended the whole thing: field
    # 2026-07-24, "if he gets blocked by my dog walking in front of it, he stops
    # and tells me so. But if my dog moves out of the way he should keep trying."
    # The firmware reports how the drive ended, which is the only signal that can
    # tell ARRIVED from STOPPED SHORT — the front ToF cannot, because a dog
    # standing half a metre away looks exactly like having reached someone.
    _, last_result = motion_controller.last_come_result()
    if int(_requested_come["approaches"]) > 0:
        if last_result == "completed":
            # The firmware believes it arrived — but "arrived" means it stopped
            # `stop_at` short of the nearest front return, and a phantom floor
            # return (mis-calibrated matrix ToF) completes the drive seconds in
            # while the requester is still across the room (field 2026-08-11:
            # `come` "completed" after 3 s, 261 front zone_blocks that session).
            # A face-size "public" read alone is NOT enough to resume: the wide-
            # angle lens lies about distance, and the resulting retry burst (three
            # `come`s in 7 s, field 2026-08-11 19:05) bulldozed him into floor
            # clutter right next to the owner. Resume only when the radial front
            # ToF ALSO shows open floor ahead — the same the-ToF-is-the-truth
            # cross-check the spontaneous approach uses. No usable reading fails
            # open (the firmware's own obstacle stop still guards the drive).
            if person.get("distance_zone") != "public":
                cancel_requested_come("arrived")
                return True
            front = _radial_front_m()
            if front is not None and front < _num("MOTION_COME_RESUME_CLEAR_M", 1.8):
                # Confirm the near reading on a SECOND tick before believing it:
                # the radial front throws single-frame speckle, and one bad
                # frame ended a whole errand as "arrived (front reads 0.62m)"
                # nowhere near the requester (field 2026-08-11 20:37).
                hits = int(_requested_come["front_near_hits"]) + 1
                _requested_come["front_near_hits"] = hits
                if hits >= 2:
                    cancel_requested_come("arrived (front reads %.2fm)" % front)
                    return True
                return True             # re-sample next tick
            _requested_come["front_near_hits"] = 0
            _log.info(
                "[motion_agency] requested come: drive completed but requester "
                "still reads far and the front is open (%s) — treating as "
                "stopped short",
                "no front reading" if front is None else "%.2fm" % front,
            )
        elif last_result is None:
            return True                 # still driving; nothing to decide yet
        # Stopped short (blocked/aborted). Reaching here already means the base is
        # idle again (the not-base_idle hold above), i.e. the path cleared. Wait a
        # short beat so a dog dawdling in front can't become a 1 Hz retry storm.
        if (now - float(_requested_come["approach_at"])) < _num(
            "MOTION_COME_RETRY_GAP_SECS", 2.0
        ):
            return True
        if int(_requested_come["approaches"]) >= int(
            _num("MOTION_COME_MAX_APPROACHES", 4)
        ):
            cancel_requested_come("path stayed blocked after repeated tries")
            return True
        _log.info("[motion_agency] requested come: path cleared (last=%s) — resuming",
                  last_result)

    stop_at = _num("MOTION_COME_REQUEST_STOP_AT_M", 1.0)
    seq = motion_controller.come(approach_heading, stop_at=stop_at)
    if seq is not None:
        _start_come_drive_gaze(seq, approach_heading)
        _requested_come["approach_at"] = now
        _requested_come["approaches"] = int(_requested_come["approaches"]) + 1
        _requested_come["align_turns"] = 0
        _log.info(
            "[motion_agency] requested come: approaching person %s "
            "(stop_at=%.2fm, obstacle-gated, try %d)",
            person.get("person_db_id") or person.get("id"), stop_at,
            int(_requested_come["approaches"]),
        )
    return True


def _min_valid_m(*vals) -> Optional[float]:
    """Smallest of the given ToF readings, in METRES, ignoring invalids. The wire
    carries mm as int16: a negative value is the error/no-data sentinel (-1); the
    clear/room-max reads (~3500 matrix, ~4000 radial) are valid large distances.
    None when no reading is usable."""
    best: Optional[float] = None
    for v in vals:
        try:
            mm = float(v)
        except (TypeError, ValueError):
            continue
        if mm < 0.0:                      # -1 = sensor error / no data
            continue
        m = mm / 1000.0
        if best is None or m < best:
            best = m
    return best


_radial_front_fallback_warned = False


def _radial_front_m() -> Optional[float]:
    """Nearest RADIAL front reading in metres — the independent second opinion.

    Reads fl_radial/fr_radial, which the firmware publishes before the 8x8 matrix
    is min-combined into fl/fr. This distinction is the whole point of the
    function: fl/fr are min(radial, matrix), so a caller cross-checking a matrix
    phantom against fl/fr is checking the phantom against itself. Field
    2026-08-20: 773 front zone_blocks, 87% of them with the base parked, and every
    guard built on this function believed it had corroboration it did not have.

    Returns None when there is no usable INDEPENDENT reading — including on
    firmware that predates the split, where believing fl/fr would silently restore
    the old false confidence. Callers already treat None as "no cross-check
    available"; that is the honest answer, and it fails closed.
    """
    global _radial_front_fallback_warned
    tele = motion.telemetry()
    if not (isinstance(tele, dict) and isinstance(tele.get("tof_mm"), dict)):
        return None
    tof = tele["tof_mm"]
    if "fl_radial" in tof or "fr_radial" in tof:
        return _min_valid_m(tof.get("fl_radial"), tof.get("fr_radial"))
    if not _radial_front_fallback_warned:
        _radial_front_fallback_warned = True
        _log.warning(
            "[motion_agency] firmware does not publish fl_radial/fr_radial — no "
            "independent front cross-check this session (flash the motion ESP32). "
            "Guards that need one fail open; clearance checks are unaffected."
        )
    return None


def _front_clearance_m() -> Optional[float]:
    """Nearest front reading in metres for "is there room to drive forward?".

    This one WANTS the conservative min-combined fl/fr — matrix included — because
    the question is whether anything at all is in the way, and the nearest return
    is the right answer even if it turns out to be a phantom. Distinct from
    _radial_front_m, which answers "is that front return corroborated by a second
    sensor?" and must never see the matrix. Conflating the two is what let a guard
    cross-check a phantom against itself (field 2026-08-20).
    """
    tele = motion.telemetry()
    if isinstance(tele, dict) and isinstance(tele.get("tof_mm"), dict):
        return _min_valid_m(tele["tof_mm"].get("fl"), tele["tof_mm"].get("fr"))
    return None


def neck_offset_fraction() -> Optional[float]:
    """Neck offset from neutral as a signed fraction of the half-span.

    + = head panned toward Rex's RIGHT (larger frame x — the face-reveal lateral
    convention; qus above neutral from the tracking logs). None when the neck
    position or channel config is unavailable (e.g. servo-less dev Mac).
    """
    try:
        from world_state import world_state
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
        neck = positions.get("neck")
        cfg = config.SERVO_CHANNELS["neck"]
        neutral = float(cfg["neutral"])
        half_span = max(1.0, min(float(cfg["max"]) - neutral, neutral - float(cfg["min"])))
        if neck is None:
            return None
        return (float(neck) - neutral) / half_span
    except Exception:
        return None


def _visible_known_person(snapshot: dict,
                          requester_id: "int | None" = None) -> Optional[dict]:
    """A known person whose face is visible RIGHT NOW, head lock or not.

    The come-here search used to require face_tracking to be LOCKED. That lock is a
    head-behavior signal, not a "can I see them" signal: it drops whenever something
    else steers the head (a speaker-gaze search, a scan), and it flickers on a small
    face across a room. When it dropped, come-here concluded nobody was there and
    swept the room — field 2026-07-24, owner ~9 ft away: "my face was detected in the
    GUI when I said come here, but he just turned left and right then around and
    never came anywhere." The GUI draws from world_state.people, which still had him.

    With ``requester_id`` set, only THAT person counts: anyone else's face is noted
    but skipped, and the search keeps looking until the requester is found (owner
    spec 2026-08-11: JT on the couch must not satisfy Bret's "come here").
    """
    skipped = None
    found = None
    for person in snapshot.get("people") or []:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        pid = person.get("person_db_id")
        if pid is None or str(pid).strip() == "":
            continue                      # unknown face — never a come-here target
        if requester_id is not None and str(pid) != str(requester_id):
            skipped = pid                 # someone else — keep looking for the requester
            continue
        found = person
        break
    if found is None and skipped is not None and requested_come_active():
        now = time.monotonic()
        if (now - float(_requested_come["skip_log_at"] or 0.0)) > 5.0:
            _requested_come["skip_log_at"] = now
            _log.info(
                "[motion_agency] requested come: seeing person %s but the requester "
                "is %s — continuing the search", skipped, requester_id,
            )
    return found


def _face_offset_fraction(person: Optional[dict]) -> Optional[float]:
    """Signed horizontal offset of a face box from frame centre, as a fraction of
    the half-width (+ = toward Rex's right). The alignment signal that still works
    when the head is NOT pointing at them, so neck offset says nothing useful."""
    if not person:
        return None
    box = person.get("face_box") or person.get("bounding_box") or person.get("bbox")
    if not box or len(box) < 4:
        return None
    try:
        x, _y, w, _h = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    try:
        from world_state import world_state
        frame = (world_state.get("self_state") or {}).get("frame_size") or {}
        width = float(frame.get("width") or 0.0)
    except Exception:
        width = 0.0
    if width <= 0.0:
        width = float(getattr(config, "CAMERA_FRAME_WIDTH", 1920) or 1920)
    if width <= 0.0:
        return None
    centre = x + w / 2.0
    return max(-1.0, min(1.0, (centre - width / 2.0) / (width / 2.0)))


def _come_neck_bearing_deg() -> Optional[float]:
    """The neck's yaw off the body's nose in REAL degrees (+ = Rex's right)."""
    frac = neck_offset_fraction()
    if frac is None:
        return None
    return frac * _num("MOTION_COME_NECK_HALF_SPAN_DEG", 45.0)


def _neck_parked_centre() -> bool:
    """True when the neck sits close enough to neutral that the camera's frame
    centre IS the body's nose. No servo readback (dev Mac) counts as parked —
    there is no neck to correct for."""
    frac = neck_offset_fraction()
    return frac is None or abs(frac) <= _num("MOTION_COME_NECK_PARK_TOLERANCE", 0.06)


def _park_head_for_alignment(now: float) -> None:
    """Glide the neck to centre with the head level (the owner's spec: he keeps
    his head level and straight ahead while lining up). Ends any dwell sweep
    first; the caller's settle window then guarantees a still frame before the
    first camera-bearing measurement."""
    _stop_come_dwell_gaze()
    try:
        from sequences import animations
        animations.travel_glance_pose("center", "level")
    except Exception:
        pass
    _requested_come["turn_done_at"] = now


def _come_bearing_deg(person: Optional[dict], *, head_locked: bool) -> Optional[float]:
    """The person's bearing off the BODY's nose in REAL degrees (+ = Rex's
    right): neck yaw plus the face's angular offset within the camera frame.

    The two components live on DIFFERENT angular scales — the neck's half-span
    is ~45° of yaw (the physical sweep is "about 90 degrees", owner) and half
    the camera frame is only ~25° — and the old fraction math mapped both
    through one 60° constant, OVERSTATING every bearing 1.5-2.4x: he saw the
    owner, computed an inflated angle, and swung ~45° past him, out of view
    (field 2026-08-11 20:15). Camera calibration from that log: a −33° base
    turn moved a face 1290 px across the 1920-wide frame ⇒ ~39 px/deg ⇒
    half-frame ≈ 25°.
    """
    neck_deg = _come_neck_bearing_deg()
    face_frac = _face_offset_fraction(person)
    face_deg = (None if face_frac is None
                else face_frac * _num("MOTION_COME_CAM_HALF_FOV_DEG", 25.0))
    if neck_deg is not None and face_deg is not None:
        return neck_deg + face_deg
    if head_locked and neck_deg is not None:
        return neck_deg
    if face_deg is not None:
        return face_deg
    return neck_deg


def _come_turn_for_bearing(bearing_deg: float, *, floor: bool = True) -> float:
    """The base turn command (deg, + = left/CCW) that faces a bearing. Clamped
    to the max expressive turn; the minimum-turn floor applies only to actual
    base turns (a `come` heading must keep a 3° residual as 3°)."""
    max_deg = _num("MOTION_FACE_TURN_MAX_DEG", 60.0)
    deg = -float(bearing_deg)
    if _flag("MOTION_FACE_TURN_INVERT", False):
        deg = -deg
    deg = max(-max_deg, min(max_deg, deg))
    if floor:
        min_deg = _num("MOTION_FACE_TURN_MIN_DEG", 10.0)
        if abs(deg) < min_deg:
            deg = min_deg if deg >= 0 else -min_deg
    return deg


def _wander_owns_neck() -> bool:
    """True while an idle head wander is driving the neck, or just settled it.

    The grace covers face-tracking hauling the head back from the wander's last
    waypoint: during that sweep the neck reads pegged even though nothing has
    run out of travel.
    """
    if not _flag("MOTION_FACE_IGNORE_WANDER_NECK", True):
        return False
    try:
        from intelligence import consciousness
        age = consciousness.idle_wander_neck_age_secs()
    except Exception:
        return False
    return age < _num("MOTION_FACE_WANDER_SETTLE_SECS", 6.0)


def _tracked_person(snapshot: dict,
                    requester_id: "int | None" = None) -> Optional[dict]:
    """The world_state person entry the head is currently locked onto, or None.

    With ``requester_id`` set, a lock on anyone ELSE returns None — the head
    happening to track the wrong person must not steer a come-here at them."""
    try:
        from world_state import world_state
        tracking = (world_state.get("self_state") or {}).get("face_tracking") or {}
        if not (tracking.get("locked") and tracking.get("visible")):
            return None
        lock_key = str(tracking.get("lock_key") or "")
        kind, _, value = lock_key.partition(":")
        value = value if _ else kind
        for person in snapshot.get("people") or []:
            if not isinstance(person, dict):
                continue
            # Face tracking uses db:<person_db_id> for recognized people and a
            # camera slot key for unknowns. Comparing db:1 only with person["id"]
            # made a recognized, visibly tracked speaker impossible to acquire.
            match = ((kind == "db" and str(person.get("person_db_id")) == value)
                     or (kind != "db" and str(person.get("id")) == value))
            if not match:
                continue
            if (requester_id is not None
                    and str(person.get("person_db_id")) != str(requester_id)):
                return None               # locked onto the wrong person
            return person
        return None
    except Exception:
        return None


def _any_visible_face(snapshot: dict) -> bool:
    """True when ANY face — known or unknown — is on camera right now."""
    for person in snapshot.get("people") or []:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        return True
    return False


def _maybe_radar_orient(snapshot: dict, now: float) -> bool:
    """ORIENT — face a radar body when the camera has nobody (owner spec
    2026-08-19: "use radar to orient towards people if there are no people in
    camera"). Neck-first, wheels last, same as everything else: a body within
    the neck's reach gets a glance the camera can act on (face tracking takes
    over the moment a face appears); only a body beyond the neck turns the
    base. Requires the body to persist across frames (the _radar_bodies
    min-frames rule) AND the no-face condition to hold for consecutive ticks,
    so one dropped detection frame never spins him away from a conversation.
    Returns True when it consumed the tick with an action."""
    if _any_visible_face(snapshot):
        _reset("orient_hits")
        return False
    cooldown = _num("MOTION_RADAR_ORIENT_COOLDOWN_SECS", 30.0)
    if (now - float(_state.get("orient_last_at") or 0.0)) < cooldown:
        return False
    # Radar bearings smear while the base rotates — only decide from a stretch
    # with no recent maneuver of ours.
    quiet = _num("MOTION_RADAR_ORIENT_QUIET_SECS", 3.0)
    busy_at = max(float(_state.get("last_turn_at") or 0.0),
                  float(_state.get("last_approach_at") or 0.0),
                  float(_state.get("last_flinch_at") or 0.0))
    if (now - busy_at) < quiet:
        return False
    window = _num("MOTION_RADAR_ORIENT_WINDOW_SECS", 2.5)
    bodies, ready = _radar_bodies(now, since=now - window, window=window)
    if not ready or not bodies:
        _reset("orient_hits")
        return False
    # A body he already turned toward and found NOBODY at is spent for a while —
    # field 2026-08-19 22:49-50: three +60° chases of a rear return (+172, +109,
    # +175) in three minutes, each spinning him away from where the owner sat,
    # never finding a face. One look per bearing per TTL; ghosts don't get laps.
    yaw = _base_yaw_deg()
    visited_ttl = _num("MOTION_RADAR_ORIENT_VISITED_TTL_SECS", 150.0)
    visited_deg = _num("MOTION_RADAR_ORIENT_VISITED_DEG", 30.0)
    visited = [
        (w, t) for (w, t) in (_state.get("orient_visited") or [])
        if (now - t) < visited_ttl
    ]
    _state["orient_visited"] = visited
    best = None
    for body in bodies:
        if yaw is not None and any(
            abs(_wrap180((yaw + float(body["bearing_deg"])) - w)) <= visited_deg
            for w, _t in visited
        ):
            continue
        best = body
        break
    if best is None:
        _reset("orient_hits")
        return False       # every persistent body has had its look already
    bearing = float(best["bearing_deg"])
    if float(best["confidence"]) < _num("MOTION_RADAR_ORIENT_MIN_CONFIDENCE", 0.30):
        _reset("orient_hits")
        return False
    if abs(bearing) < _num("MOTION_RADAR_ORIENT_MIN_BEARING_DEG", 20.0):
        _reset("orient_hits")
        return False       # already roughly facing them — the camera's problem now
    _state["orient_hits"] = int(_state.get("orient_hits") or 0) + 1
    if _state["orient_hits"] < int(_num("MOTION_RADAR_ORIENT_CONFIRM_TICKS", 3)):
        return False
    _reset("orient_hits")
    if yaw is not None:
        # Mark the WORLD bearing as looked-at the moment we commit to it; if a
        # face shows up there, tracking owns the head and this note is moot.
        visited.append((_wrap180(yaw + bearing), now))
        _state["orient_visited"] = visited

    neck_reach = _num("MOTION_RADAR_ORIENT_NECK_MAX_DEG", 40.0)
    if abs(bearing) <= neck_reach:
        # The neck can cover it — glance, and let face tracking take over the
        # moment a face lands in frame. Never fight another head owner.
        if _wander_owns_neck():
            return False
        try:
            from hardware import servos
            if servos.speech_motion_active() or servos.listening_motion_active():
                return False
        except Exception:
            pass
        side = "left" if bearing > 0 else "right"   # radar + = left/CCW (REP-103)
        frac = min(1.0, abs(bearing) / _num("MOTION_COME_NECK_HALF_SPAN_DEG", 45.0))
        try:
            from sequences import animations
            from intelligence import consciousness
            animations.travel_glance_pose(side, "level", fraction=frac)
            consciousness.hold_directed_gaze(
                side, secs=_num("MOTION_RADAR_ORIENT_NECK_HOLD_SECS", 6.0))
        except Exception:
            return False
        _state["orient_last_at"] = now
        _log.info(
            "[motion_agency] radar orient: body at %+.0f° (%.1fm), nobody on "
            "camera — neck glance %s", bearing, best["range_m"], side,
        )
        return True

    # Beyond the neck: turn the base (a drive — traction rules apply).
    if _traction_lost(now):
        return False
    max_deg = _num("MOTION_FACE_TURN_MAX_DEG", 60.0)
    deg = max(-max_deg, min(max_deg, bearing))      # turn + = left/CCW, same frame
    seq = motion_controller.turn(deg, rate=_num("MOTION_COME_SCAN_RATE_DEG_S", 40.0))
    if seq is not None:
        _state["orient_last_at"] = now
        _state["last_turn_at"] = now
        _log.info(
            "[motion_agency] radar orient: body at %+.0f° (%.1fm), nobody on "
            "camera and beyond the neck — base turn %+.0f°",
            bearing, best["range_m"], deg,
        )
    return True


# ── Object step ─────────────────────────────────────────────────────────────────
# Rex just asked about an object he can see (object_qa.note_asked → the glance
# hook). If the thing is roughly AHEAD of the body, arm one small ToF-gated step
# toward it (owner 2026-08-19: "If he sees something that grabs his attention
# that he asks about, he could move towards the object"). The step is ARMED at
# ask time but EXECUTED by the social-lane tick, which waits out the human's
# answer (mid-sentence gate) instead of driving motor noise into the very reply
# the answer latch is trying to capture.


def face_requester(person_id: "int | None" = None) -> "tuple[float | None, str]":
    """"Turn to face me" — one base turn onto the requester's bearing, then stop.

    Returns ``(deg_issued, reason)``. ``deg_issued`` is None whenever no turn was
    sent; ``reason`` is a machine key the caller maps onto a spoken line:
    ``turned`` / ``already_facing`` / ``no_bearing`` / ``ambiguous`` /
    ``come_active`` / ``busy`` / ``traction`` / ``suppressed`` / ``disabled``.

    ONE SHOT, not an errand. The come-here machinery this borrows from is a
    multi-tick search: turn, dwell, sweep, adopt, approach. "Face me" excludes all
    of that by construction — there is no search, because if he cannot work out
    where you are the honest answer is to SAY so, and the human is right there to
    say something else. So this is a synchronous function on the interaction
    thread, the shape _handle_router_motion_action already uses to drive the base.

    ONE NUMBER. `_come_bearing_deg` is this codebase's single answer to "which way
    is that person off my nose", complete with the 2026-08-11 calibration that cost
    a field session (neck half-span 45 deg, camera half-frame 25 deg; the old math
    ran both through one 60 deg constant, overstated every bearing 1.5-2.4x, and
    swung him past the owner and out of view). Nothing here derives a second one.

    The autonomous realign is deliberately NOT reused. It answers a different
    question — "is tracking failing?" — off a neck-only angle, and in the exact
    case this function serves (they are speaking, the head is on them, the face is
    centred) its `frac` is ~0, which `_turn_degrees_for` floors into a 10 degree
    turn AWAY from the person. Same floor trap as `_come_turn_for_bearing`; see the
    dead-band below.
    """
    if not _flag("MOTION_FACE_ME_ENABLED", True):
        return None, "disabled"
    if not _flag("AUTONOMOUS_MOTION_ENABLED", True) or not motion_controller.available():
        return None, "suppressed"
    now = time.monotonic()
    # A running come-here STRICTLY CONTAINS a face-turn: it is already aligning on
    # the way. Interrupting it to turn would restart its search from a heading it
    # did not choose.
    if requested_come_active():
        return None, "come_active"
    try:
        from intelligence import exploration
        if exploration.active():
            return None, "suppressed"
    except Exception:
        pass
    try:
        if motion.state() != "idle":
            return None, "busy"
    except Exception:
        return None, "busy"
    if _traction_lost(now):
        return None, "traction"
    # An explicit command lifts a standing "don't move" and the traction latch, the
    # same way come-here does: the human has plainly just asked him to move.
    release_user_hold("face request")
    note_traction_recovered("face request")

    bearing, source = _requester_bearing_deg(person_id, now)
    if bearing is None:
        return None, source                    # "no_bearing" or "ambiguous"

    if source == "camera":
        # Dead-band BEFORE the conversion, never after. _come_turn_for_bearing
        # carries a MOTION_FACE_TURN_MIN_DEG floor so a 3 degree residual does not
        # reach the wheels as a twitch — which means it maps a 0 degree bearing to a
        # +10 degree turn and a +3 to a -10 (measured on this checkout). On a
        # commanded face-me that floor would turn "you are already looking at me"
        # into a 10 degree swing off the person.
        if abs(bearing) < _num("MOTION_FACE_ME_CENTERED_DEG", 12.0):
            return None, "already_facing"
        deg = _come_turn_for_bearing(bearing)  # + = LEFT/CCW; the ONE negation
    else:
        # RADAR bearings are ALREADY in the turn frame (+ = left/CCW) and go in raw
        # — the camera's + = Rex's RIGHT is the frame that needs negating. Sending a
        # radar bearing through _come_turn_for_bearing would mirror it: he would
        # turn exactly as far the wrong way. (docs/radar-bearing-prior-spec.md;
        # _step_come_radar and _maybe_radar_orient both pass it through unnegated.)
        if abs(bearing) < _num("MOTION_FACE_ME_CENTERED_DEG", 12.0):
            return None, "already_facing"
        max_deg = _num("MOTION_FACE_ME_TURN_MAX_DEG", 180.0)
        deg = max(-max_deg, min(max_deg, _wrap180(bearing)))

    # Stamp this as a VOICE-commanded maneuver before issuing it. Two things hang
    # off that stamp, and both were missing:
    #   * motion_controller._suppressed only speaks its refusal ("Can't swing that
    #     way — I'd clip something behind me") when a human asked out loud. Without
    #     the stamp a swing-blocked face-me is silent at BOTH ends — the controller
    #     stays quiet because it reads the turn as autonomous, and this function
    #     returns "suppressed", which the caller maps to no line. Turning toward
    #     someone BEHIND him is the single likeliest swing block there is, so that
    #     is the silent no-op the whole _suppressed mechanism exists to prevent.
    #   * the drive whir plays at the autonomous half-volume instead of the
    #     commanded overlay, so the confirmation talks over it.
    motion_controller.note_user_commanded_motion()
    seq = motion_controller.turn(deg, rate=_num("MOTION_FACE_ME_TURN_RATE_DEG_S", 40.0))
    if seq is None:
        # Refused downstream — the swing check, a ToF block, the charger, manual
        # override. motion_controller has now spoken the why for the cases it can
        # explain; say nothing more rather than announcing a turn that never went.
        return None, "suppressed"
    _state["last_turn_at"] = now
    _state["face_me_at"] = now
    _log.info("[motion_agency] face request: %s bearing %+.0f deg -> turn %+.0f deg "
              "(requester=%s)", source, bearing, deg, person_id)
    return deg, "turned"


def _requester_bearing_deg(person_id: "int | None",
                           now: float) -> "tuple[float | None, str]":
    """(bearing, source) for the requester, or (None, refusal-reason).

    CAMERA FIRST and identity-resolved: it is the only source that knows WHICH
    person it is looking at. Radar is a prior, not a detector — it reports that a
    body is at a bearing, never whose (hardware/radar.py, the spec doc). With two
    people in the room a radar-first face-me would turn to whoever is more
    reflective, which is the JT-on-the-couch failure the come rework fixed.
    """
    try:
        from world_state import world_state
        snapshot = world_state.snapshot()
    except Exception as exc:
        _log.debug("face request: world snapshot unavailable: %s", exc)
        snapshot = {}

    # person_id None = an unidentified voice. It must NOT widen to "any known
    # face": that is exactly how "come here" used to deliver Rex to whoever
    # happened to be on camera instead of whoever called him (owner spec
    # 2026-08-11). An unknown requester goes straight to the radar prior.
    locked = _tracked_person(snapshot, person_id) if person_id is not None else None
    person = locked or (_visible_known_person(snapshot, person_id)
                        if person_id is not None else None)
    if person is not None:
        head_locked = locked is not None
        if not _neck_term_trustworthy():
            # Something other than face tracking put the head where it is, so the
            # neck yaw is a pose, not a bearing. The face's offset within the frame
            # still is one — it is measured against the frame, not the body — as
            # long as the neck is near centre so frame centre IS the nose.
            if _neck_parked_centre():
                frac = _face_offset_fraction(person)
                if frac is not None:
                    return frac * _num("MOTION_COME_CAM_HALF_FOV_DEG", 25.0), "camera"
        else:
            bearing = _come_bearing_deg(person, head_locked=head_locked)
            if bearing is not None:
                return bearing, "camera"

    bearing = _radar_requester_bearing(now)
    if bearing is not None:
        return bearing, "radar"
    return None, _state.pop("face_me_radar_reason", None) or "no_bearing"


def _neck_term_trustworthy() -> bool:
    """Whether the neck's yaw currently encodes a FACE bearing rather than a pose.

    The neck is a bearing only when face tracking put it there. Everything else
    that steers it — an idle wander waypoint, a speaker-gaze room scan, a directed
    look, a come dwell sweep, a gamepad — leaves a number that reads exactly like a
    bearing and is not one.

    Listening motion is deliberately absent from this list: it is the NORMAL state
    at the moment a command lands, and its amplitude (SERVO_LISTENING_NECK_QUS) is
    a sway of ~1.4 deg about a frozen face-tracking baseline, well inside the park
    tolerance. Excluding it would refuse nearly every real face-me.
    """
    try:
        if _wander_owns_neck():
            return False
        from hardware import servos
        if servos.speech_motion_active() or servos.manual_override_enabled():
            return False
        if _come_gaze_busy():
            return False
        from intelligence import consciousness
        if consciousness.directed_gaze_hold_active():
            return False
        from world_state import world_state
        tracking = (world_state.get("self_state") or {}).get("face_tracking") or {}
        if tracking.get("searching"):
            return False     # a scan parks the neck at a GUESS, not on a face
    except Exception as exc:
        _log.debug("face request: neck trust check failed: %s", exc)
    return True


def _radar_requester_bearing(now: float) -> "float | None":
    """A single unambiguous radar body's bearing (+ = left/CCW), or None.

    AMBIGUITY IS NOT A BEARING. The ring returns no identity and no track id, and
    several returns at once is the ordinary case in a furnished room — the
    (-hits, -confidence) ranking answers "which return is most solidly a thing",
    which favours furniture over a person who moved. Come-here can afford to take
    the best guess because it then LOOKS, and marks the bearing spent if nobody is
    there (field 2026-08-19: three +60 deg chases of a rear return in three
    minutes, each spinning him away from where the owner sat). A one-shot face-me
    has no look-and-retry, so more than one plausible body means say so.
    """
    _state.pop("face_me_radar_reason", None)
    quiet = _num("MOTION_FACE_ME_QUIET_SECS", 3.0)
    busy_at = max(float(_state.get("last_turn_at") or 0.0),
                  float(_state.get("last_approach_at") or 0.0),
                  float(_state.get("last_flinch_at") or 0.0))
    if (now - busy_at) < quiet:
        return None          # bearings smear across our own rotation
    window = _num("MOTION_FACE_ME_RADAR_WINDOW_SECS", 2.5)
    bodies, ready = _radar_bodies(now, since=now - window, window=window)
    if not ready or not bodies:
        return None
    min_conf = _num("MOTION_FACE_ME_MIN_CONFIDENCE", 0.30)
    # The near floor is above the shell echo the firmware's own range gate leaks
    # (RADAR_RANGE_MIN_M 0.60 was set from a measured 0.47 m self-return); a ghost
    # that is present in every frame scores maximum hits and sorts FIRST.
    near = _num("MOTION_FACE_ME_RANGE_MIN_M", 0.9)
    far = _num("MOTION_FACE_ME_RANGE_MAX_M", 5.0)
    plausible = [
        b for b in bodies
        if float(b.get("confidence") or 0.0) >= min_conf
        and near <= float(b.get("range_m") or 0.0) <= far
    ]
    if not plausible:
        return None
    if len(plausible) > 1:
        _log.info("[motion_agency] face request: %d plausible bodies (%s) — "
                  "radar cannot say which is the requester",
                  len(plausible),
                  ", ".join(f"{b['bearing_deg']:+.0f}@{b['range_m']:.1f}m"
                            for b in plausible[:4]))
        _state["face_me_radar_reason"] = "ambiguous"
        return None
    return _wrap180(float(plausible[0]["bearing_deg"]))


def request_object_step(camera_yaw_deg: float, label: str = "",
                        source: str = "") -> bool:
    """Arm a step toward an asked-about object. ``camera_yaw_deg`` is the
    object's yaw within the camera frame (+ = right of frame); the body bearing
    folds in the current neck offset. Returns True when armed."""
    if not _flag("MOTION_OBJECT_STEP_ENABLED", True):
        return False
    now = time.monotonic()
    if (now - float(_state.get("object_step_at") or 0.0)) < _num(
        "MOTION_OBJECT_STEP_COOLDOWN_SECS", 90.0
    ):
        return False
    neck_deg = _come_neck_bearing_deg() or 0.0        # + = Rex's right
    bearing = neck_deg + float(camera_yaw_deg)        # camera + = right of frame
    if abs(bearing) > _num("MOTION_OBJECT_STEP_MAX_BEARING_DEG", 15.0):
        return False                                   # not ahead — glance only
    _state["object_step"] = {"label": str(label or ""), "bearing": float(bearing),
                             "at": now, "source": str(source or "")}
    return True


def _step_object_step(profile, now: float) -> bool:
    """Execute an armed object step at the first clear moment. True = acted."""
    pending = _state.get("object_step")
    if pending is None:
        return False
    if (now - float(pending.get("at") or 0.0)) > _num(
        "MOTION_OBJECT_STEP_TTL_SECS", 15.0
    ):
        _state["object_step"] = None
        return False
    # The body moved since the ask — the stored bearing no longer points at it.
    moved_at = max(float(_state.get("last_turn_at") or 0.0),
                   float(_state.get("last_approach_at") or 0.0),
                   float(_state.get("last_flinch_at") or 0.0))
    if moved_at > float(pending["at"]):
        _state["object_step"] = None
        return False
    if getattr(profile, "interaction_busy", False):
        return False           # their answer is in flight — hold the arm, wait
    if _traction_lost(now):
        _state["object_step"] = None
        return False
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            _state["object_step"] = None
            return False
    except Exception:
        pass
    front = _front_clearance_m()
    if front is None or front < _num("MOTION_OBJECT_STEP_MIN_FRONT_M", 1.0):
        _state["object_step"] = None
        return False
    step_m = min(_num("MOTION_OBJECT_STEP_M", 0.25),
                 front - _num("MOTION_OBJECT_STEP_KEEP_CLEAR_M", 0.7))
    _state["object_step"] = None
    if step_m < 0.08:
        return False
    speed = random.uniform(_num("MOTION_OBJECT_STEP_SPEED_MIN_MS", 0.08),
                           _num("MOTION_OBJECT_STEP_SPEED_MAX_MS", 0.14))
    seq = motion_controller.move(step_m, speed=speed)
    if seq is not None:
        _state["object_step_at"] = now
        _state["last_approach_at"] = now   # quiet windows respect it
        _log.info(
            "[motion_agency] object step: leaning in %.2fm toward %r "
            "(front %.2fm, %.2f m/s, ToF-gated)",
            step_m, pending.get("label") or "?", front, speed,
        )
        return True
    return False


# ── Startup approach ────────────────────────────────────────────────────────────
def _maybe_startup_approach(person: dict, facing_them: bool,
                            front: "Optional[float]", now: float) -> bool:
    """The welcome roll-up (owner 2026-08-19: "when he started up I expected he
    would move towards me but he sat there motionless... I want it to happen
    right after he starts up if the ToF allow for it").

    Once per session, within a bounded window after the first live autonomy
    tick: the first person he's facing with genuinely open floor ahead gets
    approached to a respectful stop distance. Unlike the regular approach it
    does NOT wait for the "public" zone vote, the 120 s cooldown, or the
    proactive-speech gates — a startup greeting is usually in flight, and the
    greeting and the roll-up are one welcome gesture. The front ToF is the
    authority: no reading, or under the floor, means no drive (fails closed —
    "if the ToF allow" is the owner's own condition)."""
    if not _flag("MOTION_STARTUP_APPROACH_ENABLED", True):
        return False
    if _state.get("startup_approach_done"):
        return False
    first = float(_state.get("first_step_at") or 0.0)
    if first <= 0.0:
        return False
    if (now - first) > _num("MOTION_STARTUP_APPROACH_WINDOW_SECS", 180.0):
        _state["startup_approach_done"] = True    # window closed — stop checking
        return False
    if str(person.get("distance_zone") or "") not in ("social", "public"):
        _state["startup_hits"] = 0
        return False                              # already at conversation range
    if not facing_them:
        _state["startup_hits"] = 0
        return False
    if front is None or front < _num("MOTION_STARTUP_APPROACH_MIN_FRONT_M", 1.8):
        _state["startup_hits"] = 0
        return False                              # the ToF does NOT allow it
    if _traction_lost(now):
        return False
    _state["startup_hits"] = int(_state.get("startup_hits") or 0) + 1
    if _state["startup_hits"] < int(_num("MOTION_STARTUP_APPROACH_CONFIRM_TICKS", 2)):
        return False
    stop_at = _num("MOTION_STARTUP_APPROACH_STOP_AT_M", 1.2)
    speed = None
    if _flag("MOTION_APPROACH_SPEED_JITTER", True):
        speed = _num("MOTION_MAX_LINEAR_MS", 0.40) * random.uniform(
            _num("MOTION_APPROACH_SPEED_JITTER_LOW", 0.55), 1.0)
    seq = motion_controller.come(0.0, stop_at=stop_at, speed=speed)
    if seq is not None:
        _state["startup_approach_done"] = True
        _state["last_approach_at"] = now
        _log.info(
            "[motion_agency] startup approach: first sight of person %s with %.2fm "
            "open ahead -> rolling up (stop_at=%.2fm, ToF-guarded)",
            person.get("person_db_id") or person.get("id"), front, stop_at,
        )
        return True
    return False


# ── Idle base wander ("weight shift") ──────────────────────────────────────────
# The drive-base sibling of the idle arm/head wander (owner spec 2026-08-19:
# "much like the idle hands... more random movement back and forth with slight
# left or right movements"). TURNS are paired — a slight sway then its inverse —
# so heading, which every stored bearing relies on, never drifts. Fore/aft
# SHUFFLES are one-way drifts with a long settle in the new spot (owner field
# pass, same day: the roll-out-roll-back pair "looks like it was for no
# reason"); translation touches no bearing, and every leg is re-gated on
# clearance, so the slow unbiased walk stays bounded. Randomized timing,
# amplitude, and drowsy speed inside a deterministic safety envelope: the
# clearance gates pick what is possible, the dice pick when and how big. All
# motion goes through the ToF-gated closed-loop verbs, so the firmware reflex
# stop stays authoritative; a tight room scales the behavior down and a genuinely
# cramped one (or a no-drive room, positionally) shuts it off entirely.


def _wander_clearances() -> dict:
    """Nearest obstacle per axis in metres from the radial ring (front pair is
    matrix-fused in firmware). Missing/None values mean 'unknown' — the caller
    fails CLOSED on them; a cosmetic behavior never earns benefit of the doubt."""
    tele = motion.telemetry()
    tof = tele.get("tof_mm") if isinstance(tele, dict) else None
    if not isinstance(tof, dict):
        return {}
    return {
        "front": _min_valid_m(tof.get("fl"), tof.get("fr")),
        "rear": _min_valid_m(tof.get("rl"), tof.get("rr")),
        "left": _min_valid_m(tof.get("lf"), tof.get("lb")),
        "right": _min_valid_m(tof.get("rf"), tof.get("rb")),
    }


def _wander_roominess(clear: dict) -> float:
    """0..1 'how roomy is this spot' — the tightest KNOWN axis over the comfort
    distance. 0.0 when nothing is known (fail closed)."""
    known = [v for v in clear.values() if v is not None]
    if not known:
        return 0.0
    comfort = max(0.1, _num("MOTION_IDLE_WANDER_COMFORT_M", 1.2))
    return max(0.0, min(1.0, min(known) / comfort))


def _clear_idle_wander(reason: str = "") -> None:
    if _state.get("wander_pending") is not None:
        _state["wander_pending"] = None
        if reason:
            _log.debug("[motion_agency] idle wander pair dropped: %s", reason)


def _step_idle_wander_pending(now: float) -> bool:
    """Advance an in-flight wander sequence (a sway pair or a meander chain).
    True = this lane owns the tick."""
    pending = _state.get("wander_pending")
    if pending is None:
        return False
    if (now - float(pending.get("at") or 0.0)) > _num(
        "MOTION_IDLE_WANDER_PENDING_TTL_SECS", 12.0
    ):
        _clear_idle_wander("step timed out")
        return False
    steps = pending.get("steps") or []
    idx = int(pending.get("idx") or 0)
    if idx >= len(steps):
        _clear_idle_wander("no steps left")
        return False
    step = steps[idx]
    try:
        verdict = motion.done_result(int(pending["seq"]))
    except Exception:
        _clear_idle_wander("no verdict available")
        return False
    if verdict is None:
        return True                        # maneuver still executing
    if verdict != "completed":
        # The leg died (blocked/aborted). Never chase it with the rest of the
        # chain — the residual offset is at most one wander amplitude. An
        # aborted TURN is the same scrubbed-tyres signal the realign traction
        # detector reads.
        if step.get("op") == "turn" and verdict == "aborted":
            _state["traction_fails"] = int(_state.get("traction_fails") or 0) + 1
            if _state["traction_fails"] >= int(_num("MOTION_TRACTION_FAIL_STREAK", 2)):
                secs = _num("MOTION_TRACTION_STANDDOWN_SECS", 300.0)
                _state["no_traction_until"] = now + secs
                _log.warning(
                    "[motion_agency] no traction — %d turns aborted without physical "
                    "yaw progress. Autonomous driving stood down %.0fs; voice "
                    "commands still work.", _state["traction_fails"], secs,
                )
                _emit_traction_notice()
        _clear_idle_wander("leg ended %s" % verdict)
        return True
    if step.get("op") == "turn":
        _state["traction_fails"] = 0       # the wheels bit
    if idx + 1 >= len(steps):
        _clear_idle_wander()               # sequence complete
        return True
    if now < float(pending.get("dwell_until") or 0.0):
        return True                        # settle at this pose a beat
    nxt = steps[idx + 1]
    if nxt.get("op") == "move":
        # Re-gate the NEXT leg on live clearance — the chain was planned at
        # start, the room may have changed (someone stepped in, he drifted
        # toward the couch). The firmware reflex still guards the drive itself.
        clear = _wander_clearances()
        need = (_num("MOTION_STOP_ZONE_M", 0.15) + abs(float(nxt["amount"]))
                + _num("MOTION_IDLE_WANDER_MOVE_MARGIN_M", 0.30))
        axis = "front" if float(nxt["amount"]) >= 0 else "rear"
        if clear.get(axis) is None or clear[axis] < need:
            _clear_idle_wander("chain leg lost its clearance")
            return True
        seq = motion_controller.move(float(nxt["amount"]), speed=float(nxt["pace"]))
    else:
        seq = motion_controller.turn(float(nxt["amount"]), rate=float(nxt["pace"]))
    if seq is None:
        _clear_idle_wander("next leg refused")
        return False
    pending.update(
        idx=idx + 1, seq=int(seq), at=now,
        dwell_until=now + random.uniform(
            _num("MOTION_IDLE_WANDER_DWELL_MIN_SECS", 0.4),
            _num("MOTION_IDLE_WANDER_DWELL_MAX_SECS", 1.4)),
    )
    return True


def _maybe_idle_wander(profile, now: float) -> bool:
    """Maybe START a wander pair. Caller has already passed the sleep, master
    flag, exploration, availability, idle-state, mid-sentence, user-hold, and
    no-drive-room gates by position."""
    if _state.get("wander_pending") is not None:
        return False
    if now < float(_state.get("wander_next_at") or 0.0):
        return False
    quiet = _num("MOTION_IDLE_WANDER_QUIET_SECS", 6.0)
    busy_at = max(float(_state.get("last_turn_at") or 0.0),
                  float(_state.get("last_approach_at") or 0.0),
                  float(_state.get("last_flinch_at") or 0.0))
    if (now - busy_at) < quiet:
        return False
    if getattr(profile, "interaction_busy", False):
        return False
    if _traction_lost(now):
        return False
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            return False
    except Exception:
        pass
    try:
        from hardware import servos
        if servos.speech_motion_active() or servos.listening_motion_active():
            return False                   # motor noise into his own mic moments
    except Exception:
        pass
    clear = _wander_clearances()
    roominess = _wander_roominess(clear)
    if roominess < _num("MOTION_IDLE_WANDER_MIN_ROOMINESS", 0.35):
        return False                       # tight spot (or blind) — hold still
    if random.random() >= _num("MOTION_IDLE_WANDER_CHANCE", 0.25) * roominess:
        return False

    # What is physically on the table right now?
    stop_zone = _num("MOTION_STOP_ZONE_M", 0.15)
    margin = _num("MOTION_IDLE_WANDER_MOVE_MARGIN_M", 0.30)
    max_move = _num("MOTION_IDLE_WANDER_MOVE_MAX_M", 0.15)
    side_clear = _num("MOTION_IDLE_WANDER_TURN_SIDE_CLEAR_M", 0.35)
    chain_move_max = _num("MOTION_IDLE_WANDER_CHAIN_MOVE_MAX_M", 0.30)
    turn_ok = (clear.get("left") is not None and clear.get("right") is not None
               and min(clear["left"], clear["right"]) >= side_clear)
    options = []
    if turn_ok:
        options.append("turn")
    if (clear.get("front") is not None
            and clear["front"] >= stop_zone + max_move + margin):
        options.append("shuffle_fwd")
    if (clear.get("rear") is not None
            and clear["rear"] >= stop_zone + max_move + margin):
        options.append("shuffle_back")
    if (turn_ok and clear.get("front") is not None
            and clear["front"] >= stop_zone + chain_move_max + margin):
        # A meander needs room to swing AND room ahead for its move legs.
        options.append("meander")
    if not options:
        return False
    kind = random.choice(options)
    amp_scale = 0.5 + 0.5 * roominess      # tighter room = smaller shift
    speed_lo = _num("MOTION_IDLE_WANDER_SPEED_MIN_MS", 0.04)
    speed_hi = _num("MOTION_IDLE_WANDER_SPEED_MAX_MS", 0.09)
    rate_lo = _num("MOTION_IDLE_WANDER_TURN_RATE_MIN_DEG_S", 10.0)
    rate_hi = _num("MOTION_IDLE_WANDER_TURN_RATE_MAX_DEG_S", 22.0)

    if kind == "turn":
        # Sway pair (out + inverse): heading is what every bearing in the other
        # lanes relies on, and a small sway-and-return reads natural.
        deg = random.uniform(_num("MOTION_IDLE_WANDER_TURN_MIN_DEG", 4.0),
                             _num("MOTION_IDLE_WANDER_TURN_MAX_DEG", 10.0)) * amp_scale
        deg *= random.choice((-1.0, 1.0))
        rate = random.uniform(rate_lo, rate_hi)
        steps = [{"op": "turn", "amount": deg, "pace": rate},
                 {"op": "turn", "amount": -deg, "pace": rate}]
        cooldown = (_num("MOTION_IDLE_WANDER_COOLDOWN_MIN_SECS", 25.0),
                    _num("MOTION_IDLE_WANDER_COOLDOWN_MAX_SECS", 70.0))
        label = "sway %+.1f deg @ %.0f deg/s" % (deg, rate)
    elif kind == "meander":
        # A sustained little walk (owner 2026-08-19: "it would be cool if the
        # idle motion chained — turns left 5 degrees, moves forward 1 foot,
        # turns right 7 degrees"). Turn signs ALTERNATE so net heading stays
        # within one leg's swing; every move leg is re-gated on live clearance
        # right before it fires (the chain was planned in a room that may have
        # changed by leg three).
        steps = []
        sign = random.choice((-1.0, 1.0))
        legs = random.randint(int(_num("MOTION_IDLE_WANDER_CHAIN_LEGS_MIN", 3)),
                              int(_num("MOTION_IDLE_WANDER_CHAIN_LEGS_MAX", 6)))
        for i in range(legs):
            if i % 2 == 0:
                deg = random.uniform(
                    _num("MOTION_IDLE_WANDER_TURN_MIN_DEG", 4.0),
                    _num("MOTION_IDLE_WANDER_CHAIN_TURN_MAX_DEG", 12.0)) * amp_scale * sign
                sign = -sign
                steps.append({"op": "turn", "amount": deg,
                              "pace": random.uniform(rate_lo, rate_hi)})
            else:
                dist = random.uniform(
                    _num("MOTION_IDLE_WANDER_CHAIN_MOVE_MIN_M", 0.10),
                    chain_move_max) * amp_scale
                steps.append({"op": "move", "amount": dist,
                              "pace": random.uniform(speed_lo, speed_hi)})
        cooldown = (_num("MOTION_IDLE_WANDER_SHUFFLE_COOLDOWN_MIN_SECS", 45.0),
                    _num("MOTION_IDLE_WANDER_SHUFFLE_COOLDOWN_MAX_SECS", 120.0))
        label = "meander, %d legs" % legs
    else:
        # Shuffles are ONE-WAY drifts (owner 2026-08-19: "rolling forward then
        # straight back looks like it was for no reason"). He settles in the new
        # spot, and the longer shuffle cooldown makes the stay read deliberate.
        # Net drift is a slow unbiased walk, re-gated on clearance every leg —
        # translation doesn't touch heading, so no stored bearing goes stale.
        dist = random.uniform(_num("MOTION_IDLE_WANDER_MOVE_MIN_M", 0.05),
                              max_move) * amp_scale
        if kind == "shuffle_back":
            dist = -dist
        steps = [{"op": "move", "amount": dist,
                  "pace": random.uniform(speed_lo, speed_hi)}]
        cooldown = (_num("MOTION_IDLE_WANDER_SHUFFLE_COOLDOWN_MIN_SECS", 45.0),
                    _num("MOTION_IDLE_WANDER_SHUFFLE_COOLDOWN_MAX_SECS", 120.0))
        label = "%s %+.2f m @ %.2f m/s" % (kind, dist, steps[0]["pace"])

    first = steps[0]
    if first["op"] == "turn":
        seq = motion_controller.turn(float(first["amount"]), rate=float(first["pace"]))
    else:
        seq = motion_controller.move(float(first["amount"]), speed=float(first["pace"]))
    if seq is None:
        return False
    if len(steps) > 1:
        # Multi-leg sequences ride the pending stepper; a lone drift needs no
        # bookkeeping (nothing to chase, nothing to invert).
        _state["wander_pending"] = {
            "steps": steps, "idx": 0, "seq": int(seq), "at": now,
            "dwell_until": now + random.uniform(
                _num("MOTION_IDLE_WANDER_DWELL_MIN_SECS", 0.4),
                _num("MOTION_IDLE_WANDER_DWELL_MAX_SECS", 1.4)),
        }
    _state["wander_next_at"] = now + random.uniform(*cooldown)
    _log.info(
        "[motion_agency] idle wander: %s (roominess %.2f) — ToF-gated",
        label, roominess,
    )
    return True


def _bearing_degrees_for(frac: float) -> float:
    """The signed correction (deg, + = left/CCW per the wire protocol) for an
    offset fraction, WITHOUT the minimum-turn floor — suitable as a `come`
    heading, where a 3-degree residual must stay 3 degrees. Neck toward Rex's
    right (+frac) needs a RIGHT (CW, negative) correction;
    MOTION_FACE_TURN_INVERT flips if field testing disagrees.

    The fraction converts to REAL degrees through the neck's physical half-span
    (~45°), not the 60° turn CLAMP — the old math scaled by the clamp, so every
    realign over-rotated by a third and the comfort realigns ping-ponged
    +52/−59/+60 around the owner (field 2026-08-19 22:47-48). Same scale bug
    class the come-here bearing fusion fixed on 2026-08-11."""
    span_deg = _num("MOTION_COME_NECK_HALF_SPAN_DEG", 45.0)
    max_deg = _num("MOTION_FACE_TURN_MAX_DEG", 60.0)
    deg = -frac * span_deg
    if _flag("MOTION_FACE_TURN_INVERT", False):
        deg = -deg
    return max(-max_deg, min(max_deg, deg))


def _turn_degrees_for(frac: float) -> float:
    """Base turn that reduces an offset fraction: the bearing correction with a
    minimum-turn floor, because the chassis can't execute a tiny nudge."""
    deg = _bearing_degrees_for(frac)
    min_deg = _num("MOTION_FACE_TURN_MIN_DEG", 10.0)
    if abs(deg) < min_deg:
        deg = min_deg if deg >= 0 else -min_deg
    return deg


def _reset_flinch() -> None:
    """Drop the adaptive baselines + confirm counters (cooldown stamps persist)."""
    _flinch_state["baseline"]["fl"] = None
    _flinch_state["baseline"]["fr"] = None
    _flinch_state["clear_run"]["fl"] = 0
    _flinch_state["clear_run"]["fr"] = 0
    _flinch_state["hits"] = 0


def _flinch_side_m(v) -> Optional[float]:
    """One front-sensor reading in METRES, or None if unusable. Beyond the shared
    -1/junk rejection, drop implausibly short reads (< MOTION_FLINCH_MIN_VALID_M) as
    sensor noise so a lone near-zero flyer can't by itself trip the reflex."""
    m = _min_valid_m(v)
    if m is None or m < _num("MOTION_FLINCH_MIN_VALID_M", 0.05):
        return None
    return m


def _nearest(*vals: Optional[float]) -> Optional[float]:
    """Smallest of some already-in-METRES readings, ignoring None. (Unlike
    _min_valid_m this does NOT re-scale — feed it _flinch_side_m outputs.)"""
    xs = [v for v in vals if v is not None]
    return min(xs) if xs else None


def _update_baseline(side: str, d: Optional[float]) -> None:
    """Fold this tick's reading into the side's adaptive open-distance baseline.

    Movement is capped at MOTION_FLINCH_BASELINE_ADAPT_M per tick. The baseline may
    drift DOWN toward a nearer clear surface immediately (keeps him sensitive), but may
    only RISE after the front has read clear for MOTION_FLINCH_CLEAR_CONFIRM_TICKS in a
    row — so a multi-frame ToF dropout-to-max burst can't inflate the reference and
    fake an approach on a static object (the down direction has no such glitch risk).
    Once something is inside the trigger the baseline FREEZES, preserving "where they
    came from" across a long approach or a gated stretch."""
    if d is None:
        return
    if d < _num("MOTION_FLINCH_TRIGGER_M", 0.45):
        _flinch_state["clear_run"][side] = 0        # intrusion regime — freeze
        if _flinch_state["baseline"][side] is None:
            _flinch_state["baseline"][side] = d      # seed on first near read
        return
    _flinch_state["clear_run"][side] += 1
    b = _flinch_state["baseline"][side]
    if b is None:
        _flinch_state["baseline"][side] = d          # seed on first clear read
        return
    cap = _num("MOTION_FLINCH_BASELINE_ADAPT_M", 0.12)
    if d <= b:
        _flinch_state["baseline"][side] = b + max(-cap, d - b)     # nearer surface: track down now
    elif _flinch_state["clear_run"][side] >= max(1, int(_num("MOTION_FLINCH_CLEAR_CONFIRM_TICKS", 3))):
        _flinch_state["baseline"][side] = b + min(cap, d - b)      # sustained re-open: allow a rise
    # else: an unconfirmed clear read (possible dropout) — hold the frozen baseline.


def _side_intrudes(side: str, d: Optional[float]) -> bool:
    """True if this side shows a genuine intrusion: inside the trigger AND closed by
    at least MOTION_FLINCH_APPROACH_DROP_M vs its frozen open-distance baseline."""
    b = _flinch_state["baseline"][side]
    if d is None or b is None or d >= _num("MOTION_FLINCH_TRIGGER_M", 0.45):
        return False
    return (b - d) >= _num("MOTION_FLINCH_APPROACH_DROP_M", 0.20)


def _log_uncorroborated_flinch(front: Optional[float], now: float, kind: str) -> None:
    """Say WHY the reflex held, throttled to the flinch cooldown.

    Silence here would be its own bug: a suppressed flinch looks identical to a
    flinch that never triggered, and the whole reason this veto exists is that the
    last investigation could not tell a phantom from a person from the log."""
    if (now - float(_flinch_state.get("last_veto_log_at") or 0.0)) < _num(
        "MOTION_FLINCH_COOLDOWN_SECS", 8.0
    ):
        return
    _flinch_state["last_veto_log_at"] = now
    _log.info(
        "[motion_agency] flinch (%s) held: front reads %s but the independent "
        "radial front sees %s — matrix-only intrusions are not worth reversing for",
        kind,
        "n/a" if front is None else "%.2fm" % front,
        "open floor" if _radial_front_m() is None else "%.2fm" % _radial_front_m(),
    )


def _flinch_corroborated() -> bool:
    """True when it is safe to believe the front intrusion enough to REVERSE.

    fl/fr are min(radial, matrix) in firmware, so a matrix phantom simply IS the
    front reading — and a phantom that is temporally consistent rather than
    single-frame defeats every anti-noise mechanism the reflex has (adaptive
    baseline, clear-run confirm, consecutive-tick hits, cooldown). Field 2026-08-20:
    five retreats, ~1.5 m of unrequested reverse, off a channel dipping to
    0.07-0.11 m from >0.6 m. The radial pair watches the same personal space from a
    different sensor, so a real shin or dog registers on both.

    Fails OPEN when the firmware does not publish the independent pair: no second
    opinion means no veto, which is exactly today's behavior rather than a new
    blind spot. Also fails open when corroboration is disabled.
    """
    if not _flag("MOTION_FLINCH_REQUIRE_CORROBORATION", True):
        return True
    radial = _radial_front_m()
    if radial is None:
        return True                     # no independent reading — don't veto
    return radial <= _num("MOTION_FLINCH_CORROBORATION_MAX_M", 0.60)


def _flinch_gated(profile, now: float) -> bool:
    """Common fire gates once a trigger is present: the mid-sentence freeze (only when
    the operator opts flinch into it) and the maneuver cooldown."""
    if (getattr(profile, "user_mid_sentence", False)
            and not _flag("MOTION_FLINCH_ALLOW_MID_SENTENCE", True)):
        return True
    return (now - _state["last_flinch_at"]) < _num("MOTION_FLINCH_COOLDOWN_SECS", 6.0)


def _corner_log(msg: str, front: float, rear: Optional[float], now: float) -> None:
    """Throttled 'nowhere to retreat' log (does NOT touch the maneuver cooldown, so a
    hold never delays a real flinch once room appears behind him)."""
    if (now - _flinch_state["last_corner_log_at"]) < _num("MOTION_FLINCH_COOLDOWN_SECS", 6.0):
        return
    _flinch_state["last_corner_log_at"] = now
    _log.info(
        "[motion_agency] flinch: crowded at %.2fm but rear=%s — %s",
        front, ("%.2fm" % rear) if rear is not None else "unknown", msg,
    )


def _flinch_retreat(front: float, rear: Optional[float], now: float, reason: str) -> bool:
    """Back off from a confirmed front intrusion, capped by rear clearance. Holds
    (returns False, no cooldown stamp) when cornered or blind behind — the firmware's
    always-on rear-ToF stop only guards a reverse when the rear sensors report, so a
    blind rear has no backstop and must not take even a token step. Returns True only
    when a move was actually issued."""
    if rear is None:
        _corner_log("rear sensors blind, holding", front, rear, now)
        return False
    room = rear - _num("MOTION_FLINCH_REAR_MARGIN_M", 0.30)
    backup = min(_num("MOTION_FLINCH_BACKUP_M", 0.30), room)
    if backup < _num("MOTION_FLINCH_MIN_BACKUP_M", 0.10):
        _corner_log("cornered, holding", front, rear, now)
        return False

    speed = _num("MOTION_FLINCH_SPEED_MS", 0.20)
    seq = motion_controller.move(-backup, speed)
    if seq is None:
        return False  # suppressed (paused / gamepad owner / disconnected) — not fired
    _log.info(
        "[motion_agency] flinch (%s): front %.2fm -> back off %.2fm (rear=%.2fm, speed=%.2f)",
        reason, front, backup, rear, speed,
    )
    _state["last_flinch_at"] = now
    _reset_flinch()  # fresh baselines after the move
    try:
        from intelligence import decision_ledger
        decision_ledger.record(
            "flinch",
            f"something came at my front sensor fast ({str(reason).replace('_', ' ')}, "
            f"about {front:.1f} m away) and I backed off {backup:.1f} m",
            detail={"reason": reason, "front_m": round(front, 2), "backup_m": round(backup, 2)},
        )
    except Exception:
        pass
    return True


def _maybe_flinch(profile, now: float, state: str) -> bool:
    """Reflexive back-off from a front intrusion. Call once per tick from idle OR the
    firmware BLOCKED state; returns True only when a back-off was issued (caller stops).

    IDLE — watch each front side (fl/fr) with its own adaptive baseline + a
    consecutive-tick confirm counter, so a real approach (from either side, fast or
    slow) fires while static clutter and single-frame noise do not.

    BLOCKED — the firmware vouches only that an obstacle is close, not that a PERSON
    approached. Require the same temporal closure evidence as idle, using the baseline
    sampled before the block. This deliberately fails closed when there is no baseline:
    a stuck-close matrix return or robot body part must never make Rex reverse blindly.

    Never raises."""
    tele = motion.telemetry()
    if not isinstance(tele, dict):
        return False
    tof = tele.get("tof_mm")
    if not isinstance(tof, dict):
        return False
    rear = _min_valid_m(tof.get("rl"), tof.get("rr"))
    if state == "blocked":
        fl = _flinch_side_m(tof.get("fl"))
        fr = _flinch_side_m(tof.get("fr"))
        intruding = _side_intrudes("fl", fl) or _side_intrudes("fr", fr)
        if not intruding:
            _flinch_state["hits"] = 0
            return False
        _flinch_state["hits"] += 1
        if _flinch_state["hits"] < max(1, int(_num("MOTION_FLINCH_CONFIRM_TICKS", 2))):
            return False
        if _flinch_gated(profile, now):
            return False
        if not _flinch_corroborated():
            _log_uncorroborated_flinch(_nearest(fl, fr), now, "blocked-approach")
            return False
        return _flinch_retreat(_nearest(fl, fr), rear, now, "blocked-approach")

    fl = _flinch_side_m(tof.get("fl"))
    fr = _flinch_side_m(tof.get("fr"))

    # ── idle: sample every tick (baselines must stay fresh even while gated) ──────
    _update_baseline("fl", fl)
    _update_baseline("fr", fr)
    intruding = _side_intrudes("fl", fl) or _side_intrudes("fr", fr)
    _flinch_state["hits"] = _flinch_state["hits"] + 1 if intruding else 0

    if _flinch_state["hits"] < max(1, int(_num("MOTION_FLINCH_CONFIRM_TICKS", 2))):
        return False
    if _flinch_gated(profile, now):
        return False
    if not _flinch_corroborated():
        _log_uncorroborated_flinch(_nearest(fl, fr), now, "approach")
        return False
    return _flinch_retreat(_nearest(fl, fr), rear, now, "approach")


def step(snapshot: dict, profile) -> None:
    """One autonomy tick. Call from the consciousness loop; never raises."""
    try:
        _step_inner(snapshot, profile)
    except Exception as exc:
        _log.debug("motion agency step error: %s", exc)


def _step_inner(snapshot: dict, profile) -> None:
    if state_module.get_state() in (State.SLEEP, State.SHUTDOWN):
        _reset("neck_hits", "far_hits")
        _reset_flinch()
        cancel_requested_come("robot asleep")
        return
    if not _flag("AUTONOMOUS_MOTION_ENABLED", True):
        # Drop the flinch baselines like every other non-live path, so a person who
        # walked up while autonomy was OFF isn't read as an "approach" on re-enable.
        _reset("neck_hits", "far_hits")
        _reset_flinch()
        cancel_requested_come("autonomous motion disabled")
        return
    # A room-exploration session OWNS the base while it wanders — realign/approach
    # must not interleave a maneuver between its legs.
    try:
        from intelligence import exploration
        if exploration.active():
            _reset("neck_hits", "far_hits")
            _reset_flinch()
            cancel_requested_come("room exploration owns the base")
            return
    except Exception:
        pass
    if not motion_controller.available():
        _reset("neck_hits", "far_hits")
        _reset_flinch()
        cancel_requested_come("drive base unavailable")
        return

    # The startup-approach window opens on the first tick WITH a live base, not
    # at import — a session that boots before the ESP32 enumerates must not burn
    # its welcome window against a base that isn't there yet.
    if not float(_state.get("first_step_at") or 0.0):
        _state["first_step_at"] = time.monotonic()

    st = motion.state()

    # SIGHTING SAMPLER for an active come request — runs on EVERY tick, including
    # while the base is mid-turn. Scan turns sweep the camera across the person for
    # only a moment; the settled-state step below never runs during that moment, so
    # without this the sighting is thrown away and the sweep spins right past them
    # (field 2026-07-23: face lock on Bret during scan turn 3, sweep went to -180).
    if requested_come_active():
        _req = _requested_come["requester_id"]
        seen_locked = _tracked_person(snapshot, _req)
        seen = seen_locked or _visible_known_person(snapshot, _req)
        if seen is not None:
            _stop_come_dwell_gaze()   # face found — the sweep yields to tracking,
                                      # leaving the neck ON them (no recentre)
            _requested_come["last_seen_at"] = time.monotonic()
            bearing = _come_bearing_deg(seen, head_locked=seen_locked is not None)
            if bearing is not None and abs(bearing) > 3.0:
                # + bearing = person to Rex's right needs a negative (CW) base
                # turn; seen_sign follows the turn convention (+ = left).
                _requested_come["seen_sign"] = -1.0 if bearing >= 0 else 1.0
                # The neck and face are read TOGETHER here, so this bearing is
                # synchronized — trustworthy even mid-sweep at full neck throw.
                _requested_come["seen_deg"] = float(bearing)

    # The base must be settled (idle) to sample intrusions, but a firmware BLOCKED
    # state is itself a strong front-crowding signal the flinch should answer. Any
    # other state (moving / estop / fault / comms-lost) means the ToF is dominated by
    # our own motion or the base is unavailable — drop the baselines so a distance
    # jump across it can't later masquerade as an approach.
    if st not in ("idle", "blocked"):
        _reset_flinch()
        return

    now = time.monotonic()

    # An explicit request owns the social-motion lane. While idle it searches,
    # aligns, or starts the approach. A front BLOCK does NOT end the search — the
    # firmware always allows turning and driving AWAY from a block, and the front
    # zone flaps near furniture (field 2026-07-23: "search blocked by an obstacle"
    # killed a come request that only needed to keep turning). Only the forward
    # approach must wait for an idle base; _step_requested_come gates it on st.
    if requested_come_active():
        if _step_requested_come(snapshot, now, base_idle=(st == "idle")):
            return

    # ── FLINCH: reflexive back-off when someone crowds the front ─────────────────
    # A raw ToF reflex: no tracked/known person required, and it may fire even while
    # the human is mid-sentence (someone stepping into his face while talking) — so
    # it is evaluated BEFORE the mid-sentence freeze that guards the social behaviors.
    # It also samples the front baseline every idle tick (approach detection), which
    # is why it must run before any of the early returns below.
    if _flag("MOTION_FLINCH_ENABLED", True) and _maybe_flinch(profile, now, st):
        return  # one maneuver per tick

    # A BLOCKED base can only flinch (or hold) — it is not settled enough for the
    # social behaviors, and there is no fresh approach baseline while blocked.
    if st != "idle":
        return

    # Below here are the SOCIAL behaviors (realign/approach). Never start their
    # maneuvers while the human is mid-sentence (motor noise into the mic on THEIR
    # turn) — a reflex flinch is deliberately exempt from this, they are not.
    if getattr(profile, "user_mid_sentence", False):
        _reset("neck_hits", "far_hits")
        return

    # The human steered the body by voice (honor their placement instead of rotating
    # it back toward their face — see note_user_motion), or told him to stay put
    # outright (note_user_hold, which latches). Counters reset either way, so stale
    # off-center ticks can't fire the instant the stand-down lifts.
    if _user_motion_standdown(now) or _user_hold_active(now):
        _reset("neck_hits", "far_hits")
        return

    # An active parlor game means the players are seated around him and long
    # think-silences are normal — "nobody on camera" mid-round is a seating fact,
    # not a search cue. The social lanes below read that stillness as a reason to
    # go looking: radar-orient chased a rear return and turned the base away from
    # the Jeopardy players, and idle wander shuffled him between clues (field
    # 2026-08-25 18:55-18:56). The flinch reflex and an explicit come-here (both
    # above) still run; everything social waits for the game to end.
    if _flag("MOTION_HOLD_DURING_GAMES", True):
        try:
            from features import games as games_mod
            if games_mod.is_active():
                _reset("neck_hits", "far_hits")
                _clear_idle_wander("game active")
                return
        except Exception:
            pass

    # A room the owner has told him not to drive in (carpet, or just their house
    # rules). Persisted per place, so it re-arms every time he recognizes the room —
    # unlike the traction detector, which has to grind through a couple of failed
    # turns first. Only the SOCIAL behaviors are gated here; the flinch reflex above
    # and an explicit come-here still run.
    room = no_drive_room()
    if room is not None:
        if (now - float(_state.get("no_drive_log_at") or 0.0)) > 120.0:
            _state["no_drive_log_at"] = now
            _log.info("[motion_agency] holding still — %s is flagged no-drive (%s)",
                      room[0], room[1] or "owner's rule")
        _reset("neck_hits", "far_hits")
        _clear_idle_wander("no-drive room")
        return

    # An in-flight weight-shift pair finishes (or dies) before any other lane may
    # move the base — its inverse landing after a realign turn would corrupt both.
    if _step_idle_wander_pending(now):
        return

    # A step toward an asked-about object executes at the first clear moment
    # (it was armed at ask time; this tick has already waited out mid-sentence).
    if _step_object_step(profile, now):
        return

    person = _tracked_person(snapshot)
    if person is None:
        _reset("neck_hits", "far_hits")
        _state["neck_strain_since"] = 0.0   # no tracked face = no tracking strain
        # Nobody on camera — but the radar ring may know where they are.
        if (_flag("MOTION_RADAR_ORIENT_ENABLED", True)
                and _maybe_radar_orient(snapshot, now)):
            return
        # Nothing social to do — maybe shift his weight (one maneuver per tick).
        if _flag("MOTION_IDLE_WANDER_ENABLED", True):
            _maybe_idle_wander(profile, now)
        return
    _reset("orient_hits")

    frac = neck_offset_fraction()

    # ── TRACTION CHECK ───────────────────────────────────────────────────────
    # The firmware already knows. TURN_IMU_VERIFY closes finite turns on integrated
    # gyro yaw, and a turn that cannot make physical yaw progress is aborted after
    # TURN_VERIFY_TIMEOUT rather than grinding forever (calib.h). On carpet that is
    # exactly what happens: the tyres scrub, yaw never moves, `done result=aborted`.
    # Field 2026-07-25: seven consecutive realign turns, every one aborted at ~8 s,
    # motors grinding the whole time while the neck sat 55-98% off-centre. So take
    # the base at its word instead of second-guessing it from the neck offset —
    # `aborted` IS the no-traction signal. Two in a row (one alone can be comms
    # loss, which shares the abort code) stands autonomous driving down.
    pending = _state.get("realign_pending_seq")
    if pending is not None:
        verdict = None
        try:
            verdict = motion.done_result(int(pending))
        except Exception:
            _state["realign_pending_seq"] = None
        if verdict == "aborted":
            _state["realign_pending_seq"] = None
            _state["traction_fails"] = int(_state.get("traction_fails") or 0) + 1
            if _state["traction_fails"] >= int(_num("MOTION_TRACTION_FAIL_STREAK", 2)):
                secs = _num("MOTION_TRACTION_STANDDOWN_SECS", 300.0)
                _state["no_traction_until"] = now + secs
                _log.warning(
                    "[motion_agency] no traction — %d turns aborted without physical "
                    "yaw progress. Autonomous driving stood down %.0fs; voice "
                    "commands still work.", _state["traction_fails"], secs,
                )
                _emit_traction_notice()
        elif verdict is not None:
            _state["realign_pending_seq"] = None
            if verdict == "completed":
                _state["traction_fails"] = 0   # the wheels bit — floor is fine

    if _traction_lost(now):
        _reset("neck_hits", "far_hits")
        return      # the wheels cannot turn here — do not grind at the carpet

    # ── REALIGN: rotate the base under the head — neck first, wheels last ─────
    # The neck servo is the primary tracker. The wheels only engage once the neck
    # sweep is exhausted (parked near its travel limit) AND the face has drifted to
    # the extreme edge of the frame on that same side — i.e. face-tracking has
    # genuinely run out of neck and still can't hold them.
    if _flag("MOTION_FACE_PERSON_ENABLED", True) and frac is not None:
        # The neck offset is only a TRACKING signal when face-tracking is what put
        # the head there. An idle wander parks it at the travel limit for reasons
        # that have nothing to do with a face, and any face that happens to be on
        # that side then satisfies the edge test below — so the wheels turn on a
        # forged signal. Field 2026-08-18: a wander during an impersonation left
        # the neck at -99%, face-tracking was still hauling it back, and realign
        # spun the base +59 deg mid-performance. Wait for tracking to own the neck
        # again. (Same class as the come-here rework: one sensor, one number, one
        # actuator owner.)
        if _wander_owns_neck():
            _reset("neck_hits", "far_hits")
            _state["neck_strain_since"] = 0.0   # a wander-parked neck is not strain
            return
        threshold = _num("MOTION_FACE_NECK_FRACTION", 0.85)
        edge = _num("MOTION_FACE_EDGE_FRACTION", 0.30)
        face_frac = _face_offset_fraction(person)
        neck_exhausted = abs(frac) >= threshold
        # Same-side check: neck panned right (+) with the face escaping right (+).
        # A face on the OPPOSITE side means the neck can still sweep toward it.
        face_at_edge = (face_frac is not None and abs(face_frac) >= edge
                        and face_frac * frac > 0.0)
        if neck_exhausted and face_at_edge:
            _state["neck_hits"] += 1
        else:
            _state["neck_hits"] = 0
        confirm = int(_num("MOTION_FACE_CONFIRM_TICKS", 2))
        cooldown = _num("MOTION_FACE_TURN_COOLDOWN_SECS", 8.0)
        if (_state["neck_hits"] >= confirm
                and (now - _state["last_turn_at"]) >= cooldown):
            deg = _turn_degrees_for(frac)
            # A realign is housekeeping, not a commanded maneuver — glide it.
            # The default 75°/s made a 60° correction read as a jarring snap
            # mid-conversation (owner 2026-08-19).
            seq = motion_controller.turn(
                deg, rate=_num("MOTION_FACE_TURN_RATE_DEG_S", 40.0))
            if seq is not None:
                _log.info(
                    "[motion_agency] realign: neck %.0f%% off-center -> base turn %+.0f deg "
                    "(person=%s)",
                    frac * 100.0, deg, person.get("person_db_id") or person.get("id"),
                )
                _state["last_turn_at"] = now
                _state["realign_pending_seq"] = seq    # did it actually rotate?
            _reset("neck_hits")
            return  # one maneuver per tick

        # ── COMFORT REALIGN (owner 2026-08-19: "He's turning his head to face
        # me, but he looks strained. He should eventually turn his body to face
        # and straighten out his neck servo while he does it.") ────────────────
        # The hard trigger above only fires when tracking is LOSING the person —
        # exhausted neck AND face escaping the frame. Field 2026-08-19 22:07:
        # the neck sat at 70-100% of its throw for most of a session (mean 69%)
        # with the face held perfectly centered, so the wheels never came
        # around and he chatted cranked sideways. This trigger is the ease-in:
        # a neck held past the comfort fraction for a sustained stretch turns
        # the base by the neck angle, and face tracking straightens the neck as
        # the body comes around — same mechanism, gentler cause. The timer
        # freezes out anything that parks the neck deliberately (directed-gaze
        # holds; wanders are already excluded above).
        comfort = _num("MOTION_FACE_COMFORT_FRACTION", 0.60)
        gaze_held = False
        try:
            from intelligence import consciousness
            gaze_held = bool(consciousness.directed_gaze_hold_active())
        except Exception:
            pass
        if abs(frac) < comfort or gaze_held:
            _state["neck_strain_since"] = 0.0
        elif not float(_state.get("neck_strain_since") or 0.0):
            _state["neck_strain_since"] = now
        elif ((now - float(_state["neck_strain_since"]))
                >= _num("MOTION_FACE_COMFORT_SECS", 12.0)
                and (now - _state["last_turn_at"]) >= cooldown):
            deg = _turn_degrees_for(frac)
            seq = motion_controller.turn(
                deg, rate=_num("MOTION_FACE_TURN_RATE_DEG_S", 40.0))
            if seq is not None:
                _log.info(
                    "[motion_agency] comfort realign: neck held %.0f%% off-center "
                    "for %.0fs -> base turn %+.0f deg, neck straightens as it "
                    "comes around (person=%s)",
                    frac * 100.0, now - float(_state["neck_strain_since"]), deg,
                    person.get("person_db_id") or person.get("id"),
                )
                _state["last_turn_at"] = now
                _state["realign_pending_seq"] = seq    # traction detector watches it
                _state["neck_strain_since"] = 0.0
                return  # one maneuver per tick

    centered = _num("MOTION_APPROACH_CENTERED_FRACTION", 0.18)
    facing_them = frac is None or abs(frac) < centered
    # Face width lies on a wide-angle lens: a face 3-4 ft away can read under the
    # "public" fraction, so face size alone said "far" about someone within arm's
    # reach (field 2026-07-31: drove at the owner from 3-4 ft, got awkwardly close).
    # The front ToF is the truth — unless it shows genuinely open floor ahead, the
    # "they're far" vote doesn't count. Fails open only when there is no usable
    # front reading (the firmware's obstacle stop still guards the drive itself).
    # Clearance, not corroboration: the conservative min-combined pair is exactly
    # what "is the floor ahead open" wants (see _front_clearance_m).
    front = _front_clearance_m()

    # ── STARTUP APPROACH: the welcome roll-up ──────────────────────────────────
    if _maybe_startup_approach(person, facing_them, front, now):
        return  # one maneuver per tick

    # Shared gates for the volunteered drives below. Critical battery: stop
    # VOLUNTEERING (voice-commanded motion still obeys — the BMS is the hard
    # protection; this is Rex pacing himself). suppress_proactive/interaction_busy:
    # a whole-base drive is a big proactive act — same social gates as unsolicited
    # speech. These used to END the tick, which starved the idle wander below for
    # entire conversations (field 2026-08-19: minutes of statue while chatting) —
    # now they only skip the lanes they gate.
    battery_ok = True
    try:
        from intelligence import battery_awareness
        battery_ok = not battery_awareness.battery_critical()
    except Exception:
        pass
    proactive_ok = not (getattr(profile, "suppress_proactive", False)
                        or getattr(profile, "interaction_busy", False))

    # ── APPROACH: close distance to a far person ──────────────────────────────
    if not (_flag("MOTION_APPROACH_ENABLED", True) and battery_ok and proactive_ok):
        _reset("far_hits")
    else:
        far_enough = not (
            front is not None and front < _num("MOTION_APPROACH_MIN_START_M", 1.8)
        )
        if person.get("distance_zone") == "public" and facing_them and far_enough:
            _state["far_hits"] += 1
        else:
            _state["far_hits"] = 0
        confirm = int(_num("MOTION_APPROACH_CONFIRM_TICKS", 4))
        cooldown = _num("MOTION_APPROACH_COOLDOWN_SECS", 120.0)
        if (_state["far_hits"] >= confirm
                and (now - _state["last_approach_at"]) >= cooldown):
            # Spontaneous approaches keep a respectful distance: stop farther out
            # than the explicit come-here default (nobody asked him to come this
            # time). And nobody asked, so he doesn't have to hurry — the pace is
            # randomized (owner 2026-08-19). Explicit come-here keeps full pace.
            stop_at = _num("MOTION_APPROACH_STOP_AT_M", 1.0)
            speed = None
            if _flag("MOTION_APPROACH_SPEED_JITTER", True):
                speed = _num("MOTION_MAX_LINEAR_MS", 0.40) * random.uniform(
                    _num("MOTION_APPROACH_SPEED_JITTER_LOW", 0.55), 1.0)
            seq = motion_controller.come(0.0, stop_at=stop_at, speed=speed)
            if seq is not None:
                _log.info(
                    "[motion_agency] approach: person %s at public distance -> come "
                    "(stop_at=%.2fm, ToF-guarded)",
                    person.get("person_db_id") or person.get("id"), stop_at,
                )
                _state["last_approach_at"] = now
            _reset("far_hits")
            return  # one maneuver per tick

    # ── EDGE-IN: drift a step closer mid-conversation (owner 2026-08-19: "If
    # he's having a conversation, he should try to get closer") ────────────────
    # The public-zone approach above closes big gaps; this closes SOCIAL distance
    # by one short, slow step — at most once per cooldown, only while genuinely
    # facing them, and only when the front ToF (the lens-truth check) shows the
    # room for it. The step keeps MOTION_EDGE_IN_KEEP_CLEAR_M of front clearance,
    # so he settles at the near edge of social distance, never in their lap.
    if (_flag("MOTION_EDGE_IN_ENABLED", True)
            and battery_ok and proactive_ok
            and getattr(profile, "conversation_active", False)
            and person.get("distance_zone") == "social"
            and facing_them):
        min_front = _num("MOTION_EDGE_IN_MIN_FRONT_M", 1.4)
        if front is not None and front >= min_front:
            _state["edge_hits"] += 1
        else:
            _state["edge_hits"] = 0
        if (_state["edge_hits"] >= int(_num("MOTION_EDGE_IN_CONFIRM_TICKS", 6))
                and (now - float(_state.get("edge_last_at") or 0.0))
                >= _num("MOTION_EDGE_IN_COOLDOWN_SECS", 240.0)):
            step_m = min(_num("MOTION_EDGE_IN_STEP_M", 0.25),
                         front - _num("MOTION_EDGE_IN_KEEP_CLEAR_M", 1.0))
            _reset("edge_hits")
            if step_m >= 0.08:
                speed = random.uniform(_num("MOTION_EDGE_IN_SPEED_MIN_MS", 0.08),
                                       _num("MOTION_EDGE_IN_SPEED_MAX_MS", 0.14))
                seq = motion_controller.move(step_m, speed=speed)
                if seq is not None:
                    _log.info(
                        "[motion_agency] edge-in: conversation at social distance, "
                        "front clear %.2fm -> one step %.2fm closer (%.2f m/s)",
                        front, step_m, speed,
                    )
                    _state["edge_last_at"] = now
                    _state["last_approach_at"] = now   # quiet windows respect it
                    return  # one maneuver per tick
    else:
        _state["edge_hits"] = 0

    # ── IDLE WANDER: nothing social needed the base this tick ─────────────────
    # Weight-shift micro-motion also runs while a person is tracked (he fidgets
    # in company, like the idle hands do) — the quiet window inside keeps it
    # clear of any maneuver the lanes above just issued.
    if _flag("MOTION_IDLE_WANDER_ENABLED", True):
        _maybe_idle_wander(profile, now)
