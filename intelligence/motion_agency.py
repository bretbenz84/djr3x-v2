"""
intelligence/motion_agency.py — autonomous base motion (owner spec 2026-07-06).

Four behaviors, evaluated once per consciousness tick (~1 Hz), highest priority
first:

REQUESTED COME — after an explicit "come here" command, rotate in bounded search
steps until face tracking acquires a person, use the tracked neck offset to turn the
base toward them, then issue the firmware `come` command with a 1 m stop distance.
The forward ToF target may be the person or an intervening obstacle, so furniture and
walls stop the approach just as safely as the intended person does.

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

REALIGN — turn the base to face the person the head is tracking. The neck servo is
the signal: face-tracking keeps the FACE centered in frame, so frame error goes to
zero even when the body points the wrong way — but the neck's offset from neutral is
exactly the body's misalignment. When the neck sits past MOTION_FACE_NECK_FRACTION of
its half-span for MOTION_FACE_CONFIRM_TICKS consecutive ticks, the base turns by a
proportional chunk and face-tracking naturally re-centers the neck as it comes around.
Iterative small corrections + a cooldown, never one exact spin (no oscillation).

APPROACH — when the tracked person stays at "public" distance (vision/proxemics:
face width < 30% of frame) for MOTION_APPROACH_CONFIRM_TICKS ticks AND the base is
already roughly facing them, issue `come`: the firmware turns to heading 0 and
advances until the nearest FORWARD ToF obstacle is MOTION_COME_STOP_AT_M away — the
person's own body is the stop target, and anything in between (furniture, wall) stops
the base the same way. No cliff sensing needed or used (owner: never upstairs).

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
    "last_turn_at": 0.0,
    "last_approach_at": 0.0,
    "last_flinch_at": 0.0,
    "user_motion_at": 0.0,   # last explicit voice motion command (stand-down window)
}


def note_user_motion() -> None:
    """Record an explicit voice motion command. The social realign/approach
    behaviors stand down for MOTION_USER_MOTION_STANDDOWN_SECS afterwards — the
    human deliberately pointed the body, and realign was rotating it right back
    (field 2026-07-23: "turn right a little" -> -45, then realign +30 toward the
    face 13 s later, reading as "I tell it to turn right, it turns left"). The
    flinch reflex and an explicit come-here request are unaffected."""
    _state["user_motion_at"] = time.monotonic()


def _user_motion_standdown(now: float) -> bool:
    at = float(_state.get("user_motion_at") or 0.0)
    if at <= 0.0:
        return False
    window = _num("MOTION_USER_MOTION_STANDDOWN_SECS", 45.0)
    return (now - at) < window

_requested_come = {
    "active": False,
    "started_at": 0.0,
    "search_turns": 0,
    "last_turn_at": 0.0,    # when the LAST chassis turn (align or scan) was issued
    "scan_sign": 1.0,       # which side the person was last known on (sweep starts there)
    "last_seen_at": 0.0,    # last time face tracking held the person — sampled EVERY
                            # tick, including while the base is mid-turn (see step()):
                            # a sighting during a scan turn must not be thrown away
    "seen_sign": 0.0,       # which way to turn to re-center that sighting (+ = left)
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


def request_come_here() -> bool:
    """Arm a bounded search/align/approach sequence for an explicit voice request."""
    if not _flag("AUTONOMOUS_MOTION_ENABLED", True) or not motion_controller.available():
        return False
    # An explicit "come here" outranks the autonomous explorer — stop it and take
    # the base (field 2026-07-23: come requests died with "room exploration owns
    # the base" and Rex kept wandering instead of coming).
    try:
        from intelligence import exploration
        if exploration.active():
            exploration.stop("come-here request takes the base")
    except Exception:
        pass
    _requested_come.update(
        active=True,
        started_at=time.monotonic(),
        search_turns=0,
        last_turn_at=0.0,
        scan_sign=1.0,
        last_seen_at=0.0,
        seen_sign=0.0,
    )
    _reset("neck_hits", "far_hits")
    _log.info("[motion_agency] requested come: searching for a visible person")
    return True


def cancel_requested_come(reason: str = "cancelled") -> None:
    if _requested_come["active"]:
        _log.info("[motion_agency] requested come: %s", reason)
    _requested_come.update(active=False, started_at=0.0, search_turns=0,
                           last_turn_at=0.0, scan_sign=1.0,
                           last_seen_at=0.0, seen_sign=0.0)


def _step_requested_come(snapshot: dict, now: float, base_idle: bool = True) -> bool:
    """Run one settled-state step. True means this mode consumed the autonomy tick.

    base_idle=False means the firmware is in BLOCKED: scanning/aligning turns are
    still safe (turning away from a block is always allowed), but the forward
    approach must wait for a clear front.
    """
    if not requested_come_active():
        return False
    timeout = _num("MOTION_COME_SEARCH_TIMEOUT_SECS", 45.0)
    max_turns = max(1, int(_num("MOTION_COME_SEARCH_MAX_TURNS", 8)))
    if ((now - float(_requested_come["started_at"])) >= timeout
            or int(_requested_come["search_turns"]) >= max_turns):
        cancel_requested_come("no person found before search limit")
        return True

    person = _tracked_person(snapshot)
    if person is None:
        # Re-acquire grace: a chassis turn WE issued swings the camera, so face
        # tracking loses the person for a beat even when they never moved (field
        # 2026-07-21: align +30° -> "person gone" -> scan spiral -> bookshelf).
        # After any issued turn, wait out the grace before concluding they're lost.
        grace = _num("MOTION_COME_REACQUIRE_GRACE_SECS", 3.0)
        last_turn = float(_requested_come["last_turn_at"])
        if last_turn > 0.0 and (now - last_turn) < grace:
            return True
        # RESIGHT: face tracking held the person moments ago — typically mid-scan,
        # when the sweeping camera swept PAST them and a follow-up micro-turn (e.g.
        # compass correction) lost the lock again (field 2026-07-23: lock on Bret at
        # scan turn 3, sweep continued to -180 and Rex pirouetted instead of coming).
        # Turn a small step back toward that sighting and restart the sweep budget
        # centered there, instead of taking the next ever-bigger sweep leg away.
        fresh = _num("MOTION_COME_SIGHT_FRESH_SECS", 6.0)
        seen_at = float(_requested_come["last_seen_at"])
        seen_sign = float(_requested_come["seen_sign"])
        if seen_at > 0.0 and (now - seen_at) < fresh and seen_sign != 0.0:
            deg = abs(_num("MOTION_COME_RESIGHT_TURN_DEG", 30.0))
            seq = motion_controller.turn(seen_sign * deg)
            if seq is not None:
                _requested_come["last_turn_at"] = now
                _requested_come["search_turns"] = 0
                _requested_come["scan_sign"] = seen_sign
                _log.info(
                    "[motion_agency] requested come: recent sighting %.1fs ago — "
                    "turning back %+.0f deg toward it",
                    now - seen_at, seen_sign * deg,
                )
            return True
        # Sweep AROUND the last-known side instead of spiraling one direction: net
        # offsets +45, -45, +90, -90, ... (x scan_sign), so the search stays centered
        # on where the person actually was. Relative command i (1-based):
        #   sign = scan_sign * (-1)^(i+1),  magnitude = deg * i.
        deg = abs(_num("MOTION_COME_SEARCH_TURN_DEG", 45.0))
        i = int(_requested_come["search_turns"]) + 1
        sign = float(_requested_come["scan_sign"]) * (1.0 if i % 2 == 1 else -1.0)
        rel = sign * deg * i
        # Always rotate the SHORT way to the target offset: the raw relative command
        # grows to +/-225, +/-270 in the later sweep steps, which the chassis executed
        # as multi-second pirouettes ("he just spins"). Same net heading, shorter arc.
        rel = ((rel + 180.0) % 360.0) - 180.0
        seq = motion_controller.turn(rel)
        if seq is not None:
            _requested_come["search_turns"] = i
            _requested_come["last_turn_at"] = now
            _log.info(
                "[motion_agency] requested come: scan turn %d/%d (%+.0f deg, sweep)",
                i, max_turns, rel,
            )
        return True

    frac = neck_offset_fraction()
    centered = _num("MOTION_APPROACH_CENTERED_FRACTION", 0.18)
    if frac is not None and abs(frac) >= centered:
        deg = _turn_degrees_for(frac)
        if motion_controller.turn(deg) is not None:
            _requested_come["last_turn_at"] = now
            # Remember which side they were on: if the align turn loses them, the
            # sweep starts back toward that side, not away from it.
            _requested_come["scan_sign"] = 1.0 if deg >= 0 else -1.0
            _requested_come["search_turns"] = 0     # fresh sweep budget after a sighting
            _log.info(
                "[motion_agency] requested come: acquired person %s, aligning %+.0f deg",
                person.get("person_db_id") or person.get("id"), deg,
            )
        return True

    if not base_idle:
        # Person found and centered but the front is momentarily blocked — hold
        # this tick; the approach starts once the zone clears (firmware final say).
        return True
    stop_at = _num("MOTION_COME_REQUEST_STOP_AT_M", 1.0)
    seq = motion_controller.come(0.0, stop_at=stop_at)
    if seq is not None:
        _log.info(
            "[motion_agency] requested come: approaching person %s "
            "(stop_at=%.2fm, obstacle-gated)",
            person.get("person_db_id") or person.get("id"), stop_at,
        )
        cancel_requested_come("approach issued")
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


def _tracked_person(snapshot: dict) -> Optional[dict]:
    """The world_state person entry the head is currently locked onto, or None."""
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
            if kind == "db" and str(person.get("person_db_id")) == value:
                return person
            if kind != "db" and str(person.get("id")) == value:
                return person
        return None
    except Exception:
        return None


def _turn_degrees_for(frac: float) -> float:
    """Base turn (deg, + = left/CCW per the wire protocol) that reduces a neck
    offset fraction. Neck toward Rex's right (+frac) needs a RIGHT (CW, negative)
    base turn; MOTION_FACE_TURN_INVERT flips if field testing disagrees."""
    max_deg = _num("MOTION_FACE_TURN_MAX_DEG", 60.0)
    min_deg = _num("MOTION_FACE_TURN_MIN_DEG", 10.0)
    deg = -frac * max_deg
    if _flag("MOTION_FACE_TURN_INVERT", False):
        deg = -deg
    if abs(deg) < min_deg:
        deg = min_deg if deg >= 0 else -min_deg
    return max(-max_deg, min(max_deg, deg))


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

    st = motion.state()

    # SIGHTING SAMPLER for an active come request — runs on EVERY tick, including
    # while the base is mid-turn. Scan turns sweep the camera across the person for
    # only a moment; the settled-state step below never runs during that moment, so
    # without this the sighting is thrown away and the sweep spins right past them
    # (field 2026-07-23: face lock on Bret during scan turn 3, sweep went to -180).
    if requested_come_active():
        seen = _tracked_person(snapshot)
        if seen is not None:
            _requested_come["last_seen_at"] = time.monotonic()
            frac = neck_offset_fraction()
            if frac is not None and abs(frac) > 0.05:
                # Same convention as _turn_degrees_for: positive neck offset
                # (person to Rex's right) needs a negative (CW) base turn.
                _requested_come["seen_sign"] = -1.0 if frac >= 0 else 1.0

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

    # The human just steered the body by voice — honor their placement instead of
    # rotating it back toward their face (see note_user_motion). Counters reset so
    # stale off-center ticks can't fire the instant the window expires.
    if _user_motion_standdown(time.monotonic()):
        _reset("neck_hits", "far_hits")
        return

    person = _tracked_person(snapshot)
    if person is None:
        _reset("neck_hits", "far_hits")
        return

    frac = neck_offset_fraction()

    # ── REALIGN: rotate the base under the head ──────────────────────────────
    if _flag("MOTION_FACE_PERSON_ENABLED", True) and frac is not None:
        threshold = _num("MOTION_FACE_NECK_FRACTION", 0.30)
        if abs(frac) >= threshold:
            _state["neck_hits"] += 1
        else:
            _state["neck_hits"] = 0
        confirm = int(_num("MOTION_FACE_CONFIRM_TICKS", 2))
        cooldown = _num("MOTION_FACE_TURN_COOLDOWN_SECS", 8.0)
        if (_state["neck_hits"] >= confirm
                and (now - _state["last_turn_at"]) >= cooldown):
            deg = _turn_degrees_for(frac)
            seq = motion_controller.turn(deg)
            if seq is not None:
                _log.info(
                    "[motion_agency] realign: neck %.0f%% off-center -> base turn %+.0f deg "
                    "(person=%s)",
                    frac * 100.0, deg, person.get("person_db_id") or person.get("id"),
                )
                _state["last_turn_at"] = now
            _reset("neck_hits")
            return  # one maneuver per tick

    # ── APPROACH: close distance to a far person ──────────────────────────────
    if not _flag("MOTION_APPROACH_ENABLED", True):
        return
    # Critical battery: stop VOLUNTEERING drives (voice-commanded motion still
    # obeys — the pack's BMS is the hard protection; this is Rex pacing himself).
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            _reset("far_hits")
            return
    except Exception:
        pass
    # A whole-base approach is a big proactive act — respect the same social gates
    # as unsolicited speech, plus require an active turn NOT being processed.
    if getattr(profile, "suppress_proactive", False) or getattr(profile, "interaction_busy", False):
        _reset("far_hits")
        return
    centered = _num("MOTION_APPROACH_CENTERED_FRACTION", 0.18)
    facing_them = frac is None or abs(frac) < centered
    if person.get("distance_zone") == "public" and facing_them:
        _state["far_hits"] += 1
    else:
        _state["far_hits"] = 0
    confirm = int(_num("MOTION_APPROACH_CONFIRM_TICKS", 4))
    cooldown = _num("MOTION_APPROACH_COOLDOWN_SECS", 120.0)
    if (_state["far_hits"] >= confirm
            and (now - _state["last_approach_at"]) >= cooldown):
        seq = motion_controller.come(0.0)  # firmware stops MOTION_COME_STOP_AT_M short
        if seq is not None:
            _log.info(
                "[motion_agency] approach: person %s at public distance -> come "
                "(stop_at=%.2fm, ToF-guarded)",
                person.get("person_db_id") or person.get("id"),
                _num("MOTION_COME_STOP_AT_M", 0.60),
            )
            _state["last_approach_at"] = now
        _reset("far_hits")
