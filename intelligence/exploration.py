"""Room exploration mode — an invited, self-directed wander.

Someone invites Rex to look around ("feel free to explore the room", "make
yourself at home"). The normal turn-based conversation hands off to this mode:
Rex takes the floor, drives several varied legs around the room, snaps pictures at
each stop, sends them to one OpenAI vision call that ranks what is interesting
(art / oddities / people over generic furniture), riffs whimsically, and
eventually FIXATES on something worth a bigger beat — never on the first stop.
The fixation ends the walk and seeds the conversation with what he found.

Ownership: while a session is live this module OWNS the base (motion_agency
stands down), the head (face tracking stands down + a session gaze hold), and the
conversational floor (`speech_engine.can_proactive_speak` denies while `active()`).
It is interruptible by voice at any point via `handle_user_turn`.

The session runs on a dedicated daemon WORKER thread: the sequence blocks on
motion completion + multi-second vision calls and must not stall the ~1 Hz
consciousness tick. The worker re-checks an abort `Event` between every step; the
consciousness tick only SUPERVISES (`supervise()` force-cleans a wedged session).

Hardware reality (see config's EXPLORE_* header): the live robot build enables the
front 8x8 matrix ToF, so the ESP32 applies its own forward SLOW/STOP reflex. ToF is
compile-time gated and bare-board firmware builds still report clear. All driving
uses finite, closed-loop `turn`+`move` legs (never streamed `drive`, never the
person-seeking `come`) and retains a per-stop VISION floor-check for cables, clutter,
and navigation context the distance sensor cannot classify.

Every heavy dependency is imported LAZILY inside functions: this module is
imported by speech_engine / motion_agency / consciousness / interaction, so a
top-level import of those would cycle.
"""

from __future__ import annotations

import logging
import math
import re
import random
import threading
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# ── Session state (single session at a time, guarded by _lock) ────────────────

_lock = threading.RLock()
_session: Optional["_Session"] = None


class _Session:
    """One exploration run. Mutated by the worker thread; read under `_lock`."""

    def __init__(self, person_id: Optional[int], person_name: Optional[str], source: str):
        self.person_id = person_id if isinstance(person_id, int) else None
        self.person_name = person_name or None
        self.source = source or "invite"
        self.state = "announce"            # announce|exploring|paused|fixate|handoff|done
        self.created_at = time.monotonic()
        self.started_at = time.monotonic()
        self.abort = threading.Event()
        self.abort_reason = ""             # "" while running; set on stop/abort
        self.thread: Optional[threading.Thread] = None
        # Progress / budget counters.
        self.stops_done = 0
        self.legs_done = 0
        self.blocked_legs = 0
        self.vision_calls = 0
        self.vision_failures = 0
        self.lines_spoken = 0
        self.fixated = False               # a real fixation beat was delivered
        self.had_base = base_available()   # session started with a drive base attached
        # Perception bookkeeping.
        self.best: Optional[dict] = None   # best appraisal candidate seen so far
        self.riffed_keys: set[str] = set()  # candidate names/categories already riffed (dedup)
        self.dead_headings: set[int] = set()  # coarse 8-bucket headings that blocked
        self.last_riff_stop = -1           # stop index of the last riff (skip-boring cadence)
        self.last_open_direction = ""      # nav hint from the previous appraisal
        self.last_floor_hazard = ""        # veto the next forward leg when non-empty
        self.last_appraise_ok = False      # last stop produced a real vision read (floor known)
        # Odometry tether (session-start pose).
        self.start_xy: Optional[tuple] = None
        # Pause bookkeeping.
        self.pause_turns = 0               # non-encouragement turns taken while paused
        self.paused_at = 0.0

    def aborting(self) -> bool:
        return self.abort.is_set()

    def halt_requested(self) -> bool:
        """True when the worker must NOT start new motion/speech: an abort OR a
        pause (a pause halts driving/narration but keeps the session alive)."""
        return self.abort.is_set() or self.state == "paused"


# ── Public API ────────────────────────────────────────────────────────────────


def active() -> bool:
    """True while an exploration session owns the floor (running OR paused).

    Cheap, never raises. Consulted by speech_engine.can_proactive_speak,
    motion_agency, and the interaction idle loop, so it must stay fast and safe.
    """
    try:
        with _lock:
            sess = _session
            if sess is None:
                return False
            if sess.state in ("done",):
                return False
            # TTL guard so a WEDGED session can't own the floor forever — but a
            # healthy session must never lose ownership mid-run, so the effective
            # TTL always exceeds the configured duration cap (a user raising
            # EXPLORE_MAX_DURATION_SECS above the flat TTL used to release the
            # floor/base/head while the worker was still driving). Normal exits go
            # through _handoff -> state "done"; this bound is the last resort.
            ttl = max(
                float(getattr(config, "EXPLORE_STEP_TTL_SECS", 240.0)),
                float(getattr(config, "EXPLORE_MAX_DURATION_SECS", 180.0)) + 60.0,
            )
            return (time.monotonic() - sess.created_at) <= ttl
    except Exception:
        return False


def status() -> dict:
    """A shallow snapshot for logs / GUI / the interaction status pending dict."""
    try:
        with _lock:
            sess = _session
            if sess is None:
                return {"active": False}
            return {
                "active": active(),
                "state": sess.state,
                "person_id": sess.person_id,
                "stops_done": sess.stops_done,
                "legs_done": sess.legs_done,
                "vision_calls": sess.vision_calls,
                "lines_spoken": sess.lines_spoken,
                "best": (sess.best or {}).get("name") if sess.best else None,
                "best_score": (sess.best or {}).get("score") if sess.best else None,
                "age_secs": round(time.monotonic() - sess.created_at, 1),
            }
    except Exception:
        return {"active": False}


def enabled() -> bool:
    return bool(getattr(config, "EXPLORE_ENABLED", True))


def no_drive_room() -> "Optional[tuple]":
    """The (name, reason) of the current room if the owner flagged it no-drive.

    Same source of truth as the other motion behaviors (motion_agency owns the
    place lookup). None when drivable, unknown, or the check is unavailable.
    """
    try:
        from intelligence import motion_agency
        return motion_agency.no_drive_room()
    except Exception:
        return None


def can_start() -> Optional[str]:
    """Return None if a session may start, else a short reason string (for logs).

    Does NOT include the no-base case — the caller decides between the no-base
    verbal denial and a head-only fallback.
    """
    if not enabled():
        return "disabled"
    if active():
        return "already_active"
    # A room the owner told him not to drive in (carpet, house rules). The
    # traction detector would eventually grind to the same conclusion, but the
    # owner's rule is authoritative BEFORE the wheels scrub — a walk must not
    # start here while a drive base is attached. (A head-only fallback session
    # drives nothing, so the rule doesn't apply without a base.)
    if base_available() and no_drive_room() is not None:
        return "room_no_drive"
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            return "battery_critical"
    except Exception:
        pass
    try:
        from features import games
        if games.is_active():
            return "game_active"
    except Exception:
        pass
    try:
        from features import dj
        if dj.is_playing():
            return "dj_playing"
    except Exception:
        pass
    return None


def base_available() -> bool:
    try:
        from intelligence import motion_controller
        return bool(motion_controller.available())
    except Exception:
        return False


def start(person_id: Optional[int], person_name: Optional[str], source: str = "invite") -> bool:
    """Begin an exploration session. Returns True if it started.

    Preconditions are the caller's responsibility for the no-base branch; this
    re-checks `can_start()` and refuses (returns False) on any blocker. Spawns the
    worker thread and returns immediately — the ack line is spoken by the worker so
    it lands without blocking the turn pipeline.
    """
    global _session
    reason = can_start()
    if reason is not None:
        _log.info("[explore] start refused: %s", reason)
        return False
    if not (base_available() or bool(getattr(config, "EXPLORE_HEADONLY_FALLBACK_ENABLED", False))):
        _log.info("[explore] start refused: no_base")
        return False
    with _lock:
        if _session is not None and active():
            return False
        sess = _Session(person_id, person_name, source)
        _session = sess
    worker = threading.Thread(target=_run_session, args=(sess,), name="exploration", daemon=True)
    sess.thread = worker
    worker.start()
    _log.info(
        "[explore] session started (person_id=%s source=%s base=%s)",
        person_id, source, base_available(),
    )
    return True


def stop(reason: str = "stop") -> None:
    """Abort the live session from any thread. Idempotent; halts the base at once."""
    with _lock:
        sess = _session
    if sess is None:
        return
    if not sess.abort_reason:
        sess.abort_reason = reason or "stop"
    sess.abort.set()
    try:
        from intelligence import motion_controller
        motion_controller.stop()
    except Exception:
        pass


def supervise() -> None:
    """Consciousness-tick watchdog: force-clean a wedged/overrun session.

    The worker owns its own duration checks; this is the belt-and-suspenders
    backstop for a worker blocked in a long wait or a thread that died without
    tearing down. Never raises.
    """
    try:
        with _lock:
            sess = _session
        if sess is None:
            return
        overran = (time.monotonic() - sess.started_at) > float(
            getattr(config, "EXPLORE_MAX_DURATION_SECS", 180.0)
        )
        thread_dead = sess.thread is not None and not sess.thread.is_alive()
        if sess.state == "done":
            with _lock:
                if _session is sess:
                    _session = None
            return
        if overran and not sess.aborting():
            _log.warning("[explore] supervisor: session overran — aborting")
            stop("watchdog_timeout")
        elif thread_dead and sess.state != "done":
            # The worker died without running teardown — clean up so the floor is freed.
            _log.warning("[explore] supervisor: worker thread dead — force teardown")
            _handoff(sess)
            with _lock:
                if _session is sess:
                    _session = None
    except Exception as exc:
        _log.debug("[explore] supervise error: %s", exc)


def handle_user_turn(text: str, speaker_id: Optional[int]) -> Optional[str]:
    """Consume a user turn while a session is active (called BEFORE routers).

    Returns a spoken line (turn consumed) or None (turn released to normal
    routing). Precedence: stop-words end the whole mode; encouragement continues;
    anything else PAUSES the walk and defers to the normal pipeline.
    """
    with _lock:
        sess = _session
    if sess is None or not active():
        return None
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    # 1. Stop / recall — end the mode instantly and speak a sign-off.
    if _is_stop_word(cleaned):
        stop("user_recall")
        return random.choice(list(getattr(config, "EXPLORE_ABORT_LINES", []) or ["Okay, stopping."]))

    # 2. Encouragement — consume with a brief canned ack, keep exploring.
    if _is_encouragement(cleaned):
        if sess.state == "paused":
            _resume(sess)
        return random.choice(
            list(getattr(config, "EXPLORE_ENCOURAGE_ACK_LINES", []) or ["On it."])
        )

    # 3. An explicit drive command is a MANUAL TAKEOVER — end the walk and release
    # the turn so the normal motion path executes it. Pausing instead meant the
    # walk resumed 4 s later and immediately turned wherever ITS plan pointed,
    # right after the user had aimed the body ("I tell it to turn right, it turns
    # left, as if it's decided to already turn", field 2026-07-23).
    try:
        from intelligence import action_router
        motion_decision = action_router.classify_explicit_motion(cleaned)
        if motion_decision is not None and motion_decision.action in {
            "motion.turn", "motion.move", "motion.arc", "motion.come",
        }:
            stop("user_motion_command")
            return None
    except Exception:
        pass

    # 4. Anything else — a real question/comment. Pause and defer to the pipeline.
    if sess.state == "paused":
        # A SECOND non-encouragement turn while already paused: they clearly want
        # to talk. End the mode and release the turn to normal routing.
        stop("user_engaged")
        return None
    _pause(sess)
    return None  # released — the normal reply machinery answers this turn


# ── Interruption helpers ──────────────────────────────────────────────────────

_STOP_WORDS = (
    "stop", "halt", "freeze", "come back", "get back here", "that's enough",
    "thats enough", "enough exploring", "enough already", "okay okay", "ok ok",
    "quit exploring", "stop exploring", "knock it off", "cut it out", "wait wait",
    "whoa whoa", "abort",
)
_ENCOURAGE_WORDS = (
    "keep going", "keep at it", "carry on", "go on", "what else", "anything good",
    "anything interesting", "find anything", "keep looking", "keep exploring",
    "nice", "cool", "go for it",
)


def _is_stop_word(cleaned: str) -> bool:
    low = cleaned.lower()
    # Bare / near-bare stop imperative, or an explicit recall.
    if low in _STOP_WORDS:
        return True
    for phrase in _STOP_WORDS:
        if " " in phrase and phrase in low:
            return True
    # Bare single-word stop imperatives ("stop.", "halt!").
    stripped = low.strip(" .!?")
    return stripped in {"stop", "halt", "freeze", "abort"}


def _is_encouragement(cleaned: str) -> bool:
    low = cleaned.lower().strip(" .!?")
    if low in _ENCOURAGE_WORDS:
        return True
    return any(phrase in low for phrase in _ENCOURAGE_WORDS if " " in phrase)


def _pause(sess: "_Session") -> None:
    with _lock:
        if sess.state in ("exploring", "announce"):
            sess.state = "paused"
            sess.paused_at = time.monotonic()
            sess.pause_turns += 1
    try:
        from intelligence import motion_controller
        motion_controller.stop()
    except Exception:
        pass
    _log.info("[explore] paused for a user turn")


def _resume(sess: "_Session") -> None:
    with _lock:
        if sess.state == "paused":
            sess.state = "exploring"
            sess.paused_at = 0.0
    _log.info("[explore] resumed")


# ── The worker (FSM) ──────────────────────────────────────────────────────────


def _run_session(sess: "_Session") -> None:
    """Worker-thread entry point. Runs the FSM to completion; always tears down."""
    try:
        _hold_head(sess)
        # Capture the tether ORIGIN from the session-start pose (before any leg) —
        # measured from where the walk began, not from the first leg's destination.
        # The post-leg _note_tether call remains as a fallback for late telemetry.
        _note_tether(sess)
        _announce(sess)
        _explore_loop(sess)
    except Exception as exc:
        _log.debug("[explore] worker error: %s", exc)
    finally:
        try:
            _handoff(sess)
        except Exception as exc:
            _log.debug("[explore] handoff error: %s", exc)
        with _lock:
            global _session
            sess.state = "done"
            if _session is sess:
                _session = None


def _explore_loop(sess: "_Session") -> bool:
    """The main PLAN_LEG -> TRAVEL -> SURVEY -> APPRAISE -> RIFF loop.

    Returns True if the loop reached a natural end (fixation / wind-down), False if it
    bailed on an abort (the interaction thread already spoke any sign-off). The return
    is advisory only — `_handoff` decides what to seed from `sess.fixated`, not this.
    """
    max_stops = int(getattr(config, "EXPLORE_MAX_STOPS", 6))
    max_dur = float(getattr(config, "EXPLORE_MAX_DURATION_SECS", 180.0))
    max_blocked = int(getattr(config, "EXPLORE_MAX_BLOCKED_LEGS", 3))
    max_fail = int(getattr(config, "EXPLORE_VISION_MAX_FAILURES", 2))
    max_calls = int(getattr(config, "EXPLORE_VISION_MAX_CALLS", 8))

    with _lock:
        sess.state = "exploring"

    while True:
        if not _check_can_continue(sess):
            return False
        if (time.monotonic() - sess.started_at) > max_dur:
            _log.info("[explore] duration cap reached")
            break
        if sess.stops_done >= max_stops:
            _log.info("[explore] stop budget reached")
            break
        if sess.blocked_legs >= max_blocked:
            _log.info("[explore] too many blocked legs")
            break
        if sess.vision_failures >= max_fail:
            _log.info("[explore] too many vision failures — ending")
            break
        if sess.vision_calls >= max_calls:
            _log.info("[explore] vision call budget reached")
            break

        # Honor a pause: wait quietly until resumed / timed out / aborted.
        if not _await_resume_if_paused(sess):
            return False

        # PLAN_LEG + TRAVEL (a no-op entirely when locomotion is disabled / no base).
        # The very first beat OPENS with a chassis turn toward the most open ToF
        # direction plus a short capped ToF-planned move: an invite should visibly
        # answer with the body, not a camera pan (owner 2026-07-23: "when I invite
        # him to look around, he needs to first move — turn, then drive"; 2026-08-08:
        # "needs more autonomous driving"). Normal legs stay vision-gated; only the
        # bounded opening move runs pre-vision on ToF authority.
        if sess.stops_done > 0:
            _travel_one_leg(sess)
            if not _check_can_continue(sess):
                return False
        else:
            _opening_leg(sess)
            if not _check_can_continue(sess):
                return False

        # SURVEY + APPRAISE + RIFF for this stop.
        views = _survey(sess)
        if not _check_can_continue(sess):
            return False
        appraisal = _appraise(sess, views)
        sess.stops_done += 1
        if appraisal is not None:
            _update_best(sess, appraisal)  # cheap; keep best current even if we pause
            # If the user interrupted DURING the survey/vision call, don't riff or
            # fixate over the reply the normal pipeline is now giving — loop back to
            # the top where the pause/abort is honored (the base is already halted).
            if not sess.halt_requested():
                _riff(sess, appraisal)
                # FIXATION gate.
                if _should_fixate(sess):
                    _fixate(sess, sess.best)
                    return True

    # Budget/timeout/blocked exhaustion — fixate on best-so-far, else wind down.
    if not _check_can_continue(sess):
        return False
    fallback = float(getattr(config, "EXPLORE_FIXATE_FALLBACK_SCORE", 0.55))
    if (
        sess.best is not None
        and float(sess.best.get("score", 0.0)) >= fallback
        and sess.stops_done >= 1
    ):
        _fixate(sess, sess.best)
        return True
    _wind_down(sess)
    return True


def _check_can_continue(sess: "_Session") -> bool:
    """False when the session must abort NOW (abort flag or an external takeover)."""
    if sess.aborting():
        return False
    # Gamepad grab / interaction pause / base disconnect / firmware fault or estop
    # end the session (the design contract in the plan's abort table).
    try:
        from hardware import motion
        if sess.had_base and not base_available():
            _log.info("[explore] base disconnected mid-session — aborting")
            sess.abort_reason = sess.abort_reason or "base_disconnected"
            sess.abort.set()
            return False
        if base_available() and motion.owner() == "manual":
            _log.info("[explore] gamepad took the base — aborting")
            sess.abort_reason = sess.abort_reason or "manual_override"
            sess.abort.set()
            return False
        if base_available() and motion.state() in ("fault", "estop", "comms_lost"):
            _log.info("[explore] base in terminal state %r — aborting", motion.state())
            sess.abort_reason = sess.abort_reason or "base_" + motion.state()
            sess.abort.set()
            return False
        if bool(getattr(config, "INTERACTION_PAUSED", False)):
            sess.abort_reason = sess.abort_reason or "interaction_paused"
            sess.abort.set()
            return False
    except Exception:
        pass
    # The room belief can change MID-SESSION (place recognition publishes live):
    # a walk that carries him into — or re-recognizes — a no-drive room must stop
    # driving at the next step boundary, before the traction detector has to
    # grind through aborted turns to learn the same rule. Head-only sessions
    # drive nothing, so the rule doesn't end them.
    if sess.had_base and no_drive_room() is not None:
        _log.info("[explore] room is flagged no-drive — ending the walk")
        sess.abort_reason = sess.abort_reason or "room_no_drive"
        sess.abort.set()
        return False
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            _log.info("[explore] battery critical — aborting")
            sess.abort_reason = sess.abort_reason or "battery_critical"
            sess.abort.set()
            return False
    except Exception:
        pass
    # A game or music that started while the walk was paused takes over the floor.
    try:
        from features import games
        if games.is_active():
            sess.abort_reason = sess.abort_reason or "game_started"
            sess.abort.set()
            return False
    except Exception:
        pass
    try:
        from features import dj
        if dj.is_playing():
            sess.abort_reason = sess.abort_reason or "dj_started"
            sess.abort.set()
            return False
    except Exception:
        pass
    return True


def _await_resume_if_paused(sess: "_Session") -> bool:
    """Block while paused until resumed, aborted, or the resume-quiet window elapses.

    Returns False if the session should end (aborted / paused too long). While
    paused the worker drives nothing and stays silent; the normal reply pipeline is
    answering the user's turn.
    """
    if sess.state != "paused":
        return True
    resume_after = float(getattr(config, "EXPLORE_RESUME_DELAY_SECS", 4.0))
    no_reply_grace = float(getattr(config, "EXPLORE_PAUSE_NO_REPLY_GRACE_SECS", 10.0))
    # The pipeline is answering the user's released turn. We must resume only AFTER
    # that reply is delivered — NOT during the multi-second LLM+TTS generation gap,
    # when the speech queue is briefly idle. Strategy: while any output is busy, push
    # the resume deadline forward (count quiet only AFTER the reply finishes). If no
    # reply ever plays (a silent turn), fall back to a longer no-reply grace so we
    # don't resume on top of a slow-to-start answer.
    deadline = sess.paused_at + max(resume_after, no_reply_grace)
    while True:
        if not _check_can_continue(sess):
            return False
        with _lock:
            st = sess.state
        if st != "paused":
            return True  # resumed (encouragement) — keep going
        now = time.monotonic()
        if _output_busy():
            deadline = now + resume_after  # reply is playing — quiet clock restarts after it
        elif now >= deadline:
            _resume(sess)
            return True
        # A hard cap so a stalled pause can't hold the floor to the duration limit.
        if (now - sess.paused_at) > max(30.0, resume_after * 6, no_reply_grace * 3):
            _log.info("[explore] pause exceeded — ending")
            sess.abort_reason = sess.abort_reason or "pause_timeout"
            sess.abort.set()
            return False
        time.sleep(0.2)


# ── ANNOUNCE / RIFF / FIXATE / WIND-DOWN (speech) ─────────────────────────────


def _announce(sess: "_Session") -> None:
    line = random.choice(list(getattr(config, "EXPLORE_ACK_LINES", []) or ["On it."]))
    _speak(sess, line, emotion="playful", tag="explore:ack")


def _riff(sess: "_Session", appraisal: dict) -> None:
    """Speak at most one whimsical line about this stop (skip a boring stop when a
    riff already fired at the previous stop — silence reads as searching)."""
    if sess.lines_spoken >= int(getattr(config, "EXPLORE_MAX_LINES", 7)):
        return
    cand = appraisal.get("top")
    if not cand:
        return
    key = _cand_key(cand)
    boring = float(cand.get("score", 0.0)) <= float(getattr(config, "EXPLORE_BORING_MAX_SCORE", 0.35))
    if boring and sess.last_riff_stop == sess.stops_done - 1:
        return  # don't narrate two dull stops in a row
    if key in sess.riffed_keys:
        return
    sess.riffed_keys.add(key)
    directive = _riff_directive(sess, cand, boring=boring)
    line = _generate(directive)
    if not line:
        return
    if _speak(sess, line, emotion="curious", tag="explore:riff"):
        sess.last_riff_stop = sess.stops_done  # cadence tracks DELIVERED riffs only


def _fixate(sess: "_Session", cand: Optional[dict]) -> None:
    """Orient toward the winning find, speak the fixation beat (+ maybe a question)."""
    with _lock:
        sess.state = "fixate"
    if not cand:
        _wind_down(sess)
        return
    # Turn the head toward the view the find was seen in (a bounded glance).
    view = str(cand.get("view") or "center")
    _glance(view)
    ask = random.random() < float(getattr(config, "EXPLORE_FIXATE_QUESTION_PROB", 0.7))
    directive = _fixate_directive(sess, cand, ask=ask)
    line = _generate(directive)
    if not line:
        line = "Okay, THIS I have opinions about."
    # A fixation only EXISTS if the beat was actually delivered — a line dropped
    # because the user started talking (or the queue failed) must not persist a
    # fixation into memory/topic seeding that Rex never said out loud.
    if _speak(sess, line, emotion="excited", tag="explore:fixate",
              register_frame=cand if ask else None):
        sess.fixated = True
        _log.info(
            "[explore] fixated on %r (score=%.2f) after %d stops",
            cand.get("name"), float(cand.get("score", 0.0)), sess.stops_done,
        )
    else:
        _log.info("[explore] fixation line dropped — no fixation persisted (%r)", cand.get("name"))


def _wind_down(sess: "_Session") -> None:
    line = random.choice(list(getattr(config, "EXPLORE_WINDDOWN_LINES", []) or ["Tour complete."]))
    _speak(sess, line, emotion="dry", tag="explore:winddown")


# ── SURVEY + APPRAISE (perception) ────────────────────────────────────────────


def _survey(sess: "_Session") -> list:
    """Head-only gaze sweep at the current stop. Returns [(view, frame), ...].

    Mirrors features.games._ispy_scan_room: hold the gaze, pose the head, capture
    from that pose, recenter. Degrades to a single current-gaze frame without servos.
    """
    try:
        from vision import camera
    except Exception:
        return []
    views_cfg = list(getattr(config, "EXPLORE_GAZE_VIEWS", ("left", "center", "right")))
    settle = float(getattr(config, "EXPLORE_SETTLE_SECS", 0.35))

    scan_possible = True
    try:
        from hardware import servos
        scan_possible = servos.connected()
    except Exception:
        scan_possible = False
    if not scan_possible:
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []

    try:
        from intelligence import consciousness
        from sequences import animations
    except Exception:
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []

    out: list = []
    for view in views_cfg:
        if sess.aborting():
            break
        try:
            consciousness.hold_directed_gaze(view, secs=6.0)
            animations.directed_look_pose(view)
            frame = camera.capture_current_gaze(settle_secs=settle)
            if frame is not None:
                out.append((view, frame))
        except Exception as exc:
            _log.debug("[explore] survey pose %r failed: %s", view, exc)
    try:
        from sequences import animations
        animations.directed_look_pose("center")
    except Exception:
        pass
    if not out:
        frame = camera.get_frame()
        return [("center", frame)] if frame is not None else []
    return out


def _appraise(sess: "_Session", views: list) -> Optional[dict]:
    """One multi-image OpenAI call ranking what's interesting at this stop.

    Returns a normalized dict:
        {"top": <candidate|None>, "candidates": [...],
         "open_direction": str, "floor_hazard": str}
    or None on failure (which bumps the failure counter). Each candidate:
        {"name","view","category","score" (0..1),"riff_hook","novelty"}
    """
    if not views:
        # No frames at all (dead/blind camera) counts as a vision FAILURE so the
        # "never wander blind" cap trips — otherwise the loop would keep navigating
        # with no visual floor read. Firmware ToF remains the hard reflex on equipped
        # builds, but it is not permission for exploration to navigate blind.
        sess.last_appraise_ok = False
        sess.vision_failures += 1
        return None
    if sess.vision_calls >= int(getattr(config, "EXPLORE_VISION_MAX_CALLS", 8)):
        sess.last_appraise_ok = False
        return None
    sess.vision_calls += 1
    try:
        raw = _appraise_call(views)
    except Exception as exc:
        _log.debug("[explore] appraise call raised: %s", exc)
        raw = None
    if raw is None:
        sess.vision_failures += 1
        sess.last_appraise_ok = False
        return None
    parsed = _parse_appraisal(raw, views)
    if parsed is None:
        sess.vision_failures += 1
        sess.last_appraise_ok = False
        return None
    sess.vision_failures = 0
    sess.last_appraise_ok = True
    # Record the nav / safety hints for the next leg.
    sess.last_open_direction = str(parsed.get("open_direction") or "")
    sess.last_floor_hazard = str(parsed.get("floor_hazard") or "")
    # Feed the room model (novelty baseline) with what we confirmed by sight.
    _record_room_labels(parsed.get("candidates") or [])
    return parsed


def _appraise_call(views: list) -> Optional[str]:
    """Build + send the multi-image appraisal request. Returns raw model text."""
    try:
        import apikeys
        from openai import OpenAI
        from vision.image_utils import encode_jpeg_base64
    except Exception as exc:
        _log.debug("[explore] appraise deps unavailable: %s", exc)
        return None

    detail = config.VISION_DETAIL.get("explore", config.VISION_DETAIL.get("scene_analysis", "low"))
    known_names = _visible_known_names()

    content: list = []
    labeled: list = []
    for view, frame in views:
        b64 = encode_jpeg_base64(frame, quality=85)
        if b64 is None:
            continue
        labeled.append(view)
        content.append({"type": "text", "text": f"View looking {view}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    if not labeled:
        return None

    view_names = ", ".join(f'"{v}"' for v in labeled)
    names_clause = (
        f" People you already recognize (safe to name): {', '.join(known_names)}."
        if known_names else ""
    )
    content.append({
        "type": "text",
        "text": (
            "These are views of the same room from a robot (DJ-R3X) looking around while "
            "exploring. Rank what is INTERESTING to fixate on and joke about. "
            "Interesting = art, posters, paintings, instruments, memorabilia, collections, "
            "unusual machines/gadgets, strange or personal objects, pets, and PEOPLE. "
            "BORING (score <= 0.2 unless something specific is genuinely odd) = generic "
            "furniture and toys: chairs, couches, tables, desks, lamps, balls, cups, boxes.\n"
            f"{names_clause}\n"
            "Do NOT identify, guess age, health, or sensitive traits of unknown people. "
            "Do NOT read private screen text. No race/ethnicity/religion/disability/medical.\n"
            "Return ONLY a JSON object with these keys:\n"
            '  "candidates": array (up to 4) of {"name": short label, '
            f'"view": one of {view_names}, "category": one of '
            '"art"/"decor"/"object"/"person"/"animal"/"collection"/"oddity", '
            '"interest": number 0..1 (higher = more worth fixating on), '
            '"riff_hook": one concrete visible detail worth a light joke, '
            '"novelty": short note on whether it seems unusual for this room},\n'
            f'  "open_direction": which view shows the most open, unobstructed floor '
            f'to drive toward next — one of {view_names} or "none",\n'
            '  "floor_hazards": short text describing cables/steps/clutter directly '
            'ahead, or "" if the floor ahead looks clear.\n'
            "No preamble, no markdown."
        ),
    })

    try:
        # HARD request timeout so a hung OpenAI call can't wedge the worker thread
        # past the duration cap (mirrors intelligence/llm.py's client-wide timeout).
        timeout = float(getattr(config, "EXPLORE_VISION_TIMEOUT_SECS", 25.0))
        client = OpenAI(api_key=apikeys.OPENAI_API_KEY, timeout=timeout)
        resp = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        _log.error("[explore] appraisal vision call failed: %s", exc)
        return None


def _parse_appraisal(raw: str, views: list) -> Optional[dict]:
    data = _parse_json(raw)
    if not isinstance(data, dict):
        return None
    legal_views = [v for v, _ in views]
    cands: list = []
    for c in (data.get("candidates") or []):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        view = str(c.get("view") or "").strip().lower()
        if view not in legal_views:
            view = legal_views[0] if legal_views else "center"
        try:
            interest = float(c.get("interest"))
        except (TypeError, ValueError):
            interest = 0.0
        interest = max(0.0, min(1.0, interest))
        cands.append({
            "name": name,
            "view": view,
            "category": str(c.get("category") or "object").strip().lower(),
            "interest": interest,
            "riff_hook": str(c.get("riff_hook") or "").strip(),
            "novelty": str(c.get("novelty") or "").strip(),
        })
    if not cands:
        return None
    # Local scoring (boring clamp + novelty boost) → final "score".
    _score_candidates(cands)
    cands.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    open_dir = str(data.get("open_direction") or "").strip().lower()
    if open_dir not in legal_views:
        open_dir = "none"
    return {
        "top": cands[0],
        "candidates": cands,
        "open_direction": open_dir,
        "floor_hazard": _normalize_floor_hazard(data.get("floor_hazards")),
    }


# Floor dangers the ToF reflex genuinely cannot see: thin snag/trip lines, level
# changes, and spills. Anything solid enough to be an "obstacle" IS visible to the
# firmware ToF (radial ring + floor-rejected front matrix), which stops the base on
# its own — vision prose about generic mess must not ground the robot.
_FLOOR_HAZARD_WORDS = (
    "cable", "cord", "wire", "charger", "strap", "string", "ribbon", "leash",
    "step", "stair", "ledge", "drop", "edge of", "threshold",
    "spill", "liquid", "puddle", "wet", "broken glass", "shard",
    "rug edge", "curled",
)


def _normalize_floor_hazard(value) -> str:
    """Keep only floor dangers the ToF cannot catch; everything else is clear.

    The vision model happily writes travelogue ("Some clutter around the kitchen
    area.") and any non-empty text used to veto the forward leg — field 2026-07-23:
    every leg in a messy room became turn-only and "look around the room" produced a
    robot that pans and never drives. Solid clutter is the firmware ToF's job (it
    stops for it reflexively at 0.32 m/s); the semantic gate is ONLY for what the
    ToF physically misses: cables/cords, steps/ledges, and liquids.
    """
    text = str(value or "").strip()
    lowered = text.lower().strip(" .!-")
    if not lowered or lowered in {"none", "no", "n/a", "clear", "no hazards"}:
        return ""
    if ("clear" in lowered or "unobstructed" in lowered) and not any(
        word in lowered for word in ("not clear", "unclear", "cable", "cord", "step")
    ):
        return ""
    if not any(word in lowered for word in _FLOOR_HAZARD_WORDS):
        _log.info("[explore] floor note without a ToF-blind hazard (%r) — driving anyway", text)
        return ""
    return text


# ── Scoring (deterministic, after the vision call) ────────────────────────────


def _person_candidate_allowed() -> bool:
    """Fail-closed gate for riffing/fixating on a PERSON candidate.

    Mirrors the roast-vision safety policy: a person may be the subject of an
    exploration beat ONLY when every currently visible person is a RECOGNIZED,
    known NON-minor. An unidentified face on camera (could be a child), any known
    minor, or any error resolving identity/age -> False — never assume adult.
    Person candidates that fail this gate are clamped at scoring time so they can
    never win a riff or a fixation (they remain fine as scene context).
    """
    try:
        from world_state import world_state
        from vision import face
        from intelligence import profile_questions
        people = world_state.get("people") or []
        visible = [
            p for p in people
            if isinstance(p, dict)
            and p.get("face_visible") is not False
            and not p.get("face_missing")
        ]
        if not visible:
            return False  # a person candidate with nobody visibly tracked = stale/phantom
        for p in visible:
            if p.get("person_db_id") is None:
                return False  # unidentified person on camera — could be a minor
        known_ids = {pid for pid, _name in face.visible_known_people()}
        if not known_ids:
            return False
        for pid in known_ids:
            if profile_questions.person_is_minor(int(pid)):
                return False
        return True
    except Exception:
        return False  # never assume adult


def _score_candidates(cands: list) -> None:
    """Set each candidate's final "score": model interest, boring-clamped +
    novelty-boosted. Person candidates additionally pass the fail-closed
    minor/identity gate or are clamped like boring items. Mutates in place."""
    boring_labels = set(getattr(config, "EXPLORE_BORING_LABELS", set()))
    boring_cap = float(getattr(config, "EXPLORE_BORING_MAX_SCORE", 0.35))
    boost = float(getattr(config, "EXPLORE_NOVELTY_BOOST", 0.15))
    sightings = _label_sightings([c["name"] for c in cands] + [c["category"] for c in cands])
    # Evaluate the person gate at most once per stop, and only when needed.
    person_allowed: Optional[bool] = None
    for c in cands:
        score = float(c.get("interest", 0.0))
        name_l = c["name"].lower()
        cat_l = c["category"].lower()
        is_person = cat_l == "person" or name_l in ("person", "people", "someone", "a person")
        if is_person:
            if person_allowed is None:
                person_allowed = _person_candidate_allowed()
            if not person_allowed:
                # Blocked person subject: clamp like a boring item so it can never
                # become best / riff / fixation. Stays in candidates as context.
                c["boring"] = True
                c["person_blocked"] = True
                c["score"] = min(score, boring_cap)
                continue
        is_boring = not is_person and (
            # People are interesting BY SPEC; their names often carry locational
            # furniture words ("person by the window") that must not clamp them.
            cat_l in boring_labels
            or name_l in boring_labels
            or any(w in boring_labels for w in name_l.split())
        )
        if is_boring:
            score = min(score, boring_cap)
        else:
            # Novelty boost: a label Rex has rarely/never logged stands out.
            seen = min(
                sightings.get(name_l, 0),
                sightings.get(cat_l, 0) if cat_l in sightings else 999,
            )
            if seen <= 1:
                score = min(1.0, score + boost)
        c["boring"] = is_boring
        c["score"] = max(0.0, min(1.0, score))


def _update_best(sess: "_Session", appraisal: dict) -> Optional[dict]:
    top = appraisal.get("top")
    if not top:
        return sess.best
    if sess.best is None or float(top.get("score", 0.0)) > float(sess.best.get("score", 0.0)):
        sess.best = dict(top)
    return sess.best


def _should_fixate(sess: "_Session") -> bool:
    """Fixation gate: enough observation and travel, threshold, not boring."""
    min_stops = int(getattr(config, "EXPLORE_MIN_STOPS_BEFORE_FIXATE", 2))
    if sess.stops_done < min_stops:
        return False
    # A mobile Rex should actually wander before the first strong visual candidate
    # ends the mode. Head-only fallback has no locomotion requirement.
    if (
        sess.had_base
        and bool(getattr(config, "EXPLORE_LOCOMOTION_ENABLED", True))
        and sess.legs_done < int(getattr(config, "EXPLORE_MIN_LEGS_BEFORE_FIXATE", 3))
    ):
        return False
    if sess.best is None:
        return False
    if sess.best.get("boring"):
        return False
    return float(sess.best.get("score", 0.0)) >= float(getattr(config, "EXPLORE_FIXATE_MIN_SCORE", 0.75))


# ── LOCOMOTION (PLAN_LEG + TRAVEL) ────────────────────────────────────────────


def _opening_leg(sess: "_Session") -> None:
    """First-beat chassis motion: turn toward the most open ToF direction, then a
    short capped forward move when the front ToF shows room.

    Runs before the first survey so accepting an invite LOOKS like exploring from
    the first second — and answers it with real driving, not just a pan (owner
    2026-08-08: "an invitation to look around the room needs more autonomous
    driving"). No vision read exists yet, so the opening move deliberately runs on
    ToF authority alone: distance is planned from the live fl/fr clearance, capped
    at EXPLORE_OPENING_LEG_MAX_M (shorter than a normal leg), and the firmware
    reflex still guards the path. The vision floor-gate picks up from stop 1.

    Invited while the base is ALREADY moving (e.g. mid "drive forward"): leave the
    in-flight command alone — the travel gaze sweeps the head until the base
    stops, which is exactly the "look around while you drive" the invite asked for.
    """
    if not bool(getattr(config, "EXPLORE_LOCOMOTION_ENABLED", True)):
        return
    if not base_available() or sess.halt_requested():
        return
    try:
        from intelligence import motion_controller
    except Exception:
        return
    gaze = _start_travel_gaze(sess)
    try:
        if motion_controller.is_moving():
            _ride_out_current_motion(sess)
            return
        deg = _plan_leg_heading(sess)
        min_deg = float(getattr(config, "EXPLORE_OPENING_TURN_MIN_DEG", 30.0))
        if abs(deg) < min_deg:
            # An open corridor straight ahead plans ~0° — still make the invite
            # visibly physical with a modest scan turn toward the wider ToF side.
            deg = min_deg if deg >= 0.0 else -min_deg
        rate = float(getattr(config, "EXPLORE_TURN_RATE_DEG_S", 70.0))
        seq = motion_controller.turn(deg, rate=rate)
        if seq is None:
            _log.info("[explore] opening turn suppressed (gated)")
            return
        _log.info("[explore] opening turn %+.0f deg (invite answered with the base)", deg)
        if not _wait_leg_done(sess, seq) or sess.halt_requested():
            return

        cap = float(getattr(config, "EXPLORE_OPENING_LEG_MAX_M", 0.8))
        if cap <= 0.0:
            return
        dist = _plan_leg_distance()
        if dist is None:
            _log.info("[explore] no forward ToF clearance — opening turn only")
            return
        dist = min(dist, cap)
        speed = float(getattr(config, "EXPLORE_LEG_SPEED_MS", 0.32))
        seq = motion_controller.move(dist, speed=speed)
        if seq is None:
            return
        _log.info("[explore] opening move %.2fm (ToF-planned, pre-vision)", dist)
        result = _wait_leg_done(sess, seq)
        sess.legs_done += 1
        if result == "blocked":
            sess.blocked_legs += 1
            sess.dead_headings.add(_heading_bucket(sess))
        elif result:
            sess.blocked_legs = 0
        _note_tether(sess)
    finally:
        _stop_travel_gaze(gaze)


def _ride_out_current_motion(sess: "_Session") -> None:
    """Wait (bounded) for an in-flight base command to finish on its own."""
    try:
        from intelligence import motion_controller
    except Exception:
        return
    _log.info("[explore] invited mid-drive — sweeping the head until the base stops")
    deadline = time.monotonic() + float(
        getattr(config, "EXPLORE_LEG_DONE_TIMEOUT_SECS", 12.0)
    )
    while time.monotonic() < deadline:
        if sess.halt_requested() or not motion_controller.is_moving():
            return
        time.sleep(0.1)


def _travel_one_leg(sess: "_Session") -> None:
    """Turn to a chosen heading, then drive one short forward leg. Closed-loop.

    No-op when locomotion is disabled or no base is connected (Phase-1 stationary
    behavior). Marks a blocked heading and bumps blocked_legs on a blocked/failed
    leg; resets the counter on a clean leg.
    """
    if not bool(getattr(config, "EXPLORE_LOCOMOTION_ENABLED", True)):
        return
    if not base_available():
        return
    if sess.halt_requested():
        return
    # NEVER DRIVE BLIND: firmware ToF is the hard distance reflex on the live build;
    # the per-stop vision read is complementary protection for cables/clutter and also
    # confirms that this particular firmware/hardware session can perceive the route.
    # If the prior appraisal failed, stay put and re-survey.
    if not sess.last_appraise_ok:
        _log.info("[explore] no vision read from the last stop — holding position (no blind leg)")
        return
    try:
        from intelligence import motion_controller
    except Exception:
        return

    # PLAN_LEG: pick a varied heading change, biased by open floor + the tether.
    deg = _plan_leg_heading(sess)
    gaze = _start_travel_gaze(sess)

    try:
        # Turn (bounded, closed-loop) while the head independently looks around.
        rate = float(getattr(config, "EXPLORE_TURN_RATE_DEG_S", 70.0))
        if abs(deg) >= 1.0:
            seq = motion_controller.turn(deg, rate=rate)
            if seq is None:
                _log.info("[explore] turn suppressed (gated) — skipping leg")
                return
            if not _wait_leg_done(sess, seq):
                return

        # A stop/abort OR a pause that landed during the turn must cancel the forward
        # move. Checked immediately before the send to keep the TOCTOU window minimal.
        if sess.halt_requested():
            return

        # Vision adds a semantic floor gate on top of the firmware ToF reflex.
        if sess.last_floor_hazard:
            _log.info("[explore] floor hazard ahead (%r) — turn-only leg", sess.last_floor_hazard)
            sess.last_floor_hazard = ""
            return

        # Re-read the now-forward ToF pair after the turn. The front matrix is
        # min-combined into fl/fr by firmware, so this clearance includes both the
        # radial beams and all floor-rejected 8x8 zones. Travel a fraction of the
        # measured opening, leaving a fixed body margin — and while the front STAYS
        # open after a move, chain further segments (up to EXPLORE_LEG_SEGMENTS_MAX)
        # before stopping to survey: a look-around should cross the room, not inch
        # across it one vision call at a time.
        dist = _plan_leg_distance()
        if dist is None:
            _log.info("[explore] no safe forward ToF clearance — holding position")
            return
        speed = float(getattr(config, "EXPLORE_LEG_SPEED_MS", 0.32))
        max_segments = int(getattr(config, "EXPLORE_LEG_SEGMENTS_MAX", 3))
        continue_min = float(getattr(config, "EXPLORE_LEG_CONTINUE_MIN_M", 0.40))
        segments = 0
        while True:
            seq = motion_controller.move(dist, speed=speed)
            if seq is None:
                _log.info("[explore] move suppressed (gated)")
                return
            result = _wait_leg_done(sess, seq)
            sess.legs_done += 1
            segments += 1
            _note_tether(sess)
            if result == "blocked":
                sess.blocked_legs += 1
                sess.dead_headings.add(_heading_bucket(sess))
                _log.info("[explore] leg blocked — heading marked dead")
                return
            if not result:
                return
            sess.blocked_legs = 0
            if segments >= max_segments or sess.halt_requested() or _beyond_tether(sess):
                return
            nxt = _plan_leg_distance()
            if nxt is None or nxt < continue_min:
                return
            dist = nxt
            _log.info("[explore] front still open (%.2fm planned) — continuing the leg", nxt)
    finally:
        _stop_travel_gaze(gaze)


def _plan_leg_heading(sess: "_Session") -> float:
    """Choose the most open live ToF direction (+ = left/CCW).

    The eight radial sensors form an occupancy ring. Firmware overlays the front
    matrix onto fl/fr, so front candidates also account for its 64 zones. Dead
    world headings are rejected and near-ties favor smaller turns, producing a
    purposeful open-space walk instead of a random tolerance sample.
    """
    max_deg = float(getattr(config, "EXPLORE_TURN_MAX_DEG", 120.0))
    min_deg = min(max_deg, float(getattr(config, "EXPLORE_TURN_MIN_DEG", 35.0)))
    # Tether: if we've wandered past the leash, turn back toward start.
    if _beyond_tether(sess):
        toward = _heading_toward_start(sess)
        if toward is not None:
            return max(-max_deg, min(max_deg, toward))
    candidates = _tof_heading_candidates(sess, max_deg)
    if candidates:
        # Max clearance first. Within 15 cm, prefer the smaller chassis rotation.
        best_clear = max(c[1] for c in candidates)
        near_best = [c for c in candidates if c[1] >= best_clear - 0.15]
        deg, clearance, sensor = min(near_best, key=lambda c: abs(c[0]))
        _log.info(
            "[explore] ToF route: sensor=%s clearance=%.2fm turn=%+.0fdeg",
            sensor, clearance, deg,
        )
        return deg

    # Sensor data unavailable: retain the semantic camera hint for orientation,
    # but the distance planner below will still refuse to drive blind.
    hint = (sess.last_open_direction or "").lower()
    if hint == "left":
        return min_deg
    if hint == "right":
        return -min_deg
    if hint == "center":
        return 0.0
    return min_deg if (sess.legs_done % 2 == 0) else -min_deg


_TOF_RELATIVE_DEG = {
    "fl": 22.5, "lf": 67.5, "lb": 112.5, "rl": 157.5,
    "rr": -157.5, "rb": -112.5, "rf": -67.5, "fr": -22.5,
}


def _tof_mm() -> dict:
    try:
        from hardware import motion
        tof = (motion.telemetry() or {}).get("tof_mm") or {}
        return tof if isinstance(tof, dict) else {}
    except Exception:
        return {}


def _valid_tof_m(value) -> Optional[float]:
    try:
        mm = float(value)
    except (TypeError, ValueError):
        return None
    return mm / 1000.0 if mm >= 0.0 else None


def _tof_heading_candidates(sess: "_Session", max_deg: float) -> list[tuple[float, float, str]]:
    tof = _tof_mm()
    try:
        from hardware import motion
        theta = float(((motion.telemetry() or {}).get("odom") or {}).get("theta") or 0.0)
    except Exception:
        theta = 0.0
    out = []
    # The two ±22.5° front beams (and the matrix halves overlaid onto them) jointly
    # describe a straight-ahead corridor. Add that corridor explicitly so an open
    # room does not cause a needless 22.5° turn on every leg.
    front_pair = [_valid_tof_m(tof.get(k)) for k in ("fl", "fr")]
    front_pair = [d for d in front_pair if d is not None]
    if front_pair:
        world_bucket = int(round(math.degrees(theta) / 45.0)) % 8
        if world_bucket not in sess.dead_headings:
            out.append((0.0, min(front_pair), "front+matrix"))
    for sensor, deg in _TOF_RELATIVE_DEG.items():
        if abs(deg) > max_deg:
            continue
        clearance = _valid_tof_m(tof.get(sensor))
        if clearance is None:
            continue
        world_bucket = int(round((math.degrees(theta) + deg) / 45.0)) % 8
        if world_bucket in sess.dead_headings:
            continue
        out.append((deg, clearance, sensor))
    return out


def _plan_leg_distance() -> Optional[float]:
    """Derive forward travel from current fl/fr clearance; never choose randomly."""
    tof = _tof_mm()
    front = [_valid_tof_m(tof.get(k)) for k in ("fl", "fr")]
    front = [d for d in front if d is not None]
    if not front:
        return None
    clearance = min(front)
    margin = float(getattr(config, "EXPLORE_CLEARANCE_MARGIN_M", 0.45))
    fraction = float(getattr(config, "EXPLORE_CLEARANCE_FRACTION", 0.65))
    max_dist = float(getattr(config, "EXPLORE_LEG_MAX_M", 1.50))
    min_dist = float(getattr(config, "EXPLORE_LEG_MIN_M", 0.20))
    dist = min(max_dist, max(0.0, clearance - margin) * fraction)
    return dist if dist >= min_dist else None


def _wait_leg_done(sess: "_Session", seq) -> "str | bool":
    """Wait for a finite motion command to finish. Returns the done result string
    ("completed"/"blocked"/...), True on a truthy-but-unknown done, or False if it
    timed out / aborted."""
    if not isinstance(seq, int):
        return True  # arc/unknown — nothing to wait on
    timeout = float(getattr(config, "EXPLORE_LEG_DONE_TIMEOUT_SECS", 12.0))
    try:
        from hardware import motion
        done = motion.wait_done(seq, timeout=timeout)
    except Exception:
        done = None
    if sess.halt_requested():
        return False
    if done is None:
        # Timed out — stop the base defensively and re-check state.
        try:
            from intelligence import motion_controller
            motion_controller.stop()
        except Exception:
            pass
        return False
    result = str(done.get("result") or "").lower()
    return result or True


def _note_tether(sess: "_Session") -> None:
    xy = _current_xy()
    if xy is None:
        return
    if sess.start_xy is None:
        sess.start_xy = xy


def _beyond_tether(sess: "_Session") -> bool:
    if sess.start_xy is None:
        return False
    cur = _current_xy()
    if cur is None:
        return False
    r = float(getattr(config, "EXPLORE_TETHER_RADIUS_M", 3.0))
    dx = cur[0] - sess.start_xy[0]
    dy = cur[1] - sess.start_xy[1]
    return (dx * dx + dy * dy) > (r * r)


def _heading_toward_start(sess: "_Session") -> Optional[float]:
    """A bounded heading change that points roughly back toward the start pose."""
    try:
        import math
        from hardware import motion
        tel = motion.telemetry() or {}
        odom = tel.get("odom") or {}
        cur = _current_xy()
        theta = float(odom.get("theta") or 0.0)
        if cur is None or sess.start_xy is None:
            return None
        dx = sess.start_xy[0] - cur[0]
        dy = sess.start_xy[1] - cur[1]
        target = math.atan2(dy, dx)
        delta = math.degrees(_wrap_angle(target - theta))
        max_deg = float(getattr(config, "EXPLORE_TURN_MAX_DEG", 75.0))
        return max(-max_deg, min(max_deg, delta))
    except Exception:
        return None


def _current_xy() -> Optional[tuple]:
    try:
        from hardware import motion
        tel = motion.telemetry() or {}
        odom = tel.get("odom") or {}
        return (float(odom.get("x") or 0.0), float(odom.get("y") or 0.0))
    except Exception:
        return None


def _heading_bucket(sess: "_Session") -> int:
    try:
        import math
        from hardware import motion
        tel = motion.telemetry() or {}
        theta = float((tel.get("odom") or {}).get("theta") or 0.0)
        return int(round(math.degrees(theta) / 45.0)) % 8
    except Exception:
        return 0


def _wrap_angle(a: float) -> float:
    import math
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ── Head / gaze ownership ─────────────────────────────────────────────────────


def _start_travel_gaze(
    sess: "_Session",
) -> Optional[tuple[threading.Event, threading.Thread]]:
    """Start independent head motion for the duration of one base leg.

    The base worker continues issuing and waiting on finite ESP32 commands while
    this companion thread chooses its own glance sequence. Camera appraisal still
    happens only after both settle at the next stop, avoiding motion-blurred frames.
    Returns an opaque ``(stop_event, thread)`` handle or None.
    """
    if not bool(getattr(config, "EXPLORE_TRAVEL_GAZE_ENABLED", True)):
        return None
    try:
        from hardware import servos
        if not servos.connected():
            return None
    except Exception:
        return None
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_travel_gaze_loop,
        args=(sess, stop_event),
        name="exploration-gaze",
        daemon=True,
    )
    worker.start()
    return stop_event, worker


def _travel_gaze_loop(sess: "_Session", stop_event: threading.Event) -> None:
    """Sweep the head left-right while the base drives; settle center at the stop.

    The sweep alternates sides deterministically — a back-and-forth watch of the
    room while he rolls, not random darting (per-glance randomness read as
    stutter). Each swing draws a pitch with equal weight from
    EXPLORE_TRAVEL_GAZE_PITCHES, so he looks down (floor/low shelves) as often as
    level and up. When the leg ends the head settles back to center — the
    stationary survey scan then starts from a composed pose.
    """
    try:
        from sequences import animations
    except Exception:
        return
    pitches = [
        str(p).lower()
        for p in getattr(config, "EXPLORE_TRAVEL_GAZE_PITCHES", ("down", "level", "up"))
        if str(p).lower() in ("down", "level", "up")
    ] or ["level"]
    hold = float(getattr(config, "EXPLORE_TRAVEL_GAZE_HOLD_SECS", 0.8))
    side = random.choice(("left", "right"))
    try:
        while not stop_event.is_set() and not sess.halt_requested():
            try:
                animations.travel_glance_pose(side, random.choice(pitches))
            except Exception as exc:
                _log.debug("[explore] travel glance failed: %s", exc)
                return
            side = "right" if side == "left" else "left"
            if stop_event.wait(max(0.1, hold)):
                break
    finally:
        # Settle in the middle when the base stops.
        try:
            animations.travel_glance_pose("center", "level")
        except Exception:
            pass


def _stop_travel_gaze(
    handle: Optional[tuple[threading.Event, threading.Thread]],
) -> None:
    """Stop and briefly join a per-leg gaze worker before stationary capture."""
    if handle is None:
        return
    try:
        stop_event, worker = handle
        stop_event.set()
        # directed_look_pose is bounded but blocking. Joining prevents its current
        # servo move from racing the stationary survey's first capture pose.
        worker.join(timeout=8.0)
    except Exception:
        pass


def _hold_head(sess: "_Session") -> None:
    """Pin a session gaze hold so the face-tracking loop doesn't fight the walk."""
    try:
        from intelligence import consciousness
        secs = float(getattr(config, "EXPLORE_MAX_DURATION_SECS", 180.0)) + 10.0
        consciousness.hold_directed_gaze("hold", secs=secs)
    except Exception:
        pass


def _release_head() -> None:
    try:
        from intelligence import consciousness
        consciousness.clear_directed_gaze_hold()
        from sequences import animations
        animations.directed_look_pose("center")
    except Exception:
        pass


def _glance(view: str) -> None:
    try:
        if str(view or "").lower() not in ("left", "right", "up", "down", "center"):
            return
        from sequences import animations
        animations.directed_look_pose(view)
    except Exception:
        pass


# ── HANDOFF ───────────────────────────────────────────────────────────────────


def _handoff(sess: "_Session") -> None:
    """Release the floor + head, seed the conversation with the find, capture memory.

    Runs on EVERY exit path (finally-block) so an abort can never leave gates
    latched. Idempotent-ish (guarded so a double-call is harmless).
    """
    if getattr(sess, "_handed_off", False):
        return
    sess._handed_off = True
    with _lock:
        sess.state = "handoff"
    # Stop the base + release the head/gaze holds.
    try:
        from intelligence import motion_controller
        motion_controller.stop()
    except Exception:
        pass
    _release_head()

    reason = sess.abort_reason or "complete"
    # Seed the conversation + memory only when the walk actually FIXATED on a find
    # (not a wind-down, a user-recall, or a silent abort).
    fixated = sess.fixated and sess.best is not None
    if reason in ("user_recall", "user_engaged"):
        fixated = False

    if fixated and sess.best is not None:
        _seed_topic(sess, sess.best)
        _record_episode(sess, sess.best)
        _bank_callback(sess, sess.best)

    _log.info(
        "[explore] session ended (reason=%s stops=%d legs=%d blocked=%d vision=%d "
        "lines=%d best=%r score=%s)",
        reason, sess.stops_done, sess.legs_done, sess.blocked_legs, sess.vision_calls,
        sess.lines_spoken, (sess.best or {}).get("name"),
        (sess.best or {}).get("score"),
    )


def _seed_topic(sess: "_Session", cand: dict) -> None:
    """Register the fixation as a live topic + reply frame so the next turn binds."""
    label = str(cand.get("name") or "").strip()
    if not label:
        return
    try:
        from intelligence import interaction
        interaction._register_rex_utterance(
            f"I found {label}.",
            source="exploration",
            topic=label,
            target_person_id=sess.person_id,
            expected_reply_types=["answer", "statement"],
        )
    except Exception as exc:
        _log.debug("[explore] topic seed failed: %s", exc)


def _record_episode(sess: "_Session", cand: dict) -> None:
    if sess.legs_done < 1 and bool(getattr(config, "EXPLORE_LOCOMOTION_ENABLED", True)) and base_available():
        # Only diary a real wander; a stationary head-only sweep isn't "I explored".
        return
    label = str(cand.get("name") or "something").strip()
    summary = f"I explored the room and got fixated on {label}."
    try:
        from intelligence import episodic_hooks
        episodic_hooks.exploration(summary, person_name=sess.person_name, person_id=sess.person_id)
    except Exception as exc:
        _log.debug("[explore] episodic capture failed: %s", exc)


# Labels that must never become a personal callback. The detector names whatever it
# sees, including PEOPLE and its own hedges, and every one of those got banked as a
# fact about the owner: the nine rows in the field DB were "has Bret Benziger in
# their space" (the owner, filed as his own furniture), "has Mysterious black object
# in their space", "has Plant in their space" AND "has Plant Display in their space"
# (the same plant twice). A callback is meant to be a personal detail worth raising
# later — "you have a thing near you" is not one.
_LABEL_REJECT_RE = re.compile(
    r"^(?:a|an|the)?\s*(?:mystery|mysterious|unknown|unidentified|unrecognized|"
    r"possible|probable|some(?:thing|)|indistinct|blurry|obscured|generic|"
    r"assorted|misc(?:ellaneous)?)\b",
    re.IGNORECASE,
)
# Too generic to be a callback even when the detector is sure.
_LABEL_TOO_GENERIC = frozenset({
    "object", "objects", "item", "items", "thing", "things", "stuff", "shape",
    "person", "people", "human", "man", "woman", "figure", "face", "background",
    "wall", "floor", "ceiling", "room", "light", "shadow", "surface",
})


def _label_bankable(label: str, person_name: "str | None") -> bool:
    low = " ".join(str(label or "").split()).lower()
    if not low or len(low) < 3:
        return False
    if _LABEL_REJECT_RE.match(low):
        return False
    if low in _LABEL_TOO_GENERIC:
        return False
    if all(w in _LABEL_TOO_GENERIC for w in low.split()):
        return False
    # A detected PERSON is not a possession. Reject the current human by name and
    # anyone Rex knows — "has Bret Benziger in their space" was a real stored row.
    if person_name and low == str(person_name).strip().lower():
        return False
    try:
        from memory import people as _people
        for known in (_people.list_person_names() or []):
            if low == str(known).strip().lower():
                return False
    except Exception:
        pass
    return True


def _bank_callback(sess: "_Session", cand: dict) -> None:
    if not bool(getattr(config, "EXPLORE_BANK_CALLBACK_ENABLED", True)):
        return
    if sess.person_id is None:
        return
    label = str(cand.get("name") or "").strip()
    if not label:
        return
    if not _label_bankable(label, sess.person_name):
        _log.info("[explore] not banking %r as a callback — not a personal detail", label)
        return
    try:
        from memory import callbacks
        callbacks.bank(
            sess.person_id,
            f"has {label} in their space",
            category="quirk",
            topic=label,
            sensitivity=callbacks.SENSITIVITY_SAFE,
            source_quote="",
            volunteered_playfully=False,
            # Rex SAW this; nobody told him. Storing it as 'explicit' put a camera
            # guess on the same footing as the person's own words.
            source="observed",
        )
    except Exception as exc:
        _log.debug("[explore] callback bank failed: %s", exc)


# ── Speech + generation helpers ───────────────────────────────────────────────


def _speak(
    sess: "_Session",
    text: str,
    *,
    emotion: str = "neutral",
    tag: str = "explore",
    register_frame: Optional[dict] = None,
) -> bool:
    """Speak one exploration line and block (bounded) for pacing.

    The mode OWNS the floor, so it enqueues directly (bypassing the proactive gate,
    which `active()` would otherwise trip against the mode's own speech). Yields to
    a user who is already talking. Registers a reply frame for the fixation question.
    Returns True only when the line was actually ENQUEUED — callers that persist
    state about a spoken beat (fixation, riff cadence) must key off this, never
    assume delivery.
    """
    text = (text or "").strip()
    if not text or sess.aborting():
        return False
    try:
        from audio import speech_queue, barge_guard
        if barge_guard.user_speaking_now():
            _log.info("[explore] user speaking — dropping line: %r", text)
            return False
        ev = speech_queue.enqueue(text, emotion, priority=1, tag=tag)
        sess.lines_spoken += 1
        # Transcript + speech-state coherence.
        try:
            from memory import conversations as conv_memory
            from utils import conv_log
            conv_memory.add_to_transcript("Rex", text)
        except Exception:
            pass
        if register_frame is not None:
            _register_fixation_frame(sess, text, register_frame)
        else:
            _note_line(text)
        # Pace: wait for playback to finish (bounded) so legs/riffs don't overlap.
        try:
            ev.wait(timeout=float(getattr(config, "EXPLORE_SPEAK_MAX_WAIT_SECS", 12.0)))
        except Exception:
            pass
        return True
    except Exception as exc:
        _log.debug("[explore] speak failed: %s", exc)
        return False


def _note_line(text: str) -> None:
    try:
        from intelligence import consciousness
        consciousness.note_rex_utterance(text, open_response_wait=False, source="exploration")
    except Exception:
        pass


def _register_fixation_frame(sess: "_Session", text: str, cand: dict) -> None:
    try:
        from intelligence import interaction
        interaction._register_rex_utterance(
            text,
            source="exploration",
            topic=str(cand.get("name") or ""),
            target_person_id=sess.person_id,
            expected_reply_types=["answer", "statement"],
        )
    except Exception as exc:
        _log.debug("[explore] fixation frame register failed: %s", exc)


def _generate(directive: str) -> str:
    """One in-persona line via the lean one-voice path (get_response → stream_directive)."""
    try:
        from intelligence.llm import get_response
        text = get_response(directive)
        return (text or "").strip()
    except Exception as exc:
        _log.debug("[explore] generate failed: %s", exc)
        return ""


def _person_safety_clause(cand: dict) -> str:
    """Hard rule appended when the subject is a PERSON (already gated to known
    adults): the humor never targets their body or appearance."""
    if str(cand.get("category") or "").lower() != "person":
        return ""
    return (
        " The subject is a PERSON: absolutely NO remarks about their body, face, "
        "age, or appearance — react warmly to what they're DOING, and aim any dig "
        "at yourself or the situation, never at them."
    )


def _riff_directive(sess: "_Session", cand: dict, *, boring: bool) -> str:
    hook = str(cand.get("riff_hook") or "").strip()
    name = str(cand.get("name") or "something").strip()
    hook_clause = f' The concrete detail to riff on: "{hook}".' if hook else ""
    tone = (
        "Under-react — a dry, mildly disappointed aside."
        if boring else
        "Delight FIRST, then a light dig. 'Ooooh, what do we have HERE' energy."
    )
    return (
        f"You are DJ-R3X rolling around a room, exploring. You just stopped and are "
        f"looking at: {name}.{hook_clause} React with ONE short whimsical, witty line "
        f"(<= 18 words). {tone} Do NOT ask a question here — you're just noticing it out "
        f"loud. Never invent objects you can't see; only riff on {name}."
        f"{_person_safety_clause(cand)} No stage directions, no asterisks."
    )


def _fixate_directive(sess: "_Session", cand: dict, *, ask: bool) -> str:
    hook = str(cand.get("riff_hook") or "").strip()
    name = str(cand.get("name") or "something").strip()
    hook_clause = f' The detail that grabbed you: "{hook}".' if hook else ""
    who = f" You're talking to {sess.person_name}." if sess.person_name else ""
    if ask:
        tail = (
            "Then ask ONE curious, playful question about it (to them if they're here)."
            " Two sentences MAX total."
        )
    else:
        tail = "One or two short sentences MAX. No question."
    return (
        f"You are DJ-R3X. While exploring the room you got FIXATED on: {name}.{hook_clause}"
        f"{who} React with real, whimsical delight (and a light dig if it fits) — this is "
        f"the thing you can't look away from. {tail} Never invent details you can't see; "
        f"only about {name}.{_person_safety_clause(cand)} No stage directions, no asterisks."
    )


# ── Small leaf utilities (lazy deps) ──────────────────────────────────────────


def _cand_key(cand: dict) -> str:
    return (str(cand.get("name") or "") + "|" + str(cand.get("category") or "")).lower().strip()


def _output_busy() -> bool:
    try:
        from audio import speech_queue, output_gate
        return bool(speech_queue.is_speaking() or output_gate.is_busy())
    except Exception:
        return False


def _visible_known_names() -> list:
    try:
        from vision import face
        return list(face.visible_known_names() or [])
    except Exception:
        return []


def _label_sightings(labels) -> dict:
    try:
        from memory import room_model
        return room_model.label_sightings(labels) or {}
    except Exception:
        return {}


def _record_room_labels(cands: list) -> None:
    try:
        from memory import room_model
        objs = [
            {"label": c.get("name"), "position": c.get("view") or "unknown"}
            for c in cands if c.get("name")
        ]
        if objs:
            room_model.record_objects(objs)
    except Exception:
        pass


def _parse_json(text: str):
    """Tolerant JSON extractor (raw → code-fence → outermost brace slice)."""
    if not text:
        return None
    import json
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        try:
            return json.loads(stripped)
        except Exception:
            pass
    for oc, cc in (("{", "}"), ("[", "]")):
        s = text.find(oc)
        e = text.rfind(cc)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return None
