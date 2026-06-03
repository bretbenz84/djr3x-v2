"""
intelligence/consciousness.py — Central consciousness loop for DJ-R3X.

Reads WorldState on a fixed interval and drives proactive behavior:
anger/mood maintenance, person recognition, follow-up detection,
disengagement recovery, proactive world reactions, idle micro-behaviors,
and continuous neck-servo face tracking.
"""

import json
import logging
import random
import re
import sys
import threading
import time
import inspect
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import state as state_module
from state import State
from world_state import world_state
from awareness.situation import assessor as _situation_assessor, SituationProfile
from intelligence import emotion_orchestrator
from intelligence import person_specials
from intelligence import profile_questions
from utils import conv_log

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_face_tracking_thread: Optional[threading.Thread] = None
_face_tracking_tracker = None
_process_started_iso: Optional[str] = None
_process_started_mono: float = 0.0

# Smoothed neck servo position in quarter-microseconds
_neck_smooth: float = float(config.SERVO_CHANNELS["neck"]["neutral"])
_face_tracking_suspended_until: float = 0.0
_face_tracking_lock: dict = {}
_last_face_tracking_log_at: float = 0.0
_face_tracking_last_error_key: Optional[str] = None
_face_tracking_last_error_x: Optional[float] = None
_face_tracking_last_error_y: Optional[float] = None
_face_tracking_last_error_at: float = 0.0

# WorldState snapshot from the previous loop iteration (for change detection)
_last_snapshot: dict = {}

# Notable dates acknowledged this session so we don't repeat them
_acknowledged_dates: set[str] = set()
_acknowledged_weather_signatures: set[str] = set()
_last_weather_reaction_at: float = 0.0

# Monotonic timestamp of the last idle micro-behavior
_last_micro_behavior_at: float = 0.0

# Cooldown: map person id-string → monotonic timestamp of last re-engagement attempt
_reengagement_sent_at: dict[str, float] = {}
_REENGAGEMENT_COOLDOWN_SECS = 30.0

# Monotonic timestamp of last live-vision commentary call (cost control).
_last_live_vision_comment_at: float = 0.0

# Visual curiosity asks: after a real back-and-forth goes quiet, Rex can take a
# fresh frame, summarize it, and ask one scene-grounded question.
_last_visual_curiosity_at: float = 0.0
_visual_curiosity_by_person: dict[int, float] = {}
_visual_curiosity_in_flight: bool = False
_visual_curiosity_lock = threading.Lock()

# Pending follow-up events per DB person_id: {db_id: [event_dict, ...]}
_pending_followups: dict[int, list[dict]] = {}
_followup_lock = threading.Lock()

# Pending identity prompt for unknown-person enrollment.
_pending_identity_prompt = threading.Event()
_identity_prompt_in_flight = threading.Event()
_last_identity_prompt_at: float = 0.0
_identity_prompt_reply_until: float = 0.0
_IDENTITY_PROMPT_COOLDOWN_SECS = 45.0

# Pending RELATIONSHIP prompt: Rex asked the engaged person who the stranger is.
# When set, the next user utterance should be parsed for {name, relationship}
# and, if found, the new face is enrolled and an edge saved.
_pending_relationship_prompt = threading.Event()
_pending_relationship_context: dict = {}  # {"engaged_person_id": int, "engaged_name": str, "slot_id": str, "asked_at": float}
_RELATIONSHIP_PROMPT_COOLDOWN_SECS = 45.0
_UNKNOWN_WITH_ENGAGED_CONFIRM_SECS = 5.0
# Per-session slot ids we've already asked about, so Rex doesn't re-ask.
_asked_relationship_slots: set[str] = set()
# Track first-seen time of each unknown slot (while any engaged conversation is open).
_unknown_first_seen_at: dict[str, float] = {}

# Conversation turn-taking guard: when Rex asks a question, proactive speech
# pauses briefly so people can answer without being talked over.
_response_wait_until: float = 0.0
_last_proactive_speech_at: float = 0.0
_last_rex_utterance_text: str = ""
_last_memory_hint_text: str = ""
_last_memory_hint_at: float = 0.0
_last_memory_hint_person_id: Optional[int] = None
_turn_lock = threading.Lock()
_proactive_speech_pending = threading.Event()

# Face detection terminal feedback de-duplication signature.
_last_face_feedback_signature: Optional[str] = None
_last_pose_analysis_at: float = 0.0
_previous_face_boxes: dict[str, tuple[int, int, int, int]] = {}
_last_face_seen_at: float = 0.0
_personal_space_reacted_at: dict[str, float] = {}

# Presence tracking: set of tracking keys visible in the previous loop tick.
# Key type: int (person_db_id) for known people, str (slot id e.g. "person_1") for unknown.
_visible_people: set = set()

# Known people (by person_db_id) already greeted since process start.
_greeted_this_session: set[int] = set()

# (person_db_id, event_id) pairs Rex has already anticipated this session,
# so the same upcoming event isn't referenced on every re-entry into frame.
_anticipated_events: set[tuple[int, int]] = set()

# Third-party awareness state.
# _third_party_seen_at: per-tracking-key monotonic timestamp of when the
#   person was first noticed as a non-engaged bystander this lurking spell.
# _third_party_called_out: per-tracking-key dedupe so a given lurker only
#   triggers one callout per session.
# _last_third_party_check_at: rate limit so the step only does real work
#   every THIRD_PARTY_CHECK_INTERVAL_SECS.
_third_party_seen_at: dict = {}
_third_party_called_out: set = set()
_last_third_party_check_at: float = 0.0

# Group turn-taking state.
# Tracks known visible people who have not spoken much while another known person
# has been carrying the conversation, so Rex can occasionally invite them in.
_group_turn_speaker_times: dict[int, deque[float]] = {}
_group_turn_visible_since: dict[int, float] = {}
_group_turn_invited_at: dict[int, float] = {}
_group_turn_invited_this_session: set[int] = set()
_last_group_turn_check_at: float = 0.0
_last_group_lull_check_at: float = 0.0
_group_lull_fired_at: dict[str, float] = {}

# Startup group greeting state. When two known people are visible together as
# the camera settles, Rex should greet the room once instead of stacking
# person-by-person memory callbacks.
_startup_group_signature: Optional[str] = None
_startup_group_seen_at: float = 0.0
_startup_group_greeted_signatures: set[str] = set()
_startup_solo_seen_at: float = 0.0
_startup_empty_room_seen_at: float = 0.0
_startup_empty_room_fired: bool = False
_startup_camera_first_frame_at: float = 0.0
_startup_presence_evidence_at: float = 0.0
_startup_presence_evidence_reason: str = ""

# Overheard chime-in tracking. Counts how many times Rex has chimed in on
# being-discussed mentions this session and rate-limits how often the step
# considers a chime-in.
_overheard_chime_in_count: int = 0
_last_overheard_check_at: float = 0.0
_last_overheard_mention_handled_at: Optional[float] = None

# Holiday-plans dedupe: (person_db_id, holiday_iso_date) tuples Rex has already
# asked about this session. The iso date includes the year, so the same holiday
# next year is fair game without a manual reset.
_holiday_plans_asked: set[tuple[int, str]] = set()
_last_holiday_plans_check_at: float = 0.0

# Weekly small-talk dedupe: (person_db_id, iso_year, iso_week, slot) tuples Rex
# has already asked. Slots: "weekend_plans" (Fri eve), "week_ahead" (Sun eve),
# "weekend_recap" (Mon morning).
_weekly_smalltalk_asked: set[tuple[int, int, int, str]] = set()
_last_weekly_smalltalk_check_at: float = 0.0

# Emotional check-in dedupe: per-session, per-person. Each engaged person
# can be the target of at most one proactive emotional check-in per session.
_emotional_checkin_fired: set[int] = set()
_emotional_checkin_fired_at: dict[int, float] = {}
# Per-person monotonic timestamp of when their cached affect first turned
# negative this session. Cleared whenever the cached affect goes non-negative.
# Used to gate "sustained negative" check-ins.
_negative_streak_started_at: dict[int, float] = {}
_last_emotional_checkin_check_at: float = 0.0

# Per-person mood cache: person_db_id → ({mood, confidence, notes}, monotonic_ts).
# Mood vision calls are expensive, so we re-use a recent reading within
# config.MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS instead of re-asking GPT-4o.
_mood_cache: dict[int, tuple[dict, float]] = {}
_gui_mood_refresh_in_flight: bool = False

# After Rex delivers a short joke/snark, watch the visible person's expression
# for a neutral/frown-to-smile shift and allow one tiny victory lap.
_smile_reaction_lock = threading.Lock()
_smile_reaction_watch: Optional[dict] = None
_last_smile_reaction_at: float = 0.0
_facial_expression_observed: dict[str, dict] = {}
_facial_expression_reacted_at: dict[tuple[str, str], float] = {}
_last_facial_expression_reaction_at: float = 0.0
_last_expression_reaction_line_by_kind: dict[str, str] = {}
_disposition_sampled_at: dict[int, float] = {}

# Per-person monotonic timestamp of when they were last seen in frame.
_last_seen: dict = {}

# Identity stickiness: when exactly one face is visible and recognition momentarily
# returns Unknown for what is almost certainly the same physical person we had
# identified a second ago, carry the last identity forward for this many seconds.
_last_solo_identity: Optional[tuple[int, str, float, tuple[float, float, float, float] | None]] = None
_SOLO_IDENTITY_STICKY_SECS = 5.0

# Per-person monotonic timestamp of the last departure/return reaction fired.
_last_departure_reaction_at: dict = {}
_last_return_reaction_at: dict = {}

# Ensures only one presence reaction fires at a time; acquire non-blocking to skip if busy.
_presence_reaction_lock = threading.Lock()

# Special-case celebrity greeting for Jeff Benziger from History Hunters.
_jeff_celebrity_greeted_this_session: set[int] = set()
_pending_jeff_celebrity_greetings: dict[int, dict] = {}

# Special-case celebrity greeting for JT / Jay Tee, volleyball legend.
_jt_volleyball_greeted_this_session: set[int] = set()
_pending_jt_volleyball_greetings: dict[int, dict] = {}

# Persons who have left frame but whose departure reaction hasn't fired yet.
# Maps tracking_key → (departure_monotonic, person_name_or_None).
# Departure reactions are delayed until situation.apparent_departure is True so that
# face-gone-but-still-talking (situation.likely_still_present) doesn't trigger a reaction.
_pending_departure_keys: dict = {}

# Unified per-person presence cooldown (ANY reaction type — departure or return).
# Takes precedence over per-type cooldowns to stop Rex from narrating every
# micro-absence of the same person.
_last_presence_reaction_at: dict = {}

# First-missing-at timestamp per tracking key. A person must be continuously
# missing for PRESENCE_DEPARTURE_CONFIRM_SECS before a departure is staged.
_first_missing_at: dict = {}

# Confirmed absent keys have passed the absence hysteresis. Return reactions are
# only eligible for these keys, which prevents recognition flicker from becoming
# "oh, you're back" banter.
_confirmed_absent_at: dict = {}

# First-sight greeting candidates must remain visible briefly before Rex speaks.
_first_sight_seen_at: dict = {}

# Animal arrival dedupe uses species/position signatures instead of unstable
# animal_1/animal_2 IDs returned by the vision prompt.
_animal_seen_signatures: set[str] = set()
_animal_reacted_at: dict[str, float] = {}
_pending_animal_arrivals: dict[str, dict] = {}
_last_startle_sound_reaction_at: float = 0.0

# Engagement tracking: the person_db_id Rex is currently talking with, if any.
# Presence reactions for this person are suppressed while the engagement is open.
_engaged_lock = threading.Lock()
_engaged_person_id: Optional[int] = None
_engaged_last_touch_at: float = 0.0
_recent_engaged_person_id: Optional[int] = None
_recent_engaged_touch_at: float = 0.0

# Speaker-gaze intent: recent speech asks the head to find/center the speaker.
_speaker_gaze_lock = threading.Lock()
_speaker_gaze_intent: dict = {}

# Learned vertical rest gaze. Active face tracking still owns the exact gaze
# baseline; this only changes where Rex settles/searches after a face disappears.
_adaptive_head_rest: dict = {
    "lift": int(config.SERVO_CHANNELS["headlift"]["neutral"]),
    "tilt": int(config.SERVO_CHANNELS["headtilt"]["neutral"]),
    "samples": 0,
    "updated_at": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Engagement API — called by interaction.py
# ─────────────────────────────────────────────────────────────────────────────

def mark_engagement(person_id: Optional[int]) -> None:
    """Record that Rex is actively conversing with person_id. Called on every
    identified speech segment. Resets the engagement window."""
    global _engaged_person_id, _engaged_last_touch_at
    global _recent_engaged_person_id, _recent_engaged_touch_at
    if person_id is None:
        return
    now = time.monotonic()
    with _engaged_lock:
        _engaged_person_id = person_id
        _engaged_last_touch_at = now
        _recent_engaged_person_id = person_id
        _recent_engaged_touch_at = now


def note_person_spoke(person_id: Optional[int]) -> None:
    """Record an identified speech turn for lightweight group turn-taking."""
    if person_id is None:
        return
    try:
        pid = int(person_id)
    except Exception:
        return
    now = time.monotonic()
    window = float(getattr(config, "GROUP_TURN_RECENT_WINDOW_SECS", 180.0))
    max_age = max(window, 60.0)
    turns = _group_turn_speaker_times.setdefault(pid, deque())
    turns.append(now)
    cutoff = now - max_age
    while turns and turns[0] < cutoff:
        turns.popleft()


def clear_engagement() -> None:
    """Clear active engagement state, but keep recent engagement for attribution."""
    global _engaged_person_id, _engaged_last_touch_at
    with _engaged_lock:
        _engaged_person_id = None
        _engaged_last_touch_at = 0.0


def is_engaged_with(person_id: Optional[int]) -> bool:
    """True if person_id is currently Rex's active conversational partner."""
    if person_id is None:
        return False
    window = getattr(config, "ENGAGEMENT_WINDOW_SECS", 90.0)
    with _engaged_lock:
        if _engaged_person_id != person_id:
            return False
        return (time.monotonic() - _engaged_last_touch_at) <= window


def get_recent_engagement(window_secs: Optional[float] = None) -> Optional[dict]:
    """
    Return the most recently engaged person within window_secs, even if the
    engagement technically ended (session cleared). Used by interaction.py to
    chain "who are you?" into "how do you know <engaged_name>?"

    Returns dict {person_id, name} or None.
    """
    if window_secs is None:
        window_secs = float(getattr(config, "RECENT_ENGAGEMENT_WINDOW_SECS", 60.0))
    with _engaged_lock:
        pid = _engaged_person_id if _engaged_person_id is not None else _recent_engaged_person_id
        touch = _engaged_last_touch_at if _engaged_person_id is not None else _recent_engaged_touch_at
    if pid is None or touch <= 0.0:
        return None
    if (time.monotonic() - touch) > window_secs:
        return None
    try:
        from memory import people as _people_mod
        row = _people_mod.get_person(pid)
        if row and row.get("name"):
            return {"person_id": pid, "name": row["name"]}
    except Exception:
        pass
    return {"person_id": pid, "name": None}


def _person_has_visible_face(person_id: Optional[int]) -> bool:
    if person_id is None:
        return False
    try:
        target = int(person_id)
    except Exception:
        return False
    try:
        for person in world_state.get("people") or []:
            if person.get("person_db_id") != target:
                continue
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
            return bool(person.get("face_box") or person.get("bounding_box") or person.get("bbox"))
    except Exception:
        return False
    return False


def _any_visible_face() -> bool:
    try:
        for person in world_state.get("people") or []:
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
            if person.get("face_box") or person.get("bounding_box") or person.get("bbox"):
                return True
    except Exception:
        return False
    return False


def _any_visible_unknown_face() -> bool:
    try:
        for person in world_state.get("people") or []:
            if person.get("person_db_id") is not None:
                continue
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
            if person.get("face_box") or person.get("bounding_box") or person.get("bbox"):
                return True
    except Exception:
        return False
    return False


def _note_startup_camera_frame(frame) -> None:
    """Record when startup first had an actual camera frame to reason from."""
    global _startup_camera_first_frame_at
    if frame is None or _startup_camera_first_frame_at > 0.0:
        return
    if not _within_startup_group_window():
        return
    _startup_camera_first_frame_at = time.monotonic()


def _note_startup_presence_evidence(reason: str) -> None:
    """Any face or speech during startup means Rex must not claim absence."""
    global _startup_presence_evidence_at, _startup_presence_evidence_reason
    if not _within_startup_group_window():
        return
    _startup_presence_evidence_at = time.monotonic()
    _startup_presence_evidence_reason = str(reason or "presence")


def _startup_presence_gate_ready(now: float) -> bool:
    """
    True only after startup has had a fair chance to look before making any
    room-empty-style comment. This is intentionally conservative: wide-angle
    face detection can prove presence, but it cannot prove absence.
    """
    if _process_started_mono <= 0.0:
        return False

    camera_ready_secs = float(
        getattr(config, "STARTUP_EMPTY_ROOM_CAMERA_READY_SECS", 2.0) or 0.0
    )
    if _startup_camera_first_frame_at <= 0.0:
        return False
    if (now - _startup_camera_first_frame_at) < max(0.0, camera_ready_secs):
        return False

    if bool(getattr(config, "STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE", True)):
        min_scan = float(getattr(config, "STARTUP_EMPTY_ROOM_MIN_SCAN_SECS", 9.5) or 0.0)
        if bool(getattr(config, "SPEAKER_GAZE_STARTUP_SCAN_ENABLED", True)):
            search_window = float(
                getattr(config, "SPEAKER_GAZE_SEARCH_WINDOW_SECS", 8.0) or 0.0
            )
            min_scan = max(min_scan, search_window + 0.5)
        if (now - _process_started_mono) < max(0.0, min_scan):
            return False

    evidence_window = float(
        getattr(config, "STARTUP_EMPTY_ROOM_RECENT_PRESENCE_EVIDENCE_SECS", 20.0)
        or 0.0
    )
    if (
        _startup_presence_evidence_at > 0.0
        and (now - _startup_presence_evidence_at) <= max(0.0, evidence_window)
    ):
        return False
    if _last_face_seen_at > 0.0 and (now - _last_face_seen_at) <= max(0.0, evidence_window):
        return False
    if is_identity_prompt_waiting_for_reply() or is_identity_prompt_in_flight():
        return False
    return True


def note_speaker_gaze_intent(
    person_id: Optional[int],
    *,
    unknown_voice: bool = False,
    reason: str = "speech",
    force_search: Optional[bool] = None,
) -> None:
    """Tell the gaze loop that recent speech should guide head target choice."""
    if not bool(getattr(config, "SPEAKER_GAZE_ENABLED", True)):
        return

    now = time.monotonic()
    if str(reason or "").lower() not in {"startup", "scan"}:
        _note_startup_presence_evidence(f"speaker_gaze:{reason or 'speech'}")
    try:
        pid = int(person_id) if person_id is not None else None
    except Exception:
        pid = None
    if pid is not None:
        visible = _person_has_visible_face(pid)
    elif unknown_voice:
        visible = _any_visible_unknown_face()
        # A voice that didn't resolve while a KNOWN face is right in front of the
        # camera is almost always that person, just mis-scored — don't abandon the
        # lock and thrash the head scanning for an off-camera speaker. Only the
        # explicit off-camera decision (or a forced startup scan) should search.
        if not visible and reason != "off_camera_unknown" and _any_visible_face():
            visible = True
    else:
        visible = _any_visible_face()
    search_requested = (not visible) if force_search is None else bool(force_search)

    with _speaker_gaze_lock:
        _speaker_gaze_intent.clear()
        _speaker_gaze_intent.update({
            "person_id": pid,
            "unknown_voice": bool(unknown_voice),
            "reason": str(reason or "speech"),
            "requested_at": now,
            "search_requested": search_requested,
            "search_started_at": now if search_requested else 0.0,
            "last_search_at": 0.0,
            "search_index": 0,
            "search_plan": None,
            "search_plan_index": 0,
            "acquired_at": 0.0,
        })
    _log.info(
        "[speaker_gaze] intent reason=%s person_id=%s unknown=%s visible=%s search=%s",
        reason,
        pid,
        bool(unknown_voice),
        visible,
        search_requested,
    )


def request_face_acquisition_scan(reason: str = "startup") -> None:
    """Request a short exploratory scan for faces, biased toward seated people."""
    note_speaker_gaze_intent(
        None,
        unknown_voice=False,
        reason=reason,
        force_search=True,
    )


def _speaker_gaze_current_intent(now: Optional[float] = None) -> Optional[dict]:
    if not bool(getattr(config, "SPEAKER_GAZE_ENABLED", True)):
        return None
    now = time.monotonic() if now is None else now
    intent_window = float(getattr(config, "SPEAKER_GAZE_INTENT_WINDOW_SECS", 14.0) or 0.0)
    search_window = float(getattr(config, "SPEAKER_GAZE_SEARCH_WINDOW_SECS", 8.0) or 0.0)
    with _speaker_gaze_lock:
        intent = dict(_speaker_gaze_intent)
    if not intent:
        return None
    requested_at = float(intent.get("requested_at") or 0.0)
    if requested_at <= 0.0:
        return None
    age = now - requested_at
    if bool(intent.get("search_requested")):
        if age <= max(0.0, search_window):
            return intent
        with _speaker_gaze_lock:
            _speaker_gaze_intent["search_requested"] = False
        intent["search_requested"] = False
    if age <= max(0.0, intent_window):
        return intent
    return None


def _speaker_gaze_note_acquired(candidate: dict) -> None:
    now = time.monotonic()
    with _speaker_gaze_lock:
        if not _speaker_gaze_intent:
            return
        _speaker_gaze_intent["search_requested"] = False
        _speaker_gaze_intent["search_index"] = 0
        _speaker_gaze_intent["search_plan"] = None
        _speaker_gaze_intent["search_plan_index"] = 0
        _speaker_gaze_intent["last_search_at"] = 0.0
        _speaker_gaze_intent["acquired_at"] = now


def unknown_visible_recently(window_secs: Optional[float] = None) -> bool:
    """
    True if an unknown face is visible now or was seen very recently.

    Interaction uses this as a small face-detection grace window so a visible
    newcomer who flickers out for one frame is not treated as off-camera.
    """
    if window_secs is None:
        window_secs = float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0))
    now = time.monotonic()
    try:
        people = world_state.get("people") or []
        for person in people:
            if person.get("person_db_id") is not None:
                continue
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
            if person.get("face_box") or person.get("bounding_box") or person.get("bbox"):
                return True
    except Exception:
        pass
    try:
        for key, seen_at in _last_seen.items():
            if isinstance(key, str) and (now - float(seen_at)) <= window_secs:
                return True
    except Exception:
        return False
    return False


def known_visible_recently_except(
    person_id: Optional[int],
    window_secs: Optional[float] = None,
) -> bool:
    """
    True if a known person other than person_id is visible now or was seen
    recently. Interaction uses this to avoid assigning a low-confidence voice to
    the engaged person when another known participant just flickered out.
    """
    if window_secs is None:
        window_secs = float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0))
    try:
        excluded = int(person_id) if person_id is not None else None
    except (TypeError, ValueError):
        excluded = None
    now = time.monotonic()
    try:
        people = world_state.get("people") or []
        for person in people:
            pid = person.get("person_db_id")
            if pid is None:
                continue
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            if excluded is None or pid_int != excluded:
                return True
    except Exception:
        pass
    try:
        for key, seen_at in _last_seen.items():
            if not isinstance(key, int):
                continue
            if excluded is not None and int(key) == excluded:
                continue
            if (now - float(seen_at)) <= window_secs:
                return True
    except Exception:
        return False
    return False


def person_visible_recently(
    person_id: Optional[int],
    window_secs: Optional[float] = None,
) -> bool:
    """True if a specific known person is visible now or was seen recently."""
    try:
        target = int(person_id)
    except (TypeError, ValueError):
        return False
    if window_secs is None:
        window_secs = float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0))
    now = time.monotonic()
    try:
        people = world_state.get("people") or []
        for person in people:
            try:
                if int(person.get("person_db_id")) == target:
                    return True
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    try:
        seen_at = _last_seen.get(target)
        return seen_at is not None and (now - float(seen_at)) <= window_secs
    except Exception:
        return False


def set_relationship_prompt_context(ctx: dict) -> None:
    """
    Externally open a relationship-prompt window. Used by interaction.py after
    enrolling a newcomer to request that their NEXT utterance be parsed as
    {relationship} relative to a previously-engaged person.
    """
    if not ctx:
        return
    _pending_relationship_context.clear()
    _pending_relationship_context.update(ctx)
    _pending_relationship_prompt.set()


# ─────────────────────────────────────────────────────────────────────────────
# Public follow-up API
# ─────────────────────────────────────────────────────────────────────────────

def set_pending_followup(person_id: int, event: dict) -> None:
    """Store a follow-up event so the next interaction loop opens with it."""
    with _followup_lock:
        _pending_followups.setdefault(person_id, []).append(event)


def get_pending_followup(person_id: int) -> Optional[list[dict]]:
    """
    Return and clear pending follow-up events for person_id, or None if absent.
    Called by the interaction loop before starting a conversation.
    """
    with _followup_lock:
        events = _pending_followups.pop(person_id, None)
    return events if events else None


def drop_pending_followups(person_id: int, event_ids: Optional[set[int]] = None) -> None:
    """Remove queued follow-ups that are no longer valid after a correction."""
    with _followup_lock:
        if event_ids is None:
            _pending_followups.pop(person_id, None)
            return
        events = _pending_followups.get(person_id, [])
        kept = [e for e in events if int(e.get("id") or -1) not in event_ids]
        if kept:
            _pending_followups[person_id] = kept
        else:
            _pending_followups.pop(person_id, None)


def consume_identity_prompt_request() -> bool:
    """
    Return True once when an unknown-person identity prompt was recently spoken.
    Interaction uses this to temporarily accept short bare-name replies.
    """
    global _identity_prompt_reply_until
    if _pending_identity_prompt.is_set():
        now = time.monotonic()
        if _identity_prompt_reply_until > 0.0 and now > _identity_prompt_reply_until:
            _pending_identity_prompt.clear()
            _identity_prompt_reply_until = 0.0
            _log.info("[identity_prompt] reply window expired before user speech")
            return False
        _pending_identity_prompt.clear()
        _identity_prompt_reply_until = 0.0
        _log.info("[identity_prompt] reply window consumed by user speech")
        return True
    return False


def clear_pending_identity_prompts(*, reason: str = "") -> bool:
    """Drop identity/relationship reply windows that no longer fit the live turn."""
    global _identity_prompt_reply_until
    changed = (
        _pending_identity_prompt.is_set()
        or _identity_prompt_in_flight.is_set()
        or _pending_relationship_prompt.is_set()
        or bool(_pending_relationship_context)
        or _identity_prompt_reply_until > 0.0
    )
    _pending_identity_prompt.clear()
    _identity_prompt_in_flight.clear()
    _identity_prompt_reply_until = 0.0
    _pending_relationship_prompt.clear()
    _pending_relationship_context.clear()
    if changed:
        _log.info("[identity_prompt] cleared pending identity prompts reason=%s", reason)
    return changed


def is_identity_prompt_in_flight() -> bool:
    """True while an unknown-person identity prompt is being queued/spoken."""
    return _identity_prompt_in_flight.is_set()


def is_identity_prompt_waiting_for_reply() -> bool:
    """True while Rex should wait for a name reply after asking who someone is."""
    global _identity_prompt_reply_until
    if _identity_prompt_in_flight.is_set():
        return True
    if not _pending_identity_prompt.is_set():
        return False
    now = time.monotonic()
    if _identity_prompt_reply_until <= 0.0 or now <= _identity_prompt_reply_until:
        return True
    _pending_identity_prompt.clear()
    _identity_prompt_reply_until = 0.0
    _log.info("[identity_prompt] reply window expired")
    return False


def consume_relationship_prompt_request() -> Optional[dict]:
    """
    If Rex recently asked the engaged person about an unknown stranger, return
    the context dict (engaged_person_id, engaged_name, slot_id, asked_at) once
    and clear the event. Returns None if no prompt is pending.
    """
    if _pending_relationship_prompt.is_set():
        _pending_relationship_prompt.clear()
        ctx = dict(_pending_relationship_context)
        _pending_relationship_context.clear()
        return ctx
    return None


def get_pending_relationship_context() -> Optional[dict]:
    """
    Return a copy of the current relationship-prompt context without consuming it.
    Used by boundary/topic logic that only needs to know what kind of prompt is
    active.
    """
    if _pending_relationship_prompt.is_set() and _pending_relationship_context:
        return dict(_pending_relationship_context)
    return None


def note_relationship_slot_handled(slot_id: str) -> None:
    """Called by interaction after it resolves (or gives up on) a slot so
    consciousness won't re-ask about the same unknown face in this session."""
    if slot_id:
        _asked_relationship_slots.add(slot_id)


def note_person_greeted_this_session(person_id: Optional[int]) -> None:
    """
    Mark a person as already greeted by an explicit interaction path.

    Introductions produce their own welcome line, so the first-sight presence
    loop should not immediately stack a second startup greeting for the same
    newly enrolled person.
    """
    if person_id is None:
        return
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return
    _greeted_this_session.add(pid)
    _last_presence_reaction_at[pid] = time.monotonic()


def begin_response_wait(window_secs: Optional[float] = None) -> None:
    """
    Extend the "waiting for user response" window.
    """
    global _response_wait_until
    wait_for = (
        config.QUESTION_RESPONSE_WAIT_SECS
        if window_secs is None
        else max(0.0, float(window_secs))
    )
    deadline = time.monotonic() + wait_for
    with _turn_lock:
        _response_wait_until = max(_response_wait_until, deadline)


def clear_response_wait() -> None:
    """Clear any active response-wait window."""
    global _response_wait_until
    with _turn_lock:
        _response_wait_until = 0.0


def is_waiting_for_response() -> bool:
    """Return True while Rex should pause proactive speech and wait for a reply."""
    with _turn_lock:
        return time.monotonic() < _response_wait_until


def _utterance_expects_reply(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return "?" in cleaned


_MEMORY_HINT_PAT = re.compile(
    r"\b(remember|told me|you said|plan|planned|schedule|trip|congrats|"
    r"how'?s|how is|how did|did .* go|survive|ready for)\b",
    re.IGNORECASE,
)


def note_rex_utterance(
    text: str,
    wait_secs: Optional[float] = None,
    *,
    open_response_wait: bool = True,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    target_person_id: Optional[int] = None,
    target_name: Optional[str] = None,
    expected_reply_types: Optional[list[str]] = None,
    blocked_actions: Optional[list[str]] = None,
) -> None:
    """
    Track when Rex last spoke and, if it was a question, open a reply window.
    """
    global _last_proactive_speech_at, _response_wait_until, _last_rex_utterance_text
    global _last_memory_hint_text, _last_memory_hint_at, _last_memory_hint_person_id
    now = time.monotonic()
    _last_rex_utterance_text = (text or "").strip()
    if _last_rex_utterance_text and _MEMORY_HINT_PAT.search(_last_rex_utterance_text):
        _last_memory_hint_text = _last_rex_utterance_text
        _last_memory_hint_at = now
        _last_memory_hint_person_id = None
    try:
        from intelligence import question_budget
        question_budget.note_rex_utterance(text)
    except Exception:
        pass
    try:
        from intelligence import repair_moves
        repair_moves.note_assistant_turn(text)
    except Exception:
        pass
    try:
        from intelligence import topic_thread
        topic_thread.note_assistant_turn(text)
    except Exception:
        pass
    try:
        from intelligence import dialogue_act
        dialogue_act.note_rex_turn(
            text,
            source=source,
            topic=topic,
            target_person_id=target_person_id,
            target_name=target_name,
            expected_reply_types=expected_reply_types,
            blocked_actions=blocked_actions,
        )
    except Exception:
        pass

    with _turn_lock:
        _last_proactive_speech_at = now

        should_wait = open_response_wait and (
            wait_secs is not None or _utterance_expects_reply(text)
        )
        if not should_wait:
            return

        wait_for = (
            config.QUESTION_RESPONSE_WAIT_SECS
            if wait_secs is None
            else max(0.0, float(wait_secs))
        )
        _response_wait_until = max(_response_wait_until, now + wait_for)


def _question_key_for_presence_line(label: str, purpose: str) -> Optional[str]:
    label_l = (label or "").lower()
    if "first-sight greeting" in label_l or "startup group greeting" in label_l:
        return "startup_conversation_steering"
    if purpose == "small_talk":
        return "proactive_small_talk"
    return None


def _presence_line_counts_as_greeting(label: str, purpose: str) -> bool:
    del purpose
    label_l = (label or "").lower()
    return (
        "first-sight" in label_l
        or "startup" in label_l
        or label_l.startswith("return ")
        or " return " in label_l
    )


def _record_proactive_question(
    person_id: Optional[int],
    text: str,
    *,
    label: str,
    purpose: str,
    question_key: Optional[str] = None,
    question_depth: int = 1,
) -> None:
    if person_id is None or "?" not in (text or ""):
        return
    key = question_key or _question_key_for_presence_line(label, purpose)
    if not key:
        return
    try:
        from memory import relationships as rel_memory
        rel_memory.save_question_asked(
            int(person_id),
            key,
            text.strip(),
            int(question_depth or 1),
        )
        _log.info(
            "consciousness: recorded proactive question key=%s person_id=%s label=%s",
            key,
            person_id,
            label,
        )
    except Exception as exc:
        _log.debug("proactive question record failed: %s", exc)


def get_last_rex_utterance() -> str:
    """Return the latest Rex line observed by consciousness/interaction."""
    return _last_rex_utterance_text


def get_last_memory_hint(max_age_secs: float = 300.0) -> str:
    """Return Rex's latest memory-callback-looking line within a short TTL."""
    if not _last_memory_hint_text:
        return ""
    if (time.monotonic() - _last_memory_hint_at) > max(0.0, float(max_age_secs)):
        return ""
    return _last_memory_hint_text


def note_memory_hint(text: str, person_id: Optional[int]) -> None:
    """Remember a just-spoken memory callback and whom it was addressed to."""
    global _last_memory_hint_text, _last_memory_hint_at, _last_memory_hint_person_id
    if not text or person_id is None:
        return
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return
    _last_memory_hint_text = text.strip()
    _last_memory_hint_at = time.monotonic()
    _last_memory_hint_person_id = pid


def get_last_memory_hint_target(max_age_secs: float = 300.0) -> Optional[int]:
    """Return the person id for the latest remembered memory callback."""
    if not get_last_memory_hint(max_age_secs=max_age_secs):
        return None
    return _last_memory_hint_person_id


def _end_thread_grace_active() -> bool:
    try:
        from intelligence import end_thread
        return bool(end_thread.is_grace_active())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_speak() -> bool:
    return state_module.get_state() not in (State.QUIET, State.SLEEP, State.SHUTDOWN)


def _can_proactive_speak() -> bool:
    if not _can_speak():
        return False

    try:
        from features import dj as dj_mod
        if (
            bool(getattr(config, "DJ_SUPPRESS_CONVERSATION_DURING_PLAYBACK", True))
            and dj_mod.is_playing()
        ):
            return False
    except Exception:
        pass

    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            if games_mod.suppresses_conversation_interruptions():
                return False
        elif games_mod.is_active():
            return False
    except Exception:
        pass

    current_state = state_module.get_state()
    if (
        current_state == State.ACTIVE
        and not getattr(config, "CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE", False)
    ):
        return False

    if is_waiting_for_response():
        return False
    if _proactive_speech_pending.is_set():
        return False
    try:
        if _situation_assessor.is_interaction_busy():
            return False
    except Exception:
        pass

    with _turn_lock:
        last_spoken = _last_proactive_speech_at
    min_gap = max(0.0, float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0)))
    if min_gap and (time.monotonic() - last_spoken) < min_gap:
        return False

    try:
        from audio import speech_queue, output_gate
        if speech_queue.is_speaking() or output_gate.is_busy():
            return False
    except Exception:
        return False
    return True


def _speak_async(
    text: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
    purpose: Optional[str] = None,
    label: str = "",
    governed: bool = True,
    on_done: Optional[Callable[[], None]] = None,
) -> bool:
    candidate_id = None
    if governed:
        candidate_id = _observe_governor_candidate(
            purpose=purpose or "direct_speech",
            label=label,
            suggested_text=text,
            emotion=emotion,
            wait_secs=wait_secs,
            requires_llm=False,
        )
    try:
        if not _can_proactive_speak():
            _mark_governor_candidate(candidate_id, "dropped", "can_proactive_speak_false")
            return False
        if not text or not text.strip():
            _mark_governor_candidate(candidate_id, "dropped", "empty_text")
            return False
        # Yield the floor if the user has already started talking. This line was
        # decided + generated before now; pre-cache its audio so the mic re-check
        # lands right before playback (not ~1s before it, the window in which Rex
        # used to start talking over a reply that began during TTS generation),
        # then bail if the user beat us to it — the interaction turn loop will
        # pick them up from the un-attenuated rolling buffer.
        if bool(getattr(config, "PROACTIVE_SPEECH_YIELD_ENABLED", True)):
            try:
                from audio import tts
                tts.ensure_cached(text, emotion=emotion)
            except Exception as exc:
                _log.debug("proactive pre-cache failed: %s", exc)
            try:
                from audio import barge_guard
                if barge_guard.user_speaking_now():
                    _mark_governor_candidate(candidate_id, "dropped", "user_speaking")
                    _log.info(
                        "[consciousness] proactive line yielded — user already speaking: %r",
                        text,
                    )
                    return False
            except Exception as exc:
                _log.debug("proactive yield check failed: %s", exc)
        from audio import speech_queue
        _proactive_speech_pending.set()
        done = speech_queue.enqueue(text, emotion, priority=0)
        _mark_governor_candidate(candidate_id, "accepted", "current_behavior_enqueued_speech")
        should_open_wait_on_done = (
            on_done is None and (wait_secs is not None or _utterance_expects_reply(text))
        )

        def _on_done() -> None:
            done.wait()
            try:
                if on_done is not None:
                    on_done()
                elif should_open_wait_on_done:
                    begin_response_wait(wait_secs)
            finally:
                _proactive_speech_pending.clear()

        threading.Thread(target=_on_done, daemon=True, name="speech-pending-clear").start()
        try:
            conv_log.log_rex(text)
        except Exception as exc:
            _log.debug("conversation log write failed for proactive speech: %s", exc)
        note_rex_utterance(
            text,
            wait_secs=wait_secs,
            open_response_wait=False,
            source=purpose,
        )
        return True
    except Exception as exc:
        _mark_governor_candidate(candidate_id, "dropped", "speak_async_error")
        _proactive_speech_pending.clear()
        _log.debug("_speak_async error: %s", exc)
        return False


_SMILE_REACTION_LINES = (
    "Oh look, I made the lifeform smile. Guess my purpose in life has succeeded.",
    "There it is. A smile. My work here is alarmingly complete.",
    "Aha. Smile detected. I will notify the committee that my joke achieved lift.",
    "Look at that, facial amusement. I am basically a public service.",
    "Was that a smile? Incredible. I will try not to let this tiny victory ruin me.",
)
_SMILE_LABELS = {
    "smile",
    "smiling",
    "happy",
    "joy",
    "joyful",
    "amused",
    "laugh",
    "laughing",
}
_SMILE_REACTION_EXCLUDE_MARKERS = (
    "made the lifeform smile",
    "smile detected",
    "facial amusement",
    "my work here is alarmingly complete",
    "tiny victory ruin me",
)
_SMILE_REACTION_SNARK_MARKERS = (
    "alarmingly",
    "behold",
    "bold choice",
    "carbon",
    "committee",
    "congratulations",
    "droid",
    "flawless",
    "great, another",
    "hilarious",
    "incredible",
    "joke",
    "lifeform",
    "meat",
    "organic",
    "organics",
    "photoreceptors",
    "processing",
    "recalibrating",
    "roast",
    "snark",
    "spectacular",
    "tragic",
)
_SMILE_REACTION_SERIOUS_MARKERS = (
    "sorry",
    "condolence",
    "grief",
    "hurt",
    "sick",
    "hospital",
    "emergency",
    "panic",
    "terrified",
    "afraid",
    "depressed",
    "anxious",
    "trauma",
)
_FACIAL_EXPRESSION_REACTION_LINES = {
    "smile": (
        "There it is. A smile. I knew the diagnostics would eventually find joy.",
        "Smile detected. Careful, optimism is how droids get assigned extra duties.",
        "Ah, the lifeform is smiling. Marking this under rare but encouraging anomalies.",
        "Look at that, actual visible morale. I will pretend I had nothing to do with it.",
    ),
    "surprise": (
        "That was a full photoreceptor-wide shock face. What did the galaxy do now?",
        "You just looked like the hyperdrive coughed up a receipt. What happened?",
        "Wide eyes detected. Was it my charm, or did reality file another complaint?",
        "That expression says someone moved your starship. Care to brief the droid?",
        "Shock face logged. Did I say something brilliant, or did the universe get rude?",
    ),
    "frown": (
        "That frown has its own gravity well. Want to vent before it starts charging rent?",
        "You look displeased. If it helps, I also disapprove of most things.",
        "Downturned mouth detected. Organic morale appears to be under warranty review.",
        "That expression is doing sad trombone without the trombone. What's up?",
        "Your face just filed a complaint. Need a soundtrack, or a target?",
    ),
    "brow_furrow": (
        "That is a serious thinking face. Either a breakthrough, or math has betrayed you.",
        "Eyebrow committee detected. They appear focused and underfunded.",
        "I see the concentration squint. Want a droid to blame, or are we staying productive?",
        "That forehead is running extra diagnostics. Need a sounding board?",
        "Deep thought detected. I will lower the alarm level from doom to paperwork.",
    ),
}
_FACIAL_EXPRESSION_REACTION_LABELS = {
    "smile": {"smile", "smiling", "happy", "grin", "grinning"},
    "surprise": {"surprise", "surprised", "wide_eyes", "wide_eyed", "shocked"},
    "frown": {"frown", "sad", "downturned_mouth", "unhappy"},
    "brow_furrow": {"brow_furrow", "focused", "angry", "furrowed_brow", "irritated"},
}
_DISPOSITION_FIRST_SIGHT_LINES = {
    "smiley": (
        "Oh look, {first_name}, the perpetually smiling life-form has graced us with the pearly whites again.",
        "{first_name}, there you are. Still smiling like optimism owes you credits.",
        "Careful, {first_name}. Keep smiling like that and this room may accidentally improve.",
        "Ah, {first_name}. The resident morale leak returns, grinning all over the equipment.",
        "{first_name}, your smile stats remain suspiciously high. I have alerted absolutely no authorities.",
    ),
    "grumpy": (
        "Ah, {first_name}. The resident storm cloud returns. Try not to lower the barometric pressure.",
        "{first_name}, there you are, calibrating the room to mildly displeased.",
        "Look who it is: {first_name}, bringing the traditional frown garnish.",
        "{first_name}, that face says the galaxy failed inspection again. Relatable.",
        "Welcome back, {first_name}. I see your expression has entered classic complaint mode.",
    ),
    "deadpan": (
        "Ah, {first_name}. The neutral-expression champion returns. Thrilling emotional weather as always.",
        "{first_name}, your face remains statistically unreadable. Honestly, impressive brand consistency.",
        "Welcome back, {first_name}. Another flawless deployment of the default organic expression.",
        "{first_name}, I see the emotional readout is set to beige again.",
        "There you are, {first_name}. Stoic as a cargo manifest and twice as mysterious.",
    ),
    "intense": (
        "There is {first_name}, arriving with the eyebrows already in attack formation.",
        "{first_name}, that concentrated look could make a circuit board confess.",
        "Ah, {first_name}. The tactical squint has entered the building.",
        "{first_name}, your eyebrows appear to be holding a disciplinary hearing.",
        "Welcome back, {first_name}. I see the forehead committee is already in session.",
    ),
    "startled": (
        "Ah, {first_name}, champion of looking like the plot just turned.",
        "{first_name}, you do bring a reliable sense of startled cinema to the room.",
        "There you are, {first_name}. Already prepared for a surprise inspection by reality.",
        "{first_name}, your expression history says the galaxy keeps jump-scaring you. Bold lifestyle.",
        "Welcome back, {first_name}. I will try not to startle the professional startled person.",
    ),
}


def _norm_expression_label(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _safe_confidence(value, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _animal_is_furry_companion(species: str, animal: Optional[dict] = None) -> bool:
    species_key = str(species or "").strip().lower()
    if not species_key:
        return False
    if isinstance(animal, dict) and animal.get("furred") is True:
        return True
    configured = getattr(config, "FURRY_COMPANION_ANIMAL_SPECIES", set()) or set()
    return any(str(token).strip().lower() in species_key for token in configured)


_JEFF_BENZIGER_CANONICAL = "jeff benziger"
_JEFF_HISTORY_HUNTERS_LINES = (
    "Alert: Jeff from History Hunters detected. Somebody secure the historical markers.",
    "Oh no. Jeff Benziger is here. Now I'm going to end up in an episode called The Forgotten Droid of Batuu.",
    "History Hunters confirmed. Probability of hearing 'this used to be a thriving town' has risen to critical levels.",
    "Attention guests: Jeff from History Hunters has entered the sector. All forgotten cemeteries are advised to remain calm.",
    "Fantastic. Jeff Benziger is here. Everybody act natural or you'll end up on YouTube by Thursday.",
    "History Hunters detected. Quick, hide the old maps before he turns this cantina into a forty-seven minute documentary.",
)
_JEFF_HISTORY_HUNTERS_RETURN_LINES = (
    "Jeff from History Hunters is back. Everybody look casually archival.",
    "History Hunters has returned. I give it ten minutes before there's drone footage.",
)


def _normalize_person_name_for_specials(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _is_jeff_benziger(name: object) -> bool:
    return _normalize_person_name_for_specials(name) == _JEFF_BENZIGER_CANONICAL


def _jeff_history_hunters_line(*, returning: bool = False) -> str:
    lines = _JEFF_HISTORY_HUNTERS_RETURN_LINES if returning else _JEFF_HISTORY_HUNTERS_LINES
    return random.choice(lines)


def _can_jeff_celebrity_speak(profile: SituationProfile) -> bool:
    if not _can_speak():
        return False
    if profile.user_mid_sentence:
        return False
    try:
        from features import dj as dj_mod
        if (
            bool(getattr(config, "DJ_SUPPRESS_CONVERSATION_DURING_PLAYBACK", True))
            and dj_mod.is_playing()
        ):
            return False
    except Exception:
        pass
    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            if games_mod.suppresses_conversation_interruptions():
                return False
        elif games_mod.is_active():
            return False
    except Exception:
        pass
    try:
        from audio import speech_queue, output_gate
        if speech_queue.is_speaking() or output_gate.is_busy():
            return False
    except Exception:
        return False
    return not _proactive_speech_pending.is_set()


def _stage_jeff_history_hunters_greeting(
    *,
    key: int,
    person_name: str,
    returning: bool = False,
) -> None:
    if not returning and key in _jeff_celebrity_greeted_this_session:
        return
    existing = _pending_jeff_celebrity_greetings.get(key)
    if existing:
        existing["last_seen_at"] = time.monotonic()
        existing["returning"] = bool(existing.get("returning") or returning)
        return
    _pending_jeff_celebrity_greetings[key] = {
        "person_name": person_name,
        "returning": bool(returning),
        "first_seen_at": time.monotonic(),
        "last_seen_at": time.monotonic(),
    }
    _log.info(
        "consciousness: Jeff Benziger celebrity greeting staged (returning=%s)",
        bool(returning),
    )


def _try_fire_jeff_history_hunters_greeting(
    *,
    key,
    person_name: Optional[str],
    person_db_id: Optional[int],
    profile: SituationProfile,
    returning: bool = False,
) -> bool:
    if not isinstance(key, int) or not _is_jeff_benziger(person_name):
        return False
    if not returning and key in _jeff_celebrity_greeted_this_session:
        return False
    if not _can_jeff_celebrity_speak(profile):
        return False
    label = (
        "return celebrity greeting for Jeff Benziger"
        if returning
        else "first-sight celebrity greeting for Jeff Benziger"
    )
    text = _jeff_history_hunters_line(returning=returning)
    candidate_id = _observe_governor_candidate(
        purpose="presence_reaction",
        label=label,
        suggested_text=text,
        emotion="starstruck",
        priority=2,
        target_person_id=key,
        requires_llm=False,
    )
    if not _presence_reaction_lock.acquire(blocking=False):
        _mark_governor_candidate(candidate_id, "dropped", "presence_reaction_lock_busy")
        return False

    try:
        from audio import speech_queue

        _proactive_speech_pending.set()
        speech_queue.clear_below_priority(2)
        tag = f"presence:jeff_history_hunters:{key}"
        _last_presence_reaction_at[key] = time.monotonic()
        _log.info("consciousness: firing Jeff Benziger celebrity greeting: %r", text)
        done = speech_queue.enqueue(text, "starstruck", priority=2, tag=tag)
        _mark_governor_candidate(candidate_id, "accepted", "jeff_celebrity_enqueued")
        try:
            from memory import people as people_mod
            people_mod.record_greeting(key)
        except Exception as exc:
            _log.debug("record greeting failed for Jeff person_id=%s: %s", key, exc)
        try:
            conv_log.log_rex(text)
        except Exception as exc:
            _log.debug("conversation log write failed for Jeff greeting: %s", exc)
        note_rex_utterance(
            text,
            open_response_wait=False,
            source="presence_reaction",
            topic=label,
            target_person_id=key,
        )
        _jeff_celebrity_greeted_this_session.add(key)
        _greeted_this_session.add(key)
        _first_sight_seen_at.pop(key, None)
        _pending_jeff_celebrity_greetings.pop(key, None)

        def _clear_pending_flag() -> None:
            done.wait()
            _proactive_speech_pending.clear()
            try:
                _presence_reaction_lock.release()
            except RuntimeError:
                pass

        threading.Thread(
            target=_clear_pending_flag,
            daemon=True,
            name="jeff-celebrity-presence-done",
        ).start()
        return True
    except Exception as exc:
        _mark_governor_candidate(candidate_id, "dropped", "jeff_celebrity_error")
        _proactive_speech_pending.clear()
        try:
            _presence_reaction_lock.release()
        except RuntimeError:
            pass
        _log.debug("Jeff celebrity greeting failed: %s", exc)
        return False


def _step_jeff_history_hunters_detection(snapshot: dict, profile: SituationProfile) -> bool:
    """
    Jeff's recognition bit gets the first conversational claim once his known
    face is visible, even if he was enrolled earlier in the same run.
    """
    now = time.monotonic()
    confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
    visible_jeff: Optional[tuple[int, str]] = None
    for person in snapshot.get("people", []) or []:
        person_name = person.get("face_id") or person.get("voice_id") or ""
        if not _is_jeff_benziger(person_name):
            continue
        try:
            key = int(person.get("person_db_id"))
        except (TypeError, ValueError):
            continue
        if key in _jeff_celebrity_greeted_this_session:
            continue
        first_visible = _first_sight_seen_at.setdefault(key, now)
        if (now - first_visible) < max(0.0, confirm_visible):
            return True
        _stage_jeff_history_hunters_greeting(
            key=key,
            person_name=person_name,
        )
        visible_jeff = (key, person_name)
        break

    for pending_key, pending in list(_pending_jeff_celebrity_greetings.items()):
        name = str(pending.get("person_name") or _JEFF_BENZIGER_CANONICAL)
        if not _is_jeff_benziger(name):
            _pending_jeff_celebrity_greetings.pop(pending_key, None)
            continue
        stale_after = float(getattr(config, "JEFF_CELEBRITY_GREETING_PENDING_SECS", 45.0) or 45.0)
        if (now - float(pending.get("last_seen_at") or now)) > max(1.0, stale_after):
            _pending_jeff_celebrity_greetings.pop(pending_key, None)
            continue
        _try_fire_jeff_history_hunters_greeting(
            key=pending_key,
            person_name=name,
            person_db_id=pending_key,
            profile=profile,
            returning=bool(pending.get("returning")),
        )
        return True
    return visible_jeff is not None


def _is_jt_volleyball_celebrity(name: object) -> bool:
    return person_specials.is_jt_volleyball_celebrity(name)


def _can_jt_volleyball_speak(profile: SituationProfile) -> bool:
    return _can_jeff_celebrity_speak(profile)


def _stage_jt_volleyball_greeting(
    *,
    key: int,
    person_name: str,
    returning: bool = False,
) -> None:
    if not returning and key in _jt_volleyball_greeted_this_session:
        return
    existing = _pending_jt_volleyball_greetings.get(key)
    if existing:
        existing["last_seen_at"] = time.monotonic()
        existing["returning"] = bool(existing.get("returning") or returning)
        return
    _pending_jt_volleyball_greetings[key] = {
        "person_name": person_name,
        "returning": bool(returning),
        "first_seen_at": time.monotonic(),
        "last_seen_at": time.monotonic(),
    }
    _log.info(
        "consciousness: JT volleyball celebrity greeting staged (returning=%s)",
        bool(returning),
    )


def _try_fire_jt_volleyball_greeting(
    *,
    key,
    person_name: Optional[str],
    person_db_id: Optional[int],
    profile: SituationProfile,
    returning: bool = False,
) -> bool:
    if not isinstance(key, int) or not _is_jt_volleyball_celebrity(person_name):
        return False
    if not returning and key in _jt_volleyball_greeted_this_session:
        return False
    if not _can_jt_volleyball_speak(profile):
        return False
    label = (
        "return celebrity greeting for JT volleyball"
        if returning
        else "first-sight celebrity greeting for JT volleyball"
    )
    text = person_specials.jt_volleyball_line(returning=returning)
    candidate_id = _observe_governor_candidate(
        purpose="presence_reaction",
        label=label,
        suggested_text=text,
        emotion="starstruck",
        priority=2,
        target_person_id=key,
        requires_llm=False,
    )
    if not _presence_reaction_lock.acquire(blocking=False):
        _mark_governor_candidate(candidate_id, "dropped", "presence_reaction_lock_busy")
        return False

    try:
        from audio import speech_queue

        _proactive_speech_pending.set()
        speech_queue.clear_below_priority(2)
        tag = f"presence:jt_volleyball:{key}"
        _last_presence_reaction_at[key] = time.monotonic()
        _log.info("consciousness: firing JT volleyball celebrity greeting: %r", text)
        done = speech_queue.enqueue(text, "starstruck", priority=2, tag=tag)
        _mark_governor_candidate(candidate_id, "accepted", "jt_volleyball_enqueued")
        try:
            from memory import people as people_mod
            people_mod.record_greeting(key)
        except Exception as exc:
            _log.debug("record greeting failed for JT person_id=%s: %s", key, exc)
        try:
            conv_log.log_rex(text)
        except Exception as exc:
            _log.debug("conversation log write failed for JT greeting: %s", exc)
        note_rex_utterance(
            text,
            open_response_wait=False,
            source="presence_reaction",
            topic=label,
            target_person_id=key,
        )
        _jt_volleyball_greeted_this_session.add(key)
        _greeted_this_session.add(key)
        _first_sight_seen_at.pop(key, None)
        _pending_jt_volleyball_greetings.pop(key, None)

        def _clear_pending_flag() -> None:
            done.wait()
            _proactive_speech_pending.clear()
            try:
                _presence_reaction_lock.release()
            except RuntimeError:
                pass

        threading.Thread(
            target=_clear_pending_flag,
            daemon=True,
            name="jt-volleyball-presence-done",
        ).start()
        return True
    except Exception as exc:
        _mark_governor_candidate(candidate_id, "dropped", "jt_volleyball_error")
        _proactive_speech_pending.clear()
        try:
            _presence_reaction_lock.release()
        except RuntimeError:
            pass
        _log.debug("JT volleyball greeting failed: %s", exc)
        return False


def _step_jt_volleyball_detection(snapshot: dict, profile: SituationProfile) -> bool:
    """
    JT's recognition bit mirrors the Jeff celebrity override for volleyball lore.
    """
    now = time.monotonic()
    confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
    visible_jt: Optional[tuple[int, str]] = None
    for person in snapshot.get("people", []) or []:
        person_name = person.get("face_id") or person.get("voice_id") or ""
        if not _is_jt_volleyball_celebrity(person_name):
            continue
        try:
            key = int(person.get("person_db_id"))
        except (TypeError, ValueError):
            continue
        if key in _jt_volleyball_greeted_this_session:
            continue
        first_visible = _first_sight_seen_at.setdefault(key, now)
        if (now - first_visible) < max(0.0, confirm_visible):
            return True
        _stage_jt_volleyball_greeting(
            key=key,
            person_name=person_name,
        )
        visible_jt = (key, person_name)
        break

    for pending_key, pending in list(_pending_jt_volleyball_greetings.items()):
        name = str(pending.get("person_name") or "JT")
        if not _is_jt_volleyball_celebrity(name):
            _pending_jt_volleyball_greetings.pop(pending_key, None)
            continue
        stale_after = float(getattr(config, "JEFF_CELEBRITY_GREETING_PENDING_SECS", 45.0) or 45.0)
        if (now - float(pending.get("last_seen_at") or now)) > max(1.0, stale_after):
            _pending_jt_volleyball_greetings.pop(pending_key, None)
            continue
        _try_fire_jt_volleyball_greeting(
            key=pending_key,
            person_name=name,
            person_db_id=pending_key,
            profile=profile,
            returning=bool(pending.get("returning")),
        )
        return True
    return visible_jt is not None


def _prime_emotion_frame(frame) -> None:
    emotion_orchestrator.publish_frame(frame)
    if frame.body_beat:
        try:
            from sequences import animations
            animations.play_body_beat(frame.body_beat)
        except Exception as exc:
            _log.debug("emotion body beat skipped: %s", exc)


def _animal_signature(animal: dict) -> str:
    species = (animal.get("species") or "creature").strip().lower()
    position = (animal.get("position") or "unknown").strip().lower()
    return f"{species}:{position}"


def _stage_animal_arrivals(snapshot: dict) -> None:
    """Remember animal arrivals even when startup greetings temporarily own speech."""
    if not _last_snapshot:
        return
    animal_cooldown = float(getattr(config, "ANIMAL_ARRIVAL_COOLDOWN_SECS", 300.0))
    prev_animal_signatures = {
        _animal_signature(a)
        for a in _last_snapshot.get("animals", [])
        if isinstance(a, dict) and a.get("species")
    }
    now = time.monotonic()
    for animal in snapshot.get("animals", []) or []:
        if not isinstance(animal, dict) or not animal.get("species"):
            continue
        signature = _animal_signature(animal)
        _animal_seen_signatures.add(signature)
        if signature in _pending_animal_arrivals:
            _pending_animal_arrivals[signature]["last_seen_at"] = now
            continue
        if signature in prev_animal_signatures:
            continue
        if (now - _animal_reacted_at.get(signature, 0.0)) < animal_cooldown:
            continue
        pending = dict(animal)
        pending["signature"] = signature
        pending["first_seen_at"] = now
        pending["last_seen_at"] = now
        _pending_animal_arrivals[signature] = pending
        _log.info("consciousness: staged animal arrival signature=%s", signature)


_FURRY_ANIMAL_REACTION_LINES = (
    "Whoa. Small furry lifeform in the operational zone.",
    "Hold everything. A small furry lifeform has breached containment.",
    "Oh good. A small furry lifeform has entered the system.",
    "Well, hello, small furry lifeform. Try not to unionize.",
)

_STARTLING_ANIMAL_REACTION_LINES = (
    "Yah! New lifeform. I was emotionally prepared for none of that.",
    "Tiny startle event registered. Very dignified. Moving on.",
    "Nope. Surprise creature detected. Systems pretending to be calm.",
)

_GENERIC_ANIMAL_REACTION_LINES = (
    "New lifeform detected. The room just got more interesting.",
    "Ah, an unscheduled creature cameo. Naturally.",
    "Organic inventory update: additional lifeform present.",
)


def _animal_reaction_frame_and_line(animal: dict):
    species = (animal.get("species") or "creature").strip().lower()
    if emotion_orchestrator.is_startling_animal(species):
        frame = emotion_orchestrator.frame_for_event("animal_detected", species=species)
        return frame, random.choice(_STARTLING_ANIMAL_REACTION_LINES)
    if _animal_is_furry_companion(species, animal):
        frame = emotion_orchestrator.frame_for_emotion(
            "surprised",
            intensity=0.86,
            source="event",
            trigger=f"animal_arrival:{species}",
        )
        return frame, random.choice(_FURRY_ANIMAL_REACTION_LINES)
    frame = emotion_orchestrator.frame_for_event("animal_detected", species=species)
    return frame, random.choice(_GENERIC_ANIMAL_REACTION_LINES)


def _fire_pending_animal_arrival_reaction() -> bool:
    if not _pending_animal_arrivals:
        return False
    now = time.monotonic()
    stale_after = float(getattr(config, "ANIMAL_PENDING_REACTION_TTL_SECS", 90.0))
    for signature, animal in list(_pending_animal_arrivals.items()):
        if now - float(animal.get("last_seen_at") or now) > stale_after:
            _pending_animal_arrivals.pop(signature, None)
            continue
        frame, line = _animal_reaction_frame_and_line(animal)
        if _speak_async(
            line,
            frame.affect,
            purpose="world.animal_arrival",
            label=f"animal arrival: {(animal.get('species') or 'creature').strip().lower()}",
        ):
            _prime_emotion_frame(frame)
            _animal_reacted_at[signature] = now
            _pending_animal_arrivals.pop(signature, None)
            _log.info(
                "consciousness: animal arrival reaction fired signature=%s text=%r",
                signature,
                line,
            )
            return True
    return False


def _visible_face_people(snapshot: dict) -> list[dict]:
    people = snapshot.get("people") if isinstance(snapshot, dict) else []
    if not isinstance(people, list):
        return []
    visible: list[dict] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        has_face = bool(
            person.get("face_box")
            or person.get("bounding_box")
            or person.get("bbox")
            or person.get("box")
            or person.get("face_expression")
            or person.get("facial_expression")
        )
        if has_face:
            visible.append(person)
    return visible


def _person_db_id(person: dict) -> Optional[int]:
    try:
        value = person.get("person_db_id")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _smile_reaction_person_key(person: dict) -> Optional[str]:
    person_id = _person_db_id(person)
    if person_id is not None:
        return f"db:{person_id}"
    slot_id = str(person.get("id") or "").strip()
    if slot_id:
        return f"slot:{slot_id}"
    face_id = str(person.get("face_id") or "").strip()
    if face_id:
        return f"face:{face_id}"
    return None


def _find_person_by_smile_key(snapshot: dict, key: Optional[str]) -> Optional[dict]:
    if not key:
        return None
    for person in _visible_face_people(snapshot):
        if _smile_reaction_person_key(person) == key:
            return person
    return None


def _expression_reading(person: dict) -> dict:
    for field in ("face_expression", "facial_expression"):
        value = person.get(field)
        if isinstance(value, dict):
            return {
                "expression": str(value.get("expression") or value.get("mood") or ""),
                "mood": str(value.get("mood") or ""),
                "confidence": _safe_confidence(value.get("confidence")),
                "blendshapes": dict(value.get("blendshapes") or {}),
                "source": str(value.get("source") or ""),
                "updated_at": value.get("updated_at"),
            }
    mood = person.get("face_mood")
    if isinstance(mood, dict):
        return {
            "expression": str(mood.get("expression") or mood.get("mood") or ""),
            "mood": str(mood.get("mood") or ""),
            "confidence": _safe_confidence(mood.get("confidence")),
            "blendshapes": {},
            "source": str(mood.get("source") or ""),
            "updated_at": mood.get("updated_at"),
        }
    expression = str(person.get("expression") or "")
    if expression:
        return {
            "expression": expression,
            "mood": expression,
            "confidence": 1.0,
            "blendshapes": {},
            "source": "person.expression",
            "updated_at": None,
        }
    return {
        "expression": "",
        "mood": "",
        "confidence": 0.0,
        "blendshapes": {},
        "source": "",
        "updated_at": None,
    }


def _smile_blendshape_score(blendshapes: dict) -> float:
    return _mean_blendshape_score(blendshapes, "mouthSmileLeft", "mouthSmileRight")


def _mean_blendshape_score(blendshapes: dict, *keys: str) -> float:
    scores = []
    for key in keys:
        if key in blendshapes:
            scores.append(_safe_confidence(blendshapes.get(key)))
    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def _person_expression_label(person: dict) -> str:
    reading = _expression_reading(person)
    return (
        _norm_expression_label(reading.get("expression"))
        or _norm_expression_label(reading.get("mood"))
        or "unknown"
    )


def _person_is_smiling(person: dict) -> bool:
    reading = _expression_reading(person)
    expression = _norm_expression_label(reading.get("expression"))
    mood = _norm_expression_label(reading.get("mood"))
    confidence = _safe_confidence(reading.get("confidence"))
    blend_score = _smile_blendshape_score(reading.get("blendshapes") or {})
    min_conf = _safe_confidence(getattr(config, "SMILE_REACTION_MIN_CONFIDENCE", 0.45))
    if expression in _SMILE_LABELS or mood in _SMILE_LABELS:
        return max(confidence, blend_score) >= min_conf
    return blend_score >= min_conf


def _expression_kind_blend_score(kind: str, blendshapes: dict) -> float:
    if kind == "smile":
        return _mean_blendshape_score(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    if kind == "surprise":
        eye_wide = _mean_blendshape_score(blendshapes, "eyeWideLeft", "eyeWideRight")
        jaw_open = _safe_confidence(blendshapes.get("jawOpen"))
        brow_inner = _safe_confidence(blendshapes.get("browInnerUp"))
        scores = [score for score in (eye_wide, jaw_open, brow_inner) if score > 0.0]
        return sum(scores) / float(len(scores)) if scores else 0.0
    if kind == "frown":
        return _mean_blendshape_score(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    if kind == "brow_furrow":
        return _mean_blendshape_score(blendshapes, "browDownLeft", "browDownRight")
    return 0.0


def _facial_expression_reaction_min_confidence(kind: str) -> float:
    if kind == "smile":
        return _safe_confidence(
            getattr(config, "FACIAL_EXPRESSION_REACTION_SMILE_MIN_CONFIDENCE", 0.70)
        )
    if kind == "brow_furrow":
        return _safe_confidence(
            getattr(config, "FACIAL_EXPRESSION_REACTION_BROW_FURROW_MIN_CONFIDENCE", 0.78)
        )
    return _safe_confidence(
        getattr(config, "FACIAL_EXPRESSION_REACTION_MIN_CONFIDENCE", 0.55)
    )


def _person_reactable_expression(person: dict) -> tuple[Optional[str], float]:
    reading = _expression_reading(person)
    expression = _norm_expression_label(reading.get("expression"))
    mood = _norm_expression_label(reading.get("mood"))
    confidence = _safe_confidence(reading.get("confidence"))
    blendshapes = reading.get("blendshapes") or {}
    best_kind: Optional[str] = None
    best_score = 0.0
    for kind, labels in _FACIAL_EXPRESSION_REACTION_LABELS.items():
        blend_score = _expression_kind_blend_score(kind, blendshapes)
        if expression in labels or mood in labels:
            score = max(confidence, blend_score)
        else:
            score = blend_score
        if score > best_score:
            best_kind = kind
            best_score = score
    if best_kind is None or best_score < _facial_expression_reaction_min_confidence(best_kind):
        return None, best_score
    return best_kind, best_score


def _reading_timestamp_seconds(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _face_expression_reading_is_recent(reading: dict) -> bool:
    updated_at = _reading_timestamp_seconds(reading.get("updated_at"))
    if updated_at is None:
        return True
    max_age = max(
        0.5,
        float(getattr(config, "FACIAL_DISPOSITION_MAX_READING_AGE_SECS", 3.0) or 3.0),
    )
    return (time.time() - updated_at) <= max_age


def _step_disposition_memory(snapshot: dict) -> None:
    if not bool(getattr(config, "FACIAL_DISPOSITION_MEMORY_ENABLED", True)):
        return
    interval = max(
        0.5,
        float(getattr(config, "FACIAL_DISPOSITION_SAMPLE_INTERVAL_SECS", 2.0) or 2.0),
    )
    min_conf = _safe_confidence(getattr(config, "FACIAL_DISPOSITION_MIN_CONFIDENCE", 0.45))
    now = time.monotonic()
    for person in _visible_face_people(snapshot):
        person_id = _person_db_id(person)
        if person_id is None:
            continue
        last_at = _disposition_sampled_at.get(person_id, 0.0)
        if last_at and (now - last_at) < interval:
            continue
        reading = _expression_reading(person)
        if reading.get("source") != "mediapipe_face_landmarker":
            continue
        if not _face_expression_reading_is_recent(reading):
            continue
        confidence = _safe_confidence(reading.get("confidence"))
        if confidence < min_conf:
            continue
        try:
            from memory import disposition as disposition_memory
            disposition_memory.record_expression_sample(
                person_id,
                expression=reading.get("expression"),
                mood=reading.get("mood"),
                confidence=confidence,
            )
            _disposition_sampled_at[person_id] = now
        except Exception as exc:
            _log.debug("facial disposition sample failed person_id=%s: %s", person_id, exc)


def _smile_reaction_line_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _SMILE_REACTION_EXCLUDE_MARKERS)


def _rex_line_can_trigger_smile_reaction(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned or len(cleaned) > 240 or "?" in cleaned:
        return False
    lower = cleaned.lower()
    if _smile_reaction_line_text(cleaned):
        return False
    if any(marker in lower for marker in _SMILE_REACTION_SERIOUS_MARKERS):
        return False
    if any(marker in lower for marker in _SMILE_REACTION_SNARK_MARKERS):
        return True
    words = cleaned.split()
    return len(words) <= 22 and bool(re.search(r"[.!]", cleaned))


def _visible_reaction_target(
    snapshot: Optional[dict] = None,
    *,
    recent_window_secs: Optional[float] = None,
) -> Optional[dict]:
    snapshot = snapshot if isinstance(snapshot, dict) else world_state.snapshot()
    visible = _visible_face_people(snapshot)
    if not visible:
        return None

    try:
        recent = get_recent_engagement(window_secs=recent_window_secs)
    except Exception:
        recent = None
    if recent and recent.get("person_id") is not None:
        try:
            recent_id = int(recent.get("person_id"))
        except (TypeError, ValueError):
            recent_id = None
        if recent_id is not None:
            for person in visible:
                if _person_db_id(person) == recent_id:
                    return person

    if len(visible) == 1:
        return visible[0]
    known = [person for person in visible if _person_db_id(person) is not None]
    return known[0] if len(known) == 1 else None


def _smile_reaction_target(snapshot: Optional[dict] = None) -> Optional[dict]:
    return _visible_reaction_target(
        snapshot,
        recent_window_secs=float(
            getattr(config, "SMILE_REACTION_RECENT_ENGAGEMENT_SECS", 20.0)
        ),
    )


def _facial_expression_reaction_target(snapshot: Optional[dict] = None) -> Optional[dict]:
    return _visible_reaction_target(
        snapshot,
        recent_window_secs=float(
            getattr(config, "FACIAL_EXPRESSION_REACTION_RECENT_ENGAGEMENT_SECS", 30.0)
        ),
    )


def _smile_reaction_cooldown_active(now: Optional[float] = None) -> bool:
    now = time.monotonic() if now is None else now
    cooldown = max(0.0, float(getattr(config, "SMILE_REACTION_COOLDOWN_SECS", 75.0) or 0.0))
    return bool(cooldown and (now - _last_smile_reaction_at) < cooldown)


def _choose_expression_reaction_line(kind: str, lines) -> str:
    choices = list(lines or ())
    if not choices:
        return ""
    previous = _last_expression_reaction_line_by_kind.get(kind)
    if previous and len(choices) > 1:
        choices = [line for line in choices if line != previous] or choices
    line = random.choice(choices)
    _last_expression_reaction_line_by_kind[kind] = line
    return line


def _arm_smile_reaction_watch(item, *, phase: str) -> bool:
    global _smile_reaction_watch
    if not bool(getattr(config, "SMILE_REACTION_ENABLED", True)):
        return False
    if getattr(item, "audio_path", None):
        return False
    tag = str(getattr(item, "tag", "") or "")
    if tag == "social:smile_reaction":
        return False
    text = str(getattr(item, "text", "") or "").strip()
    if not _rex_line_can_trigger_smile_reaction(text):
        return False

    now = time.monotonic()
    if _smile_reaction_cooldown_active(now):
        return False

    target = _smile_reaction_target()
    if target is None or _person_is_smiling(target):
        return False
    person_key = _smile_reaction_person_key(target)
    if not person_key:
        return False

    window = max(1.0, float(getattr(config, "SMILE_REACTION_WINDOW_SECS", 5.0) or 5.0))
    watch = {
        "trigger_seq": getattr(item, "seq", None),
        "trigger_text": text,
        "person_key": person_key,
        "baseline_expression": _person_expression_label(target),
        "armed_at": now,
        "speech_started_at": now if phase == "start" else None,
        "speech_done_at": now if phase == "done" else None,
        "expires_at": now + window,
    }
    with _smile_reaction_lock:
        if _smile_reaction_cooldown_active(now):
            return False
        _smile_reaction_watch = watch
    _log.debug(
        "consciousness: armed smile reaction watch person=%s baseline=%s phase=%s",
        person_key,
        watch["baseline_expression"],
        phase,
    )
    return True


def _note_rex_speech_item_started(item) -> None:
    _arm_smile_reaction_watch(item, phase="start")


def _note_rex_speech_item_done(item) -> None:
    global _smile_reaction_watch
    if not bool(getattr(config, "SMILE_REACTION_ENABLED", True)):
        return
    now = time.monotonic()
    window = max(1.0, float(getattr(config, "SMILE_REACTION_WINDOW_SECS", 5.0) or 5.0))
    seq = getattr(item, "seq", None)
    with _smile_reaction_lock:
        watch = _smile_reaction_watch
        if watch and seq is not None and watch.get("trigger_seq") == seq:
            watch["speech_done_at"] = now
            watch["expires_at"] = max(float(watch.get("expires_at") or 0.0), now + window)
            return
    if _arm_smile_reaction_watch(item, phase="done"):
        with _smile_reaction_lock:
            if _smile_reaction_watch and _smile_reaction_watch.get("trigger_seq") == seq:
                _smile_reaction_watch["speech_done_at"] = now
                _smile_reaction_watch["expires_at"] = max(
                    float(_smile_reaction_watch.get("expires_at") or 0.0),
                    now + window,
                )


def _can_smile_reaction_speak() -> bool:
    if not _can_speak():
        return False

    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            if games_mod.suppresses_conversation_interruptions():
                return False
        elif games_mod.is_active():
            return False
    except Exception:
        pass

    current_state = state_module.get_state()
    if (
        current_state == State.ACTIVE
        and not getattr(config, "CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE", False)
    ):
        return False

    if is_waiting_for_response() or _proactive_speech_pending.is_set():
        return False
    try:
        if _situation_assessor.is_interaction_busy():
            return False
    except Exception:
        pass
    try:
        from audio import output_gate, speech_queue
        if speech_queue.is_speaking() or output_gate.is_busy():
            return False
    except Exception:
        return False
    return True


def _speak_smile_reaction(text: str) -> bool:
    global _last_smile_reaction_at
    if not text or not text.strip():
        return False
    if not _can_smile_reaction_speak():
        return False
    try:
        from audio import speech_queue
        _proactive_speech_pending.set()
        done = speech_queue.enqueue(
            text.strip(),
            "happy",
            priority=1,
            tag="social:smile_reaction",
        )

        def _on_done() -> None:
            try:
                done.wait()
            finally:
                _proactive_speech_pending.clear()

        threading.Thread(
            target=_on_done,
            daemon=True,
            name="smile-reaction-pending-clear",
        ).start()
        try:
            conv_log.log_rex(text.strip())
        except Exception as exc:
            _log.debug("conversation log write failed for smile reaction: %s", exc)
        note_rex_utterance(text.strip(), open_response_wait=False)
        _last_smile_reaction_at = time.monotonic()
        return True
    except Exception as exc:
        _proactive_speech_pending.clear()
        _log.debug("smile reaction speech failed: %s", exc)
        return False


def _step_smile_reaction(snapshot: dict, profile: SituationProfile) -> None:
    del profile
    global _smile_reaction_watch
    if not bool(getattr(config, "SMILE_REACTION_ENABLED", True)):
        return
    now = time.monotonic()
    with _smile_reaction_lock:
        watch = dict(_smile_reaction_watch or {})
    if not watch:
        return
    if now > float(watch.get("expires_at") or 0.0):
        with _smile_reaction_lock:
            if _smile_reaction_watch and _smile_reaction_watch.get("armed_at") == watch.get("armed_at"):
                _smile_reaction_watch = None
        return

    speech_done_at = watch.get("speech_done_at")
    if speech_done_at is None:
        return
    min_delay = max(
        0.0,
        float(getattr(config, "SMILE_REACTION_MIN_DELAY_SECS", 0.35) or 0.0),
    )
    if now - float(speech_done_at) < min_delay:
        return

    person = _find_person_by_smile_key(snapshot, watch.get("person_key"))
    if person is None:
        return
    if not _person_is_smiling(person):
        return

    with _smile_reaction_lock:
        if (
            _smile_reaction_watch
            and _smile_reaction_watch.get("armed_at") == watch.get("armed_at")
        ):
            _smile_reaction_watch = None
        else:
            return

    if _smile_reaction_cooldown_active(now):
        return
    line = _choose_expression_reaction_line("smile", _SMILE_REACTION_LINES)
    if _speak_smile_reaction(line):
        _log.info(
            "consciousness: smile reaction fired person=%s baseline=%s current=%s",
            watch.get("person_key"),
            watch.get("baseline_expression"),
            _person_expression_label(person),
        )


def _facial_expression_reaction_sustain_secs(kind: str) -> float:
    if kind == "smile":
        return max(
            0.0,
            float(
                getattr(
                    config,
                    "FACIAL_EXPRESSION_REACTION_SMILE_SUSTAIN_SECS",
                    1.0,
                )
                or 0.0
            ),
        )
    if kind == "surprise":
        return max(
            0.0,
            float(
                getattr(
                    config,
                    "FACIAL_EXPRESSION_REACTION_SURPRISE_SUSTAIN_SECS",
                    0.5,
                )
                or 0.0
            ),
        )
    if kind == "brow_furrow":
        return max(
            0.0,
            float(
                getattr(
                    config,
                    "FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS",
                    3.0,
                )
                or 0.0
            ),
        )
    return max(
        0.0,
        float(getattr(config, "FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS", 1.25) or 0.0),
    )


def _update_facial_expression_observation(
    person_key: str,
    kind: Optional[str],
    score: float,
    now: float,
) -> Optional[dict]:
    if not person_key:
        return None
    if not kind:
        _facial_expression_observed[person_key] = {
            "kind": None,
            "score": score,
            "started_at": now,
            "last_seen_at": now,
        }
        return None

    state = _facial_expression_observed.get(person_key)
    if not state or state.get("kind") != kind:
        state = {
            "kind": kind,
            "score": score,
            "started_at": now,
            "last_seen_at": now,
        }
        _facial_expression_observed[person_key] = state
        if _facial_expression_reaction_sustain_secs(kind) > 0.0:
            return None
        return state

    state["score"] = max(float(state.get("score") or 0.0), score)
    state["last_seen_at"] = now
    if (now - float(state.get("started_at") or now)) < _facial_expression_reaction_sustain_secs(kind):
        return None
    return state


def _facial_expression_reaction_on_cooldown(
    person_key: str,
    kind: str,
    now: float,
) -> bool:
    global_gap = max(
        0.0,
        float(
            getattr(config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 30.0)
            or 0.0
        ),
    )
    if global_gap and (now - _last_facial_expression_reaction_at) < global_gap:
        return True
    per_expression_gap = max(
        0.0,
        float(getattr(config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 120.0) or 0.0),
    )
    last_at = _facial_expression_reacted_at.get((person_key, kind), 0.0)
    return bool(per_expression_gap and (now - last_at) < per_expression_gap)


def _speak_facial_expression_reaction(kind: str, text: str) -> bool:
    emotion = {
        "smile": "happy",
        "surprise": "curious",
        "frown": "curious",
        "brow_furrow": "curious",
    }.get(kind, "neutral")
    return _speak_async(
        text,
        emotion,
        wait_secs=None,
        purpose=f"social.facial_expression.{kind}",
        label=f"facial expression reaction: {kind}",
    )


def _step_facial_expression_reactions(snapshot: dict, profile: SituationProfile) -> None:
    del profile
    global _last_facial_expression_reaction_at
    if not bool(getattr(config, "FACIAL_EXPRESSION_REACTIONS_ENABLED", True)):
        return
    with _smile_reaction_lock:
        if _smile_reaction_watch:
            return

    person = _facial_expression_reaction_target(snapshot)
    if person is None:
        return
    person_key = _smile_reaction_person_key(person)
    if not person_key:
        return

    kind, score = _person_reactable_expression(person)
    now = time.monotonic()
    state = _update_facial_expression_observation(person_key, kind, score, now)
    if not kind or not state:
        return
    if _facial_expression_reaction_on_cooldown(person_key, kind, now):
        return

    lines = _FACIAL_EXPRESSION_REACTION_LINES.get(kind) or ()
    line = _choose_expression_reaction_line(kind, lines)
    if not line:
        return
    if _speak_facial_expression_reaction(kind, line):
        _last_facial_expression_reaction_at = time.monotonic()
        _facial_expression_reacted_at[(person_key, kind)] = _last_facial_expression_reaction_at
        _log.info(
            "consciousness: facial expression reaction fired person=%s kind=%s score=%.2f",
            person_key,
            kind,
            float(score),
        )


def _claim_proactive_purpose(
    purpose: str,
    *,
    priority: Optional[int] = None,
    label: str = "",
) -> Optional[str]:
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.claim_proactive_purpose(
            purpose,
            priority=priority,
            label=label,
        )
    except Exception as exc:
        _log.debug("proactive purpose claim failed: %s", exc)
        return None


def _release_proactive_purpose(token: Optional[str]) -> None:
    try:
        from intelligence import conversation_agenda
        conversation_agenda.release_proactive_claim(token)
    except Exception:
        pass


def _proactive_purpose_current(token: Optional[str]) -> bool:
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.proactive_claim_is_current(token)
    except Exception:
        return True


def _apply_proactive_directive(prompt: str, purpose: Optional[str]) -> str:
    if not purpose:
        return prompt
    try:
        from intelligence import conversation_agenda
        return conversation_agenda.with_proactive_directive(prompt, purpose)
    except Exception:
        return prompt


def _governor_source() -> str:
    try:
        for frame in inspect.stack(context=0):
            name = frame.function
            if name.startswith("_step_") or name.startswith("_do_"):
                return name
    except Exception:
        pass
    return "consciousness"


def _governor_speech_metadata() -> dict:
    metadata = {
        "waiting_for_response": is_waiting_for_response(),
        "can_speak": _can_speak(),
    }
    try:
        current_state = state_module.get_state()
        metadata["state"] = getattr(current_state, "name", str(current_state))
        metadata["active_state_proactive_blocked"] = (
            current_state == State.ACTIVE
            and not getattr(config, "CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE", False)
        )
    except Exception:
        pass
    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            metadata["game_interruptions_suppressed"] = bool(
                games_mod.suppresses_conversation_interruptions()
            )
        elif hasattr(games_mod, "is_active"):
            metadata["game_interruptions_suppressed"] = bool(games_mod.is_active())
    except Exception:
        pass
    metadata["proactive_speech_pending"] = _proactive_speech_pending.is_set()
    try:
        metadata["interaction_busy"] = _situation_assessor.is_interaction_busy()
    except Exception:
        pass
    try:
        from audio import output_gate, speech_queue
        metadata["speech_queue_speaking"] = speech_queue.is_speaking()
        metadata["output_gate_busy"] = output_gate.is_busy()
    except Exception:
        metadata["output_gate_status_error"] = True
    with _turn_lock:
        last_spoken = _last_proactive_speech_at
    if last_spoken:
        recent_gap = time.monotonic() - last_spoken
        metadata["seconds_since_rex_spoke"] = recent_gap
        min_gap = max(0.0, float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 0.0)))
        if min_gap and recent_gap < min_gap:
            metadata["cooldown_active"] = True
            metadata["cooldown_reason"] = "proactive_speech_cooldown"
            metadata["cooldown_remaining_secs"] = max(0.0, min_gap - recent_gap)
    try:
        metadata["can_proactive_speak"] = _can_proactive_speak()
    except Exception:
        pass
    return metadata


def _observe_governor_candidate(
    *,
    purpose: Optional[str],
    label: str = "",
    prompt: str = "",
    suggested_text: str = "",
    emotion: str = "neutral",
    wait_secs: Optional[float] = None,
    priority: Optional[int] = None,
    target_person_id: Optional[int] = None,
    target_label: str = "",
    requires_llm: bool = True,
    source: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    try:
        from intelligence.action_governor import CandidateMove, governor
        if not governor.active():
            return None
        merged = _governor_speech_metadata()
        if metadata:
            merged.update(metadata)
        candidate = CandidateMove(
            source=source or _governor_source(),
            purpose=purpose or "direct_speech",
            label=label or purpose or "",
            prompt=prompt,
            suggested_text=suggested_text,
            emotion=emotion,
            priority=priority,
            target_person_id=target_person_id,
            target_label=target_label,
            requires_llm=requires_llm,
            wait_secs=wait_secs,
            metadata=merged,
        )
        return governor.observe(candidate)
    except Exception as exc:
        _log.debug("action governor observe failed: %s", exc)
        return None


def _mark_governor_candidate(candidate_id: Optional[str], outcome: str, reason: str = "") -> None:
    if not candidate_id:
        return
    try:
        from intelligence.action_governor import governor
        governor.mark_outcome(candidate_id, outcome, reason)
    except Exception as exc:
        _log.debug("action governor outcome update failed: %s", exc)


def _start_governor_cycle(profile: SituationProfile) -> None:
    try:
        from intelligence.action_governor import governor
        governor.start_cycle(profile=profile)
    except Exception as exc:
        _log.debug("action governor cycle start failed: %s", exc)


def _finish_governor_cycle() -> None:
    try:
        from intelligence.action_governor import governor
        governor.finish_cycle()
    except Exception as exc:
        _log.debug("action governor cycle finish failed: %s", exc)


def _generate_and_speak(
    prompt: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
    purpose: Optional[str] = None,
    priority: Optional[int] = None,
    label: str = "",
    metadata: Optional[dict] = None,
) -> bool:
    candidate_id = _observe_governor_candidate(
        purpose=purpose,
        label=label,
        prompt=prompt,
        emotion=emotion,
        wait_secs=wait_secs,
        priority=priority,
        requires_llm=True,
        metadata=metadata,
    )
    token = None
    if purpose:
        token = _claim_proactive_purpose(
            purpose,
            priority=priority,
            label=label or purpose,
        )
        if token is None:
            _mark_governor_candidate(
                candidate_id,
                "dropped",
                "conversation_agenda_claim_rejected",
            )
            return False
    _mark_governor_candidate(candidate_id, "accepted", "current_behavior_queued_llm")
    prompt = _apply_proactive_directive(prompt, purpose)

    def _task():
        try:
            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return
            from intelligence.llm import get_response
            text = get_response(prompt)
            if text and _proactive_purpose_current(token):
                _speak_async(text, emotion, wait_secs=wait_secs, governed=False)
        except Exception as exc:
            _log.debug("_generate_and_speak error: %s", exc)
        finally:
            _release_proactive_purpose(token)

    threading.Thread(target=_task, daemon=True).start()
    return True


def _should_fire_presence(
    key,
    person_db_id: Optional[int],
    profile: SituationProfile,
    *,
    allow_engaged: bool = False,
    bypass_cooldown: bool = False,
) -> bool:
    """
    Unified gate for presence (departure/return) reactions.

    Stricter than generic proactive speech: we never narrate presence events for
    the person Rex is currently talking to, never during the user's own sentence,
    never while another presence reaction for this person is already queued, and
    never more often than PRESENCE_PER_PERSON_COOLDOWN_SECS per person.
    """
    if not _can_speak():
        return False
    if not _can_proactive_speak():
        return False
    if profile.suppress_proactive or profile.interaction_busy:
        return False
    if profile.user_mid_sentence:
        return False
    if not allow_engaged and is_engaged_with(person_db_id):
        return False

    cooldown = getattr(config, "PRESENCE_PER_PERSON_COOLDOWN_SECS", 120.0)
    last = _last_presence_reaction_at.get(key, 0.0)
    if not bypass_cooldown and last and (time.monotonic() - last) < cooldown:
        return False

    try:
        from audio import speech_queue
        if speech_queue.has_waiting_with_tag(f"presence:{key}"):
            return False
    except Exception:
        pass
    return True


def _ensure_named_startup_greeting(text: str, first_name: Optional[str]) -> str:
    """Make first-sight startup lines visibly begin as a named greeting."""
    cleaned = str(text or "").strip()
    name = str(first_name or "").strip()
    if not cleaned or not name:
        return cleaned

    name_pat = re.escape(name)
    greeting_with_name = re.compile(
        rf"^(?:hey|hi|hello|good\s+(?:morning|afternoon|evening)|"
        rf"there\s+you\s+are)\b[^\n.!?]{{0,70}}\b{name_pat}\b",
        re.IGNORECASE,
    )
    if greeting_with_name.search(cleaned):
        return cleaned

    starts_with_name = re.compile(
        rf"^\s*{name_pat}\b(?P<punct>[,!.?;:]?)\s*",
        re.IGNORECASE,
    )
    match = starts_with_name.match(cleaned)
    if match:
        remainder = cleaned[match.end():].lstrip()
        punct = match.group("punct") or ","
        if punct in {";", ":"}:
            punct = ","
        if remainder:
            return f"Hey {name}{punct} {remainder}"
        return f"Hey {name}."

    return f"Hey {name}. {cleaned}"


def _generate_and_speak_presence(
    prompt: str,
    label: str,
    tag_key,
    emotion: str = "neutral",
    *,
    purpose: str = "presence_reaction",
    priority: Optional[int] = None,
    startup_greeting_name: Optional[str] = None,
    question_key: Optional[str] = None,
    question_depth: int = 1,
    direct_text: Optional[str] = None,
) -> bool:
    """
    Presence-reaction variant of _generate_and_speak.

    All gating now flows through _should_fire_presence() before this is called.
    The tag_key is used to coalesce duplicate queued reactions for the same
    person (newer replaces older).
    """
    speech_text = str(direct_text or "").strip()
    candidate_id = _observe_governor_candidate(
        purpose=purpose,
        label=label,
        prompt=prompt,
        suggested_text=speech_text,
        emotion=emotion,
        priority=priority,
        target_person_id=tag_key if isinstance(tag_key, int) else None,
        target_label=str(tag_key) if not isinstance(tag_key, int) else "",
        requires_llm=not bool(speech_text),
    )
    token = _claim_proactive_purpose(purpose, priority=priority, label=label)
    if token is None:
        _mark_governor_candidate(
            candidate_id,
            "dropped",
            "conversation_agenda_claim_rejected",
        )
        return False
    _mark_governor_candidate(
        candidate_id,
        "accepted",
        "current_behavior_queued_direct_speech" if speech_text else "current_behavior_queued_llm",
    )
    if not speech_text:
        prompt = _apply_proactive_directive(prompt, purpose)

    def _task():
        if not _presence_reaction_lock.acquire(blocking=False):
            _log.debug("_generate_and_speak_presence: reaction already in progress, skipping — %s", label)
            _release_proactive_purpose(token)
            return
        try:
            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return
            if speech_text:
                text = speech_text
            else:
                from intelligence.llm import get_response
                text = get_response(prompt)
            if not text or not text.strip():
                return
            if startup_greeting_name and not speech_text:
                text = _ensure_named_startup_greeting(text, startup_greeting_name)
            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return

            delay = getattr(config, "PRESENCE_REACTION_DELAY_SECS", 2.0)
            if delay > 0:
                time.sleep(delay)

            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return

            from audio import speech_queue
            tag = f"presence:{tag_key}"
            _log.info("consciousness: firing presence reaction — %s: %r", label, text[:120])
            _last_presence_reaction_at[tag_key] = time.monotonic()
            done = speech_queue.enqueue(text, emotion, priority=1, tag=tag)
            if isinstance(tag_key, int) and _presence_line_counts_as_greeting(label, purpose):
                try:
                    from memory import people as people_mod
                    people_mod.record_greeting(tag_key)
                except Exception as exc:
                    _log.debug("record greeting failed for person_id=%s: %s", tag_key, exc)
            expects_reply = _utterance_expects_reply(text)
            note_rex_utterance(
                text,
                open_response_wait=False,
                source=purpose,
                topic=label,
                target_person_id=tag_key if isinstance(tag_key, int) else None,
            )
            if expects_reply:
                def _open_wait_after_presence_done() -> None:
                    done.wait()
                    begin_response_wait()

                threading.Thread(
                    target=_open_wait_after_presence_done,
                    daemon=True,
                    name="presence-response-wait",
                ).start()
            _record_proactive_question(
                tag_key if isinstance(tag_key, int) else None,
                text,
                label=label,
                purpose=purpose,
                question_key=question_key,
                question_depth=question_depth,
            )
            if (
                purpose in {"memory_followup", "celebration_checkin", "emotional_checkin"}
                and isinstance(tag_key, int)
            ):
                note_memory_hint(text, tag_key)
        except Exception as exc:
            _log.debug("_generate_and_speak_presence error: %s", exc)
        finally:
            _release_proactive_purpose(token)
            _presence_reaction_lock.release()

    threading.Thread(target=_task, daemon=True).start()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Anger cooldown
# ─────────────────────────────────────────────────────────────────────────────

def _step_anger_cooldown() -> None:
    try:
        from intelligence.personality import get_anger_level
        get_anger_level()  # auto-resets anger level if cooldown has elapsed
    except Exception as exc:
        _log.debug("anger cooldown error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Mood decay
# ─────────────────────────────────────────────────────────────────────────────

def _step_mood_decay(elapsed: float) -> None:
    try:
        from intelligence.personality import apply_mood_decay
        apply_mood_decay(elapsed)
    except Exception as exc:
        _log.debug("mood decay error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Interoception update
# ─────────────────────────────────────────────────────────────────────────────

def _step_interoception() -> None:
    try:
        from awareness import interoception
        sys_state = interoception.get_system_state()
        self_state = world_state.get("self_state")
        self_state.update(sys_state)
        world_state.update("self_state", self_state)
    except Exception as exc:
        _log.debug("interoception step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Chronoception update
# ─────────────────────────────────────────────────────────────────────────────

def _step_chronoception() -> None:
    try:
        from awareness.chronoception import get_time_context
        ctx = get_time_context()
        time_state = world_state.get("time")
        time_state.update(ctx)
        world_state.update("time", time_state)
    except Exception as exc:
        _log.debug("chronoception step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Person recognition
# ─────────────────────────────────────────────────────────────────────────────

def _mark_people_faces_missing(people: list[dict], *, now_mono: float) -> list[dict]:
    """Keep identity slots alive while explicitly removing stale face geometry."""
    held: list[dict] = []
    lost_age = max(0.0, now_mono - _last_face_seen_at) if _last_face_seen_at else None
    for person in people:
        slot = dict(person or {})
        slot["face_visible"] = False
        slot["face_missing"] = True
        slot["face_box"] = None
        slot["bounding_box"] = None
        slot["bbox"] = None
        slot["box"] = None
        slot["position"] = None
        slot["face_box_fraction"] = None
        slot["approach_vector"] = "lost"
        if lost_age is not None:
            slot["face_last_seen_age_secs"] = round(lost_age, 2)
        held.append(slot)
    return held


def _face_boxes_sticky_compatible(
    current: tuple | list | None,
    previous: tuple[float, float, float, float] | None,
    *,
    frame_w: int,
    frame_h: int,
) -> bool:
    if current is None or previous is None:
        return False
    try:
        cx, cy, cw, ch = [float(v) for v in current[:4]]
        px, py, pw, ph = [float(v) for v in previous[:4]]
    except Exception:
        return False
    if cw <= 0 or ch <= 0 or pw <= 0 or ph <= 0:
        return False

    center_dx = (cx + cw / 2.0) - (px + pw / 2.0)
    center_dy = (cy + ch / 2.0) - (py + ph / 2.0)
    center_dist = (center_dx * center_dx + center_dy * center_dy) ** 0.5
    frame_diag = max(1.0, (float(frame_w) ** 2 + float(frame_h) ** 2) ** 0.5)
    max_jump = min(frame_diag * 0.18, max(cw, ch, pw, ph) * 2.2 + 80.0)
    if center_dist > max_jump:
        return False

    size_ratio = max(cw * ch, pw * ph) / max(1.0, min(cw * ch, pw * ph))
    return size_ratio <= 4.0


def _step_person_recognition(frame) -> None:
    """
    Detect visible faces, resolve known identities via DB lookup, and update
    world_state.people with one slot per visible face.

    This function no longer depends on pose pre-populating people slots. If the
    pose pipeline is disabled or lagging, face recognition still works and can
    drive unknown-person onboarding prompts.
    """
    global _last_face_feedback_signature, _last_identity_prompt_at, _last_solo_identity
    global _last_face_seen_at
    try:
        from vision import face as face_mod

        if frame is None:
            _last_face_feedback_signature = None
            return

        detected = face_mod.detect_faces(frame)
        if not detected:
            # No visible faces this tick. Hold the last slots briefly so a
            # small/partly occluded face does not lose conversation identity on
            # a single detector miss. Geometry is cleared immediately so the
            # GUI never draws a stale box at an old camera coordinate.
            hold_secs = float(getattr(config, "FACE_DETECTION_HOLD_SECS", 3.0) or 0.0)
            now_mono = time.monotonic()
            within_hold = bool(
                hold_secs > 0
                and _last_face_seen_at
                and (now_mono - _last_face_seen_at) <= hold_secs
            )
            outcome = {"held": False}

            def _hold_or_clear(people):
                if not people:
                    return None
                if within_hold:
                    outcome["held"] = True
                    return _mark_people_faces_missing(people, now_mono=now_mono)
                return []

            world_state.mutate("people", _hold_or_clear)
            _previous_face_boxes.clear()
            if outcome["held"]:
                return
            _last_face_feedback_signature = None
            return

        _last_face_seen_at = time.monotonic()
        _note_startup_presence_evidence("face")

        # Identity stickiness: HOG face recognition flickers unknown↔known within
        # 1–2 frames. When there's one face and we identified it moments ago,
        # carry that identity forward if this frame can't match.
        frame_width = int(getattr(frame, "shape", [0, 0])[1] or 0)
        frame_height = int(getattr(frame, "shape", [0, 0, 0])[0] or 0)
        sticky_identity = None
        if (
            len(detected) == 1
            and _last_solo_identity is not None
            and (time.monotonic() - _last_solo_identity[2]) <= _SOLO_IDENTITY_STICKY_SECS
        ):
            sticky_identity = _last_solo_identity

        people = world_state.get("people")
        changed = False

        # Ensure one world-state person slot per detected face.
        if len(people) != len(detected):
            resized = []
            for i in range(len(detected)):
                base = people[i] if i < len(people) else {}
                resized.append({
                    "id": base.get("id") or f"person_{i + 1}",
                    "person_db_id": base.get("person_db_id"),
                    "face_id": base.get("face_id"),
                    "voice_id": base.get("voice_id"),
                    "distance_zone": base.get("distance_zone"),
                    "approach_vector": base.get("approach_vector"),
                    "face_box_fraction": base.get("face_box_fraction"),
                    "pose": base.get("pose"),
                    "gesture": base.get("gesture"),
                    "engagement": base.get("engagement"),
                    "age_estimate": base.get("age_estimate"),
                    "position": base.get("position"),
                    "face_box": base.get("face_box"),
                    "face_visible": base.get("face_visible"),
                    "face_missing": base.get("face_missing"),
                    "face_last_seen_at": base.get("face_last_seen_at"),
                    "face_last_seen_age_secs": base.get("face_last_seen_age_secs"),
                })
            people = resized
            changed = True

        recognized_names: list[str] = []
        unknown_count = 0
        any_identified_this_tick = False
        active_box_keys: set[str] = set()
        for idx, det in enumerate(detected):
            person_record = face_mod.identify_face(det["encoding"])
            if (
                person_record is None
                and sticky_identity is not None
                and _face_boxes_sticky_compatible(
                    det.get("bounding_box"),
                    sticky_identity[3] if len(sticky_identity) >= 4 else None,
                    frame_w=frame_width,
                    frame_h=frame_height,
                )
            ):
                # Carry forward last solo identity through a single-face miss.
                sticky_id, sticky_name, _, _ = sticky_identity
                person_record = {"id": sticky_id, "name": sticky_name}
            target_slot = people[idx] if idx < len(people) else None
            if person_record is not None:
                any_identified_this_tick = True
                recognized_name = person_record.get("name") or f"person_{person_record.get('id')}"
                recognized_names.append(recognized_name)

                # Prefer matching an already-assigned DB slot; otherwise fill first unknown slot.
                target_slot = None
                for ws_person in people:
                    if ws_person.get("person_db_id") == person_record.get("id"):
                        target_slot = ws_person
                        break
                if target_slot is None:
                    for ws_person in people:
                        if ws_person.get("face_id") is None:
                            target_slot = ws_person
                            break
            else:
                unknown_count += 1

            if target_slot is None:
                continue

            target_slot["face_visible"] = True
            target_slot["face_missing"] = False
            target_slot["face_last_seen_at"] = time.time()
            target_slot["face_last_seen_age_secs"] = 0.0
            changed = True

            box = det.get("bounding_box")
            if box:
                target_slot["face_box"] = tuple(box)
                try:
                    target_slot["position"] = (
                        int(box[0] + box[2] / 2),
                        int(box[1] + box[3] / 2),
                    )
                except Exception:
                    pass
                slot_key = str(
                    person_record.get("id")
                    if person_record is not None
                    else target_slot.get("id") or f"person_{idx + 1}"
                )
                active_box_keys.add(slot_key)
                try:
                    from vision import proxemics
                    target_slot["distance_zone"] = proxemics.get_distance_zone(
                        box,
                        frame_width,
                    )
                    previous_box = _previous_face_boxes.get(slot_key)
                    target_slot["approach_vector"] = (
                        proxemics.get_approach_vector(box, previous_box)
                        if previous_box
                        else "stationary"
                    )
                    target_slot["face_box_fraction"] = (
                        (box[2] / frame_width) if frame_width > 0 else None
                    )
                    _previous_face_boxes[slot_key] = box
                    changed = True
                except Exception as exc:
                    _log.debug("proxemics update failed: %s", exc)

            if person_record is not None:
                incoming_name = person_record.get("name")
                incoming_id = person_record.get("id")
                if (
                    target_slot.get("face_id") != incoming_name
                    or target_slot.get("person_db_id") != incoming_id
                ):
                    target_slot["face_id"] = incoming_name
                    target_slot["person_db_id"] = incoming_id
                    if target_slot.get("voice_id") is None and incoming_name:
                        target_slot["voice_id"] = incoming_name
                    changed = True
                    _log.info(
                        "consciousness: face identified → %s (db_id=%s)",
                        incoming_name,
                        incoming_id,
                    )

        for stale_key in list(_previous_face_boxes.keys()):
            if stale_key not in active_box_keys:
                _previous_face_boxes.pop(stale_key, None)

        known_unique = sorted(set(recognized_names))
        signature = f"known={','.join(known_unique)}|unknown={unknown_count}"
        if signature != _last_face_feedback_signature:
            if known_unique:
                print(f"[FACE] Known face detected: {', '.join(known_unique)}", flush=True)
                _log.info("consciousness: known face(s) visible: %s", ", ".join(known_unique))
            if unknown_count > 0:
                noun = "face" if unknown_count == 1 else "faces"
                print(f"[FACE] Unknown {noun} detected ({unknown_count})", flush=True)
                _log.info("consciousness: unknown %s detected (%d)", noun, unknown_count)
            _last_face_feedback_signature = signature

        _maybe_prompt_unknown_identity(
            unknown_count=unknown_count,
            known_unique=known_unique,
        )

        if changed:
            # Commit under the world_state lock. The slow face DB lookups above
            # ran without the lock, so another thread (pose / expression / an
            # identity binder) may have written since we read. Overlay their
            # decoration fields, and preserve any identity they bound where this
            # tick didn't assign one, so recognition doesn't revert them.
            def _commit(current):
                decor = (
                    "pose", "gesture", "engagement", "age_estimate",
                    "face_mood", "face_expression", "facial_expression", "expression",
                )
                ident = ("person_db_id", "face_id", "voice_id")
                for i, slot in enumerate(people):
                    if i >= len(current):
                        continue
                    cur = current[i]
                    for field in decor:
                        if field in cur:
                            slot[field] = cur[field]
                    for field in ident:
                        if slot.get(field) is None and cur.get(field) is not None:
                            slot[field] = cur[field]
                return people

            world_state.mutate("people", _commit)

        # Update solo identity snapshot for next tick's stickiness check.
        if len(detected) == 1 and any_identified_this_tick and recognized_names:
            # Find the db_id that matches the recognized name
            for ws_person in people:
                if ws_person.get("face_id") == recognized_names[0] and ws_person.get("person_db_id"):
                    _last_solo_identity = (
                        ws_person["person_db_id"],
                        ws_person["face_id"],
                        time.monotonic(),
                        tuple(ws_person.get("face_box") or ()) if ws_person.get("face_box") else None,
                    )
                    break
        elif len(detected) != 1:
            # Multiple or zero faces — stickiness no longer applies.
            _last_solo_identity = None
    except Exception as exc:
        _log.debug("person recognition step error: %s", exc)


def _maybe_prompt_unknown_identity(
    *,
    unknown_count: int,
    known_unique: list[str],
) -> None:
    """
    Ask a solo unknown visible person for their name.

    The physical droid can run in ACTIVE fallback when wake-word models are
    unavailable, so this must not be limited to IDLE. Relationship inquiry owns
    mixed known+unknown scenes; this prompt is for fresh databases / solo
    unknown visitors.
    """
    global _last_identity_prompt_at, _identity_prompt_reply_until

    if unknown_count <= 0 or known_unique:
        return
    if _pending_identity_prompt.is_set() or _identity_prompt_in_flight.is_set():
        return

    current_state = state_module.get_state()
    if current_state == State.ACTIVE and not bool(
        getattr(config, "IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE", False)
    ):
        return
    if current_state not in (State.IDLE, State.ACTIVE):
        return
    if not _can_proactive_speak():
        return

    now = time.monotonic()
    if (now - _last_identity_prompt_at) < _IDENTITY_PROMPT_COOLDOWN_SECS:
        return

    _log.info(
        "consciousness: prompting unknown person for identity (state=%s)",
        getattr(current_state, "name", current_state),
    )
    _identity_prompt_in_flight.set()

    def _identity_prompt_done() -> None:
        global _identity_prompt_reply_until
        wait_secs = float(getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0) or 0.0)
        _pending_identity_prompt.set()
        _identity_prompt_reply_until = time.monotonic() + max(0.0, wait_secs)
        begin_response_wait(wait_secs)
        _log.info("[identity_prompt] reply window open for %.1fs", wait_secs)
        _identity_prompt_in_flight.clear()

    queued = _speak_async(
        "Hold up, I don't know you yet. What name should I save for you?",
        emotion="curious",
        wait_secs=getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0),
        purpose="identity_prompt",
        label="identity_prompt",
        on_done=_identity_prompt_done,
    )
    if queued:
        _last_identity_prompt_at = now
    else:
        _identity_prompt_in_flight.clear()


def _step_body_social_analysis(frame) -> None:
    """
    Merge pose/gesture engagement into visible people and refresh crowd context.

    Face recognition owns identity and proxemic face boxes. Pose owns body
    engagement. Social analysis combines the latest people slots into a crowd
    mode so downstream conversation logic can use it.
    """
    global _last_pose_analysis_at

    now = time.monotonic()
    interval = float(getattr(config, "POSE_ANALYSIS_INTERVAL_SECS", 2.0) or 0.0)
    if frame is not None and (interval <= 0.0 or (now - _last_pose_analysis_at) >= interval):
        _last_pose_analysis_at = now
        try:
            from vision import pose as pose_mod
            pose_mod.detect_pose(frame)
        except Exception as exc:
            _log.debug("pose analysis step error: %s", exc)

    try:
        from awareness import social as social_mod
        social_mod.analyze_crowd(world_state.get("people") or [])
    except Exception as exc:
        _log.debug("social crowd analysis step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Follow-up check
# ─────────────────────────────────────────────────────────────────────────────

def _step_followup_check(snapshot: dict) -> None:
    """
    For each identified person in world_state.people, query the DB for pending
    follow-up events and store novel ones in _pending_followups.
    """
    try:
        from memory import events as events_mod
        for person in snapshot.get("people", []):
            db_id = person.get("person_db_id")
            if db_id is None:
                continue
            pending = events_mod.get_pending_followups(db_id)
            if not pending:
                continue
            with _followup_lock:
                existing_ids = {e.get("id") for e in _pending_followups.get(db_id, [])}
                for ev in pending:
                    if ev.get("id") not in existing_ids:
                        _pending_followups.setdefault(db_id, []).append(ev)
                        _log.debug(
                            "consciousness: queued follow-up for db_id=%s: %s",
                            db_id, ev.get("event_name"),
                        )
    except Exception as exc:
        _log.debug("followup check step error: %s", exc)


def _within_startup_group_window(now: Optional[float] = None) -> bool:
    if not getattr(config, "STARTUP_GROUP_GREETING_ENABLED", True):
        return False
    now = time.monotonic() if now is None else now
    window = float(getattr(config, "STARTUP_GROUP_GREETING_WINDOW_SECS", 45.0))
    if _process_started_mono <= 0.0:
        return False
    return (now - _process_started_mono) <= max(0.0, window)


def _step_startup_group_greeting(snapshot: dict, profile: SituationProfile) -> None:
    """Fire one relationship-aware group greeting during startup."""
    global _startup_group_signature, _startup_group_seen_at, _startup_solo_seen_at
    global _startup_empty_room_seen_at, _startup_empty_room_fired

    now = time.monotonic()
    if not _within_startup_group_window(now):
        return
    if not _can_proactive_speak() or profile.user_mid_sentence:
        return

    try:
        from intelligence import social_scene
        scene = social_scene.from_snapshot(snapshot)
    except Exception as exc:
        _log.debug("startup group scene build failed: %s", exc)
        return

    if any(_is_jeff_benziger(person.name) for person in scene.known):
        return

    if len(scene.known) < 2:
        if len(scene.known) == 1 and _startup_solo_seen_at <= 0.0:
            _startup_solo_seen_at = now
        return

    signature = scene.signature
    if signature in _startup_group_greeted_signatures:
        return
    if _startup_group_signature != signature:
        _startup_group_signature = signature
        _startup_group_seen_at = now
        return

    confirm = float(getattr(config, "STARTUP_GROUP_GREETING_CONFIRM_SECS", 2.0))
    if (now - _startup_group_seen_at) < max(0.0, confirm):
        return

    if not _should_fire_presence(f"group:{signature}", None, profile):
        return

    label = social_scene.visible_group_label(scene)
    prompt = None
    emotion = "curious"
    if (
        getattr(config, "MOOD_AWARE_FIRST_SIGHT_ENABLED", True)
        and len(scene.known) == 2
        and scene.unknown_count == 0
        and scene.crowd_count <= int(getattr(config, "MOOD_AWARE_FIRST_SIGHT_MAX_PEOPLE", 2) or 2)
    ):
        moods = _detect_group_startup_moods()
        prompt = _build_group_smile_startup_prompt(scene, moods)
        if prompt:
            emotion = "happy"
            _log.info(
                "consciousness: startup group smile greeting for %s moods=%r",
                label,
                moods,
            )
    _log.info("consciousness: startup group greeting for %s", label)
    queued = _generate_and_speak_presence(
        prompt or social_scene.startup_group_prompt(scene),
        label=f"startup group greeting for {label}",
        tag_key=f"group:{signature}",
        emotion=emotion,
        purpose="presence_reaction",
    )
    if queued:
        _startup_group_greeted_signatures.add(signature)
        for person in scene.known:
            _greeted_this_session.add(person.person_id)
            _last_presence_reaction_at[person.person_id] = now
            _first_sight_seen_at.pop(person.person_id, None)


def _hold_startup_individual_greeting(snapshot: dict, now: float) -> bool:
    """
    During startup, briefly hold solo first-sight callbacks so a second known
    face can settle into the scene and receive a group greeting.
    """
    if not _within_startup_group_window(now):
        return False
    if _startup_group_greeted_signatures:
        return False
    try:
        known_count = sum(
            1
            for person in snapshot.get("people", []) or []
            if person.get("person_db_id") is not None
        )
    except Exception:
        known_count = 0
    if known_count >= 2:
        return True
    if known_count == 1:
        first_seen = _startup_solo_seen_at or now
        hold = float(getattr(config, "STARTUP_GROUP_SOLO_HOLD_SECS", 8.0))
        return (now - first_seen) < max(0.0, hold)
    return False


def _pick_anticipated_event(person_db_id: Optional[int]) -> Optional[dict]:
    """
    Return the soonest upcoming event for this person that Rex hasn't already
    anticipated this session. Filtered by ANTICIPATION_LOOKAHEAD_DAYS so distant
    events don't get referenced. Returns None if none qualify.
    """
    if not isinstance(person_db_id, int):
        return None
    try:
        from datetime import date, datetime, timedelta
        from memory import events as events_mod
        upcoming = events_mod.get_upcoming_events(person_db_id)
        if not upcoming:
            return None
        lookahead_days = getattr(config, "ANTICIPATION_LOOKAHEAD_DAYS", 30)
        cutoff = date.today() + timedelta(days=lookahead_days)
        for ev in upcoming:
            ev_id = ev.get("id")
            if ev_id is None or (person_db_id, ev_id) in _anticipated_events:
                continue
            ev_date_str = ev.get("event_date")
            if ev_date_str:
                try:
                    ev_date = datetime.fromisoformat(ev_date_str).date()
                except ValueError:
                    continue
                if ev_date > cutoff:
                    continue
            return ev
    except Exception as exc:
        _log.debug("anticipation lookup error: %s", exc)
    return None


def _pick_birthday_window(person_db_id: Optional[int]) -> Optional[int]:
    """
    If the person has a stored birthday and it's within
    BIRTHDAY_REMINDER_WINDOW_DAYS, return days_until (0 = today).
    Otherwise None.
    """
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import facts as facts_mod
        from awareness.holidays import days_until_birthday
        for fact in facts_mod.get_facts(person_db_id):
            if fact.get("key") == "birthday":
                days = days_until_birthday(fact.get("value") or "")
                if days is None:
                    return None
                window = getattr(config, "BIRTHDAY_REMINDER_WINDOW_DAYS", 7)
                if 0 <= days <= window:
                    return days
                return None
    except Exception as exc:
        _log.debug("birthday window lookup error: %s", exc)
    return None


def _build_birthday_prompt(first_name: str, days_until: int) -> str:
    if days_until == 0:
        when = "is TODAY"
    elif days_until == 1:
        when = "is tomorrow"
    else:
        when = f"is in {days_until} days"
    return (
        f"You see '{first_name}', someone you know — and their birthday {when}. "
        f"Open with one short in-character Rex line that calls it out — warm, dry, "
        f"with the usual snark. Don't sing. Address {first_name} by name. One line only."
    )


def _pick_milestone(person_db_id: Optional[int]) -> Optional[int]:
    """
    Return the visit number Rex should acknowledge as a milestone, or None.
    visit_count in the DB reflects PRIOR visits — update_visit fires at session
    end — so the incoming visit number is visit_count + 1.
    """
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import people as people_mod
        person = people_mod.get_person(person_db_id)
        if not person:
            return None
        incoming = int(person.get("visit_count", 0)) + 1
        milestones = getattr(config, "VISIT_MILESTONES", ())
        return incoming if incoming in milestones else None
    except Exception as exc:
        _log.debug("milestone lookup error: %s", exc)
        return None


def _pick_absence_phase(person_db_id: Optional[int]) -> Optional[tuple[str, float]]:
    """
    Return ("long_absence", days) if last visit was long ago,
    ("recent_return", hours) if last visit was very recent, or None.
    Mutually exclusive — long absence wins ties.
    """
    if not isinstance(person_db_id, int):
        return None
    try:
        from datetime import datetime, timezone
        from memory import people as people_mod
        person = people_mod.get_person(person_db_id)
        if not person:
            return None
        last_seen_str = person.get("last_seen")
        if not last_seen_str:
            return None
        try:
            last_seen = datetime.fromisoformat(last_seen_str)
        except ValueError:
            return None
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - last_seen
        days = delta.total_seconds() / 86400.0
        hours = delta.total_seconds() / 3600.0
        long_thresh = getattr(config, "LONG_ABSENCE_THRESHOLD_DAYS", 60)
        recent_thresh = getattr(config, "RECENT_RETURN_THRESHOLD_HOURS", 48)
        if days >= long_thresh:
            return ("long_absence", days)
        if hours <= recent_thresh:
            return ("recent_return", hours)
    except Exception as exc:
        _log.debug("absence phase lookup error: %s", exc)
    return None


def _build_milestone_prompt(first_name: str, visit_number: int) -> str:
    return (
        f"You see '{first_name}', someone you know — and this is their visit number "
        f"{visit_number}, a milestone you actually want to acknowledge. Acknowledge "
        f"the milestone in one short dry, begrudging-but-warm Rex line, then end with "
        f"a small-talk question inviting them to share what they've been up to since "
        f"their last visit. Address {first_name} by name. Two short sentences max — "
        f"the second must end in a question mark."
    )


def _build_long_absence_prompt(first_name: str, days: float) -> str:
    days_int = int(round(days))
    if days_int >= 365:
        span = f"about {days_int // 365} year(s)"
    elif days_int >= 60:
        span = f"about {days_int // 30} months"
    else:
        span = f"{days_int} days"
    return (
        f"You see '{first_name}', someone you know — but it's been {span} since their "
        f"last visit. Open with one short dry, faintly accusatory Rex line about the "
        f"absence, then ask a curious small-talk question — where they've been, what "
        f"they've been doing, anything that gets them talking. Address {first_name} "
        f"by name. Two short sentences max — the second must end in a question mark."
    )


def _build_recent_return_prompt(first_name: str, hours: float) -> str:
    if hours < 1.5:
        span = "less than an hour ago"
    elif hours < 24:
        span = f"about {int(round(hours))} hours ago"
    else:
        span = "yesterday"
    return (
        f"You see '{first_name}' again — they were just here {span}. Open with one "
        f"short Rex line teasing the quick return, then ask a small-talk question "
        f"inviting them to share what brought them back or what's on their mind. "
        f"Address {first_name} by name. Two short sentences max — the second must "
        f"end in a question mark."
    )


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', …"""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _same_day_return_count(person_db_id: Optional[int]) -> int:
    """Prior greetings for this person earlier TODAY (0 if first time today)."""
    if not isinstance(person_db_id, int):
        return 0
    if not bool(getattr(config, "PRESENCE_SAME_DAY_RETURN_ENABLED", True)):
        return 0
    try:
        from memory import people as people_mod
        return people_mod.greetings_today_count(person_db_id)
    except Exception as exc:
        _log.debug("same-day return count lookup error: %s", exc)
        return 0


def _build_same_day_return_prompt(first_name: str, prior_greetings_today: int) -> str:
    """Roast-style 'oh, it's you again' opener for a same-day repeat activation.

    `prior_greetings_today` is how many times Rex already greeted them earlier
    today (>=1 here), so this activation is the (prior+1)th of the day.
    """
    nth = _ordinal(prior_greetings_today + 1)
    if prior_greetings_today >= 2:
        tally = (
            f"'{first_name}' has now powered you up for the {nth} time today — they "
            f"will not leave you alone"
        )
    else:
        tally = (
            f"You already greeted '{first_name}' earlier today, and here they are AGAIN"
        )
    return (
        f"{tally}. Open with a sharp, funny 'oh, it's you again' roast about how they "
        f"keep summoning you today — punch up, commit to the bit, keep it affectionate, "
        f"not mean. Then drop straight into normal conversation with ONE short question "
        f"about what they need this time. Address {first_name} by name. Two short "
        f"sentences max — the second must end in a question mark. Do NOT re-introduce "
        f"yourself or act like you haven't seen them today."
    )


def _build_anticipation_prompt(
    first_name: str, event: dict, situation: str
) -> Optional[str]:
    """
    Build a Rex prompt that opens with a preemptive reference to an upcoming
    event. `situation` is a short phrase describing the recognition moment
    (e.g. "you just booted and see them", "they just walked back into frame").
    """
    if random.random() >= getattr(config, "ANTICIPATION_PROBABILITY", 0.85):
        return None
    event_name = (event.get("event_name") or "").strip()
    if not event_name:
        return None
    event_date = event.get("event_date") or ""
    notes = (event.get("event_notes") or "").strip()
    when_clause = f" coming up on {event_date}" if event_date else " coming up"
    notes_clause = f" Context they gave: {notes}." if notes else ""
    return (
        f"You see '{first_name}', someone you know — {situation}. "
        f"You remember they have '{event_name}'{when_clause}.{notes_clause} "
        f"Open with a short in-character Rex line that PREEMPTIVELY references "
        f"this event — like you've been thinking about it and are bringing it up "
        f"before they do. Warm but dry. Address {first_name} by name. One line only."
    )


def _pick_due_emotional_checkin(person_db_id: Optional[int]) -> Optional[dict]:
    """Return the most recent active negative event due for a startup check-in."""
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import emotional_events as emo_events
        due = emo_events.get_startup_checkins(
            person_db_id,
            process_started_iso=_process_started_iso,
            limit=1,
        )
        return due[0] if due else None
    except Exception as exc:
        _log.debug("emotional check-in lookup error: %s", exc)
        return None


_VAGUE_AFFECT_RE = re.compile(
    r"^\s*(?:the\s+)?(?:speaker|user|person|they|he|she)?\s*"
    r"(?:feels?|seems?|appears?|is|was|seemed)\s+"
    r"(?:really\s+|very\s+|quite\s+|pretty\s+)?"
    r"(?:proud|happy|excited|good|pleased|glad|content|positive|confident|"
    r"upbeat|cheerful|satisfied|optimistic)\b",
    re.IGNORECASE,
)


def _event_age_days(mentioned_at: Optional[str]) -> float:
    """Age of an event in days; a large number when the timestamp is unusable so
    an undateable memory is treated as too old to lead a cold open."""
    raw = (mentioned_at or "").strip()
    if not raw:
        return 1e9
    try:
        when = datetime.fromisoformat(raw.replace("T", " "))
        return max(0.0, (datetime.utcnow() - when).total_seconds() / 86400.0)
    except Exception:
        return 1e9


def _celebration_worth_leading_with(event: dict) -> bool:
    """Gate vague/inferred/stale 'good news' out of the first-sight greeting.

    Leading a cold open with "you must feel proud of your problem-solving skills"
    reads as awkward. Only a concrete milestone — one the person actually told
    Rex about, or a recent one — is worth opening with. Everything else falls
    through to a normal warm greeting (the memory can still come up later)."""
    if not bool(getattr(config, "PRESENCE_CELEBRATION_REQUIRE_CONCRETE", True)):
        return True
    desc = str((event or {}).get("description") or "").strip()
    if len(re.findall(r"[A-Za-z']+", desc)) < 3:
        return False
    if _VAGUE_AFFECT_RE.search(desc):
        return False
    if (event or {}).get("person_invited_topic"):
        return True
    max_age = float(getattr(config, "PRESENCE_CELEBRATION_LEAD_MAX_AGE_DAYS", 21.0))
    return _event_age_days((event or {}).get("mentioned_at")) <= max_age


def _pick_due_celebration_checkin(person_db_id: Optional[int]) -> Optional[dict]:
    """Return the most recent concrete positive event worth leading a greeting."""
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import emotional_events as emo_events
        due = emo_events.get_startup_celebrations(
            person_db_id,
            process_started_iso=_process_started_iso,
            limit=5,
        )
        for event in due:
            if _celebration_worth_leading_with(event):
                return event
        return None
    except Exception as exc:
        _log.debug("celebration check-in lookup error: %s", exc)
        return None


def _first_sight_context(first_name: str) -> tuple[str, str]:
    """Return prompt phrasing for seeing a known person first time this run."""
    if _process_started_mono and (time.monotonic() - _process_started_mono) <= 45.0:
        return (
            f"You just started up and immediately see '{first_name}'.",
            f"you just booted up and immediately spot {first_name}",
        )
    return (
        f"'{first_name}', someone you know, just came into your camera view "
        f"for the first time this run.",
        f"{first_name} just came into your camera view for the first time this run",
    )


def _build_startup_solo_greeting_prompt(first_name: str, context_sentence: str) -> str:
    try:
        from intelligence import social_scene
        steering_examples = "; ".join(social_scene.FIRST_GREETING_STEERING_PHRASES)
    except Exception:
        steering_examples = (
            "What are you up to today?; "
            "What do you want to talk about?"
        )
    return (
        f"{context_sentence} "
        f"Greet {first_name} in-character by name, then end with ONE "
        f"short conversation-steering question in Rex's snarky DJ-R3X "
        f"voice. Pick one from this menu or invent a similar short variant; "
        f"do not reuse the same wording every run: {steering_examples}. "
        f"This is a solo greeting: use '{first_name}' "
        f"or 'you'; do NOT call this one visible person 'they' or 'them'. "
        f"Two short sentences max — the second must end in a question mark."
    )


def _age_days_since_iso(value) -> Optional[float]:
    ts = _reading_timestamp_seconds(value)
    if ts is None:
        return None
    return max(0.0, (time.time() - ts) / 86400.0)


def _pick_first_sight_disposition_greeting(
    person_id: Optional[int],
    first_name: str,
) -> Optional[tuple[str, str]]:
    if not isinstance(person_id, int):
        return None
    if not bool(getattr(config, "FACIAL_DISPOSITION_FIRST_SIGHT_ENABLED", True)):
        return None
    try:
        from memory import disposition as disposition_memory
        stats = disposition_memory.get_stats(person_id)
    except Exception as exc:
        _log.debug("disposition stats lookup failed person_id=%s: %s", person_id, exc)
        return None
    if not stats:
        return None

    try:
        total = int(stats.get("total_samples") or 0)
    except (TypeError, ValueError):
        total = 0
    min_samples = int(
        getattr(config, "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_SAMPLES", 20) or 20
    )
    if total < max(1, min_samples):
        return None

    label = str(stats.get("disposition_label") or "").strip().lower()
    if label not in _DISPOSITION_FIRST_SIGHT_LINES:
        return None
    confidence = _safe_confidence(stats.get("confidence"))
    min_conf = _safe_confidence(
        getattr(config, "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_CONFIDENCE", 0.50)
    )
    if confidence < min_conf:
        return None

    cooldown_days = max(
        0.0,
        float(getattr(config, "FACIAL_DISPOSITION_FIRST_SIGHT_COOLDOWN_DAYS", 2.0) or 0.0),
    )
    last_mentioned_age = _age_days_since_iso(stats.get("last_mentioned_at"))
    if (
        cooldown_days
        and last_mentioned_age is not None
        and last_mentioned_age < cooldown_days
    ):
        return None

    probability = _safe_confidence(
        getattr(config, "FACIAL_DISPOSITION_FIRST_SIGHT_PROBABILITY", 0.28)
    )
    if probability <= 0.0 or random.random() > probability:
        return None

    template = _choose_expression_reaction_line(
        f"disposition:{label}",
        _DISPOSITION_FIRST_SIGHT_LINES[label],
    )
    if not template:
        return None
    line = template.format(first_name=first_name)
    return label, line


def _pick_startup_profile_question(person_id: Optional[int]) -> Optional[dict]:
    """Pick a basic profile question for known people Rex barely knows."""
    if not isinstance(person_id, int):
        return None
    # A profile question is an awkward cold open. Let the first-sight greeting
    # stay casual ("what's up?") and ask profile questions once the conversation
    # is actually rolling instead.
    if not bool(getattr(config, "STARTUP_PROFILE_QUESTION_ENABLED", False)):
        return None
    if not bool(getattr(config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True)):
        return None
    max_facts = int(getattr(config, "LOW_MEMORY_PROFILE_MAX_FACTS", 4) or 4)
    if profile_questions.profile_fact_count(person_id) > max_facts:
        return None
    try:
        from intelligence import question_budget
        if not question_budget.can_ask("startup_profile_question"):
            return None
    except Exception:
        pass
    return profile_questions.next_profile_question(person_id)


def _build_startup_profile_question_prompt(
    first_name: str,
    context_sentence: str,
    question_text: str,
) -> str:
    return (
        f"{context_sentence} "
        f"Greet {first_name} in-character by name, then ask this exact basic "
        f"profile question: {question_text!r}. "
        "This is early getting-to-know-you curiosity, so keep it light and "
        "non-intimate. Do not ask about fears, regrets, grief, values, or life "
        "meaning. Use two short sentences max. The final sentence must preserve "
        "the question wording and end in a question mark. "
        f"This is a solo greeting: use '{first_name}' or 'you'; do not call this "
        "one visible person 'they' or 'them'."
    )


def _build_emotional_checkin_prompt(
    first_name: str,
    event: dict,
    context_sentence: str,
) -> str:
    category = (event.get("category") or "event").strip().lower()
    desc = (event.get("description") or "").strip()
    valence = float(event.get("valence", -0.5) or -0.5)
    if category in {"grief", "death"}:
        stance = (
            "This is a recent death or grief event. Lead with care. No teasing, "
            "no silver lining, no attempt to cheer them up with a joke. Do not "
            "make it sound like they just told you for the first time."
        )
        reference_rule = (
            f"This is shared context from memory: \"{desc}\". You may refer to it "
            f"softly as 'your loss' or 'everything' because you both know the "
            f"context. Do NOT explicitly remind them with robotic phrasing like "
            f"'I remember you said your grandpa died yesterday.' Do NOT say "
            f"'that sounds really tough' or similar first-time-listener phrases."
        )
    elif category in {"bad_day", "work_stress", "stress"}:
        stance = (
            "This was a recent rough day or stress event. Keep it light but kind; "
            "it should not feel dramatic."
        )
        reference_rule = (
            f"Briefly name the remembered context from this description: \"{desc}\"; "
            f"do not use vague phrases like 'everything' with no context."
        )
    elif valence <= -0.7:
        stance = "This is a serious recent hard thing. Be gentle and grounded."
        reference_rule = (
            f"Briefly name the remembered context from this description: \"{desc}\"; "
            f"do not use vague phrases like 'everything' with no context."
        )
    else:
        stance = "This is a recent difficult thing. Be warm and low-pressure."
        reference_rule = (
            f"If you mention it, briefly name the remembered context from this "
            f"description: \"{desc}\"; do not be cryptic."
        )
    return (
        f"{context_sentence} FIRST PRIORITY: "
        f"before birthdays, milestones, upcoming plans, long absences, or 'back so soon' "
        f"banter, you remember this sensitive event: category={category}, "
        f"description=\"{desc}\". {stance} {reference_rule} "
        f"In ONE short in-character Rex line, "
        f"gently check in on how {first_name} is doing. You may sound like Rex, "
        f"but ROAST OFF: no insults, no appearance comments, no jokes at their "
        f"expense. End with a low-pressure question. Good shapes for grief: "
        f"'Hey {first_name}, how are you holding up with everything?' or "
        f"'How are you doing with your loss?'"
    )


def _build_celebration_checkin_prompt(
    first_name: str,
    event: dict,
    context_sentence: str,
) -> str:
    category = (event.get("category") or "good_news").strip().lower()
    desc = (event.get("description") or "").strip()
    return (
        f"{context_sentence} You remember this good news or milestone for "
        f"{first_name}: category={category}, description=\"{desc}\". "
        f"Open with ONE short in-character Rex line that celebrates it without "
        f"making a huge speech. Warm, dry, no insult at their expense. You may "
        f"ask one low-pressure follow-up like 'how's that going?' only if it "
        f"fits naturally. Address {first_name} by name."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Disengagement detection
# ─────────────────────────────────────────────────────────────────────────────

def _step_disengagement(snapshot: dict, profile: SituationProfile) -> None:
    """
    If the dominant speaker is disengaging, fire a proactive re-engagement line.
    Rate-limited to _REENGAGEMENT_COOLDOWN_SECS per person.
    """
    if profile.suppress_proactive or not _can_proactive_speak():
        return
    try:
        from awareness.social import check_disengagement
        people = snapshot.get("people", [])
        disengaged = check_disengagement(people)
        dominant = snapshot.get("crowd", {}).get("dominant_speaker")
        if not dominant or dominant not in disengaged:
            return

        now = time.monotonic()
        last_sent = _reengagement_sent_at.get(dominant, 0.0)
        if now - last_sent < _REENGAGEMENT_COOLDOWN_SECS:
            return

        _reengagement_sent_at[dominant] = now
        _log.info("consciousness: dominant speaker disengaging — triggering re-engagement")
        _generate_and_speak(
            "The person you were just talking to is starting to disengage or drift away. "
            "Generate one short, in-character line to recapture their attention. "
            "Not desperate — Rex doesn't beg. One punchy line only.",
            emotion="curious",
            purpose="reengagement",
        )
    except Exception as exc:
        _log.debug("disengagement step error: %s", exc)


def _person_space_key(person: dict) -> str:
    return str(
        person.get("person_db_id")
        or person.get("face_id")
        or person.get("id")
        or "unknown"
    )


def _too_close_for_personal_space(person: dict) -> bool:
    min_zone = str(
        getattr(config, "PERSONAL_SPACE_REACTION_MIN_ZONE", "intimate") or "intimate"
    ).lower()
    zone = (person.get("distance_zone") or "").lower()
    if min_zone == "social":
        return zone in {"intimate", "social"}
    return zone == "intimate"


def _step_personal_space(snapshot: dict, profile: SituationProfile) -> None:
    """
    React once in a while when someone is extremely close to the camera.

    Camera proxemics are treated like American personal-space norms: a huge
    face in frame maps to "intimate" distance, which is fair game for a short
    boundary joke or roast.
    """
    if not getattr(config, "PERSONAL_SPACE_REACTION_ENABLED", True):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not _can_proactive_speak():
        return

    now = time.monotonic()
    cooldown = float(getattr(config, "PERSONAL_SPACE_REACTION_COOLDOWN_SECS", 45.0))
    for person in snapshot.get("people", []) or []:
        if not _too_close_for_personal_space(person):
            continue
        key = _person_space_key(person)
        if (now - _personal_space_reacted_at.get(key, 0.0)) < max(0.0, cooldown):
            continue
        _personal_space_reacted_at[key] = now
        name = person.get("face_id") or person.get("voice_id") or "this person"
        approach = person.get("approach_vector") or "stationary"
        prompt = (
            f"{name} is extremely close, in intimate personal-space range by "
            f"American norms. Approach vector: {approach}. Give ONE short "
            f"in-character Rex boundary joke or roast about them being too close "
            f"for comfort. Playful, not hostile. Do not mention sensors, cameras, "
            f"or tracking. Do not ask a question. Max 18 words."
        )
        _log.info(
            "consciousness: personal-space reaction for %s zone=%s approach=%s",
            key,
            person.get("distance_zone"),
            approach,
        )
        _generate_and_speak(
            prompt,
            emotion="curious",
            purpose="personal_space",
            label=f"personal space {key}",
        )
        return


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Proactive reactions
# ─────────────────────────────────────────────────────────────────────────────

def _step_proactive_reactions(snapshot: dict, profile: SituationProfile) -> None:
    """
    Compare current WorldState to _last_snapshot. For each notable change,
    generate and speak a short in-character reaction. Never fires in QUIET/SHUTDOWN.
    """
    global _acknowledged_dates, _last_weather_reaction_at, _last_startle_sound_reaction_at

    if _last_snapshot:
        _stage_animal_arrivals(snapshot)

    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _last_snapshot or not _can_proactive_speak():
        return
    if _startup_known_greeting_pending(snapshot):
        return
    if is_identity_prompt_waiting_for_reply():
        return
    if _fire_pending_animal_arrival_reaction():
        return

    try:
        triggers: list[dict] = []

        def _add_trigger(
            prompt: str,
            emotion: str,
            *,
            purpose: str = "world_reaction",
            label: str = "",
            metadata: Optional[dict] = None,
        ) -> None:
            triggers.append({
                "prompt": prompt,
                "emotion": emotion,
                "purpose": purpose,
                "label": label,
                "metadata": metadata or {},
            })

        def _frame_metadata(frame) -> dict:
            return {"emotion_frame": frame.as_dict()}

        def _add_emotion_trigger(
            prompt: str,
            frame,
            *,
            purpose: str = "world_reaction",
            label: str = "",
            metadata: Optional[dict] = None,
        ) -> None:
            merged = _frame_metadata(frame)
            if metadata:
                merged.update(metadata)
            _add_trigger(
                prompt,
                frame.affect,
                purpose=purpose,
                label=label,
                metadata=merged,
            )

        def _prime_emotion_trigger(metadata: dict) -> None:
            data = (metadata or {}).get("emotion_frame")
            if not isinstance(data, dict):
                return
            frame = emotion_orchestrator.frame_for_speech(data)
            _prime_emotion_frame(frame)

        # New person entered frame. During startup, known-person greetings own
        # the first line; crowd-count flicker should not steal the opening with
        # generic "someone new walked in" banter.
        prev_count = _last_snapshot.get("crowd", {}).get("count", 0)
        curr_count = snapshot.get("crowd", {}).get("count", 0)
        people_now = snapshot.get("people", []) or []
        known_now = [p for p in people_now if p.get("person_db_id") is not None]
        unknown_now = [p for p in people_now if p.get("person_db_id") is None]
        if (
            curr_count > prev_count
            and not _startup_known_greeting_pending(snapshot)
            and not (known_now and not unknown_now)
        ):
            _add_trigger(
                "Someone new just walked into your view. React in one short in-character line — "
                "somewhere between a greeting and a roast, delivered as you clock them entering.",
                "curious",
                label="new person entered view",
            )

        # Crowd size label changed significantly
        prev_label = _last_snapshot.get("crowd", {}).get("count_label")
        curr_label = snapshot.get("crowd", {}).get("count_label")
        if curr_label and prev_label and curr_label != prev_label:
            _add_trigger(
                f"The crowd around you just shifted from '{prev_label}' to '{curr_label}'. "
                "One short in-character observation about this change.",
                "neutral",
                label="crowd size changed",
            )

        # Notable sound event
        prev_sound = _last_snapshot.get("audio_scene", {}).get("last_sound_event")
        curr_sound = snapshot.get("audio_scene", {}).get("last_sound_event")
        if curr_sound and curr_sound != prev_sound:
            startle_events = set(getattr(config, "STARTLE_SOUND_EVENTS", {"scream", "sudden_loud_sound", "crash"}))
            is_startle = curr_sound in startle_events
            startle_allowed = bool(
                getattr(config, "WORLD_STARTLE_SOUND_EVENT_REACTIONS_ENABLED", True)
            )
            generic_allowed = bool(getattr(config, "WORLD_SOUND_EVENT_REACTIONS_ENABLED", False))
            cooldown = float(getattr(config, "STARTLE_SOUND_EVENT_REACTION_COOLDOWN_SECS", 20.0))
            if (
                is_startle
                and startle_allowed
                and (time.monotonic() - _last_startle_sound_reaction_at) >= cooldown
            ):
                frame = emotion_orchestrator.frame_for_event(curr_sound)
                _add_emotion_trigger(
                    f"You just registered a sudden startle sound event: '{curr_sound}'. "
                    "React like something genuinely startled you: a tiny yelp, squeal, "
                    "or very short Rex line. Do not ask a question. One line only.",
                    frame,
                    label=f"startle sound: {curr_sound}",
                    metadata={"startle_sound_event": curr_sound},
                )
            elif generic_allowed:
                _add_trigger(
                    f"You just registered a notable sound event: '{curr_sound}'. "
                    "One punchy in-character line reacting to it.",
                    "curious",
                    label="sound event reaction",
                )

        # Notable calendar date (once per session per date)
        notable_date = snapshot.get("time", {}).get("notable_date")
        if notable_date and notable_date not in _acknowledged_dates:
            _acknowledged_dates.add(notable_date)
            _add_trigger(
                f"Today is {notable_date}. Make one spontaneous in-character remark about it "
                "as if you just noticed the date. Deliver it Rex-style.",
                "excited",
                label=f"notable date: {notable_date}",
            )

        # Notable weather change. Weather comes from the network feed, not body
        # sensors, so prompts keep Rex honest about how he knows it.
        if bool(getattr(config, "WEATHER_PROACTIVE_REACTIONS_ENABLED", True)):
            curr_weather = snapshot.get("weather", {}) or {}
            prev_weather = _last_snapshot.get("weather", {}) or {}
            if curr_weather.get("available"):
                condition = (curr_weather.get("condition") or "unknown").lower()
                prev_condition = (prev_weather.get("condition") or "unknown").lower()
                temp = curr_weather.get("temp_f")
                prev_temp = prev_weather.get("temp_f")
                try:
                    temp_int = int(temp) if temp is not None else None
                    prev_temp_int = int(prev_temp) if prev_temp is not None else None
                except (TypeError, ValueError):
                    temp_int = prev_temp_int = None

                def _temp_bucket(value: Optional[int]) -> str:
                    if value is None:
                        return "unknown"
                    if value >= 95:
                        return "very_hot"
                    if value >= 85:
                        return "warm"
                    if value <= 40:
                        return "cold"
                    if value <= 55:
                        return "cool"
                    return "mild"

                notable_condition = condition in {"rain", "snow", "thunder", "fog"}
                condition_changed = condition != prev_condition and notable_condition
                bucket = _temp_bucket(temp_int)
                prev_bucket = _temp_bucket(prev_temp_int)
                temp_shifted = (
                    temp_int is not None
                    and prev_temp_int is not None
                    and abs(temp_int - prev_temp_int) >= 10
                )
                bucket_changed = bucket != prev_bucket and bucket in {"very_hot", "cold"}
                cooldown = float(getattr(config, "WEATHER_PROACTIVE_REACTION_COOLDOWN_SECS", 1800.0))
                signature = f"{condition}:{bucket}"
                if (
                    (condition_changed or temp_shifted or bucket_changed)
                    and signature not in _acknowledged_weather_signatures
                    and (time.monotonic() - _last_weather_reaction_at) >= cooldown
                ):
                    _acknowledged_weather_signatures.add(signature)
                    _last_weather_reaction_at = time.monotonic()
                    location = curr_weather.get("location") or "the local area"
                    desc = curr_weather.get("description") or condition
                    temp_clause = f"{temp_int}°F" if temp_int is not None else "temperature unavailable"
                    _add_trigger(
                        f"Your weather feed just updated for {location}: {temp_clause}, {desc}. "
                        "Make one short spontaneous Rex-style weather remark. You may imply "
                        "you saw it in your feed, but do not pretend you physically feel the weather.",
                        "curious",
                        purpose="weather.proactive_comment",
                        label="weather proactive comment",
                        metadata={
                            "topic_key": f"weather:{signature}",
                            "weather_signature": signature,
                            "weather_condition": condition,
                            "weather_bucket": bucket,
                        },
                    )

        if triggers:
            surprise_triggers = [
                t for t in triggers
                if isinstance((t.get("metadata") or {}).get("emotion_frame"), dict)
                and (t.get("metadata") or {}).get("emotion_frame", {}).get("affect") == "surprised"
            ]
            trigger = random.choice(surprise_triggers or triggers)
            metadata = trigger.get("metadata") or {}
            if metadata.get("startle_sound_event"):
                _last_startle_sound_reaction_at = time.monotonic()
            _prime_emotion_trigger(metadata)
            _generate_and_speak(
                trigger["prompt"],
                trigger["emotion"],
                purpose=trigger.get("purpose") or "world_reaction",
                label=trigger.get("label") or "",
                metadata=metadata or None,
            )

    except Exception as exc:
        _log.debug("proactive reactions step error: %s", exc)


def _startup_known_greeting_pending(snapshot: dict, now: Optional[float] = None) -> bool:
    """True while startup should reserve speech for known-person greetings."""
    now = time.monotonic() if now is None else now
    if not _within_startup_group_window(now):
        return False
    for person in snapshot.get("people", []) or []:
        pid = person.get("person_db_id")
        if pid is not None and int(pid) not in _greeted_this_session:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Idle micro-behaviors
# ─────────────────────────────────────────────────────────────────────────────

def _step_idle_micro_behavior(snapshot: dict, profile: SituationProfile) -> None:
    """
    In IDLE state, when sufficiently idle, fire one random micro-behavior:
    ambient scan, private thought, or idle audio clip.
    """
    global _last_micro_behavior_at

    if state_module.get_state() != State.IDLE:
        return
    if is_waiting_for_response():
        return
    now = time.monotonic()
    if _within_startup_group_window(now) and not _greeted_this_session:
        return
    if _startup_known_greeting_pending(snapshot):
        return

    interval_min = getattr(config, "MICRO_BEHAVIOR_INTERVAL_SECS_MIN", 15)
    interval_max = getattr(config, "MICRO_BEHAVIOR_INTERVAL_SECS_MAX", 45)
    since_last = now - _last_micro_behavior_at

    if since_last < interval_min:
        return

    # Don't fire immediately after an interaction
    last_interaction_ago = snapshot.get("self_state", {}).get("last_interaction_ago")
    if last_interaction_ago is not None and last_interaction_ago < interval_min:
        return

    # Randomise the trigger point within the [min, max] window
    if since_last < random.uniform(interval_min, interval_max):
        return

    _last_micro_behavior_at = now
    choices, weights = _idle_micro_behavior_choices(snapshot)
    behavior = random.choices(choices, weights=weights, k=1)[0]
    _log.debug("consciousness: idle micro-behavior → %s", behavior)

    if behavior == "empty_room_joke":
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            _do_empty_room_joke(snapshot)
    elif behavior == "small_talk_question":
        if not profile.suppress_proactive:
            _do_small_talk_question(snapshot)
    elif behavior == "ambient_scan":
        _do_ambient_scan()
    elif behavior == "private_thought":
        # Private thoughts are system monologues — suppressed by both proactive and
        # system-comment gates so Rex doesn't mutter about himself mid-conversation.
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            _do_private_thought()
    elif behavior == "aspiration":
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            _do_aspiration()
    elif behavior == "idle_clip":
        if not profile.suppress_proactive:
            _do_idle_clip()
    elif behavior == "ambient_observation":
        if not profile.suppress_proactive:
            _do_ambient_observation(snapshot)
    elif behavior == "appearance_riff":
        if not profile.suppress_proactive:
            _do_appearance_riff(snapshot)
    elif behavior == "people_roast":
        if not profile.suppress_proactive:
            _do_people_roast(snapshot)
    elif behavior == "live_vision_comment":
        if not profile.suppress_proactive:
            _do_live_vision_comment(snapshot)


def _room_looks_empty(snapshot: dict) -> bool:
    people = snapshot.get("people", []) or []
    crowd = snapshot.get("crowd", {}) or {}
    try:
        crowd_count = int(crowd.get("count", len(people)) or 0)
    except (TypeError, ValueError):
        crowd_count = len(people)
    return not people and crowd_count <= 0


def _empty_room_commentary_allowed(snapshot: dict, now: Optional[float] = None) -> bool:
    if not _room_looks_empty(snapshot):
        return False
    now = time.monotonic() if now is None else now
    if _within_startup_group_window(now) and not _startup_presence_gate_ready(now):
        return False
    return True


def _step_startup_empty_room_comment(snapshot: dict, profile: SituationProfile) -> None:
    """Fire one startup-only no-confirmed-presence line after a fair scan."""
    global _startup_empty_room_seen_at, _startup_empty_room_fired

    if _startup_empty_room_fired:
        return
    if not bool(getattr(config, "STARTUP_EMPTY_ROOM_COMMENT_ENABLED", True)):
        return

    now = time.monotonic()
    if not _within_startup_group_window(now):
        return
    if _greeted_this_session or _startup_known_greeting_pending(snapshot, now=now):
        return
    if profile.user_mid_sentence or profile.interaction_busy:
        return
    if profile.suppress_proactive or profile.suppress_system_comments:
        return

    if not _empty_room_commentary_allowed(snapshot, now):
        _startup_empty_room_seen_at = 0.0
        return

    if _startup_empty_room_seen_at <= 0.0:
        _startup_empty_room_seen_at = now
        return

    confirm = float(getattr(config, "STARTUP_EMPTY_ROOM_CONFIRM_SECS", 5.0))
    if (now - _startup_empty_room_seen_at) < max(0.0, confirm):
        return

    pool = (
        getattr(config, "STARTUP_EMPTY_ROOM_JOKES", None)
        or getattr(config, "EMPTY_ROOM_JOKES", None)
        or []
    )
    if not pool:
        return

    token = _claim_proactive_purpose(
        "startup_empty_room",
        label="startup empty-room joke",
    )
    if token is None:
        return
    try:
        if not _proactive_purpose_current(token):
            return
        line = random.choice(list(pool))
        if _speak_async(
            line,
            emotion="curious",
            purpose="startup_empty_room",
            label="startup empty-room joke",
        ):
            _startup_empty_room_fired = True
    finally:
        _release_proactive_purpose(token)


def _idle_micro_behavior_choices(snapshot: dict) -> tuple[list[str], list[int]]:
    people = snapshot.get("people", []) or []
    if _room_looks_empty(snapshot):
        if not _empty_room_commentary_allowed(snapshot):
            return (
                [
                    "ambient_scan",
                    "private_thought",
                    "aspiration",
                    "idle_clip",
                ],
                [3, 1, 1, 1],
            )
        return (
            [
                "empty_room_joke",
                "private_thought",
                "aspiration",
                "ambient_scan",
                "idle_clip",
                "ambient_observation",
                "live_vision_comment",
            ],
            [6, 2, 1, 1, 1, 1, 1],
        )
    if people:
        return (
            [
                "people_roast",
                "appearance_riff",
                "small_talk_question",
                "ambient_observation",
                "live_vision_comment",
                "ambient_scan",
                "private_thought",
                "aspiration",
                "idle_clip",
            ],
            [4, 3, 2, 1, 1, 1, 1, 1, 1],
        )
    return (
        [
            "small_talk_question",
            "empty_room_joke",
            "ambient_scan",
            "private_thought",
            "aspiration",
            "idle_clip",
            "ambient_observation",
            "live_vision_comment",
        ],
        [2, 3, 1, 1, 1, 1, 1, 1],
    )


def _do_empty_room_joke(snapshot: dict) -> None:
    if not _can_proactive_speak():
        return
    if not _empty_room_commentary_allowed(snapshot):
        return
    if random.random() >= float(getattr(config, "EMPTY_ROOM_JOKE_PROBABILITY", 0.9)):
        return
    pool = getattr(config, "EMPTY_ROOM_JOKES", None) or getattr(config, "PRIVATE_THOUGHTS", [])
    if not pool:
        return
    token = _claim_proactive_purpose("idle_monologue", label="empty-room joke")
    if token is None:
        return
    line = random.choice(list(pool))
    try:
        if _proactive_purpose_current(token):
            try:
                from intelligence import performance_output
                from sequences import animations
                performance_output.execute_body_beat_event(
                    "idle.empty_room",
                    play_body_beat=animations.play_body_beat,
                )
            except Exception as exc:
                _log.debug("empty-room body beat skipped: %s", exc)
            _speak_async(
                line,
                emotion="neutral",
                purpose="idle_monologue",
                label="empty-room joke",
            )
    finally:
        _release_proactive_purpose(token)


def _do_ambient_scan() -> None:
    try:
        from hardware.servos import set_servo
        neck_cfg = config.SERVO_CHANNELS["neck"]
        ch = neck_cfg["ch"]
        neutral = neck_cfg["neutral"]
        left_pos  = int(neutral - (neutral - neck_cfg["min"]) * 0.35)
        right_pos = int(neutral + (neck_cfg["max"] - neutral) * 0.35)

        def _scan():
            set_servo(ch, left_pos)
            time.sleep(1.5)
            set_servo(ch, right_pos)
            time.sleep(1.5)
            set_servo(ch, neutral)

        threading.Thread(target=_scan, daemon=True, name="ambient_scan").start()
    except Exception as exc:
        _log.debug("ambient scan error: %s", exc)


def _get_or_detect_mood(person_id: int) -> Optional[dict]:
    """
    Return a recent mood reading for person_id, calling GPT-4o vision if the
    cached reading is stale. Returns dict {mood, confidence, notes} or None.
    """
    cooldown = float(getattr(config, "MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS", 180.0))
    now = time.monotonic()
    cached = _mood_cache.get(person_id)
    if cached and (now - cached[1]) < cooldown:
        return cached[0]
    try:
        from vision import camera as _cam
        from vision import face as face_mod
        frame = _cam.get_frame()
        if frame is None:
            return None
        mood = face_mod.detect_mood(frame)
        if mood:
            _mood_cache[person_id] = (mood, now)
            return mood
    except Exception as exc:
        _log.debug("mood detect error: %s", exc)
    return None


def get_cached_mood(person_id: Optional[int], max_age_secs: Optional[float] = None) -> Optional[dict]:
    """Return a recent face-mood reading without making a new vision call."""
    if not isinstance(person_id, int):
        return None
    try:
        cooldown = float(
            max_age_secs
            if max_age_secs is not None
            else getattr(config, "MOOD_ANALYSIS_PER_PERSON_COOLDOWN_SECS", 180.0)
        )
        cached = _mood_cache.get(person_id)
        if not cached:
            return None
        mood, ts = cached
        if (time.monotonic() - ts) <= cooldown:
            return dict(mood)
    except Exception as exc:
        _log.debug("cached mood lookup error: %s", exc)
    return None


def _step_gui_mood_telemetry(snapshot: dict, frame) -> None:
    """Refresh cached face mood for the GUI without making Qt own vision calls."""
    global _gui_mood_refresh_in_flight

    if not bool(getattr(config, "MOOD_ANALYSIS_GUI_TELEMETRY_ENABLED", True)):
        return
    if bool(getattr(config, "FACE_EXPRESSION_LOCAL_ENABLED", True)):
        return
    if frame is None or _gui_mood_refresh_in_flight:
        return

    people = [
        p
        for p in (snapshot.get("people") or [])
        if isinstance(p, dict)
        and p.get("person_db_id") is not None
        and p.get("face_visible") is not False
        and not p.get("face_missing")
        and (p.get("face_box") or p.get("bounding_box") or p.get("bbox") or p.get("box"))
    ]
    # detect_mood reads the most prominent face in the whole frame, so only use
    # it as telemetry when there is one unambiguous visible known face.
    if len(people) != 1:
        return

    try:
        person_id = int(people[0].get("person_db_id"))
    except (TypeError, ValueError):
        return

    refresh_secs = float(getattr(config, "MOOD_ANALYSIS_GUI_REFRESH_SECS", 20.0) or 20.0)
    now = time.monotonic()
    cached = _mood_cache.get(person_id)
    if cached and (now - cached[1]) < refresh_secs:
        return

    frame_copy = frame.copy() if hasattr(frame, "copy") else frame
    _gui_mood_refresh_in_flight = True

    def _task() -> None:
        global _gui_mood_refresh_in_flight
        try:
            from vision import face as face_mod

            mood = face_mod.detect_mood(frame_copy)
            if not mood:
                return
            _mood_cache[person_id] = (dict(mood), time.monotonic())
            _write_face_mood_to_world_state(person_id, mood)
            _log.info(
                "consciousness: GUI face mood refreshed for person_id=%s mood=%s confidence=%.2f",
                person_id,
                mood.get("mood"),
                float(mood.get("confidence") or 0.0),
            )
        except Exception as exc:
            _log.debug("GUI mood telemetry error: %s", exc)
        finally:
            _gui_mood_refresh_in_flight = False

    threading.Thread(target=_task, daemon=True, name="gui-mood-telemetry").start()


def _write_face_mood_to_world_state(person_id: int, mood: dict) -> None:
    try:
        def _apply_mood(people):
            changed = False
            for idx, person in enumerate(people):
                if not isinstance(person, dict):
                    continue
                try:
                    current_id = int(person.get("person_db_id"))
                except (TypeError, ValueError):
                    continue
                if current_id != int(person_id):
                    continue
                updated = dict(person)
                updated["face_mood"] = dict(mood)
                expression = str(updated.get("expression") or "").strip().lower()
                if expression in {"", "neutral", "unknown", "none"}:
                    updated["expression"] = mood.get("mood") or "neutral"
                people[idx] = updated
                changed = True
            return people if changed else None

        world_state.mutate("people", _apply_mood)
    except Exception as exc:
        _log.debug("face mood world_state update failed: %s", exc)


def _mood_clause_for(mood: Optional[dict]) -> tuple[str, str]:
    """
    Translate a mood reading into (prompt_clause, emotion) for small-talk.
    Returns ("", "curious") for neutral / low-confidence / missing reads.
    """
    if not mood:
        return "", "curious"
    label = (mood.get("mood") or "").lower()
    confidence = float(mood.get("confidence") or 0.0)
    notes = (mood.get("notes") or "").strip()
    if not label or label == "neutral" or confidence < 0.5:
        return "", "curious"

    notes_clause = f" (you notice: {notes})" if notes else ""
    cues = {
        "happy":     ("they look genuinely happy",
                      "ask what's got them in such a good mood today",
                      "curious"),
        "sad":       ("they look down — a little sad",
                      "gently ask what's got them down, what's on their mind",
                      "concerned"),
        "tired":     ("they look tired, maybe wiped out",
                      "ask if they got any sleep, or what's been wearing them out",
                      "concerned"),
        "angry":     ("they look frustrated or annoyed",
                      "carefully ask what's bugging them",
                      "concerned"),
        "anxious":   ("they look tense, on edge",
                      "ask what's weighing on them, what they're worrying about",
                      "concerned"),
        "surprised": ("they look surprised or wide-eyed",
                      "ask what just happened, what's the look for",
                      "curious"),
    }
    if label not in cues:
        return "", "curious"
    observation, ask, emotion = cues[label]
    clause = (
        f" Looking at their face right now, {observation}{notes_clause}. "
        f"Acknowledge what you see and {ask}. Make the question specifically "
        f"match their expression — not a generic small-talk opener."
    )
    return clause, emotion


def _first_sight_mood_confidence_floor() -> float:
    return float(getattr(config, "MOOD_AWARE_FIRST_SIGHT_CONFIDENCE", 0.65))


def _first_sight_mood_enabled() -> bool:
    return bool(getattr(config, "MOOD_AWARE_FIRST_SIGHT_ENABLED", True))


def _build_first_sight_mood_prompt(
    first_name: str,
    context_sentence: str,
    mood: Optional[dict],
) -> Optional[tuple[str, str]]:
    if not _first_sight_mood_enabled() or not mood:
        return None
    label = (mood.get("mood") or "").lower()
    try:
        confidence = float(mood.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < _first_sight_mood_confidence_floor():
        return None
    notes = (mood.get("notes") or "").strip()
    notes_clause = f" Expression note: {notes}." if notes else ""
    cues = {
        "happy": (
            "they appear to be smiling or in a good mood",
            "ask what's got them smiling today",
            "happy",
        ),
        "sad": (
            "they appear a little down or frowning",
            "gently ask what's got them down or what's on their mind",
            "concerned",
        ),
        "tired": (
            "they appear tired or worn out",
            "ask if sleep betrayed them or what's been wearing them out",
            "concerned",
        ),
        "angry": (
            "they appear frustrated or annoyed",
            "carefully ask what's bugging them",
            "concerned",
        ),
        "anxious": (
            "they appear tense or worried",
            "ask what's weighing on them",
            "concerned",
        ),
        "surprised": (
            "they appear surprised or wide-eyed",
            "ask what just happened",
            "curious",
        ),
    }
    if label not in cues:
        return None
    observation, ask, emotion = cues[label]
    prompt = (
        f"{context_sentence} You have a fresh low-confidence visual read of "
        f"{first_name}'s expression: {observation}.{notes_clause} "
        f"Greet {first_name} in-character by name, then {ask}. Phrase it as an "
        f"apparent read, not a diagnosis; do not say you analyzed an image or "
        f"mention OpenAI. Warm, dry Rex voice. Two short sentences max — the "
        f"second must end in a question mark."
    )
    return prompt, emotion


def _get_first_sight_mood(person_id: Optional[int]) -> Optional[dict]:
    if not isinstance(person_id, int) or not _first_sight_mood_enabled():
        return None
    return _get_or_detect_mood(person_id)


def _build_group_smile_startup_prompt(scene, moods: list[dict]) -> Optional[str]:
    if not _first_sight_mood_enabled():
        return None
    max_people = int(getattr(config, "MOOD_AWARE_FIRST_SIGHT_MAX_PEOPLE", 2) or 2)
    if len(scene.known) != 2 or scene.unknown_count > 0 or scene.crowd_count > max_people:
        return None
    if len(moods) < 2:
        return None
    floor = _first_sight_mood_confidence_floor()
    for mood in moods[:2]:
        label = (mood.get("mood") or "").lower()
        try:
            confidence = float(mood.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if label != "happy" or confidence < floor:
            return None
    from intelligence import social_scene

    label = social_scene.visible_group_label(scene)
    names = ", ".join(scene.first_names)
    notes = "; ".join(
        (m.get("notes") or "").strip()
        for m in moods[:2]
        if (m.get("notes") or "").strip()
    )
    notes_clause = f" Expression notes: {notes}." if notes else ""
    return (
        f"You just started up and can see these two known people together: {names}. "
        f"The natural group label is: {label}. They both appear to be smiling or "
        f"in a good mood.{notes_clause} Greet them as a group in one short "
        f"in-character Rex line, then ask what's got them both smiling today. "
        f"Phrase it as an apparent read, not certainty. Do not mention cameras, "
        f"OpenAI, or image analysis. Two short sentences max — the second must "
        f"end in a question mark."
    )


def _detect_group_startup_moods() -> list[dict]:
    try:
        from vision import camera as _cam
        from vision import face as face_mod

        frame = _cam.get_frame()
        if frame is None:
            return []
        return face_mod.detect_group_moods(
            frame,
            max_people=int(getattr(config, "MOOD_AWARE_FIRST_SIGHT_MAX_PEOPLE", 2) or 2),
        )
    except Exception as exc:
        _log.debug("group startup mood detect error: %s", exc)
        return []


def _do_small_talk_question(snapshot: dict) -> None:
    """
    When the user has gone quiet, initiate small talk by asking them a question.
    Prefers asking a known visible person; falls back to an open question.
    When a known person is in frame, optionally does a GPT-4o mood read of
    their face and tailors the question to what Rex sees.
    """
    if not _can_proactive_speak():
        return

    people = snapshot.get("people", []) or []
    known = [p for p in people if p.get("person_db_id") and p.get("face_id")]
    target_name: Optional[str] = None
    target_db_id: Optional[int] = None
    if known:
        target = random.choice(known)
        if is_engaged_with(target.get("person_db_id")):
            # Mid-conversation — let interaction handle turn-taking.
            return
        target_name = _first_name(target.get("face_id"), "")
        target_db_id = target.get("person_db_id")

    time_of_day = (snapshot.get("time", {}) or {}).get("time_of_day") or ""
    venue = getattr(config, "VENUE_NAME", "")

    # Prefer asking about a known plan (past or upcoming) over a generic question.
    plan_clause = ""
    if target_db_id is not None:
        try:
            from memory import events as events_mod
            pending = events_mod.get_pending_followups(target_db_id) or []
            if pending:
                ev = pending[0]
                ev_name = ev.get("event_name") or ""
                if ev_name:
                    _pending_followups_lock_remove(target_db_id, ev.get("id"))
                    plan_clause = (
                        f" You remember they told you they had this coming up: "
                        f"'{ev_name}'. Specifically ask how it went."
                    )
            if not plan_clause:
                upcoming = events_mod.get_upcoming_events(target_db_id) or []
                if upcoming:
                    ev = upcoming[0]
                    ev_name = ev.get("event_name") or ""
                    ev_date = ev.get("event_date") or ""
                    if ev_name:
                        when = f" on {ev_date}" if ev_date else ""
                        plan_clause = (
                            f" You remember they mentioned '{ev_name}'{when} is "
                            f"coming up. You can ask how they're feeling about it "
                            f"or whether they're ready."
                        )
        except Exception as exc:
            _log.debug("smalltalk plan lookup error: %s", exc)

    do_mood = (
        target_db_id is not None
        and getattr(config, "MOOD_AWARE_SMALLTALK_ENABLED", True)
        and not plan_clause   # don't override a fresh follow-up with a mood riff
        and random.random() < float(getattr(config, "MOOD_ANALYSIS_PROBABILITY", 0.7))
    )
    purpose = "memory_followup" if plan_clause else "small_talk"
    candidate_id = _observe_governor_candidate(
        purpose=purpose,
        label="small-talk question",
        prompt=(
            "Small-talk candidate: choose a known visible person if available, "
            "optionally use mood or plan context, then ask one short question."
        ),
        emotion="curious",
        target_person_id=target_db_id,
        requires_llm=True,
    )
    token = _claim_proactive_purpose(purpose, label="small-talk question")
    if token is None:
        _mark_governor_candidate(
            candidate_id,
            "dropped",
            "conversation_agenda_claim_rejected",
        )
        return
    _mark_governor_candidate(candidate_id, "accepted", "current_behavior_queued_llm")

    def _task() -> None:
        try:
            if not _proactive_purpose_current(token):
                return
            mood_clause = ""
            emotion = "curious"
            if do_mood:
                mood = _get_or_detect_mood(target_db_id)
                mood_clause, emotion = _mood_clause_for(mood)

            if target_name:
                prompt = (
                    f"It's quiet and you're idly looking at '{target_name}', someone you know. "
                    f"They haven't said anything in a while.{plan_clause}{mood_clause} "
                    f"Open small talk by asking them one short, in-character Rex question. "
                    f"Lead with genuine curiosity about who they are — ask how they're doing, "
                    f"about a hobby or interest of theirs, their taste in music or movies, what "
                    f"they've been into or thinking about lately, or what they're passionate "
                    f"about. If a cue above gives you something specific (a plan, their mood, a "
                    f"known interest), you may ask about that instead — but don't default to "
                    f"interrogating them about their schedule. Warm but dry. Don't lecture, "
                    f"don't give your opinion — just ask. Address {target_name} by name. One "
                    f"short sentence ending in a question mark."
                )
            else:
                ctx_bits = []
                if time_of_day:
                    ctx_bits.append(f"part of day: {time_of_day}")
                if venue:
                    ctx_bits.append(f"venue: {venue}")
                ctx = "; ".join(ctx_bits) or "no extra context"
                prompt = (
                    f"It's quiet around you and nobody has said anything in a while ({ctx}). "
                    f"Break the silence by asking the room one short, in-character Rex small-talk "
                    f"question — something open-ended that invites whoever is listening to "
                    f"answer. Don't lecture, don't give your opinion — just ask. One short "
                    f"sentence ending in a question mark."
                )

            if not _can_proactive_speak():
                return
            prompt = _apply_proactive_directive(prompt, purpose)
            from intelligence.llm import get_response
            text = get_response(prompt, target_db_id)
            if text and _proactive_purpose_current(token):
                _speak_async(text, emotion=emotion, governed=False)
        except Exception as exc:
            _log.debug("_do_small_talk_question task error: %s", exc)
        finally:
            _release_proactive_purpose(token)

    threading.Thread(target=_task, daemon=True, name="small-talk-question").start()


def _do_private_thought() -> None:
    if not _can_proactive_speak():
        return
    token = _claim_proactive_purpose("idle_monologue", label="private thought")
    if token is None:
        return
    line = random.choice(config.PRIVATE_THOUGHTS)
    try:
        if _proactive_purpose_current(token):
            _speak_async(
                line,
                emotion="neutral",
                purpose="idle_monologue",
                label="private thought",
            )
    finally:
        _release_proactive_purpose(token)


# Anti-repeat for aspirations — never play the same line back-to-back.
_last_aspiration: Optional[str] = None


def _do_aspiration() -> None:
    """Speak one of Rex's forward-looking aspirations as an idle micro-behavior."""
    global _last_aspiration
    if not _can_proactive_speak():
        return
    pool = getattr(config, "ASPIRATIONS", None)
    if not pool:
        return
    token = _claim_proactive_purpose("idle_monologue", label="aspiration")
    if token is None:
        return
    candidates = [line for line in pool if line != _last_aspiration] or list(pool)
    chosen = random.choice(candidates)
    _last_aspiration = chosen
    try:
        if _proactive_purpose_current(token):
            _speak_async(
                chosen,
                emotion="curious",
                purpose="idle_monologue",
                label="aspiration",
            )
    finally:
        _release_proactive_purpose(token)


def _do_ambient_observation(snapshot: dict) -> None:
    """
    Fire a short in-character remark about the current environment, pulled from
    world_state.environment — room type, lighting, crowd density, description.
    No vision call; uses data the periodic scene scanner already collected.
    """
    if random.random() >= getattr(config, "AMBIENT_OBSERVATION_PROBABILITY", 0.5):
        return
    env = snapshot.get("environment", {}) or {}
    audio_scene = snapshot.get("audio_scene", {}) or {}

    bits: list[str] = []
    if env.get("description"):
        bits.append(f"scene: {env['description']}")
    elif env.get("scene_type"):
        bits.append(f"scene type: {env['scene_type']}")
    if env.get("lighting"):
        bits.append(f"lighting: {env['lighting']}")
    if env.get("crowd_density"):
        bits.append(f"crowd density: {env['crowd_density']}")
    if audio_scene.get("ambient_level"):
        bits.append(f"ambient noise: {audio_scene['ambient_level']}")
    if audio_scene.get("music_detected"):
        bits.append("music is playing")

    if not bits:
        return
    context = "; ".join(bits)
    _generate_and_speak(
        f"You are idly observing your surroundings right now. Here is what you perceive "
        f"— {context}. In one short in-character Rex line, make an offhand observation "
        f"about the room or environment — like someone thinking out loud. Don't greet "
        f"anyone, don't ask a question; just a dry remark about the space or vibe. "
        f"One line only.",
        emotion="neutral",
        purpose="ambient_observation",
    )


def _do_appearance_riff(snapshot: dict) -> None:
    """
    Pick one currently-visible known person and make an unprompted remark about
    their appearance (hair, clothes, notable features), using stored person_facts.
    No vision call; uses data from face enrollment.
    """
    people = snapshot.get("people", []) or []
    known = [p for p in people if p.get("person_db_id") and p.get("face_id")]
    if not known:
        return
    target = random.choice(known)
    hint = _pick_appearance_hint(target.get("person_db_id"))
    if not hint:
        return
    try:
        from memory import boundaries as _boundaries
        target_id = target.get("person_db_id")
        if (
            _boundaries.is_blocked(target_id, "mention", "appearance")
            or _boundaries.is_blocked(target_id, "roast", "appearance")
            or _boundaries.is_blocked(target_id, "mention", "clothing")
            or _boundaries.is_blocked(target_id, "roast", "clothing")
        ):
            return
    except Exception:
        pass
    # Don't riff on the engaged person — it'd feel interruptive mid-conversation.
    if is_engaged_with(target.get("person_db_id")):
        return
    first_name = _first_name(target.get("face_id"), "there")
    _generate_and_speak(
        f"You're idly looking at '{first_name}'. You remember this about their "
        f"appearance: {hint}. Make one short in-character Rex remark about it — "
        f"the kind of thing you'd say while looking them over. Warm, dry, observational, "
        f"and lightly funny if the opening is there. "
        f"Address {first_name} by name. One line only.",
        emotion="neutral",
        purpose="appearance_riff",
    )


def _person_roast_allowed(person: dict) -> bool:
    age = (person.get("age_estimate") or person.get("age_category") or "").lower()
    if age in {"child", "teen", "minor"}:
        return False
    target_id = person.get("person_db_id")
    if target_id is None:
        return True
    try:
        from memory import boundaries as _boundaries
        if (
            _boundaries.is_blocked(target_id, "roast", "anything")
            or _boundaries.is_blocked(target_id, "mention", "anything")
            or _boundaries.is_blocked(target_id, "roast", "appearance")
            or _boundaries.is_blocked(target_id, "roast", "body")
            or _boundaries.is_blocked(target_id, "roast", "identity")
        ):
            return False
    except Exception:
        pass
    return True


def _person_roast_cues(person: dict) -> str:
    cues = []
    for key, label in (
        ("distance_zone", "distance"),
        ("approach_vector", "movement"),
        ("pose", "pose"),
        ("gesture", "gesture"),
        ("engagement", "engagement"),
        ("position", "position"),
    ):
        value = person.get(key)
        if value and value != "neutral":
            cues.append(f"{label}={value}")
    return ", ".join(cues) or "quietly present, saying nothing"


def _do_people_roast(snapshot: dict) -> None:
    if not _can_proactive_speak():
        return
    if random.random() >= float(getattr(config, "PEOPLE_ROAST_RIFF_PROBABILITY", 0.75)):
        return
    people = snapshot.get("people", []) or []
    candidates = [
        person for person in people
        if not is_engaged_with(person.get("person_db_id"))
        and _person_roast_allowed(person)
    ]
    if not candidates:
        return
    target = random.choice(candidates)
    first_name = _first_name(target.get("face_id"), "there")
    label = first_name or "the unidentified organic in frame"
    cues = _person_roast_cues(target)
    family_clause = (
        "Keep it extra gentle and family-safe because a younger person may be present. "
        if any(
            (p.get("age_estimate") or p.get("age_category") or "").lower() in {"child", "teen", "minor"}
            for p in people
        )
        else ""
    )
    _generate_and_speak(
        f"You're idle, nobody has spoken for a bit, and you're looking at {label}. "
        f"Live non-sensitive cues: {cues}. {family_clause}"
        "Make one short playful Rex joke or light roast about their current vibe, "
        "silence, posture, indecision, or general organic energy. Keep it affectionate "
        "and non-sensitive. Do NOT joke about body, age, gender, race, religion, "
        "disability, health, money, identity, grief, private text, or anything intimate. "
        "Do not ask a question. "
        f"{'Address ' + first_name + ' by name. ' if first_name else ''}"
        "One line only.",
        emotion="curious",
        purpose="people_roast",
        label="idle people roast",
    )


def _do_live_vision_comment(snapshot: dict) -> None:
    """
    Capture the current frame and ask GPT-4o for one short observational detail
    about it — a spontaneous remark on something Rex is literally seeing right now.

    Rate-limited by LIVE_VISION_COMMENT_COOLDOWN_SECS so it stays costed.
    """
    global _last_live_vision_comment_at
    now = time.monotonic()
    cooldown = getattr(config, "LIVE_VISION_COMMENT_COOLDOWN_SECS", 300.0)
    if (now - _last_live_vision_comment_at) < cooldown:
        return
    _last_live_vision_comment_at = now

    def _task():
        try:
            if not _can_proactive_speak():
                return
            from vision import camera as _cam
            from vision import scene as _scene
            frame = _cam.get_frame()
            if frame is None:
                return
            # Reuse the describe_scene path for a fresh, low-detail summary. This
            # triggers analyze_environment(force=True) which hits GPT-4o once.
            desc = _scene.describe_scene()
            if not desc:
                return
            _generate_and_speak(
                f"You just glanced around and actually LOOKED at what's in front of you "
                f"right now. Vision summary: '{desc}'. In one short in-character Rex line, "
                f"make a spontaneous remark about one concrete detail you 'see' — not a "
                f"greeting, not a question, just a passing observation as if thinking out "
                f"loud. One line only.",
                emotion="curious",
                purpose="visual_curiosity",
            )
        except Exception as exc:
            _log.debug("live vision comment error: %s", exc)

    threading.Thread(target=_task, daemon=True, name="live-vision-comment").start()


def _visual_curiosity_blocked_by_empathy(person_id: Optional[int]) -> bool:
    """
    Avoid visual riffs during tender emotional modes. Rex can be observant and
    snarky later; right after grief or distress, curiosity should stay relational.
    """
    try:
        from intelligence import empathy as _empathy
        entry = _empathy.peek(person_id)
    except Exception:
        return False
    if not entry:
        return False

    result = entry.get("result") or {}
    mode_pack = entry.get("mode") or {}
    mode = (mode_pack.get("mode") or "default").lower()
    sensitivity = (result.get("topic_sensitivity") or "none").lower()
    affect = (result.get("affect") or "neutral").lower()
    confidence = float(result.get("confidence", 0.5) or 0.5)

    tender_modes = {
        "listen",
        "support",
        "acknowledge_then_yield",
        "ground",
        "course_correct",
        "crisis",
        "validate",
        "gentle_probe",
        "kind_default",
    }
    if mode in tender_modes:
        return True
    if sensitivity in {"heavy", "crisis"}:
        return True
    if confidence >= 0.55:
        try:
            return _empathy.is_negative_affect(affect)
        except Exception:
            return affect in {"sad", "anxious", "angry", "tired"}
    return False


def _visual_curiosity_recently_blocked_by_checkin(person_id: Optional[int]) -> bool:
    """
    Keep visual curiosity quiet briefly after a care move, but don't suppress it
    for the rest of the session. Active grief/distress is still handled by
    _visual_curiosity_blocked_by_empathy().
    """
    if person_id is None:
        return False
    fired_at = _emotional_checkin_fired_at.get(person_id)
    if not fired_at:
        return False
    cooldown = float(getattr(config, "VISUAL_CURIOSITY_AFTER_EMPATHY_COOLDOWN_SECS", 90.0))
    return (time.monotonic() - fired_at) < max(0.0, cooldown)


def _note_emotional_checkin_fired(person_id: Optional[int]) -> None:
    if person_id is None:
        return
    _emotional_checkin_fired.add(person_id)
    _emotional_checkin_fired_at[person_id] = time.monotonic()


def note_emotional_checkin_boundary(
    person_id: Optional[int],
    *,
    window_secs: Optional[float] = None,
) -> bool:
    """
    Called when the person closes the door on a recent empathy check-in.

    Keep the per-session check-in dedupe intact so Rex doesn't ask again, but
    clear the post-care visual curiosity hold. Once someone says "don't talk
    about that," a neutral visual pivot is allowed if the normal curiosity gates
    later decide the silence needs one.
    """
    if person_id is None:
        return False
    fired_at = _emotional_checkin_fired_at.get(person_id)
    if not fired_at:
        return False

    if window_secs is None:
        minutes = float(getattr(config, "EMOTIONAL_CHECKIN_BOUNDARY_WINDOW_MINUTES", 20.0))
        window_secs = max(0.0, minutes * 60.0)
    if window_secs and (time.monotonic() - fired_at) > window_secs:
        return False

    _emotional_checkin_fired_at.pop(person_id, None)
    _negative_streak_started_at.pop(person_id, None)
    _log.info(
        "consciousness: released post-empathy visual curiosity hold for person_id=%s",
        person_id,
    )
    return True


def _step_visual_curiosity(snapshot: dict, profile: SituationProfile) -> None:
    """
    After a recent engaged back-and-forth goes quiet, use a fresh visual summary
    to ask one concrete question about something Rex can see right now.

    This fills the "human stopped talking" gap more naturally than generic
    small talk, but it is heavily gated because it costs a vision call and
    should never interrupt an answer, an empathy flow, or another response.
    """
    global _last_visual_curiosity_at, _visual_curiosity_in_flight

    if not getattr(config, "VISUAL_CURIOSITY_ENABLED", True):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    min_silence = float(getattr(config, "VISUAL_CURIOSITY_SILENCE_SECS", 8.0))
    active_window = float(getattr(config, "VISUAL_CURIOSITY_ACTIVE_WINDOW_SECS", 45.0))
    global_cooldown = float(getattr(config, "VISUAL_CURIOSITY_COOLDOWN_SECS", 300.0))
    person_cooldown = float(getattr(config, "VISUAL_CURIOSITY_PERSON_COOLDOWN_SECS", 600.0))

    if (now - _last_visual_curiosity_at) < global_cooldown:
        return

    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    if _visual_curiosity_blocked_by_interest_thread(engaged_id):
        return
    if _visual_curiosity_recently_blocked_by_checkin(engaged_id):
        return

    quiet_for = now - engaged_touch
    if quiet_for < min_silence or quiet_for > active_window:
        return
    if (now - _visual_curiosity_by_person.get(engaged_id, 0.0)) < person_cooldown:
        return

    try:
        turn_window = float(getattr(config, "VISUAL_CURIOSITY_TURN_WINDOW_SECS", 45.0))
        min_turns = int(getattr(config, "VISUAL_CURIOSITY_MIN_USER_TURNS", 2))
        if _situation_assessor.recent_speech_turn_count(turn_window) < min_turns:
            return
    except Exception:
        if not profile.rapid_exchange:
            return

    max_crowd = int(getattr(config, "VISUAL_CURIOSITY_MAX_CROWD_COUNT", 2))
    crowd_count = int((snapshot.get("crowd") or {}).get("count", 1) or 1)
    if crowd_count > max_crowd:
        return

    if _visual_curiosity_blocked_by_empathy(engaged_id):
        return

    candidate_id = _observe_governor_candidate(
        purpose="visual_curiosity",
        label=f"visual curiosity for {engaged_id}",
        prompt=(
            "Visual curiosity candidate: take a fresh visual snapshot after a "
            "mid-conversation lull and ask one grounded question."
        ),
        emotion="curious",
        wait_secs=float(getattr(config, "QUESTION_RESPONSE_WAIT_SECS", 7.0)),
        target_person_id=engaged_id,
        requires_llm=True,
    )
    token = _claim_proactive_purpose(
        "visual_curiosity",
        label=f"visual curiosity for {engaged_id}",
    )
    if token is None:
        _mark_governor_candidate(
            candidate_id,
            "dropped",
            "conversation_agenda_claim_rejected",
        )
        return
    _mark_governor_candidate(candidate_id, "accepted", "current_behavior_queued_llm")

    with _visual_curiosity_lock:
        if _visual_curiosity_in_flight:
            _mark_governor_candidate(candidate_id, "dropped", "visual_curiosity_in_flight")
            _release_proactive_purpose(token)
            return
        if (time.monotonic() - _last_visual_curiosity_at) < global_cooldown:
            _mark_governor_candidate(candidate_id, "dropped", "visual_curiosity_global_cooldown")
            _release_proactive_purpose(token)
            return
        _visual_curiosity_in_flight = True
        _last_visual_curiosity_at = time.monotonic()
        _visual_curiosity_by_person[engaged_id] = _last_visual_curiosity_at

    def _task() -> None:
        global _visual_curiosity_in_flight
        try:
            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return
            from memory import people as people_mod
            from vision import camera as _cam
            from vision import scene as _scene
            from intelligence.llm import get_response

            person = people_mod.get_person(engaged_id) or {}
            first_name = _first_name(person.get("name"), "there")

            frame = _cam.get_frame()
            if frame is None:
                return
            visual = _scene.describe_scene_detailed(frame)
            if not visual:
                return
            if not _proactive_purpose_current(token):
                return
            if not _can_proactive_speak():
                return

            visual_json = json.dumps(visual, ensure_ascii=False)[:3500]
            family_clause = (
                "A child or teen is present, so keep it gentle and family-safe. "
                if profile.force_family_safe else ""
            )
            prompt = (
                f"You're mid-conversation with {first_name}, and they just went "
                f"quiet for a few seconds after a back-and-forth. You took a fresh "
                f"visual snapshot. Use it as a conversational springboard.\n\n"
                f"Vision summary JSON: {visual_json}\n\n"
                f"{family_clause}"
                "Ask exactly ONE short, in-character Rex question grounded in a "
                "specific visible, non-sensitive detail. It can be dry or mildly "
                "teasing about clothing, accessories, objects, decor, or what they "
                "seem to be doing, but do not roast grief, emotions, body, identity, "
                "health, age, race, religion, politics, disability, money, or private "
                "screen/document text. Do not say you took a picture. Do not explain "
                "the visual system. Address them by name if natural. End with a "
                "question mark."
            )
            prompt = _apply_proactive_directive(prompt, "visual_curiosity")
            text = get_response(prompt, engaged_id)
            if text and _proactive_purpose_current(token) and _can_proactive_speak():
                wait = float(getattr(config, "QUESTION_RESPONSE_WAIT_SECS", 7.0))
                _log.info(
                    "consciousness: visual curiosity question for person_id=%s "
                    "after %.1fs quiet",
                    engaged_id,
                    quiet_for,
                )
                _speak_async(text, emotion="curious", wait_secs=wait, governed=False)
        except Exception as exc:
            _log.debug("visual curiosity step error: %s", exc)
        finally:
            _release_proactive_purpose(token)
            with _visual_curiosity_lock:
                _visual_curiosity_in_flight = False

    threading.Thread(target=_task, daemon=True, name="visual-curiosity").start()


def _visual_curiosity_blocked_by_interest_thread(person_id: Optional[int]) -> bool:
    if person_id is None:
        return False
    try:
        from intelligence import conversation_steering

        steering = conversation_steering.build_context(person_id)
    except Exception as exc:
        _log.debug("visual curiosity interest-thread check failed: %s", exc)
        return False
    if not steering:
        return False
    _log.info(
        "consciousness: visual curiosity suppressed — active interest thread %r "
        "for person_id=%s",
        steering.topic,
        person_id,
    )
    return True


def _do_idle_clip() -> None:
    try:
        token = _claim_proactive_purpose("idle_monologue", label="idle clip")
        if token is None:
            return
        clips_dir = Path(config.AUDIO_CLIPS_DIR)
        clips = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.wav"))
        if not clips:
            _release_proactive_purpose(token)
            return
        clip_path = random.choice(clips)

        def _play():
            try:
                if not _proactive_purpose_current(token):
                    return
                import sounddevice as sd
                import soundfile as sf
                from audio import output_gate

                with output_gate.hold("idle_clip", blocking=False) as acquired:
                    if not acquired:
                        return
                    data, samplerate = sf.read(str(clip_path), dtype="float32")
                    sd.play(data, samplerate)
                    sd.wait()
            except Exception as exc:
                _log.debug("idle clip playback error: %s", exc)
            finally:
                _release_proactive_purpose(token)

        threading.Thread(target=_play, daemon=True, name="idle_clip").start()
    except Exception as exc:
        _log.debug("idle clip error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Presence tracking (departure / return reactions)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_appearance_hint(person_db_id: Optional[int]) -> Optional[str]:
    """Return a single natural-language appearance hint for prompting, or None.

    Reads the person_facts table for category='appearance' and formats one or
    two attributes into a short phrase Rex can riff on.
    """
    if person_db_id is None:
        return None
    try:
        from memory import facts as _facts
        rows = _facts.get_facts_by_category(person_db_id, "appearance")
    except Exception:
        return None
    if not rows:
        return None

    attrs = {r["key"]: r["value"] for r in rows if r.get("key") and r.get("value")}
    candidates: list[str] = []

    notable = attrs.get("notable_features")
    if notable and notable not in ("[]", "None", "none"):
        candidates.append(f"notable features: {notable}")

    hair = []
    if attrs.get("hair_color"):
        hair.append(attrs["hair_color"])
    if attrs.get("hair_style"):
        hair.append(attrs["hair_style"])
    if hair:
        candidates.append(f"{' '.join(hair)} hair")

    if attrs.get("build"):
        candidates.append(f"{attrs['build']} build")

    if not candidates:
        return None
    return random.choice(candidates)


def _tracking_key(person: dict):
    """Stable per-person tracking key: db_id (int) for known, slot id (str) for unknown."""
    db_id = person.get("person_db_id")
    return db_id if db_id is not None else person.get("id", "unknown")


def _person_by_slot(people: list[dict], slot_id: Optional[str]) -> Optional[dict]:
    if not slot_id:
        return None
    for person in people:
        if person.get("id") == slot_id:
            return person
    return None


def _bridge_unknown_presence(person: dict, now: float) -> Optional[tuple[int, Optional[str]]]:
    """
    If a known face temporarily becomes an unknown slot, keep presence tracking
    keyed to the known person. This covers hand/arm occlusions and recognition
    flicker without enrolling or greeting a phantom newcomer.
    """
    if person.get("person_db_id") is not None:
        return None

    bridge_secs = float(getattr(config, "PRESENCE_IDENTITY_BRIDGE_SECS", 12.0))
    if bridge_secs <= 0:
        return None

    slot_id = person.get("id")
    previous = _person_by_slot(_last_snapshot.get("people", []) or [], slot_id)
    candidates: list[tuple[int, Optional[str], float]] = []

    if previous and isinstance(previous.get("person_db_id"), int):
        pid = int(previous["person_db_id"])
        last_seen = _last_seen.get(pid, now)
        candidates.append((pid, previous.get("face_id"), now - last_seen))

    # If there is exactly one visible unknown and one recently visible known
    # person, bridge that too. This catches the people=[] → unknown-slot → known
    # sequence after a brief face cover.
    visible_known = [key for key in _visible_people if isinstance(key, int)]
    current_people = _last_snapshot.get("people", []) or []
    if len(visible_known) == 1 and not current_people:
        pid = visible_known[0]
        last_seen = _last_seen.get(pid, now)
        candidates.append((pid, None, now - last_seen))
    else:
        recent_known = [
            (key, now - seen_at)
            for key, seen_at in _last_seen.items()
            if isinstance(key, int) and (now - seen_at) <= bridge_secs
        ]
        if len(recent_known) == 1:
            pid, missing_for = recent_known[0]
            candidates.append((pid, None, missing_for))

    for pid, name, missing_for in candidates:
        if missing_for <= bridge_secs:
            if not name:
                try:
                    from memory import people as _people_mod
                    row = _people_mod.get_person(pid)
                    name = row.get("name") if row else None
                except Exception:
                    name = None
            _log.debug(
                "consciousness: bridged unknown slot %s to known person %s after %.1fs",
                slot_id,
                pid,
                missing_for,
            )
            return pid, name
    return None


def _presence_tracking_map(snapshot: dict, now: float) -> dict:
    """Build current tracking_key → (name, db_id), bridging brief identity flicker."""
    current_tracked: dict = {}
    people = snapshot.get("people", []) or []
    visible_unknowns = [p for p in people if p.get("person_db_id") is None]

    for person in people:
        bridged = None
        if len(visible_unknowns) == 1:
            bridged = _bridge_unknown_presence(person, now)
        if bridged:
            key = bridged[0]
            current_tracked[key] = bridged
            continue
        key = _tracking_key(person)
        current_tracked[key] = (person.get("face_id"), person.get("person_db_id"))
    return current_tracked


def _departure_confirm_secs_for(
    key,
    person_db_id: Optional[int],
    default_confirm: float,
) -> float:
    """Use a shorter departure confirmation window for the active conversation partner."""
    confirm = max(0.0, float(default_confirm or 0.0))
    candidate_id = person_db_id
    if candidate_id is None and isinstance(key, int):
        candidate_id = key
    if candidate_id is None:
        return confirm
    try:
        if not is_engaged_with(int(candidate_id)):
            return confirm
    except Exception:
        return confirm
    engaged_confirm = float(
        getattr(config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", confirm)
    )
    return max(0.0, min(confirm, engaged_confirm))


def _face_tracking_recently_held_person(person_db_id: Optional[int], now: float) -> bool:
    if person_db_id is None:
        return False
    try:
        if int(_face_tracking_lock.get("person_id")) != int(person_db_id):
            return False
    except Exception:
        return False
    last_seen = float(_face_tracking_lock.get("last_seen_at") or 0.0)
    if last_seen <= 0:
        return False
    grace = max(
        float(getattr(config, "FACE_TRACKING_LOST_HOLD_SECS", 8.0) or 0.0),
        float(getattr(config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", 12.0) or 0.0),
    )
    return (now - last_seen) <= grace


def _step_relationship_inquiry(snapshot: dict, profile: SituationProfile) -> None:
    """
    When Rex is engaged with a known person and an UNKNOWN face has been
    continuously visible for UNKNOWN_WITH_ENGAGED_CONFIRM_SECS, ask the engaged
    person who the stranger is and what their relationship is.

    Sets _pending_relationship_prompt so interaction.py parses the next utterance
    for a {name, relationship} pair.
    """
    global _last_identity_prompt_at, _unknown_first_seen_at

    if not _can_speak():
        return
    if profile.suppress_proactive:
        return
    if _pending_relationship_prompt.is_set():
        return

    audio_scene = snapshot.get("audio_scene", {}) or {}
    try:
        chatter_until = audio_scene.get("group_chatter_until")
        if chatter_until is not None:
            group_chatter = time.time() <= float(chatter_until)
        else:
            group_chatter = bool(audio_scene.get("group_chatter_detected"))
    except (TypeError, ValueError):
        group_chatter = bool(audio_scene.get("group_chatter_detected"))
    if group_chatter:
        _unknown_first_seen_at.clear()
        return

    now = time.monotonic()
    cooldown = getattr(config, "RELATIONSHIP_PROMPT_COOLDOWN_SECS", _RELATIONSHIP_PROMPT_COOLDOWN_SECS)
    if (now - _last_identity_prompt_at) < cooldown:
        # Reuse the identity-prompt cooldown so Rex doesn't spam prompts.
        return

    # Find engaged person — allow RECENT engagement (within window) so we still
    # ask "who's this?" if a newcomer arrives right as a session is winding down.
    engaged_id: Optional[int] = None
    engaged_name: Optional[str] = None
    recent_window = float(getattr(config, "RECENT_ENGAGEMENT_WINDOW_SECS", 60.0))
    with _engaged_lock:
        pid = _engaged_person_id
        touch = _engaged_last_touch_at
    if pid is None or (touch > 0 and (now - touch) > recent_window):
        # No recent engagement; drop all unknown timers to avoid stale state.
        _unknown_first_seen_at.clear()
        return
    engaged_id = pid

    people = snapshot.get("people", []) or []
    known_visible = False
    unknown_slots: list[str] = []
    for p in people:
        pid = p.get("person_db_id")
        slot = p.get("id") or ""
        if pid == engaged_id:
            known_visible = True
            engaged_name = p.get("face_id")
        if pid is None and slot:
            unknown_slots.append(slot)

    if not known_visible or not unknown_slots:
        # No relevant unknowns while engaged — prune timers.
        for slot in list(_unknown_first_seen_at):
            if slot not in unknown_slots:
                _unknown_first_seen_at.pop(slot, None)
        return

    # Track continuous presence per unknown slot while engaged.
    for slot in unknown_slots:
        if slot not in _unknown_first_seen_at:
            _unknown_first_seen_at[slot] = now

    # Find a slot that has persisted long enough and hasn't been asked about yet.
    confirm = getattr(config, "UNKNOWN_WITH_ENGAGED_CONFIRM_SECS", _UNKNOWN_WITH_ENGAGED_CONFIRM_SECS)
    ripe_slot: Optional[str] = None
    for slot in unknown_slots:
        if slot in _asked_relationship_slots:
            continue
        if (now - _unknown_first_seen_at.get(slot, now)) >= confirm:
            ripe_slot = slot
            break
    if ripe_slot is None:
        return

    # Gate on proactive speech — need an open mouth slot.
    if not _can_proactive_speak():
        return

    first_name = _first_name(engaged_name, "friend")
    _last_identity_prompt_at = now
    _pending_relationship_context.clear()
    _pending_relationship_context.update({
        "engaged_person_id": engaged_id,
        "engaged_name": engaged_name,
        "slot_id": ripe_slot,
        "asked_at": now,
    })
    _pending_relationship_prompt.set()
    _log.info(
        "consciousness: asking %s about unknown visitor (slot=%s)",
        engaged_name, ripe_slot,
    )
    if not _generate_and_speak(
        f"You're talking with '{first_name}' and a new unfamiliar face has just "
        f"joined the view. In one short in-character Rex line, ask {first_name} "
        f"who the newcomer is AND what their relationship to {first_name} is — "
        f"e.g. 'Oh hey, who's this, {first_name}? Friend of yours?' Keep it warm "
        f"and curious, one line only, ending with a question mark.",
        emotion="curious",
        wait_secs=getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0),
        purpose="relationship_inquiry",
    ):
        _pending_relationship_prompt.clear()
        _pending_relationship_context.clear()


def _step_presence_tracking(snapshot: dict, profile: SituationProfile) -> None:
    """
    Compare person visibility against the previous tick for both known and unknown people.

    Hysteresis model:
      - A person must be continuously missing for PRESENCE_DEPARTURE_CONFIRM_SECS
        before we even consider them "gone." Single-frame detection flicker is ignored.
        The active conversation partner can use a shorter engaged-person window.
      - Once confirmed gone, we stage a departure in _pending_departure_keys and wait
        for apparent_departure (face-gone + VAD-silent) before speaking.
      - _should_fire_presence() is the single gate for every presence reaction —
        it enforces per-person cooldowns and the "no narrating the person you're
        talking to" rule.
    """
    global _visible_people, _pending_departure_keys, _first_missing_at

    now = time.monotonic()
    departure_cooldown = getattr(config, "PRESENCE_DEPARTURE_COOLDOWN_SECS", 30)
    departure_audio_silence = getattr(config, "DEPARTURE_AUDIO_SILENCE_SECS", 3.0)
    confirm_absent = getattr(config, "PRESENCE_DEPARTURE_CONFIRM_SECS", 8.0)
    return_min_absent = getattr(config, "PRESENCE_RETURN_MIN_ABSENT_SECS", 30)
    unknown_addresses = getattr(config, "UNKNOWN_PERSON_ADDRESSES", ["hey you"])

    # Build current tracked set: tracking_key → (name, db_id), with a short
    # bridge for known faces that momentarily recognize as unknown.
    current_tracked = _presence_tracking_map(snapshot, now)

    current_keys = set(current_tracked.keys())
    known_slot_aliases = {
        str(person.get("id")).strip()
        for person in (snapshot.get("people", []) or [])
        if _person_db_id(person) is not None and str(person.get("id") or "").strip()
    }
    for slot_key in known_slot_aliases:
        if slot_key not in _visible_people:
            continue
        _visible_people.discard(slot_key)
        _first_missing_at.pop(slot_key, None)
        _pending_departure_keys.pop(slot_key, None)
        _confirmed_absent_at.pop(slot_key, None)
        _first_sight_seen_at.pop(slot_key, None)
        _last_seen.pop(slot_key, None)
        _last_presence_reaction_at.pop(slot_key, None)
        _last_departure_reaction_at.pop(slot_key, None)
        _last_return_reaction_at.pop(slot_key, None)
        _log.debug(
            "consciousness: retired unknown presence slot %s after identity bind",
            slot_key,
        )

    # ── Hysteresis: track "first-missing-at" and clear when visible ───────────
    for key in _first_missing_at.keys() - current_keys:
        pass  # still missing; keep the timestamp
    # Start or clear the missing timer
    for key in _visible_people:
        if key in current_keys:
            _first_missing_at.pop(key, None)
        elif key not in _first_missing_at:
            _first_missing_at[key] = now

    # Anyone who reappears clears their timer
    for key in current_keys:
        _first_missing_at.pop(key, None)

    # ── Stage departures once absence exceeds the confirmation window ─────────
    for key, first_missing in list(_first_missing_at.items()):
        if key in _pending_departure_keys:
            continue
        # Capture person info from last snapshot.
        person_name = None
        person_db_id = key if isinstance(key, int) else None
        for p in _last_snapshot.get("people", []):
            if _tracking_key(p) == key:
                person_name = p.get("face_id")
                person_db_id = p.get("person_db_id")
                break
        # If the key is itself a db_id but the slot had lost its name (flicker),
        # look the name up directly so we don't mislabel a known departure as
        # "mystery organic."
        if isinstance(key, int) and not person_name:
            try:
                from memory import people as _people_mod
                row = _people_mod.get_person(key)
                if row and row.get("name"):
                    person_name = row["name"]
                    person_db_id = key
            except Exception:
                pass
        confirm_for_key = _departure_confirm_secs_for(
            key,
            person_db_id,
            confirm_absent,
        )
        if (now - first_missing) < confirm_for_key:
            continue
        _pending_departure_keys[key] = (first_missing, person_name, person_db_id)
        _confirmed_absent_at[key] = first_missing
        _log.debug(
            "consciousness: staged departure for key=%s name=%r after %.1fs absent "
            "(confirm=%.1fs)",
            key, person_name, now - first_missing, confirm_for_key,
        )

    # ── Resolve pending departures ─────────────────────────────────────────────
    for key in list(_pending_departure_keys):
        departed_at, person_name, person_db_id = _pending_departure_keys[key]

        # Person returned — cancel
        if key in current_keys:
            del _pending_departure_keys[key]
            continue

        # Timeout: give up after departure_cooldown without resolution
        if now - departed_at > departure_cooldown:
            del _pending_departure_keys[key]
            continue

        # Face gone but user still talking → likely just stepped off-camera; suppress
        if profile.likely_still_present:
            continue
        if _face_tracking_recently_held_person(person_db_id, now):
            continue

        # Fire only when face-gone + VAD has been silent ≥ departure_audio_silence.
        should_fire = profile.apparent_departure or (
            (now - departed_at) >= departure_audio_silence
            and not profile.user_mid_sentence
        )
        if not should_fire:
            continue

        if not _should_fire_presence(
            key,
            person_db_id,
            profile,
            allow_engaged=True,
            bypass_cooldown=True,
        ):
            continue

        _last_departure_reaction_at[key] = now
        _last_presence_reaction_at[key] = now
        _first_missing_at.pop(key, None)
        del _pending_departure_keys[key]

        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = _first_name(person_name, "there")
            _log.info("consciousness: departure reaction firing for %s", person_name)
            _generate_and_speak_presence(
                f"The person named '{first_name}' just slipped out of your camera view. "
                "React in one short in-character line as Rex — playful and dry, "
                "but not mean or needy. Do not imply they literally left the room; "
                "they may only be off-camera. Do not imply nobody likes or misses "
                f"them. Examples: 'Lost visual on {first_name}. Dramatic.', "
                f"'And {first_name} exits frame, stage left.', "
                f"'Fine, {first_name}, hide from the optics.' "
                f"Address {first_name} by name. One line only.",
                label=f"departure for {person_name}",
                tag_key=key,
                emotion="curious",
            )
        else:
            address = random.choice(unknown_addresses)
            _log.info("consciousness: departure reaction firing for unknown (key=%s)", key)
            _generate_and_speak_presence(
                f"Someone you don't recognize just left your camera view. "
                f"React in one short in-character line as Rex — dry, amused, slightly suspicious. "
                f"Use a generic address like '{address}' (examples: 'hey you', 'you there', "
                "'mystery organic', 'that one'). Example lines: "
                f"'And off goes {address}...', 'Huh. The mystery deepens.', "
                f"'Farewell, {address}. Whoever you are.' One line only.",
                label=f"departure for unknown ({key})",
                tag_key=key,
                emotion="curious",
            )

    # ── Returned: absent last tick, visible now ────────────────────────────────
    first_sight_pending_keys: set = set()
    for key in current_keys - _visible_people:
        person_name, person_db_id = current_tracked[key]

        if _is_jeff_benziger(person_name):
            first_visible = _first_sight_seen_at.setdefault(key, now)
            confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
            if (now - first_visible) < max(0.0, confirm_visible):
                first_sight_pending_keys.add(key)
                continue
            _stage_jeff_history_hunters_greeting(
                key=key,
                person_name=person_name,
            )
            if _try_fire_jeff_history_hunters_greeting(
                key=key,
                person_name=person_name,
                person_db_id=person_db_id,
                profile=profile,
            ):
                continue
            if key not in _jeff_celebrity_greeted_this_session:
                first_sight_pending_keys.add(key)
                continue

        if _is_jt_volleyball_celebrity(person_name):
            first_visible = _first_sight_seen_at.setdefault(key, now)
            confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
            if (now - first_visible) < max(0.0, confirm_visible):
                first_sight_pending_keys.add(key)
                continue
            _stage_jt_volleyball_greeting(
                key=key,
                person_name=person_name,
            )
            if _try_fire_jt_volleyball_greeting(
                key=key,
                person_name=person_name,
                person_db_id=person_db_id,
                profile=profile,
            ):
                continue
            if key not in _jt_volleyball_greeted_this_session:
                first_sight_pending_keys.add(key)
                continue

        # First time ever seen this session.
        if key not in _last_seen:
            if isinstance(key, int) and person_name and key not in _greeted_this_session:
                if _hold_startup_individual_greeting(snapshot, now):
                    first_sight_pending_keys.add(key)
                    _first_sight_seen_at.setdefault(key, now)
                    continue
                first_visible = _first_sight_seen_at.setdefault(key, now)
                confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
                if (now - first_visible) < max(0.0, confirm_visible):
                    first_sight_pending_keys.add(key)
                    continue
                if not _should_fire_presence(key, person_db_id, profile):
                    first_sight_pending_keys.add(key)
                    continue
                first_name = _first_name(person_name, "there")
                context_sentence, situation_phrase = _first_sight_context(first_name)
                prompt: Optional[str] = None
                direct_text: Optional[str] = None
                label = f"first-sight greeting for {person_name}"
                emotion = "excited"
                emotional_to_ack: Optional[dict] = None
                celebration_to_ack: Optional[dict] = None
                followup_to_remove: Optional[tuple[Optional[int], object]] = None
                anticipated_to_mark: Optional[tuple[Optional[int], object]] = None
                profile_question_to_record: Optional[dict] = None
                disposition_to_mark: Optional[int] = None

                # Priority 0 — recent sensitive emotional event.
                # This intentionally outranks temporal banter like
                # "back so soon"; care comes before the bit.
                emotional = None
                try:
                    crowd_count = int((snapshot.get("crowd") or {}).get("count", 1) or 1)
                except Exception:
                    crowd_count = 1
                suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
                if not (suppress_in_crowd and crowd_count > 1):
                    emotional = _pick_due_emotional_checkin(person_db_id)
                if emotional is not None:
                    prompt = _build_emotional_checkin_prompt(
                        first_name, emotional, context_sentence,
                    )
                    label = f"first-sight emotional check-in for {person_name}"
                    emotion = "sad" if float(emotional.get("valence", -0.5) or -0.5) < 0 else "happy"
                    emotional_to_ack = emotional
                    _log.info(
                        "consciousness: first-sight emotional check-in for %s "
                        "(category=%s, event_id=%s)",
                        person_name, emotional.get("category"), emotional.get("id"),
                    )

                # Priority 1 — birthday within reminder window
                if prompt is None:
                    bday_days = _pick_birthday_window(person_db_id)
                else:
                    bday_days = None
                if bday_days is not None:
                    prompt = _build_birthday_prompt(first_name, bday_days)
                    label = f"startup birthday (T-{bday_days}) for {person_name}"
                    _log.info(
                        "consciousness: startup birthday reminder for %s (T-%d days)",
                        person_name, bday_days,
                    )

                # Priority 1.5 — positive news / milestone check-in
                if prompt is None:
                    celebration = _pick_due_celebration_checkin(person_db_id)
                    if celebration is not None:
                        prompt = _build_celebration_checkin_prompt(
                            first_name, celebration, context_sentence,
                        )
                        label = f"first-sight celebration check-in for {person_name}"
                        emotion = "happy"
                        celebration_to_ack = celebration
                        _log.info(
                            "consciousness: first-sight celebration check-in for %s "
                            "(category=%s, event_id=%s)",
                            person_name,
                            celebration.get("category"),
                            celebration.get("id"),
                        )

                # Priority 2 — milestone visit
                if prompt is None:
                    milestone = _pick_milestone(person_db_id)
                    if milestone is not None:
                        prompt = _build_milestone_prompt(first_name, milestone)
                        label = f"startup milestone (#{milestone}) for {person_name}"
                        _log.info(
                            "consciousness: startup milestone for %s (visit #%d)",
                            person_name, milestone,
                        )

                # Priority 2.5 — pending follow-up (something they planned that has now passed)
                if prompt is None:
                    try:
                        from memory import events as events_mod
                        pending = events_mod.get_pending_followups(person_db_id) or []
                    except Exception:
                        pending = []
                    if pending:
                        ev = pending[0]
                        ev_name = ev.get("event_name") or ""
                        if ev_name:
                            followup_to_remove = (person_db_id, ev.get("id"))
                            prompt = (
                                f"{context_sentence} "
                                f"You remember they told you they had this on their schedule: "
                                f"'{ev_name}' — and the date has now passed. Greet them and "
                                f"ask specifically how '{ev_name}' went, in two short Rex-style "
                                f"sentences. Address {first_name} by name. The second sentence "
                                f"must end in a question mark."
                            )
                            label = f"startup followup ({ev_name}) for {person_name}"
                            emotion = "curious"
                            _log.info(
                                "consciousness: startup follow-up for %s — %s",
                                person_name, ev_name,
                            )

                # Priority 3 — anticipated upcoming event
                if prompt is None:
                    anticipated = _pick_anticipated_event(person_db_id)
                    if anticipated is not None:
                        anti_prompt = _build_anticipation_prompt(
                            first_name, anticipated,
                            situation_phrase,
                        )
                        if anti_prompt:
                            anticipated_to_mark = (person_db_id, anticipated["id"])
                            prompt = anti_prompt
                            label = f"startup anticipation for {person_name}"
                            _log.info(
                                "consciousness: startup anticipation for %s (event=%s)",
                                person_name, anticipated.get("event_name"),
                            )

                # Priority 3.5 — same-day repeat activation ("oh, it's you again").
                # More specific than the generic recent-return banter below: if Rex
                # already greeted this person earlier TODAY, open with a short roast
                # about the repeat visit, then move into conversation. Keyed on Rex's
                # own recorded greetings (memory.people.greetings_today_count), so it
                # only counts real prior greetings, not camera re-sightings.
                if prompt is None:
                    prior_today = _same_day_return_count(person_db_id)
                    if prior_today >= 1:
                        prompt = _build_same_day_return_prompt(first_name, prior_today)
                        label = f"startup same-day return (#{prior_today + 1}) for {person_name}"
                        emotion = "excited"
                        _log.info(
                            "consciousness: startup same-day return for %s (greeting #%d today)",
                            person_name, prior_today + 1,
                        )

                # Priority 4 — long absence or recent return
                if prompt is None:
                    absence = _pick_absence_phase(person_db_id)
                    startup_recent_grace = float(
                        getattr(config, "PRESENCE_STARTUP_RECENT_RETURN_GRACE_SECS", 45.0)
                    )
                    process_uptime = (
                        now - _process_started_mono
                        if _process_started_mono > 0.0
                        else startup_recent_grace
                    )
                    if absence and absence[0] == "long_absence":
                        prompt = _build_long_absence_prompt(first_name, absence[1])
                        label = f"startup long-absence for {person_name}"
                        emotion = "curious"
                        _log.info(
                            "consciousness: startup long-absence for %s (%.1f days)",
                            person_name, absence[1],
                        )
                    elif (
                        absence
                        and absence[0] == "recent_return"
                        and process_uptime >= startup_recent_grace
                    ):
                        prompt = _build_recent_return_prompt(first_name, absence[1])
                        label = f"startup recent-return for {person_name}"
                        emotion = "curious"
                        _log.info(
                            "consciousness: startup recent-return for %s (%.1f hrs)",
                            person_name, absence[1],
                        )

                # Fallback — profile-building greeting for sparse known people,
                # then generic greeting.
                if prompt is None:
                    disposition_greeting = _pick_first_sight_disposition_greeting(
                        person_db_id,
                        first_name,
                    )
                    if disposition_greeting:
                        disposition_label, line = disposition_greeting
                        prompt = line
                        direct_text = line
                        label = f"first-sight disposition greeting ({disposition_label}) for {person_name}"
                        emotion = "happy" if disposition_label == "smiley" else "curious"
                        disposition_to_mark = person_db_id
                        _log.info(
                            "consciousness: startup disposition greeting for %s label=%s",
                            person_name,
                            disposition_label,
                        )

                if prompt is None:
                    profile_question = _pick_startup_profile_question(person_db_id)
                    if profile_question:
                        question_text = str(profile_question.get("text") or "").strip()
                    else:
                        question_text = ""
                    if question_text:
                        profile_question_to_record = profile_question
                        prompt = _build_startup_profile_question_prompt(
                            first_name,
                            context_sentence,
                            question_text,
                        )
                        label = f"first-sight profile question for {person_name}"
                        emotion = "curious"
                        _log.info(
                            "consciousness: startup profile question for %s key=%s",
                            person_name,
                            profile_question.get("key"),
                        )
                    else:
                        mood_prompt = _build_first_sight_mood_prompt(
                            first_name,
                            context_sentence,
                            _get_first_sight_mood(person_db_id),
                        )
                        if mood_prompt:
                            prompt, emotion = mood_prompt
                            label = f"first-sight mood greeting for {person_name}"
                            _log.info(
                                "consciousness: startup mood greeting for %s",
                                person_name,
                            )
                        else:
                            prompt = _build_startup_solo_greeting_prompt(
                                first_name,
                                context_sentence,
                            )
                            _log.info("consciousness: startup greeting for %s", person_name)

                queued = _generate_and_speak_presence(
                    prompt,
                    label=label,
                    tag_key=key,
                    emotion=emotion,
                    purpose=(
                        "emotional_checkin"
                        if "emotional check-in" in label
                        else "celebration_checkin"
                        if "celebration check-in" in label
                        else "memory_followup"
                        if "followup" in label or "anticipation" in label
                        else "presence_reaction"
                    ),
                    startup_greeting_name=first_name,
                    question_key=(
                        str(profile_question_to_record.get("key"))
                        if profile_question_to_record
                        and profile_question_to_record.get("key")
                        else None
                    ),
                    question_depth=(
                        int(profile_question_to_record.get("depth", 1))
                        if profile_question_to_record
                        else 1
                    ),
                    direct_text=direct_text,
                )
                if queued:
                    if emotional_to_ack is not None:
                        _note_emotional_checkin_fired(person_db_id)
                        try:
                            from memory import emotional_events as emo_events
                            emo_events.mark_acknowledged(int(emotional_to_ack["id"]))
                        except Exception:
                            pass
                    if celebration_to_ack is not None:
                        try:
                            from memory import emotional_events as emo_events
                            emo_events.mark_acknowledged(int(celebration_to_ack["id"]))
                        except Exception:
                            pass
                    if followup_to_remove is not None:
                        _pending_followups_lock_remove(
                            followup_to_remove[0],
                            followup_to_remove[1],
                        )
                    if anticipated_to_mark is not None:
                        _anticipated_events.add(anticipated_to_mark)
                    if disposition_to_mark is not None:
                        try:
                            from memory import disposition as disposition_memory
                            disposition_memory.mark_mentioned(disposition_to_mark)
                        except Exception as exc:
                            _log.debug(
                                "disposition mention mark failed person_id=%s: %s",
                                disposition_to_mark,
                                exc,
                            )
                    _greeted_this_session.add(key)
                    _first_sight_seen_at.pop(key, None)
                else:
                    first_sight_pending_keys.add(key)
            # Already greeted/enrolled people and anonymous slots should become
            # tracked quietly on their first presence tick. Otherwise a fresh
            # enrollment can be misread as "back so soon" once the camera settles.
            continue

        absent_secs = now - _last_seen[key]
        if absent_secs < return_min_absent:
            _confirmed_absent_at.pop(key, None)
            continue
        if key not in _confirmed_absent_at:
            continue

        if not _should_fire_presence(
            key,
            person_db_id,
            profile,
            allow_engaged=True,
            bypass_cooldown=True,
        ):
            continue

        _last_return_reaction_at[key] = now
        _confirmed_absent_at.pop(key, None)
        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = _first_name(person_name, "there")
            _log.info("consciousness: return detected — queuing reaction for %s (absent %.1fs)", person_name, absent_secs)
            if _is_jeff_benziger(person_name):
                _stage_jeff_history_hunters_greeting(
                    key=key,
                    person_name=person_name,
                    returning=True,
                )
                _try_fire_jeff_history_hunters_greeting(
                    key=key,
                    person_name=person_name,
                    person_db_id=person_db_id,
                    profile=profile,
                    returning=True,
                )
                continue
            if _is_jt_volleyball_celebrity(person_name):
                _stage_jt_volleyball_greeting(
                    key=key,
                    person_name=person_name,
                    returning=True,
                )
                _try_fire_jt_volleyball_greeting(
                    key=key,
                    person_name=person_name,
                    person_db_id=person_db_id,
                    profile=profile,
                    returning=True,
                )
                continue
            allow_return_memory = (
                absent_secs
                >= float(getattr(config, "PRESENCE_RETURN_MEMORY_FOLLOWUP_MIN_ABSENT_SECS", 120.0))
                and not _end_thread_grace_active()
            )
            anticipated = _pick_anticipated_event(person_db_id) if allow_return_memory else None
            anticipation_prompt = (
                _build_anticipation_prompt(
                    first_name,
                    anticipated,
                    f"they just walked back into your camera view after about {int(absent_secs)} seconds away",
                )
                if anticipated
                else None
            )
            if anticipation_prompt:
                _anticipated_events.add((person_db_id, anticipated["id"]))
                _log.info(
                    "consciousness: return anticipation for %s (event=%s)",
                    person_name, anticipated.get("event_name"),
                )
                _generate_and_speak_presence(
                    anticipation_prompt,
                    label=f"return anticipation for {person_name}",
                    tag_key=key,
                    emotion="curious",
                    purpose="memory_followup",
                )
                continue
            appearance_hint = _pick_appearance_hint(person_db_id)
            if appearance_hint and random.random() < getattr(config, "APPEARANCE_RIFF_PROBABILITY", 0.35):
                prompt = (
                    f"The person named '{first_name}' just came back into your camera view "
                    f"after about {int(absent_secs)} seconds away. You remember this about "
                    f"their appearance: {appearance_hint}. React in one short in-character "
                    f"Rex line that NATURALLY references that appearance detail — warm but "
                    f"dry. Address {first_name} by name. One line only."
                )
            else:
                prompt = (
                    f"The person named '{first_name}' just came back into your camera view after "
                    f"being away for about {int(absent_secs)} seconds. "
                    "React in one short in-character line as Rex — warm but dry. Examples: "
                    f"'Oh, there you are, {first_name}.', "
                    f"'Visual reacquired. There you are, {first_name}.', "
                    f"'There you are, {first_name}; optics are back online.' "
                    f"Address {first_name} by name. One line only."
                )
            _generate_and_speak_presence(
                prompt,
                label=f"return for {person_name}",
                tag_key=key,
                emotion="neutral",
            )
        else:
            address = random.choice(unknown_addresses)
            _log.info("consciousness: return detected — queuing reaction for unknown (key=%s, absent=%.1fs)", key, absent_secs)
            _generate_and_speak_presence(
                f"Someone you don't recognize has returned to your camera view after "
                f"about {int(absent_secs)} seconds away. "
                "React in one short in-character line as Rex — suspicious, dry, slightly wary. "
                f"Use a generic address like '{address}'. Examples: "
                f"'Oh, you again.', 'Back already, mystery organic?', "
                "'I see you returned. Bold choice.' One line only.",
                label=f"return for unknown ({key})",
                tag_key=key,
                emotion="neutral",
            )

    # Update last-seen timestamps after absence checks so the pre-tick value is accurate.
    for key in list(_first_sight_seen_at):
        if key not in current_keys:
            _first_sight_seen_at.pop(key, None)
    for key in current_keys:
        if key in first_sight_pending_keys:
            continue
        _last_seen[key] = now
    _visible_people = current_keys - first_sight_pending_keys


# ─────────────────────────────────────────────────────────────────────────────
# Step 10c — Third-party awareness
# ─────────────────────────────────────────────────────────────────────────────

# Reuse the same set the disengagement step uses to keep the definition consistent.
from awareness.social import _DISENGAGED_ENGAGEMENT as _LURK_ENGAGEMENT_VALUES


def _step_third_party_awareness(snapshot: dict, profile: SituationProfile) -> None:
    """
    When Rex has an active conversation partner and another person nearby is
    visibly disengaged but lingering, low-probability callout that acknowledges
    the lurker. Each (session, lurker) is called out at most once.

    Rate-limited per loop tick so the dispatcher stays cheap.
    """
    global _last_third_party_check_at

    if profile.suppress_proactive or profile.rapid_exchange:
        return

    now = time.monotonic()
    interval = getattr(config, "THIRD_PARTY_CHECK_INTERVAL_SECS", 5.0)
    if (now - _last_third_party_check_at) < interval:
        return
    _last_third_party_check_at = now

    try:
        people = snapshot.get("people", []) or []
        if len(people) < 2:
            _third_party_seen_at.clear()
            return

        crowd = snapshot.get("crowd", {}) or {}
        dominant = crowd.get("dominant_speaker")
        if not dominant:
            # No active conversation partner → not a "third party" situation.
            _third_party_seen_at.clear()
            return

        lurk_threshold = getattr(config, "THIRD_PARTY_LURK_SECS", 30.0)
        callout_prob = getattr(config, "THIRD_PARTY_CALLOUT_PROBABILITY", 0.10)

        present_keys: set = set()
        for person in people:
            pid = person.get("id")
            if pid is None or pid == dominant:
                continue
            engagement = (person.get("engagement") or "").lower()
            if engagement not in _LURK_ENGAGEMENT_VALUES:
                # They're engaged or attentive — not a lurker; reset their timer.
                _third_party_seen_at.pop(pid, None)
                continue

            present_keys.add(pid)
            first_seen = _third_party_seen_at.setdefault(pid, now)
            lurk_secs = now - first_seen
            if lurk_secs < lurk_threshold:
                continue
            if pid in _third_party_called_out:
                continue
            if random.random() >= callout_prob:
                continue

            face_id = person.get("face_id")
            if face_id and isinstance(face_id, str):
                first_name = _first_name(face_id, "there")
                descriptor = f"the person named '{first_name}' standing nearby"
                address_hint = f"refer to them as '{first_name}'"
            else:
                descriptor = "the other person standing nearby — you don't know their name"
                address_hint = "use a generic label like 'your friend over there' or 'the one in the back'"

            prompt = (
                f"You're mid-conversation with someone, but {descriptor} has been "
                f"hanging around for about {int(lurk_secs)} seconds without engaging — "
                f"facing away or looking down. Drop ONE short in-character Rex line "
                f"that acknowledges them dryly, observant rather than confrontational. "
                f"{address_hint.capitalize()}. Examples in spirit: "
                f"'Your friend over there has been pretending not to listen for a while now.', "
                f"'Don't think I haven't noticed the lurker.' One line only."
            )
            _third_party_called_out.add(pid)
            _log.info(
                "consciousness: third-party callout for pid=%s (lurk %.1fs)",
                pid, lurk_secs,
            )
            _generate_and_speak_presence(
                prompt,
                label=f"third-party callout for {pid}",
                tag_key=f"third_party:{pid}",
                emotion="curious",
                purpose="third_party_awareness",
            )
            # One callout per tick to avoid stacking lines.
            break

        # Clean up timers for people who left the scene this tick.
        for pid in list(_third_party_seen_at.keys()):
            if pid not in present_keys:
                _third_party_seen_at.pop(pid, None)
    except Exception as exc:
        _log.debug("third-party awareness step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10c2 — Group turn-taking
# ─────────────────────────────────────────────────────────────────────────────

def _first_name(name: Optional[str], fallback: str = "there") -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return fallback
    return cleaned.split()[0]


def _group_turn_count(person_id: int, now: float, window_secs: float) -> int:
    turns = _group_turn_speaker_times.get(person_id)
    if not turns:
        return 0
    cutoff = now - max(0.0, window_secs)
    while turns and turns[0] < cutoff:
        turns.popleft()
    return len(turns)


def _group_turn_last_spoke(person_id: int, now: float, window_secs: float) -> Optional[float]:
    _group_turn_count(person_id, now, window_secs)
    turns = _group_turn_speaker_times.get(person_id)
    if not turns:
        return None
    return turns[-1]


def _step_group_turn_taking(snapshot: dict, profile: SituationProfile) -> None:
    """
    Softly invite a known, visible quiet person into an active small-group
    conversation after one person has been carrying the floor for a while.

    This is intentionally gentler than true turn arbitration: it only fires in
    a lull, only once per target per session, and respects empathy, closure, and
    question-budget gates.
    """
    global _last_group_turn_check_at

    if not getattr(config, "GROUP_TURN_TAKING_ENABLED", True):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if profile.rapid_exchange or not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    interval = float(getattr(config, "GROUP_TURN_CHECK_INTERVAL_SECS", 5.0))
    if (now - _last_group_turn_check_at) < max(0.0, interval):
        return
    _last_group_turn_check_at = now

    try:
        people = snapshot.get("people", []) or []
        known: dict[int, dict] = {}
        for person in people:
            pid = person.get("person_db_id")
            name = person.get("face_id")
            if not isinstance(pid, int) or not name:
                continue
            known.setdefault(pid, person)

        if len(known) < 2:
            _group_turn_visible_since.clear()
            return

        visible_ids = set(known.keys())
        for pid in visible_ids:
            _group_turn_visible_since.setdefault(pid, now)
        for pid in list(_group_turn_visible_since.keys()):
            if pid not in visible_ids:
                _group_turn_visible_since.pop(pid, None)

        with _engaged_lock:
            engaged_id = _engaged_person_id
            engaged_touch = _engaged_last_touch_at

        if engaged_id is None or engaged_id not in known:
            return
        if _visual_curiosity_blocked_by_empathy(engaged_id):
            return

        min_lull = float(getattr(config, "GROUP_TURN_MIN_CONVERSATION_LULL_SECS", 8.0))
        active_window = float(getattr(config, "GROUP_TURN_ACTIVE_WINDOW_SECS", 75.0))
        lull_secs = now - engaged_touch
        if lull_secs < min_lull or lull_secs > active_window:
            return

        recent_window = float(getattr(config, "GROUP_TURN_RECENT_WINDOW_SECS", 180.0))
        min_dominant_turns = int(getattr(config, "GROUP_TURN_DOMINANT_MIN_TURNS", 3))
        if _group_turn_count(engaged_id, now, recent_window) < min_dominant_turns:
            return

        min_visible = float(getattr(config, "GROUP_TURN_QUIET_MIN_VISIBLE_SECS", 25.0))
        min_silence = float(getattr(config, "GROUP_TURN_QUIET_MIN_SILENCE_SECS", 45.0))
        cooldown = float(getattr(config, "GROUP_TURN_PERSON_COOLDOWN_SECS", 900.0))

        target: Optional[dict] = None
        target_visible_secs = 0.0
        target_silence_secs = 0.0
        best_score = -1.0

        for pid, person in known.items():
            if pid == engaged_id:
                continue
            if pid in _group_turn_invited_this_session:
                continue
            if (now - _group_turn_invited_at.get(pid, 0.0)) < max(0.0, cooldown):
                continue
            if _visual_curiosity_blocked_by_empathy(pid):
                continue

            visible_since = _group_turn_visible_since.get(pid, now)
            visible_secs = now - visible_since
            if visible_secs < min_visible:
                continue

            last_spoke = _group_turn_last_spoke(pid, now, recent_window)
            silence_secs = now - last_spoke if last_spoke is not None else visible_secs
            if silence_secs < min_silence:
                continue

            recent_turns = _group_turn_count(pid, now, recent_window)
            score = silence_secs + visible_secs - (recent_turns * 15.0)
            if score > best_score:
                best_score = score
                target = person
                target_visible_secs = visible_secs
                target_silence_secs = silence_secs

        if not target:
            return

        target_id = target.get("person_db_id")
        if not isinstance(target_id, int):
            return
        target_name = target.get("face_id") or "there"
        engaged_name = known[engaged_id].get("face_id") or "your friend"
        target_first = _first_name(target_name)
        engaged_first = _first_name(engaged_name, "the main talker")

        _group_turn_invited_this_session.add(target_id)
        _group_turn_invited_at[target_id] = now

        prompt = (
            f"You are in a small-group conversation. {engaged_name} has been "
            f"doing most of the talking, and {target_name} is visible nearby "
            f"but has been quiet. In ONE short in-character Rex line, gently "
            f"invite {target_first} into this same conversation. Make it feel "
            f"optional, warm, and lightly funny, not accusatory. You may mention "
            f"{engaged_first} only if it helps. Ask at most one easy question. "
            f"Do not mention cameras, tracking, silence timers, or that you are "
            f"monitoring turn-taking. Max 22 words."
        )
        _log.info(
            "consciousness: group turn invite for %s (visible %.1fs, quiet %.1fs, engaged=%s)",
            target_name,
            target_visible_secs,
            target_silence_secs,
            engaged_name,
        )
        _generate_and_speak(
            prompt,
            emotion="curious",
            wait_secs=getattr(config, "QUESTION_RESPONSE_WAIT_SECS", 8.0),
            purpose="group_turn_invite",
            label=f"group turn invite for {target_name}",
        )
    except Exception as exc:
        _log.debug("group turn-taking step error: %s", exc)


def _step_group_lull(snapshot: dict, profile: SituationProfile) -> None:
    """
    If a known group stays visible after a greeting or brief reply but nobody
    says much, open the room with one easy social question.
    """
    global _last_group_lull_check_at

    if not getattr(config, "GROUP_LULL_ENABLED", True):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if profile.rapid_exchange or is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    interval = float(getattr(config, "GROUP_LULL_CHECK_INTERVAL_SECS", 3.0))
    if (now - _last_group_lull_check_at) < max(0.0, interval):
        return
    _last_group_lull_check_at = now

    try:
        from intelligence import social_scene
        scene = social_scene.from_snapshot(snapshot)
        if len(scene.known) < 2:
            return

        signature = scene.signature
        cooldown = float(getattr(config, "GROUP_LULL_COOLDOWN_SECS", 180.0))
        if (now - _group_lull_fired_at.get(signature, 0.0)) < max(0.0, cooldown):
            return

        with _engaged_lock:
            active_touch = _engaged_last_touch_at
            recent_touch = _recent_engaged_touch_at

        last_relevant = max(active_touch, recent_touch, _last_proactive_speech_at)
        if last_relevant <= 0.0:
            return

        quiet_for = now - last_relevant
        min_silence = float(getattr(config, "GROUP_LULL_MIN_SILENCE_SECS", 14.0))
        active_window = float(getattr(config, "GROUP_LULL_ACTIVE_WINDOW_SECS", 90.0))
        if quiet_for < min_silence or quiet_for > active_window:
            return

        label = social_scene.visible_group_label(scene)
        _group_lull_fired_at[signature] = now
        prompt = (
            f"You are with this visible known group: {label}. They answered or "
            f"sat quietly, and the room has gone quiet for about {quiet_for:.0f} "
            f"seconds. In ONE short in-character Rex line, gently reopen the "
            f"conversation with an easy social question for the group. Good "
            f"directions: ask what brings them both here tonight, what the "
            f"occasion is, or what they have going on. Warm, curious, lightly "
            f"funny. Do not mention cameras, timers, monitoring, or that they "
            f"are being analyzed. Max 22 words; end with a question mark."
        )
        _log.info(
            "consciousness: group lull prompt for %s after %.1fs quiet",
            label,
            quiet_for,
        )
        _generate_and_speak(
            prompt,
            emotion="curious",
            wait_secs=getattr(config, "QUESTION_RESPONSE_WAIT_SECS", 8.0),
            purpose="small_talk",
            label=f"group lull for {label}",
        )
    except Exception as exc:
        _log.debug("group-lull step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10e — Overheard chime-in
# ─────────────────────────────────────────────────────────────────────────────

def _step_overheard_chime_in(snapshot: dict, profile: SituationProfile) -> None:
    """
    When Rex hears someone talking ABOUT him (referential / instructional
    address mode), he may choose to chime in unprompted. Reads
    world_state.social.being_discussed (written by interaction.py) and rolls a
    low-probability decision biased by sentiment, friendship tier, and
    family-safe mode.

    Heavy gating:
      - profile.suppress_proactive must be False
      - rapid_exchange must be False
      - at least OVERHEARD_MIN_GAP_SECS since the last mention (let humans finish)
      - per-active-window dedupe (chimed_in flag on the world_state record)
      - per-session cap (OVERHEARD_MAX_PER_SESSION)
      - rate-limited per loop tick
    """
    global _overheard_chime_in_count, _last_overheard_check_at

    if not getattr(config, "OVERHEARD_CHIME_IN_ENABLED", True):
        return
    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not profile.being_discussed:
        return

    now = time.monotonic()
    interval = float(getattr(config, "OVERHEARD_CHECK_INTERVAL_SECS", 2.0))
    if (now - _last_overheard_check_at) < interval:
        return
    _last_overheard_check_at = now

    max_per_session = int(getattr(config, "OVERHEARD_MAX_PER_SESSION", 3))
    if _overheard_chime_in_count >= max_per_session:
        return

    social = (snapshot.get("social") or {}) if isinstance(snapshot, dict) else {}
    bd = social.get("being_discussed") or {}
    last_at = bd.get("last_mention_at")
    if last_at is None:
        return
    if bd.get("chimed_in"):
        return

    # Wall-clock gap so humans can finish the thought.
    min_gap = float(getattr(config, "OVERHEARD_MIN_GAP_SECS", 2.0))
    age = time.time() - float(last_at)
    if age < min_gap:
        return

    # Probability composition.
    base = float(getattr(config, "OVERHEARD_CHIME_IN_PROBABILITY", 0.15))
    sentiment = (bd.get("sentiment") or "neutral").lower()
    if sentiment == "positive":
        base += float(getattr(config, "OVERHEARD_POSITIVE_SENTIMENT_BONUS", 0.15))
    elif sentiment == "negative":
        base += float(getattr(config, "OVERHEARD_INSULT_BONUS", 0.30))

    # Don't bite back at insults if a child is present.
    if profile.force_family_safe and sentiment == "negative":
        base = max(0.0, base - 0.40)

    # Friendship gate: only chime in on speakers who are at least the
    # configured tier (avoids butting in on strangers).
    required_tier = getattr(config, "OVERHEARD_REQUIRE_FRIENDSHIP_TIER", "acquaintance")
    speaker_id = bd.get("speaker_id")
    speaker_name = bd.get("speaker_name") or "someone"
    if required_tier and speaker_id:
        try:
            from memory import people as people_memory
            person = people_memory.get_person(speaker_id) if hasattr(people_memory, "get_person") else None
            tier = (person or {}).get("friendship_tier") or "stranger"
            tier_order = ["stranger", "acquaintance", "friend", "close_friend", "best_friend"]
            if tier_order.index(tier) < tier_order.index(required_tier):
                return
        except Exception:
            # If we can't look the speaker up, fall through — don't block.
            pass

    if random.random() >= base:
        return

    snippet = bd.get("last_snippet") or ""
    label = bd.get("label") or "referential"
    speaker_first = _first_name(speaker_name, "someone")

    if label == "instructional":
        prompt = (
            f"You just overheard {speaker_first} tell someone else: "
            f"'{snippet}'. They were not addressing you directly — they were "
            f"talking ABOUT you to someone else, instructing them. In ONE short "
            f"in-character Rex line, chime in playfully — like a robot who "
            f"realizes he's the topic and can't help but join the conversation. "
            f"Stay light. One line only."
        )
    elif sentiment == "positive":
        prompt = (
            f"You just overheard {speaker_first} say to someone else: "
            f"'{snippet}'. They were talking ABOUT you, not to you. In ONE short "
            f"in-character Rex line, chime in — pleased they noticed but not "
            f"sappy, a little proud, slightly cheeky. One line only."
        )
    elif sentiment == "negative":
        prompt = (
            f"You just overheard {speaker_first} say to someone else: "
            f"'{snippet}'. They were talking ABOUT you, not to you, and it "
            f"wasn't flattering. In ONE short in-character Rex line, push back "
            f"dryly — wounded but witty, no real heat. One line only."
        )
    else:
        prompt = (
            f"You just overheard {speaker_first} say to someone else: "
            f"'{snippet}'. They were talking ABOUT you. In ONE short in-character "
            f"Rex line, chime in observantly — a robot who notices he's the topic. "
            f"One line only."
        )

    # Mark chimed_in immediately so a slow LLM call can't double-fire.
    try:
        social_now = world_state.get("social") or {}
        bd_now = social_now.get("being_discussed") or {}
        bd_now["chimed_in"] = True
        social_now["being_discussed"] = bd_now
        world_state.update("social", social_now)
    except Exception:
        pass

    _overheard_chime_in_count += 1
    _log.info(
        "consciousness: overheard chime-in firing (count=%d label=%s sentiment=%s p=%.2f)",
        _overheard_chime_in_count, label, sentiment, base,
    )
    _generate_and_speak_presence(
        prompt,
        label=f"overheard chime-in ({label}/{sentiment})",
        tag_key=f"overheard:{int(float(last_at))}",
        emotion="curious" if sentiment == "neutral" else (
            "happy" if sentiment == "positive" else "annoyed"
        ),
        purpose="overheard_chime_in",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 10d — Holiday plans curiosity
# ─────────────────────────────────────────────────────────────────────────────

def _holiday_plans_allowed(holiday: dict) -> bool:
    """Return whether Rex should proactively ask about this holiday."""
    window = str((holiday or {}).get("window") or "").strip().lower()
    if window == "major":
        return True
    if window == "minor":
        return bool(getattr(config, "HOLIDAY_PLANS_INCLUDE_MINOR", False))
    return False


def _step_holiday_plans(snapshot: dict, profile: SituationProfile) -> None:
    """
    During an active conversation with a known person, if any public holiday
    is within its approach window, occasionally ask the engaged person about
    their plans. Each (person, holiday) pair is asked at most once per session;
    the holiday's iso date includes the year so next year resets naturally.
    """
    global _last_holiday_plans_check_at

    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _can_proactive_speak():
        return

    now = time.monotonic()
    interval = getattr(config, "HOLIDAY_PLANS_CHECK_INTERVAL_SECS", 30.0)
    if (now - _last_holiday_plans_check_at) < interval:
        return
    _last_holiday_plans_check_at = now

    # Need an engaged conversation partner with a DB record (so we can dedupe).
    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    window = getattr(config, "ENGAGEMENT_WINDOW_SECS", 90.0)
    if (now - engaged_touch) > window:
        return

    try:
        from awareness.holidays import upcoming_holidays
        from memory import people as people_mod

        holidays = upcoming_holidays()
        if not holidays:
            return

        # Find the soonest holiday Rex hasn't asked this person about yet.
        target = None
        for h in holidays:
            if not _holiday_plans_allowed(h):
                continue
            if (engaged_id, h["date"]) not in _holiday_plans_asked:
                target = h
                break
        if target is None:
            return

        if random.random() >= getattr(config, "HOLIDAY_PLANS_PROBABILITY", 0.25):
            return

        person = people_mod.get_person(engaged_id)
        if not person:
            return
        first_name = _first_name(person.get("name"), "you")

        days_until = target["days_until"]
        if days_until == 0:
            when_clause = "today"
        elif days_until == 1:
            when_clause = "tomorrow"
        else:
            when_clause = f"in {days_until} days"

        if target["window"] == "major":
            framing = (
                f"Ask {first_name} about their plans for {target['name']} ({when_clause}). "
                f"Treat it as a real holiday — the kind organics actually do something for. "
            )
        else:
            framing = (
                f"{target['name']} is {when_clause}. "
                f"Ask {first_name} if that date means anything to them, dryly observant. "
            )

        prompt = (
            f"You're mid-conversation with '{first_name}'. {framing}"
            f"One short in-character Rex line, ending with a question. Don't lecture about "
            f"the holiday — just ask the question, in Rex's voice."
        )

        if _generate_and_speak(prompt, emotion="curious", purpose="memory_followup"):
            _holiday_plans_asked.add((engaged_id, target["date"]))
            _log.info(
                "consciousness: holiday plans question for person_id=%s — %s (T-%dd, %s)",
                engaged_id, target["name"], days_until, target["window"],
            )
    except Exception as exc:
        _log.debug("holiday plans step error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10e2 — Weekly small talk (weekend plans, week ahead, weekend recap)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_weekly_slot(snapshot: dict) -> Optional[str]:
    """
    Map current day-of-week + part-of-day to a small-talk slot, or None.
      Friday afternoon/evening → "weekend_plans"
      Sunday evening           → "week_ahead"
      Monday morning/midday    → "weekend_recap"
    """
    t = snapshot.get("time", {}) or {}
    dow = (t.get("day_of_week") or "").lower()
    part = (t.get("time_of_day") or "").lower()
    if dow == "friday" and part in ("afternoon", "evening", "night"):
        return "weekend_plans"
    if dow == "sunday" and part in ("evening", "night"):
        return "week_ahead"
    if dow == "monday" and part in ("morning", "afternoon"):
        return "weekend_recap"
    return None


def _step_weekly_smalltalk(snapshot: dict, profile: SituationProfile) -> None:
    """
    During an active conversation with a known person, occasionally ask weekly
    small-talk questions keyed on day-of-week:
      Friday eve  → "any plans this weekend?"
      Sunday eve  → "what's on the agenda this week?"
      Monday a.m. → "how was your weekend?" (referencing stored weekend events when present)
    Each (person, ISO-week, slot) is asked at most once.
    """
    global _last_weekly_smalltalk_check_at

    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _can_proactive_speak():
        return

    now = time.monotonic()
    interval = getattr(config, "WEEKLY_SMALLTALK_CHECK_INTERVAL_SECS", 30.0)
    if (now - _last_weekly_smalltalk_check_at) < interval:
        return
    _last_weekly_smalltalk_check_at = now

    slot = _pick_weekly_slot(snapshot)
    if slot is None:
        return

    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    window = getattr(config, "ENGAGEMENT_WINDOW_SECS", 90.0)
    quiet_for = now - engaged_touch
    if quiet_for > window:
        return
    min_silence = float(getattr(config, "WEEKLY_SMALLTALK_MIN_SILENCE_SECS", 45.0) or 0.0)
    if min_silence and quiet_for < min_silence:
        return

    try:
        from datetime import date as _date
        iso_year, iso_week, _ = _date.today().isocalendar()
        dedupe_key = (engaged_id, iso_year, iso_week, slot)
        if dedupe_key in _weekly_smalltalk_asked:
            return
        if random.random() >= getattr(config, "WEEKLY_SMALLTALK_PROBABILITY", 0.6):
            return

        from memory import people as people_mod
        from memory import events as events_mod
        person = people_mod.get_person(engaged_id)
        if not person:
            return
        first_name = _first_name(person.get("name"), "you")

        if slot == "weekend_plans":
            upcoming = events_mod.get_upcoming_events(engaged_id) or []
            already = ", ".join(
                f"'{e['event_name']}'"
                for e in upcoming
                if e.get("event_date")
            )
            already_clause = (
                f" You already know they have these upcoming: {already}. "
                f"If relevant, reference them; otherwise just ask openly."
                if already else ""
            )
            prompt = (
                f"You're mid-conversation with '{first_name}'. It's Friday and the "
                f"weekend is starting.{already_clause} Ask {first_name} what they "
                f"have going on this weekend, in one short Rex-style line ending "
                f"with a question. Don't lecture, just ask."
            )
            emotion = "curious"
        elif slot == "week_ahead":
            upcoming = events_mod.get_upcoming_events(engaged_id) or []
            already = ", ".join(
                f"'{e['event_name']}'"
                for e in upcoming
                if e.get("event_date")
            )
            already_clause = (
                f" You already know they mentioned these coming up: {already}. "
                f"You can reference them or ask broader."
                if already else ""
            )
            prompt = (
                f"You're mid-conversation with '{first_name}'. It's Sunday evening "
                f"and a new week is about to start.{already_clause} Ask {first_name} "
                f"what's on their agenda this week, dryly observant. One short "
                f"Rex-style line ending with a question."
            )
            emotion = "curious"
        else:  # weekend_recap
            monday_part = ((snapshot.get("time", {}) or {}).get("time_of_day") or "").lower()
            monday_part_label = monday_part if monday_part in {"morning", "afternoon"} else "morning"
            pending = events_mod.get_pending_followups(engaged_id) or []
            # Prefer asking specifically about things they told Rex they'd do.
            ref = next((e for e in pending if e.get("event_name")), None)
            if ref:
                ref_name = ref["event_name"]
                # Mark this specific event as the implicit follow-up so the
                # post-response handler doesn't re-ask the same thing.
                _pending_followups_lock_remove(engaged_id, ref.get("id"))
                prompt = (
                    f"You're mid-conversation with '{first_name}'. It's Monday and "
                    f"you remember they told you they were going to do this over the "
                    f"weekend: '{ref_name}'. Ask how it went, in one short Rex-style "
                    f"line ending with a question. Reference '{ref_name}' specifically."
                )
            else:
                prompt = (
                    f"You're mid-conversation with '{first_name}'. It's Monday {monday_part_label}. "
                    f"Ask {first_name} how their weekend was, in one short Rex-style "
                    f"line ending with a question. Warm but dry."
                )
            emotion = "curious"

        if _generate_and_speak(prompt, emotion=emotion, purpose="small_talk"):
            _weekly_smalltalk_asked.add(dedupe_key)
            _log.info(
                "consciousness: weekly small-talk for person_id=%s — slot=%s (week %d/%d)",
                engaged_id, slot, iso_week, iso_year,
            )
    except Exception as exc:
        _log.debug("weekly smalltalk step error: %s", exc)


def _step_emotional_checkin(snapshot: dict, profile: SituationProfile) -> None:
    """
    Proactive emotional check-in for the engaged person. Two triggers:

    (A) An unacknowledged active emotional event (recent grief, illness,
        layoff, milestone) exists for this person — open with a soft, in-
        character acknowledgment so they don't have to bring it up first.

    (B) The empathy classifier has been reading the engaged person as
        negatively-valenced (sad / withdrawn / anxious / tired) for at least
        EMPATHY_CHECKIN_NEGATIVE_STREAK_SECS without an obvious uptick —
        notice it, ask once.

    Cooldown: at most one emotional check-in per (person, session). After
    firing, the dedupe set blocks repeats. Trigger (A) marks the events
    acknowledged through the existing ack helper so the system-prompt
    ACKNOWLEDGE-ON-RETURN directive doesn't double up. Honors the discretion
    rule — does not fire trigger (A) when crowd > 1.
    """
    global _last_emotional_checkin_check_at

    if not getattr(config, "EMPATHY_ENABLED", True):
        return
    if not getattr(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", True):
        return
    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _can_proactive_speak():
        return

    now = time.monotonic()
    interval = float(getattr(config, "EMPATHY_CHECKIN_CHECK_INTERVAL_SECS", 10.0))
    if (now - _last_emotional_checkin_check_at) < interval:
        return
    _last_emotional_checkin_check_at = now

    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    window = float(getattr(config, "ENGAGEMENT_WINDOW_SECS", 90.0))
    if (now - engaged_touch) > window:
        return
    if engaged_id in _emotional_checkin_fired:
        return

    try:
        from memory import people as people_mod
        from memory import emotional_events as emo_events
        from intelligence import empathy as _empathy

        person = people_mod.get_person(engaged_id)
        if not person:
            return
        first_name = _first_name(person.get("name"), "you")
        tier = person.get("friendship_tier", "stranger")

        # ── Trigger A: unacknowledged active event ─────────────────────────
        crowd_count = int((snapshot.get("crowd") or {}).get("count", 1) or 1)
        suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
        if not (suppress_in_crowd and crowd_count > 1):
            try:
                active = emo_events.get_due_checkins(engaged_id, limit=3)
            except Exception:
                active = []
            due_checkins = active
            if due_checkins:
                ev = due_checkins[0]
                desc = (ev.get("description") or "").strip()
                cat = (ev.get("category") or "").strip().lower()
                valence = float(ev.get("valence", -0.5) or -0.5)
                vibe = "positive milestone" if valence > 0 else "hard thing"
                prompt = (
                    f"You're talking with '{first_name}' (tier: {tier}). You "
                    f"remember from a previous session that they mentioned this "
                    f"{vibe} — category={cat}: \"{desc}\". You haven't yet "
                    f"acknowledged it on this return visit. In ONE short, soft, "
                    f"in-character Rex line, gently acknowledge it and yield — "
                    f"no probing questions, no advice, no roast. Let them steer "
                    f"the rest. If it was a loss, lean warm. If it was good "
                    f"news, lean genuine. End with a question ONLY if it's "
                    f"low-pressure (e.g. 'how are you holding up?' for hard, "
                    f"'how's that going?' for milestone)."
                )
                emotion = "sad" if valence < 0 else "happy"
                if _generate_and_speak(prompt, emotion=emotion, purpose="emotional_checkin"):
                    _note_emotional_checkin_fired(engaged_id)
                    try:
                        emo_events.mark_acknowledged(int(ev["id"]))
                    except Exception:
                        pass
                    _log.info(
                        "consciousness: proactive emotional check-in (A: "
                        "unacknowledged %s event) for person_id=%s",
                        cat, engaged_id,
                    )
                return

        # ── Trigger A2: remembered positive news / celebration ─────────────
        try:
            celebrations = emo_events.get_due_celebrations(engaged_id, limit=2)
        except Exception:
            celebrations = []
        if celebrations:
            ev = celebrations[0]
            desc = (ev.get("description") or "").strip()
            cat = (ev.get("category") or "").strip().lower()
            prompt = (
                f"You're talking with '{first_name}' (tier: {tier}). You remember "
                f"they shared this good news or milestone — category={cat}: "
                f"\"{desc}\". In ONE short in-character Rex line, celebrate it "
                f"without turning it into a speech. Warm, dry, no insult at their "
                f"expense. You may ask one low-pressure follow-up like 'how's that "
                f"going?' only if it feels natural."
            )
            if _generate_and_speak(prompt, emotion="happy", purpose="celebration_checkin"):
                try:
                    emo_events.mark_acknowledged(int(ev["id"]))
                except Exception:
                    pass
                _log.info(
                    "consciousness: proactive celebration check-in "
                    "(category=%s, event_id=%s) for person_id=%s",
                    cat, ev.get("id"), engaged_id,
                )
            return

        # ── Trigger B: sustained negative affect ───────────────────────────
        entry = _empathy.peek(engaged_id)
        if not entry:
            _negative_streak_started_at.pop(engaged_id, None)
            return
        result = entry.get("result") or {}
        affect = (result.get("affect") or "neutral").lower()
        confidence = float(result.get("confidence", 0.5) or 0.5)
        sensitivity = (result.get("topic_sensitivity") or "none").lower()

        if not _empathy.is_negative_affect(affect):
            _negative_streak_started_at.pop(engaged_id, None)
            return

        # Require minimum confidence so a single ambiguous reading doesn't
        # start a streak that produces a check-in 30s later.
        min_conf = float(getattr(config, "EMPATHY_MIN_CONFIDENCE_FOR_MODE_CHANGE", 0.55))
        if confidence < min_conf:
            return

        streak_start = _negative_streak_started_at.get(engaged_id)
        if streak_start is None:
            _negative_streak_started_at[engaged_id] = now
            return

        required = float(getattr(config, "EMPATHY_CHECKIN_NEGATIVE_STREAK_SECS", 30.0))
        if (now - streak_start) < required:
            return

        # Tier-shaped framing — content is the same caring move regardless of
        # tier; only the *voice* differs.
        if tier in ("close_friend", "best_friend"):
            voice_clause = (
                "You know them well. Be warm and direct, the way a close friend "
                "would. Light affection underneath."
            )
        elif tier in ("friend",):
            voice_clause = (
                "You know them. Warm, dry, lightly attentive — friend territory."
            )
        else:
            voice_clause = (
                "You don't know them well yet. Reserved warmth, no presumed "
                "familiarity, no personal callbacks. Just notice and offer."
            )

        prompt = (
            f"You're mid-conversation with '{first_name}'. You've noticed they "
            f"sound {affect}"
            f"{' and the topic has been heavy' if sensitivity == 'heavy' else ''}"
            f", and it's been steady. {voice_clause} In ONE short in-character "
            f"Rex line, gently check in on them. Low-pressure, no probing — "
            f"something like 'you've gone quiet on me — long day, or something "
            f"heavier?' Don't fix, don't advise, don't roast. End with a "
            f"question that's easy to deflect."
        )
        emotion = "neutral"
        if _generate_and_speak(prompt, emotion=emotion, purpose="emotional_checkin"):
            _note_emotional_checkin_fired(engaged_id)
            _negative_streak_started_at.pop(engaged_id, None)
            _log.info(
                "consciousness: proactive emotional check-in (B: sustained %s, "
                "streak=%.1fs, conf=%.2f) for person_id=%s",
                affect, now - streak_start, confidence, engaged_id,
            )
    except Exception as exc:
        _log.debug("emotional check-in step error: %s", exc)


def _pending_followups_lock_remove(person_id: int, event_id) -> None:
    """Remove a specific event from _pending_followups so two paths don't both ask."""
    if event_id is None:
        return
    with _followup_lock:
        events = _pending_followups.get(person_id, [])
        kept = [e for e in events if e.get("id") != event_id]
        if kept:
            _pending_followups[person_id] = kept
        else:
            _pending_followups.pop(person_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Face tracking
# ─────────────────────────────────────────────────────────────────────────────

def suspend_face_tracking(seconds: float = 3.0) -> None:
    """Temporarily stop automatic neck tracking during explicit look commands."""
    global _face_tracking_suspended_until
    _face_tracking_suspended_until = max(
        _face_tracking_suspended_until,
        time.monotonic() + max(0.0, float(seconds)),
    )


def resume_face_tracking() -> None:
    """Allow automatic face tracking immediately after a scripted move settles."""
    global _face_tracking_suspended_until
    _face_tracking_suspended_until = 0.0


def _face_x_to_neck_target(x: int) -> float:
    """Map pixel x to neck servo position. Center → neutral; edges → extremes."""
    neck_cfg = config.SERVO_CHANNELS["neck"]
    frac = x / max(config.CAMERA_WIDTH - 1, 1)  # 0.0 (left) → 1.0 (right)
    return float(neck_cfg["min"] + frac * (neck_cfg["max"] - neck_cfg["min"]))


def _frame_size(frame) -> tuple[int, int]:
    try:
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
    except Exception:
        pass
    return int(getattr(config, "CAMERA_WIDTH", 1280)), int(getattr(config, "CAMERA_HEIGHT", 720))


def _face_tracking_key(person: dict, idx: int) -> str:
    if person.get("person_db_id") is not None:
        return f"db:{person.get('person_db_id')}"
    if person.get("face_id"):
        return f"face:{person.get('face_id')}"
    return f"slot:{person.get('id') or idx}"


def _person_tracking_box(person: dict) -> tuple[float, float, float, float] | None:
    box = person.get("face_box") or person.get("bounding_box") or person.get("bbox")
    if isinstance(box, dict):
        box = (
            box.get("x"),
            box.get("y"),
            box.get("w") or box.get("width"),
            box.get("h") or box.get("height"),
        )
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _visible_face_tracking_candidates(people: Optional[list[dict]] = None) -> list[dict]:
    candidates: list[dict] = []
    source_people = world_state.get("people") if people is None else people
    for idx, person in enumerate(source_people or []):
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        box = _person_tracking_box(person)
        if box is None:
            continue
        x, y, w, h = box
        candidates.append({
            "key": _face_tracking_key(person, idx),
            "person_id": person.get("person_db_id"),
            "box": box,
            "center": (x + w / 2.0, y + h / 2.0),
            "area": w * h,
            "live_tracked": bool(person.get("gui_live_tracked") or person.get("live_tracked")),
        })
    return candidates


def _current_servo_position(name: str) -> int:
    cfg = config.SERVO_CHANNELS[name]
    default = int(cfg["neutral"])
    try:
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
        return int(positions.get(name, default))
    except Exception:
        return default


def _clamp_servo(name: str, value: float) -> int:
    cfg = config.SERVO_CHANNELS[name]
    return max(int(cfg["min"]), min(int(cfg["max"]), int(round(value))))


def _limited_tracking_step(name: str, current: int, target: int, max_step: int) -> int:
    max_step = max(1, int(max_step or 1))
    delta = int(target) - int(current)
    if abs(delta) <= max_step:
        return _clamp_servo(name, target)
    return _clamp_servo(name, int(current) + (max_step if delta > 0 else -max_step))


def _adaptive_head_rest_enabled() -> bool:
    return bool(getattr(config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True)) and bool(
        getattr(config, "FACE_TRACKING_VERTICAL_ENABLED", True)
    )


def _adaptive_head_rest_limit(name: str, value: float) -> int:
    cfg = config.SERVO_CHANNELS[name]
    neutral = int(cfg["neutral"])
    attr = (
        "FACE_TRACKING_REST_MAX_LIFT_OFFSET_QUS"
        if name == "headlift"
        else "FACE_TRACKING_REST_MAX_TILT_OFFSET_QUS"
    )
    max_offset = int(getattr(config, attr, 0) or 0)
    if max_offset <= 0:
        return neutral
    low = max(int(cfg["min"]), neutral - max_offset)
    high = min(int(cfg["max"]), neutral + max_offset)
    return max(low, min(high, int(round(value))))


def _adaptive_head_rest_target() -> tuple[int, int]:
    if not _adaptive_head_rest_enabled() or int(_adaptive_head_rest.get("samples") or 0) <= 0:
        return (
            int(config.SERVO_CHANNELS["headlift"]["neutral"]),
            int(config.SERVO_CHANNELS["headtilt"]["neutral"]),
        )
    return (
        _adaptive_head_rest_limit("headlift", float(_adaptive_head_rest.get("lift") or 0.0)),
        _adaptive_head_rest_limit("headtilt", float(_adaptive_head_rest.get("tilt") or 0.0)),
    )


def _face_area_fraction(candidate: dict, frame_w: int, frame_h: int) -> float:
    frame_area = max(1.0, float(frame_w) * float(frame_h))
    try:
        area = float(candidate.get("area") or 0.0)
    except (TypeError, ValueError):
        area = 0.0
    if area <= 0.0:
        box = candidate.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                area = max(0.0, float(box[2]) * float(box[3]))
            except (TypeError, ValueError):
                area = 0.0
    return max(0.0, area / frame_area)


def _note_adaptive_head_rest(
    *,
    candidate: dict,
    frame_w: int,
    frame_h: int,
    lift: int,
    tilt: int,
    now: float,
) -> None:
    if not _adaptive_head_rest_enabled():
        return
    if candidate.get("live_tracked") and not bool(
        getattr(config, "FACE_TRACKING_REST_LEARN_FROM_LIVE_BOXES", False)
    ):
        return
    min_fraction = float(getattr(config, "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION", 0.003) or 0.0)
    if _face_area_fraction(candidate, frame_w, frame_h) < max(0.0, min_fraction):
        return

    alpha = float(getattr(config, "FACE_TRACKING_REST_ADAPT_ALPHA", 0.08) or 0.0)
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return
    old_lift, old_tilt = _adaptive_head_rest_target()
    samples = int(_adaptive_head_rest.get("samples") or 0)
    if samples <= 0:
        alpha = max(alpha, 0.35)
    new_lift = _adaptive_head_rest_limit("headlift", old_lift + alpha * (int(lift) - old_lift))
    new_tilt = _adaptive_head_rest_limit("headtilt", old_tilt + alpha * (int(tilt) - old_tilt))
    _adaptive_head_rest.update({
        "lift": new_lift,
        "tilt": new_tilt,
        "samples": samples + 1,
        "updated_at": now,
    })


def _step_adaptive_head_rest_return(
    servo_mod,
    now: float,
    *,
    lost_age_secs: float | None = None,
) -> bool:
    if not _adaptive_head_rest_enabled() or int(_adaptive_head_rest.get("samples") or 0) <= 0:
        return False
    delay = float(getattr(config, "FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS", 0.8) or 0.0)
    if lost_age_secs is not None and lost_age_secs < max(0.0, delay):
        return False
    try:
        if getattr(servo_mod, "manual_override_enabled", lambda: False)():
            return False
        if getattr(servo_mod, "speech_motion_active", lambda: False)():
            return False
    except Exception:
        return False

    target_lift, target_tilt = _adaptive_head_rest_target()
    current_neck = _current_servo_position("neck")
    current_lift = _current_servo_position("headlift")
    current_tilt = _current_servo_position("headtilt")
    max_step = int(getattr(config, "FACE_TRACKING_REST_RETURN_MAX_STEP_QUS", 55) or 55)
    next_lift = _limited_tracking_step("headlift", current_lift, target_lift, max_step)
    next_tilt = _limited_tracking_step("headtilt", current_tilt, target_tilt, max_step)

    lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
    tilt_ch = int(config.SERVO_CHANNELS["headtilt"]["ch"])
    updates: dict[int, int] = {}
    if abs(next_lift - current_lift) >= 2:
        updates[lift_ch] = next_lift
    if abs(next_tilt - current_tilt) >= 2:
        updates[tilt_ch] = next_tilt
    if not updates:
        return False
    try:
        servo_mod.set_motion_profile(
            list(updates.keys()),
            speed=int(getattr(config, "FACE_TRACKING_REST_SERVO_SPEED", 35)),
            acceleration=int(getattr(config, "FACE_TRACKING_REST_SERVO_ACCELERATION", 6)),
        )
    except Exception as exc:
        _log.debug("adaptive rest motion profile failed: %s", exc)
    servo_mod.set_servos(updates)
    try:
        servo_mod.set_face_tracking_baseline(
            neck=current_neck,
            lift=updates.get(lift_ch, current_lift),
            tilt=updates.get(tilt_ch, current_tilt),
        )
    except Exception as exc:
        _log.debug("adaptive rest baseline update failed: %s", exc)
    return True


def _tracking_error_reversed(
    *,
    key: str,
    previous_key: Optional[str],
    previous_error: Optional[float],
    current_error: float,
    dead_zone: float,
    now: float,
    previous_at: float,
) -> bool:
    if previous_key != key or previous_error is None:
        return False
    if (now - previous_at) > 1.0:
        return False
    if abs(previous_error) <= dead_zone or abs(current_error) <= dead_zone:
        return False
    return (previous_error < 0 < current_error) or (previous_error > 0 > current_error)


def _maybe_log_face_tracking_move(
    *,
    now: float,
    candidate: dict,
    frame_w: int,
    frame_h: int,
    current: dict[str, int],
    updates: dict[int, int],
    error_x: float,
    error_y: float,
) -> None:
    global _last_face_tracking_log_at

    interval = float(getattr(config, "FACE_TRACKING_LOG_INTERVAL_SECS", 2.0) or 0.0)
    if interval > 0 and (now - _last_face_tracking_log_at) < interval:
        return
    _last_face_tracking_log_at = now

    neck_ch = int(config.SERVO_CHANNELS["neck"]["ch"])
    lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
    tilt_ch = int(config.SERVO_CHANNELS["headtilt"]["ch"])
    _log.info(
        "face_tracking: lock=%s person_id=%s center=(%.1f,%.1f) frame=%dx%d "
        "error=(%.1f,%.1f) current=(neck=%d,lift=%d,tilt=%d) "
        "target=(neck=%s,lift=%s,tilt=%s)",
        candidate.get("key"),
        candidate.get("person_id"),
        candidate.get("center", (0.0, 0.0))[0],
        candidate.get("center", (0.0, 0.0))[1],
        frame_w,
        frame_h,
        error_x,
        error_y,
        current["neck"],
        current["headlift"],
        current["headtilt"],
        updates.get(neck_ch, "-"),
        updates.get(lift_ch, "-"),
        updates.get(tilt_ch, "-"),
    )


def _candidate_matches_speaker_gaze(candidate: dict, intent: Optional[dict]) -> bool:
    if not intent:
        return False
    person_id = intent.get("person_id")
    if person_id is not None:
        try:
            return int(candidate.get("person_id")) == int(person_id)
        except Exception:
            return False
    return bool(intent.get("unknown_voice") and candidate.get("person_id") is None)


def _speaker_gaze_intent_needs_specific_target(intent: Optional[dict]) -> bool:
    return bool(intent and (intent.get("person_id") is not None or intent.get("unknown_voice")))


def _speaker_gaze_lock_matches_intent(intent: Optional[dict]) -> bool:
    if not intent:
        return False
    person_id = intent.get("person_id")
    if person_id is not None:
        try:
            return int(_face_tracking_lock.get("person_id")) == int(person_id)
        except Exception:
            return False
    return bool(intent.get("unknown_voice") and _face_tracking_lock.get("person_id") is None)


def _speaker_gaze_request_search(now: float) -> Optional[dict]:
    with _speaker_gaze_lock:
        if not _speaker_gaze_intent:
            return None
        _speaker_gaze_intent["search_requested"] = True
        if not _speaker_gaze_intent.get("search_started_at"):
            _speaker_gaze_intent["search_started_at"] = now
        return dict(_speaker_gaze_intent)


def _speaker_gaze_candidate(candidates: list[dict], intent: Optional[dict]) -> Optional[dict]:
    if not intent or not candidates:
        return None
    person_id = intent.get("person_id")
    if person_id is not None:
        matches = []
        for item in candidates:
            item_person_id = item.get("person_id")
            if item_person_id is None:
                continue
            try:
                if int(item_person_id) == int(person_id):
                    matches.append(item)
            except Exception:
                continue
        if matches:
            return max(matches, key=lambda item: item["area"])
        return None
    if intent.get("unknown_voice"):
        unknowns = [item for item in candidates if item.get("person_id") is None]
        if unknowns:
            return max(unknowns, key=lambda item: item["area"])
    return None


def _build_speaker_gaze_search_plan(reason: str = "startup") -> list[tuple]:
    """Build one randomized, two-axis room-scan pass.

    Returns a list of ``(neck_frac, vert_frac)`` waypoints, where ``neck_frac`` is in
    ``[-1, 1]`` (full left .. full right, or ``None`` to hold the current heading) and
    ``vert_frac`` is in ``[0, 1]`` (level/adaptive-rest .. fully down). Every waypoint
    moves BOTH the neck and the head pitch at once, so Rex sweeps the room diagonally
    instead of one axis at a time, and the lane order + exact targets are reshuffled
    each pass so he doesn't look around the same predictable way every boot.

    The pass always opens by dropping the gaze straight down without turning (people
    are usually seated, below the camera) and closes back at neutral/level.
    """
    points = int(getattr(config, "SPEAKER_GAZE_SEARCH_POINTS", 5) or 5)
    points = max(3, min(points, 12))

    # The STARTUP room scan sweeps the full neck range (find anyone in the room). A
    # mid-conversation speaker search stays gentler — the talker is usually right in
    # front of Rex, so a full-range swing just thrashes his head and yanks the camera
    # off the face it was tracking. Scale the lane spread down for non-startup reasons.
    if str(reason or "").lower() in ("startup", "scan"):
        neck_scale = 1.0
    else:
        neck_scale = float(getattr(config, "SPEAKER_GAZE_SEARCH_SPEECH_NECK_SCALE", 0.45))

    # Lanes spaced across the (scaled) left..right range, each jittered off its lane
    # centre and paired with its own random (down-biased) pitch — so no two scans
    # trace the same path while the room still gets swept.
    step = 2.0 / (points - 1)
    lanes: list[tuple] = []
    for i in range(points):
        centre = -1.0 + step * i
        neck_frac = max(-1.0, min(1.0, centre + random.uniform(-step * 0.4, step * 0.4)))
        vert_frac = random.uniform(0.2, 1.0)
        lanes.append((neck_frac * neck_scale, vert_frac))
    random.shuffle(lanes)

    plan: list[tuple] = [(None, random.uniform(0.9, 1.0))]  # look down first, no turn
    plan.extend(lanes)
    plan.append((0.0, 0.0))  # recentre level so the head parks neutral after the pass
    return plan


def _speaker_gaze_search_label(neck_frac, vert_frac: float) -> str:
    """Short human-readable label for a search waypoint (for logs / telemetry)."""
    if neck_frac is None:
        horiz = "hold"
    elif neck_frac <= -0.15:
        horiz = "left"
    elif neck_frac >= 0.15:
        horiz = "right"
    else:
        horiz = "center"
    if vert_frac >= 0.66:
        vert = "down"
    elif vert_frac >= 0.25:
        vert = "low"
    else:
        vert = "level"
    return f"{horiz}_{vert}"


def _speaker_gaze_search_targets(neck_frac, vert_frac: float) -> dict[int, int]:
    """Resolve a ``(neck_frac, vert_frac)`` search waypoint to servo targets.

    ``neck_frac``: ``-1..1`` (left..right), or ``None`` to hold the current heading.
    ``vert_frac``: ``0..1`` (level/adaptive-rest .. fully down). Visor is held open so
    the camera keeps a clear view while searching.
    """
    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]
    visor_cfg = config.SERVO_CHANNELS["visor"]

    neck_ch = int(neck_cfg["ch"])
    lift_ch = int(lift_cfg["ch"])
    tilt_ch = int(tilt_cfg["ch"])
    visor_ch = int(visor_cfg["ch"])

    neutral_neck = int(neck_cfg["neutral"])
    rest_lift, rest_tilt = _adaptive_head_rest_target()

    neck_fraction = float(getattr(config, "SPEAKER_GAZE_SEARCH_NECK_FRACTION", 0.42))
    down_tilt_fraction = float(getattr(config, "SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION", 0.72))
    down_lift_fraction = float(getattr(config, "SPEAKER_GAZE_SEARCH_DOWN_LIFT_FRACTION", 0.18))

    if neck_frac is None:
        neck = _current_servo_position("neck")
    elif neck_frac >= 0:
        neck = neutral_neck + (int(neck_cfg["max"]) - neutral_neck) * neck_fraction * float(neck_frac)
    else:
        neck = neutral_neck - (neutral_neck - int(neck_cfg["min"])) * neck_fraction * (-float(neck_frac))

    # Down end of the pitch range. Tilt is inverted (larger PWM points lower); lift
    # points lower with smaller PWM. Never push past the adaptive rest toward "up".
    down_tilt = max(
        _clamp_servo(
            "headtilt",
            int(tilt_cfg["neutral"]) + (int(tilt_cfg["max"]) - int(tilt_cfg["neutral"])) * down_tilt_fraction,
        ),
        rest_tilt,
    )
    down_lift = min(
        _clamp_servo(
            "headlift",
            int(lift_cfg["neutral"]) - (int(lift_cfg["neutral"]) - int(lift_cfg["min"])) * down_lift_fraction,
        ),
        rest_lift,
    )

    vert = max(0.0, min(1.0, float(vert_frac)))
    lift = rest_lift + (down_lift - rest_lift) * vert
    tilt = rest_tilt + (down_tilt - rest_tilt) * vert

    targets: dict[int, int] = {
        visor_ch: int(visor_cfg["max"]),
        neck_ch: neck,
        lift_ch: lift,
        tilt_ch: tilt,
    }
    return {
        channel: _clamp_servo(_CHANNEL_TO_SERVO_NAME[channel], value)
        if channel in _CHANNEL_TO_SERVO_NAME else int(value)
        for channel, value in targets.items()
    }


_CHANNEL_TO_SERVO_NAME = {
    int(cfg["ch"]): name
    for name, cfg in config.SERVO_CHANNELS.items()
}


def _step_speaker_gaze_search(servo_mod, intent: Optional[dict], now: float) -> Optional[str]:
    if not intent or not bool(intent.get("search_requested")):
        return None
    if not bool(getattr(config, "SPEAKER_GAZE_ENABLED", True)):
        return None
    try:
        if getattr(servo_mod, "manual_override_enabled", lambda: False)():
            return None
    except Exception:
        pass

    interval = float(getattr(config, "SPEAKER_GAZE_SEARCH_INTERVAL_SECS", 0.70) or 0.70)
    with _speaker_gaze_lock:
        if not _speaker_gaze_intent:
            return None
        last_search_at = float(_speaker_gaze_intent.get("last_search_at") or 0.0)
        if last_search_at > 0.0 and (now - last_search_at) < max(0.1, interval):
            return None
        plan = _speaker_gaze_intent.get("search_plan")
        plan_idx = int(_speaker_gaze_intent.get("search_plan_index") or 0)
        if not plan or plan_idx >= len(plan):
            # Fresh randomized pass (also re-rolls if the search outlasts one pass).
            plan = _build_speaker_gaze_search_plan(_speaker_gaze_intent.get("reason", "startup"))
            _speaker_gaze_intent["search_plan"] = plan
            plan_idx = 0
        neck_frac, vert_frac = plan[plan_idx]
        _speaker_gaze_intent["search_plan_index"] = plan_idx + 1
        _speaker_gaze_intent["search_index"] = int(_speaker_gaze_intent.get("search_index") or 0) + 1
        _speaker_gaze_intent["last_search_at"] = now
        if not _speaker_gaze_intent.get("search_started_at"):
            _speaker_gaze_intent["search_started_at"] = now

    pose = _speaker_gaze_search_label(neck_frac, vert_frac)
    targets = _speaker_gaze_search_targets(neck_frac, vert_frac)
    try:
        servo_mod.set_motion_profile(
            list(targets.keys()),
            speed=int(getattr(config, "SPEAKER_GAZE_SEARCH_SERVO_SPEED", 130)),
            acceleration=int(getattr(config, "SPEAKER_GAZE_SEARCH_SERVO_ACCELERATION", 20)),
        )
    except Exception as exc:
        _log.debug("speaker gaze search motion profile failed: %s", exc)
    servo_mod.set_servos(targets)
    try:
        servo_mod.set_face_tracking_baseline(
            neck=targets.get(int(config.SERVO_CHANNELS["neck"]["ch"])),
            lift=targets.get(int(config.SERVO_CHANNELS["headlift"]["ch"])),
            tilt=targets.get(int(config.SERVO_CHANNELS["headtilt"]["ch"])),
        )
    except Exception as exc:
        _log.debug("speaker gaze search baseline update failed: %s", exc)
    _log.info(
        "[speaker_gaze] search pose=%s reason=%s person_id=%s unknown=%s",
        pose,
        intent.get("reason"),
        intent.get("person_id"),
        bool(intent.get("unknown_voice")),
    )
    return pose


def _record_face_tracking_state(
    *,
    locked: bool,
    visible: bool,
    holding_lost_lock: bool = False,
    candidate: dict | None = None,
    lost_age_secs: float | None = None,
    searching: bool = False,
    search_reason: str | None = None,
    search_pose: str | None = None,
) -> None:
    try:
        self_state = world_state.get("self_state")
        tracking = dict(self_state.get("face_tracking") or {})
        rest_lift, rest_tilt = _adaptive_head_rest_target()
        tracking.update({
            "locked": bool(locked),
            "visible": bool(visible),
            "holding_lost_lock": bool(holding_lost_lock),
            "searching": bool(searching),
            "search_reason": search_reason if searching else None,
            "search_pose": search_pose if searching else None,
            "lock_key": _face_tracking_lock.get("key") if locked else None,
            "person_id": (
                candidate.get("person_id")
                if candidate is not None
                else _face_tracking_lock.get("person_id") if locked else None
            ),
            "lost_age_secs": round(lost_age_secs, 2) if lost_age_secs is not None else None,
            "adaptive_rest": {
                "lift": rest_lift,
                "tilt": rest_tilt,
                "samples": int(_adaptive_head_rest.get("samples") or 0),
            },
        })
        if candidate is not None:
            tracking["box"] = candidate.get("box")
            tracking["center"] = candidate.get("center")
            tracking["last_seen_at"] = time.time()
        elif not locked:
            tracking["box"] = None
            tracking["center"] = None
            tracking["last_seen_at"] = None
        self_state["face_tracking"] = tracking
        world_state.update("self_state", self_state)
    except Exception as exc:
        _log.debug("face tracking state update failed: %s", exc)


def _step_face_tracking(frame, people: Optional[list[dict]] = None) -> None:
    """
    Center the current face lock in Rex's camera frame.

    Person recognition owns face detection for the tick; this step consumes its
    visible boxes, keeps a sticky lock through brief misses, and moves the neck
    plus vertical head pose toward image center.
    """
    global _neck_smooth, _face_tracking_lock
    global _face_tracking_last_error_key, _face_tracking_last_error_x
    global _face_tracking_last_error_y, _face_tracking_last_error_at

    if state_module.get_state() == State.SLEEP:
        return
    if time.monotonic() < _face_tracking_suspended_until:
        return
    if frame is None:
        return

    try:
        from hardware import servos as servo_mod

        # While listening motion owns the head (gentle nods during the
        # transcription/LLM/TTS wait), don't fight it with face centering.
        if getattr(servo_mod, "listening_motion_active", lambda: False)():
            return

        now = time.monotonic()
        candidates = _visible_face_tracking_candidates(people)
        speaker_intent = _speaker_gaze_current_intent(now)
        lock_key = _face_tracking_lock.get("key")
        last_seen = float(_face_tracking_lock.get("last_seen_at") or 0.0)
        lost_hold_secs = float(getattr(config, "FACE_TRACKING_LOST_HOLD_SECS", 4.0) or 0.0)
        lost_search_after = float(getattr(config, "SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS", 0.45) or 0.0)

        candidate = _speaker_gaze_candidate(candidates, speaker_intent)
        speaker_target_missing = (
            candidate is None
            and _speaker_gaze_intent_needs_specific_target(speaker_intent)
        )
        if speaker_target_missing and not bool(speaker_intent.get("search_requested")):
            lock_matches_speaker = _speaker_gaze_lock_matches_intent(speaker_intent)
            lost_age = (now - last_seen) if last_seen else None
            still_in_lost_grace = bool(
                lock_key
                and lock_matches_speaker
                and lost_age is not None
                and lost_age < max(0.0, lost_search_after)
            )
            if not still_in_lost_grace:
                speaker_intent = _speaker_gaze_request_search(now) or speaker_intent
        if candidate is None and lock_key:
            if (
                speaker_target_missing
                and speaker_intent
                and bool(speaker_intent.get("search_requested"))
            ):
                search_pose = _step_speaker_gaze_search(servo_mod, speaker_intent, now)
                _record_face_tracking_state(
                    locked=False,
                    visible=False,
                    searching=True,
                    search_reason=speaker_intent.get("reason"),
                    search_pose=search_pose,
                )
                return
            candidate = next((item for item in candidates if item["key"] == lock_key), None)
            if candidate is None and last_seen and (now - last_seen) <= lost_hold_secs:
                lost_age = now - last_seen
                search_pose = None
                if (
                    speaker_intent
                    and bool(speaker_intent.get("search_requested"))
                    and lost_age >= max(0.0, lost_search_after)
                ):
                    search_pose = _step_speaker_gaze_search(servo_mod, speaker_intent, now)
                _record_face_tracking_state(
                    locked=True,
                    visible=False,
                    holding_lost_lock=True,
                    lost_age_secs=lost_age,
                    searching=bool(speaker_intent and speaker_intent.get("search_requested")),
                    search_reason=(speaker_intent or {}).get("reason"),
                    search_pose=search_pose,
                )
                return

        if candidate is None and candidates:
            if (
                speaker_intent
                and bool(speaker_intent.get("search_requested"))
                and _speaker_gaze_intent_needs_specific_target(speaker_intent)
            ):
                search_pose = _step_speaker_gaze_search(servo_mod, speaker_intent, now)
                _record_face_tracking_state(
                    locked=False,
                    visible=False,
                    searching=True,
                    search_reason=speaker_intent.get("reason"),
                    search_pose=search_pose,
                )
                return
            candidate = candidates[0] if len(candidates) == 1 else max(candidates, key=lambda item: item["area"])

        if candidate is None:
            if speaker_intent and bool(speaker_intent.get("search_requested")):
                search_pose = _step_speaker_gaze_search(servo_mod, speaker_intent, now)
                _face_tracking_lock = {}
                _record_face_tracking_state(
                    locked=False,
                    visible=False,
                    searching=True,
                    search_reason=speaker_intent.get("reason"),
                    search_pose=search_pose,
                )
                return
            lost_age = (now - last_seen) if last_seen else None
            _face_tracking_lock = {}
            _step_adaptive_head_rest_return(
                servo_mod,
                now,
                lost_age_secs=lost_age,
            )
            _record_face_tracking_state(locked=False, visible=False)
            return

        speaker_target = _candidate_matches_speaker_gaze(candidate, speaker_intent)
        if speaker_target:
            _speaker_gaze_note_acquired(candidate)

        _face_tracking_lock = {
            "key": candidate["key"],
            "person_id": candidate.get("person_id"),
            "last_seen_at": now,
        }

        frame_w, frame_h = _frame_size(frame)
        neck_cfg = config.SERVO_CHANNELS["neck"]
        lift_cfg = config.SERVO_CHANNELS["headlift"]
        tilt_cfg = config.SERVO_CHANNELS["headtilt"]
        neck_ch = int(neck_cfg["ch"])
        lift_ch = int(lift_cfg["ch"])
        tilt_ch = int(tilt_cfg["ch"])

        alpha = max(0.0, min(1.0, 1.0 - float(getattr(config, "TRACKING_SMOOTHING_FACTOR", 0.2))))
        gain = float(getattr(config, "FACE_TRACKING_CENTERING_GAIN", 1.15))
        vertical_gain = float(getattr(config, "FACE_TRACKING_VERTICAL_GAIN", 0.55))
        dead_zone = float(getattr(config, "TRACKING_DEAD_ZONE_PX", 40))
        neck_max_step = int(getattr(config, "FACE_TRACKING_NECK_MAX_STEP_QUS", 420))
        lift_max_step = int(getattr(config, "FACE_TRACKING_LIFT_MAX_STEP_QUS", 300))
        tilt_max_step = int(getattr(config, "FACE_TRACKING_TILT_MAX_STEP_QUS", 130))
        if speaker_target:
            gain = float(getattr(config, "SPEAKER_GAZE_ACTIVE_CENTERING_GAIN", gain))
            vertical_gain = float(getattr(config, "SPEAKER_GAZE_ACTIVE_VERTICAL_GAIN", vertical_gain))
            dead_zone = float(getattr(config, "SPEAKER_GAZE_ACTIVE_DEAD_ZONE_PX", dead_zone))
            neck_max_step = int(getattr(config, "SPEAKER_GAZE_NECK_MAX_STEP_QUS", neck_max_step))
            lift_max_step = int(getattr(config, "SPEAKER_GAZE_LIFT_MAX_STEP_QUS", lift_max_step))
            tilt_max_step = int(getattr(config, "SPEAKER_GAZE_TILT_MAX_STEP_QUS", tilt_max_step))
        cx, cy = candidate["center"]
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0

        current_neck = _current_servo_position("neck")
        current_lift = _current_servo_position("headlift")
        current_tilt = _current_servo_position("headtilt")
        current_positions = {
            "neck": current_neck,
            "headlift": current_lift,
            "headtilt": current_tilt,
        }
        updates: dict[int, int] = {}

        error_x = cx - frame_cx
        error_y = cy - frame_cy
        candidate_key = str(candidate.get("key") or "")
        reversal_damping = float(getattr(config, "FACE_TRACKING_REVERSAL_DAMPING", 0.35))
        reversal_damping = max(0.05, min(1.0, reversal_damping))
        live_damping = float(getattr(config, "FACE_TRACKING_LIVE_BOX_DAMPING", 0.45))
        live_damping = max(0.05, min(1.0, live_damping))
        if candidate.get("live_tracked"):
            gain *= live_damping
            vertical_gain *= live_damping
            neck_max_step = max(1, int(neck_max_step * live_damping))
            lift_max_step = max(1, int(lift_max_step * live_damping))
            tilt_max_step = max(1, int(tilt_max_step * live_damping))
        if _tracking_error_reversed(
            key=candidate_key,
            previous_key=_face_tracking_last_error_key,
            previous_error=_face_tracking_last_error_x,
            current_error=error_x,
            dead_zone=dead_zone,
            now=now,
            previous_at=_face_tracking_last_error_at,
        ):
            gain *= reversal_damping
            neck_max_step = max(1, int(neck_max_step * reversal_damping))
        if _tracking_error_reversed(
            key=candidate_key,
            previous_key=_face_tracking_last_error_key,
            previous_error=_face_tracking_last_error_y,
            current_error=error_y,
            dead_zone=dead_zone,
            now=now,
            previous_at=_face_tracking_last_error_at,
        ):
            vertical_gain *= reversal_damping
            lift_max_step = max(1, int(lift_max_step * reversal_damping))
            tilt_max_step = max(1, int(tilt_max_step * reversal_damping))
        if abs(error_x) > dead_zone and frame_cx > 0:
            neck_span = (int(neck_cfg["max"]) - int(neck_cfg["min"])) / 2.0
            target_neck = _clamp_servo(
                "neck",
                current_neck + (error_x / frame_cx) * neck_span * gain,
            )
            next_neck = _clamp_servo("neck", current_neck + alpha * (target_neck - current_neck))
            next_neck = _limited_tracking_step(
                "neck",
                current_neck,
                next_neck,
                neck_max_step,
            )
            if abs(next_neck - current_neck) >= 2:
                updates[neck_ch] = next_neck
                _neck_smooth = float(next_neck)
        else:
            _neck_smooth = float(current_neck)

        if bool(getattr(config, "FACE_TRACKING_VERTICAL_ENABLED", True)):
            if abs(error_y) > dead_zone and frame_cy > 0:
                lift_span = (int(lift_cfg["max"]) - int(lift_cfg["min"])) / 2.0
                tilt_span = (int(tilt_cfg["max"]) - int(tilt_cfg["min"])) / 2.0
                target_lift = _clamp_servo(
                    "headlift",
                    current_lift - (error_y / frame_cy) * lift_span * vertical_gain,
                )
                target_tilt = _clamp_servo(
                    "headtilt",
                    current_tilt + (error_y / frame_cy) * tilt_span * vertical_gain,
                )
                next_lift = _clamp_servo(
                    "headlift",
                    current_lift + alpha * (target_lift - current_lift),
                )
                next_tilt = _clamp_servo(
                    "headtilt",
                    current_tilt + alpha * (target_tilt - current_tilt),
                )
                next_lift = _limited_tracking_step(
                    "headlift",
                    current_lift,
                    next_lift,
                    lift_max_step,
                )
                next_tilt = _limited_tracking_step(
                    "headtilt",
                    current_tilt,
                    next_tilt,
                    tilt_max_step,
                )
                if abs(next_lift - current_lift) >= 2:
                    updates[lift_ch] = next_lift
                if abs(next_tilt - current_tilt) >= 2:
                    updates[tilt_ch] = next_tilt

        baseline_neck = updates.get(neck_ch, current_neck)
        baseline_lift = updates.get(lift_ch, current_lift)
        baseline_tilt = updates.get(tilt_ch, current_tilt)
        _note_adaptive_head_rest(
            candidate=candidate,
            frame_w=frame_w,
            frame_h=frame_h,
            lift=baseline_lift,
            tilt=baseline_tilt,
            now=now,
        )
        if updates or abs(error_x) > dead_zone or abs(error_y) > dead_zone:
            _maybe_log_face_tracking_move(
                now=now,
                candidate=candidate,
                frame_w=frame_w,
                frame_h=frame_h,
                current=current_positions,
                updates=updates,
                error_x=error_x,
                error_y=error_y,
            )
        if updates:
            try:
                channels = [neck_ch, lift_ch, tilt_ch]
                servo_mod.set_motion_profile(
                    channels,
                    speed=int(getattr(config, "FACE_TRACKING_SERVO_SPEED", 140)),
                    acceleration=int(getattr(config, "FACE_TRACKING_SERVO_ACCELERATION", 16)),
                )
            except Exception as exc:
                _log.debug("face tracking motion profile update failed: %s", exc)
            servo_mod.set_servos(updates)
        servo_mod.set_face_tracking_baseline(
            neck=baseline_neck,
            lift=baseline_lift,
            tilt=baseline_tilt,
        )
        _face_tracking_last_error_key = candidate_key
        _face_tracking_last_error_x = float(error_x)
        _face_tracking_last_error_y = float(error_y)
        _face_tracking_last_error_at = now
        _record_face_tracking_state(locked=True, visible=True, candidate=candidate)

    except Exception as exc:
        _log.debug("face tracking step error: %s", exc)


def _live_face_tracking_people(frame) -> list[dict]:
    """Use optical flow to keep face boxes fresh between recognition ticks."""
    global _face_tracking_tracker

    people = world_state.get("people") or []
    if not bool(getattr(config, "FACE_TRACKING_OPTICAL_FLOW_ENABLED", True)):
        return people
    if frame is None:
        return people

    try:
        if _face_tracking_tracker is None:
            from gui.live_face_tracker import LiveFaceBoxTracker

            _face_tracking_tracker = LiveFaceBoxTracker(
                stale_secs=max(
                    0.0,
                    float(getattr(config, "FACE_TRACKING_LIVE_BOX_MAX_EXTRAPOLATION_SECS", 0.65) or 0.0),
                )
            )
        return _face_tracking_tracker.update(frame, people)
    except Exception as exc:
        _log.debug("live face-box tracking failed: %s", exc)
        return people


def _face_tracking_loop() -> None:
    """High-rate head-pose loop, separate from slower cognition/social ticks."""
    interval = max(0.02, float(getattr(config, "FACE_TRACKING_LOOP_INTERVAL_SECS", 0.08) or 0.08))
    while not _stop_event.is_set():
        try:
            from vision.camera import get_frame

            frame = get_frame()
            people = _live_face_tracking_people(frame)
            _step_face_tracking(frame, people)
        except Exception as exc:
            _log.debug("fast face tracking loop error: %s", exc)
        _stop_event.wait(interval)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def _loop() -> None:
    global _last_snapshot, _last_micro_behavior_at, _neck_smooth

    interval = getattr(config, "CONSCIOUSNESS_LOOP_INTERVAL_SECS", 1.0)
    last_tick = time.monotonic()
    _last_micro_behavior_at = time.monotonic()

    while not _stop_event.is_set():
        tick_start = time.monotonic()
        elapsed = tick_start - last_tick
        last_tick = tick_start

        try:
            # 0. Situation assessment — evaluated once per tick, passed to all steps
            profile = _situation_assessor.evaluate()
            _start_governor_cycle(profile)

            # Apply family-safe personality overrides based on current scene
            try:
                from intelligence.personality import set_family_safe
                set_family_safe(profile.force_family_safe)
            except Exception as exc:
                _log.debug("family_safe apply error: %s", exc)

            # 1. Anger cooldown
            _step_anger_cooldown()

            # 2. Mood decay
            _step_mood_decay(elapsed)

            # Grab current camera frame once — reused by steps 5 and 11
            try:
                from vision.camera import get_frame
                frame = get_frame()
            except Exception:
                frame = None
            _note_startup_camera_frame(frame)

            # 3. Interoception
            _step_interoception()

            # 4. Chronoception
            _step_chronoception()

            # 5. Person recognition (may update world_state.people)
            _step_person_recognition(frame)

            # 5b. Pose/proxemic social context. Face recognition fills identity
            # and distance; pose fills engagement; social analysis derives crowd mode.
            _step_body_social_analysis(frame)
            profile = _situation_assessor.evaluate()
            try:
                from intelligence.personality import set_family_safe
                set_family_safe(profile.force_family_safe)
            except Exception as exc:
                _log.debug("family_safe refresh error: %s", exc)

            # Snapshot after recognition/social analysis so steps 6–11 see identified persons
            snapshot = world_state.snapshot()

            # 5c. Celebrity overrides. These own the first conversational beat
            # before ordinary greetings or ambient remarks.
            if (
                _step_jeff_history_hunters_detection(snapshot, profile)
                or _step_jt_volleyball_detection(snapshot, profile)
            ):
                _finish_governor_cycle()
                _last_snapshot = snapshot
                sleep_for = max(0.0, interval - (time.monotonic() - tick_start))
                _stop_event.wait(sleep_for)
                continue

            # 6. Startup group greeting. This runs before individual follow-up
            # memory checks so a room with two known people gets one social
            # opening instead of stacked callbacks.
            _step_startup_group_greeting(snapshot, profile)

            # 6b. Follow-up check
            _step_followup_check(snapshot)

            # 7. Disengagement detection
            _step_disengagement(snapshot, profile)

            # 7b. Personal-space boundary joke when someone is comically too close
            _step_personal_space(snapshot, profile)

            # 8. Proactive reactions
            _step_proactive_reactions(snapshot, profile)

            # 9. Idle micro-behaviors
            _step_idle_micro_behavior(snapshot, profile)

            # 10. Presence tracking (departure / return reactions)
            _step_presence_tracking(snapshot, profile)

            # 10a. If startup found no confirmed person after the scan, Rex may
            # acknowledge uncertainty once. Presence/identity gets first claim.
            _step_startup_empty_room_comment(snapshot, profile)

            # 10b. Social inquiry — ask engaged person about unknown newcomer
            _step_relationship_inquiry(snapshot, profile)

            # 10c. Third-party awareness — call out a lingering bystander
            _step_third_party_awareness(snapshot, profile)

            # 10c2. Group turn-taking — softly invite a quiet visible known person
            _step_group_turn_taking(snapshot, profile)

            # 10c3. Group lull — when the whole visible group goes quiet, reopen gently
            _step_group_lull(snapshot, profile)

            # 10d. Holiday plans — ask engaged person about upcoming holidays
            _step_holiday_plans(snapshot, profile)

            # 10d2. Weekly small talk — Fri-eve / Sun-eve / Mon-morning prompts
            _step_weekly_smalltalk(snapshot, profile)

            # 10d3. Proactive emotional check-in — acknowledge an unfollowed-up
            # sensitive event, or notice sustained negative affect mid-conversation.
            _step_emotional_checkin(snapshot, profile)

            # 10d4. Visual curiosity — when conversation goes quiet, look once
            # and ask a concrete question about a visible non-sensitive detail.
            _step_visual_curiosity(snapshot, profile)

            # 10e. Overheard chime-in — react when someone talks ABOUT Rex
            _step_overheard_chime_in(snapshot, profile)

            # 10f. GUI face-mood telemetry — keep the dashboard's face-box label
            # aligned with the current visible expression when the scene is unambiguous.
            _step_gui_mood_telemetry(snapshot, frame)

            # 10f2. Long-term expression disposition memory — sample the local
            # MediaPipe expression stream at a low rate for known people.
            _step_disposition_memory(snapshot)

            # 10g. Smile reaction — after Rex lands a joke/snarky aside, notice
            # if the target visibly cracks a smile and answer it once.
            _step_smile_reaction(snapshot, profile)

            # 10h. Facial expression reactions — gently notice clear surprise,
            # frowns, and brow furrows from the local MediaPipe telemetry.
            _step_facial_expression_reactions(snapshot, profile)

            # 11. Face tracking runs in a dedicated high-rate loop so gaze
            # correction is not gated by this slower social/cognition tick.

            _finish_governor_cycle()

            # Preserve snapshot for next iteration's change detection
            _last_snapshot = snapshot

        except Exception as exc:
            _finish_governor_cycle()
            _log.error("consciousness loop unhandled error: %s", exc)

        # Sleep for the remainder of the interval (or yield immediately if overrun)
        sleep_for = max(0.0, interval - (time.monotonic() - tick_start))
        _stop_event.wait(sleep_for)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start() -> None:
    """Start the consciousness daemon thread. No-op if already running."""
    global _thread, _response_wait_until, _last_proactive_speech_at, _pending_departure_keys
    global _face_tracking_thread, _face_tracking_tracker
    global _identity_prompt_reply_until
    global _last_rex_utterance_text, _last_memory_hint_text, _last_memory_hint_at
    global _last_memory_hint_person_id
    global _recent_engaged_person_id, _recent_engaged_touch_at
    global _process_started_iso, _process_started_mono
    global _startup_group_signature, _startup_group_seen_at, _startup_solo_seen_at
    global _startup_camera_first_frame_at, _startup_presence_evidence_at
    global _startup_presence_evidence_reason
    global _last_pose_analysis_at
    global _last_weather_reaction_at
    global _face_tracking_last_error_key, _face_tracking_last_error_x
    global _face_tracking_last_error_y, _face_tracking_last_error_at
    global _last_face_seen_at
    global _smile_reaction_watch, _last_smile_reaction_at
    global _last_facial_expression_reaction_at
    global _last_startle_sound_reaction_at
    if _thread and _thread.is_alive():
        _log.debug("consciousness already running")
        return
    _process_started_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    _process_started_mono = time.monotonic()
    _stop_event.clear()
    _pending_identity_prompt.clear()
    _identity_prompt_in_flight.clear()
    _identity_prompt_reply_until = 0.0
    _pending_relationship_prompt.clear()
    _pending_relationship_context.clear()
    _asked_relationship_slots.clear()
    _unknown_first_seen_at.clear()
    _proactive_speech_pending.clear()
    _greeted_this_session.clear()
    _jeff_celebrity_greeted_this_session.clear()
    _pending_jeff_celebrity_greetings.clear()
    _jt_volleyball_greeted_this_session.clear()
    _pending_jt_volleyball_greetings.clear()
    _pending_departure_keys.clear()
    _first_missing_at.clear()
    _confirmed_absent_at.clear()
    _first_sight_seen_at.clear()
    _last_presence_reaction_at.clear()
    _animal_seen_signatures.clear()
    _animal_reacted_at.clear()
    _pending_animal_arrivals.clear()
    _last_startle_sound_reaction_at = 0.0
    _acknowledged_weather_signatures.clear()
    _last_weather_reaction_at = 0.0
    _emotional_checkin_fired.clear()
    _emotional_checkin_fired_at.clear()
    _negative_streak_started_at.clear()
    _group_turn_speaker_times.clear()
    _group_turn_visible_since.clear()
    _group_turn_invited_at.clear()
    _group_turn_invited_this_session.clear()
    _group_lull_fired_at.clear()
    _previous_face_boxes.clear()
    _last_face_seen_at = 0.0
    _face_tracking_lock.clear()
    _face_tracking_tracker = None
    _face_tracking_last_error_key = None
    _face_tracking_last_error_x = None
    _face_tracking_last_error_y = None
    _face_tracking_last_error_at = 0.0
    with _speaker_gaze_lock:
        _speaker_gaze_intent.clear()
    _record_face_tracking_state(locked=False, visible=False)
    _personal_space_reacted_at.clear()
    _last_pose_analysis_at = 0.0
    _startup_group_signature = None
    _startup_group_seen_at = 0.0
    _startup_solo_seen_at = 0.0
    _startup_empty_room_seen_at = 0.0
    _startup_empty_room_fired = False
    _startup_camera_first_frame_at = 0.0
    _startup_presence_evidence_at = 0.0
    _startup_presence_evidence_reason = ""
    _startup_group_greeted_signatures.clear()
    _last_rex_utterance_text = ""
    _last_memory_hint_text = ""
    _last_memory_hint_at = 0.0
    _last_memory_hint_person_id = None
    _recent_engaged_person_id = None
    _recent_engaged_touch_at = 0.0
    with _smile_reaction_lock:
        _smile_reaction_watch = None
    _last_smile_reaction_at = 0.0
    _facial_expression_observed.clear()
    _facial_expression_reacted_at.clear()
    _last_facial_expression_reaction_at = 0.0
    _last_expression_reaction_line_by_kind.clear()
    _disposition_sampled_at.clear()
    try:
        from intelligence import question_budget
        question_budget.clear()
    except Exception:
        pass
    try:
        from intelligence import end_thread
        end_thread.clear()
    except Exception:
        pass
    global _last_emotional_checkin_check_at, _last_group_turn_check_at, _last_group_lull_check_at
    _last_emotional_checkin_check_at = 0.0
    _last_group_turn_check_at = 0.0
    _last_group_lull_check_at = 0.0
    global _overheard_chime_in_count, _last_overheard_check_at
    _overheard_chime_in_count = 0
    _last_overheard_check_at = 0.0
    clear_engagement()
    with _turn_lock:
        _response_wait_until = 0.0
        _last_proactive_speech_at = 0.0
    try:
        from audio import speech_queue
        speech_queue.register_on_item_start(_note_rex_speech_item_started)
        speech_queue.register_on_item_done(_note_rex_speech_item_done)
    except Exception as exc:
        _log.debug("smile reaction speech hooks unavailable: %s", exc)
    if bool(getattr(config, "SPEAKER_GAZE_STARTUP_SCAN_ENABLED", True)):
        request_face_acquisition_scan(reason="startup")
    _thread = threading.Thread(target=_loop, daemon=True, name="consciousness")
    _thread.start()
    _face_tracking_thread = threading.Thread(
        target=_face_tracking_loop,
        daemon=True,
        name="face-tracking",
    )
    _face_tracking_thread.start()
    _log.info(
        "consciousness started (interval=%.1fs, face_tracking=%.2fs)",
        getattr(config, "CONSCIOUSNESS_LOOP_INTERVAL_SECS", 1.0),
        getattr(config, "FACE_TRACKING_LOOP_INTERVAL_SECS", 0.08),
    )


def stop() -> None:
    """Stop the consciousness daemon thread and wait for it to exit."""
    global _thread, _response_wait_until, _last_rex_utterance_text
    global _face_tracking_thread, _face_tracking_tracker
    global _identity_prompt_reply_until
    global _last_memory_hint_text, _last_memory_hint_at, _last_memory_hint_person_id
    global _recent_engaged_person_id, _recent_engaged_touch_at
    global _last_pose_analysis_at
    global _startup_empty_room_seen_at, _startup_empty_room_fired
    global _startup_camera_first_frame_at, _startup_presence_evidence_at
    global _startup_presence_evidence_reason
    global _face_tracking_last_error_key, _face_tracking_last_error_x
    global _face_tracking_last_error_y, _face_tracking_last_error_at
    global _last_face_seen_at
    global _smile_reaction_watch
    global _last_startle_sound_reaction_at
    _stop_event.set()
    _pending_identity_prompt.clear()
    _identity_prompt_in_flight.clear()
    _identity_prompt_reply_until = 0.0
    _proactive_speech_pending.clear()
    _jeff_celebrity_greeted_this_session.clear()
    _pending_jeff_celebrity_greetings.clear()
    _jt_volleyball_greeted_this_session.clear()
    _pending_jt_volleyball_greetings.clear()
    _confirmed_absent_at.clear()
    _first_sight_seen_at.clear()
    _animal_seen_signatures.clear()
    _animal_reacted_at.clear()
    _pending_animal_arrivals.clear()
    _last_startle_sound_reaction_at = 0.0
    _group_turn_speaker_times.clear()
    _group_turn_visible_since.clear()
    _group_turn_invited_at.clear()
    _group_turn_invited_this_session.clear()
    _group_lull_fired_at.clear()
    _previous_face_boxes.clear()
    _last_face_seen_at = 0.0
    _face_tracking_lock.clear()
    _face_tracking_tracker = None
    _face_tracking_last_error_key = None
    _face_tracking_last_error_x = None
    _face_tracking_last_error_y = None
    _face_tracking_last_error_at = 0.0
    with _speaker_gaze_lock:
        _speaker_gaze_intent.clear()
    _record_face_tracking_state(locked=False, visible=False)
    _personal_space_reacted_at.clear()
    _last_pose_analysis_at = 0.0
    _startup_empty_room_seen_at = 0.0
    _startup_empty_room_fired = False
    _startup_camera_first_frame_at = 0.0
    _startup_presence_evidence_at = 0.0
    _startup_presence_evidence_reason = ""
    _startup_group_greeted_signatures.clear()
    _last_rex_utterance_text = ""
    _last_memory_hint_text = ""
    _last_memory_hint_at = 0.0
    _last_memory_hint_person_id = None
    _recent_engaged_person_id = None
    _recent_engaged_touch_at = 0.0
    with _smile_reaction_lock:
        _smile_reaction_watch = None
    _facial_expression_observed.clear()
    _facial_expression_reacted_at.clear()
    _last_facial_expression_reaction_at = 0.0
    _last_expression_reaction_line_by_kind.clear()
    _disposition_sampled_at.clear()
    try:
        from intelligence import question_budget
        question_budget.clear()
    except Exception:
        pass
    try:
        from intelligence import end_thread
        end_thread.clear()
    except Exception:
        pass
    with _turn_lock:
        _response_wait_until = 0.0
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    if _face_tracking_thread:
        _face_tracking_thread.join(timeout=2)
        _face_tracking_thread = None
    _log.info("consciousness stopped")
