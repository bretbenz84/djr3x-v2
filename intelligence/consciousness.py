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
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
import state as state_module
from state import State
from world_state import world_state
from awareness.situation import assessor as _situation_assessor, SituationProfile

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_process_started_iso: Optional[str] = None
_process_started_mono: float = 0.0

# Smoothed neck servo position in quarter-microseconds
_neck_smooth: float = float(config.SERVO_CHANNELS["neck"]["neutral"])
_face_tracking_suspended_until: float = 0.0

# WorldState snapshot from the previous loop iteration (for change detection)
_last_snapshot: dict = {}

# Notable dates acknowledged this session so we don't repeat them
_acknowledged_dates: set[str] = set()

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
_last_identity_prompt_at: float = 0.0
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
_turn_lock = threading.Lock()
_proactive_speech_pending = threading.Event()

# Face detection terminal feedback de-duplication signature.
_last_face_feedback_signature: Optional[str] = None

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

# Per-person monotonic timestamp of when they were last seen in frame.
_last_seen: dict = {}

# Identity stickiness: when exactly one face is visible and recognition momentarily
# returns Unknown for what is almost certainly the same physical person we had
# identified a second ago, carry the last identity forward for this many seconds.
_last_solo_identity: Optional[tuple[int, str, float]] = None  # (db_id, name, monotonic)
_SOLO_IDENTITY_STICKY_SECS = 5.0

# Per-person monotonic timestamp of the last departure/return reaction fired.
_last_departure_reaction_at: dict = {}
_last_return_reaction_at: dict = {}

# Ensures only one presence reaction fires at a time; acquire non-blocking to skip if busy.
_presence_reaction_lock = threading.Lock()

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

# Engagement tracking: the person_db_id Rex is currently talking with, if any.
# Presence reactions for this person are suppressed while the engagement is open.
_engaged_lock = threading.Lock()
_engaged_person_id: Optional[int] = None
_engaged_last_touch_at: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Engagement API — called by interaction.py
# ─────────────────────────────────────────────────────────────────────────────

def mark_engagement(person_id: Optional[int]) -> None:
    """Record that Rex is actively conversing with person_id. Called on every
    identified speech segment. Resets the engagement window."""
    global _engaged_person_id, _engaged_last_touch_at
    if person_id is None:
        return
    with _engaged_lock:
        _engaged_person_id = person_id
        _engaged_last_touch_at = time.monotonic()


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
    """Clear engagement state — called on session end."""
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
        pid = _engaged_person_id
        touch = _engaged_last_touch_at
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


def consume_identity_prompt_request() -> bool:
    """
    Return True once when an unknown-person identity prompt was recently spoken.
    Interaction uses this to temporarily accept short bare-name replies.
    """
    if _pending_identity_prompt.is_set():
        _pending_identity_prompt.clear()
        return True
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


def note_rex_utterance(text: str, wait_secs: Optional[float] = None) -> None:
    """
    Track when Rex last spoke and, if it was a question, open a reply window.
    """
    global _last_proactive_speech_at, _response_wait_until
    now = time.monotonic()
    try:
        from intelligence import question_budget
        question_budget.note_rex_utterance(text)
    except Exception:
        pass

    with _turn_lock:
        _last_proactive_speech_at = now

        should_wait = wait_secs is not None or _utterance_expects_reply(text)
        if not should_wait:
            return

        wait_for = (
            config.QUESTION_RESPONSE_WAIT_SECS
            if wait_secs is None
            else max(0.0, float(wait_secs))
        )
        _response_wait_until = max(_response_wait_until, now + wait_for)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_speak() -> bool:
    return state_module.get_state() not in (State.QUIET, State.SHUTDOWN)


def _can_proactive_speak() -> bool:
    if not _can_speak():
        return False

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
) -> bool:
    try:
        if not _can_proactive_speak():
            return False
        if not text or not text.strip():
            return False
        from audio import speech_queue
        _proactive_speech_pending.set()
        done = speech_queue.enqueue(text, emotion, priority=0)

        def _on_done() -> None:
            done.wait()
            _proactive_speech_pending.clear()

        threading.Thread(target=_on_done, daemon=True, name="speech-pending-clear").start()
        note_rex_utterance(text, wait_secs=wait_secs)
        return True
    except Exception as exc:
        _proactive_speech_pending.clear()
        _log.debug("_speak_async error: %s", exc)
        return False


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


def _generate_and_speak(
    prompt: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
    purpose: Optional[str] = None,
    priority: Optional[int] = None,
    label: str = "",
) -> bool:
    token = None
    if purpose:
        token = _claim_proactive_purpose(
            purpose,
            priority=priority,
            label=label or purpose,
        )
        if token is None:
            return False
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
                _speak_async(text, emotion, wait_secs=wait_secs)
        except Exception as exc:
            _log.debug("_generate_and_speak error: %s", exc)
        finally:
            _release_proactive_purpose(token)

    threading.Thread(target=_task, daemon=True).start()
    return True


def _should_fire_presence(key, person_db_id: Optional[int], profile: SituationProfile) -> bool:
    """
    Unified gate for presence (departure/return) reactions.

    Stricter than generic proactive speech: we never narrate presence events for
    the person Rex is currently talking to, never during the user's own sentence,
    never while another presence reaction for this person is already queued, and
    never more often than PRESENCE_PER_PERSON_COOLDOWN_SECS per person.
    """
    if not _can_speak():
        return False
    if profile.user_mid_sentence:
        return False
    if is_engaged_with(person_db_id):
        return False

    cooldown = getattr(config, "PRESENCE_PER_PERSON_COOLDOWN_SECS", 120.0)
    last = _last_presence_reaction_at.get(key, 0.0)
    if last and (time.monotonic() - last) < cooldown:
        return False

    try:
        from audio import speech_queue
        if speech_queue.has_waiting_with_tag(f"presence:{key}"):
            return False
    except Exception:
        pass
    return True


def _generate_and_speak_presence(
    prompt: str,
    label: str,
    tag_key,
    emotion: str = "neutral",
    *,
    purpose: str = "presence_reaction",
    priority: Optional[int] = None,
) -> None:
    """
    Presence-reaction variant of _generate_and_speak.

    All gating now flows through _should_fire_presence() before this is called.
    The tag_key is used to coalesce duplicate queued reactions for the same
    person (newer replaces older).
    """
    token = _claim_proactive_purpose(purpose, priority=priority, label=label)
    if token is None:
        return
    prompt = _apply_proactive_directive(prompt, purpose)

    def _task():
        if not _presence_reaction_lock.acquire(blocking=False):
            _log.debug("_generate_and_speak_presence: reaction already in progress, skipping — %s", label)
            _release_proactive_purpose(token)
            return
        try:
            if not _proactive_purpose_current(token):
                return
            if not _can_speak():
                return
            from intelligence.llm import get_response
            text = get_response(prompt)
            if not text or not text.strip():
                return
            if not _proactive_purpose_current(token):
                return
            if not _can_speak():
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
            speech_queue.enqueue(text, emotion, priority=1, tag=tag)
            note_rex_utterance(text)
        except Exception as exc:
            _log.debug("_generate_and_speak_presence error: %s", exc)
        finally:
            _release_proactive_purpose(token)
            _presence_reaction_lock.release()

    threading.Thread(target=_task, daemon=True).start()


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

def _step_person_recognition(frame) -> None:
    """
    Detect visible faces, resolve known identities via DB lookup, and update
    world_state.people with one slot per visible face.

    This function no longer depends on pose pre-populating people slots. If the
    pose pipeline is disabled or lagging, face recognition still works and can
    drive unknown-person onboarding prompts.
    """
    global _last_face_feedback_signature, _last_identity_prompt_at, _last_solo_identity
    try:
        from vision import face as face_mod

        if frame is None:
            _last_face_feedback_signature = None
            return

        detected = face_mod.detect_faces(frame)
        if not detected:
            # No visible faces this tick — clear transient person slots.
            if world_state.get("people"):
                world_state.update("people", [])
            _last_face_feedback_signature = None
            return

        # Identity stickiness: HOG face recognition flickers unknown↔known within
        # 1–2 frames. When there's one face and we identified it moments ago,
        # carry that identity forward if this frame can't match.
        apply_sticky = (
            len(detected) == 1
            and _last_solo_identity is not None
            and (time.monotonic() - _last_solo_identity[2]) <= _SOLO_IDENTITY_STICKY_SECS
        )

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
                    "pose": base.get("pose"),
                    "gesture": base.get("gesture"),
                    "engagement": base.get("engagement"),
                    "age_estimate": base.get("age_estimate"),
                    "position": base.get("position"),
                })
            people = resized
            changed = True

        recognized_names: list[str] = []
        unknown_count = 0
        any_identified_this_tick = False
        for det in detected:
            person_record = face_mod.identify_face(det["encoding"])
            if person_record is None and apply_sticky:
                # Carry forward last solo identity through a single-face miss.
                sticky_id, sticky_name, _ = _last_solo_identity
                person_record = {"id": sticky_id, "name": sticky_name}
            if person_record is None:
                unknown_count += 1
                continue
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

            if target_slot is None:
                continue

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

                # If someone unknown appears while idle, ask who they are so
                # interaction can enroll them in the person database.
                now = time.monotonic()
                if (
                    _can_proactive_speak()
                    and state_module.get_state() == State.IDLE
                    and (now - _last_identity_prompt_at) >= _IDENTITY_PROMPT_COOLDOWN_SECS
                ):
                    _last_identity_prompt_at = now
                    _pending_identity_prompt.set()
                    _log.info("consciousness: prompting unknown person for identity")
                    _generate_and_speak(
                        "You can see someone you don't recognize. "
                        "In one short in-character line, ask who they are and what name "
                        "you should store for them.",
                        emotion="curious",
                        wait_secs=getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0),
                        purpose="identity_prompt",
                    )
            _last_face_feedback_signature = signature

        if changed:
            world_state.update("people", people)

        # Update solo identity snapshot for next tick's stickiness check.
        if len(detected) == 1 and any_identified_this_tick and recognized_names:
            # Find the db_id that matches the recognized name
            for ws_person in people:
                if ws_person.get("face_id") == recognized_names[0] and ws_person.get("person_db_id"):
                    _last_solo_identity = (
                        ws_person["person_db_id"],
                        ws_person["face_id"],
                        time.monotonic(),
                    )
                    break
        elif len(detected) != 1:
            # Multiple or zero faces — stickiness no longer applies.
            _last_solo_identity = None
    except Exception as exc:
        _log.debug("person recognition step error: %s", exc)


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


def _pick_due_celebration_checkin(person_db_id: Optional[int]) -> Optional[dict]:
    """Return the most recent positive event due for a startup celebration."""
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import emotional_events as emo_events
        due = emo_events.get_startup_celebrations(
            person_db_id,
            process_started_iso=_process_started_iso,
            limit=1,
        )
        return due[0] if due else None
    except Exception as exc:
        _log.debug("celebration check-in lookup error: %s", exc)
        return None


def _first_sight_context(first_name: str) -> tuple[str, str]:
    """Return prompt phrasing for seeing a known person first time this run."""
    if _process_started_mono and (time.monotonic() - _process_started_mono) <= 45.0:
        return (
            f"You just started up and immediately see '{first_name}'.",
            "you just booted up and immediately spot them",
        )
    return (
        f"'{first_name}', someone you know, just came into your camera view "
        f"for the first time this run.",
        "they just came into your camera view for the first time this run",
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Proactive reactions
# ─────────────────────────────────────────────────────────────────────────────

def _step_proactive_reactions(snapshot: dict, profile: SituationProfile) -> None:
    """
    Compare current WorldState to _last_snapshot. For each notable change,
    generate and speak a short in-character reaction. Never fires in QUIET/SHUTDOWN.
    """
    global _acknowledged_dates

    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _last_snapshot or not _can_proactive_speak():
        return

    try:
        triggers: list[tuple[str, str]] = []  # (llm_prompt, emotion)

        # New person entered frame
        prev_count = _last_snapshot.get("crowd", {}).get("count", 0)
        curr_count = snapshot.get("crowd", {}).get("count", 0)
        if curr_count > prev_count:
            triggers.append((
                "Someone new just walked into your view. React in one short in-character line — "
                "somewhere between a greeting and a roast, delivered as you clock them entering.",
                "curious",
            ))

        # Animal detected. Animal IDs are positional and can be unstable across
        # scans, so dedupe by species + rough position with a cooldown.
        animal_cooldown = float(getattr(config, "ANIMAL_ARRIVAL_COOLDOWN_SECS", 300.0))
        prev_animal_signatures = {
            f"{(a.get('species') or 'creature').strip().lower()}:"
            f"{(a.get('position') or 'unknown').strip().lower()}"
            for a in _last_snapshot.get("animals", [])
            if a.get("species")
        }
        for animal in snapshot.get("animals", []):
            species = (animal.get("species") or "creature").strip().lower()
            position = (animal.get("position") or "unknown").strip().lower()
            if not species:
                continue
            signature = f"{species}:{position}"
            last_reacted = _animal_reacted_at.get(signature, 0.0)
            if signature in prev_animal_signatures:
                _animal_seen_signatures.add(signature)
                continue
            if (time.monotonic() - last_reacted) < animal_cooldown:
                continue
            _animal_seen_signatures.add(signature)
            _animal_reacted_at[signature] = time.monotonic()
            if species == "dog":
                prompt = (
                    f"You just spotted a dog in your immediate environment at {position}. "
                    "One short in-character Rex reaction — delighted but dry. Do not ask "
                    "who the dog is unless a human has introduced it. One line only."
                )
            else:
                prompt = (
                    f"You just spotted a {species} in your immediate environment at {position}. "
                    "One short in-character reaction — genuinely surprised, unmistakably Rex. "
                    "One line only."
                )
            triggers.append((prompt, "excited"))

        # Crowd size label changed significantly
        prev_label = _last_snapshot.get("crowd", {}).get("count_label")
        curr_label = snapshot.get("crowd", {}).get("count_label")
        if curr_label and prev_label and curr_label != prev_label:
            triggers.append((
                f"The crowd around you just shifted from '{prev_label}' to '{curr_label}'. "
                "One short in-character observation about this change.",
                "neutral",
            ))

        # Notable sound event
        prev_sound = _last_snapshot.get("audio_scene", {}).get("last_sound_event")
        curr_sound = snapshot.get("audio_scene", {}).get("last_sound_event")
        if curr_sound and curr_sound != prev_sound:
            triggers.append((
                f"You just registered a notable sound event: '{curr_sound}'. "
                "One punchy in-character line reacting to it.",
                "curious",
            ))

        # Notable calendar date (once per session per date)
        notable_date = snapshot.get("time", {}).get("notable_date")
        if notable_date and notable_date not in _acknowledged_dates:
            _acknowledged_dates.add(notable_date)
            triggers.append((
                f"Today is {notable_date}. Make one spontaneous in-character remark about it "
                "as if you just noticed the date. Deliver it Rex-style.",
                "excited",
            ))

        if triggers:
            prompt, emotion = random.choice(triggers)
            _generate_and_speak(prompt, emotion, purpose="world_reaction")

    except Exception as exc:
        _log.debug("proactive reactions step error: %s", exc)


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
    # Include ambient_observation and appearance_riff/live_vision so Rex talks
    # about his surroundings and the people in them, not just himself.
    behavior = random.choices(
        [
            "small_talk_question",
            "ambient_scan",
            "private_thought",
            "aspiration",
            "idle_clip",
            "ambient_observation",
            "appearance_riff",
            "live_vision_comment",
        ],
        # Bias toward asking the user something when they've gone quiet — Rex
        # should pull them into conversation, not just narrate his own opinions.
        weights=[3, 1, 1, 1, 1, 1, 1, 1],
        k=1,
    )[0]
    _log.debug("consciousness: idle micro-behavior → %s", behavior)

    if behavior == "small_talk_question":
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
    elif behavior == "live_vision_comment":
        if not profile.suppress_proactive:
            _do_live_vision_comment(snapshot)


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
        target_name = (target.get("face_id") or "").split()[0] or None
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
    token = _claim_proactive_purpose(purpose, label="small-talk question")
    if token is None:
        return

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
                    f"If a specific cue above tells you what to ask about, ask about THAT. "
                    f"Otherwise ask something open — how their day is going, what they've "
                    f"been working on, what's on their mind, what they're listening to lately. "
                    f"Warm but dry. Don't lecture, don't give your opinion — just ask. "
                    f"Address {target_name} by name. One short sentence ending in a question mark."
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
                _speak_async(text, emotion=emotion)
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
            _speak_async(line, emotion="neutral")
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
            _speak_async(chosen, emotion="curious")
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
    first_name = (target.get("face_id") or "").split()[0] or "there"
    _generate_and_speak(
        f"You're idly looking at '{first_name}'. You remember this about their "
        f"appearance: {hint}. Make one short in-character Rex remark about it — "
        f"the kind of thing you'd say while looking them over. Warm, dry, observational. "
        f"Address {first_name} by name. One line only.",
        emotion="neutral",
        purpose="appearance_riff",
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

    token = _claim_proactive_purpose(
        "visual_curiosity",
        label=f"visual curiosity for {engaged_id}",
    )
    if token is None:
        return

    with _visual_curiosity_lock:
        if _visual_curiosity_in_flight:
            _release_proactive_purpose(token)
            return
        if (time.monotonic() - _last_visual_curiosity_at) < global_cooldown:
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
            first_name = (person.get("name") or "").split()[0] or "there"

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
                _speak_async(text, emotion="curious", wait_secs=wait)
        except Exception as exc:
            _log.debug("visual curiosity step error: %s", exc)
        finally:
            _release_proactive_purpose(token)
            with _visual_curiosity_lock:
                _visual_curiosity_in_flight = False

    threading.Thread(target=_task, daemon=True, name="visual-curiosity").start()


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

    first_name = (engaged_name or "").split()[0] or "friend"
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
      - Once confirmed gone, we stage a departure in _pending_departure_keys and wait
        for apparent_departure (face-gone + VAD-silent) before speaking.
      - _should_fire_presence() is the single gate for every presence reaction —
        it enforces per-person cooldowns and the "no narrating the person you're
        talking to" rule.
    """
    global _visible_people, _pending_departure_keys, _first_missing_at

    if not _can_speak():
        return

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
        if (now - first_missing) < confirm_absent:
            continue
        # Capture person info from last snapshot.
        person_name = None
        person_db_id = None
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
        _pending_departure_keys[key] = (first_missing, person_name, person_db_id)
        _confirmed_absent_at[key] = first_missing
        _log.debug(
            "consciousness: staged departure for key=%s name=%r after %.1fs absent",
            key, person_name, now - first_missing,
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

        # Fire only when face-gone + VAD has been silent ≥ departure_audio_silence.
        should_fire = profile.apparent_departure or (
            (now - departed_at) >= departure_audio_silence
            and not profile.user_mid_sentence
        )
        if not should_fire:
            continue

        if not _should_fire_presence(key, person_db_id, profile):
            del _pending_departure_keys[key]
            continue

        _last_departure_reaction_at[key] = now
        _last_presence_reaction_at[key] = now
        _first_missing_at.pop(key, None)
        del _pending_departure_keys[key]

        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = person_name.split()[0]
            _log.info("consciousness: departure reaction firing for %s", person_name)
            _generate_and_speak_presence(
                f"The person named '{first_name}' just left your camera view. "
                "React in one short in-character line as Rex — playful and dry, "
                "but not mean. Do not imply nobody likes or misses them. Examples: "
                f"'Where are you going, {first_name}?', 'Oh, leaving already?', "
                "'Don't go too far, I can't roast you from a distance.' "
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

        # First time ever seen this session.
        if key not in _last_seen:
            if isinstance(key, int) and person_name and key not in _greeted_this_session:
                first_visible = _first_sight_seen_at.setdefault(key, now)
                confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
                if (now - first_visible) < max(0.0, confirm_visible):
                    first_sight_pending_keys.add(key)
                    continue
                _greeted_this_session.add(key)
                if _should_fire_presence(key, person_db_id, profile):
                    first_name = person_name.split()[0]
                    context_sentence, situation_phrase = _first_sight_context(first_name)
                    prompt: Optional[str] = None
                    label = f"first-sight greeting for {person_name}"
                    emotion = "excited"

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
                        _note_emotional_checkin_fired(person_db_id)
                        try:
                            from memory import emotional_events as emo_events
                            emo_events.mark_acknowledged(int(emotional["id"]))
                        except Exception:
                            pass
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
                            try:
                                from memory import emotional_events as emo_events
                                emo_events.mark_acknowledged(int(celebration["id"]))
                            except Exception:
                                pass
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
                                _pending_followups_lock_remove(person_db_id, ev.get("id"))
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
                                _anticipated_events.add((person_db_id, anticipated["id"]))
                                prompt = anti_prompt
                                label = f"startup anticipation for {person_name}"
                                _log.info(
                                    "consciousness: startup anticipation for %s (event=%s)",
                                    person_name, anticipated.get("event_name"),
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

                    # Fallback — generic greeting
                    if prompt is None:
                        prompt = (
                            f"{context_sentence} "
                            f"Greet them in-character, then ask them a small-talk question — how "
                            f"they're doing, what they've been up to, what's on the agenda — "
                            f"something that invites them to actually talk to you, not just listen "
                            f"to you have an opinion. Address {first_name} by name. Two short "
                            f"sentences max — the second must end in a question mark."
                        )
                        _log.info("consciousness: startup greeting for %s", person_name)

                    _generate_and_speak_presence(
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
                    )
            continue

        absent_secs = now - _last_seen[key]
        if absent_secs < return_min_absent:
            _confirmed_absent_at.pop(key, None)
            continue
        if key not in _confirmed_absent_at:
            continue

        if not _should_fire_presence(key, person_db_id, profile):
            continue

        _last_return_reaction_at[key] = now
        _confirmed_absent_at.pop(key, None)
        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = person_name.split()[0]
            _log.info("consciousness: return detected — queuing reaction for %s (absent %.1fs)", person_name, absent_secs)
            anticipated = _pick_anticipated_event(person_db_id)
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
                    f"'Oh, you're back.', 'Miss me already, {first_name}?', "
                    "'I knew you couldn't stay away.' "
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
                first_name = face_id.split()[0]
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
    speaker_first = (speaker_name or "someone").split()[0]

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
        first_name = (person.get("name") or "").split()[0] or "you"

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
                f"{target['name']} is {when_clause} — a long weekend in their tradition. "
                f"Ask {first_name} if they have any 3-day-weekend plans, dryly observant. "
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
    if (now - engaged_touch) > window:
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
        first_name = (person.get("name") or "").split()[0] or "you"

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
                    f"You're mid-conversation with '{first_name}'. It's Monday morning. "
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
        first_name = (person.get("name") or "").split()[0] or "you"
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


def _face_x_to_neck_target(x: int) -> float:
    """Map pixel x to neck servo position. Center → neutral; edges → extremes."""
    neck_cfg = config.SERVO_CHANNELS["neck"]
    frac = x / max(config.CAMERA_WIDTH - 1, 1)  # 0.0 (left) → 1.0 (right)
    return float(neck_cfg["min"] + frac * (neck_cfg["max"] - neck_cfg["min"]))


def _step_face_tracking(frame) -> None:
    """
    Get the largest face position from the current frame, smooth toward the neck
    servo target, and issue a set_servo command when change exceeds the dead zone.
    Suspended during SLEEP state.
    """
    global _neck_smooth

    if state_module.get_state() == State.SLEEP:
        return
    if time.monotonic() < _face_tracking_suspended_until:
        return

    try:
        from vision import face as face_mod
        from hardware import servos as servo_mod

        neck_cfg = config.SERVO_CHANNELS["neck"]
        neck_ch  = neck_cfg["ch"]
        neutral  = float(neck_cfg["neutral"])
        alpha    = 1.0 - config.TRACKING_SMOOTHING_FACTOR  # fraction to close per tick

        face_pos = face_mod.get_face_position(frame) if frame is not None else None

        if face_pos is None:
            # No face detected: drift back toward neutral
            _neck_smooth += alpha * (neutral - _neck_smooth)
            return

        x, _y = face_pos
        frame_cx = config.CAMERA_WIDTH / 2.0

        # Pixel dead zone: if face is close enough to center, do nothing
        if abs(x - frame_cx) <= config.TRACKING_DEAD_ZONE_PX:
            return

        target = _face_x_to_neck_target(x)
        new_smooth = _neck_smooth + alpha * (target - _neck_smooth)

        # Only send servo command if smoothed position moved beyond dead zone in servo space
        dead_zone_qus = (
            config.TRACKING_DEAD_ZONE_PX
            / config.CAMERA_WIDTH
            * (neck_cfg["max"] - neck_cfg["min"])
        )
        if abs(new_smooth - _neck_smooth) >= dead_zone_qus:
            _neck_smooth = new_smooth
            clamped = max(neck_cfg["min"], min(neck_cfg["max"], int(_neck_smooth)))
            servo_mod.set_servo(neck_ch, clamped)
            servo_mod.set_face_tracking_baseline(neck=clamped)

    except Exception as exc:
        _log.debug("face tracking step error: %s", exc)


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

            # 3. Interoception
            _step_interoception()

            # 4. Chronoception
            _step_chronoception()

            # 5. Person recognition (may update world_state.people)
            _step_person_recognition(frame)

            # Snapshot after recognition so steps 6–11 see identified persons
            snapshot = world_state.snapshot()

            # 6. Follow-up check
            _step_followup_check(snapshot)

            # 7. Disengagement detection
            _step_disengagement(snapshot, profile)

            # 8. Proactive reactions
            _step_proactive_reactions(snapshot, profile)

            # 9. Idle micro-behaviors
            _step_idle_micro_behavior(snapshot, profile)

            # 10. Presence tracking (departure / return reactions)
            _step_presence_tracking(snapshot, profile)

            # 10b. Social inquiry — ask engaged person about unknown newcomer
            _step_relationship_inquiry(snapshot, profile)

            # 10c. Third-party awareness — call out a lingering bystander
            _step_third_party_awareness(snapshot, profile)

            # 10c2. Group turn-taking — softly invite a quiet visible known person
            _step_group_turn_taking(snapshot, profile)

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

            # 11. Face tracking
            _step_face_tracking(frame)

            # Preserve snapshot for next iteration's change detection
            _last_snapshot = snapshot

        except Exception as exc:
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
    global _process_started_iso, _process_started_mono
    if _thread and _thread.is_alive():
        _log.debug("consciousness already running")
        return
    _process_started_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    _process_started_mono = time.monotonic()
    _stop_event.clear()
    _pending_identity_prompt.clear()
    _pending_relationship_prompt.clear()
    _pending_relationship_context.clear()
    _asked_relationship_slots.clear()
    _unknown_first_seen_at.clear()
    _proactive_speech_pending.clear()
    _greeted_this_session.clear()
    _pending_departure_keys.clear()
    _first_missing_at.clear()
    _confirmed_absent_at.clear()
    _first_sight_seen_at.clear()
    _last_presence_reaction_at.clear()
    _animal_seen_signatures.clear()
    _animal_reacted_at.clear()
    _emotional_checkin_fired.clear()
    _emotional_checkin_fired_at.clear()
    _negative_streak_started_at.clear()
    _group_turn_speaker_times.clear()
    _group_turn_visible_since.clear()
    _group_turn_invited_at.clear()
    _group_turn_invited_this_session.clear()
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
    global _last_emotional_checkin_check_at, _last_group_turn_check_at
    _last_emotional_checkin_check_at = 0.0
    _last_group_turn_check_at = 0.0
    global _overheard_chime_in_count, _last_overheard_check_at
    _overheard_chime_in_count = 0
    _last_overheard_check_at = 0.0
    clear_engagement()
    with _turn_lock:
        _response_wait_until = 0.0
        _last_proactive_speech_at = 0.0
    _thread = threading.Thread(target=_loop, daemon=True, name="consciousness")
    _thread.start()
    _log.info(
        "consciousness started (interval=%.1fs)",
        getattr(config, "CONSCIOUSNESS_LOOP_INTERVAL_SECS", 1.0),
    )


def stop() -> None:
    """Stop the consciousness daemon thread and wait for it to exit."""
    global _thread, _response_wait_until
    _stop_event.set()
    _pending_identity_prompt.clear()
    _proactive_speech_pending.clear()
    _confirmed_absent_at.clear()
    _first_sight_seen_at.clear()
    _animal_seen_signatures.clear()
    _animal_reacted_at.clear()
    _group_turn_speaker_times.clear()
    _group_turn_visible_since.clear()
    _group_turn_invited_at.clear()
    _group_turn_invited_this_session.clear()
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
    _log.info("consciousness stopped")
