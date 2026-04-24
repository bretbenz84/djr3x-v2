"""
intelligence/consciousness.py — Central consciousness loop for DJ-R3X.

Reads WorldState on a fixed interval and drives proactive behavior:
anger/mood maintenance, person recognition, follow-up detection,
disengagement recovery, proactive world reactions, idle micro-behaviors,
and continuous neck-servo face tracking.
"""

import logging
import random
import sys
import threading
import time
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

# Smoothed neck servo position in quarter-microseconds
_neck_smooth: float = float(config.SERVO_CHANNELS["neck"]["neutral"])

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


def note_relationship_slot_handled(slot_id: str) -> None:
    """Called by interaction after it resolves (or gives up on) a slot so
    consciousness won't re-ask about the same unknown face in this session."""
    if slot_id:
        _asked_relationship_slots.add(slot_id)


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


def _generate_and_speak(
    prompt: str,
    emotion: str = "neutral",
    *,
    wait_secs: Optional[float] = None,
) -> None:
    def _task():
        try:
            if not _can_proactive_speak():
                return
            from intelligence.llm import get_response
            text = get_response(prompt)
            if text:
                _speak_async(text, emotion, wait_secs=wait_secs)
        except Exception as exc:
            _log.debug("_generate_and_speak error: %s", exc)

    threading.Thread(target=_task, daemon=True).start()


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
) -> None:
    """
    Presence-reaction variant of _generate_and_speak.

    All gating now flows through _should_fire_presence() before this is called.
    The tag_key is used to coalesce duplicate queued reactions for the same
    person (newer replaces older).
    """
    def _task():
        if not _presence_reaction_lock.acquire(blocking=False):
            _log.debug("_generate_and_speak_presence: reaction already in progress, skipping — %s", label)
            return
        try:
            if not _can_speak():
                return
            from intelligence.llm import get_response
            text = get_response(prompt)
            if not text or not text.strip():
                return
            if not _can_speak():
                return

            delay = getattr(config, "PRESENCE_REACTION_DELAY_SECS", 2.0)
            if delay > 0:
                time.sleep(delay)

            if not _can_speak():
                return

            from audio import speech_queue
            tag = f"presence:{tag_key}"
            _log.info("consciousness: firing presence reaction — %s: %r", label, text[:120])
            _last_presence_reaction_at[tag_key] = time.monotonic()
            speech_queue.enqueue(text, emotion, priority=1, tag=tag)
        except Exception as exc:
            _log.debug("_generate_and_speak_presence error: %s", exc)
        finally:
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

        # Animal detected
        prev_animal_ids = {a.get("id") for a in _last_snapshot.get("animals", [])}
        for animal in snapshot.get("animals", []):
            if animal.get("id") not in prev_animal_ids:
                species = animal.get("species", "creature")
                triggers.append((
                    f"You just spotted a {species} in your immediate environment. "
                    "One short in-character reaction — genuinely surprised, unmistakably Rex.",
                    "excited",
                ))

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
            _generate_and_speak(prompt, emotion)

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
    behavior = random.choice([
        "ambient_scan",
        "private_thought",
        "idle_clip",
        "ambient_observation",
        "appearance_riff",
        "live_vision_comment",
    ])
    _log.debug("consciousness: idle micro-behavior → %s", behavior)

    if behavior == "ambient_scan":
        _do_ambient_scan()
    elif behavior == "private_thought":
        # Private thoughts are system monologues — suppressed by both proactive and
        # system-comment gates so Rex doesn't mutter about himself mid-conversation.
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            _do_private_thought()
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


def _do_private_thought() -> None:
    if not _can_proactive_speak():
        return
    line = random.choice(config.PRIVATE_THOUGHTS)
    _speak_async(line, emotion="neutral")


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
            )
        except Exception as exc:
            _log.debug("live vision comment error: %s", exc)

    threading.Thread(target=_task, daemon=True, name="live-vision-comment").start()


def _do_idle_clip() -> None:
    try:
        clips_dir = Path(config.AUDIO_CLIPS_DIR)
        clips = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.wav"))
        if not clips:
            return
        clip_path = random.choice(clips)

        def _play():
            try:
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

    # Find engaged person (if any) from current snapshot.
    engaged_id: Optional[int] = None
    engaged_name: Optional[str] = None
    with _engaged_lock:
        if _engaged_person_id is None:
            # Not currently engaged; drop all unknown timers to avoid stale state.
            _unknown_first_seen_at.clear()
            return
        engaged_id = _engaged_person_id

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
    _generate_and_speak(
        f"You're talking with '{first_name}' and a new unfamiliar face has just "
        f"joined the view. In one short in-character Rex line, ask {first_name} "
        f"who the newcomer is AND what their relationship to {first_name} is — "
        f"e.g. 'Oh hey, who's this, {first_name}? Friend of yours?' Keep it warm "
        f"and curious, one line only, ending with a question mark.",
        emotion="curious",
        wait_secs=getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0),
    )


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

    # Build current tracked set: tracking_key → (name, db_id).
    current_tracked: dict = {}
    for person in snapshot.get("people", []):
        key = _tracking_key(person)
        current_tracked[key] = (person.get("face_id"), person.get("person_db_id"))

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
        del _pending_departure_keys[key]

        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = person_name.split()[0]
            _log.info("consciousness: departure reaction firing for %s", person_name)
            _generate_and_speak_presence(
                f"The person named '{first_name}' just left your camera view. "
                "React in one short in-character line as Rex. Examples: "
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
    for key in current_keys - _visible_people:
        person_name, person_db_id = current_tracked[key]

        # First time ever seen this session.
        if key not in _last_seen:
            if isinstance(key, int) and person_name and key not in _greeted_this_session:
                _greeted_this_session.add(key)
                if _should_fire_presence(key, person_db_id, profile):
                    first_name = person_name.split()[0]
                    _log.info("consciousness: startup greeting for %s", person_name)
                    _generate_and_speak_presence(
                        f"You just started up and immediately see '{first_name}', someone you know. "
                        f"Greet them in one short in-character Rex line. "
                        f"Address {first_name} by name. Make it feel like Rex just booted and is glad to see a familiar face.",
                        label=f"startup greeting for {person_name}",
                        tag_key=key,
                        emotion="excited",
                    )
            continue

        absent_secs = now - _last_seen[key]
        if absent_secs < return_min_absent:
            continue

        if not _should_fire_presence(key, person_db_id, profile):
            continue

        _last_return_reaction_at[key] = now
        is_known = isinstance(key, int) and person_name

        if is_known:
            first_name = person_name.split()[0]
            _log.info("consciousness: return detected — queuing reaction for %s (absent %.1fs)", person_name, absent_secs)
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
    for key in current_keys:
        _last_seen[key] = now
    _visible_people = current_keys


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Face tracking
# ─────────────────────────────────────────────────────────────────────────────

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

    try:
        from vision import face as face_mod
        from hardware.servos import set_servo

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
            set_servo(neck_ch, clamped)

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
    if _thread and _thread.is_alive():
        _log.debug("consciousness already running")
        return
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
    _last_presence_reaction_at.clear()
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
    with _turn_lock:
        _response_wait_until = 0.0
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    _log.info("consciousness stopped")
