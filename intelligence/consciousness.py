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
from intelligence import episodic_hooks
from intelligence import gaze_engine
from intelligence import greeting_cadence
from intelligence import person_specials
from intelligence import profile_questions
from intelligence import speech_engine
from utils import conv_log

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Proactive-speech ENGINE — re-export shims. The implementations were extracted to
# intelligence/speech_engine.py to shrink this module; these aliases keep the historical
# `_name` call sites (~100 of them) and the test patch targets working unchanged. Patching
# consciousness._generate_and_speak still affects every caller that resolves it here.
# (note_rex_utterance + the shared speech state stay in this module — the engine reads them
# back via its consciousness proxy.)
# ─────────────────────────────────────────────────────────────────────────────
_can_proactive_speak = speech_engine.can_proactive_speak
_claim_proactive_purpose = speech_engine.claim_proactive_purpose
_release_proactive_purpose = speech_engine.release_proactive_purpose
_proactive_purpose_current = speech_engine.proactive_purpose_current
_apply_proactive_directive = speech_engine.apply_proactive_directive
_generate_and_speak = speech_engine.generate_and_speak
_speak_async = speech_engine.speak_async
_generate_and_speak_presence = speech_engine.generate_and_speak_presence
_observe_governor_candidate = speech_engine.observe_governor_candidate
_mark_governor_candidate = speech_engine.mark_governor_candidate
_start_governor_cycle = speech_engine.start_governor_cycle
_finish_governor_cycle = speech_engine.finish_governor_cycle
_governor_enforcing = speech_engine.governor_enforcing
_governor_source = speech_engine.governor_source
_governor_speech_metadata = speech_engine.governor_speech_metadata

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
# Jump-rejection state: the last ACCEPTED face-box center (so a spurious teleport can be
# measured against it) and a pending position being confirmed as a genuine fast move.
_face_tracking_last_center: Optional[dict] = None
_face_tracking_pending_center: Optional[dict] = None
_face_tracking_last_jump_log_at: float = 0.0

# WorldState snapshot from the previous loop iteration (for change detection)
_last_snapshot: dict = {}

# Notable dates acknowledged this session so we don't repeat them
_acknowledged_dates: set[str] = set()
_acknowledged_weather_signatures: set[str] = set()
_last_weather_reaction_at: float = 0.0
# Parts of day (morning/afternoon/evening/night/late_night) Rex has already remarked on
# this session, so a transition fires at most once.
_acknowledged_tod: set[str] = set()

# Monotonic timestamp of the last idle micro-behavior
_last_micro_behavior_at: float = 0.0

# Cooldown: map person id-string → monotonic timestamp of last re-engagement attempt
_reengagement_sent_at: dict[str, float] = {}
_REENGAGEMENT_COOLDOWN_SECS = 30.0

# Monotonic timestamp of last live-vision commentary call (cost control).
# Monotonic timestamp of last bored environmental-snark riff (cost control).

# Visual curiosity asks: after a real back-and-forth goes quiet, Rex can take a
# fresh frame, summarize it, and ask one scene-grounded question.
_last_visual_curiosity_at: float = 0.0
_visual_curiosity_by_person: dict[int, float] = {}
_visual_curiosity_in_flight: bool = False
_visual_curiosity_lock = threading.Lock()

# Lull callback: after a back-and-forth goes quiet, resurface one banked
# "fun fact" premise about the engaged person (intelligence/callback_engine).
_last_lull_callback_at: float = 0.0
_lull_callback_by_person: dict[int, float] = {}
_last_news_remark_at: float = 0.0
_news_remarks_this_session: int = 0
_open_thread_asked_persons: set = set()   # once per person per session

# Pending follow-up events per DB person_id: {db_id: [event_dict, ...]}
_pending_followups: dict[int, list[dict]] = {}
_followup_lock = threading.Lock()

# Pending identity prompt for unknown-person enrollment.
_pending_identity_prompt = threading.Event()
_identity_prompt_in_flight = threading.Event()
# When the in-flight latch was set. A governor-REJECTED candidate never runs its
# speak_fn, so no callback clears the latch — a stale timestamp is the recovery
# signal (live-logged 2026-07-06-19-20: one suppressed ask muted Rex all session).
_identity_prompt_in_flight_at: float = 0.0
_last_identity_prompt_at: float = 0.0
_identity_prompt_reply_until: float = 0.0
_IDENTITY_PROMPT_COOLDOWN_SECS = 45.0

# Pending RELATIONSHIP prompt: Rex asked the engaged person who the stranger is.
# When set, the next user utterance should be parsed for {name, relationship}
# and, if found, the new face is enrolled and an edge saved.
_pending_relationship_prompt = threading.Event()
_pending_relationship_context: dict = {}  # {"engaged_person_id": int, "engaged_name": str, "slot_id": str, "asked_at": float}
# Set between submitting a "who's this?" candidate and it actually speaking, so the reactor
# doesn't re-submit a duplicate every tick before the first one wins arbitration. Under
# ENFORCE the governor can REJECT the candidate (its speak_fn/on_spoke never run), so the
# latch has a stale-timeout auto-clear (RELATIONSHIP_PROMPT_INFLIGHT_STALE_SECS) — mirrors
# _identity_prompt_in_flight.
_relationship_prompt_in_flight = threading.Event()
_relationship_prompt_in_flight_at: float = 0.0
_RELATIONSHIP_PROMPT_COOLDOWN_SECS = 45.0
_UNKNOWN_WITH_ENGAGED_CONFIRM_SECS = 5.0
# Per-session slot ids we've already asked about, so Rex doesn't re-ask.
_asked_relationship_slots: set[str] = set()
# Track first-seen time of each unknown slot (while any engaged conversation is open).
_unknown_first_seen_at: dict[str, float] = {}
# Monotonic time a SOLO unknown face (no known person in frame) first appeared, for the
# identity-prompt grace window. A known face reads as "unknown" for the tick or two it
# takes recognition to resolve at startup; without a grace period Rex fired "I don't
# know you yet" one tick before recognizing Bret. Reset whenever the scene is no longer
# solo-unknown (a known face resolved, or the unknown left).
_solo_unknown_since: float = 0.0

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


def event_recently_anticipated(person_id: int, event_id: int) -> bool:
    """True if the PROACTIVE anticipation path has already raised this (person, event)
    pair this session — so the reply path's open-plans injection can skip it and the two
    don't both bring up the same upcoming plan."""
    try:
        return (int(person_id), int(event_id)) in _anticipated_events
    except (TypeError, ValueError):
        return False


def note_event_anticipated(person_id: int, event_id: int) -> None:
    """Mark a (person, event) as raised this session so the reply context and proactive path
    don't double-surface it. Used by open-plans and open-commitments (the accountability
    needle marks itself here so the same promise isn't ribbed every turn). Session-scoped."""
    if person_id is None or event_id is None:
        return
    try:
        _anticipated_events.add((int(person_id), int(event_id)))
    except (TypeError, ValueError):
        pass

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

# Startup-only OpenAI presence fallback (runs once per boot when the dlib room scan
# finds nobody). `_started` gates the one spawn; `_active` blocks the empty-room line
# while a verification sweep is in flight; `_verified_empty_at` marks a confirmed
# empty room (telemetry — the "no organics" line is now truthful).
_startup_presence_fallback_started: bool = False
_startup_presence_fallback_active: bool = False
_startup_openai_verified_empty_at: float = 0.0

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
# Wave-back: last time Rex waved back (global) + per-person last-reacted (debounce so a
# single wave fires one reaction, and the same person isn't re-waved-at for a cooldown).
# _pending_wave_back latches a detected wave (a transient ~2s gesture) so it's still
# answered once Rex is free — a wave seen mid-turn isn't lost. {"key","name","at"}.
_last_wave_reaction_at: float = 0.0
_wave_reacted_keys: dict[str, float] = {}
_pending_wave_back: Optional[dict] = None
# Repeat-wave comedy bit: per-person (last_response_monotonic, consecutive_wave_count).
# Consecutive waves escalate (greet → silent wave → joke → give-up → ignore); the count
# resets after WAVE_BACK_ESCALATION_RESET_SECS with no wave from that person.
_wave_escalation: dict[str, tuple[float, int]] = {}
# Stability gate: consecutive ticks each visible person has read gesture=='waving'.
# A wave is only trusted once its streak reaches WAVE_BACK_CONFIRM_FRAMES — a held human
# wave persists across ticks; a flickering non-human blob (a pillow) does not.
_wave_streak: dict[str, int] = {}
_last_wave_close_log_at: float = 0.0  # throttle for the "face too close" suppression log
_last_wave_static_log_at: float = 0.0  # throttle for the static-wrist (no-motion) veto log
# Land-the-laugh / take-a-bow: per-session count + last-fire time (a plain dict so
# mutation needs no `global`; reset on session reset alongside the wave state).
_room_reacted: dict[str, float] = {"count": 0.0, "last_at": 0.0}
# "Wait, that's new" change detection: per-session cooldown/cap + per-label de-dup.
_room_change_state: dict[str, float] = {"count": 0.0, "last_at": 0.0}
_room_change_remarked: set[str] = set()
# Held-object remark ("what's that you're drinking?"): per-session cooldown/cap,
# per-label de-dup, and first-seen persistence tracking so a one-frame near_person
# flicker never fires it.
_held_object_state: dict[str, float] = {"count": 0.0, "last_at": 0.0}
_held_object_remarked: set[str] = set()
_held_object_first_seen: dict[str, float] = {}
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
# Temporal hysteresis for a single visible face: (candidate_person_id, consecutive_count)
# of recognition ticks that disagree with the currently-held identity. The bound
# identity only switches once a NEW person is seen for FACE_IDENTITY_SWITCH_CONFIRM_FRAMES
# consecutive ticks — damps the known<->known HOG flicker (Bret<->Wade).
_pending_solo_switch: Optional[tuple[int, int]] = None

# Temporal persistence for UNKNOWN faces: consecutive recognition ticks on which any
# unknown face was detected. An unknown is only exposed as a real "person" (and allowed
# to arm the who's-the-guest agenda) after FACE_UNKNOWN_CONFIRM_FRAMES consecutive ticks,
# so transient phantom faces (clutter, a shape on the wall, a glance at a messy shelf) are
# ignored while a genuine newcomer — who persists — still clears the gate in ~1s.
_unknown_visible_streak = 0


def _unknown_confirm_frames() -> int:
    return max(1, int(getattr(config, "FACE_UNKNOWN_CONFIRM_FRAMES", 3) or 1))


def _update_unknown_streak(had_raw_unknown: bool) -> int:
    """Advance the unknown-face persistence streak for one recognition tick (resets to 0
    on a tick with no unknown face). Returns the new streak."""
    global _unknown_visible_streak
    _unknown_visible_streak = (_unknown_visible_streak + 1) if had_raw_unknown else 0
    return _unknown_visible_streak


# Throttle for the low-confidence unidentified-face veto log.
_last_lowconf_face_log_at: float = 0.0


def _unknown_face_conf_ok(det: dict) -> bool:
    """True if a face that FAILED identification scores high enough on the detector to
    count as an unknown person at all.

    A KNOWN face is protected by the embedding match; an unknown has nothing but the
    detector's own score — and SCRFD's clutter false-positives hug the 0.5 accept
    threshold while real faces score well above it (live-logged 2026-08-05: a workshop
    shelf minted a persistent "face" that survived the pose guard once the room emptied
    and got the full "what name should I save for you?" treatment). Below
    FACE_UNKNOWN_MIN_CONFIDENCE the face is presumptively clutter: it neither counts as
    an unknown person nor feeds the persistence streak. dlib detections carry no score
    and pass unchanged; 0 disables the gate."""
    global _last_lowconf_face_log_at
    conf = det.get("confidence") if isinstance(det, dict) else None
    if not isinstance(conf, (int, float)):
        return True  # dlib backend / no score — gate is insightface-only
    floor = float(getattr(config, "FACE_UNKNOWN_MIN_CONFIDENCE", 0.62) or 0.0)
    if floor <= 0.0 or float(conf) >= floor:
        return True
    now = time.monotonic()
    if (now - _last_lowconf_face_log_at) > 10.0:
        _last_lowconf_face_log_at = now
        box = det.get("bounding_box") or ()
        _log.info(
            "[face_conf_gate] unidentified face det_score=%.2f below %.2f floor — "
            "ignored as clutter (box=%s)",
            float(conf), floor, tuple(int(v) for v in box[:4]) if box else "n/a",
        )
    return False

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

# Special-case celebrity greeting for Joy / T-Joy / Exudica, galactic hair-styling legend.
_hair_stylist_greeted_this_session: set[int] = set()
_pending_hair_stylist_greetings: dict[int, dict] = {}

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

# Arrival (monotonic) of the CURRENT visit, per person key — set once per visit and
# held until departure (unlike _first_sight_seen_at, which is popped at greeting time),
# so the departure reaction can log "I spent ~N minutes with <name>" to rex.db.
_visit_started_at: dict = {}

# Animal arrival dedupe uses species/position signatures instead of unstable
# animal_1/animal_2 IDs returned by the vision prompt.
_animal_seen_signatures: set[str] = set()
_animal_reacted_at: dict[str, float] = {}
# Species-level announce cooldown (owner 2026-08-02: the dog running in and out
# of frame re-announced "small furry lifeform" repeatedly — the per-signature
# cooldown keys on species:POSITION, so every new position was a "new" animal).
_animal_species_reacted_at: dict[str, float] = {}
_pending_animal_arrivals: dict[str, dict] = {}
# Species-level PRESENCE ledger — the dynamic comings-and-goings bit (owner
# 2026-08-03: "if the animal goes away and then comes back, he should mention it
# as a joke; repeats can trigger more — but not too annoying"). Replaces the flat
# 5-minute cooldowns as the staging logic: first sighting reacts, a REAL departure
# (out of frame past a grace window, so frame flicker doesn't count) followed by a
# return earns an escalating return joke, paced by a min remark gap + session cap.
# Per species: {present, first_seen_at, last_seen_at, departed_at, return_count,
# remarks_spoken, last_remark_at}.
_animal_presence: dict[str, dict] = {}
_last_startle_sound_reaction_at: float = 0.0

# Crowd-change reaction debounce. The camera crowd count flickers (a face lost for
# one frame reads pair->alone->pair), and the raw "label changed" check fired a
# "now it's just us" line the same second Rex greeted the pair — a bystander read
# the contradiction + repeat as "your code glitched". We only react once a NEW
# label has PERSISTED past CROWD_CHANGE_SETTLE_SECS.
_crowd_change_reacted_label: str = ""
_crowd_change_pending_label: str = ""
_crowd_change_pending_since: float = 0.0

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

# Directed-gaze hold: an explicit "look down" (etc.) pins the head where it was
# told to look for a cooldown window, instead of letting the speaker room-scan,
# the adaptive rest drift, or the idle wander pull it back up to level.
_directed_gaze_hold_lock = threading.Lock()
_directed_gaze_hold: dict = {"until": 0.0, "direction": None, "started_at": 0.0}

# Learned vertical rest gaze. Active face tracking still owns the exact gaze
# baseline; this only changes where Rex settles/searches after a face disappears.
_adaptive_head_rest: dict = {
    "lift": int(config.SERVO_CHANNELS["headlift"]["neutral"]),
    "tilt": int(config.SERVO_CHANNELS["headtilt"]["neutral"]),
    "samples": 0,
    "updated_at": 0.0,
}

# Last time a mood-driven idle body gesture fired (monotonic), for cooldown spacing.
_last_mood_gesture_at: float = 0.0
# Whether the mood layer currently owns the visor (so it knows to release it to neutral
# when the mood decays), and the last breathing cadence it asserted (to avoid churn).
_mood_owns_visor: bool = False
_last_mood_breathing: Optional[str] = None

# Idle "mind of his own" head wander: when the conversation lulls while a face is locked,
# Rex sometimes looks AWAY around the room, then returns his gaze and may re-greet. The
# face-tracking loop (12.5Hz) drives the motion when `active`; the 1Hz consciousness loop
# decides when to start one and whether to re-greet on re-acquiring the face. Guarded by
# _idle_wander_lock since two threads touch it.
_idle_wander_lock = threading.Lock()
_idle_wander: dict = {
    "active": False,        # currently looking around (face loop drives the motion)
    "until": 0.0,           # monotonic deadline for the wander
    "waypoints": [],        # list of (neck, lift, tilt) poses to visit; last = return gaze
    "index": 0,             # current waypoint
    "reached_at": 0.0,      # when the current waypoint was reached (for dwell)
    "last_at": 0.0,         # last wander finish time (cooldown)
    "pending_regreet": False,  # a wander just finished; eligible to re-greet on re-lock
    "regreet_deadline": 0.0,   # how long the re-greet opportunity stays open
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
    # An explicit directed look ("look down") owns the head right now — don't let
    # fresh speech kick off a room scan that would sweep his gaze back up.
    if search_requested and directed_gaze_hold_active(now):
        search_requested = False

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
            "waypoint_committed_at": 0.0,
            "waypoint_pose": None,
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


def hold_directed_gaze(direction: str, *, secs: Optional[float] = None) -> None:
    """Commit Rex to a user-directed gaze (e.g. "look down") for a cooldown.

    While the hold is active the speaker-search room scan and the adaptive
    head-rest drift are suppressed, and the idle wander stands down, so Rex's
    head stays where he was told to look instead of popping back up to level.
    Face tracking still runs: if he spots someone he locks on and keeps watching
    them. The hold lapses after the cooldown so he resumes looking around.
    """
    if not bool(getattr(config, "DIRECTED_LOOK_HOLD_ENABLED", True)):
        return
    norm = (direction or "").strip().lower()
    # Re-centering to neutral isn't a gaze worth pinning in place.
    if norm in ("", "center", "centre", "current", "front", "forward", "ahead", "straight"):
        clear_directed_gaze_hold()
        return
    if secs is None:
        secs = float(getattr(config, "DIRECTED_LOOK_HOLD_SECS", 25.0))
    secs = max(0.0, float(secs))
    if secs <= 0.0:
        return
    now = time.monotonic()
    with _directed_gaze_hold_lock:
        _directed_gaze_hold.update({
            "until": now + secs,
            "direction": norm,
            "started_at": now,
        })
    # A fresh directed look supersedes any in-flight speaker room-scan: stop the
    # search so it can't sweep the head back up out from under the hold.
    with _speaker_gaze_lock:
        if _speaker_gaze_intent:
            _speaker_gaze_intent["search_requested"] = False
            _speaker_gaze_intent["search_plan"] = None
            _speaker_gaze_intent["search_plan_index"] = 0
            _speaker_gaze_intent["waypoint_committed_at"] = 0.0
            _speaker_gaze_intent["waypoint_pose"] = None
    _log.info("[directed_gaze] hold direction=%s secs=%.1f", norm, secs)


def directed_gaze_hold_active(now: Optional[float] = None) -> bool:
    """True while an explicit directed look is still being held."""
    now = time.monotonic() if now is None else now
    with _directed_gaze_hold_lock:
        return float(_directed_gaze_hold.get("until") or 0.0) > now


def clear_directed_gaze_hold() -> None:
    """Release any directed-gaze hold so normal idle behavior resumes."""
    with _directed_gaze_hold_lock:
        _directed_gaze_hold.update({"until": 0.0, "direction": None, "started_at": 0.0})


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
        _speaker_gaze_intent["waypoint_committed_at"] = 0.0
        _speaker_gaze_intent["waypoint_pose"] = None
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


def _face_recognized_chirp() -> None:
    """Warm 'oh it's you' recognition ding, fired the first time Rex greets a known
    person this session. Best-effort; the effect layer owns the cooldown/no-audio gate."""
    try:
        from audio import sound_effects
        sound_effects.play("face_recognized")
    except Exception:
        pass


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
    # If this line asked someone how THEY are doing, spend the ask so the next
    # greeting / reply doesn't run the same ritual at them. Detected from the final
    # TEXT rather than from which prompt-builder ran, because the builder only says
    # what Rex was told to do — the text says what he actually said.
    try:
        _ask_target = target_person_id
        if _ask_target is None:
            _engaged = get_recent_engagement()
            _ask_target = (_engaged or {}).get("person_id")
        greeting_cadence.note_wellbeing_ask(_ask_target, text)
    except Exception:
        pass

    # One-shot spend of the landed-reaction awareness for PROACTIVE lines too (the
    # reply path spends via interaction._register_rex_utterance) — a lull line whose
    # prompt carried the "they smiled" beat uses it up the same as a reply would.
    try:
        from intelligence import reaction_awareness
        reaction_awareness.note_rex_spoke()
    except Exception:
        pass

    # A Rex line from another behavior (smile reaction, greeting, idle banter)
    # during an active "tell me about" briefing leaves the teller unsure
    # whether the file is still open — let the flow queue its re-anchor
    # question. Lazy import: interaction imports consciousness at module level.
    try:
        from intelligence import interaction as interaction_mod
        interaction_mod.tell_about_on_external_rex_line(source)
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
    # INTERACTION_PAUSED (Memory Banks editor open) halts ALL proactive speech: this is
    # the gate every proactive path checks, and speech_engine.can_proactive_speak() calls
    # it too — so no presence/idle/curiosity LLM calls fire while editing.
    if getattr(config, "INTERACTION_PAUSED", False):
        return False
    return state_module.get_state() not in (State.QUIET, State.SLEEP, State.SHUTDOWN)


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
# Authored fallbacks used when the context-aware LLM reaction is unavailable. They
# must NOT narrate the camera ("Smile detected", "logged", "detected") — that breaks
# the illusion; Rex reacts to the face like a person would, never announcing a sensor.
_FACIAL_EXPRESSION_REACTION_LINES = {
    "smile": (
        "There it is. A smile. I knew the diagnostics would eventually find joy.",
        "Careful — that kind of optimism is how droids get assigned extra duties.",
        "There's the grin. I'll file it under rare but encouraging anomalies.",
        "Look at that, actual visible morale. I will pretend I had nothing to do with it.",
    ),
    "surprise": (
        "That was a full photoreceptor-wide shock face. What did the galaxy do now?",
        "You just looked like the hyperdrive coughed up a receipt. What happened?",
        "Whoa, the wide eyes. Was it my charm, or did reality file another complaint?",
        "That expression says someone moved your starship. Care to brief the droid?",
        "Did I say something brilliant, or did the universe just get rude?",
    ),
    "frown": (
        "That frown has its own gravity well. Want to vent before it starts charging rent?",
        "You look displeased. If it helps, I also disapprove of most things.",
        "Organic morale appears to be under warranty review. What's the damage?",
        "That expression is doing sad trombone without the trombone. What's up?",
        "Your face just filed a complaint. Need a soundtrack, or a target?",
    ),
    "brow_furrow": (
        "That is a serious thinking face. Either a breakthrough, or math has betrayed you.",
        "The eyebrows have formed a committee. Focused and underfunded, by the look of it.",
        "I see the concentration squint. Want a droid to blame, or are we staying productive?",
        "That forehead is running extra diagnostics. Need a sounding board?",
        "Deep thought, by the look of it. I'll lower the alarm level from doom to paperwork.",
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
        # "I met Jeff Benziger" → rex.db (celebrity easter egg).
        episodic_hooks.celebrity(key, person_name, "Jeff Benziger (History Hunters)", returning=returning)

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


def mark_jt_volleyball_greeted(person_id: int) -> None:
    """External identity flows (the off-camera who's-that ask) delivered the JT
    volleyball intro bit — don't repeat it when his face later hits the camera."""
    try:
        _jt_volleyball_greeted_this_session.add(int(person_id))
    except (TypeError, ValueError):
        pass


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
        # "I met JT" → rex.db (celebrity easter egg).
        episodic_hooks.celebrity(key, person_name, "JT (volleyball legend)", returning=returning)

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


# ── Galactic hair-styling legend (Joy / T-Joy / Exudica) — mirrors the JT bit ──────────────

def _is_galactic_hair_stylist(name: object) -> bool:
    return person_specials.is_galactic_hair_stylist(name)


def _can_hair_stylist_speak(profile: SituationProfile) -> bool:
    return _can_jeff_celebrity_speak(profile)


def _stage_hair_stylist_greeting(*, key: int, person_name: str, returning: bool = False) -> None:
    if not returning and key in _hair_stylist_greeted_this_session:
        return
    existing = _pending_hair_stylist_greetings.get(key)
    if existing:
        existing["last_seen_at"] = time.monotonic()
        existing["returning"] = bool(existing.get("returning") or returning)
        return
    _pending_hair_stylist_greetings[key] = {
        "person_name": person_name,
        "returning": bool(returning),
        "first_seen_at": time.monotonic(),
        "last_seen_at": time.monotonic(),
    }
    _log.info(
        "consciousness: hair-stylist celebrity greeting staged (returning=%s)", bool(returning)
    )


def _try_fire_hair_stylist_greeting(
    *,
    key,
    person_name: Optional[str],
    person_db_id: Optional[int],
    profile: SituationProfile,
    returning: bool = False,
) -> bool:
    if not isinstance(key, int) or not _is_galactic_hair_stylist(person_name):
        return False
    if not returning and key in _hair_stylist_greeted_this_session:
        return False
    if not _can_hair_stylist_speak(profile):
        return False
    label = (
        "return celebrity greeting for hair stylist"
        if returning
        else "first-sight celebrity greeting for hair stylist"
    )
    text = person_specials.galactic_hair_stylist_line(returning=returning)
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
        tag = f"presence:hair_stylist:{key}"
        _last_presence_reaction_at[key] = time.monotonic()
        _log.info("consciousness: firing hair-stylist celebrity greeting: %r", text)
        done = speech_queue.enqueue(text, "starstruck", priority=2, tag=tag)
        _mark_governor_candidate(candidate_id, "accepted", "hair_stylist_enqueued")
        try:
            from memory import people as people_mod
            people_mod.record_greeting(key)
        except Exception as exc:
            _log.debug("record greeting failed for hair stylist person_id=%s: %s", key, exc)
        try:
            conv_log.log_rex(text)
        except Exception as exc:
            _log.debug("conversation log write failed for hair-stylist greeting: %s", exc)
        note_rex_utterance(
            text,
            open_response_wait=False,
            source="presence_reaction",
            topic=label,
            target_person_id=key,
        )
        _hair_stylist_greeted_this_session.add(key)
        _greeted_this_session.add(key)
        _first_sight_seen_at.pop(key, None)
        _pending_hair_stylist_greetings.pop(key, None)
        # "I met Joy" → rex.db (celebrity easter egg).
        episodic_hooks.celebrity(key, person_name, "Joy (galactic hair-styling legend)", returning=returning)

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
            name="hair-stylist-presence-done",
        ).start()
        return True
    except Exception as exc:
        _mark_governor_candidate(candidate_id, "dropped", "hair_stylist_error")
        _proactive_speech_pending.clear()
        try:
            _presence_reaction_lock.release()
        except RuntimeError:
            pass
        _log.debug("hair-stylist greeting failed: %s", exc)
        return False


def _step_hair_stylist_detection(snapshot: dict, profile: SituationProfile) -> bool:
    """Joy / T-Joy / Exudica hair-styling legend — mirrors the JT volleyball detection bit."""
    now = time.monotonic()
    confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
    visible_stylist: Optional[tuple[int, str]] = None
    for person in snapshot.get("people", []) or []:
        person_name = person.get("face_id") or person.get("voice_id") or ""
        if not _is_galactic_hair_stylist(person_name):
            continue
        try:
            key = int(person.get("person_db_id"))
        except (TypeError, ValueError):
            continue
        if key in _hair_stylist_greeted_this_session:
            continue
        first_visible = _first_sight_seen_at.setdefault(key, now)
        if (now - first_visible) < max(0.0, confirm_visible):
            return True
        _stage_hair_stylist_greeting(key=key, person_name=person_name)
        visible_stylist = (key, person_name)
        break

    for pending_key, pending in list(_pending_hair_stylist_greetings.items()):
        name = str(pending.get("person_name") or "Joy")
        if not _is_galactic_hair_stylist(name):
            _pending_hair_stylist_greetings.pop(pending_key, None)
            continue
        stale_after = float(getattr(config, "JEFF_CELEBRITY_GREETING_PENDING_SECS", 45.0) or 45.0)
        if (now - float(pending.get("last_seen_at") or now)) > max(1.0, stale_after):
            _pending_hair_stylist_greetings.pop(pending_key, None)
            continue
        _try_fire_hair_stylist_greeting(
            key=pending_key,
            person_name=name,
            person_db_id=pending_key,
            profile=profile,
            returning=bool(pending.get("returning")),
        )
        return True
    return visible_stylist is not None


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


def _stage_animal_remark(species: str, animal: dict, *, kind: str,
                         return_count: int, now: float) -> None:
    """Queue one pending animal remark for this species (arrival or return joke)."""
    pending = dict(animal)
    pending["species"] = species
    pending["signature"] = _animal_signature(animal)
    pending["kind"] = kind
    pending["return_count"] = return_count
    pending["first_seen_at"] = now
    pending["last_seen_at"] = now
    _pending_animal_arrivals[species] = pending
    _log.info("consciousness: staged animal %s species=%s return_count=%d",
              kind, species, return_count)


def _stage_animal_arrivals(snapshot: dict) -> None:
    """Species-level presence tracking → staged arrival/return remarks.

    The old shape was one remark per species per flat 5-minute window, so the dog
    leaving and wandering back was just silence. New shape (owner 2026-08-03): the
    bit follows the animal's comings and goings —
      * first sighting: the arrival reaction, unchanged;
      * out of frame under ANIMAL_DEPARTURE_GRACE_SECS: NOT a departure (the
        floor-level wide-angle loses the dog constantly — flicker stays silent,
        which is what the old species cooldown was really protecting);
      * a real departure then a sighting: a RETURN joke, escalating with
        return_count ("womp rat energy" → "doing laps" → "get it a badge");
      * anti-annoyance pacing: at least ANIMAL_RETURN_REMARK_MIN_GAP_SECS between
        SPOKEN remarks per species and at most ANIMAL_REMARK_SESSION_CAP spoken
        remarks per species per run — beyond either, returns update state silently;
      * absence of ANIMAL_FRESH_ARRIVAL_AFTER_SECS or more resets the bit — the
        next sighting is a fresh arrival again, not "back again" after 3 hours.
    Staged remarks still speak via _fire_pending_animal_arrival_reaction (pending
    survives startup greetings; TTL applies)."""
    if not _last_snapshot:
        return
    now = time.monotonic()
    grace = float(getattr(config, "ANIMAL_DEPARTURE_GRACE_SECS", 30.0))
    fresh_after = float(getattr(config, "ANIMAL_FRESH_ARRIVAL_AFTER_SECS", 1800.0))
    min_gap = float(getattr(config, "ANIMAL_RETURN_REMARK_MIN_GAP_SECS", 120.0))
    cap = int(getattr(config, "ANIMAL_REMARK_SESSION_CAP", 4))

    seen: dict[str, dict] = {}
    for animal in snapshot.get("animals", []) or []:
        if not isinstance(animal, dict) or not animal.get("species"):
            continue
        species = str(animal["species"]).strip().lower()
        seen.setdefault(species, animal)
        _animal_seen_signatures.add(_animal_signature(animal))

    # Departures: time-based grace, so one dropped frame never counts as leaving.
    for species, rec in _animal_presence.items():
        if rec.get("present") and species not in seen:
            last = float(rec.get("last_seen_at") or now)
            if now - last >= grace:
                rec["present"] = False
                rec["departed_at"] = last
                _log.info("consciousness: animal departed species=%s (unseen %.0fs)",
                          species, now - last)

    for species, animal in seen.items():
        if species in _pending_animal_arrivals:
            _pending_animal_arrivals[species]["last_seen_at"] = now
        rec = _animal_presence.get(species)
        departed_at = float((rec or {}).get("departed_at") or 0.0)
        if rec is None or (not rec.get("present")
                           and departed_at and now - departed_at >= fresh_after):
            # Never seen this run, or gone long enough that "back again" would read
            # weird — either way the bit starts fresh.
            _animal_presence[species] = {
                "present": True, "first_seen_at": now, "last_seen_at": now,
                "departed_at": None, "return_count": 0,
                "remarks_spoken": 0, "last_remark_at": 0.0,
            }
            _stage_animal_remark(species, animal, kind="arrival",
                                 return_count=0, now=now)
            continue
        rec["last_seen_at"] = now
        if rec.get("present"):
            continue  # still here (or frame flicker) — nothing new to say
        # A real departure followed by a sighting: the return bit.
        rec["present"] = True
        rec["return_count"] = int(rec.get("return_count") or 0) + 1
        if int(rec.get("remarks_spoken") or 0) >= cap:
            continue  # bit is spent for this run — welcome back silently
        if (now - float(rec.get("last_remark_at") or 0.0)) < min_gap:
            continue  # too soon after the last remark — let it breathe
        _stage_animal_remark(species, animal, kind="return",
                             return_count=rec["return_count"], now=now)


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

# Return-joke ladder: the bit escalates with how many times the animal has left
# and wandered back this run. Furry companions get the warm riff; everything else
# gets the neutral pool (a returning bird is a cameo, not a roommate).
_ANIMAL_RETURN_LINES_FIRST = (
    "Whoa — the furry lifeform is back. It's got the energy of a womp rat on a sugar run.",
    "The furry lifeform has returned. Apparently I'm on its patrol route now.",
    "Furry lifeform re-entry detected. Smooth landing. No customs check.",
)
_ANIMAL_RETURN_LINES_SECOND = (
    "That's twice now. The furry one is either doing laps or casing the place.",
    "Back AGAIN. I'm starting to feel like a waypoint on a smuggling run.",
    "Second re-entry logged. Someone is optimizing a patrol route.",
)
_ANIMAL_RETURN_LINES_MANY = (
    "I've lost count. The furry lifeform now has standing docking clearance.",
    "In. Out. In again. I admire the cardio, honestly.",
    "At this point the creature works here. Somebody get it a badge.",
)
_ANIMAL_RETURN_LINES_GENERIC = (
    "The creature has returned. Bold.",
    "Re-entry detected. The lifeform's commute continues.",
    "Ah. The creature again. We meet as equals.",
)


def _animal_reaction_frame_and_line(animal: dict):
    species = (animal.get("species") or "creature").strip().lower()
    if (animal.get("kind") or "arrival") == "return":
        # A return isn't a surprise — the joke is that Rex has clocked the pattern.
        # Warm/amused frame, escalating line pool by how many round-trips so far.
        count = int(animal.get("return_count") or 1)
        if _animal_is_furry_companion(species, animal):
            pool = (_ANIMAL_RETURN_LINES_FIRST if count <= 1
                    else _ANIMAL_RETURN_LINES_SECOND if count == 2
                    else _ANIMAL_RETURN_LINES_MANY)
        else:
            pool = _ANIMAL_RETURN_LINES_GENERIC
        frame = emotion_orchestrator.frame_for_emotion(
            "happy",
            intensity=0.6,
            source="event",
            trigger=f"animal_return:{species}",
        )
        return frame, random.choice(pool)
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


# ───────────────────────────────────────────────────────────────────────────────────
# Episodic memory CAPTURE (Phase 1) lives in intelligence/episodic_hooks.py — the
# thin, gated, failure-safe rex.db capture wrappers were extracted there to keep this
# module smaller. Call them as episodic_hooks.<name>(...). Nothing reads them back yet
# (Phase 2 will); the startup-image latch + scene-change dedupe live in that module.
# ───────────────────────────────────────────────────────────────────────────────────


def _fire_pending_animal_arrival_reaction() -> bool:
    if not _pending_animal_arrivals:
        return False
    now = time.monotonic()
    stale_after = float(getattr(config, "ANIMAL_PENDING_REACTION_TTL_SECS", 90.0))
    for pending_key, animal in list(_pending_animal_arrivals.items()):
        if now - float(animal.get("last_seen_at") or now) > stale_after:
            _pending_animal_arrivals.pop(pending_key, None)
            continue
        frame, line = _animal_reaction_frame_and_line(animal)

        _ep_species = (animal.get("species") or "creature")
        _ep_position = animal.get("position")
        _ep_kind = (animal.get("kind") or "arrival")

        def _on_spoke(pending_key=pending_key, frame=frame, line=line, now=now,
                      species=_ep_species, position=_ep_position,
                      kind=_ep_kind) -> None:
            # Prime the face + retire the pending remark only on an actual spoken
            # reaction — under ENFORCE a losing candidate must not pop the queue.
            _prime_emotion_frame(frame)
            species_key = (species or "creature").strip().lower()
            _animal_reacted_at[pending_key] = now
            _animal_species_reacted_at[species_key] = now
            # The presence ledger paces the bit on SPOKEN remarks only — a staged
            # line that lost the governor race must not burn the session cap.
            rec = _animal_presence.get(species_key)
            if rec is not None:
                rec["remarks_spoken"] = int(rec.get("remarks_spoken") or 0) + 1
                rec["last_remark_at"] = now
            _pending_animal_arrivals.pop(pending_key, None)
            if kind == "arrival":
                episodic_hooks.animal(species, position)  # "I saw a dog" → rex.db
            _log.info(
                "consciousness: animal %s reaction fired species=%s text=%r",
                kind,
                species_key,
                line,
            )

        if _speak_async(
            line,
            frame.affect,
            purpose="world.animal_arrival",
            label=(f"animal {_ep_kind}: "
                   f"{(animal.get('species') or 'creature').strip().lower()}"),
            on_spoke=_on_spoke,
            force_salient=True,
        ):
            return True
    return False


# Scenery-change remark: a one-line "did we move?" when this run's startup scene differs
# from the last run's. episodic_hooks computes the remark (off the tick, after captioning);
# this speaks it once when Rex can.
_scenery_remark_pending = None


def _step_scenery_change() -> None:
    """Speak the queued change-of-scenery remark once, if there is one and Rex can talk."""
    global _scenery_remark_pending
    if _scenery_remark_pending is None:
        try:
            taken = episodic_hooks.take_scenery_remark()
        except Exception:
            taken = None
        if not taken:
            return
        _scenery_remark_pending = taken
    remark = _scenery_remark_pending

    def _on_spoke(remark=remark) -> None:
        global _scenery_remark_pending
        _scenery_remark_pending = None
        _log.info("consciousness: scenery-change remark spoken: %r", remark)

    _speak_async(
        remark,
        "curious",
        purpose="world.scenery_change",
        label="scenery change",
        on_spoke=_on_spoke,
        force_salient=True,
    )


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
    """The CLASSIFIER label (per-face adaptive baseline, vision/face_expression) is the
    sole trigger. Raw blendshapes only corroborate confidence when the label already says
    smiling — they must never trigger alone: MediaPipe over-reads mouthSmile on resting
    faces at the robot's upward camera angle, which is exactly the false '[laughs] there
    it is' / 'comedy validated' misfire on a non-smiling face."""
    reading = _expression_reading(person)
    expression = _norm_expression_label(reading.get("expression"))
    mood = _norm_expression_label(reading.get("mood"))
    confidence = _safe_confidence(reading.get("confidence"))
    blend_score = _smile_blendshape_score(reading.get("blendshapes") or {})
    min_conf = _safe_confidence(getattr(config, "SMILE_REACTION_MIN_CONFIDENCE", 0.45))
    if expression in _SMILE_LABELS or mood in _SMILE_LABELS:
        return max(confidence, blend_score) >= min_conf
    return False


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
    if kind == "surprise":
        return _safe_confidence(
            getattr(config, "FACIAL_EXPRESSION_REACTION_SURPRISE_MIN_CONFIDENCE", 0.50)
        )
    return _safe_confidence(
        getattr(config, "FACIAL_EXPRESSION_REACTION_MIN_CONFIDENCE", 0.55)
    )


def _person_reactable_expression(person: dict) -> tuple[Optional[str], float]:
    """Only the CLASSIFIER's baseline-corrected label can nominate an expression kind;
    raw blendshapes merely raise its confidence. (They previously nominated alone, which
    let a resting face's over-read mouthSmile/browDown fire reactions the classifier
    had already rejected as neutral.)"""
    reading = _expression_reading(person)
    expression = _norm_expression_label(reading.get("expression"))
    mood = _norm_expression_label(reading.get("mood"))
    confidence = _safe_confidence(reading.get("confidence"))
    blendshapes = reading.get("blendshapes") or {}
    best_kind: Optional[str] = None
    best_score = 0.0
    for kind, labels in _FACIAL_EXPRESSION_REACTION_LABELS.items():
        if expression not in labels and mood not in labels:
            continue
        score = max(confidence, _expression_kind_blend_score(kind, blendshapes))
        if score > best_score:
            best_kind = kind
            best_score = score
    if best_kind is None or best_score < _facial_expression_reaction_min_confidence(best_kind):
        return None, best_score
    return best_kind, best_score


def _expression_is_habitual_disposition(person_id: Optional[int], kind: Optional[str]) -> bool:
    """True when `kind` is the person's KNOWN dominant resting expression.

    Some people read as habitually brow-furrowed/intense (or perpetually smiling).
    Reacting to that baseline ("you're not exactly sold on this, are you?") mistakes a
    visual habit for a live emotional signal — the disposition trend that already feeds
    the prompt explicitly flags it as "a light visual habit, not a diagnosis". Gate on a
    minimum sample count so a thin profile can't suppress a genuine reaction.
    """
    if not kind or not isinstance(person_id, int):
        return False
    if not bool(getattr(config, "FACIAL_EXPRESSION_REACTION_RESPECT_DISPOSITION", True)):
        return False
    try:
        from memory import disposition as disposition_memory
        stats = disposition_memory.get_stats(person_id)
    except Exception as exc:
        _log.debug("disposition suppression lookup failed person_id=%s: %s", person_id, exc)
        return False
    if not stats:
        return False
    try:
        total = int(stats.get("total_samples") or 0)
    except (TypeError, ValueError):
        total = 0
    min_samples = int(
        getattr(config, "FACIAL_EXPRESSION_REACTION_DISPOSITION_MIN_SAMPLES", 20) or 20
    )
    if total < max(1, min_samples):
        return False
    dominant = str(stats.get("dominant_expression") or "").strip().lower()
    return bool(dominant) and dominant == str(kind).strip().lower()


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


def _wave_person_key(person: dict) -> str:
    """Stable-ish key for wave-back debounce: the db id when known, else the world_state
    slot id, else a generic unknown bucket."""
    pid = person.get("person_db_id")
    if pid is not None:
        return f"db:{pid}"
    slot = person.get("id")
    return f"slot:{slot}" if slot else "unknown"


def _wave_face_too_close(person: dict) -> bool:
    """True when the waver's face fills too much of the frame to be a real across-the-room
    wave — i.e. they're right at the camera (a desk/laptop webcam), where a detected 'wave'
    is almost always a near-camera artifact (an arm/object) and a wave-back makes no sense.
    Gated by WAVE_BACK_MAX_FACE_FRACTION (face box HEIGHT / frame height; 0 disables)."""
    max_frac = float(getattr(config, "WAVE_BACK_MAX_FACE_FRACTION", 0.0) or 0.0)
    if max_frac <= 0.0:
        return False
    frac = person.get("face_box_height_fraction")
    return isinstance(frac, (int, float)) and float(frac) >= max_frac


def _wave_back_line(first_name: str) -> str:
    """A short warm wave-back line, with the person's name woven in when known."""
    name = (first_name or "").strip()
    if name:
        pool = list(getattr(config, "WAVE_BACK_LINES", []) or [])
        line = random.choice(pool) if pool else "Hey, {name}!"
        return line.replace("{name}", name)
    pool = list(getattr(config, "WAVE_BACK_LINES_NO_NAME", []) or [])
    return random.choice(pool) if pool else "Hey there!"


def _wave_joke_line() -> str:
    pool = list(getattr(config, "WAVE_BACK_JOKE_LINES", []) or [])
    return random.choice(pool) if pool else "Still waving? We've established I have arms."


def _wave_giveup_line() -> str:
    pool = list(getattr(config, "WAVE_BACK_GIVEUP_LINES", []) or [])
    return random.choice(pool) if pool else "Okay, that's the last one."


def _wave_response_plan(level: int, first_name: str) -> tuple[Optional[str], bool, bool]:
    """The escalating repeat-wave bit. Returns (line, should_speak, should_gesture) for the
    Nth consecutive wave from a person:
      1 greet + wave · 2 silent wave · 3 joke + wave · 4 give-up joke (no wave) · 5+ ignore.
    """
    if level <= 1:
        return (_wave_back_line(first_name), True, True)
    if level == 2:
        return (None, False, True)            # silent wave-back
    if level == 3:
        return (_wave_joke_line(), True, True)
    if level == 4:
        return (_wave_giveup_line(), True, False)  # protest, no wave
    return (None, False, False)               # level >= 5: he's done — ignore


def _mirrored_half_period(user_speed: Optional[float]) -> Optional[float]:
    """Map the user's measured wave speed (normalized-x/sec from vision.pose) to Rex's wave
    half-period so a slow wave gets a slow wave-back and a fast one a fast wave-back. Linear
    from [SLOW..FAST] user speed onto [SLOW..FAST] half-period, clamped. None → caller uses
    the fixed default (mirroring disabled or no measurement)."""
    if user_speed is None or not bool(getattr(config, "WAVE_SPEED_MIRROR_ENABLED", True)):
        return None
    slow_s = float(getattr(config, "WAVE_SPEED_MIRROR_SLOW", 0.25))
    fast_s = float(getattr(config, "WAVE_SPEED_MIRROR_FAST", 1.20))
    if fast_s <= slow_s:
        return None
    slow_hp = float(getattr(config, "WAVE_BACK_WRIST_HALF_PERIOD_SLOW_SECS", 0.48))
    fast_hp = float(getattr(config, "WAVE_BACK_WRIST_HALF_PERIOD_FAST_SECS", 0.18))
    frac = max(0.0, min(1.0, (float(user_speed) - slow_s) / (fast_s - slow_s)))
    return slow_hp + frac * (fast_hp - slow_hp)


def _step_exploration(snapshot: dict, profile: SituationProfile) -> None:
    """Supervise an active room-exploration session (a WATCHDOG only).

    The whole wander/survey/narrate sequence runs on exploration.py's own worker
    thread (it blocks on motion completion + multi-second vision calls and must not
    stall this ~1 Hz tick). This step only force-cleans a session that overran the
    duration cap or whose worker died — freeing the base/head/floor it owns."""
    try:
        from intelligence import exploration
        exploration.supervise()
    except Exception as exc:
        _log.debug("exploration supervise error: %s", exc)


def _step_autonomous_motion(snapshot: dict, profile: SituationProfile) -> None:
    """Autonomous base motion: rotate to face the tracked person, approach a far one.
    All decision logic lives in intelligence/motion_agency.py (turn/come are
    closed-loop firmware commands under the ESP32's ToF reflexes)."""
    try:
        from intelligence import motion_agency
        motion_agency.step(snapshot, profile)
    except Exception as exc:
        _log.debug("autonomous motion step error: %s", exc)


def _step_battery_awareness(snapshot: dict, profile: SituationProfile) -> None:
    """Pack-voltage awareness from base telemetry (intelligence/battery_awareness.py).
    Dormant until the INA226 is wired (firmware reports batt_mv=-1 without it)."""
    try:
        from intelligence import battery_awareness
        battery_awareness.step(snapshot, profile)
    except Exception as exc:
        _log.debug("battery awareness step error: %s", exc)


def _step_wave_reaction(snapshot: dict, profile: SituationProfile) -> None:
    """If a visible person waves, wave back (+ one short warm greeting) — the way you'd
    return a wave from across a room.

    Two phases, because a wave is a transient (~2s) gesture but Rex may be busy (mid-turn,
    speaking) when it lands:
      (A) DETECT + LATCH — record the wave the moment the pose pipeline classifies
          'waving' onto world_state.people, even mid-turn, so it isn't forgotten.
      (B) FIRE WHEN FREE — voice the latched greeting as soon as Rex can speak (not over
          live speech / the user / music / a game). The latch expires after a short TTL so
          a stale wave isn't answered late. Debounced per person + globally.

    Diagnostic logging (info) records detect / fire / expire so a "why didn't it speak?"
    question is answerable from the log instead of guesswork."""
    global _last_wave_reaction_at, _pending_wave_back, _last_wave_close_log_at
    global _last_wave_static_log_at
    if not bool(getattr(config, "WAVE_BACK_ENABLED", True)):
        return
    now = time.monotonic()
    per_person_cd = float(getattr(config, "WAVE_BACK_PER_PERSON_COOLDOWN_SECS", 6.0))

    # ── (A) Detect a fresh wave and latch it (runs even while Rex is busy) ──────────
    # Stability gate: a wave must persist across WAVE_BACK_CONFIRM_FRAMES consecutive
    # ticks before it's trusted. A held human wave spans 2-3 of these ~1 Hz ticks; a
    # flickering non-human blob (a pillow MediaPipe momentarily skeletonizes) cycles
    # random gestures per appearing-frame and virtually never reads 'waving' twice in a
    # row — so it never accumulates a streak and is rejected (live-logged 2026-06-26).
    confirm = max(1, int(getattr(config, "WAVE_BACK_CONFIRM_FRAMES", 2)))
    waving_now: set[str] = set()
    too_close_frac = 0.0
    for person in snapshot.get("people", []) or []:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False:
            continue
        if (person.get("gesture") or "") != "waving":
            continue
        # Too close to the camera (face fills the frame) → a wave at this range is almost
        # always a near-camera artifact (an arm/object by a desk webcam); skip it so it
        # never accumulates a streak.
        if _wave_face_too_close(person):
            try:
                too_close_frac = max(too_close_frac, float(person.get("face_box_height_fraction") or 0.0))
            except Exception:
                pass
            continue
        waving_now.add(_wave_person_key(person))
    if too_close_frac > 0.0 and not waving_now and (now - _last_wave_close_log_at) > 10.0:
        _last_wave_close_log_at = now
        _log.info(
            "consciousness: wave ignored — waver's face fills %.0f%% of the frame height "
            "(>= %.0f%% threshold); too close to the camera for a real wave",
            too_close_frac * 100.0,
            float(getattr(config, "WAVE_BACK_MAX_FACE_FRACTION", 0.0)) * 100.0,
        )
    # Drop any tracked key that isn't waving this tick (gesture changed / gone), then
    # advance the streak for those that are.
    for stale_key in [k for k in _wave_streak if k not in waving_now]:
        del _wave_streak[stale_key]
    for waving_key in waving_now:
        _wave_streak[waving_key] = _wave_streak.get(waving_key, 0) + 1

    for person in snapshot.get("people", []) or []:
        if not isinstance(person, dict):
            continue
        if person.get("face_visible") is False:
            continue
        if (person.get("gesture") or "") != "waving":
            continue
        key = _wave_person_key(person)
        if _wave_streak.get(key, 0) < confirm:
            continue  # wave not yet stable across enough ticks — reject flicker
        if (now - float(_wave_reacted_keys.get(key, 0.0))) < per_person_cd:
            continue  # already waved back at this person recently
        # Capture how fast they're waving NOW (refreshed each tick while waving) so the
        # wave-back can mirror the speed; None if it couldn't be measured. Read BEFORE the
        # latch so it can also gate:
        speed = None
        try:
            from vision import pose as pose_mod
            speed = pose_mod.recent_wave_speed()
        except Exception as exc:
            _log.debug("wave speed read failed: %s", exc)
        # STATIC-WRIST VETO (live-logged 2026-08-05): in a cluttered room MediaPipe plants
        # "wrist" landmarks on chair armrests / shelf edges at face height, which the
        # single-frame posture check reads as 'waving' — but those wrists don't MOVE
        # (measured 0.05–0.09 normalized-x/s vs the 0.25+ a real wave sweeps, per
        # WAVE_SPEED_MIRROR_SLOW). A wave with measurable motion below the floor is
        # furniture, not a greeting. None (no measurement yet) passes: the confirm streak
        # already spans 1-2s of pose ticks, so a real raised hand has samples by now.
        min_speed = float(getattr(config, "WAVE_BACK_MIN_SPEED", 0.15) or 0.0)
        if min_speed > 0.0 and isinstance(speed, (int, float)) and speed < min_speed:
            if (now - _last_wave_static_log_at) > 10.0:
                _last_wave_static_log_at = now
                _log.info(
                    "consciousness: wave ignored for %s — wrist speed %.2f below "
                    "WAVE_BACK_MIN_SPEED %.2f (raised-but-motionless wrist = armrest/"
                    "clutter pose artifact, not a wave)", key, speed, min_speed,
                )
            continue
        name = _first_name(person.get("face_id") or person.get("name"), "")
        if not _pending_wave_back or _pending_wave_back.get("key") != key:
            _fh = person.get("face_box_height_fraction")
            _fh_s = f"{float(_fh):.2f}" if isinstance(_fh, (int, float)) else "n/a"
            _log.info("consciousness: wave detected for %s — queued wave-back "
                      "(face_height=%s speed=%s)", key, _fh_s,
                      ("%.2f" % speed) if isinstance(speed, (int, float)) else "n/a")
        _pending_wave_back = {"key": key, "name": name, "at": now, "speed": speed}
        break

    # ── (B) Fire the latched wave-back as soon as Rex is free ───────────────────────
    pending = _pending_wave_back
    if not pending:
        return
    ttl = float(getattr(config, "WAVE_BACK_PENDING_TTL_SECS", 8.0))
    if (now - float(pending.get("at", 0.0))) > ttl:
        _pending_wave_back = None
        _log.info("consciousness: wave-back expired for %s — Rex wasn't free within %.0fs",
                  pending.get("key"), ttl)
        return
    if (now - _last_wave_reaction_at) < float(getattr(config, "WAVE_BACK_MIN_GAP_SECS", 4.0)):
        return

    key = pending["key"]
    name = pending.get("name") or ""

    # Escalation level = how many consecutive waves from this person (resets after a gap).
    level = 1
    if bool(getattr(config, "WAVE_BACK_ESCALATION_ENABLED", True)):
        reset = float(getattr(config, "WAVE_BACK_ESCALATION_RESET_SECS", 30.0))
        last_t, prev = _wave_escalation.get(key, (0.0, 0))
        if (now - last_t) <= reset:
            level = prev + 1
    line, should_speak, should_gesture = _wave_response_plan(level, name)

    # MID-CONVERSATION wave (field 2026-07-18: a wave 45s into a flowing exchange
    # produced a spoken "Hi there, Bret!" — a duplicate greeting that derailed the
    # conversation): when the person spoke recently, respond like a human would —
    # wave BACK silently, don't restart the greeting.
    try:
        from intelligence import interaction as _intx
        _recent_speech = (
            _intx._last_user_content_at > 0.0
            and (time.monotonic() - _intx._last_user_content_at)
            <= float(getattr(config, "WAVE_BACK_SILENT_IN_CONVERSATION_SECS", 90.0))
        )
    except Exception:
        _recent_speech = False
    if _recent_speech and should_speak:
        should_speak, line = False, None
        should_gesture = True

    # Speaking levels must clear the speech gates; if blocked, HOLD the latch and retry
    # without advancing the bit (so the joke lands when Rex is actually free, not lost).
    # reactive=True breaks through awaiting-reply/active-conversation/pacing but still yields
    # to live speech/music/games; governed=False routes past the action-governor tournament.
    if should_speak:
        if profile.user_mid_sentence:
            return  # don't talk over the person while they're speaking
        if not _can_proactive_speak(reactive=True):
            return
        spoke = _speak_async(
            line, emotion="happy", purpose="wave_back", label="wave back",
            governed=False, reactive=True,
        )
        if not spoke:
            return

    # Responded (spoke and/or waved, or deliberately ignored at high levels) — advance the
    # bit and consume the wave so a sustained wave doesn't re-trigger it.
    user_speed = pending.get("speed")
    half_period = _mirrored_half_period(user_speed)  # None → gesture uses the fixed default
    _wave_escalation[key] = (now, level)
    _wave_reacted_keys[key] = now
    _pending_wave_back = None
    if should_speak or should_gesture:
        _last_wave_reaction_at = now
    if should_gesture:
        # Raise the arm and sweep the wrist between both travel limits, mirroring the user's
        # wave speed when measured (non-blocking; failure-safe / no-ops without servos).
        try:
            from sequences import animations
            animations.wave_back_gesture(half_period=half_period)
        except Exception as exc:
            _log.debug("wave-back animation skipped: %s", exc)
    _log.info(
        "consciousness: wave-back for %s — level=%d speak=%s gesture=%s "
        "user_speed=%s half_period=%s",
        key, level, should_speak, should_gesture,
        ("%.2f" % user_speed) if isinstance(user_speed, (int, float)) else None,
        ("%.2f" % half_period) if isinstance(half_period, (int, float)) else None,
    )


def _visible_amused_person(snapshot: dict) -> bool:
    """True when a currently visible face carries a fresh, confident smile.

    Used to corroborate the audio scene's laughter/applause booleans — the
    MediaPipe expression telemetry ("happy"/smile, per-face adaptive baseline)
    is the witness that a human actually reacted, not Rex's own noise floor.
    """
    min_conf = _safe_confidence(
        getattr(config, "ROOM_REACTION_AMUSEMENT_MIN_CONFIDENCE", 0.5)
    )
    for person in _visible_face_people(snapshot):
        reading = _expression_reading(person)
        if str(reading.get("mood") or "") != "happy":
            continue
        if not _face_expression_reading_is_recent(reading):
            continue
        if _safe_confidence(reading.get("confidence")) >= min_conf:
            return True
    return False


def _step_room_reaction(snapshot: dict, profile: SituationProfile) -> None:
    """Land the laugh / take a bow: react to the ROOM responding to Rex's material.

    The audio scene exposes momentary `applause_detected` / `laughter_detected` booleans
    (and analysis is suppressed while Rex is speaking, so these land just AFTER his line).
    Applause → a quick take-a-bow (`proud_dj_pose` + a line); laughter → a dry
    follow-through. Gated on a recent-Rex-utterance window so ambient noise / music / TV
    can't set him off, a global cooldown that also de-dups one multi-cycle burst, and a
    LOW per-session cap so it never reads as needy. No latch — fire when free, else skip
    (a take-a-bow 10s late is worse than none).
    """
    if not bool(getattr(config, "ROOM_REACTION_ENABLED", True)):
        return

    audio = snapshot.get("audio_scene") or {}
    applause = bool(audio.get("applause_detected"))
    laughter = bool(audio.get("laughter_detected"))
    if not (applause or laughter):
        return

    # The burst detectors can't tell a human laugh from Rex's OWN mechanicals —
    # servo whine, drive-base motor noise, and sound-effect chirps all read as
    # rhythmic bursts (field 2026-07-30: "See? That one was free." at a
    # not-laughing owner right after a back-up move + motion whir; and an
    # applause bow at plain servo noise). Two extra gates:
    #   1. self-noise: skip while the base is moving or within a short window
    #      of any sound-effect start (sfx accompany every autonomous maneuver).
    if bool(getattr(config, "ROOM_REACTION_SELF_NOISE_GUARD_ENABLED", True)):
        try:
            from intelligence import motion_controller
            if motion_controller.is_moving():
                return
        except Exception:
            pass
        try:
            from audio import sound_effects
            if sound_effects.seconds_since_last_play() < float(
                getattr(config, "ROOM_REACTION_SELF_NOISE_GUARD_SECS", 4.0)
            ):
                return
        except Exception:
            pass
    #   2. visual corroboration: only credit the laugh/applause when a visible
    #      face actually looks amused right now. A bow at a stone-faced (or
    #      empty) room reads as a glitch, so no fresh smile → no reaction.
    if bool(getattr(config, "ROOM_REACTION_REQUIRE_VISIBLE_AMUSEMENT", True)):
        if not _visible_amused_person(snapshot):
            return

    now = time.monotonic()
    # Only react when this is plausibly a response to REX — i.e. his last line FINISHED
    # within the window. speech_queue tracks EVERY spoken line (reply, roast, proactive),
    # so a roast's laugh counts where the proactive-only timestamp would miss it; ambient
    # laughter while Rex has been idle does not. (Audio analysis is suppressed during his
    # TTS, so the laugh/applause lands just after he stops.)
    after_rex = float(getattr(config, "ROOM_REACTION_AFTER_REX_SECS", 12.0))
    try:
        from audio import speech_queue
        since_spoke = speech_queue.seconds_since_last_speech()
    except Exception:
        since_spoke = float("inf")
    if since_spoke > after_rex:
        return
    # ...but NOT instantly. The first analysis window after his TTS unmutes still
    # holds his own decaying tail + room echo, which reads as applause/laughter —
    # field 2026-07-24 19:58:31: Rex took a bow at a silent, seated room ~10 s after
    # his own line. Real applause from a human starts later than his own reverb.
    min_after_rex = float(getattr(config, "ROOM_REACTION_MIN_AFTER_REX_SECS", 1.5))
    if since_spoke < min_after_rex:
        return
    # Global cooldown (also collapses the 2-3 consecutive True reads of one burst).
    if (now - float(_room_reacted.get("last_at", 0.0))) < float(
        getattr(config, "ROOM_REACTION_MIN_GAP_SECS", 20.0)
    ):
        return
    if _room_reacted.get("count", 0.0) >= float(getattr(config, "ROOM_REACTION_SESSION_CAP", 3)):
        return
    if profile.user_mid_sentence:
        return
    if not _can_proactive_speak(reactive=True):
        return

    # Applause is the bigger moment — it wins if both fire in the same cycle.
    if applause:
        lines = getattr(config, "ROOM_APPLAUSE_REACTION_LINES", []) or []
        kind, emotion, beat = "applause", "happy", "proud_dj_pose"
    else:
        lines = getattr(config, "ROOM_LAUGHTER_REACTION_LINES", []) or []
        kind, emotion, beat = "laughter", "happy", None
    if not lines:
        return
    # Never the same victory lap twice in one session ("Encore's included…" fired
    # twice in 10 minutes and read as a glitch, not a bit).
    pool = [l for l in lines if l != _room_reacted.get("last_line")] or list(lines)
    line = random.choice(pool)
    _room_reacted["last_line"] = line

    spoke = _speak_async(
        line, emotion=emotion, purpose="room_reaction", label=f"land the {kind}",
        governed=False, reactive=True,
    )
    if not spoke:
        return
    _room_reacted["last_at"] = now
    _room_reacted["count"] = _room_reacted.get("count", 0.0) + 1
    if beat:
        try:
            from sequences import animations
            animations.play_body_beat(beat)
        except Exception as exc:
            _log.debug("room-reaction beat skipped: %s", exc)
    _log.info(
        "consciousness: room reaction — %s (count=%d/%s)",
        kind, int(_room_reacted["count"]),
        getattr(config, "ROOM_REACTION_SESSION_CAP", 3),
    )


def _step_room_change(snapshot: dict, profile: SituationProfile) -> None:
    """Wait — that's new. When the room is KNOWN (Rex has an established object baseline)
    and a genuinely new object shows up — currently present, low recorded sighting count,
    never a fixture — Rex clocks it ONCE with a dry one-liner. Heavily gated because the
    COCO detector is noisy: it needs a baseline first (no fresh-install flood), fires only
    in a lull (plain _can_proactive_speak yields to active conversation), and is bounded by
    a cooldown + a low per-session cap + per-label de-dup."""
    if not bool(getattr(config, "ROOM_CHANGE_REMARK_ENABLED", True)):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return

    labels = {
        str(o.get("label") or "").strip().lower()
        for o in (snapshot.get("objects") or [])
        if isinstance(o, dict) and o.get("label")
    }
    labels = {lbl for lbl in labels if lbl and lbl not in _room_change_remarked}
    if not labels:
        return

    now = time.monotonic()
    if (now - float(_room_change_state.get("last_at", 0.0))) < float(
        getattr(config, "ROOM_CHANGE_COOLDOWN_SECS", 120.0)
    ):
        return
    if _room_change_state.get("count", 0.0) >= float(getattr(config, "ROOM_CHANGE_SESSION_CAP", 3)):
        return

    try:
        from memory import room_model
        # Rex must KNOW the room first — without an established baseline, a fresh install
        # (or a freshly-cleared rex.db) would flag every fixture as "new".
        established_min = int(getattr(config, "ROOM_MODEL_ESTABLISHED_SIGHTINGS", 20))
        if room_model.established_count(established_min) < int(
            getattr(config, "ROOM_CHANGE_MIN_BASELINE", 4)
        ):
            return
        counts = room_model.label_sightings(labels)
    except Exception as exc:
        _log.debug("room-change model read failed: %s", exc)
        return

    lo = int(getattr(config, "ROOM_CHANGE_MIN_SIGHTINGS", 2))
    hi = int(getattr(config, "ROOM_CHANGE_MAX_SIGHTINGS", 12))
    new_labels = sorted(lbl for lbl in labels if lo <= counts.get(lbl, 0) <= hi)
    # DETECTOR HUMILITY (field 2026-07-18: a pillow misread as "handbag" for 4
    # seconds became "New handbag. The room's redecorating without consulting me"):
    # a real new object PERSISTS — require its sightings to span a minimum wall-
    # clock window, and never remark on soft/carriable labels sitting next to a
    # person (usually the person's own stuff, or a misread of them/their couch).
    min_span = float(getattr(config, "ROOM_CHANGE_MIN_SPAN_SECS", 45.0))
    if min_span > 0 and new_labels:
        try:
            spans = room_model.label_spans(new_labels)
            new_labels = [l for l in new_labels if spans.get(l, 0.0) >= min_span]
        except Exception:
            pass
    # FURNITURE never "appears" — the camera PANS, so large fixed classes enter
    # the frame all the time (field 2026-07-18: the bed, misread as "couch",
    # "just appeared out of nowhere"). These classes are excluded outright.
    furniture = {
        s.strip().lower()
        for s in getattr(config, "ROOM_CHANGE_FURNITURE_LABELS", (
            "couch", "bed", "chair", "dining table", "tv", "refrigerator",
            "toilet", "sink", "oven", "microwave", "potted plant",
        ))
    }
    new_labels = [l for l in new_labels if l not in furniture]
    soft = {
        s.strip().lower()
        for s in getattr(config, "ROOM_CHANGE_SOFT_LABELS", (
            "handbag", "backpack", "suitcase", "tie", "umbrella", "cell phone",
            "book", "cup", "bottle", "remote",
        ))
    }
    near_person_labels = {
        str(o.get("label") or "").strip().lower()
        for o in (snapshot.get("objects") or [])
        if isinstance(o, dict) and o.get("near_person")
    }
    new_labels = [l for l in new_labels if not (l in soft and l in near_person_labels)]
    if not new_labels:
        return
    if not _can_proactive_speak():
        return

    label = new_labels[0]
    lines = getattr(config, "ROOM_CHANGE_REMARK_LINES", []) or []
    if not lines:
        return
    # De-dup the label NOW, before speaking: even if the enqueue races and fails, this new
    # object is "handled" for the session — a flickering detection must never re-fire it.
    # (The cooldown + session cap only advance on an actual remark, below.)
    _room_change_remarked.add(label)

    # PERSON PRESENT → the new object is a conversation OPENER, not a room note.
    # Owner feedback 2026-07-06 (RF-DETR spotted his sandwich, Rex said "A wild
    # sandwich appears. The room's got range."): a closed quip wastes the moment —
    # ask about the thing instead ("What kind of sandwich are we dealing with?").
    # The canned observational one-liners remain the ALONE behavior (muttering at
    # the room when there's nobody to ask).
    person_name = _room_change_addressee(snapshot)
    if person_name is not None and bool(
        getattr(config, "ROOM_CHANGE_ASK_WHEN_PERSON_PRESENT", True)
    ):
        prompt = (
            f"You just properly noticed a {label} you hadn't paid attention to "
            f"before, and {person_name} is right here. Your camera pans around, so "
            f"do NOT claim it 'appeared' or 'came out of nowhere' — you just hadn't "
            f"clocked it. React in ONE short in-character Rex line that INVITES "
            f"them to talk about it — genuinely curious, ask something natural about "
            f"the {label} (what kind it is / how it is / where it came from / what's "
            f"the occasion — whatever fits a {label}). Warm and dry, not an "
            f"interrogation; ONE question max. Address {person_name} casually."
        )
        spoke = _generate_and_speak_presence(
            prompt,
            label=f"room change ask: {label}",
            tag_key=f"room_change:{label}",
            emotion="curious",
            purpose="room_change",
        )
    else:
        line = random.choice(list(lines)).replace("{label}", label)
        spoke = _speak_async(
            line, emotion="curious", purpose="room_change", label=f"room change: {label}",
            governed=False,
        )
    if not spoke:
        return
    _room_change_state["last_at"] = now
    _room_change_state["count"] = _room_change_state.get("count", 0.0) + 1
    # Arm the correction latch: "actually, that's a pillow" in the next turns
    # should RENAME the object in the room model, not fall into the person-fact
    # correction machinery (field 2026-07-18: it did, and emitted a canned
    # failure line).
    try:
        from intelligence import room_questions
        room_questions.note_room_remark(label)
    except Exception:
        pass
    _log.info(
        "consciousness: room-change remark — %s (sightings=%d, count=%d/%s, asked=%s)",
        label, counts.get(label, 0), int(_room_change_state["count"]),
        getattr(config, "ROOM_CHANGE_SESSION_CAP", 3), person_name is not None,
    )


def _step_held_object_remark(snapshot: dict, profile: SituationProfile) -> None:
    """Someone is HOLDING something — ask about it ("what's that you're drinking?").

    The direct payoff of person-oriented object salience (owner 2026-07-08: "comment
    on objects I'm holding more often" — he held a cup through two whole sessions and
    Rex never asked). Event-driven, not lull-taxonomy: fires soon after a near_person
    object PERSISTS (first-seen tracking absorbs one-frame flickers), yields to live
    conversation via _can_proactive_speak, and is bounded by a per-label session
    de-dup + cooldown + session cap. Unlike _step_room_change it needs NO room-model
    baseline — a held object is salient on a fresh install too."""
    if not bool(getattr(config, "HELD_OBJECT_REMARK_ENABLED", True)):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return

    now = time.monotonic()
    held_now: dict[str, dict] = {}
    for obj in snapshot.get("objects") or []:
        if not isinstance(obj, dict) or not obj.get("near_person"):
            continue
        label = str(obj.get("label") or "").strip().lower()
        if label:
            held_now[label] = obj
    # First-seen persistence: a label must stay near_person for MIN_HOLD_SECS.
    for label in list(_held_object_first_seen):
        if label not in held_now:
            del _held_object_first_seen[label]
    for label in held_now:
        _held_object_first_seen.setdefault(label, now)

    min_hold = float(getattr(config, "HELD_OBJECT_REMARK_MIN_HOLD_SECS", 5.0))
    ready = [
        label for label, seen_at in _held_object_first_seen.items()
        if label not in _held_object_remarked and (now - seen_at) >= min_hold
    ]
    if not ready:
        return
    if (now - float(_held_object_state.get("last_at", 0.0))) < float(
        getattr(config, "HELD_OBJECT_REMARK_COOLDOWN_SECS", 90.0)
    ):
        return
    if _held_object_state.get("count", 0.0) >= float(
        getattr(config, "HELD_OBJECT_REMARK_SESSION_CAP", 3)
    ):
        return
    if not _can_proactive_speak():
        return

    label = ready[0]
    holder = str(held_now.get(label, {}).get("near_person_name") or "").strip()
    holder = holder.split()[0] if holder else (_room_change_addressee(snapshot) or "them")
    # De-dup NOW, before speaking, so a failed enqueue can't re-fire the same label.
    _held_object_remarked.add(label)

    prompt = (
        f"{holder} is holding a {label} right now (or it's right beside them). React in "
        f"ONE short in-character Rex line that shows genuine curiosity about THEIR "
        f"{label} — the natural small-talk move (\"what's that you're drinking?\" / "
        f"\"where'd that come from?\" / \"what is that?\" — whatever fits a {label}). "
        f"Casual and warm, ONE question max, address {holder} directly."
    )
    spoke = _generate_and_speak_presence(
        prompt,
        label=f"held object ask: {label}",
        tag_key=f"held_object:{label}",
        emotion="curious",
        purpose="held_object_remark",
    )
    if not spoke:
        return
    _held_object_state["last_at"] = now
    _held_object_state["count"] = _held_object_state.get("count", 0.0) + 1
    _log.info(
        "consciousness: held-object remark — %s (holder=%s, count=%d/%s)",
        label, holder, int(_held_object_state["count"]),
        getattr(config, "HELD_OBJECT_REMARK_SESSION_CAP", 3),
    )


def _room_change_addressee(snapshot: dict) -> Optional[str]:
    """First name of a visibly present person to ask about a new object, or None
    when the room is empty (alone → the canned observational line instead).
    An unknown-but-present person still gets asked, generically."""
    try:
        for person in snapshot.get("people") or []:
            if not isinstance(person, dict):
                continue
            if not (person.get("face_visible") or person.get("face_box")):
                continue
            pid = person.get("person_db_id")
            if pid is not None:
                try:
                    from memory import people as people_mod
                    record = people_mod.get_person(int(pid)) or {}
                    name = str(record.get("name") or "").strip()
                    if name:
                        return name.split()[0]
                except Exception:
                    pass
            return "them"   # visible but unidentified — still worth asking
    except Exception as exc:
        _log.debug("room-change addressee lookup failed: %s", exc)
    return None


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
    # Owner rework 2026-08-05: a confirmed landed smile no longer SPEAKS a canned
    # interjection ("Oh look, I made the lifeform smile" — a sensor report wearing
    # a joke, and it over-triggered). It feeds reaction_awareness instead, so Rex's
    # NEXT generated line knows the joke landed and can enjoy it in first person —
    # or not mention it at all. Everything downstream (diary hook, giddy body mood,
    # the cooldown) is shared by both paths.
    fired = False
    if bool(getattr(config, "SMILE_REACTION_CANNED_LINES_ENABLED", False)):
        line = _choose_expression_reaction_line("smile", _SMILE_REACTION_LINES)
        fired = _speak_smile_reaction(line)
    else:
        try:
            from intelligence import reaction_awareness
            reaction_awareness.note_reaction(
                _person_db_id(person) if person else None,
                _first_name((person or {}).get("face_id"), "them"),
                "smile",
                trigger_text=str(watch.get("trigger_text") or ""),
            )
            # The awareness path never enqueues audio, so arm the cooldown here
            # (the canned path arms it inside _speak_smile_reaction) — otherwise
            # a held smile would re-mint the awareness every tick.
            global _last_smile_reaction_at
            _last_smile_reaction_at = now
            fired = True
        except Exception as exc:
            _log.debug("smile reaction awareness note failed: %s", exc)
    if fired:
        _log.info(
            "consciousness: smile reaction fired person=%s baseline=%s current=%s canned=%s",
            watch.get("person_key"),
            watch.get("baseline_expression"),
            _person_expression_label(person),
            bool(getattr(config, "SMILE_REACTION_CANNED_LINES_ENABLED", False)),
        )
        # "I made <name> smile" → rex.db. person_key is a STRING ("db:123"), so pull
        # the int id from the person dict instead. Tag it with the live topic so the
        # callback is specific ("I made Bret smile about his fantasy team").
        _laugh_topic = ""
        try:
            from intelligence import topic_thread as _tt
            _snap = _tt.snapshot() or {}
            _laugh_topic = str(_snap.get("label") or "")
        except Exception:
            _laugh_topic = ""
        episodic_hooks.made_laugh(
            _person_db_id(person) if person else None,
            _first_name((person or {}).get("face_id"), "them"),
            kind="smile",
            topic=_laugh_topic,
        )
        # Landing a laugh delights Rex → let it carry into his body mood/posture.
        try:
            from intelligence import body_mood
            body_mood.set_mood("giddy", source="made_laugh")
        except Exception:
            pass


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


def _speak_facial_expression_reaction(
    kind: str, text: str, *, on_spoke: Optional[Callable[[], None]] = None
) -> bool:
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
        on_spoke=on_spoke,
    )


_expr_vision_last_fresh_at = 0.0


def _expression_vision_context(person_db_id: Optional[int]) -> str:
    """A short GPT-vision read of the moment to ground an expression reaction — budgeted:
    the FREE local classifier is the trigger; the per-person mood cache is consulted first
    (no tokens); a fresh vision call is allowed at most once per
    EXPRESSION_REACTION_VISION_MIN_INTERVAL_SECS across all people. Returns "" when there
    is nothing affordable/useful."""
    global _expr_vision_last_fresh_at
    if person_db_id is None:
        return ""
    if not bool(getattr(config, "EXPRESSION_REACTION_VISION_ENABLED", True)):
        return ""

    def _notes(mood: Optional[dict]) -> str:
        if not isinstance(mood, dict):
            return ""
        notes = str(mood.get("notes") or "").strip()
        label = str(mood.get("mood") or "").strip()
        if notes and label:
            return f"{label} — {notes}"
        return notes or label

    # Free first: a recent cached read.
    text = _notes(get_cached_mood(person_db_id))
    if text:
        return text
    # One fresh read, globally rate-limited.
    now = time.monotonic()
    min_gap = float(getattr(config, "EXPRESSION_REACTION_VISION_MIN_INTERVAL_SECS", 120.0))
    if (now - _expr_vision_last_fresh_at) < min_gap:
        return ""
    _expr_vision_last_fresh_at = now
    return _notes(_get_or_detect_mood(person_db_id))


def _generate_contextual_expression_reaction(
    kind: str, person_id: Optional[int]
) -> str:
    """A conversation-aware facial-expression reaction via the main LLM, or "" to
    fall back to the authored bank. Lazy import to avoid a consciousness->llm import
    cycle; any failure is swallowed so the bank still fires."""
    if not bool(getattr(config, "FACIAL_EXPRESSION_REACTION_LLM_ENABLED", True)):
        return ""
    try:
        from intelligence import llm
        visual = _expression_vision_context(person_id)
        return llm.generate_expression_reaction(
            kind, person_id=person_id, visual_context=visual
        )
    except Exception as exc:
        _log.debug("contextual expression reaction failed: %s", exc)
        return ""


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
    if _expression_is_habitual_disposition(_person_db_id(person), kind):
        # The detected expression IS this person's resting face — don't read a visual
        # habit as a live emotional reaction (e.g. a habitually brow-furrowed person
        # getting "you're not exactly sold on this, are you?" at startup).
        _log.info(
            "consciousness: facial expression reaction suppressed (habitual disposition) "
            "person=%s kind=%s score=%.2f",
            person_key, kind, float(score),
        )
        return
    if _facial_expression_reaction_on_cooldown(person_key, kind, now):
        return

    # SMILE diverts to first-person awareness (owner 2026-08-05: "get rid of the
    # sometimes triggered canned text"). This is the SPONTANEOUS-smile system — no
    # Rex quip caused it — and it used to emit interjection candidates ("You're
    # smiling like you know exactly what you're doing") into the governor. Now the
    # noticing feeds reaction_awareness so his next generated line can acknowledge
    # it woven-in, or not at all. Same cooldowns arm (a held smile must not re-mint
    # every tick); surprise/brow_furrow keep the spoken path — those are prompted
    # reactions to a live moment, not the canned smile commentary the owner cut.
    if (
        kind == "smile"
        and bool(getattr(config, "REACTION_AWARENESS_ENABLED", True))
        and not bool(getattr(config, "SMILE_REACTION_CANNED_LINES_ENABLED", False))
    ):
        try:
            from intelligence import reaction_awareness
            reaction_awareness.note_reaction(
                _person_db_id(person),
                _first_name((person or {}).get("face_id"), "them"),
                "smile",
                spontaneous=True,
            )
            _last_facial_expression_reaction_at = now
            _facial_expression_reacted_at[(person_key, kind)] = now
            _log.info(
                "consciousness: spontaneous smile → reaction awareness person=%s score=%.2f",
                person_key, float(score),
            )
        except Exception as exc:
            _log.debug("spontaneous smile awareness note failed: %s", exc)
        return

    # Prefer a context-aware, conversation-grounded reaction (judges surprise vs.
    # whether Rex just said something provocative; never narrates the camera). Fall
    # back to the authored bank when the LLM path is disabled or returns nothing.
    line = _generate_contextual_expression_reaction(kind, _person_db_id(person))
    if not line:
        lines = _FACIAL_EXPRESSION_REACTION_LINES.get(kind) or ()
        line = _choose_expression_reaction_line(kind, lines)
    if not line:
        return
    def _on_spoke() -> None:
        # Cooldown arms only when the line ACTUALLY speaks — under ENFORCE a smile
        # reaction that loses the tick must NOT suppress itself.
        global _last_facial_expression_reaction_at
        _last_facial_expression_reaction_at = time.monotonic()
        _facial_expression_reacted_at[(person_key, kind)] = _last_facial_expression_reaction_at
        _log.info(
            "consciousness: facial expression reaction fired person=%s kind=%s score=%.2f",
            person_key,
            kind,
            float(score),
        )

    _speak_facial_expression_reaction(kind, line, on_spoke=_on_spoke)


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


def _apply_solo_switch_hysteresis(
    person_record: Optional[dict],
    box: tuple | list | None,
    frame_w: int,
    frame_h: int,
) -> Optional[dict]:
    """For a SINGLE visible face, resist switching the bound identity to a different
    known person until it has been seen for FACE_IDENTITY_SWITCH_CONFIRM_FRAMES
    consecutive ticks. Returns the identity to use THIS tick (the held one while a switch
    is still pending). Only holds against a recent, box-compatible prior identity, so a
    genuinely new person / new position still resolves promptly."""
    global _pending_solo_switch
    confirm = int(getattr(config, "FACE_IDENTITY_SWITCH_CONFIRM_FRAMES", 2) or 1)
    if confirm <= 1 or person_record is None or _last_solo_identity is None:
        _pending_solo_switch = None
        return person_record
    prev_id, prev_name, prev_ts = _last_solo_identity[0], _last_solo_identity[1], _last_solo_identity[2]
    prev_box = _last_solo_identity[3] if len(_last_solo_identity) >= 4 else None
    if (time.monotonic() - prev_ts) > _SOLO_IDENTITY_STICKY_SECS:
        _pending_solo_switch = None
        return person_record
    cand_id = person_record.get("id")
    if cand_id == prev_id:
        _pending_solo_switch = None
        return person_record
    if not _face_boxes_sticky_compatible(box, prev_box, frame_w=frame_w, frame_h=frame_h):
        _pending_solo_switch = None  # new position — treat as a real new face
        return person_record
    count = (_pending_solo_switch[1] + 1) if (_pending_solo_switch and _pending_solo_switch[0] == cand_id) else 1
    if count >= confirm:
        _pending_solo_switch = None
        return person_record  # confirmed across enough ticks — accept the switch
    _pending_solo_switch = (cand_id, count)
    _log.info(
        "consciousness: face identity switch held %s -> %s (%d/%d ticks)",
        prev_name, person_record.get("name"), count, confirm,
    )
    return {"id": prev_id, "name": prev_name}


# Last non-empty pose head anchors + capture time. The pose pipeline ticks at
# ~1 Hz and misses beats; on a miss tick the guard used to no-op, and that gap
# is exactly when wall phantoms leaked into world_state (field 2026-08-03: a
# busy workshop wall kept minting faces — the head snapped up/down chasing them
# and Rex waved at the wall). Anchors are in PIXEL space and the head can pan,
# so the cache lives only POSE_FACE_GUARD_ANCHOR_TTL_SECS and is judged with a
# wider radius (POSE_FACE_GUARD_CACHED_DIST_MULT) than live anchors.
_pose_anchor_cache: list = []
_pose_anchor_cache_at: float = 0.0


def _reject_faces_off_body(detected: list, frame_w: int, frame_h: int) -> list:
    """Drop detected faces that are far from EVERY pose head (phantom dlib faces).

    The MediaPipe pose heads (nose/eyes/ears) track real heads reliably even when dlib
    throws a spurious face elsewhere — so a face within POSE_FACE_GUARD_MAX_DIST_MULT
    head-widths of ANY tracked body is kept, and only a face far from every body is
    treated as a phantom. Multi-person aware: with POSE_MAX_PEOPLE>1 a second real person
    has their OWN pose head, so their face survives (the prior single-head version dropped
    it). No pose heads this tick → fall back to RECENTLY-cached anchors (wider radius);
    only with no live and no fresh cached anchors does face detection stand on its own."""
    global _pose_anchor_cache, _pose_anchor_cache_at
    if not detected or not bool(getattr(config, "POSE_FACE_GUARD_ENABLED", True)):
        return detected
    try:
        from vision import pose as pose_mod
        anchors = pose_mod.head_anchors_px(int(frame_w or 0), int(frame_h or 0))
    except Exception as exc:
        _log.debug("[pose_face_guard] head anchor lookup failed: %s", exc)
        anchors = []
    now = time.monotonic()
    cached = False
    if anchors:
        _pose_anchor_cache = list(anchors)
        _pose_anchor_cache_at = now
    else:
        ttl = float(getattr(config, "POSE_FACE_GUARD_ANCHOR_TTL_SECS", 2.5))
        if _pose_anchor_cache and (now - _pose_anchor_cache_at) <= ttl:
            anchors = _pose_anchor_cache
            cached = True
        else:
            return detected  # no live or fresh cached pose heads — can't guard

    mult = float(getattr(config, "POSE_FACE_GUARD_MAX_DIST_MULT", 1.5))
    if cached:
        # The head may have panned since capture — widen the accept radius so a
        # stale anchor can't drop the real face, while a wall phantom (far from
        # any body) still dies.
        mult = float(getattr(config, "POSE_FACE_GUARD_CACHED_DIST_MULT", 3.0))
    kept = []
    for face in detected:
        box = face.get("bounding_box") if isinstance(face, dict) else None
        if not (isinstance(box, (list, tuple)) and len(box) >= 4):
            kept.append(face)
            continue
        x, y, w, h = [float(v) for v in box[:4]]
        fx, fy = x + w / 2.0, y + h / 2.0
        near_any = any(
            ((fx - hx) ** 2 + (fy - hy) ** 2) ** 0.5 <= mult * float(head_w)
            for (hx, hy, head_w) in anchors
        )
        if near_any:
            kept.append(face)
        else:
            _log.info(
                "[pose_face_guard] dropped phantom face center=(%.0f,%.0f) — far from all "
                "%d%s pose head(s)", fx, fy, len(anchors),
                " cached" if cached else "",
            )
    return kept


def _step_person_recognition(frame) -> None:
    """
    Detect visible faces, resolve known identities via DB lookup, and update
    world_state.people with one slot per visible face.

    This function no longer depends on pose pre-populating people slots. If the
    pose pipeline is disabled or lagging, face recognition still works and can
    drive unknown-person onboarding prompts.
    """
    global _last_face_feedback_signature, _last_identity_prompt_at, _last_solo_identity
    global _pending_solo_switch
    global _last_face_seen_at
    try:
        from vision import face as face_mod

        if frame is None:
            _last_face_feedback_signature = None
            return

        detected = face_mod.detect_faces(frame)
        # Reject phantom faces (dlib false positives off the body) using the pose head as
        # source of truth. If all faces this tick were phantoms, `detected` becomes empty
        # and the hold/clear path below keeps the last good identity instead of jumping.
        _frame_h_px = int(getattr(frame, "shape", [0, 0, 0])[0] or 0)
        _frame_w_px = int(getattr(frame, "shape", [0, 0, 0])[1] or 0)
        detected = _reject_faces_off_body(detected, _frame_w_px, _frame_h_px)
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
                    # Visual active-speaker signal (vision/active_speaker.py).
                    # Written by a separate module under the same world_state lock;
                    # carried forward here so a slot resize doesn't drop it.
                    "is_speaking": base.get("is_speaking"),
                    "speaking_confidence": base.get("speaking_confidence"),
                    "speaking_updated_at": base.get("speaking_updated_at"),
                })
            people = resized
            changed = True

        recognized_names: list[str] = []
        unknown_count = 0
        unknown_scores: list = []  # det scores of counted unknowns (floor calibration)
        had_raw_unknown = False
        # An unknown face only counts as a real person once it has PERSISTED — compute
        # this tick's exposure from the running streak (+1 for this tick if it has one).
        expose_unknown = (_unknown_visible_streak + 1) >= _unknown_confirm_frames()
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
            if len(detected) == 1:
                # Damp known<->known flicker: hold the current identity until a different
                # person has been seen for FACE_IDENTITY_SWITCH_CONFIRM_FRAMES ticks.
                person_record = _apply_solo_switch_hysteresis(
                    person_record, det.get("bounding_box"), frame_width, frame_height
                )
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
                if target_slot is None and len(detected) == len(people):
                    # A positive re-ID must claim a slot even when every slot is already
                    # bound to a DIFFERENT known person — otherwise the overlay stays stuck
                    # on the stale identity (symptom B: turning the camera back to Bret
                    # never updated because the slot kept face_id='Bro'/'Broski').
                    target_slot = people[idx] if idx < len(people) else None
            else:
                # Detector-confidence floor: a face nothing identifies AND the detector
                # itself only half-believes is clutter, not a stranger — drop it before
                # it can feed the persistence streak or the identity prompt.
                if not _unknown_face_conf_ok(det):
                    continue
                had_raw_unknown = True
                # Persistence gate: a transient unknown (clutter, a shape on the wall, a
                # glance at a messy shelf) must NOT become a visible "person" — that armed
                # the badgering "who's the mystery guest?" agenda on phantom faces. Skip
                # exposing it until it has persisted FACE_UNKNOWN_CONFIRM_FRAMES ticks.
                if not expose_unknown:
                    continue
                unknown_count += 1
                _c = det.get("confidence")
                unknown_scores.append(round(float(_c), 2) if isinstance(_c, (int, float)) else None)

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
                    # HEIGHT fraction (face box height / frame height) — robust to wide
                    # 16:9 webcams where a close-up face is TALL, not wide, so the width
                    # fraction above under-reads closeness (proxemics logged distance=public
                    # for a face filling ~half the frame height). Used by the wave-back
                    # close gate: a face taller than ~a third of the frame is a desk-webcam
                    # close-up, where a "wave" is a near-camera artifact.
                    target_slot["face_box_height_fraction"] = (
                        (box[3] / frame_height) if frame_height > 0 else None
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

        # Commit this tick's unknown-persistence streak (resets when no unknown was seen).
        _update_unknown_streak(had_raw_unknown)
        if had_raw_unknown and not expose_unknown:
            _log.debug(
                "[face] unknown face held (persistence %d/%d) — not yet treated as a person",
                _unknown_visible_streak, _unknown_confirm_frames(),
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
                _log.info("consciousness: unknown %s detected (%d) det_scores=%s",
                          noun, unknown_count, unknown_scores)
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
                    "pose", "gesture", "engagement", "age_estimate", "pose_keypoints",
                    "face_mood", "face_expression", "facial_expression", "expression",
                    # Active-speaker fields (vision/active_speaker.py) — overlay a
                    # fresh speaker write so a concurrent identity re-bind on this
                    # slow-path tick doesn't drop it. NOTE: this overlay is
                    # positional; active_speaker keys its WRITES by person_db_id
                    # (stable across a slot resize) so a mis-aligned index here can
                    # only carry a stale value forward, never mis-attribute.
                    "is_speaking", "speaking_confidence", "speaking_updated_at",
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
            _pending_solo_switch = None
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
    global _last_identity_prompt_at, _identity_prompt_reply_until, _solo_unknown_since
    global _identity_prompt_in_flight_at

    if unknown_count <= 0 or known_unique:
        _solo_unknown_since = 0.0  # not a solo-unknown scene — reset the grace timer
        return
    if _pending_identity_prompt.is_set():
        return
    if _identity_prompt_in_flight.is_set():
        # The governor can REJECT the submitted candidate (e.g. the 5s
        # situation-suppression window right after ACTIVE->IDLE), in which case its
        # speak_fn/on_done never run and nothing clears this latch. Live-logged
        # 2026-07-06-19-20: one rejected ask left the latch set and Rex never spoke
        # to the unknown visitor again. A latch older than the stale window is dead.
        stale = float(getattr(config, "IDENTITY_PROMPT_INFLIGHT_STALE_SECS", 10.0) or 10.0)
        if (time.monotonic() - _identity_prompt_in_flight_at) < stale:
            return
        _log.info(
            "[identity_prompt] stale in-flight latch (>%.0fs, governor likely rejected) "
            "— clearing and retrying", stale,
        )
        _identity_prompt_in_flight.clear()

    # Grace: require the solo-unknown face to PERSIST before concluding it's truly a
    # stranger. A KNOWN face takes a tick or two to resolve (detect -> encode -> DB
    # match) and reads as "unknown" until then; without this, Rex asked "what's your
    # name?" one tick before recognizing a known person (the GUI already showed Bret).
    now_grace = time.monotonic()
    grace_secs = float(getattr(config, "IDENTITY_PROMPT_UNKNOWN_GRACE_SECS", 2.5) or 0.0)
    if _solo_unknown_since <= 0.0:
        _solo_unknown_since = now_grace
    if (now_grace - _solo_unknown_since) < grace_secs:
        return

    current_state = state_module.get_state()
    if current_state == State.ACTIVE and not bool(
        getattr(config, "IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE", True)
    ):
        return
    if current_state not in (State.IDLE, State.ACTIVE):
        return
    # salient=True: an unacknowledged stranger standing in front of Rex is
    # time-sensitive — without it, plain proactive speech is blocked for the whole
    # ACTIVE period (the first ~60s after boot), which is exactly when a visitor
    # walks up. Salient still yields to live speech, awaiting-a-reply, DJ, games,
    # and open flows.
    if not _can_proactive_speak(salient=True):
        return

    now = time.monotonic()
    if (now - _last_identity_prompt_at) < _IDENTITY_PROMPT_COOLDOWN_SECS:
        return

    _log.info(
        "consciousness: prompting unknown person for identity (state=%s)",
        getattr(current_state, "name", current_state),
    )
    _identity_prompt_in_flight.set()
    _identity_prompt_in_flight_at = now

    def _identity_prompt_spoke() -> None:
        # Arm the re-ask cooldown only when the line is actually committed to the
        # speech queue. speak_async returns True on governor SUBMISSION, so arming
        # on that return burned the 45s cooldown on candidates the governor then
        # rejected — and the visitor got silence.
        global _last_identity_prompt_at
        _last_identity_prompt_at = time.monotonic()

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
        force_salient=True,
        on_done=_identity_prompt_done,
        on_spoke=_identity_prompt_spoke,
    )
    if not queued:
        _identity_prompt_in_flight.clear()


def _step_body_social_analysis(frame) -> None:
    """
    Refresh crowd context from the latest people slots.

    Pose/gesture detection now runs in vision.pose's own background loop (started in
    main.py), so it is no longer pulled from this ~1 Hz tick — that keeps the GUI
    skeleton overlay and wave-back live. Face recognition owns identity and proxemic
    face boxes; pose decorates engagement/gesture/keypoints onto the same slots. Here
    we only combine the latest slots into a crowd mode for downstream conversation.
    """
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
        _face_recognized_chirp()
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
        from datetime import date, datetime, timedelta, timezone
        from memory import events as events_mod
        upcoming = events_mod.get_upcoming_events(person_db_id)
        if not upcoming:
            return None
        lookahead_days = getattr(config, "ANTICIPATION_LOOKAHEAD_DAYS", 30)
        cutoff = date.today() + timedelta(days=lookahead_days)
        cooldown_hours = float(getattr(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20) or 0.0)
        now_utc = datetime.now(timezone.utc)
        for ev in upcoming:
            ev_id = ev.get("id")
            if ev_id is None or (person_db_id, ev_id) in _anticipated_events:
                continue
            # Cross-session throttle: don't re-anticipate the same event on every launch
            # (the 'Juneteenth every launch' fix). Keys on anticipated_at — when REX last
            # spoke an anticipation — never on mentioned_at, which is set by the human's
            # own mention (field 2026-07-18: the river float mentioned at 1 AM was still
            # inside the 20h window at 9 PM, so Rex never once brought it up). A
            # never-anticipated event is never throttled.
            anticipated_at = ev.get("anticipated_at")
            if anticipated_at and cooldown_hours > 0:
                try:
                    m_dt = datetime.fromisoformat(str(anticipated_at))
                    if m_dt.tzinfo is None:
                        m_dt = m_dt.replace(tzinfo=timezone.utc)
                    if (now_utc - m_dt) < timedelta(hours=cooldown_hours):
                        continue
                except (ValueError, TypeError):
                    pass
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

    Only fires once per milestone: last_milestone_greeted records the highest
    milestone already announced, so Rex doesn't repeat "your 5th visit" every
    startup while visit_count sits at the same value.
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
        if incoming not in milestones:
            return None
        if incoming <= int(person.get("last_milestone_greeted") or 0):
            return None
        return incoming
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


def _greeting_profile(person_db_id: Optional[int]) -> tuple[str, bool]:
    """Return (tone_instruction, warm_default) for a greeting, based on the relationship.

    A greeting should sound like how that relationship would ACTUALLY greet you — warm
    and simple for a close friend or Rex's creator, friendlier-but-reserved for an
    acquaintance. warm_default=True means a plain "how are you?" hello is the right
    opener (no getting-to-know-you hook needed)."""
    person = None
    try:
        if person_db_id is not None:
            from memory import people as people_mod
            person = people_mod.get_person(int(person_db_id))
    except Exception:
        person = None
    name = str((person or {}).get("name") or "")
    tier = str((person or {}).get("friendship_tier") or "stranger").lower()
    try:
        is_creator = bool(name and person_specials.is_rex_creator(name))
    except Exception:
        is_creator = False
    if is_creator:
        return (
            "This is your MAKER — greet them with genuine warmth and familiarity, glad "
            "to see them, the way you'd greet the person you trust and care about most.",
            True,
        )
    if tier in {"best_friend", "close_friend"}:
        return ("This is a close friend — warm and familiar, genuinely glad to see them.", True)
    if tier == "friend":
        return ("This is a friend — warm and friendly, happy to see them.", True)
    if tier == "acquaintance":
        return ("You know them a little — friendly and easygoing, not too familiar yet.", False)
    return ("You barely know them — a polite, friendly hello.", False)


def _presence_relationship_tone(person_db_id: Optional[int]) -> str:
    """Relationship-edge directive for a RETURN / DEPARTURE line so the tone scales with
    the relationship — a sharper rib for a friend who needles Rex, a warmer one for a
    close friend, "" (the line stays plain) for a near-stranger or neutral relationship.
    Reuses the same tested logic the reply path uses (llm._relationship_tone_rule over the
    person's warmth/antagonism/tier). Arrivals already scale via _greeting_profile."""
    if not bool(getattr(config, "PRESENCE_RELATIONSHIP_TONE_ENABLED", True)):
        return ""
    if not isinstance(person_db_id, int):
        return ""
    try:
        from memory import people as people_mod
        from intelligence import llm as _llm
        person = people_mod.get_person(person_db_id)
        if not person:
            return ""
        return _llm._relationship_tone_rule(person, person.get("name") or "") or ""
    except Exception as exc:
        _log.debug("presence relationship tone failed: %s", exc)
        return ""


# Short, warm "hello" openers rotated for REPEAT greetings within the day / an ~8h
# window, so seeing Bret a second/third time isn't always "how are you, Bret?". All are
# question-style so they fit the "ends in a question mark" instruction. The default
# ("how are you") is reserved for a FIRST greeting; repeats rotate through the rest.
_GREETING_OPENERS = (
    "how are you",        # default — first greeting of the window
    "what's up",
    "what's new",
    "how's it going",
    "how've you been",
    "what's good",
    "how's your day going",
)

# First-greeting STYLE variety for an established regular (owner gripe 2026-07-06:
# "hey Bret, what's up?" fires too often at startup — fine sometimes, stale as the
# default). Question-phrase rotation alone kept the same SHAPE every time; this
# table mixes in statement hellos and time-of-day hellos, which drop the
# question-mark requirement. (opener_phrase, is_question). "time_of_day" renders
# "good morning/afternoon/evening" at build time.
_FIRST_GREETING_STYLES = (
    ("how are you", True),
    ("good to see you", False),
    ("what's up", True),
    ("time_of_day", False),
    ("what's new", True),
    ("hey, welcome back", False),
    ("how's your day going", True),
)


def _time_of_day_hello() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "good morning"
    if hour < 17:
        return "good afternoon"
    return "good evening"


def _first_greeting_style(visits: int) -> tuple[str, bool]:
    """(opener_phrase, is_question) for a regular's first greeting of the day,
    rotated on their persistent visit count so it varies across days."""
    phrase, is_question = _FIRST_GREETING_STYLES[visits % len(_FIRST_GREETING_STYLES)]
    if phrase == "time_of_day":
        phrase = _time_of_day_hello()
    return phrase, is_question


def _repeat_greeting_opener(greeting_ordinal: int) -> Optional[str]:
    """Pick a rotated short opener for a REPEAT greeting so a same-day / within-window
    return varies instead of always "how are you". ``greeting_ordinal`` is 1-based (1 =
    first greeting of the window). Rotation is keyed on that count, which persists in the
    DB (greetings_today), so even restarting the program within the window cycles the
    opener. Returns None for the first greeting (use the default warm "how are you")."""
    variants = _GREETING_OPENERS[1:]
    if greeting_ordinal <= 1 or not variants:
        return None
    return variants[(greeting_ordinal - 2) % len(variants)]


def _build_simple_greeting_prompt(
    first_name: str, tone: str, *, note: str = "", opener: Optional[str] = None,
    allow_familiarity: bool = False, require_question: bool = True,
) -> str:
    """A plain, warm, human greeting — the way a real friend says hello. No roast, no
    clever theme, no interest hook; just a friendly hello. `note` optionally sets the
    situation (seen earlier today, been a while) so it lands naturally. `opener` sets the
    hello STYLE (e.g. "what's up") for repeat-visit variety; defaults to "how are you".
    `allow_familiarity` drops ONLY the "it's you again" ban (for an established regular whose
    `note` invites that warmth) — every other ban (roast, clever bit, interest hook) stays.
    `require_question=False` allows a STATEMENT hello ("Hey Bret — good to see you.") for
    the non-question opener styles, so not every startup greeting is a question."""
    note_clause = (note.strip() + " ") if note else ""
    opener = (opener or "how are you").strip()
    again_ban = "" if allow_familiarity else "NO 'oh it's you again', "
    if require_question:
        shape = (
            f"(e.g. 'Hey {first_name}, {opener}?', or a close, natural variant of that). "
        )
        ending = "just a warm hello by name that ends in a question mark."
    else:
        shape = (
            f"(e.g. 'Hey {first_name} — {opener}.', or a close, natural variant of that). "
        )
        ending = (
            "just a warm hello by name. A question is OPTIONAL — a plain warm "
            "statement hello is completely fine this time."
        )
    return (
        f"You see {first_name}. {note_clause}{tone} Give a simple, natural, warm hello "
        f"that opens with a \"{opener}\"-style greeting — exactly how a real friend says "
        f"hello {shape}"
        f"Keep it to ONE short, genuine line. NO roast, {again_ban}NO clever "
        f"bit or Star Wars one-liner, NO 'what do you need / what are you up to / working "
        f"on / tinkering with', and NO interest callbacks — {ending}"
    )


def _build_long_absence_prompt(first_name: str, days: float, *, tone: str = "") -> str:
    days_int = int(round(days))
    if days_int >= 365:
        span = f"about {days_int // 365} year(s)"
    elif days_int >= 60:
        span = f"about {days_int // 30} months"
    else:
        span = f"{days_int} days"
    return _build_simple_greeting_prompt(
        first_name, tone,
        note=f"You haven't seen {first_name} in {span} — it's good to have them back.",
    )


def _build_recent_return_prompt(
    first_name: str, hours: float, *, tone: str = "", opener: Optional[str] = None,
) -> str:
    """First greeting of the day for someone last seen RECENTLY (<48h) — the hello
    should lightly acknowledge the quick return ('back already', 'saw you
    yesterday') instead of a generic 'how are you'. Same fix as the same-day
    builder: it used to delegate to the plain template, which dropped the note.
    `opener` kept for caller compatibility, unused."""
    if hours < 1.5:
        span = "just a little while ago"
        examples = f"'Back already? Missed me.', 'That was quick. Hey {first_name}.'"
    elif hours < 24:
        span = f"about {int(round(hours))} hours ago"
        examples = f"'Twice in one day — I'm flattered.', 'Back again? Good.'"
    else:
        span = "yesterday"
        examples = (
            f"'Hey {first_name} — two days running. I could get used to this.', "
            f"'Back again. Yesterday clearly went well.'"
        )
    tone_clause = f" {tone}" if tone else ""
    return (
        f"You see {first_name}; you last saw them {span}, so this is a QUICK return."
        f"{tone_clause} Greet them with ONE short, warm line that lightly acknowledges "
        f"seeing them again so soon — never a generic 'how are you / what's up' hello. "
        f"Glad, never annoyed; NO roast. Shape examples: {examples} A question is "
        f"optional."
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


def _greeting_recency(person_db_id: Optional[int]) -> tuple:
    """(bucket, age_secs) for how recently Rex greeted this person — see
    intelligence/greeting_cadence.py. (None, …) means the normal ladder should run.

    Deliberately DB-backed rather than in-memory: the point is to survive the restart
    that resets every other repeat-visit guard.
    """
    try:
        return greeting_cadence.recency(person_db_id)
    except Exception as exc:
        _log.debug("greeting recency lookup failed: %s", exc)
        return (None, None)


def _wellbeing_ask_clause(person_db_id: Optional[int]) -> str:
    """A clause appended to ANY greeting prompt when Rex already asked this person how
    they're doing recently — so the suppression holds no matter which ladder branch
    won, including the LLM-improvised ones that could smuggle the question back in."""
    try:
        return greeting_cadence.suppression_line(person_db_id)
    except Exception as exc:
        _log.debug("wellbeing ask clause lookup failed: %s", exc)
        return ""


def _build_same_day_return_prompt(
    first_name: str, prior_greetings_today: int, *, tone: str = "",
    opener: Optional[str] = None,
) -> str:
    """A warm 'oh, you're back' for a same-day repeat activation — NOT a roast.

    Owner gripe 2026-07-06: repeat visits got plain 'Hey Bret, what's up?' with no
    back-again acknowledgment. This used to delegate to the simple-greeting
    template, which buried the return context in an ignorable note, forced the
    'Hey {name}, {opener}?' shape via its example, and BANNED "it's you again"
    (allow_familiarity defaults False) — the acknowledgment was literally
    prohibited. The return IS the greeting now. `opener` is accepted for caller
    compatibility but unused — the back-again beat replaces the hello style."""
    if prior_greetings_today >= 2:
        situation = (
            f"You've already seen {first_name} a couple of times today and here they "
            f"are AGAIN — you're glad they keep coming back, not annoyed by it."
        )
        examples = (
            f"'Back again? I'll allow it.', '{first_name}! Round "
            f"{_ordinal(prior_greetings_today + 1)}.', 'You keep showing up. Good.'"
        )
    else:
        situation = f"You greeted {first_name} earlier today and here they are again."
        examples = (
            f"'Hey, you're back.', 'Oh — round two. Hey {first_name}.', "
            f"'Look who's back already.'"
        )
    tone_clause = f" {tone}" if tone else ""
    return (
        f"You see {first_name}. {situation}{tone_clause} Greet them with ONE short, "
        f"warm line that ACKNOWLEDGES the return — 'you're back' is the whole point, "
        f"never a generic 'how are you / what's up' hello. Glad, never annoyed; NO "
        f"roast, NO guilt-tripping about leaving. Shape examples: {examples} "
        f"A question is optional."
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
    # Spell out the relative day with LOCAL dates — the model has no reliable
    # "today" and guessed (field bug: a July-4 event opened as happening
    # "tonight" at 9:33pm on July 3).
    if event_date:
        try:
            from datetime import date as _d
            days = (_d.fromisoformat(str(event_date)[:10]) - _d.today()).days
            rel = (
                "TODAY" if days == 0 else "TOMORROW" if days == 1
                else f"in {days} days" if days > 1 else "already past"
            )
            when_clause = (
                f" coming up on {event_date} — that is {rel}; phrase any time "
                f"reference accordingly and never guess a different day"
            )
        except (ValueError, TypeError):
            pass
    notes_clause = f" Context they gave: {notes}." if notes else ""
    # A HEDGED plan ("might", "thinking about") must never be asserted as a
    # scheduled fact — "I might move the couch this weekend" opened the next
    # boot as "the couch move is today" (field 2026-08-01). Ask, don't declare.
    if bool(event.get("hedged")):
        return (
            f"You see '{first_name}', someone you know — {situation}. "
            f"You remember they said they MIGHT do '{event_name}'{when_clause} — "
            f"it was tentative, NOT a commitment.{notes_clause} "
            f"Open with a short in-character Rex line that asks whether it's "
            f"still the plan / whether they're actually going to do it. Do NOT "
            f"state that it is happening or treat it as scheduled. Warm but dry. "
            f"Address {first_name} by name. One line only."
        )
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

# Concrete-milestone cues — a description with one of these reads as a real event
# worth opening with (vs. a borderline "things are going well"), so it scores high
# on the concreteness axis of the cold-open ranker.
_CONCRETE_MILESTONE_RE = re.compile(
    r"\b(won|win|wins|winning|champion|championship|promot|graduat|hire[sd]?|"
    r"got\s+(?:the|a|an)\s+(?:job|gig|role|offer)|new\s+(?:job|gig|role|house|home|car)|"
    r"award|prize|medal|trophy|finished|completed|launch|publish|released|"
    r"married|engaged|wedding|baby|newborn|bought|passed|exam|degree|diploma|"
    r"record|milestone|signed|closed\s+the\s+deal|first\s+place|personal\s+best)\b",
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


def _celebration_lead_score(event: dict) -> float:
    """Rank a gate-passing celebration for leading a cold open: did-they-invite-it
    (they told Rex themselves) dominates, then recency, then how concrete/specific
    the milestone is. Higher = more worth opening with. Tunable via the
    PRESENCE_CELEBRATION_W_* weights."""
    ev = event or {}
    halflife = float(getattr(config, "PRESENCE_CELEBRATION_RECENCY_HALFLIFE_DAYS", 14.0))
    recency = 1.0 / (1.0 + max(0.0, _event_age_days(ev.get("mentioned_at"))) / max(1.0, halflife))
    invited = 1.0 if ev.get("person_invited_topic") else 0.0
    desc = str(ev.get("description") or "")
    # A milestone keyword is the strong concreteness signal; without one, length is
    # only a weak proxy and must NEVER outscore a real milestone (a wordy "had a
    # decent day" should not beat "won the championship").
    if _CONCRETE_MILESTONE_RE.search(desc):
        concreteness = 1.0
    else:
        concreteness = min(0.6, len(re.findall(r"[A-Za-z']+", desc)) / 12.0)
    return (
        float(getattr(config, "PRESENCE_CELEBRATION_W_INVITED", 1.0)) * invited
        + float(getattr(config, "PRESENCE_CELEBRATION_W_RECENCY", 0.6)) * recency
        + float(getattr(config, "PRESENCE_CELEBRATION_W_CONCRETE", 0.3)) * concreteness
    )


def _pick_due_celebration_checkin(person_db_id: Optional[int]) -> Optional[dict]:
    """Return the celebration worth leading a greeting with — the BEST of the
    gate-passing candidates (ranked by recency x concreteness x invited), not just
    the first/most-recent that happens to pass."""
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import emotional_events as emo_events
        # Honor existing "don't bring up X" boundaries (even ones set in a prior
        # session) before choosing what to lead with — mutes matching events.
        try:
            from memory import boundaries as _boundaries
            _boundaries.reconcile_event_mutes(person_db_id)
        except Exception as exc:
            _log.debug("boundary→event reconcile skipped: %s", exc)
        due = emo_events.get_startup_celebrations(
            person_db_id,
            process_started_iso=_process_started_iso,
            limit=5,
        )
        worthy = [event for event in due if _celebration_worth_leading_with(event)]
        if not worthy:
            return None
        if bool(getattr(config, "PRESENCE_CELEBRATION_RANK_ENABLED", True)):
            return max(worthy, key=_celebration_lead_score)
        return worthy[0]
    except Exception as exc:
        _log.debug("celebration check-in lookup error: %s", exc)
        return None


def _cold_open_lead_score(cand: dict) -> float:
    """Unified cold-open lead-score, the SAME shape as _celebration_lead_score
    (invited dominant → recency → concreteness) but for an interest/fact callback
    candidate {invited, recency_iso, text, base}. Lets the cold-open picker rank a
    remembered-interest opener against a fact the same way it ranks a celebration —
    so Rex leads with the single best thing to bring up, not just a celebration."""
    ev = cand or {}
    halflife = float(getattr(config, "PRESENCE_CELEBRATION_RECENCY_HALFLIFE_DAYS", 14.0))
    recency = 1.0 / (1.0 + max(0.0, _event_age_days(ev.get("recency_iso"))) / max(1.0, halflife))
    invited = 1.0 if ev.get("invited") else 0.0
    text = str(ev.get("text") or "")
    if _CONCRETE_MILESTONE_RE.search(text):
        concreteness = 1.0
    else:
        concreteness = min(0.6, len(re.findall(r"[A-Za-z']+", text)) / 12.0)
    return (
        float(getattr(config, "PRESENCE_CELEBRATION_W_INVITED", 1.0)) * invited
        + float(getattr(config, "PRESENCE_CELEBRATION_W_RECENCY", 0.6)) * recency
        + float(getattr(config, "PRESENCE_CELEBRATION_W_CONCRETE", 0.3)) * concreteness
        + float(ev.get("base", 0.0))
    )


# Fact categories that make warm "how's X going?" cold-open material — genuine
# ACTIVITIES the person DOES, not static preferences ("favorite ice cream") or
# identity/sensitive facts (those read awkwardly as "how's the ice cream going?").
# Interests proper come from the interests table, not here.
_COLD_OPEN_FACT_CATEGORIES = {"hobby", "project", "activity"}

# An interest is mis-stored as a "hobby" all the time ("mint chocolate chip ice cream",
# "my clothes", "you now"). Those read absurd as a greeting opener ("what's the latest
# scoop on your ice cream adventures?"), so they're excluded from cold-open LEADS. A
# consumable/static favorite is a preference, not an activity to ask "how's it going?".
_COLD_OPEN_INTEREST_EXCLUDE_RE = re.compile(
    r"\bice\s?cream\b|\bchocolate\b|\bcandy\b|\bcookies?\b|\bsnacks?\b|\bpizza\b|"
    r"\bburgers?\b|\bcoffee\b|\bsoda\b|\bdessert\b|"
    r"\bfavou?rite\s+(?:food|colou?r|snack|drink|flavou?r)\b"
    r"|\b(?:my|your)\s+(?:clothes|bed|outfit|stuff|hair)\b"
    r"|\b(?:hang(?:ing)?\s+out|in\s+(?:my|the)\s+bed|be\s+in\s+there|you\s+now)\b",
    re.IGNORECASE,
)


def _cold_open_interest_worthy(name: str) -> bool:
    """True when an interest is substantive enough to LEAD a greeting (an activity/
    fandom/skill), not a static favorite or junk fragment."""
    cleaned = (name or "").strip()
    if len(re.findall(r"[A-Za-z']+", cleaned)) < 1:
        return False
    return _COLD_OPEN_INTEREST_EXCLUDE_RE.search(cleaned) is None


def _cold_open_callback_candidates(person_db_id: int) -> list[dict]:
    """Gather interest-hook + warm-fact candidates worth OPENING a greeting with,
    normalized for _cold_open_lead_score. Interests are things the person told Rex
    (invited=True) and get a small base bump over inferred facts."""
    cands: list[dict] = []
    try:
        from memory import interests as interests_mem
        for hook in (interests_mem.get_interest_hooks(person_db_id) or [])[:8]:
            name = str(hook.get("name") or "").strip()
            if not name or not _cold_open_interest_worthy(name):
                continue
            cands.append({
                "kind": "interest",
                "topic": name,
                "text": name,
                "invited": True,
                "recency_iso": hook.get("last_mentioned_at") or hook.get("first_mentioned_at"),
                "base": 0.20,
            })
    except Exception as exc:
        _log.debug("cold-open interest candidates error: %s", exc)
    try:
        from memory import facts as facts_mem
        for fact in (facts_mem.get_prompt_worthy_facts(person_db_id, limit=8) or []):
            category = str(fact.get("category") or "").strip().lower()
            value = str(fact.get("value") or "").strip()
            if category not in _COLD_OPEN_FACT_CATEGORIES or not value:
                continue
            if str(fact.get("freshness_label")) == "stale":
                continue
            cands.append({
                "kind": "fact",
                "topic": value,
                "text": value,
                "invited": _normalize_source_is_volunteered(fact.get("source")),
                "recency_iso": fact.get("last_mentioned_at") or fact.get("created_at"),
                "base": 0.0,
            })
    except Exception as exc:
        _log.debug("cold-open fact candidates error: %s", exc)
    return cands


def _normalize_source_is_volunteered(source) -> bool:
    return str(source or "").strip().lower() in {"explicit", "corrected", "volunteered"}


def _pick_cold_open_callback(person_db_id: Optional[int]) -> Optional[dict]:
    """The single remembered interest/fact worth LEADING a cold open with — the
    best of the gate-passing candidates by _cold_open_lead_score (invited × recency ×
    concreteness). Extends the celebration ranker across facts/interests. Returns the
    candidate dict (with 'topic'), or None when there's nothing worth opening with."""
    if not isinstance(person_db_id, int):
        return None
    if not bool(getattr(config, "COLD_OPEN_INTEREST_RANK_ENABLED", True)):
        return None
    try:
        cands = _cold_open_callback_candidates(person_db_id)
        if not cands:
            return None
        return max(cands, key=_cold_open_lead_score)
    except Exception as exc:
        _log.debug("cold-open callback pick error: %s", exc)
        return None


def _build_cold_open_callback_prompt(
    first_name: str, candidate: dict, context_sentence: str,
) -> str:
    topic = str((candidate or {}).get("topic") or "").strip()
    return (
        f"{context_sentence} You remember {first_name} is into '{topic}'. Greet them by "
        f"name and lead with genuine curiosity about it — ask how '{topic}' is going or "
        f"what they've been up to with it lately, in one or two short in-character Rex "
        f"sentences. The last sentence must end in a question mark. Don't invent details "
        f"you don't have; just open the door."
    )


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

def _crowd_change_settled(curr_label: str) -> Optional[str]:
    """Debounce crowd-count flicker. Returns the prior STABLE label when ``curr_label``
    is a settled change (the same new label has persisted past CROWD_CHANGE_SETTLE_SECS),
    else None. Silently adopts the first observed label as the baseline so startup
    doesn't fire a spurious "crowd changed" line."""
    global _crowd_change_reacted_label, _crowd_change_pending_label
    global _crowd_change_pending_since
    settle = float(getattr(config, "CROWD_CHANGE_SETTLE_SECS", 2.5))
    now = time.monotonic()
    baseline = _crowd_change_reacted_label
    if not baseline:
        _crowd_change_reacted_label = curr_label
        _crowd_change_pending_label = ""
        return None
    if curr_label == baseline:
        _crowd_change_pending_label = ""  # back to stable — cancel any pending change
        return None
    if curr_label != _crowd_change_pending_label:
        _crowd_change_pending_label = curr_label
        _crowd_change_pending_since = now
        return None
    if (now - _crowd_change_pending_since) >= settle:
        _crowd_change_reacted_label = curr_label
        _crowd_change_pending_label = ""
        return baseline
    return None


def _step_proactive_reactions(snapshot: dict, profile: SituationProfile) -> None:
    """
    Compare current WorldState to _last_snapshot. For each notable change,
    generate and speak a short in-character reaction. Never fires in QUIET/SHUTDOWN.
    """
    global _acknowledged_dates, _acknowledged_tod, _last_weather_reaction_at, _last_startle_sound_reaction_at

    if _last_snapshot:
        _stage_animal_arrivals(snapshot)
        # A newly-arrived (or returning) animal is a salient, time-sensitive event
        # and deserves a reaction even mid-conversation — so attempt it BEFORE the
        # general proactive-suppression gates below, which would otherwise starve it
        # the way they did when a dog was held up during an active conversation. The
        # fire path is marked salient (force_salient) so it can interrupt ACTIVE /
        # skip the pacing cooldown, but it still yields to live user speech and to a
        # pending startup greeting or identity prompt, and the governor still
        # arbitrates its priority (85).
        if (
            not _startup_known_greeting_pending(snapshot)
            and not is_identity_prompt_waiting_for_reply()
            and _fire_pending_animal_arrival_reaction()
        ):
            return

    if profile.suppress_proactive or profile.rapid_exchange:
        return
    if not _last_snapshot or not _can_proactive_speak():
        return
    if _startup_known_greeting_pending(snapshot):
        return
    if is_identity_prompt_waiting_for_reply():
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

        # Crowd size label changed significantly — debounced so a one-frame camera
        # flicker (pair->alone->pair) can't fire a "now it's just us" line that
        # contradicts the greeting Rex just gave the group.
        curr_label = snapshot.get("crowd", {}).get("count_label")
        if curr_label:
            settled_prev = _crowd_change_settled(curr_label)
            if settled_prev:
                _add_trigger(
                    f"The crowd around you just shifted from '{settled_prev}' to "
                    f"'{curr_label}'. One short in-character observation about this change.",
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
            # Stash the dedupe key; only consume it if THIS trigger is the one chosen
            # below (else a co-occurring weather/tod change would swallow it unspoken).
            _add_trigger(
                f"Today is {notable_date}. Make one spontaneous in-character remark about it "
                "as if you just noticed the date. Deliver it Rex-style.",
                "excited",
                label=f"notable date: {notable_date}",
                metadata={"ack_date": notable_date},
            )

        # Part of day rolled over (morning → afternoon → evening → night). The hour
        # bucket is computed every tick but was only ever passive prompt context, so
        # Rex never NOTICED the day turning over. Fire once per transition per session,
        # mirroring the weather/notable-date change blocks. The line is LLM-generated
        # (via _generate_and_speak below) so it varies run to run.
        if bool(getattr(config, "TIME_OF_DAY_REACTIONS_ENABLED", True)):
            curr_tod = (snapshot.get("time", {}) or {}).get("time_of_day")
            prev_tod = (_last_snapshot.get("time", {}) or {}).get("time_of_day")
            if (
                curr_tod
                and prev_tod
                and curr_tod != prev_tod
                and curr_tod not in _acknowledged_tod
            ):
                _add_trigger(
                    f"The time of day just rolled over to {str(curr_tod).replace('_', ' ')}. "
                    "Make one short, spontaneous in-character remark as if you just "
                    "noticed the hour/light shifting. Don't recite the literal clock time.",
                    "curious",
                    label=f"time of day: {curr_tod}",
                    metadata={"ack_tod": curr_tod},
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
                            "ack_weather_signature": signature,
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
            # Consume dedupe state ONLY for the trigger actually chosen this tick — the
            # un-chosen co-occurring triggers stay un-acknowledged so they can fire next
            # tick instead of being permanently swallowed.
            if metadata.get("ack_date"):
                _acknowledged_dates.add(metadata["ack_date"])
            if metadata.get("ack_tod"):
                _acknowledged_tod.add(metadata["ack_tod"])
            if metadata.get("ack_weather_signature"):
                _acknowledged_weather_signatures.add(metadata["ack_weather_signature"])
                _last_weather_reaction_at = time.monotonic()
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
    # Lazy import avoids a top-level cycle (idle_behaviors imports consciousness back).
    from intelligence import idle_behaviors

    if state_module.get_state() != State.IDLE:
        return
    if is_waiting_for_response():
        return
    now = time.monotonic()
    if _within_startup_group_window(now) and not _greeted_this_session:
        return
    if _startup_known_greeting_pending(snapshot):
        return
    if bool(getattr(config, "BOREDOM_ENABLED", True)) and _room_looks_empty(snapshot):
        # The four-phase empty-room arc owns all speech when Rex is genuinely
        # alone. Random legacy micro-behaviors could otherwise jump ahead to
        # "I'm bored" during phase 1 or keep joking after the left-on phase began.
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
    # Novelty drive (curiosity Phase 2): a stale room tilts the idle mix toward
    # the LOOKING behaviors — Rex goes hunting for something new instead of
    # monologuing. Multiplier only touches scan/observation/vision entries.
    try:
        from awareness import novelty_drive
        if novelty_drive.is_stale():
            boost = float(getattr(config, "NOVELTY_STALE_LOOK_BOOST", 3.0))
            weights = [
                w * boost if c in ("ambient_scan", "ambient_observation",
                                   "live_vision_comment") else w
                for c, w in zip(choices, weights)
            ]
    except Exception:
        pass
    # Episodic recall (Phase 2) is opt-in: only offer the "memory musing" behavior
    # when enabled, so the idle mix is unchanged while the feature is off.
    if getattr(config, "EPISODIC_RECALL_ENABLED", False) and "memory_musing" not in choices:
        choices = choices + ["memory_musing"]
        weights = weights + [2]
    behavior = random.choices(choices, weights=weights, k=1)[0]
    _log.debug("consciousness: idle micro-behavior → %s", behavior)

    if behavior == "empty_room_joke":
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            idle_behaviors.do_empty_room_joke(snapshot)
    elif behavior == "small_talk_question":
        if not profile.suppress_proactive:
            _do_small_talk_question(snapshot)
    elif behavior == "ambient_scan":
        idle_behaviors.do_ambient_scan()
    elif behavior == "private_thought":
        # Private thoughts are system monologues — suppressed by both proactive and
        # system-comment gates so Rex doesn't mutter about himself mid-conversation.
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            idle_behaviors.do_private_thought()
    elif behavior == "aspiration":
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            idle_behaviors.do_aspiration()
    elif behavior == "memory_musing":
        # A system monologue like private_thought — suppressed by both gates so Rex
        # doesn't reminisce out loud mid-conversation.
        if not profile.suppress_proactive and not profile.suppress_system_comments:
            idle_behaviors.do_memory_musing()
    elif behavior == "idle_clip":
        if not profile.suppress_proactive:
            idle_behaviors.do_idle_clip()
    elif behavior == "ambient_observation":
        if not profile.suppress_proactive:
            idle_behaviors.do_ambient_observation(snapshot)
    elif behavior == "appearance_riff":
        if not profile.suppress_proactive:
            idle_behaviors.do_appearance_riff(snapshot)
    elif behavior == "people_roast":
        if not profile.suppress_proactive:
            idle_behaviors.do_people_roast(snapshot)
    elif behavior == "live_vision_comment":
        if not profile.suppress_proactive:
            idle_behaviors.do_live_vision_comment(snapshot)
    elif behavior == "bored_env_snark":
        if not profile.suppress_proactive:
            idle_behaviors.do_bored_environment_snark(snapshot)


# ── Boredom escalation: grumble when left alone, then doze off to SLEEP ──────────
_boredom_started_at: float = 0.0        # monotonic time boredom began (0.0 = not bored)
_last_boredom_comment_at: float = 0.0
_last_empty_room_observation_at: float = 0.0
_boredom_sleeping: bool = False         # True while the doze-off sleep flow is in flight
_boredom_loop_started_at: float = 0.0   # anchor so a never-engaged droid still gets bored


def _human_idle_secs(now: float) -> float:
    """Seconds since a HUMAN last engaged Rex. Unlike _conversation_idle_secs this does
    NOT count Rex's own proactive chatter, so his bored grumbling can't reset the
    boredom→sleep clock. Anchored to the loop start so a droid nobody has ever spoken
    to still eventually gets bored."""
    last_human = max(_engaged_last_touch_at, _recent_engaged_touch_at)
    anchor = last_human if last_human > 0.0 else _boredom_loop_started_at
    if anchor <= 0.0:
        return 0.0
    return max(0.0, now - anchor)


def _speak_boredom_line(bored_for: float) -> None:
    """Speak phase 2 (bored) or phase 3 (left activated) with no API spend."""
    sleep_after = float(getattr(config, "BOREDOM_SLEEP_AFTER_SECS", 600.0))
    left_on_fraction = float(getattr(config, "BOREDOM_LEFT_ON_PHASE_FRACTION", 0.60))
    left_on = sleep_after > 0 and bored_for >= sleep_after * left_on_fraction
    early_lines = list(getattr(config, "BOREDOM_LINES_EARLY", []) or [])
    late_lines = list(getattr(config, "BOREDOM_LINES_LATE", []) or [])
    left_on_lines = list(getattr(config, "BOREDOM_LINES_LEFT_ON", []) or [])
    # Mix the two boredom banks during phase 2, then switch cleanly to the
    # "somebody left me powered on" premise for phase 3.
    bored_lines = early_lines + late_lines
    pool = (left_on_lines or bored_lines) if left_on else (bored_lines or left_on_lines)
    if not pool:
        pool = ["...is anyone even here?"]
    _speak_async(
        random.choice(pool),
        emotion=("annoyed" if left_on else "neutral"),
        # Dedicated purpose (NOT idle_monologue): the lean brain suppresses the
        # silence-fill purposes because its impulse replaces them — but the lean
        # impulse never fires in an EMPTY room, so riding idle_monologue silently
        # killed the entire boredom arc when lean went live (owner noticed
        # 2026-07-07: "R3X is supposed to act bored when everyone leaves").
        # "boredom" is exempt from lean suppression AND from the presence-cadence
        # clamp: it self-paces (55-95s) and the arc terminates in SLEEP anyway.
        purpose="boredom",
        label="boredom grumble",
    )


def _trigger_boredom_sleep() -> None:
    """Doze off via interaction's sleep flow (sleepy line → SLEEP state → sleep pose),
    on a daemon thread so the spoken line doesn't block the consciousness loop."""
    _log.info("[boredom] bored too long with no interaction — dozing off into SLEEP.")

    def _task() -> None:
        global _boredom_sleeping
        try:
            from intelligence import interaction  # lazy: interaction imports consciousness
            lines = list(getattr(config, "BOREDOM_SLEEP_RESIGNATION_LINES", []) or [])
            resignation = random.choice(lines) if lines else None
            interaction._enter_sleep_mode(transition_line=resignation)
        except Exception as exc:
            _log.warning("[boredom] sleep transition failed: %s", exc)
            _boredom_sleeping = False  # recover so boredom can re-arm

    threading.Thread(target=_task, daemon=True, name="boredom-sleep").start()


def _step_boredom_escalation(snapshot: dict, profile: "SituationProfile") -> None:
    """Run the four-phase empty-room arc: observe → bored → left-on → sleep."""
    global _boredom_started_at, _last_boredom_comment_at
    global _last_empty_room_observation_at, _boredom_sleeping

    if not bool(getattr(config, "BOREDOM_ENABLED", True)):
        return

    # A doze-off is in flight (sleepy line still playing). Hold until SLEEP lands —
    # don't grumble over the going-to-sleep line — then disarm.
    if _boredom_sleeping:
        if state_module.get_state() != State.IDLE:
            _boredom_sleeping = False
        return

    # Only get bored while idle and not mid-exchange. Any other state (active convo,
    # already asleep, shutting down) clears the boredom clock.
    if state_module.get_state() != State.IDLE or is_waiting_for_response():
        _boredom_started_at = 0.0
        _last_empty_room_observation_at = 0.0
        return

    # The boredom arc is the EMPTY-ROOM show (owner design: "act bored when
    # everyone leaves... then sleep"). Someone visibly present resets it — a
    # present-but-quiet person is the lean impulse / re-engagement's territory,
    # and grumbling "I'm bored" AT them reads as needy.
    if not _room_looks_empty(snapshot):
        _boredom_started_at = 0.0
        _last_empty_room_observation_at = 0.0
        return

    now = time.monotonic()
    human_idle = _human_idle_secs(now)
    observation_onset = float(getattr(config, "EMPTY_ROOM_OBSERVATION_ONSET_SECS", 30.0))
    boredom_onset = float(getattr(config, "BOREDOM_ONSET_SECS", 150.0))
    if human_idle < observation_onset:
        _boredom_started_at = 0.0   # engaged recently / not alone long enough
        _last_empty_room_observation_at = 0.0
        return

    # Phase 1 — he is alone long enough to notice the room, but not bored yet.
    # Deliberately paced by the same interval as later comments so he looks alive
    # without narrating every camera scan.
    if human_idle < boredom_onset:
        if profile.suppress_proactive or profile.suppress_system_comments:
            return
        interval = random.uniform(
            float(getattr(config, "BOREDOM_COMMENT_INTERVAL_SECS_MIN", 55.0)),
            float(getattr(config, "BOREDOM_COMMENT_INTERVAL_SECS_MAX", 95.0)),
        )
        if (now - _last_empty_room_observation_at) < interval or not _can_proactive_speak():
            return
        _last_empty_room_observation_at = now
        from intelligence import idle_behaviors
        idle_behaviors.do_empty_room_observation(snapshot)
        return

    # Phase 2 begins: boredom has set in. Phase 3 is selected inside
    # _speak_boredom_line once enough of the boredom-to-sleep window has elapsed.
    if _boredom_started_at <= 0.0:
        _boredom_started_at = now
        _last_boredom_comment_at = 0.0
        _log.info("[boredom] no interaction for ~%.0fs — Rex is bored.", human_idle)

    bored_for = now - _boredom_started_at

    # Doze off after enough boredom.
    if bored_for >= float(getattr(config, "BOREDOM_SLEEP_AFTER_SECS", 600.0)):
        _boredom_started_at = 0.0
        _last_boredom_comment_at = 0.0
        _boredom_sleeping = True
        _trigger_boredom_sleep()
        return

    # Periodic bored grumbling (yield to anything suppressing proactive speech).
    if profile.suppress_proactive or profile.suppress_system_comments:
        return
    interval = random.uniform(
        float(getattr(config, "BOREDOM_COMMENT_INTERVAL_SECS_MIN", 55.0)),
        float(getattr(config, "BOREDOM_COMMENT_INTERVAL_SECS_MAX", 95.0)),
    )
    if (now - _last_boredom_comment_at) < interval:
        return
    if not _can_proactive_speak():
        return
    _last_boredom_comment_at = now
    _speak_boredom_line(bored_for)


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
    # Never declare the room empty while the OpenAI presence sweep is still verifying.
    if _startup_presence_fallback_active:
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

        def _on_spoke() -> None:
            # Latch the once-per-session flag only when the joke actually speaks —
            # under ENFORCE a losing candidate must not permanently suppress it.
            global _startup_empty_room_fired
            _startup_empty_room_fired = True

        _speak_async(
            line,
            emotion="curious",
            purpose="startup_empty_room",
            label="startup empty-room joke",
            on_spoke=_on_spoke,
        )
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
                "bored_env_snark",
            ],
            # Weighted toward room/environment commentary (ambient_observation,
            # live_vision_comment, bored_env_snark) so an idle tick is more likely
            # to land on "comment on the room" than self-monologue.
            [6, 1, 1, 1, 1, 3, 2, 4],
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
                "bored_env_snark",
            ],
            # Room/environment commentary (ambient_observation 1->3,
            # live_vision_comment 1->2, bored_env_snark 2->3) gets more share so
            # Rex riffs on the room, not just on the person, when idle.
            [4, 3, 2, 3, 2, 1, 1, 1, 1, 3],
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
            "bored_env_snark",
        ],
        # More room/environment commentary share (ambient_observation 1->3,
        # live_vision_comment 1->2) here too.
        [2, 3, 1, 1, 1, 1, 3, 2, 3],
    )


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
                        f"'{ev_name}'. Specifically ask whether it ended up "
                        f"happening and how it went — don't assert that it did."
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
    # `token` is the legacy conversation_agenda claim (LEGACY) or None (ENFORCE — the
    # governor arbitrates; None is claim-safe). The idle micro-behavior caller already
    # rate-limits how often small talk is attempted, so ENFORCE needs no extra cooldown.
    token = None
    enforcing = _governor_enforcing()
    if not enforcing:
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

            try:
                from intelligence.lean_brain import _fresh_angles_clause
                angles = _fresh_angles_clause()
            except Exception:
                angles = ""
            if target_name:
                prompt = (
                    f"It's quiet and you're idly looking at '{target_name}', someone you know. "
                    f"They haven't said anything in a while.{plan_clause}{mood_clause} "
                    f"Open small talk by asking them one short, in-character Rex question. "
                    f"Lead with genuine curiosity about who they are — ask how they're doing, "
                    f"about a hobby or interest of theirs, what "
                    f"they've been into or thinking about lately, or what they're passionate "
                    f"about. You are a DJ, so your reflex is to ask about music — RESIST it; "
                    f"music/song questions are your most overused opener, do NOT ask one."
                    f"{angles} "
                    f"If a cue above gives you something specific (a plan, their mood, a "
                    f"known interest), you may ask about that instead — but don't default to "
                    f"interrogating them about their schedule. Pick something they have NOT "
                    f"already covered in this conversation — a fresh subject or a new angle, "
                    f"never a repeat of what they just told you. Warm but dry. Don't lecture, "
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
            if token is not None:
                _release_proactive_purpose(token)

    if enforcing:
        # ENFORCE: submit the candidate carrying the deferred generate+ask; the
        # governor runs it ONLY if small talk wins this tick.
        _observe_governor_candidate(
            purpose=purpose,
            label="small-talk question",
            prompt=(
                "Small-talk candidate: choose a known visible person if available, "
                "optionally use mood or plan context, then ask one short question."
            ),
            emotion="curious",
            target_person_id=target_db_id,
            requires_llm=True,
            speak_fn=lambda: threading.Thread(
                target=_task, daemon=True, name="small-talk-question"
            ).start(),
        )
        return

    threading.Thread(target=_task, daemon=True, name="small-talk-question").start()


def _voice_pov_as_micro_behavior(label: str, prompt: str, *, emotion: str) -> bool:
    """When Rex has an active preoccupation (rex_pov) and REX_POV_FEEDS_MICRO_BEHAVIORS
    is on, VOICE it through the reply LLM (which already injects the POV via §6c) as the
    idle micro-behavior — so his mutterings are about the thing he's actually chewing
    on, not a random canned line. `_generate_and_speak` does its own claim + governor
    routing. Returns True if it handled the behavior (caller skips the canned fallback)."""
    if not bool(getattr(config, "REX_POV_FEEDS_MICRO_BEHAVIORS", True)):
        return False
    try:
        from intelligence import rex_pov
        if not rex_pov.active_pov_text():
            return False
    except Exception:
        return False
    _generate_and_speak(prompt, emotion=emotion, purpose="idle_monologue", label=label)
    return True


# Anti-repeat for aspirations — never play the same line back-to-back.


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


def _visual_curiosity_objects_line() -> str:
    """A compact 'Confirmed objects in view' line from the LOCAL object detector
    (world_state.objects), or '' when disabled / nothing confident enough. These are
    detector-verified, so the curiosity prompt may name them by label without risking
    an invented prop — the substrate for object-grounded curiosity."""
    if not bool(getattr(config, "VISUAL_CURIOSITY_USE_OBJECTS", True)):
        return ""
    try:
        objects = world_state.get("objects") or []
    except Exception:
        return ""
    min_conf = float(getattr(config, "VISUAL_CURIOSITY_OBJECTS_MIN_CONFIDENCE", 0.40))
    keep = [
        o for o in objects
        if isinstance(o, dict) and o.get("label")
        and float(o.get("confidence") or 0.0) >= min_conf
    ]
    if not keep:
        return ""
    keep.sort(key=lambda o: float(o.get("confidence") or 0.0), reverse=True)
    cap = int(getattr(config, "VISUAL_CURIOSITY_OBJECTS_MAX", 6))
    # Novelty (room model): float objects that are NEW to the room (low/zero recorded
    # sightings) to the front and tell the LLM to prefer them, so curiosity asks about
    # what changed rather than the fixtures Rex sees every day. Degrades to plain order.
    novel_note = ""
    try:
        from memory import room_model
        counts = room_model.label_sightings({o.get("label") for o in keep})
        nmax = int(getattr(config, "ROOM_MODEL_NOVELTY_MAX_SIGHTINGS", 6))

        def _is_novel(o) -> bool:
            return counts.get(str(o.get("label") or "").strip().lower(), 0) < nmax

        novel = [o for o in keep if _is_novel(o)]
        if novel and len(novel) < len(keep):  # only worth flagging if SOME are fixtures
            keep = novel + [o for o in keep if not _is_novel(o)]
            novel_note = (
                f" The {novel[0].get('label')} is NEW to the room (Rex hasn't logged it "
                "here before) — prefer asking about that."
            )
    except Exception:
        pass
    # Person-oriented salience OUTRANKS novelty: what someone is HOLDING is the most
    # interesting thing in the room, period (live-logged 2026-07-08: Bret held a cup
    # for minutes while curiosity picked the background chair).
    held_note = ""
    held = [o for o in keep if o.get("near_person")]
    if held:
        keep = held + [o for o in keep if not o.get("near_person")]
        holder = str(held[0].get("near_person_name") or "the person")
        held_note = (
            f" The {held[0].get('label')} is IN {holder}'s hands or right beside them — "
            "ask about THAT ('what are you drinking?' energy), not the background."
        )
    # Learned names (curiosity Phase 1 write-back): a human told Rex what some of
    # these objects actually ARE — speak of "the sourdough starter", not "potted
    # plant". Single-source names (confidence 1) are marked so the LLM can hedge.
    def _display(o) -> str:
        label = str(o.get("label") or "")
        try:
            from memory import room_model
            named = room_model.human_label(label)
        except Exception:
            named = None
        if named:
            hedge = "" if int(named.get("confidence") or 0) >= 2 else " — one person's word"
            return f"{label} (they call it \"{named['name']}\"{hedge})"
        return label
    items = ", ".join(
        f"{_display(o)} "
        + ("(in their hands)" if o.get("near_person") else f"({o.get('position') or 'in view'})")
        for o in keep[:cap]
    )
    return (
        "Confirmed objects in view (a local object detector verified these are really "
        f"there — safe to name; USE the they-call-it name where one is given): "
        f"{items}.{held_note or novel_note}\n\n"
    )


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

    # `token` is the legacy conversation_agenda claim (LEGACY) or None (ENFORCE — the
    # governor arbitrates the win, so the claim is moot; None is claim-safe).
    token = None
    enforcing = _governor_enforcing()
    if enforcing:
        # ENFORCE: do NOT claim — arm the cooldowns NOW (on submit) so the top-of-
        # function gates (global + per-person) stop us re-submitting every tick while
        # arbitration is pending; a loser just waits out the cooldown and re-submits.
        # The candidate (submitted below, once _task is defined) carries the deferred
        # vision+ask as its speak_fn; only the governor winner runs it.
        _last_visual_curiosity_at = now
        _visual_curiosity_by_person[engaged_id] = now
    else:
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
            objects_line = _visual_curiosity_objects_line()
            family_clause = (
                "A child or teen is present, so keep it gentle and family-safe. "
                if profile.force_family_safe else ""
            )
            prefer_object = (
                "PREFER grounding it in one of the confirmed objects above when one "
                "leads somewhere natural — those are detector-verified, so you can name "
                "them with confidence. "
                if objects_line else ""
            )
            prompt = (
                f"You're mid-conversation with {first_name}, and they just went "
                f"quiet for a few seconds after a back-and-forth. You took a fresh "
                f"visual snapshot. Use it as a conversational springboard.\n\n"
                f"Vision summary JSON: {visual_json}\n\n"
                f"{objects_line}"
                f"{family_clause}"
                "Ask exactly ONE short, in-character Rex question grounded in a "
                f"specific visible, non-sensitive detail. {prefer_object}It can be dry "
                "or mildly teasing about clothing, accessories, objects, decor, or what "
                "they seem to be doing, but do not roast grief, emotions, body, identity, "
                "health, age, race, religion, politics, disability, money, or private "
                "screen/document text. Reference only things actually in view — a confirmed "
                "object or something in the vision summary — never invent an object. Do not "
                "say you took a picture. Do not explain the visual system. Address them by "
                "name if natural. End with a question mark."
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
            if token is not None:
                _release_proactive_purpose(token)
            with _visual_curiosity_lock:
                _visual_curiosity_in_flight = False

    if enforcing:
        # Submit the candidate carrying the deferred task. _winner_speak runs ONLY if
        # visual curiosity wins this tick; it sets the in-flight latch (cleared in
        # _task's finally) and spawns the worker so the slow vision+LLM work stays off
        # the consciousness tick.
        def _winner_speak() -> None:
            global _visual_curiosity_in_flight
            with _visual_curiosity_lock:
                if _visual_curiosity_in_flight:
                    return
                _visual_curiosity_in_flight = True
            threading.Thread(target=_task, daemon=True, name="visual-curiosity").start()

        _observe_governor_candidate(
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
            speak_fn=_winner_speak,
        )
        return

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


def _step_lull_callback(snapshot: dict, profile: SituationProfile) -> None:
    """
    Banked-callback humor in a mid-conversation lull: after a real
    back-and-forth goes quiet, resurface ONE stored "fun fact" premise about
    the engaged person as a dry callback line — the "counting ceiling panels
    again, fewer than the stars you pretend to photograph" slot.

    The tone/consent gating lives in callback_engine.lull_gates_clear (empathy,
    sober-room, unacked events, boundaries, tier, crowd, the shared pacing
    ledger); this step owns the lull envelope and its own trigger cooldowns,
    then hands composition+arbitration to the proactive speech engine
    (purpose="lull_callback", priority 58 — above visual_curiosity, below all
    sincerity flows). Trigger cooldowns are armed AT SUBMIT (anti-resubmit; a
    governor loss costs only the cooldown); the premise itself is spent ONLY in
    on_spoke, after the line actually played.
    """
    global _last_lull_callback_at

    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    min_silence = float(getattr(config, "CALLBACK_LULL_MIN_SILENCE_SECS", 12.0))
    active_window = float(getattr(config, "CALLBACK_LULL_ACTIVE_WINDOW_SECS", 60.0))
    global_cooldown = float(getattr(config, "CALLBACK_LULL_COOLDOWN_SECS", 600.0))
    person_cooldown = float(getattr(config, "CALLBACK_LULL_PERSON_COOLDOWN_SECS", 900.0))
    if (now - _last_lull_callback_at) < global_cooldown:
        return

    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    quiet_for = now - engaged_touch
    if quiet_for < min_silence or quiet_for > active_window:
        return
    if (now - _lull_callback_by_person.get(engaged_id, 0.0)) < person_cooldown:
        return

    # Only after a REAL exchange — a lull presumes there was conversation.
    try:
        turn_window = float(getattr(config, "VISUAL_CURIOSITY_TURN_WINDOW_SECS", 45.0))
        if _situation_assessor.recent_speech_turn_count(turn_window) < 2:
            return
    except Exception:
        if not profile.rapid_exchange:
            return

    try:
        from intelligence import callback_engine
        if not callback_engine.lull_gates_clear(engaged_id):
            return
        premise = callback_engine.pick_lull_premise(engaged_id)
    except Exception as exc:
        _log.debug("lull callback step error: %s", exc)
        return
    if not premise:
        return

    _last_lull_callback_at = now
    _lull_callback_by_person[engaged_id] = now

    first_name = "there"
    try:
        from memory import people as people_mod
        person = people_mod.get_person(engaged_id) or {}
        first_name = _first_name(person.get("name"), "there")
    except Exception:
        pass

    prompt = callback_engine.build_lull_prompt(first_name, premise)
    _log.info(
        "consciousness: lull callback candidate for person_id=%s after %.1fs quiet "
        "(premise id=%s)",
        engaged_id, quiet_for, premise.get("id"),
    )
    _generate_and_speak(
        prompt,
        emotion="amused",
        purpose="lull_callback",
        priority=int(getattr(config, "CALLBACK_LULL_PRIORITY", 58)),
        label=f"lull callback for {engaged_id}",
        metadata={"topic_key": f"callback:{engaged_id}:{premise.get('id')}"},
        on_spoke=lambda: callback_engine.spend_lull_premise(premise),
        # Tone state can shift between submit and the governor win (a heavy
        # disclosure mid-tick); re-run the engine gates right before composing.
        pre_speak_check=lambda: callback_engine.lull_gates_clear(engaged_id),
    )


_last_self_explore_at: float = 0.0


def _step_self_exploration(snapshot: dict, profile: SituationProfile) -> None:
    """
    Curiosity Phase 2, OPT-IN (EXPLORE_SELF_TRIGGER_ENABLED, default OFF — it
    MOVES THE ROBOT unprompted): when the novelty clock has been stale a long
    time, nobody is around, and the pack is healthy, Rex takes himself on the
    same supervised wander an invitation would start. The walk feeds the room
    model, which feeds the learn-by-asking queue — boredom literally produces
    questions for the next visitor.
    """
    global _last_self_explore_at
    if not bool(getattr(config, "EXPLORE_SELF_TRIGGER_ENABLED", False)):
        return
    if not bool(getattr(config, "EXPLORE_ENABLED", True)):
        return
    if profile.suppress_proactive:
        return
    if not _room_looks_empty(snapshot):
        return                                  # never self-wander with company
    try:
        from awareness import novelty_drive
        if novelty_drive.staleness_secs() < float(
            getattr(config, "EXPLORE_SELF_TRIGGER_STALENESS_SECS", 3600.0)
        ):
            return
    except Exception:
        return
    now = time.monotonic()
    if (now - _last_self_explore_at) < float(
        getattr(config, "EXPLORE_SELF_TRIGGER_COOLDOWN_SECS", 7200.0)
    ):
        return
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            return
    except Exception:
        pass
    try:
        from hardware import motion
        if (motion.telemetry() or {}).get("charging"):
            return                              # firmware locks the wheels anyway
    except Exception:
        pass
    try:
        from intelligence import exploration
        if exploration.active():
            return
        _last_self_explore_at = now
        _log.info("consciousness: self-triggered exploration (novelty staleness) — starting walk")
        exploration.start(None, None, source="boredom")
    except Exception as exc:
        _log.debug("self exploration start failed: %s", exc)


def _step_open_thread_followup(snapshot: dict, profile: SituationProfile) -> None:
    """
    Cross-session follow-through in a mid-conversation lull: the diary stored
    what this person left unresolved last time (open_threads on their
    conversation_summary episodes); ask about ONE of them — "hey, did the
    dentist thing ever happen?" Highest-priority lull candidate
    (OPEN_THREAD_PRIORITY 62 > lull callbacks 58 > news 54): remembering
    someone's actual life beats banked humor and headlines.
    """
    if not bool(getattr(config, "OPEN_THREAD_FOLLOWUP_ENABLED", True)):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    min_silence = float(getattr(config, "CALLBACK_LULL_MIN_SILENCE_SECS", 12.0))
    active_window = float(getattr(config, "CALLBACK_LULL_ACTIVE_WINDOW_SECS", 60.0))
    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None or engaged_id in _open_thread_asked_persons:
        return
    quiet_for = now - engaged_touch
    if quiet_for < min_silence or quiet_for > active_window:
        return
    try:
        turn_window = float(getattr(config, "VISUAL_CURIOSITY_TURN_WINDOW_SECS", 45.0))
        if _situation_assessor.recent_speech_turn_count(turn_window) < 2:
            return
    except Exception:
        if not profile.rapid_exchange:
            return

    try:
        from intelligence import open_threads
        candidates = open_threads.pending_for_person(engaged_id)
    except Exception as exc:
        _log.debug("open thread step error: %s", exc)
        return
    if not candidates:
        return
    pick = candidates[0]                     # freshest unresolved thread

    _open_thread_asked_persons.add(engaged_id)   # armed at submit (anti-resubmit)

    first_name = "there"
    try:
        from memory import people as people_mod
        person = people_mod.get_person(engaged_id) or {}
        first_name = _first_name(person.get("name"), "there")
    except Exception:
        pass

    when = open_threads.describe_age(pick["age_days"])
    prompt = (
        f"You're talking with {first_name}. {when.capitalize()}, they left "
        f"something unresolved: {pick['thread']}. The conversation just hit a "
        "lull — casually check in on it in ONE short in-character line "
        "(\"hey, did ... ever happen?\" energy). Warm and genuinely curious, "
        "not an interrogation; don't recite the memory back at them."
    )
    _log.info(
        "consciousness: open-thread follow-up for person_id=%s after %.1fs quiet (%r)",
        engaged_id, quiet_for, pick["thread"],
    )
    _generate_and_speak(
        prompt,
        emotion="curious",
        purpose="open_thread_followup",
        priority=int(getattr(config, "OPEN_THREAD_PRIORITY", 62)),
        label=f"open thread for {engaged_id}",
        metadata={"topic_key": f"thread:{engaged_id}:{pick['episode_id']}"},
        on_spoke=lambda: open_threads.mark_asked(pick["episode_id"], pick["thread"]),
    )


def _step_news_remark(snapshot: dict, profile: SituationProfile) -> None:
    """
    Current-events conversation invitation in a mid-conversation lull: surface
    ONE of today's cached stories (awareness/current_events.py, fetched once
    per day at startup) as a "hey, did you hear about ...?" opener that invites
    the person to pick the thread up.

    Deliberately B-material: same lull envelope as the banked-callback step but
    LOWER priority (NEWS_REMARK_PRIORITY 54 < lull_callback 58), a session cap
    (default 1), and a long cooldown — news competes for airtime, it never owns
    it. The story is spent (persisted) only in on_spoke, so a governor loss
    keeps it available for later.
    """
    global _last_news_remark_at, _news_remarks_this_session

    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return
    if _news_remarks_this_session >= int(getattr(config, "NEWS_REMARK_SESSION_CAP", 1)):
        return

    now = time.monotonic()
    if (now - _last_news_remark_at) < float(getattr(config, "NEWS_REMARK_COOLDOWN_SECS", 900.0)):
        return
    # Same lull window the callback step uses: a real exchange that went quiet.
    min_silence = float(getattr(config, "CALLBACK_LULL_MIN_SILENCE_SECS", 12.0))
    active_window = float(getattr(config, "CALLBACK_LULL_ACTIVE_WINDOW_SECS", 60.0))
    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None:
        return
    quiet_for = now - engaged_touch
    if quiet_for < min_silence or quiet_for > active_window:
        return
    try:
        turn_window = float(getattr(config, "VISUAL_CURIOSITY_TURN_WINDOW_SECS", 45.0))
        if _situation_assessor.recent_speech_turn_count(turn_window) < 2:
            return
    except Exception:
        if not profile.rapid_exchange:
            return

    # Interest-tailored first: when the engaged person is known and has stored
    # interests, prefer news about THEIR topics ("seen the new Strange New
    # Worlds episode?") over the generic headline pool. The per-topic fetch is
    # kicked in the background here (daily-cached, budget-capped), so the
    # first lull of a session may still serve general news while it lands.
    try:
        from awareness import current_events
    except Exception as exc:
        _log.debug("news remark step error: %s", exc)
        return

    first_name = ""
    interest_topics: list[str] = []
    if bool(getattr(config, "INTEREST_NEWS_ENABLED", True)):
        try:
            interest_topics, first_name = _engaged_interest_topics(engaged_id)
            if interest_topics:
                current_events.start_interest_refresh(interest_topics)
        except Exception as exc:
            _log.debug("interest news refresh kick failed: %s", exc)

    story = None
    interest_topic = None
    try:
        if interest_topics:
            picked = current_events.pick_interest_story(interest_topics)
            if picked:
                interest_topic, story = picked
        if story is None:
            story = current_events.pick_story()
    except Exception as exc:
        _log.debug("news remark step error: %s", exc)
        return
    if not story:
        return

    _last_news_remark_at = now          # armed at submit (anti-resubmit); the
    _news_remarks_this_session += 1     # story itself is spent only on_spoke

    if interest_topic:
        who = first_name or "the person you're with"
        prompt = (
            f"You know {who} loves {interest_topic}, and you read some news on it. "
            f"The story: {story['headline']} — {story['summary']} "
            "A conversational lull just opened — bring it up naturally in ONE "
            f"short in-character line aimed at {who} PERSONALLY (\"have you seen/"
            "heard ...\" energy — you remembered their interest and found them "
            "something). Tease the concrete detail, invite them in, don't recite "
            "the summary."
        )
        label = f"interest news lull remark ({interest_topic})"
        on_spoke = (lambda t=interest_topic, s=story:
                    current_events.mark_interest_story_mentioned(t, s))
    else:
        prompt = (
            "You read some news this morning and a conversational lull just opened. "
            f"The story: {story['headline']} — {story['summary']} "
            "Bring it up naturally in ONE short in-character line that INVITES the "
            "other person into the topic (\"hey, did you hear about ...\" energy — "
            "a conversation opener, not a news broadcast). Don't recite the whole "
            "summary; tease the interesting part and let them ask."
        )
        label = "news lull remark"
        on_spoke = lambda s=story: current_events.mark_mentioned(s)

    _log.info(
        "consciousness: news remark candidate after %.1fs quiet (interest=%s story=%r)",
        quiet_for, interest_topic or "-", story["headline"],
    )
    _generate_and_speak(
        prompt,
        emotion="amused",
        purpose="news_remark",
        priority=int(getattr(config, "NEWS_REMARK_PRIORITY", 54)),
        label=label,
        metadata={"topic_key": f"news:{story['headline'][:60]}"},
        on_spoke=on_spoke,
    )


def _engaged_interest_topics(person_db_id: Optional[int]) -> "tuple[list[str], str]":
    """The engaged person's top interest names (for tailored news) and their
    first name. ([], "") for unknown people or on any failure."""
    if person_db_id is None:
        return [], ""
    try:
        from memory import interests as interests_mem
        from memory import people as people_mem
        rows = interests_mem.get_interests_for_prompt(
            int(person_db_id),
            limit=int(getattr(config, "INTEREST_NEWS_TOPICS_PER_PERSON", 3)),
        ) or []
        topics = [str(r.get("name") or "").strip() for r in rows]
        topics = [t for t in topics if len(t) >= 3]
        name = ""
        try:
            person = people_mem.get_person(int(person_db_id)) or {}
            name = str(person.get("name") or "").strip().split()[0] if person.get("name") else ""
        except Exception:
            name = ""
        return topics, name
    except Exception as exc:
        _log.debug("engaged interest topics lookup failed: %s", exc)
        return [], ""


_last_interest_discovery_at: float = 0.0
_interest_discovery_sessions_asked: set = set()


def _step_interest_discovery(snapshot: dict, profile: SituationProfile) -> None:
    """Lull question that grows the interest catalogue: when the engaged person
    has FEW stored interests, ask what they're into that they haven't shared
    ("so, is there anything you're into you haven't told me before?"). The
    normal extraction pipeline harvests the answer — this step only asks.

    Same lull envelope as the news remark but rarer: once per person per
    session, a long cooldown, and it stands down entirely once the catalogue
    is rich (INTEREST_DISCOVERY_MAX_KNOWN — the point is filling gaps for NEW
    people, who today are barely asked about themselves)."""
    global _last_interest_discovery_at

    if not bool(getattr(config, "INTEREST_DISCOVERY_ENABLED", True)):
        return
    if profile.suppress_proactive or profile.user_mid_sentence or profile.interaction_busy:
        return
    if not profile.conversation_active:
        return
    if is_waiting_for_response() or not _can_proactive_speak():
        return

    now = time.monotonic()
    if (now - _last_interest_discovery_at) < float(
        getattr(config, "INTEREST_DISCOVERY_COOLDOWN_SECS", 1800.0)
    ):
        return
    min_silence = float(getattr(config, "CALLBACK_LULL_MIN_SILENCE_SECS", 12.0))
    active_window = float(getattr(config, "CALLBACK_LULL_ACTIVE_WINDOW_SECS", 60.0))
    with _engaged_lock:
        engaged_id = _engaged_person_id
        engaged_touch = _engaged_last_touch_at
    if engaged_id is None or engaged_id in _interest_discovery_sessions_asked:
        return
    quiet_for = now - engaged_touch
    if quiet_for < min_silence or quiet_for > active_window:
        return
    try:
        turn_window = float(getattr(config, "VISUAL_CURIOSITY_TURN_WINDOW_SECS", 45.0))
        if _situation_assessor.recent_speech_turn_count(turn_window) < 2:
            return
    except Exception:
        if not profile.rapid_exchange:
            return

    try:
        from memory import interests as interests_mem
        known = interests_mem.get_interests_for_prompt(
            int(engaged_id),
            limit=int(getattr(config, "INTEREST_DISCOVERY_MAX_KNOWN", 5)),
        ) or []
    except Exception as exc:
        _log.debug("interest discovery lookup failed: %s", exc)
        return
    if len(known) >= int(getattr(config, "INTEREST_DISCOVERY_MAX_KNOWN", 5)):
        return

    _, first_name = _engaged_interest_topics(engaged_id)
    known_names = ", ".join(
        str(r.get("name") or "").strip() for r in known if r.get("name")
    )
    who = first_name or "them"
    known_part = (
        f"You already know they're into: {known_names}. Ask about something NEW — "
        "not those. " if known_names else
        "You know almost nothing about what they're into yet. "
    )
    prompt = (
        f"A conversational lull just opened with {who}. {known_part}"
        "In ONE short, warm, in-character line, ask what they're into that they "
        "haven't told you about yet — hobbies, shows, obsessions, whatever "
        "(\"so, is there anything you're into you haven't told me before?\" "
        "energy). One genuine question, no list of examples longer than two."
    )
    _last_interest_discovery_at = now
    _interest_discovery_sessions_asked.add(engaged_id)
    _log.info(
        "consciousness: interest discovery question after %.1fs quiet "
        "(person_id=%s known=%d)", quiet_for, engaged_id, len(known),
    )
    _generate_and_speak(
        prompt,
        emotion="curious",
        purpose="interest_discovery",
        priority=int(getattr(config, "INTEREST_DISCOVERY_PRIORITY", 52)),
        label=f"interest discovery question ({who})",
        metadata={"topic_key": f"interest_discovery:{engaged_id}"},
        on_spoke=lambda: begin_response_wait(
            float(getattr(config, "INTEREST_DISCOVERY_RESPONSE_WAIT_SECS", 20.0))
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Presence tracking (departure / return reactions)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_appearance_hint(person_db_id: Optional[int]) -> Optional[str]:
    """Return one safe, non-body visual hint for a gentle riff, or ``None``.

    Appearance enrollment is deliberately broader than conversational material.
    In particular, height, build, age, and arbitrary "notable features" do not
    belong in an unprompted comment.  Keep this shared helper to hair and a small
    allowlist of neutral accessories, so return greetings and Lean cues follow the
    same rule.
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
        try:
            features = json.loads(notable) if isinstance(notable, str) else notable
        except (TypeError, ValueError):
            features = [notable]
        if not isinstance(features, list):
            features = [features]
        safe_accessories = {
            "glasses", "sunglasses", "hat", "cap", "beanie", "headphones",
            "earbuds", "scarf",
        }
        for feature in features:
            value = str(feature or "").strip().lower()
            if value in safe_accessories:
                candidates.append(f"a familiar {value}")

    hair = []
    if attrs.get("hair_color"):
        hair.append(attrs["hair_color"])
    if attrs.get("hair_style"):
        hair.append(attrs["hair_style"])
    if hair:
        candidates.append(f"{' '.join(hair)} hair")

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


def person_visibly_facing(
    person_id: Optional[int], max_age_secs: float = 6.0
) -> bool:
    """True when face-tracking has held a lock on this person's FACE within
    ``max_age_secs`` — i.e. they are present and oriented toward Rex.

    A detected, tracked face is inherently a mostly-frontal face (the
    wide-angle lens rarely detects profiles), so a live face lock is honest
    "they're facing me" evidence with no iris/gaze estimation needed. Used by
    the lean impulse gate: someone who stays turned toward Rex during a lull
    is WAITING, not withdrawing (owner 2026-08-03: he sat looking straight at
    Rex for a minute while the low-energy read kept Rex silent)."""
    if person_id is None:
        return False
    try:
        if int(_face_tracking_lock.get("person_id")) != int(person_id):
            return False
        last_seen = float(_face_tracking_lock.get("last_seen_at") or 0.0)
    except Exception:
        return False
    return last_seen > 0 and (time.monotonic() - last_seen) <= float(max_age_secs)


def _step_relationship_inquiry(snapshot: dict, profile: SituationProfile) -> None:
    """
    When Rex is engaged with a known person and an UNKNOWN face has been
    continuously visible for UNKNOWN_WITH_ENGAGED_CONFIRM_SECS, ask the engaged
    person who the stranger is and what their relationship is.

    Sets _pending_relationship_prompt so interaction.py parses the next utterance
    for a {name, relationship} pair.
    """
    global _last_identity_prompt_at, _unknown_first_seen_at
    global _relationship_prompt_in_flight_at

    if not _can_speak():
        return
    if profile.suppress_proactive:
        return
    if _pending_relationship_prompt.is_set():
        return
    if _relationship_prompt_in_flight.is_set():
        # A candidate is submitted but hasn't spoken yet — don't stack a duplicate. But the
        # governor can REJECT it (e.g. a higher-priority emotional_checkin wins the tick), in
        # which case its speak_fn/on_spoke never run and nothing clears this latch. A latch
        # older than the stale window is dead — clear it and allow a retry.
        stale = float(getattr(config, "RELATIONSHIP_PROMPT_INFLIGHT_STALE_SECS", 10.0) or 10.0)
        if (time.monotonic() - _relationship_prompt_in_flight_at) < stale:
            return
        _log.info(
            "[relationship_inquiry] stale in-flight latch (>%.0fs, governor likely rejected) "
            "— clearing and retrying", stale,
        )
        _relationship_prompt_in_flight.clear()

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
    rel_ctx = {
        "engaged_person_id": engaged_id,
        "engaged_name": engaged_name,
        "slot_id": ripe_slot,
        "asked_at": now,
    }
    _log.info(
        "consciousness: asking %s about unknown visitor (slot=%s)",
        engaged_name, ripe_slot,
    )

    def _relationship_inquiry_spoke() -> None:
        # Arm the reply window + cooldown ONLY when the line actually speaks. Under ENFORCE,
        # _generate_and_speak returns True at governor SUBMISSION, so the old pre-speak arming
        # + `if not _generate_and_speak(): clear` self-heal was dead code: on a candidate the
        # governor then REJECTED (e.g. a higher-priority emotional_checkin won the tick),
        # _pending_relationship_prompt stayed set with no question asked, and the next user
        # statement got mis-parsed as the answer. Arming here — the on_spoke hook that fires
        # only after the line enqueues — closes that hole (mirrors _step_identity_prompt).
        global _last_identity_prompt_at
        _last_identity_prompt_at = time.monotonic()
        _pending_relationship_context.clear()
        _pending_relationship_context.update(rel_ctx)
        _pending_relationship_prompt.set()
        _relationship_prompt_in_flight.clear()

    _relationship_prompt_in_flight.set()
    _relationship_prompt_in_flight_at = now
    if not _generate_and_speak(
        f"You're talking with '{first_name}' and a new unfamiliar face has just "
        f"joined the view. In one short in-character Rex line, ask {first_name} "
        f"who the newcomer is AND what their relationship to {first_name} is — "
        f"e.g. 'Oh hey, who's this, {first_name}? Friend of yours?' Keep it warm "
        f"and curious, one line only, ending with a question mark.",
        emotion="curious",
        wait_secs=getattr(config, "IDENTITY_RESPONSE_WAIT_SECS", 20.0),
        purpose="relationship_inquiry",
        on_spoke=_relationship_inquiry_spoke,
    ):
        # Submission itself failed (legacy claim rejected / not enforcing) → release the latch.
        _relationship_prompt_in_flight.clear()


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
        # staged_at rides along: the quip-resolution window below is measured from
        # STAGING, not first-missing (the confirm window already consumed 12-40 s of
        # first-missing time — measuring the old 30 s timeout from first-missing made
        # non-engaged departures, confirm 40 s, expire the instant they staged).
        _pending_departure_keys[key] = (first_missing, person_name, person_db_id, now)
        _confirmed_absent_at[key] = first_missing
        _log.debug(
            "consciousness: staged departure for key=%s name=%r after %.1fs absent "
            "(confirm=%.1fs)",
            key, person_name, now - first_missing, confirm_for_key,
        )

    # ── Resolve pending departures ─────────────────────────────────────────────
    for key in list(_pending_departure_keys):
        departed_at, person_name, person_db_id, staged_at = _pending_departure_keys[key]

        # Person returned — cancel
        if key in current_keys:
            del _pending_departure_keys[key]
            continue

        # Explicit recent goodbye + absence past the confirm window: latch the
        # conversation closed NOW, ahead of the quip gates below. A stale
        # face-tracking hold or ambient room audio must not keep the conversation
        # open after the person SAID they were leaving and then left the view
        # (field-logged 2026-07-11: the latch never armed and Rex questioned an
        # empty room for 2+ minutes). Worst case self-heals: if they said bye but
        # stayed, their next turn or reappearance clears the latch and re-greets.
        try:
            from intelligence import end_thread
            if end_thread.recent_farewell():
                end_thread.note_farewell_departure()
                _first_missing_at.pop(key, None)
                del _pending_departure_keys[key]
                _last_departure_reaction_at[key] = now
                _last_presence_reaction_at[key] = now
                _visit_arrival = _visit_started_at.pop(key, None)
                if isinstance(key, int) and person_name:
                    _log.info(
                        "consciousness: %s left after an explicit goodbye — "
                        "conversation closed, suppressing departure quip",
                        person_name,
                    )
                    episodic_hooks.visit_departure(
                        person_db_id, person_name, _visit_arrival, departed_at,
                    )
                else:
                    _log.info(
                        "consciousness: unknown (key=%s) left after a goodbye — "
                        "conversation closed, suppressing departure quip",
                        key,
                    )
                continue
        except Exception:
            _log.debug("farewell-departure check failed", exc_info=True)

        # Quip window expired: resolve as a SILENT departure — clean up the missing
        # timer and log the visit, just without the spoken quip. The old bare delete
        # left _first_missing_at armed, so the entry re-staged and re-deleted every
        # tick FOREVER: no quip, no farewell latch, no visit log, and the person
        # haunted presence state until shutdown (the 2026-07-11 empty-room bug).
        if now - staged_at > departure_cooldown:
            _first_missing_at.pop(key, None)
            del _pending_departure_keys[key]
            _visit_arrival = _visit_started_at.pop(key, None)
            if isinstance(key, int) and person_name:
                _log.info(
                    "consciousness: silent departure resolved for %s "
                    "(quip window missed)", person_name,
                )
                episodic_hooks.visit_departure(
                    person_db_id, person_name, _visit_arrival, departed_at,
                )
            else:
                _log.info(
                    "consciousness: silent departure resolved for key=%s", key,
                )
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
        _visit_arrival = _visit_started_at.pop(key, None)  # arrival of the just-ended visit

        is_known = isinstance(key, int) and person_name
        # (The explicit-goodbye latch used to live here, AFTER the quip gates — a
        # stale face-track hold could block it until the resolve window expired.
        # It now runs at the TOP of the resolve loop, ahead of every gate.)

        if is_known:
            first_name = _first_name(person_name, "there")
            _log.info("consciousness: departure reaction firing for %s", person_name)
            tone = _presence_relationship_tone(person_db_id)
            tone_clause = f" {tone}" if tone else ""
            _generate_and_speak_presence(
                f"The person named '{first_name}' just slipped out of your camera view. "
                "React in one short in-character line as Rex — playful and dry, "
                "but not mean or needy. Do not imply they literally left the room; "
                "they may only be off-camera. Do not imply nobody likes or misses "
                f"them.{tone_clause} Examples: 'Lost visual on {first_name}. Dramatic.', "
                f"'And {first_name} exits frame, stage left.', "
                f"'Fine, {first_name}, hide from the optics.' "
                f"Address {first_name} by name. One line only.",
                label=f"departure for {person_name}",
                tag_key=key,
                emotion="curious",
            )
            # "I spent about 40 minutes with Bret" → rex.db (visit arrival → departure).
            episodic_hooks.visit_departure(
                person_db_id, person_name, _visit_arrival, departed_at,
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

        if _is_galactic_hair_stylist(person_name):
            first_visible = _first_sight_seen_at.setdefault(key, now)
            confirm_visible = float(getattr(config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 3.0))
            if (now - first_visible) < max(0.0, confirm_visible):
                first_sight_pending_keys.add(key)
                continue
            _stage_hair_stylist_greeting(key=key, person_name=person_name)
            if _try_fire_hair_stylist_greeting(
                key=key,
                person_name=person_name,
                person_db_id=person_db_id,
                profile=profile,
            ):
                continue
            if key not in _hair_stylist_greeted_this_session:
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
                followup_event_name: str = ""
                anticipated_to_mark: Optional[tuple[Optional[int], object]] = None
                milestone_to_mark: Optional[int] = None
                profile_question_to_record: Optional[dict] = None
                disposition_to_mark: Optional[int] = None

                # Birthday window, computed up front. On the ACTUAL day (T-0) the
                # birthday OUTRANKS even the sensitive emotional check-in below
                # (config.BIRTHDAY_WINS_ON_DAY) — you should hear "happy birthday" on
                # your birthday. In the LEAD-UP days it stays BELOW the check-in
                # ("care before the bit"), handled as Priority 1 lower down.
                bday_days = _pick_birthday_window(person_db_id)
                if bday_days == 0 and bool(getattr(config, "BIRTHDAY_WINS_ON_DAY", True)):
                    prompt = _build_birthday_prompt(first_name, bday_days)
                    label = f"startup birthday (T-0) for {person_name}"
                    emotion = "happy"
                    _log.info(
                        "consciousness: startup birthday (T-0, wins-on-day) for %s",
                        person_name,
                    )

                # Priority 0 — recent sensitive emotional event.
                # This intentionally outranks temporal banter like
                # "back so soon"; care comes before the bit.
                if prompt is None:
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

                # Priority 1 — birthday within reminder window (LEAD-UP days; the
                # actual day T-0 was already handled above, before the check-in).
                if prompt is None and bday_days is not None:
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
                        milestone_to_mark = milestone
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
                            followup_event_name = ev_name
                            prompt = (
                                f"{context_sentence} "
                                f"You remember they told you they had this on their schedule: "
                                f"'{ev_name}' — and the date has now passed. Greet them and "
                                f"ask whether '{ev_name}' ended up happening and how it went, "
                                f"in two short Rex-style sentences. A plan is not a fact — do "
                                f"NOT assert that it happened; if it fell through, your "
                                f"question should still land naturally. Address {first_name} "
                                f"by name. The second sentence must end in a question mark."
                            )
                            label = f"startup followup ({ev_name}) for {person_name}"
                            emotion = "curious"
                            _log.info(
                                "consciousness: startup follow-up for %s — %s",
                                person_name, ev_name,
                            )

                # Priority 2.6 — session-opener continuity: an UNDATED thread from a
                # previous session that never got resolved ("last night you never told
                # me how the soup turned out"). Dated events are Priority 2.5; this
                # covers dateless plans that would otherwise wait FOLLOWUP_UNDATED_DAYS.
                if prompt is None and bool(
                    getattr(config, "SESSION_OPENER_CONTINUITY_ENABLED", True)
                ):
                    try:
                        from memory import events as events_mod
                        threads = events_mod.get_recent_open_threads(person_db_id) or []
                    except Exception:
                        threads = []
                    if threads:
                        ev = threads[0]
                        ev_name = ev.get("event_name") or ""
                        if ev_name:
                            when_label = events_mod.mentioned_when_label(ev.get("mentioned_at"))
                            followup_to_remove = (person_db_id, ev.get("id"))
                            followup_event_name = ev_name
                            # when_label is when they MENTIONED it, not when the
                            # event happens — the old phrasing conflated the two
                            # (field 2026-08-01 18:08: a trip planned for TOMORROW,
                            # mentioned earlier today, was greeted with "How'd Lake
                            # Folsom go earlier today?").
                            prompt = (
                                f"{context_sentence} "
                                f"{when_label.capitalize()} they MENTIONED this and you never "
                                f"heard how it turned out: '{ev_name}'. You do not know when — "
                                f"or whether — it actually happened; {when_label} is only when "
                                f"they told you about it. Greet {first_name} by name, warm not "
                                f"roasty, then pick the thread back up — ask whether "
                                f"'{ev_name}' ended up happening / how it turned out. Never "
                                f"state that the event itself was {when_label}. Two short "
                                f"Rex-style sentences; the second must end in a question mark."
                            )
                            label = f"startup continuity ({ev_name}) for {person_name}"
                            emotion = "curious"
                            _log.info(
                                "consciousness: session-opener continuity for %s — %r (%s)",
                                person_name, ev_name, when_label,
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

                # Relationship-aware greeting tone: warm/familiar for a close friend or
                # Rex's creator, friendlier-but-reserved for an acquaintance. Drives the
                # return greetings AND the simple warm default below.
                greeting_tone, greeting_warm_default = _greeting_profile(person_db_id)

                # Repeat-visit opener variety: rotate a short hello ("what's up?",
                # "what's new?") for a same-day / within-window return so it isn't always
                # "how are you?". Keyed on greetings_today (persists across runs), so even
                # restarting the program within the window cycles the opener. None on the
                # first greeting → the default warm "how are you".
                prior_today = _same_day_return_count(person_db_id)
                greeting_opener = _repeat_greeting_opener(prior_today + 1)

                # How long since Rex last greeted THIS person, read from the DB so it
                # survives a restart (owner gripe 2026-08-05). Every in-memory guard —
                # _greeted_this_session, the monotonic presence cooldowns — is wiped by
                # a reboot, which made the single most common repeat-visit case (relaunch
                # him twenty minutes later) the one case nothing could suppress.
                greet_bucket, greet_age = _greeting_recency(person_db_id)

                # Priority 3.4 — a QUICK return: he greeted them minutes or a couple of
                # hours ago. Outranks the same-day beat below, which is calendar-day
                # coarse and would still run a full "you're back" hello for someone who
                # stepped out for coffee. People don't re-greet at that range; they say
                # four words, or nothing.
                if prompt is None and greet_bucket is not None:
                    constraint = greeting_cadence.greeting_constraint(greet_bucket, greet_age)
                    if constraint:
                        # Tone FIRST, constraint LAST. The relationship tone says
                        # things like "greet them with genuine warmth" — which reads
                        # as an instruction to run a full hello, exactly what the
                        # constraint forbids. Whichever comes last wins, and here the
                        # constraint has to.
                        prompt = (
                            f"You see {first_name} again. {greeting_tone} {constraint}"
                        )
                        label = f"startup quick-return ({greet_bucket}) for {person_name}"
                        emotion = "happy" if greet_bucket == greeting_cadence.RECENT else "amused"
                        _log.info(
                            "consciousness: startup quick-return for %s (%s, %.0fs since last greeting)",
                            person_name, greet_bucket, greet_age or 0.0,
                        )

                # Priority 3.5 — same-day repeat activation. A warm "good to see you back",
                # NOT a roast, keyed on Rex's recorded greetings_today_count.
                if prompt is None:
                    if prior_today >= 1:
                        prompt = _build_same_day_return_prompt(
                            first_name, prior_today, tone=greeting_tone,
                            opener=greeting_opener)
                        label = f"startup same-day return (#{prior_today + 1}) for {person_name}"
                        emotion = "happy"
                        _log.info(
                            "consciousness: startup same-day return for %s (greeting #%d today)",
                            person_name, prior_today + 1,
                        )

                # Priority 3.8 — visit-cadence trend (streak / frequency / medium gap).
                # The human-shaped "we've been seeing a lot of each other" awareness:
                # "third day in a row", "4 visits this week", "first time in ~2 weeks"
                # (the 2–60-day band no other hook covered). First greeting of the day
                # only, and computed from existing session rows — zero extra tokens.
                if prompt is None and prior_today == 0 and bool(
                    getattr(config, "TREND_GREETING_HOOK_ENABLED", True)
                ):
                    # Cadence remarks are for ESTABLISHED relationships. Someone Rex
                    # barely knows (sparse profile) gets the getting-to-know-you flow
                    # instead — "you're becoming a regular" before knowing their name
                    # is backwards.
                    try:
                        _sparse = profile_questions.profile_fact_count(person_db_id) <= int(
                            getattr(config, "LOW_MEMORY_PROFILE_MAX_FACTS", 4) or 4
                        )
                    except Exception:
                        _sparse = False
                    hook = None
                    if not _sparse:
                        try:
                            from memory import trends as _trends
                            hook = _trends.cadence_hook(person_db_id)
                        except Exception:
                            hook = None
                    if hook is not None:
                        kind, phrase = hook
                        if kind == "medium_gap":
                            detail = (
                                f"It's been {phrase} since you last saw them — notice it "
                                f"warmly (glad they're back, maybe a dry 'the place was too "
                                f"quiet'), never guilt-trippy."
                            )
                        else:
                            detail = (
                                f"This is {phrase} they've come by — you've genuinely "
                                f"noticed they're around a lot, and you like it. Remark on "
                                f"it the way a friend would (warm, a little dry — 'you're "
                                f"becoming a regular' energy), never as a logged statistic."
                            )
                        prompt = (
                            f"You see '{first_name}', someone you know. {detail} "
                            f"{greeting_tone} Then ask one small open question. Address "
                            f"{first_name} by name. Two short sentences max — the second "
                            f"must end in a question mark."
                        )
                        label = f"startup cadence ({kind}) for {person_name}"
                        emotion = "happy"
                        _log.info(
                            "consciousness: startup cadence hook for %s (%s: %s)",
                            person_name, kind, phrase,
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
                        prompt = _build_long_absence_prompt(
                            first_name, absence[1], tone=greeting_tone)
                        label = f"startup long-absence for {person_name}"
                        emotion = "happy"
                        _log.info(
                            "consciousness: startup long-absence for %s (%.1f days)",
                            person_name, absence[1],
                        )
                    elif (
                        absence
                        and absence[0] == "recent_return"
                        and process_uptime >= startup_recent_grace
                    ):
                        prompt = _build_recent_return_prompt(
                            first_name, absence[1], tone=greeting_tone,
                            opener=(greeting_opener if absence[1] <= 8.0 else None))
                        label = f"startup recent-return for {person_name}"
                        emotion = "happy"
                        _log.info(
                            "consciousness: startup recent-return for %s (%.1f hrs)",
                            person_name, absence[1],
                        )

                # Priority 4.5 — default warm greeting for known friends/creator: a plain
                # "how are you?", scaled by relationship. This takes priority over the
                # disposition roast / interest cold-open / profile question below, which
                # are reserved for people Rex is still getting to know (acquaintances) —
                # per Bret's feedback that a greeting should just be a friendly hello, not
                # a themed hook or a roast.
                if prompt is None and greeting_warm_default:
                    # Returning-regular flavor: for an established regular, add a warm
                    # "look who's back" familiarity note and rotate the opener by visit_count
                    # so even the first boot of the day varies (it otherwise hard-defaults to
                    # "how are you" every cold boot — the field "no greeting variation" gripe).
                    _greet_note = ""
                    _greet_opener = greeting_opener
                    _allow_familiarity = False
                    _greet_require_question = True
                    try:
                        if (
                            bool(getattr(config, "PRESENCE_RETURNING_REGULAR_GREETING_ENABLED", True))
                            and isinstance(person_db_id, int)
                        ):
                            from memory import people as _people_mod
                            _prow = _people_mod.get_person(person_db_id) or {}
                            _visits = int(_prow.get("visit_count") or 0)
                            if _visits >= int(getattr(config, "PRESENCE_RETURNING_REGULAR_MIN_VISITS", 4)):
                                _allow_familiarity = True
                                _greet_note = (
                                    f"You know {first_name} well — a regular you're always "
                                    f"glad to see, so a warm, familiar 'look who's back / hey, "
                                    f"it's you' vibe fits."
                                )
                                if not _greet_opener:
                                    # STYLE rotation, not just phrase rotation: statement
                                    # and time-of-day hellos mixed with the question forms
                                    # so cold boots don't all open "Hey Bret, what's up?"
                                    # (owner gripe 2026-07-06 — fine sometimes, stale as
                                    # the default).
                                    _greet_opener, _q = _first_greeting_style(_visits)
                                    _greet_require_question = _q
                    except Exception as exc:
                        _log.debug("[greeting] returning-regular flavor failed: %s", exc)
                    prompt = _build_simple_greeting_prompt(
                        first_name, greeting_tone, note=_greet_note,
                        opener=_greet_opener, allow_familiarity=_allow_familiarity,
                        require_question=_greet_require_question)
                    label = f"first-sight warm greeting for {person_name}"
                    emotion = "happy"
                    _log.info(
                        "consciousness: first-sight warm greeting for %s (friend/creator, familiar=%s)",
                        person_name, _allow_familiarity,
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

                # Interest/fact cold-open — lead with something Rex already KNOWS they
                # care about (ranked across interests+facts by the same lead-score as
                # celebrations) before falling to a generic profile question.
                if prompt is None:
                    callback = _pick_cold_open_callback(person_db_id)
                    if callback is not None:
                        prompt = _build_cold_open_callback_prompt(
                            first_name, callback, context_sentence,
                        )
                        label = f"first-sight interest cold-open ({callback.get('kind')}) for {person_name}"
                        emotion = "curious"
                        _log.info(
                            "consciousness: first-sight interest cold-open for %s — %s:%r",
                            person_name, callback.get("kind"), callback.get("topic"),
                        )
                        # Anti-repeat: mark the interest asked so the cold-open ROTATES
                        # instead of re-leading with the same one every startup (the
                        # reactive path already marks; the cold-open never did, so the
                        # top interest — e.g. ice cream — opened forever).
                        if callback.get("kind") == "interest" and callback.get("topic"):
                            try:
                                from memory import interests as interests_mem
                                interests_mem.mark_interest_asked(
                                    person_db_id,
                                    str(callback["topic"]),
                                    cooldown_days=int(getattr(
                                        config, "COLD_OPEN_INTEREST_COOLDOWN_DAYS", 21)),
                                )
                            except Exception as exc:
                                _log.debug("cold-open interest cooldown mark failed: %s", exc)

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

                # Whichever branch won, if Rex already asked this person how they're
                # doing recently, the greeting must not ask again. Applied HERE rather
                # than inside each builder so it also covers the LLM-improvised
                # branches, which could otherwise smuggle the question back in.
                if prompt is not None and direct_text is None:
                    _no_reask = _wellbeing_ask_clause(person_db_id)
                    if _no_reask:
                        prompt = f"{prompt} {_no_reask}"

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
                        # Arm the resolver so the user's NEXT reply ("I never went")
                        # closes this event in memory — otherwise the passed-date plan
                        # stays 'planned' and Rex re-asks about it every run.
                        try:
                            from intelligence import interaction as _interaction
                            _interaction.set_awaiting_followup_event(
                                followup_to_remove[0],
                                followup_to_remove[1],
                                followup_event_name,
                            )
                        except Exception as exc:
                            _log.debug("arm startup follow-up resolution failed: %s", exc)
                    if anticipated_to_mark is not None:
                        _anticipated_events.add(anticipated_to_mark)
                        try:
                            from memory import events as _events_mod
                            _events_mod.mark_anticipated(int(anticipated_to_mark[1]))
                        except Exception:
                            pass
                    if milestone_to_mark is not None:
                        try:
                            from memory import people as people_mod
                            people_mod.record_milestone_greeted(
                                person_db_id, milestone_to_mark
                            )
                        except Exception as exc:
                            _log.debug(
                                "milestone mark failed person_id=%s: %s",
                                person_db_id, exc,
                            )
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
                    _face_recognized_chirp()
                    # "I saw <name>" → rex.db (once per known person per run).
                    episodic_hooks.person_seen(person_db_id, person_name)
                    # Memorable greeting tiers (birthday/celebration/milestone/reunion/
                    # check-in) → rex.db, keyed on the dispatched greeting's label.
                    episodic_hooks.greeting_from_label(label, person_db_id, person_name)
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
        # Back in frame — lift any "conversation closed after goodbye" dormancy so
        # the welcome-back below (and normal proactive life) can resume.
        try:
            from intelligence import end_thread
            end_thread.note_presence_return()
        except Exception:
            _log.debug("presence-return latch clear failed", exc_info=True)
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
            if _is_galactic_hair_stylist(person_name):
                _stage_hair_stylist_greeting(
                    key=key,
                    person_name=person_name,
                    returning=True,
                )
                _try_fire_hair_stylist_greeting(
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
                try:
                    from memory import events as _events_mod
                    _events_mod.mark_anticipated(int(anticipated["id"]))
                except Exception:
                    pass
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
            tone = _presence_relationship_tone(person_db_id)
            tone_clause = f" {tone}" if tone else ""
            if appearance_hint and random.random() < getattr(config, "APPEARANCE_RIFF_PROBABILITY", 0.35):
                prompt = (
                    f"The person named '{first_name}' just came back into your camera view "
                    f"after about {int(absent_secs)} seconds away. You remember this about "
                    f"their appearance: {appearance_hint}. React in one short in-character "
                    f"Rex line that NATURALLY references that appearance detail — warm but "
                    f"dry.{tone_clause} Address {first_name} by name. One line only."
                )
            else:
                prompt = (
                    f"The person named '{first_name}' just came back into your camera view after "
                    f"being away for about {int(absent_secs)} seconds. "
                    f"React in one short in-character line as Rex — warm but dry.{tone_clause} Examples: "
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
        # Stamp the visit's arrival once; held until departure for visit-duration recall.
        _visit_started_at.setdefault(key, now)
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


def _next_holiday_plan_for_person(person_id: Optional[int]) -> Optional[dict]:
    """Return the soonest upcoming holiday this person has not been asked about.

    Lean Brain uses this same lookup so it owns conversational timing while the
    calendar and per-person/date dedupe stay shared with the classic fallback.
    """
    if person_id is None:
        return None
    try:
        person_id = int(person_id)
    except (TypeError, ValueError):
        return None
    try:
        from awareness import holidays as holidays_mod
        from memory import relationships as rel_memory

        for holiday in holidays_mod.upcoming_holidays():
            date_key = str(holiday.get("date") or "").strip()
            if not date_key or not _holiday_plans_allowed(holiday):
                continue
            if (person_id, date_key) in _holiday_plans_asked:
                continue
            if rel_memory.was_proactive_asked(person_id, f"holiday_plans:{date_key}"):
                continue
            days_until = int(holiday.get("days_until", 0) or 0)
            return {
                **holiday,
                "when": holidays_mod._holiday_when_phrase(days_until),
            }
    except Exception as exc:
        _log.debug("holiday-plan lookup failed for person_id=%s: %s", person_id, exc)
    return None


def _mark_holiday_plan_asked(person_id: Optional[int], holiday: Optional[dict]) -> None:
    """Record a holiday question only after it genuinely spoke, per person/date."""
    if person_id is None or not holiday:
        return
    try:
        person_id = int(person_id)
    except (TypeError, ValueError):
        return
    date_key = str(holiday.get("date") or "").strip()
    if not date_key:
        return
    _holiday_plans_asked.add((person_id, date_key))
    try:
        from memory import relationships as rel_memory
        rel_memory.mark_proactive_asked(person_id, f"holiday_plans:{date_key}")
    except Exception as exc:
        _log.debug("persist holiday plans asked failed: %s", exc)


def _step_holiday_plans(snapshot: dict, profile: SituationProfile) -> None:
    """
    During an active conversation with a known person, if any public holiday
    is within its approach window, occasionally ask the engaged person about
    their plans. Each (person, holiday) pair is asked at most once per session;
    the holiday's iso date includes the year so next year resets naturally.
    """
    global _last_holiday_plans_check_at

    # Lean Brain owns conversational lulls. It consumes the same calendar cue,
    # so do not submit a competing legacy candidate in Lean mode.
    if bool(getattr(config, "LEAN_BRAIN_ENABLED", False)):
        return

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
        from memory import people as people_mod
        target = _next_holiday_plan_for_person(engaged_id)
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

        def _on_spoke() -> None:
            _mark_holiday_plan_asked(engaged_id, target)
            _log.info(
                "consciousness: holiday plans question for person_id=%s — %s (T-%dd, %s)",
                engaged_id, target["name"], days_until, target["window"],
            )

        _generate_and_speak(
            prompt, emotion="curious", purpose="memory_followup", on_spoke=_on_spoke,
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
                    f"weekend: '{ref_name}'. Ask whether it ended up happening and "
                    f"how it went, in one short Rex-style line ending with a "
                    f"question. Reference '{ref_name}' specifically; don't assert "
                    f"that it happened."
                )
            else:
                prompt = (
                    f"You're mid-conversation with '{first_name}'. It's Monday {monday_part_label}. "
                    f"Ask {first_name} how their weekend was, in one short Rex-style "
                    f"line ending with a question. Warm but dry."
                )
            emotion = "curious"

        def _on_spoke() -> None:
            # Mark this week's small-talk slot used only on an actual spoken turn.
            _weekly_smalltalk_asked.add(dedupe_key)
            _log.info(
                "consciousness: weekly small-talk for person_id=%s — slot=%s (week %d/%d)",
                engaged_id, slot, iso_week, iso_year,
            )

        _generate_and_speak(
            prompt, emotion=emotion, purpose="small_talk", on_spoke=_on_spoke,
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
                def _on_spoke() -> None:
                    # Mark the event acknowledged only when Rex actually SPOKE the
                    # check-in — under ENFORCE a losing candidate must not silently
                    # mark an event done it never voiced.
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
                    # "I checked in on <name> about a hard thing" → rex.db.
                    episodic_hooks.checkin(
                        engaged_id, person.get("name"),
                        f"I checked in on {first_name} about a {vibe} on their mind.",
                        detail={"category": cat, "valence": valence, "trigger": "remembered_event"},
                    )

                _generate_and_speak(
                    prompt, emotion=emotion, purpose="emotional_checkin",
                    on_spoke=_on_spoke,
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
            def _on_spoke() -> None:
                # Acknowledge only on an actual spoken celebration (ENFORCE-safe).
                try:
                    emo_events.mark_acknowledged(int(ev["id"]))
                except Exception:
                    pass
                _log.info(
                    "consciousness: proactive celebration check-in "
                    "(category=%s, event_id=%s) for person_id=%s",
                    cat, ev.get("id"), engaged_id,
                )
                # "I celebrated <name>'s good news" → rex.db.
                episodic_hooks.celebration(
                    engaged_id, person.get("name"),
                    f"I celebrated {first_name}'s good news with them.",
                    detail={"category": cat, "trigger": "remembered_celebration"},
                )

            _generate_and_speak(
                prompt, emotion="happy", purpose="celebration_checkin",
                on_spoke=_on_spoke,
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
        def _on_spoke() -> None:
            # Arm cooldown / clear the streak only on an actual spoken check-in.
            _note_emotional_checkin_fired(engaged_id)
            _negative_streak_started_at.pop(engaged_id, None)
            _log.info(
                "consciousness: proactive emotional check-in (B: sustained %s, "
                "streak=%.1fs, conf=%.2f) for person_id=%s",
                affect, now - streak_start, confidence, engaged_id,
            )
            # "I checked in on <name> when they sounded down" → rex.db.
            episodic_hooks.checkin(
                engaged_id, person.get("name"),
                f"I checked in on {first_name} when they sounded {affect}.",
                detail={"affect": affect, "trigger": "sustained_negative"},
            )

        _generate_and_speak(
            prompt, emotion=emotion, purpose="emotional_checkin",
            on_spoke=_on_spoke,
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


def _neck_saturated_at_rail(current: float, target: float, cfg: dict) -> bool:
    """True when the neck is already pinned at a mechanical limit AND the centering
    target still wants to push it further into that same rail — so a correction can't
    reduce the error and only jitters the head. Caller holds position instead."""
    if not bool(getattr(config, "FACE_TRACKING_RAIL_DAMP_ENABLED", True)):
        return False
    eps = int(getattr(config, "FACE_TRACKING_RAIL_DAMP_EPSILON_QUS", 60) or 0)
    if eps <= 0:
        return False
    lo = int(cfg["min"])
    hi = int(cfg["max"])
    at_high = current >= hi - eps and target >= hi - eps
    at_low = current <= lo + eps and target <= lo + eps
    return bool(at_high or at_low)


# ── Idle "mind of his own" head wander ───────────────────────────────────────

def _idle_head_wander_enabled() -> bool:
    return bool(getattr(config, "IDLE_HEAD_WANDER_ENABLED", True))


def _conversation_idle_secs(now: float) -> float:
    """Seconds since the LAST interaction (user spoke to Rex OR Rex spoke). 0.0 when
    nothing has happened yet, so Rex doesn't wander before the conversation begins."""
    last = max(_engaged_last_touch_at, _recent_engaged_touch_at, _last_proactive_speech_at)
    if last <= 0.0:
        return 0.0
    return max(0.0, now - last)


def _face_tracking_has_fresh_lock(now: float) -> bool:
    """True when face-tracking currently holds a recent lock on someone (something to
    look away from / return to)."""
    key = _face_tracking_lock.get("key")
    if key is None:
        return False
    last = float(_face_tracking_lock.get("last_seen_at") or 0.0)
    hold = float(getattr(config, "FACE_TRACKING_LOST_HOLD_SECS", 4.0) or 4.0)
    return (now - last) < hold


def _start_idle_head_wander(now: float) -> None:
    """Stop staring and pick a short look-around route. The route ends back at the
    pre-wander gaze (where the person was), so he reliably looks back and can re-acquire."""
    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]
    neck_neutral = int(neck_cfg["neutral"])
    lift_neutral = int(lift_cfg["neutral"])
    tilt_neutral = int(tilt_cfg["neutral"])

    start_pose = (
        _current_servo_position("neck"),
        _current_servo_position("headlift"),
        _current_servo_position("headtilt"),
    )
    neck_range = int(getattr(config, "IDLE_HEAD_WANDER_NECK_RANGE_QUS", 2600) or 0)
    lift_range = int(getattr(config, "IDLE_HEAD_WANDER_LIFT_RANGE_QUS", 800) or 0)
    tilt_range = int(getattr(config, "IDLE_HEAD_WANDER_TILT_RANGE_QUS", 200) or 0)
    wp_min = int(getattr(config, "IDLE_HEAD_WANDER_WAYPOINTS_MIN", 2) or 1)
    wp_max = max(wp_min, int(getattr(config, "IDLE_HEAD_WANDER_WAYPOINTS_MAX", 3) or 1))
    count = random.randint(wp_min, wp_max)

    waypoints: list[tuple[int, int, int]] = []
    for _ in range(count):
        waypoints.append((
            _clamp_servo("neck", neck_neutral + random.uniform(-neck_range, neck_range)),
            _clamp_servo("headlift", lift_neutral + random.uniform(-lift_range, lift_range)),
            _clamp_servo("headtilt", tilt_neutral + random.uniform(-tilt_range, tilt_range)),
        ))
    # Final waypoint: look back where the person was, so he re-acquires the face.
    waypoints.append((int(start_pose[0]), int(start_pose[1]), int(start_pose[2])))

    dur = random.uniform(
        float(getattr(config, "IDLE_HEAD_WANDER_MIN_DURATION_SECS", 3.0)),
        float(getattr(config, "IDLE_HEAD_WANDER_MAX_DURATION_SECS", 7.0)),
    )
    with _idle_wander_lock:
        _idle_wander.update({
            "active": True,
            "until": now + dur,
            "waypoints": waypoints,
            "index": 0,
            "reached_at": 0.0,
        })
    # Let go of the lock — he's deliberately looking elsewhere now. Mutate in place
    # (.clear() is atomic under the GIL) rather than reassigning the global, so the
    # face-tracking thread never sees a torn/orphaned _face_tracking_lock.
    _face_tracking_lock.clear()
    _log.info(
        "consciousness: idle head wander started (%d waypoints, %.1fs) — looking around.",
        len(waypoints), dur,
    )


def _finish_idle_head_wander(now: float, *, allow_regreet: bool) -> None:
    with _idle_wander_lock:
        was_active = bool(_idle_wander.get("active"))
        _idle_wander["active"] = False
        _idle_wander["index"] = 0
        _idle_wander["reached_at"] = 0.0
        if was_active:
            _idle_wander["last_at"] = now
            _idle_wander["pending_regreet"] = bool(allow_regreet)
            _idle_wander["regreet_deadline"] = (
                now + float(getattr(config, "IDLE_HEAD_WANDER_REGREET_WINDOW_SECS", 6.0))
                if allow_regreet else 0.0
            )


def _drive_idle_head_wander(servo_mod, now: float) -> None:
    """Called from the face-tracking loop while a wander is active: step the head toward
    the current waypoint, dwell, then advance. Aborts (snapping back to engage) if the
    person starts interacting again or the robot starts speaking/listening."""
    try:
        if getattr(servo_mod, "manual_override_enabled", lambda: False)():
            _finish_idle_head_wander(now, allow_regreet=False)
            return
        # Conversation resumed or Rex is talking → stop wandering and re-engage.
        if (
            getattr(servo_mod, "speech_motion_active", lambda: False)()
            or getattr(servo_mod, "listening_motion_active", lambda: False)()
            or _conversation_idle_secs(now) < 2.0
        ):
            _finish_idle_head_wander(now, allow_regreet=False)
            return

        with _idle_wander_lock:
            until = float(_idle_wander.get("until") or 0.0)
            waypoints = list(_idle_wander.get("waypoints") or [])
            index = int(_idle_wander.get("index") or 0)
            reached_at = float(_idle_wander.get("reached_at") or 0.0)

        if now >= until or index >= len(waypoints):
            _finish_idle_head_wander(now, allow_regreet=True)
            return

        target = waypoints[index]
        names = ("neck", "headlift", "headtilt")
        cur = (
            _current_servo_position("neck"),
            _current_servo_position("headlift"),
            _current_servo_position("headtilt"),
        )
        max_step = int(getattr(config, "IDLE_HEAD_WANDER_MAX_STEP_QUS", 160))
        tol = int(getattr(config, "IDLE_HEAD_WANDER_WAYPOINT_TOLERANCE_QUS", 70))
        updates: dict[int, int] = {}
        reached = True
        next_pose = {}
        for name, c_, t_ in zip(names, cur, target):
            nxt = _limited_tracking_step(name, c_, int(t_), max_step)
            next_pose[name] = nxt
            if abs(nxt - c_) >= 2:
                updates[int(config.SERVO_CHANNELS[name]["ch"])] = nxt
            if abs(int(t_) - c_) > tol:
                reached = False

        if updates:
            try:
                servo_mod.set_motion_profile(
                    list(updates.keys()),
                    speed=int(getattr(config, "IDLE_HEAD_WANDER_SERVO_SPEED", 35)),
                    acceleration=int(getattr(config, "IDLE_HEAD_WANDER_SERVO_ACCELERATION", 8)),
                )
            except Exception:
                pass
            servo_mod.set_servos(updates)
            # Keep breathing/speech orbit anchored to where he's actually looking.
            try:
                servo_mod.set_face_tracking_baseline(
                    neck=next_pose["neck"], lift=next_pose["headlift"], tilt=next_pose["headtilt"],
                )
            except Exception:
                pass

        _record_face_tracking_state(locked=False, visible=False)

        if reached:
            dwell = float(getattr(config, "IDLE_HEAD_WANDER_DWELL_SECS", 1.0))
            with _idle_wander_lock:
                if float(_idle_wander.get("reached_at") or 0.0) <= 0.0:
                    _idle_wander["reached_at"] = now
                elif now - float(_idle_wander["reached_at"]) >= dwell:
                    _idle_wander["index"] = int(_idle_wander.get("index") or 0) + 1
                    _idle_wander["reached_at"] = 0.0
    except Exception as exc:
        _log.debug("idle head wander drive error: %s", exc)
        _finish_idle_head_wander(now, allow_regreet=False)


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


def _mood_rest_bias() -> tuple[int, int]:
    """(headlift_delta, headtilt_delta) the current body mood adds to the RESTING head
    pose — where the head settles when no face is actively being centered. Riding the
    rest pose (not the live centering output) means mood never fights face-tracking.
    Clamped to config offsets; gated + failure-safe (0,0 when disabled/unavailable)."""
    try:
        from intelligence import body_mood
        if not body_mood.enabled():
            return (0, 0)
        lift_d, tilt_d = body_mood.head_bias()
    except Exception:
        return (0, 0)
    max_lift = int(getattr(config, "BODY_MOOD_REST_MAX_LIFT_OFFSET_QUS", 1100) or 0)
    max_tilt = int(getattr(config, "BODY_MOOD_REST_MAX_TILT_OFFSET_QUS", 320) or 0)
    lift_d = max(-max_lift, min(max_lift, int(lift_d)))
    tilt_d = max(-max_tilt, min(max_tilt, int(tilt_d)))
    return (lift_d, tilt_d)


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
    # The mood bias can express posture even before an adaptive rest pose is learned,
    # so proceed when EITHER a learned rest exists OR a mood bias is active.
    bias_lift, bias_tilt = _mood_rest_bias()
    has_learned_rest = _adaptive_head_rest_enabled() and int(_adaptive_head_rest.get("samples") or 0) > 0
    if not has_learned_rest and bias_lift == 0 and bias_tilt == 0:
        return False
    delay = float(getattr(config, "FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS", 0.8) or 0.0)
    if lost_age_secs is not None and lost_age_secs < max(0.0, delay):
        return False
    try:
        if getattr(servo_mod, "manual_override_enabled", lambda: False)():
            return False
        if getattr(servo_mod, "speech_motion_active", lambda: False)():
            return False
        if getattr(servo_mod, "listening_motion_active", lambda: False)():
            return False
    except Exception:
        return False

    target_lift, target_tilt = _adaptive_head_rest_target()
    # Compose the mood posture onto the settling target (clamped to servo limits).
    target_lift = _clamp_servo("headlift", target_lift + bias_lift)
    target_tilt = _clamp_servo("headtilt", target_tilt + bias_tilt)
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


def _maybe_play_idle_mood_gesture(now: float, speech_active: bool, listening_active: bool) -> None:
    """Occasionally punctuate the current body mood with a brief idle gesture (a proud
    bounce, a giddy wiggle, a suspicious side-eye) when Rex is otherwise idle. Reuses the
    existing body-beat system (non-blocking lock → never fights speech). Cooldown + chance
    gated so it's a garnish, not a tic."""
    global _last_mood_gesture_at
    if not bool(getattr(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", True)):
        return
    if speech_active or listening_active:
        return
    cooldown = float(getattr(config, "BODY_MOOD_IDLE_GESTURE_COOLDOWN_SECS", 25.0))
    if now - _last_mood_gesture_at < cooldown:
        return
    try:
        from intelligence import body_mood
        beat = body_mood.idle_beat()
    except Exception:
        return
    if not beat:
        return
    if random.random() > float(getattr(config, "BODY_MOOD_IDLE_GESTURE_CHANCE", 0.35)):
        return
    try:
        from sequences import animations
        if animations.play_body_beat(beat):
            _last_mood_gesture_at = now
            _log.debug("consciousness: idle mood gesture %r fired", beat)
    except Exception as exc:
        _log.debug("idle mood gesture failed: %s", exc)


def _step_mood_expression(snapshot: dict, profile: "SituationProfile") -> None:
    """Express Rex's sustained body mood on the channels NOT owned by face-tracking:
    visor openness + breathing cadence, plus an occasional idle mood gesture. Yields to
    speech / listening / manual override; the head-pitch side of mood rides the rest pose
    (see _mood_rest_bias) so this never fights the face-centering controller.

    Visor + breathing are set-and-forget channels, so when a mood DECAYS we explicitly
    RELEASE them back to neutral (otherwise the visor would stay open / breathing stay
    excited until the next speech). The head rest-bias needs no release — it's recomputed
    from the current mood each rest tick and returns to neutral on its own."""
    global _mood_owns_visor, _last_mood_breathing
    try:
        from intelligence import body_mood
        if not body_mood.enabled():
            return
        from hardware import servos as servo_mod

        try:
            if servo_mod.manual_override_enabled():
                return
        except Exception:
            return
        try:
            speech_active = bool(servo_mod.speech_motion_active())
            listening_active = bool(servo_mod.listening_motion_active())
        except Exception:
            speech_active = listening_active = False

        now = time.monotonic()

        # Visor + breathing are Rex's free expressive channels when he isn't speaking or
        # listening (those own the visor flutter / breathing cadence themselves).
        if not speech_active and not listening_active:
            visor_ch = int(config.SERVO_CHANNELS["visor"]["ch"])
            try:
                target = body_mood.visor_target()
            except Exception:
                target = None
            if target is None and _mood_owns_visor:
                # Mood ended → release the visor back to its lens-clear resting position.
                # MUST be the lens-clear floor (VISOR_HALF), NOT the servo neutral (6000),
                # which sits below the floor and would partially cover the camera lens.
                target = int(body_mood.visor_lens_clear_floor())
                _mood_owns_visor = False
            elif target is not None:
                _mood_owns_visor = True
            if target is not None:
                try:
                    servo_mod.set_motion_profile(
                        [visor_ch],
                        speed=int(getattr(config, "BODY_MOOD_VISOR_SERVO_SPEED", 30)),
                        acceleration=int(getattr(config, "BODY_MOOD_VISOR_SERVO_ACCELERATION", 8)),
                    )
                except Exception:
                    pass
                try:
                    servo_mod.set_servo(visor_ch, int(target))
                except Exception as exc:
                    _log.debug("mood visor set failed: %s", exc)
            try:
                breath = body_mood.breathing_emotion()
                desired = breath or ("neutral" if _last_mood_breathing not in (None, "neutral") else None)
                if desired is not None and desired != _last_mood_breathing:
                    servo_mod.set_breathing_emotion(desired)
                    _last_mood_breathing = desired
            except Exception as exc:
                _log.debug("mood breathing set failed: %s", exc)

        _maybe_play_idle_mood_gesture(now, speech_active, listening_active)
    except Exception as exc:
        _log.debug("mood expression step error: %s", exc)


def _locked_person_name(snapshot: dict) -> Optional[str]:
    """Display name of the currently face-locked person, from the snapshot."""
    pid = _face_tracking_lock.get("person_id")
    if not isinstance(pid, int):
        key = _face_tracking_lock.get("key")
        pid = key if isinstance(key, int) else None
    if not isinstance(pid, int):
        return None
    for person in (snapshot.get("people") or []):
        if _person_db_id(person) == pid:
            return person.get("face_id") or person.get("name")
    return None


def _maybe_fire_wander_regreet(snapshot: dict, profile: "SituationProfile") -> None:
    """After a wander, Rex's gaze has landed back on the person — say a short, dry line
    acknowledging he drifted off and noticed them again. Respects proactive suppression."""
    if getattr(profile, "suppress_proactive", False):
        return
    # The physical look-around is fine mid-conversation, but the SPOKEN "Oh—still
    # here" re-greet during a live exchange is an interruption (it fired twice
    # mid-conversation, live-logged 2026-06-18). Suppress only the spoken line
    # while a conversation is active; the silent wander motion still happens.
    if getattr(profile, "conversation_active", False) or getattr(
        profile, "rapid_exchange", False
    ):
        return
    # profile.conversation_active forgets fast — a ~33s pause mid-conversation
    # read as inactive and the re-greet spoke "Oh—still here" TWICE in one
    # 3-minute exchange (field 2026-07-18 01:10). Direct recency check: if the
    # person spoke within this window, the wander stays a silent head motion.
    try:
        from intelligence import interaction as _intx
        _recent = float(getattr(config, "IDLE_REGREET_MIN_USER_SILENCE_SECS", 180.0))
        if (_intx._last_user_content_at > 0.0
                and (time.monotonic() - _intx._last_user_content_at) < _recent):
            return
    except Exception:
        pass
    name = _locked_person_name(snapshot)
    first = _first_name(name, "there")
    pid = _face_tracking_lock.get("person_id")
    tag_key = pid if isinstance(pid, int) else "idle_wander"
    prompt = (
        f"The conversation had gone quiet, so you idly looked around the room for a few "
        f"seconds — and now your gaze settles back on '{first}', who's still right there. "
        f"Say ONE short, dry, in-character Rex line that acknowledges you drifted off and "
        f"noticed them again: a little 'oh — still here', mildly caught-out or amused, "
        f"NOT needy and NOT a fresh interview question. One line only."
    )
    queued = _generate_and_speak_presence(
        prompt,
        label=f"idle-wander re-greet for {name or 'someone'}",
        tag_key=tag_key,
        emotion="curious",
        purpose="presence_reaction",
    )
    if queued:
        _log.info("consciousness: idle-wander re-greet fired for %s", name or "someone")


def _step_idle_head_wander(snapshot: dict, profile: "SituationProfile") -> None:
    """Give Rex a wandering attention of his own: when the conversation lulls while he's
    still locked on a face, occasionally look around the room, then return his gaze and
    maybe re-greet. Start/decision logic lives here (1Hz); the face-tracking loop drives
    the actual motion while a wander is active."""
    if not _idle_head_wander_enabled():
        return
    try:
        now = time.monotonic()
        with _idle_wander_lock:
            active = bool(_idle_wander.get("active"))
            pending = bool(_idle_wander.get("pending_regreet"))
            last_at = float(_idle_wander.get("last_at") or 0.0)
            regreet_deadline = float(_idle_wander.get("regreet_deadline") or 0.0)
            until = float(_idle_wander.get("until") or 0.0)

        # A wander is in progress — the face-tracking loop owns the motion. BACKSTOP: if
        # that loop can't drive/finish it (asleep, tracking suspended by a directed gaze,
        # camera frames missing) the wander would otherwise hang, so end it here when Rex
        # is asleep or the wander is well past its own deadline. The head can never get
        # stuck looking away.
        if active:
            overdue = now >= until + float(getattr(config, "IDLE_HEAD_WANDER_STALL_GRACE_SECS", 3.0))
            if state_module.get_state() == State.SLEEP or overdue:
                _finish_idle_head_wander(now, allow_regreet=False)
            return

        # A wander just finished: when his gaze re-acquires the face, maybe re-greet.
        if pending:
            if _face_tracking_has_fresh_lock(now):
                with _idle_wander_lock:
                    _idle_wander["pending_regreet"] = False
                if random.random() < float(getattr(config, "IDLE_HEAD_WANDER_REGREET_CHANCE", 0.4)):
                    _maybe_fire_wander_regreet(snapshot, profile)
                # else: he just keeps looking — no comment.
            elif now >= regreet_deadline:
                with _idle_wander_lock:
                    _idle_wander["pending_regreet"] = False
            return

        # Decide whether to START a wander.
        if state_module.get_state() == State.SLEEP:
            return
        if is_waiting_for_response():
            return
        if directed_gaze_hold_active(now):
            return
        if _within_startup_group_window(now) and not _greeted_this_session:
            return
        if _startup_known_greeting_pending(snapshot):
            return
        try:
            from hardware import servos as servo_mod
            if servo_mod.manual_override_enabled():
                return
            if servo_mod.speech_motion_active() or servo_mod.listening_motion_active():
                return
        except Exception:
            return
        # Only when he's actually fixed on someone and the talk has gone quiet.
        if not _face_tracking_has_fresh_lock(now):
            return
        if _conversation_idle_secs(now) < float(getattr(config, "IDLE_HEAD_WANDER_IDLE_SECS", 18.0)):
            return
        if now - last_at < float(getattr(config, "IDLE_HEAD_WANDER_COOLDOWN_SECS", 30.0)):
            return
        if random.random() < float(getattr(config, "IDLE_HEAD_WANDER_CHANCE", 0.25)):
            _start_idle_head_wander(now)
    except Exception as exc:
        _log.debug("idle head wander step error: %s", exc)


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
    if intent.get("unknown_voice"):
        return candidate.get("person_id") is None
    # "Find anyone" intent (startup / acquisition scan: no specific person, no
    # unknown voice) — ANY visible face satisfies it. Without this, the startup
    # room scan could never be marked acquired, so search_requested stayed armed
    # for the full search window even after Rex locked onto and GREETED someone;
    # the next ≥0.45s face-detection blip (routine for a seated person at the
    # frame edge) relaunched full-room waypoint snaps mid-conversation — the
    # "greets me, then immediately looks around wildly" live failure.
    return True


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

    # Inline fallbacks mirror the config defaults so config stays the one source of truth.
    neck_fraction = float(getattr(config, "SPEAKER_GAZE_SEARCH_NECK_FRACTION", 1.0))
    down_tilt_fraction = float(getattr(config, "SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION", 0.65))
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

    # Per-waypoint cadence is "snap to the pose, then HOLD STILL" so the head
    # actually stops moving long enough for dlib (which only runs on the ~1 Hz
    # cognition loop) to get a couple of clean frames and lock a face. Each
    # waypoint gets a short SETTLE (servo move finishing) followed by a longer
    # DWELL (no servo command at all — the camera is steady). Only once the dwell
    # elapses do we advance to the next waypoint.
    settle = float(getattr(config, "SPEAKER_GAZE_SEARCH_SETTLE_SECS", 0.4) or 0.0)
    dwell = float(getattr(config, "SPEAKER_GAZE_SEARCH_DWELL_SECS", 2.0) or 0.0)
    hold_total = max(0.1, settle + dwell)
    with _speaker_gaze_lock:
        if not _speaker_gaze_intent:
            return None
        committed_at = float(_speaker_gaze_intent.get("waypoint_committed_at") or 0.0)
        if committed_at > 0.0 and (now - committed_at) < hold_total:
            # Still settling/dwelling at the current waypoint — keep the head put.
            return _speaker_gaze_intent.get("waypoint_pose")
        plan = _speaker_gaze_intent.get("search_plan")
        plan_idx = int(_speaker_gaze_intent.get("search_plan_index") or 0)
        if not plan or plan_idx >= len(plan):
            # Fresh randomized pass (also re-rolls if the search outlasts one pass).
            plan = _build_speaker_gaze_search_plan(_speaker_gaze_intent.get("reason", "startup"))
            _speaker_gaze_intent["search_plan"] = plan
            plan_idx = 0
        neck_frac, vert_frac = plan[plan_idx]
        pose = _speaker_gaze_search_label(neck_frac, vert_frac)
        _speaker_gaze_intent["search_plan_index"] = plan_idx + 1
        _speaker_gaze_intent["search_index"] = int(_speaker_gaze_intent.get("search_index") or 0) + 1
        _speaker_gaze_intent["waypoint_committed_at"] = now
        _speaker_gaze_intent["waypoint_pose"] = pose
        _speaker_gaze_intent["last_search_at"] = now
        if not _speaker_gaze_intent.get("search_started_at"):
            _speaker_gaze_intent["search_started_at"] = now

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


# ── Greet-at-their-height (Part C) ──────────────────────────────────────────────

def _greeting_height_targets(vertical: str) -> tuple[int, int]:
    """Map a person's vertical position in frame to (headlift, headtilt) targets.

    "low" → head drops toward its minimum (greet a seated/short/low person where
    they are); "high" → head rises toward its maximum (meet a standing/tall person);
    "center" → neutral. Headtilt is inverted (larger = looking down) and gets a
    gentle matching nudge so the camera points the way the head leans.
    """
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]
    lift_neutral, lift_min, lift_max = int(lift_cfg["neutral"]), int(lift_cfg["min"]), int(lift_cfg["max"])
    tilt_neutral, tilt_min, tilt_max = int(tilt_cfg["neutral"]), int(tilt_cfg["min"]), int(tilt_cfg["max"])

    v = str(vertical or "center").strip().lower()
    if v == "low":
        frac = float(getattr(config, "GREET_HEIGHT_LOW_LIFT_FRACTION", 0.88))
        lift = lift_neutral - frac * (lift_neutral - lift_min)
        tilt = tilt_neutral + 0.5 * (tilt_max - tilt_neutral)   # look down
    elif v == "high":
        frac = float(getattr(config, "GREET_HEIGHT_HIGH_LIFT_FRACTION", 0.85))
        lift = lift_neutral + frac * (lift_max - lift_neutral)
        tilt = tilt_neutral - 0.4 * (tilt_neutral - tilt_min)   # look up a touch
    else:
        lift, tilt = lift_neutral, tilt_neutral
    return _clamp_servo("headlift", lift), _clamp_servo("headtilt", tilt)


def _apply_greeting_height(vertical: str) -> None:
    """Pin head-lift to greet a person at their physical height and hold it.

    Sets the lift/tilt now, makes it the face-tracking baseline (so breathing/speech
    orbit this height instead of recentering), and arms a directed-gaze hold so the
    speaker scan, idle wander, and adaptive-rest drift all stand down while the head
    holds still long enough for dlib to lock. The instant a face is locked, normal
    face tracking takes over and fine-centers on it (neck-tilt stays the fine axis).
    """
    if not bool(getattr(config, "GREET_HEIGHT_ENABLED", True)):
        return
    try:
        from hardware import servos as servo_mod
    except Exception:
        return

    lift, tilt = _greeting_height_targets(vertical)
    lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
    tilt_ch = int(config.SERVO_CHANNELS["headtilt"]["ch"])
    try:
        servo_mod.set_motion_profile(
            [lift_ch, tilt_ch],
            speed=int(getattr(config, "GREET_HEIGHT_SERVO_SPEED", 90)),
            acceleration=int(getattr(config, "SPEAKER_GAZE_SEARCH_SERVO_ACCELERATION", 20)),
        )
    except Exception as exc:
        _log.debug("greeting height motion profile failed: %s", exc)
    try:
        servo_mod.set_servos({lift_ch: lift, tilt_ch: tilt})
    except Exception as exc:
        _log.debug("greeting height set_servos failed: %s", exc)
    try:
        servo_mod.set_face_tracking_baseline(lift=lift, tilt=tilt)
    except Exception as exc:
        _log.debug("greeting height baseline failed: %s", exc)

    # The hold (not adaptive rest — its ±offset clamp can't reach a deep-low pose)
    # is what keeps the head steady at the greeting height while dlib tries to lock.
    hold_secs = float(getattr(config, "GREET_HEIGHT_HOLD_SECS", 6.0))
    hold_directed_gaze("hold", secs=hold_secs)
    _log.info("[greet_height] vertical=%s lift=%d tilt=%d hold=%.1fs", vertical, lift, tilt, hold_secs)


def _vertical_from_box(candidate: Optional[dict], frame_h: int) -> str:
    """Classify a visible face's vertical position in frame as low/center/high."""
    try:
        center = (candidate or {}).get("center")
        if not center or frame_h <= 0:
            return "center"
        vy = float(center[1]) / float(frame_h)
    except (TypeError, ValueError, IndexError):
        return "center"
    if vy < 0.4:
        return "high"
    if vy > 0.66:
        return "low"
    return "center"


# ── Startup-only OpenAI presence fallback (Part B) ──────────────────────────────

# Head directions (neck_frac, vert_frac) the fallback sweeps, reusing the search
# pose resolver. Down-biased — people usually sit/stand below the head camera.
_PRESENCE_FALLBACK_DIRECTIONS: list[tuple] = [
    (None, 0.95),   # straight down, no turn
    (-0.6, 0.8),    # down-left
    (0.6, 0.8),     # down-right
    (0.0, 0.3),     # near level, centered
]

_PRESENCE_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _presence_min_confidence_ok(confidence) -> bool:
    want = str(getattr(config, "STARTUP_OPENAI_PRESENCE_MIN_CONFIDENCE", "medium") or "medium").strip().lower()
    have = str(confidence or "low").strip().lower()
    return _PRESENCE_CONFIDENCE_ORDER.get(have, 0) >= _PRESENCE_CONFIDENCE_ORDER.get(want, 1)


def _run_openai_presence_fallback() -> None:
    """Worker: sweep a few directions and ask the vision model whether anyone is
    there before Rex declares the room empty. On the first confident hit, record
    presence evidence (suppress the false empty-room line), bump the crowd count,
    and steer the head to greet the person at their height. Runs on its own daemon
    thread because each vision call is ~1 s and must not block the servo/cognition
    loops."""
    global _startup_presence_fallback_active, _startup_openai_verified_empty_at
    try:
        from hardware import servos as servo_mod
        from vision import camera as camera_mod
        from vision import scene as scene_mod
    except Exception as exc:
        _log.debug("presence fallback imports failed: %s", exc)
        _startup_presence_fallback_active = False
        return

    try:
        if not bool(getattr(config, "STARTUP_OPENAI_PRESENCE_FALLBACK_ENABLED", True)):
            return
        # No API key → skip the head moves entirely (locate_people would only ever
        # return "nobody", and we'd waste servo travel + failed calls).
        try:
            import apikeys
            if not getattr(apikeys, "OPENAI_API_KEY", None):
                _log.info("[presence_fallback] no OpenAI key — skipping verification sweep")
                return
        except Exception:
            return

        max_dirs = max(1, int(getattr(config, "STARTUP_OPENAI_PRESENCE_MAX_DIRECTIONS", 4)))
        directions = _PRESENCE_FALLBACK_DIRECTIONS[:max_dirs]
        settle = float(getattr(config, "STARTUP_OPENAI_PRESENCE_SETTLE_SECS", 0.35))
        neck_ch = int(config.SERVO_CHANNELS["neck"]["ch"])
        lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
        tilt_ch = int(config.SERVO_CHANNELS["headtilt"]["ch"])

        for neck_frac, vert_frac in directions:
            now = time.monotonic()
            # Bail if conditions changed under us: startup window closed, dlib/another
            # signal already proved presence, or the user took the gaze.
            if not _within_startup_group_window(now):
                return
            if _last_face_seen_at > 0.0 or _startup_presence_evidence_at > 0.0:
                return
            if directed_gaze_hold_active(now):
                return
            try:
                if servo_mod.manual_override_enabled() or servo_mod.speech_motion_active():
                    return
            except Exception:
                pass
            if not camera_mod.has_recent_frame(2.0):
                return

            targets = _speaker_gaze_search_targets(neck_frac, vert_frac)
            try:
                servo_mod.set_motion_profile(
                    list(targets.keys()),
                    speed=int(getattr(config, "SPEAKER_GAZE_SEARCH_SERVO_SPEED", 130)),
                    acceleration=int(getattr(config, "SPEAKER_GAZE_SEARCH_SERVO_ACCELERATION", 20)),
                )
            except Exception as exc:
                _log.debug("presence fallback motion profile failed: %s", exc)
            servo_mod.set_servos(targets)
            try:
                servo_mod.set_face_tracking_baseline(
                    neck=targets.get(neck_ch),
                    lift=targets.get(lift_ch),
                    tilt=targets.get(tilt_ch),
                )
            except Exception:
                pass

            frame = camera_mod.capture_current_gaze(settle_secs=settle)
            if frame is None:
                continue
            result = scene_mod.locate_people(frame)
            if result.get("present") and _presence_min_confidence_ok(result.get("confidence")):
                _note_startup_presence_evidence("openai_vision")
                try:
                    scene_mod._update_crowd_count(int(result.get("count") or 1))
                except Exception as exc:
                    _log.debug("presence fallback crowd bump failed: %s", exc)
                _log.info(
                    "[presence_fallback] person found vertical=%s posture=%s conf=%s — greeting at height",
                    result.get("vertical"), result.get("posture"), result.get("confidence"),
                )
                _apply_greeting_height(str(result.get("vertical") or "center"))
                return

        # Swept every direction and found nobody — the empty room is now verified,
        # so the eventual "no organics" line is truthful rather than a missed lock.
        _startup_openai_verified_empty_at = time.monotonic()
        _log.info(
            "[presence_fallback] swept %d direction(s), no person — empty room verified",
            len(directions),
        )
    except Exception as exc:
        _log.debug("presence fallback worker error: %s", exc)
    finally:
        _startup_presence_fallback_active = False


def _step_startup_presence_fallback_trigger(snapshot: dict) -> None:
    """Spawn the OpenAI presence fallback once, after the dlib startup scan has had
    its full chance and still found nobody — before Rex says the room is empty."""
    global _startup_presence_fallback_started, _startup_presence_fallback_active

    if _startup_presence_fallback_started:
        return
    if not bool(getattr(config, "STARTUP_OPENAI_PRESENCE_FALLBACK_ENABLED", True)):
        return
    now = time.monotonic()
    if _process_started_mono <= 0.0 or not _within_startup_group_window(now):
        return
    # Only after the (now longer, dwelled) dlib scan has fully run.
    search_window = float(getattr(config, "SPEAKER_GAZE_SEARCH_WINDOW_SECS", 13.5) or 0.0)
    if (now - _process_started_mono) < (search_window + 0.5):
        return
    if _last_face_seen_at > 0.0 or _startup_presence_evidence_at > 0.0:
        return
    if not _room_looks_empty(snapshot):
        return
    if directed_gaze_hold_active(now):
        return

    _startup_presence_fallback_started = True
    _startup_presence_fallback_active = True
    try:
        threading.Thread(
            target=_run_openai_presence_fallback,
            name="startup-presence-fallback",
            daemon=True,
        ).start()
        _log.info("[presence_fallback] dlib scan found nobody — starting OpenAI verification sweep")
    except Exception as exc:
        _startup_presence_fallback_active = False
        _log.debug("presence fallback spawn failed: %s", exc)


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
    directed_hold: bool = False,
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
            "directed_hold": bool(directed_hold),
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


def _evaluate_face_jump(
    cx, cy, key, now, frame_w, frame_h, last_center, pending_center,
    *, identified=False, live_tracked=False,
):
    """Decide whether a freshly-detected face box is an implausible single-tick teleport
    (a spurious detector box) that should be ignored so the head holds its gaze.

    The rule is "random UNKNOWN face detected far away = noise": a box that dlib
    freshly identity-matched to a known enrolled person is accepted no matter how
    far it jumped — clutter can't match a face encoding, and a seated/leaning person
    legitimately appears anywhere in the frame (including the bottom edge). The
    instant-accept does NOT apply to live_tracked boxes: those inherit their
    person_id from an older recognition pass via the correlation tracker, so a
    drifted tracker box is not identity evidence for THIS position.

    Pure/deterministic (reads config only) so it can be unit-tested. Returns
    (accept, last_center, pending_center): when ``accept`` is False the caller should
    hold the current gaze this tick. An unknown big jump is accepted only once the
    jumped-to position has persisted for FACE_TRACKING_JUMP_CONFIRM_SECS (a genuine
    fast move). A rejection refreshes the reference timestamp so the same spurious
    box can't sneak in moments later through reference staleness — persistence (or
    identity) is the only way in."""
    jump_frac = float(getattr(config, "FACE_TRACKING_MAX_JUMP_FRAC", 0.0) or 0.0)
    if jump_frac <= 0.0 or frame_w <= 0 or frame_h <= 0:
        return True, {"key": key, "cx": cx, "cy": cy, "at": now}, None
    max_jump = jump_frac * ((frame_w * frame_w + frame_h * frame_h) ** 0.5)
    max_age = float(getattr(config, "FACE_TRACKING_JUMP_MAX_AGE_SECS", 5.0))
    fresh_same = (
        last_center is not None
        and last_center.get("key") == key
        and (now - float(last_center.get("at", 0.0))) <= max_age
    )
    if not fresh_same:
        # No fresh reference for this face (first lock / re-acquire / changed target).
        return True, {"key": key, "cx": cx, "cy": cy, "at": now}, None
    jump_dist = (
        (cx - float(last_center["cx"])) ** 2 + (cy - float(last_center["cy"])) ** 2
    ) ** 0.5
    if jump_dist <= max_jump:
        return True, {"key": key, "cx": cx, "cy": cy, "at": now}, None
    diag = (frame_w * frame_w + frame_h * frame_h) ** 0.5
    # A freshly identity-matched box that moved a MODERATE distance is really them
    # (they sat down / leaned / stood) — follow immediately. But a VERY LARGE identified
    # jump is treated with suspicion: dlib does occasionally false-match a transient
    # ghost (a reflection, a high-contrast blob, a face-like region above the seated
    # person) to a known face, and a 1-2 tick ghost would otherwise yank the head to it
    # and snap back (live-logged: head jerking up to a phantom box above a person seated
    # low in frame). So a big identified jump must still PERSIST briefly — shorter than
    # an unknown, since identity is partial evidence — before it's chased.
    if identified and not live_tracked:
        identified_max = float(
            getattr(config, "FACE_TRACKING_IDENTIFIED_INSTANT_JUMP_FRAC", 0.22)
        ) * diag
        if jump_dist <= identified_max:
            return True, {"key": key, "cx": cx, "cy": cy, "at": now}, None
        confirm_secs = float(getattr(config, "FACE_TRACKING_IDENTIFIED_JUMP_CONFIRM_SECS", 0.25))
    else:
        confirm_secs = float(getattr(config, "FACE_TRACKING_JUMP_CONFIRM_SECS", 0.5))
    # Big jump — accept only if the jumped-to position has been holding (a genuine fast
    # move / new arrival), never via reference staleness.
    near_pending = pending_center is not None and (
        ((cx - float(pending_center["cx"])) ** 2 + (cy - float(pending_center["cy"])) ** 2) ** 0.5
    ) <= max_jump
    if near_pending and (now - float(pending_center.get("since", now))) >= confirm_secs:
        return True, {"key": key, "cx": cx, "cy": cy, "at": now}, None
    new_pending = pending_center if near_pending else {"cx": cx, "cy": cy, "since": now}
    # Keep the reference alive while rejecting: holding the gaze means the held
    # position is still where we believe the person is. Without this, the reference
    # aged past max_age within a tick or two and the SAME spurious box was accepted
    # unconditionally (live failure: rejected (938,838) then accepted (937,835) 1s
    # later, driving the lift servo to its rails chasing clutter).
    refreshed_last = dict(last_center)
    refreshed_last["at"] = now
    return False, refreshed_last, new_pending


def _maybe_log_face_jump_reject(cx, cy, now) -> None:
    """Rate-limited note that the head is ignoring a teleporting detector box."""
    global _face_tracking_last_jump_log_at
    if (now - _face_tracking_last_jump_log_at) >= 2.0:
        _face_tracking_last_jump_log_at = now
        _log.info(
            "[face_tracking] ignoring implausible jump to (%.0f,%.0f) — holding gaze "
            "(likely a spurious detector box on clutter)",
            cx, cy,
        )


# ── Human-like gaze rhythm (intelligence/gaze_engine.py) ─────────────────────
# A stochastic ON-target / OFF-target eye-contact duty cycle (the 50/70 rule)
# layered on top of the closed-loop face-centering below. Because the camera is on
# the head, an OFF-target "look-away" carries the face out of frame, so — exactly
# like the idle head-wander — the gaze layer drives the head AWAY to an aversion
# pose and then back to a captured anchor, after which face-centering re-acquires.
# The brain (gaze_engine) is pure; this is its only live actuator, and it routes
# through the same single servo writer (no second thread fighting the head).
_gaze_drive: dict = {"phase": "idle"}  # phase: idle | away | returning


def _capture_gaze_anchor(decision) -> None:
    """Remember where the head was looking (the on-target face gaze) before averting,
    so the return lands back on the partner and centering can re-lock."""
    _gaze_drive["phase"] = "away"
    _gaze_drive["anchor"] = (
        _current_servo_position("neck"),
        _current_servo_position("headlift"),
        _current_servo_position("headtilt"),
    )
    _gaze_drive["return_since"] = 0.0
    _gaze_drive["segment"] = getattr(decision, "segment_id", None)
    # Start the look-away from rest so it RAMPS in (soft ease-in), not a snap.
    _gaze_drive["vel"] = {}


def _gaze_release() -> None:
    if _gaze_drive.get("phase") != "idle":
        _gaze_drive["phase"] = "idle"
        _gaze_drive["return_since"] = 0.0
    _gaze_drive["vel"] = {}


def _drive_gaze_aversion(servo_mod, now: float, decision, anchor: tuple) -> None:
    """Drive the head to the engine's aversion pose: a relative YAW offset from the
    captured gaze anchor, an absolute PITCH (up=visualizing / down=internalizing) and
    a POLE (head-height) engagement bias. Fast but slew-clamped; mirrors the wander
    driver so it shares the proven single-writer + baseline-update pattern."""
    try:
        cfg = gaze_engine.get_engine().cfg
    except Exception:
        return
    neck_ch = int(config.SERVO_CHANNELS["neck"]["ch"])
    lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
    tilt_ch = int(config.SERVO_CHANNELS["headtilt"]["ch"])
    anchor_neck, anchor_lift, _anchor_tilt = anchor

    neck_off = cfg.yaw_deg_to_neck_qus(decision.yaw_offset_deg) - cfg.neck_neutral
    target_neck = _clamp_servo("neck", anchor_neck + neck_off)
    target_tilt = _clamp_servo("headtilt", cfg.pitch_deg_to_tilt_qus(decision.pitch_offset_deg))
    target_lift = _clamp_servo("headlift", anchor_lift + cfg.pole_bias_qus(decision.pole_mm))

    _step_gaze_targets(
        servo_mod, target_neck, target_lift, target_tilt,
        neck_step=int(getattr(config, "GAZE_AVERSION_NECK_MAX_STEP_QUS", 240)),
        lift_step=int(getattr(config, "GAZE_AVERSION_LIFT_MAX_STEP_QUS", 70)),
        tilt_step=int(getattr(config, "GAZE_AVERSION_TILT_MAX_STEP_QUS", 130)),
    )
    _record_face_tracking_state(locked=False, visible=False)


def _drive_gaze_return(servo_mod, now: float) -> bool:
    """Step the head back toward the captured anchor (where the partner is). Returns
    True when home (within tolerance) or after a stall deadline, so the head can never
    get stuck looking away; face-centering then takes over the fine correction."""
    anchor = _gaze_drive.get("anchor")
    if not anchor:
        return True
    if not _gaze_drive.get("return_since"):
        _gaze_drive["return_since"] = now
    target_neck, target_lift, target_tilt = anchor
    reached = _step_gaze_targets(
        servo_mod, int(target_neck), int(target_lift), int(target_tilt),
        neck_step=int(getattr(config, "GAZE_AVERSION_NECK_MAX_STEP_QUS", 240)),
        lift_step=int(getattr(config, "GAZE_AVERSION_LIFT_MAX_STEP_QUS", 70)),
        tilt_step=int(getattr(config, "GAZE_AVERSION_TILT_MAX_STEP_QUS", 130)),
        tolerance=int(getattr(config, "FACE_TRACKING_NECK_MAX_STEP_QUS", 420)),
    )
    _record_face_tracking_state(locked=False, visible=False)
    stalled = (now - float(_gaze_drive.get("return_since") or now)) >= 1.2
    return bool(reached or stalled)


def _gaze_ramped_step(name: str, current: int, target: int, max_vel: int, accel: float) -> tuple[int, float]:
    """One velocity+acceleration-limited tick toward ``target`` (qus). Returns
    (next_position, velocity). The per-axis velocity is carried in ``_gaze_drive['vel']``
    so a look-away eases IN from rest (accel-limited) and eases OUT near the target
    (the desired velocity shrinks with the remaining distance) — a soft drift, not a
    constant-speed snap."""
    vel = _gaze_drive.setdefault("vel", {})
    prev_v = float(vel.get(name, 0.0))
    delta = float(int(target) - int(current))
    max_vel = max(1.0, float(max_vel))
    desired = max(-max_vel, min(max_vel, delta))  # close the gap, capped at top speed
    if desired > prev_v + accel:
        v = prev_v + accel
    elif desired < prev_v - accel:
        v = prev_v - accel
    else:
        v = desired
    nxt = current + v
    if (v > 0 and nxt > target) or (v < 0 and nxt < target):  # don't overshoot
        nxt = float(target)
        v = nxt - current
    nxt_clamped = _clamp_servo(name, nxt)
    vel[name] = v
    return int(round(nxt_clamped)), v


def _step_gaze_targets(
    servo_mod, target_neck: int, target_lift: int, target_tilt: int,
    *, neck_step: int, lift_step: int, tilt_step: int, tolerance: int = 0,
) -> bool:
    """Ramp the three head channels one tick toward the targets (soft ease-in/out via
    a per-axis velocity); return True when all are within ``tolerance``. The
    ``*_step`` args are the per-tick velocity CAP (top speed); the acceleration is that
    cap divided by GAZE_AVERSION_RAMP_TICKS. Updates the gaze baseline so breathing
    orbits the new pose (matches the wander/rest drivers)."""
    ramp = max(1.0, float(getattr(config, "GAZE_AVERSION_RAMP_TICKS", 6.0)))
    specs = (
        ("neck", int(target_neck), int(neck_step)),
        ("headlift", int(target_lift), int(lift_step)),
        ("headtilt", int(target_tilt), int(tilt_step)),
    )
    next_pose: dict[str, int] = {}
    updates: dict[int, int] = {}
    reached = True
    tol = max(2, int(tolerance))
    for name, target, max_vel in specs:
        cur = _current_servo_position(name)
        nxt, _v = _gaze_ramped_step(name, cur, target, max_vel, max(1.0, max_vel / ramp))
        next_pose[name] = nxt
        if abs(nxt - cur) >= 2:
            updates[int(config.SERVO_CHANNELS[name]["ch"])] = nxt
        if abs(target - nxt) > tol:
            reached = False
    if updates:
        try:
            servo_mod.set_motion_profile(
                list(updates.keys()),
                speed=int(getattr(config, "GAZE_AVERSION_SERVO_SPEED", 90)),
                acceleration=int(getattr(config, "GAZE_AVERSION_SERVO_ACCELERATION", 9)),
            )
        except Exception:
            pass
        servo_mod.set_servos(updates)
        try:
            servo_mod.set_face_tracking_baseline(
                neck=next_pose["neck"], lift=next_pose["headlift"], tilt=next_pose["headtilt"],
            )
        except Exception:
            pass
    return reached


def _maybe_drive_gaze(servo_mod, now: float, speech_active: bool) -> bool:
    """Run the gaze engine for one tick and, when it wants an OFF-target look-away,
    drive it (and the return) through the shared servo writer. Returns True when the
    gaze layer is actively driving the head this tick (caller suspends centering),
    False to let normal face-centering own the ON-target correction.

    Stand-down rules (so it never fights an owner): manual override, an active
    speaker-gaze room scan, a user-commanded directed gaze, or SLEEP hand the head to
    those owners outright. While Rex is SPEAKING or in the listening/think wait, the
    speech/listening motion own the head — the gaze layer starts no new aversion then,
    but it will FINISH an in-progress return so the head never stalls looking away.
    """
    try:
        if not gaze_engine.enabled() or gaze_engine.under_test_runner():
            _gaze_release()
            return False

        manual = bool(getattr(servo_mod, "manual_override_enabled", lambda: False)())
        listening_active = bool(getattr(servo_mod, "listening_motion_active", lambda: False)())
        intent = _speaker_gaze_current_intent(now)
        searching = bool(intent and intent.get("search_requested"))
        owned_by_other = (
            manual or searching
            or directed_gaze_hold_active(now)
            or state_module.get_state() == State.SLEEP
        )
        if owned_by_other:
            _gaze_release()
            return False

        fresh_lock = _face_tracking_has_fresh_lock(now)
        last_touch = max(
            _engaged_last_touch_at, _recent_engaged_touch_at, _last_proactive_speech_at
        )
        idle_secs = (now - last_touch) if last_touch > 0 else 1.0e9
        conv_active = last_touch > 0 and idle_secs < float(
            getattr(config, "GAZE_CLOSE_AFTER_IDLE_SECS", 12.0)
        )
        listening = bool(listening_active)
        speaking = bool(speech_active)
        sweep_enabled = bool(getattr(config, "GAZE_SPEAKING_SWEEP_ENABLED", True))
        # The engine now RUNS during SPEAKING (its ~50% on-target duty + the multi-person
        # include-sweep) so those decisions are produced, not idled away; it still stands
        # down while listening, or with no fresh lock / inactive convo. With the sweep kill
        # switch off, speech suppresses the engine exactly as before.
        engine_suppressed = (
            listening or not fresh_lock or not conv_active
            or (speaking and not sweep_enabled)
        )

        active_speaker = (
            _engaged_person_id if _engaged_person_id is not None else _recent_engaged_person_id
        )
        listener_bearings: list = []
        num_visible = 1
        try:
            people = world_state.get("people") or []
            num_visible = sum(
                1 for p in people
                if not (p.get("face_visible") is False or p.get("face_missing"))
            )
            if sweep_enabled:
                frame_w = float(getattr(config, "CAMERA_WIDTH", 1920) or 1920)
                max_deg = float(getattr(config, "GAZE_LISTENER_MAX_BEARING_DEG", 22.0))
                for p in people:
                    if p.get("face_visible") is False or p.get("face_missing"):
                        continue
                    pid = p.get("person_db_id")
                    if pid is not None and pid == active_speaker:
                        continue  # never sweep to the person already being addressed
                    box = p.get("face_box") or p.get("bounding_box") or p.get("bbox")
                    if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                        continue
                    cx = float(box[0]) + float(box[2]) / 2.0
                    # +deg => neck toward max => RIGHT of frame, matching the validated
                    # _face_x_to_neck_target convention; clamp so a sweep stays a glance.
                    yaw = ((cx / frame_w) - 0.5) * 2.0 * max_deg
                    listener_bearings.append((pid, max(-max_deg, min(max_deg, yaw))))
        except Exception:
            num_visible = 1
            listener_bearings = []

        inputs = gaze_engine.GazeInputs(
            now=now,
            speaking=speaking,
            listening=listening,
            conversation_active=conv_active,
            conversation_idle_secs=(idle_secs if idle_secs < 1.0e8 else 0.0),
            num_people=max(1, num_visible),
            active_speaker_id=active_speaker,
            listener_bearings=listener_bearings,
            suppressed=engine_suppressed,
        )
        decision = gaze_engine.step(inputs)

        is_sweep = (decision.kind == gaze_engine.KIND_INCLUDE_SWEEP)
        # While SPEAKING, only a bounded include-sweep may drive (a glance to include a
        # listener); off-target aversions stay suppressed so they don't fight speech head
        # motion. Listening blocks all new drives. Outside speech, aversions drive as before.
        block_new_drive = listening or (speaking and not (is_sweep and sweep_enabled))
        if not block_new_drive and decision.drive:
            if _gaze_drive.get("phase") != "away":
                _capture_gaze_anchor(decision)
            _drive_gaze_aversion(servo_mod, now, decision, _gaze_drive["anchor"])
            return True

        # ON-target, or the engine stood down: if mid-aversion, return to the anchor
        # first (even under suppression) so the head comes back to the partner.
        if _gaze_drive.get("phase") in ("away", "returning"):
            done = _drive_gaze_return(servo_mod, now)
            if not done:
                _gaze_drive["phase"] = "returning"
                return True
            _gaze_release()
        return False
    except Exception as exc:
        _log.debug("gaze engine step error: %s", exc)
        _gaze_release()
        return False


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
    global _face_tracking_last_center, _face_tracking_pending_center

    if state_module.get_state() == State.SLEEP:
        return
    if time.monotonic() < _face_tracking_suspended_until:
        return
    # A room-exploration session drives Rex's own gaze (survey sweeps + fixation
    # glances) and owns the head — don't fight it with face centering.
    try:
        from intelligence import exploration
        if exploration.active():
            return
    except Exception:
        pass

    try:
        from hardware import servos as servo_mod

        now = time.monotonic()

        # "Mind of his own": while an idle head-wander is in progress, drive the look-
        # around motion instead of centering a face (the 1Hz loop decides when to start
        # one and whether to re-greet on re-acquiring). This MUST run before the frame /
        # listening early-returns below — it needs no frame and self-aborts on listening
        # / speech / resumed conversation, so the wander always progresses and can never
        # get stuck. Read the flag under the lock to honor the _idle_wander protocol.
        with _idle_wander_lock:
            wander_active = bool(_idle_wander.get("active"))
        if wander_active:
            _drive_idle_head_wander(servo_mod, now)
            return

        if frame is None:
            return

        # While listening motion owns the head (gentle nods during the
        # transcription/LLM/TTS wait), don't fight it with face centering.
        if getattr(servo_mod, "listening_motion_active", lambda: False)():
            return

        # While SPEAKING, the speaker-gaze pose + speech wobble already move the head;
        # we soften (not suspend) centering below so they don't all fight.
        speech_active = bool(getattr(servo_mod, "speech_motion_active", lambda: False)())

        # Human-like eye-contact rhythm: when the gaze engine decides to break contact
        # (look away to think, glance up to visualize a complex reply, down to absorb
        # what was said, then return to hand over the floor), it drives the head this
        # tick and we suspend centering — exactly like the idle-wander hook above.
        if _maybe_drive_gaze(servo_mod, now, speech_active):
            return

        candidates = _visible_face_tracking_candidates(people)
        speaker_intent = _speaker_gaze_current_intent(now)
        lock_key = _face_tracking_lock.get("key")
        last_seen = float(_face_tracking_lock.get("last_seen_at") or 0.0)
        lost_hold_secs = float(getattr(config, "FACE_TRACKING_LOST_HOLD_SECS", 4.0) or 0.0)
        lost_search_after = float(getattr(config, "SPEAKER_GAZE_LOST_SEARCH_AFTER_SECS", 0.45) or 0.0)

        candidate = _speaker_gaze_candidate(candidates, speaker_intent)

        # User told Rex to hold a gaze (e.g. "look down"), OR the OpenAI presence
        # sweep is driving the head between captures. While that's true and nobody
        # is visible to track, don't scan the room or drift back to rest — just hold
        # the commanded pose (breathing still bobs around the directed baseline) and
        # let the owning code point the head. Mark it so the idle wander stands down
        # too. If any face IS visible we fall through to normal tracking below, so
        # the moment he spots someone low in frame he locks on and keeps looking down.
        if not candidates and (directed_gaze_hold_active(now) or _startup_presence_fallback_active):
            _face_tracking_lock = {}
            _record_face_tracking_state(
                locked=False,
                visible=False,
                directed_hold=True,
            )
            return

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

        _prev_lock_pid = _face_tracking_lock.get("person_id") if _face_tracking_lock else None
        _face_tracking_lock = {
            "key": candidate["key"],
            "person_id": candidate.get("person_id"),
            "last_seen_at": now,
        }
        if candidate.get("person_id") != _prev_lock_pid:
            # Make the lock SWITCH visible — the JT run silently re-targeted Bret -> JT
            # with no log of which candidate won. Aids future 2-person diagnosis.
            _log.info("face_tracking: lock switched key=%s person_id=%s (was person_id=%s)",
                      candidate.get("key"), candidate.get("person_id"), _prev_lock_pid)

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
        # Calm face-centering while Rex is speaking so it doesn't fight the speech
        # motion / speaker-gaze pose: gentler gain + steps, wider dead-zone (only
        # correct for large offsets). Speaker-gaze still points the head; this just
        # stops the rapid micro-corrections that made the head thrash.
        if speech_active and bool(getattr(config, "FACE_TRACKING_SPEECH_CALM_ENABLED", True)):
            calm = max(0.0, min(1.0, float(getattr(config, "FACE_TRACKING_SPEECH_CALM_FACTOR", 0.4))))
            gain *= calm
            vertical_gain *= calm
            neck_max_step = max(1, int(neck_max_step * calm))
            lift_max_step = max(1, int(lift_max_step * calm))
            tilt_max_step = max(1, int(tilt_max_step * calm))
            dead_zone = max(dead_zone, float(getattr(config, "FACE_TRACKING_SPEECH_DEAD_ZONE_PX", 90)))
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

        # Jump-rejection: a box that teleported across the frame this tick is almost
        # always a spurious detector box on clutter, not the person moving — hold the
        # gaze instead of chasing it. Exceptions: a box dlib freshly identified as a
        # known person is always real (followed immediately, e.g. a seated person low
        # in frame), and an unknown spot that persists gets confirmed (see helper).
        accept_box, _face_tracking_last_center, _face_tracking_pending_center = _evaluate_face_jump(
            cx, cy, candidate_key, now, frame_w, frame_h,
            _face_tracking_last_center, _face_tracking_pending_center,
            identified=candidate.get("person_id") is not None,
            live_tracked=bool(candidate.get("live_tracked")),
        )
        if not accept_box:
            _maybe_log_face_jump_reject(cx, cy, now)
            _record_face_tracking_state(locked=True, visible=True, candidate=candidate)
            return

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
        # Edge boost: the flat per-tick cap made a face far off-centre re-face in
        # ~5s (cap 180 x live damping 0.45 = 81 qus/tick). Scale the cap with the
        # error so big lateral moves sweep fast while near-centre behavior — dead
        # zone, damping, small caps — is untouched. Applied AFTER the damping
        # above so oscillation guards still bite first.
        if frame_cx > 0:
            _boost_frac = float(getattr(config, "FACE_TRACKING_EDGE_BOOST_ERROR_FRAC", 0.30))
            _boost_max = float(getattr(config, "FACE_TRACKING_EDGE_BOOST_MULT", 2.5))
            _err_frac = min(1.0, abs(error_x) / frame_cx)
            if _boost_max > 1.0 and _boost_frac < 1.0 and _err_frac > _boost_frac:
                _boost = 1.0 + (_err_frac - _boost_frac) / max(1e-6, 1.0 - _boost_frac) * (
                    _boost_max - 1.0
                )
                neck_max_step = max(1, int(neck_max_step * _boost))

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
            if _neck_saturated_at_rail(current_neck, target_neck, neck_cfg):
                # Neck is pinned at its limit and can't reduce this error — hold instead
                # of jittering against the rail.
                _neck_smooth = float(current_neck)
            elif abs(next_neck - current_neck) >= 2:
                updates[neck_ch] = next_neck
                _neck_smooth = float(next_neck)
            else:
                _neck_smooth = float(current_neck)
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
    global _boredom_loop_started_at

    interval = getattr(config, "CONSCIOUSNESS_LOOP_INTERVAL_SECS", 1.0)
    last_tick = time.monotonic()
    _last_micro_behavior_at = time.monotonic()
    _boredom_loop_started_at = time.monotonic()

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
            # Once per run: cheap GPT caption of Rex's first look → rex.db (off-tick).
            episodic_hooks.startup_image(frame)

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

            # Episodic memory: log a scene observation when the room materially changes
            # (deduped). Capture only; nothing reads it back yet.
            episodic_hooks.scene_changed(snapshot)

            # Change-of-scenery remark: if this run's startup scene differs from the last
            # run's (different room / outdoors / new place), say so once.
            _step_scenery_change()

            # 5c. Celebrity overrides. These own the first conversational beat
            # before ordinary greetings or ambient remarks.
            if (
                _step_jeff_history_hunters_detection(snapshot, profile)
                or _step_jt_volleyball_detection(snapshot, profile)
                or _step_hair_stylist_detection(snapshot, profile)
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

            # 9a. Boredom escalation — grumble when left alone, then doze into SLEEP.
            _step_boredom_escalation(snapshot, profile)

            # 9a2. Self-triggered exploration (OPT-IN; default off) — a long-stale
            # empty room sends Rex on a wander that feeds the curiosity queue.
            _step_self_exploration(snapshot, profile)

            # 9b. Mood-driven body language — visor openness, breathing cadence, and the
            # occasional idle mood gesture, expressing Rex's sustained body mood on the
            # channels face-tracking doesn't own. (Head-pitch mood rides the rest pose.)
            _step_mood_expression(snapshot, profile)

            # 9c. Idle "mind of his own" head wander — when the conversation lulls while
            # he's locked on a face, occasionally look around the room, then return his
            # gaze and maybe re-greet. (The face-tracking loop drives the motion.)
            _step_idle_head_wander(snapshot, profile)

            # 10. Presence tracking (departure / return reactions)
            _step_presence_tracking(snapshot, profile)

            # 10a. If startup found no confirmed person after the scan, verify with
            # an OpenAI vision sweep before concluding empty (dlib misses small /
            # turned-away faces). On a hit it steers Rex to greet at their height.
            _step_startup_presence_fallback_trigger(snapshot)

            # 10a2. If startup found no confirmed person after the scan, Rex may
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

            # 10d5. Lull callback — when conversation goes quiet, resurface one
            # banked fun-fact premise about the engaged person (callback humor).
            # Open-thread follow-ups and news remarks moved INTO the lean brain's
            # impulse cue ladder (2026-07-18) — the steps here RACED the lean cues
            # (field-logged: a news_remark governor win colliding with a room-change
            # ask in the same lull). The step functions remain for a lean-disabled
            # fallback but are gated off while the lean brain owns silence-fill.
            if not bool(getattr(config, "LEAN_BRAIN_ENABLED", False)):
                _step_open_thread_followup(snapshot, profile)
                _step_news_remark(snapshot, profile)
                _step_interest_discovery(snapshot, profile)

            _step_lull_callback(snapshot, profile)

            # 10e. Overheard chime-in — react when someone talks ABOUT Rex
            _step_overheard_chime_in(snapshot, profile)

            # 10f. GUI face-mood telemetry — keep the dashboard's face-box label
            # aligned with the current visible expression when the scene is unambiguous.
            _step_gui_mood_telemetry(snapshot, frame)

            # 10f2. Long-term expression disposition memory — sample the local
            # MediaPipe expression stream at a low rate for known people.
            _step_disposition_memory(snapshot)

            # 10f-b. Wave back — if a visible person waves, return the wave (+ a short
            # warm line), the way you'd wave back across a room.
            _step_wave_reaction(snapshot, profile)

            # 10f-b2. Room exploration — supervise an active self-directed wander
            # (the sequence runs on exploration.py's own worker thread; this only
            # watchdogs an overrun/dead session). Runs BEFORE autonomous motion so
            # the base stand-down is already in effect this tick.
            _step_exploration(snapshot, profile)

            # 10f-c. Autonomous base motion — rotate the base to face the tracked
            # person (the neck's standing offset is the signal) and approach someone
            # far away (`come`, ToF-guarded by the firmware reflexes).
            _step_autonomous_motion(snapshot, profile)

            # 10f-d. Battery awareness — track the pack's tier from base telemetry;
            # grumble once per downward crossing when someone's around to hear it.
            _step_battery_awareness(snapshot, profile)

            # 10g. Smile reaction — after Rex lands a joke/snarky aside, notice
            # if the target visibly cracks a smile and answer it once.
            _step_smile_reaction(snapshot, profile)

            # 10g-b. Land the laugh / take a bow — react to the ROOM applauding or
            # laughing at Rex's material (gated on a recent-Rex-line window).
            _step_room_reaction(snapshot, profile)

            # 10g-c. "Wait, that's new" — notice a genuinely new object in the room
            # (room_model permanence), in a lull, once per object per session.
            _step_room_change(snapshot, profile)

            # 10g-d. "What's that you're drinking?" — ask about an object someone is
            # HOLDING (person-oriented salience), once per label per session.
            _step_held_object_remark(snapshot, profile)

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
    global _startup_presence_evidence_reason, _solo_unknown_since
    global _last_pose_analysis_at
    global _last_weather_reaction_at
    global _face_tracking_last_error_key, _face_tracking_last_error_x
    global _face_tracking_last_error_y, _face_tracking_last_error_at
    global _last_face_seen_at
    global _smile_reaction_watch, _last_smile_reaction_at
    global _last_facial_expression_reaction_at
    global _last_startle_sound_reaction_at
    global _last_mood_gesture_at, _mood_owns_visor, _last_mood_breathing
    global _pending_wave_back
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
    _relationship_prompt_in_flight.clear()
    _asked_relationship_slots.clear()
    _unknown_first_seen_at.clear()
    _solo_unknown_since = 0.0
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
    _visit_started_at.clear()
    _last_mood_gesture_at = 0.0
    _mood_owns_visor = False
    _last_mood_breathing = None
    with _idle_wander_lock:
        _idle_wander.update({
            "active": False, "until": 0.0, "waypoints": [], "index": 0,
            "reached_at": 0.0, "last_at": 0.0, "pending_regreet": False,
            "regreet_deadline": 0.0,
        })
    try:
        from intelligence import body_mood
        body_mood.clear()
    except Exception:
        pass
    _last_presence_reaction_at.clear()
    _animal_seen_signatures.clear()
    _animal_reacted_at.clear()
    _animal_species_reacted_at.clear()
    _pending_animal_arrivals.clear()
    _animal_presence.clear()
    _update_unknown_streak(False)   # reset unknown-face persistence streak
    _last_startle_sound_reaction_at = 0.0
    _acknowledged_dates.clear()
    _acknowledged_weather_signatures.clear()
    _acknowledged_tod.clear()
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
    gaze_engine.reset()
    _gaze_release()
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
    _wave_reacted_keys.clear()
    _wave_escalation.clear()
    _wave_streak.clear()
    _pending_wave_back = None
    _room_reacted["count"] = 0.0
    _room_reacted["last_at"] = 0.0
    _room_change_state["count"] = 0.0
    _room_change_state["last_at"] = 0.0
    _room_change_remarked.clear()
    _held_object_state["count"] = 0.0
    _held_object_state["last_at"] = 0.0
    _held_object_remarked.clear()
    _held_object_first_seen.clear()
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
    global _pending_wave_back
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
    _animal_species_reacted_at.clear()
    _pending_animal_arrivals.clear()
    _animal_presence.clear()
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
    gaze_engine.reset()
    _gaze_release()
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
    _wave_reacted_keys.clear()
    _wave_escalation.clear()
    _wave_streak.clear()
    _pending_wave_back = None
    _room_reacted["count"] = 0.0
    _room_reacted["last_at"] = 0.0
    _room_change_state["count"] = 0.0
    _room_change_state["last_at"] = 0.0
    _room_change_remarked.clear()
    _held_object_state["count"] = 0.0
    _held_object_state["last_at"] = 0.0
    _held_object_remarked.clear()
    _held_object_first_seen.clear()
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
