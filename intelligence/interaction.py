"""
intelligence/interaction.py — Continuous interaction cycle for DJ-R3X.

Rex is always listening. Wake words get his attention from IDLE or wake him
from SLEEP — they are not the primary interaction model once a conversation
is active. The loop owns: VAD-gated speech accumulation, concurrent
transcription + speaker identification, command dispatch, LLM streaming,
post-response sentiment/fact/follow-up processing, and session teardown.

Public API:
    start()   — launch the listening loop and wake word detection
    stop()    — shut everything down cleanly
"""

import logging
import contextvars
import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
import state as state_module
from state import State
from audio import stream, vad, wake_word, transcription, speaker_id
from audio import speech_queue, output_gate
from audio import echo_cancel
from audio import hardware_aec
from audio import barge_guard
from audio import prosody
from intelligence import action_router, command_parser, llm, personality, rex_preferences
from intelligence import rex_pov
from intelligence import body_mood
from intelligence import performance_output
from intelligence import performance_plan
from intelligence import consciousness
from intelligence import intent_classifier
from intelligence import empathy
from intelligence import conversation_agenda
from intelligence import topic_thread
from intelligence import dialogue_act
from intelligence import user_energy
from intelligence import question_budget
from intelligence import repair_moves
from intelligence import end_thread
from intelligence import introductions
from intelligence import memory_query
from intelligence import social_frame
from intelligence import comedy_modes
from intelligence import turn_completion
from intelligence import friendship_patterns
from intelligence import conversation_steering
from intelligence import profile_questions
from intelligence import person_specials
from memory import facts as facts_memory
from memory import preferences as preferences_memory
from memory import interests as interests_memory
from memory import conversations as conv_memory
from memory import people as people_memory
from memory import events as events_memory
from memory import relationships as rel_memory
from memory import emotional_events
from memory import boundaries as boundary_memory
from memory import forgetting
from memory import person_summary
from memory import social as social_memory
from memory import episodes as episodes_memory
from memory.name_validation import (
    looks_like_initials,
    normalize_person_name,
    names_are_similar,
    normalized_name_key,
)
from awareness import interoception
from awareness import address_mode
from awareness.situation import assessor as _situation_assessor
from world_state import world_state
from utils import conv_log

_log = logging.getLogger(__name__)

# ── VAD chunk size ─────────────────────────────────────────────────────────────
# 32 ms at 16 kHz — matches the sounddevice blocksize and Silero's preferred size.
_CHUNK_SECS = 0.032

_SILENT_COMMAND_PREFIX = "__silent_command__:"
_directed_look_context: dict[str, Any] = {
    "started_at": 0.0,
    "last_at": 0.0,
    "last_direction": None,
    "bare_count": 0,
    "asked_clarify": False,
    "target_hint": "",
    "target_kind": None,
    "greeted_key": None,
}


@dataclass(frozen=True)
class _PostTtsHandoffPolicy:
    asked_question: bool
    fast_response_expected: bool
    listen_delay_secs: float
    flush_buffer: bool


@dataclass
class _RouterDecisionAudit:
    utterance: str
    router_action: Optional[str] = None
    router_confidence: Optional[float] = None
    router_requires_confirmation: Optional[bool] = None
    allowlist_result: str = "not_run"
    legacy_command: Optional[str] = None
    legacy_match_type: Optional[str] = None
    final_executed_path: Optional[str] = None
    completed: Optional[bool] = None
    handler_error: Optional[str] = None
    spoken_text_present: Optional[bool] = None
    emitted: bool = False


@dataclass
class _CharacterLoopTrace:
    turn_id: int
    utterance: str
    heard_text: Optional[str]
    from_idle_activation: bool
    turn_start: float
    raw_best_id: Optional[int] = None
    raw_best_name: Optional[str] = None
    speaker_score: Optional[float] = None
    anonymous_speaker_label: Optional[str] = None
    anonymous_speaker_match_score: Optional[float] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    speaker_label: Optional[str] = None
    identity_resolution: Optional[str] = None
    off_camera_unknown: Optional[bool] = None
    visible_known_count: Optional[int] = None
    has_unknown_visible_or_recent: Optional[bool] = None
    router_action: Optional[str] = None
    router_confidence: Optional[float] = None
    router_requires_confirmation: Optional[bool] = None
    allowlist_result: Optional[str] = None
    legacy_command: Optional[str] = None
    legacy_match_type: Optional[str] = None
    intent: Optional[str] = None
    repair_kind: Optional[str] = None
    final_executed_path: Optional[str] = None
    completed: Optional[bool] = None
    handler_error: Optional[str] = None
    spoken_text_present: Optional[bool] = None
    assistant_asked_question: Optional[bool] = None
    suppress_memory_learning: Optional[bool] = None
    transcript_ready_at: Optional[float] = None
    first_response_queued_at: Optional[float] = None
    first_response_audio_started_at: Optional[float] = None
    first_response_priority: Optional[int] = None
    first_response_preview: Optional[str] = None
    emitted: bool = False


@dataclass
class _AnonymousSpeakerSlot:
    label: str
    embedding: np.ndarray
    first_seen_at: float
    last_seen_at: float
    turns: int = 1
    raw_best_id: Optional[int] = None
    raw_best_name: Optional[str] = None
    raw_best_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_character_loop_turn_ids = count(1)
_current_character_loop_trace: contextvars.ContextVar[Optional[_CharacterLoopTrace]] = (
    contextvars.ContextVar("current_character_loop_trace", default=None)
)
_ttfs_trace_lock = threading.Lock()
_text_input_lock = threading.Lock()
_text_only_mode: bool = False
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None

# Wake word signaling ─────────────────────────────────────────────────────────
_wake_word_fired = threading.Event()
_last_wake_word: Optional[str] = None
_wake_lock = threading.Lock()
_last_wake_word_gesture_at: float = 0.0
_wake_word_gesture_lock = threading.Lock()

# Interruption signal: set when wake word fires while Rex is speaking.
# The _speak_blocking() watchdog calls sd.stop() when it sees this set.
_interrupted = threading.Event()

# Session tracking ────────────────────────────────────────────────────────────
_session_person_ids: set[int] = set()
_session_person_turn_counts: dict[int, int] = {}
_session_exchange_count: int = 0
_last_speech_at: float = 0.0  # monotonic timestamp of most recent speech chunk
_session_forget_terms: dict[int, set[str]] = {}
_session_router_control_topics: dict[int, str] = {}
_interest_idle_followups_spoken: set[tuple[Optional[int], str]] = set()
_low_memory_idle_questions_spoken: set[int] = set()
_recent_memory_candidates = deque(maxlen=12)
_idle_outro_spoken: bool = False
# Proactive idle-banter pacing. _idle_banter_count resets in _begin_user_turn
# when the user speaks again, so each silent stretch gets a fresh attempt budget.
_last_idle_banter_at: float = 0.0
_idle_banter_count: int = 0
# Wall-clock of the last self-initiated (proactive) line of any kind. Used to
# keep proactive lines from stacking back-to-back (see _proactive_line_recently_fired).
_last_proactive_line_at: float = 0.0
# Memory follow-up cadence — the moderate clamp so Rex doesn't run down a checklist
# of remembered events turn-after-turn (see config.FOLLOWUP_MIN_GAP_EXCHANGES). Clocked
# on transcript length (like rex_pov / the arc) so it self-resets at session boundaries
# with no extra reset plumbing; _fired_followup_event_ids is the per-session anti-repeat
# that stops a resolved event from being re-raised if the queue gets re-seeded.
_last_followup_exchange: int = -(10**9)
_last_followup_at: float = 0.0
_fired_followup_event_ids: set[int] = set()
# After Rex asks a real question, hold the floor until this time so he doesn't
# jump back in with idle banter before the user has had a chance to answer.
_floor_held_until: float = 0.0
_pending_music_offer: Optional[dict] = None
_no_response_recovery_token: int = 0
_no_response_recovery_lock = threading.Lock()

# Anti-repeat for latency filler and slow-path acknowledgment lines
_last_filler: Optional[str] = None
_last_slow_path_ack: Optional[str] = None
_last_wake_ack: Optional[str] = None
_last_sleep_mode_ack: Optional[str] = None
_last_wake_from_sleep_ack: Optional[str] = None
_last_vad_barge_in_suppressed_log_at: float = 0.0

# Rolling raw voice-turn history used to distinguish one unfamiliar interjection
# from background group banter. Entries are (epoch_time, label, low_confidence).
_recent_voice_turns = deque()
_anonymous_speaker_slots: list[_AnonymousSpeakerSlot] = []
_anonymous_speaker_next_id: int = 1

# Monotonic deadline before which VAD speech-onset detections are discarded.
# Set at the end of each TTS utterance; prevents Rex's own voice tail from
# immediately triggering a new speech segment.
_listen_resume_at: float = 0.0

# Earliest monotonic time the next speech capture may include. This prevents
# answer pre-roll from pulling Rex's just-finished prompt into Whisper.
_listen_capture_floor_at: float = 0.0

# When True, a post-TTS cleanup flush already happened and the next detected
# speech onset should simply clear this marker. Question handoffs usually leave
# this False so a fast human answer is not deleted.
_post_tts_flush_needed: bool = False

# Monotonic time of the most recent question/fast-response post-TTS handoff. Used
# to keep the responsive (no-flush) window sticky so a trailing-statement handoff
# from the same turn cannot downgrade it and delete a fast human answer.
_last_fast_handoff_at: float = 0.0

# Time window (set by consciousness) where a short bare-name reply is accepted.
_identity_prompt_until: float = 0.0

_IDENTITY_REPLY_WINDOW_SECS = 45.0


def _begin_user_turn() -> None:
    """Suppress proactive speech while the interaction loop handles a user turn."""
    # The user is engaging again — refresh the proactive idle-banter budget so
    # the next silent stretch gets a fresh set of re-engagement attempts, and
    # clear the proactive-line gap so the next stretch isn't suppressed by a line
    # from the previous one.
    global _idle_banter_count, _last_proactive_line_at, _floor_held_until
    _idle_banter_count = 0
    _last_proactive_line_at = 0.0
    _floor_held_until = 0.0
    try:
        _situation_assessor.set_interaction_busy(True)
    except Exception:
        pass
    # Show gentle "I'm listening / thinking" body language for the whole wait
    # (transcription → LLM → TTS) so Rex isn't frozen until he speaks. Speech
    # motion takes over automatically when he starts talking.
    try:
        from hardware import servos
        servos.start_listening_motion()
    except Exception:
        pass
    if _game_suppresses_conversation():
        return
    try:
        # User speech makes queued background/presence chatter stale. Drop any
        # waiting non-urgent speech before it can start mid-answer.
        speech_queue.clear_below_priority(2)
    except Exception:
        pass


def _latency_log(turn_start: float, stage: str, stage_start: Optional[float] = None) -> None:
    if not bool(getattr(config, "LATENCY_TELEMETRY_ENABLED", False)):
        return
    now = time.monotonic()
    if stage_start is None:
        _log.info("[latency] %s total=%.3fs", stage, now - turn_start)
    else:
        _log.info(
            "[latency] %s stage=%.3fs total=%.3fs",
            stage,
            now - stage_start,
            now - turn_start,
        )


def _elapsed_ms(start: Optional[float], end: Optional[float]) -> Optional[int]:
    if start is None or end is None:
        return None
    return int(max(0.0, end - start) * 1000)


def _preview_trace_text(text: str, *, limit: int = 80) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)]}..."


def _ttfs_timing_payload(trace: _CharacterLoopTrace) -> dict:
    queued_at = trace.first_response_queued_at
    audio_started_at = trace.first_response_audio_started_at
    return {
        "transcript_ready_from_segment_start_ms": _elapsed_ms(
            trace.turn_start,
            trace.transcript_ready_at,
        ),
        "response_queued_from_segment_start_ms": _elapsed_ms(
            trace.turn_start,
            queued_at,
        ),
        "response_queued_after_transcript_ms": _elapsed_ms(
            trace.transcript_ready_at,
            queued_at,
        ),
        "audio_started_from_segment_start_ms": _elapsed_ms(
            trace.turn_start,
            audio_started_at,
        ),
        "audio_started_after_transcript_ms": _elapsed_ms(
            trace.transcript_ready_at,
            audio_started_at,
        ),
        "audio_started_after_queue_ms": _elapsed_ms(
            queued_at,
            audio_started_at,
        ),
        "first_response_priority": trace.first_response_priority,
        "first_response_preview": trace.first_response_preview,
    }


def _log_ttfs_event(trace: Optional[_CharacterLoopTrace], event: str) -> None:
    if trace is None:
        return
    if not bool(getattr(config, "TTFS_TELEMETRY_ENABLED", True)):
        return
    payload = {
        "turn_id": trace.turn_id,
        "utterance": trace.utterance,
        "event": event,
        **_ttfs_timing_payload(trace),
    }
    _log.info("[ttfs] %s", json.dumps(payload, sort_keys=True))


def _mark_first_response_queued(
    trace: Optional[_CharacterLoopTrace],
    *,
    text: str,
    priority: int,
    queued_at: Optional[float] = None,
) -> bool:
    if trace is None:
        return False
    with _ttfs_trace_lock:
        if trace.first_response_queued_at is not None:
            return False
        trace.first_response_queued_at = queued_at if queued_at is not None else time.monotonic()
        trace.first_response_priority = int(priority)
        trace.first_response_preview = _preview_trace_text(text)
    _log_ttfs_event(trace, "first_response_queued")
    return True


def _mark_first_response_audio_started(
    trace: Optional[_CharacterLoopTrace],
    *,
    started_at: Optional[float] = None,
) -> bool:
    if trace is None:
        return False
    with _ttfs_trace_lock:
        if trace.first_response_audio_started_at is not None:
            return False
        trace.first_response_audio_started_at = (
            started_at if started_at is not None else time.monotonic()
        )
    _log_ttfs_event(trace, "first_response_audio_started")
    return True


def _end_user_turn() -> None:
    try:
        _situation_assessor.set_interaction_busy(False)
    except Exception:
        pass
    # Stop listening motion if the turn produced no speech (if Rex responds,
    # begin_speech_motion already handed off). Idempotent.
    try:
        from hardware import servos
        servos.stop_listening_motion()
    except Exception:
        pass
_NAME_MAX_WORDS = 3

# If a follow-up question ("how did X go?") is outstanding, this stores the
# event row so the next user utterance can be captured as the outcome.
_awaiting_followup_event: Optional[dict] = None

# Replies that say a remembered event never happened for the user. When answering
# Rex's "how did X go?", these resolve the follow-up even if they parse as a
# repair, so Rex stops re-asking about something the user says they didn't do.
_FOLLOWUP_DID_NOT_HAPPEN_RE = re.compile(
    r"\b(?:"
    r"never\s+went|never\s+made\s+it|never\s+happened|never\s+ended\s+up|"
    r"never\s+got\s+to|never\s+did\s+(?:go|that|it)|"
    r"did\s*n['’]?t\s+go|did\s+not\s+go|did\s*n['’]?t\s+make\s+it|did\s+not\s+make\s+it|"
    r"did\s*n['’]?t\s+happen|did\s+not\s+happen|did\s*n['’]?t\s+end\s+up|"
    r"did\s*n['’]?t\s+attend|did\s*n['’]?t\s+get\s+to|"
    r"was\s*n['’]?t\s+able\s+to|could\s*n['’]?t\s+make\s+it"
    r")\b",
    re.IGNORECASE,
)


def _followup_event_did_not_happen(text: str) -> bool:
    """True when a reply says a remembered event did not happen for the user."""
    return bool(_FOLLOWUP_DID_NOT_HAPPEN_RE.search(text or ""))

# Single-cell list so nested closures can mutate without `global`.
# When set, the post-response hook will fire a "how do you know <name>?"
# question and open a relationship-prompt window for the newcomer's next reply.
_pending_post_greet_relationship: list[Optional[dict]] = [None]

# Off-camera unknown voice: when an unrecognized voice speaks while a known
# person is engaged and nobody unknown is visible, Rex asks "who's that?" and
# stores the unknown's audio so we can enroll their voice once the engaged
# person names them. Cleared on timeout or after a successful naming.
# Shape: {audio: np.ndarray, asked_at: float, prior_engaged_id: int,
#         prior_engaged_name: Optional[str], overheard_text: str}
_pending_offscreen_identify: Optional[dict] = None

# Face-reveal confirmation: Rex heard a known voice (voice-only enrolled
# person), sees unknown face(s), and asked "is this what you look like?" or
# "are you on my left or right?" Holds cached face encodings + the person_id
# we're trying to confirm. Shape:
#   {person_id: int, name: str, mode: "binary"|"lateral",
#    candidates: [{"slot_id": str, "encoding": np.ndarray, "x": int}, ...],
#    asked_at: float}
_pending_face_reveal_confirm: Optional[dict] = None

# Explicit introduction flow. Separate from generic unknown-face curiosity:
# when someone says "this is my partner Alex", this tracks the intro as a social
# event, saves the relationship, and asks one natural "how did you meet?" beat.
_pending_introduction: Optional[dict] = None
_pending_intro_followup: Optional[dict] = None
_pending_intro_voice_capture: Optional[dict] = None

# Someone answered an identity/intro prompt with a very common first name only,
# or a returning known person still only has that common first name on file.
# Hold/enforce the last-name clarification so memory rows stay distinct.
# Shape: {first_name: str, audio: np.ndarray, asked_at: float,
#         prior_engagement: Optional[dict]}
_pending_common_first_name_identity: Optional[dict] = None
_pending_common_first_name_introduction: Optional[dict] = None
_pending_existing_common_first_name: Optional[dict] = None
_pending_identity_match_confirmation: Optional[dict] = None
_pending_prompted_name_confirmation: Optional[dict] = None
_common_first_name_prompted_this_session: set[int] = set()

# Memory wipe confirmations are destructive, so the confirmation phrase is
# exact and short-lived rather than a fuzzy command-parser match.
_pending_memory_wipe: Optional[dict] = None

# Per-session set of (person_id) we've already attempted a face reveal for
# but were declined ("no") — so Rex doesn't keep re-asking the same question.
_face_reveal_declined: set[int] = set()

# Per-session set of person_ids whose voice biometric was auto-refreshed this
# session. Cap at one refresh per person per session so we don't spam new
# biometric rows when someone speaks a lot.
_voice_refreshed_this_session: set[int] = set()

# Per-person grief / loss conversation flow. When the empathy classifier
# detects a death/grief/illness event with an identifiable subject, Rex
# walks a small structured exchange instead of streaming a free-form LLM
# reply: condolence + consent → name → name-aware acknowledgment → hand
# back to the normal LLM. Keyed by person_id. TTL guards against stale
# state when a conversation drifts away from the topic.
#   step: "awaiting_consent" | "awaiting_name" | "awaiting_description"
#   subject: e.g. "grandpa" / "dog" / "best friend"
#   subject_kind: "person" | "pet" | "other"
#   name: deceased's name once known (None until provided)
#   started_at: time.monotonic() when the flow began
_grief_flow_state: dict[int, dict] = {}
_GRIEF_FLOW_TTL_SECS = 600.0  # 10 minutes — clears stale flows
_GRIEF_FLOW_KEYWORDS = ("grief", "death", "illness")
_PRONOUN_VALUE_RE = re.compile(r"\b(he/him|she/her|they/them)\b", re.IGNORECASE)
_PRONOUN_NAMED_RE = re.compile(
    r"\b([A-Za-z][A-Za-z'\-]{1,40})\s+"
    r"(?:uses|use|goes by|go by|has|have)\s+"
    r"(he/him|she/her|they/them)\b",
    re.IGNORECASE,
)
_PRONOUN_SELF_RE = re.compile(
    r"\b(?:i\s+(?:use|go by|have)|my pronouns are)\s+"
    r"(he/him|she/her|they/them)\b",
    re.IGNORECASE,
)
_PRONOUN_THIRD_PERSON_RE = re.compile(
    r"\b(?:he|she|they)\s+(?:uses|use|goes by|go by|has|have)\s+"
    r"(he/him|she/her|they/them)\b",
    re.IGNORECASE,
)

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi am\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi['’]m\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bim\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bthis is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bcall me\s+(.+)$", re.IGNORECASE),
    re.compile(r"\brename me(?:\s+to)?\s+(.+)$", re.IGNORECASE),
]
_SELF_NAME_PATTERNS = _NAME_PATTERNS[:4]
_SELF_INTRO_NON_NAME_RE = re.compile(
    r"^\s*(?:"
    r"from|in|at|on|near|around|inside|outside|with|for|to|into|"
    r"going\b|doing\b|feeling\b|still\b|just\b|not\b|so\b|very\b|"
    r"fine\b|okay\b|ok\b|good\b|great\b|happy\b|sad\b|tired\b|busy\b|"
    r"hungry\b|thirsty\b|sorry\b|ready\b|here\b|home\b|back\b"
    r")\b",
    re.IGNORECASE,
)
_CALL_ME_NAME_RE = re.compile(
    r"\b(?:you can\s+)?call me\s+(.+)$",
    re.IGNORECASE,
)
_NAME_CORRECTION_RE = re.compile(
    r"\b(?:you\s+(?:got|have)\s+my\s+name\s+wrong|"
    r"that['’]?s\s+not\s+my\s+name|that\s+isn['’]?t\s+my\s+name|"
    r"that['’]?s\s+not\s+[A-Za-z][A-Za-z' -]{1,60}|"
    r"that\s+isn['’]?t\s+[A-Za-z][A-Za-z' -]{1,60}|"
    r"you\s+called\s+me\s+the\s+wrong\s+name|"
    r"my\s+name\s+is|call\s+me|rename\s+me)\b",
    re.IGNORECASE,
)
_PREFERRED_NAME_SPLIT_RE = re.compile(
    r"\b(?:but\s+)?(?:you can\s+)?call me\b",
    re.IGNORECASE,
)
_NAME_TRAILING_FILLER_RE = re.compile(
    r"\b(?:hi|hello|hey|wait|hold on|actually|instead|from now on|please|thanks|thank you)\b.*$",
    re.IGNORECASE,
)

_NAME_STOPWORDS = {
    "again",
    "back",
    "both",
    "everybody",
    "everyone",
    "fine",
    "good",
    "great",
    "have",
    "has",
    "here",
    "hi",
    "hello",
    "hey",
    "nobody",
    "okay",
    "ok",
    "someone",
    "somebody",
    "ready",
    "sorry",
    "there",
    "you",
    "your",
    "whoever",
}


# ─────────────────────────────────────────────────────────────────────────────
# Wake word callback (called from wake_word's background daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

def _on_wake_word(model_name: str) -> None:
    global _last_wake_word
    try:
        current_state = state_module.get_state()
    except Exception:
        current_state = State.IDLE

    # Dedicated "shut down" wake word → immediate shutdown, skipping the VAD/STT path
    # (where "shut down" is often clipped to "down" or dropped) and the LLM farewell.
    # main.py runs the normal shutdown sequence (audio + animation) on the state change.
    shutdown_model = str(getattr(config, "WAKE_WORD_SHUTDOWN_MODEL", "") or "").strip()
    if shutdown_model and model_name == shutdown_model:
        if current_state == State.SHUTDOWN:
            return
        # Confirm the utterance really was a shutdown command before powering off.
        # The acoustic model alone confuses "look down" with "shut down" (it fires
        # at ~0.5-0.8 for both), and unlike every other shutdown route this one was
        # NOT lexically gated. Transcribe ~2s of the rolling buffer (the whole
        # phrase, not a VAD fragment that clips "shut down" to "down") and only shut
        # down if it's an actual standalone shutdown command. Fails safe: an empty
        # or failed transcript is rejected, so an STT outage never powers Rex off.
        if bool(getattr(config, "WAKE_WORD_SHUTDOWN_CONFIRM_ENABLED", True)):
            try:
                secs = float(getattr(config, "WAKE_WORD_SHUTDOWN_CONFIRM_AUDIO_SECS", 2.0))
                recent = stream.get_audio_chunk(secs)
                text = transcription.transcribe(recent) if len(recent) else ""
            except Exception as exc:
                text = ""
                _log.warning("[wake_word] shutdown confirm transcription failed: %s", exc)
            if not command_parser.is_standalone_shutdown_command(text):
                _log.info(
                    "[wake_word] shut_down wake hit NOT confirmed by transcript %r — "
                    "ignoring (likely 'look down')", text,
                )
                return
            _log.info(
                "[wake_word] shutdown wake word %r confirmed by transcript %r — initiating shutdown",
                model_name, text,
            )
        else:
            _log.info(
                "[wake_word] shutdown wake word %r detected (confirm disabled) — initiating shutdown",
                model_name,
            )
        with _wake_lock:
            _last_wake_word = model_name
        try:
            _interrupted.set()  # cut off any in-progress speech immediately
        except Exception:
            pass
        state_module.set_state(State.SHUTDOWN)
        return

    if current_state != State.SLEEP and model_name == "wakeuprex":
        _log.info("[wake_word] ignored sleep-only wake word while not asleep: %s", model_name)
        return

    with _wake_lock:
        _last_wake_word = model_name

    _wake_word_recognition_gesture(model_name)

    if speech_queue.is_speaking():
        if _response_wait_active():
            _log.info(
                "[wake_word] ignored mid-question wake interruption while waiting for response"
            )
        else:
            _interrupted.set()

    _wake_word_fired.set()


def _wake_word_recognition_gesture(model_name: str) -> bool:
    """Trigger a short physical wake-word recognition wave when appropriate."""
    if not bool(getattr(config, "WAKE_WORD_RECOGNITION_GESTURE_ENABLED", True)):
        return False

    allowed = {
        str(item).strip()
        for item in getattr(config, "WAKE_WORD_RECOGNITION_GESTURE_MODELS", [])
        if str(item).strip()
    }
    if allowed and str(model_name).strip() not in allowed:
        return False

    try:
        if state_module.get_state() in (State.QUIET, State.SLEEP, State.SHUTDOWN):
            return False
    except Exception:
        return False

    global _last_wake_word_gesture_at
    cooldown = max(
        0.0,
        float(getattr(config, "WAKE_WORD_RECOGNITION_GESTURE_COOLDOWN_SECS", 1.25) or 0.0),
    )
    now = time.monotonic()
    with _wake_word_gesture_lock:
        if cooldown > 0.0 and now - _last_wake_word_gesture_at < cooldown:
            return False
        _last_wake_word_gesture_at = now

    try:
        from sequences import animations
        return bool(animations.wake_word_ack_wave())
    except Exception as exc:
        _log.debug("[wake_word] recognition gesture failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Speech helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_speak() -> bool:
    return state_module.get_state() not in (State.QUIET, State.SLEEP, State.SHUTDOWN)


def _dj_is_playing() -> bool:
    try:
        from features import dj as dj_mod
        return bool(dj_mod.is_playing())
    except Exception:
        return False


def _dj_suppresses_conversation() -> bool:
    return (
        bool(getattr(config, "DJ_SUPPRESS_CONVERSATION_DURING_PLAYBACK", True))
        and _dj_is_playing()
    )


def _stop_dj_for_wake() -> bool:
    """Stop DJ playback in response to a wake word so Rex can take a command.

    During playback the conversation loop and the VAD mic feed are intentionally
    suppressed (the music would otherwise self-trigger), which also strands the
    wake word — the user's only recourse was killing the process. The wake word
    is the deliberate override, so stop the track, clear the post-playback
    suppression tail, and let the normal listen loop resume cleanly.

    Returns True if a track was actually stopped.
    """
    stopped = False
    try:
        from features import dj as dj_mod
        if dj_mod.is_playing():
            dj_mod.stop()
            stopped = True
            _log.info("[wake_word] stopped DJ playback for wake-word barge-in")
    except Exception as exc:
        _log.debug("[wake_word] DJ barge-in stop failed: %s", exc)
    if stopped:
        # dj.stop() arms a short post-playback suppression tail; drop it so the
        # wake ack and the user's follow-up command are heard immediately.
        try:
            echo_cancel.clear_suppression_tail()
        except Exception:
            pass
    return stopped


def _start_dj_after_response(track) -> None:
    """Start DJ playback just after the current spoken response sequence closes."""
    def _play() -> None:
        try:
            from features import dj as dj_mod
            dj_mod.play(track)
        except Exception as exc:
            _log.debug("deferred DJ playback failed: %s", exc)

    delay = max(
        0.05,
        float(getattr(config, "DJ_START_AFTER_TTS_DELAY_SECS", 0.25) or 0.0),
    )
    threading.Timer(delay, _play).start()


def _game_suppresses_conversation() -> bool:
    try:
        from features import games as games_mod
        if hasattr(games_mod, "suppresses_conversation_interruptions"):
            return bool(games_mod.suppresses_conversation_interruptions())
        return bool(games_mod.is_active())
    except Exception:
        return False


def _game_active_for_router() -> bool:
    try:
        from features import games as games_mod
        return bool(games_mod.is_active())
    except Exception:
        return False


def _action_router_context(
    text: str,
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    speaker_score: float,
    recent_engagement: Optional[dict],
    off_camera_unknown: bool,
    identity_prompt_active: bool,
) -> dict:
    """Build compact turn context for the shadow action router."""
    visible_people = []
    try:
        for p in world_state.get("people") or []:
            visible_people.append({
                "person_id": p.get("person_db_id"),
                "face_id": p.get("face_id"),
                "voice_id": p.get("voice_id"),
                "age_estimate": p.get("age_estimate"),
                "is_unknown": p.get("person_db_id") is None,
            })
    except Exception:
        visible_people = []

    active_game = False
    try:
        from features import games as games_mod
        active_game = bool(games_mod.is_active())
    except Exception:
        active_game = False

    legacy_command = None
    try:
        match = command_parser.parse(text)
        if match is not None:
            legacy_command = {
                "command_key": match.command_key,
                "match_type": match.match_type,
                "args": match.args,
            }
    except Exception:
        legacy_command = None

    pending_event = None
    if _awaiting_followup_event is not None:
        pending_event = {
            "event_id": _awaiting_followup_event.get("event_id"),
            "event_name": _awaiting_followup_event.get("event_name"),
            "person_id": _awaiting_followup_event.get("person_id"),
        }

    pending_question = None
    if person_id is not None:
        try:
            pending = rel_memory.get_latest_pending_question(person_id)
            if pending:
                pending_question = {
                    "question_key": pending.get("question_key"),
                    "question_text": pending.get("question_text"),
                    "depth": pending.get("depth"),
                    "asked_at": pending.get("asked_at"),
                }
        except Exception:
            pending_question = None

    return {
        "speaker": {
            "person_id": person_id,
            "name": person_name,
            "raw_best_id": raw_best_id,
            "raw_best_name": raw_best_name,
            "voice_score": round(float(speaker_score or 0.0), 3),
            "off_camera_unknown": off_camera_unknown,
        },
        "visible_people": visible_people[:6],
        "recent_engagement": recent_engagement,
        "active_game": active_game,
        "active_music": _dj_is_playing(),
        "pending": {
            "identity_prompt_active": identity_prompt_active,
            "pending_question": pending_question,
            "awaiting_followup_event": pending_event,
            "offscreen_identify": _pending_offscreen_identify is not None,
            "introduction": _pending_introduction is not None,
            "intro_followup": _pending_intro_followup is not None,
            "prompted_name_confirmation": _pending_prompted_name_confirmation is not None,
        },
        "legacy": {
            "command_key": (legacy_command or {}).get("command_key"),
            "command_match": legacy_command,
        },
        "last_rex_frame": dialogue_act.active_frame_context(person_id),
    }


def _router_decision_executable(
    decision: Optional[action_router.ActionDecision],
    *,
    text: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> bool:
    return _router_execution_block_reason(
        decision,
        text=text,
        context=context,
        dialogue_decision=dialogue_decision,
    ) is None


def _router_execute_allowlist() -> Optional[set[str]]:
    configured = getattr(config, "ACTION_ROUTER_EXECUTE_ACTIONS", None)
    if configured is None:
        return None
    if isinstance(configured, str):
        return {
            item.strip()
            for item in configured.split(",")
            if item.strip()
        }
    try:
        return {str(item).strip() for item in configured if str(item).strip()}
    except TypeError:
        return set()


def _router_execution_block_reason(
    decision: Optional[action_router.ActionDecision],
    *,
    text: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> Optional[str]:
    if decision is None:
        return "no_decision"
    if bool(decision.requires_confirmation):
        return "requires_confirmation"
    if decision.action not in action_router.EXECUTABLE_ACTIONS:
        return "not_executable"
    if decision.action == "performance.body_beat":
        args = decision.args or {}
        beat = (
            args.get("body_beat")
            or args.get("beat")
            or args.get("gesture")
            or args.get("pose")
            or ""
        )
        if performance_plan.canonical_body_beat(str(beat)) is None:
            return "unknown_body_beat"
    if decision.action == "performance.mood_pose":
        args = decision.args or {}
        mood = (
            args.get("mood")
            or args.get("emotion")
            or args.get("pose")
            or ""
        )
        if performance_plan.canonical_mood_pose(str(mood)) is None:
            return "unknown_mood_pose"
    if decision.action == "game.answer" and not _game_active_for_router():
        return "game_inactive"
    if (
        dialogue_decision is not None
        and dialogue_act.action_blocked_by_dialogue(decision.action, dialogue_decision)
        and not _dialogue_allows_action_breakout(
            decision.action,
            text or "",
            dialogue_decision,
        )
    ):
        return "blocked_by_dialogue_act"
    if text is not None:
        evidence_reason = action_router.missing_required_evidence_reason(
            text,
            decision,
            context=context,
        )
        if evidence_reason:
            return evidence_reason
    allowlist = _router_execute_allowlist()
    if allowlist is not None and decision.action not in allowlist:
        return "not_in_execute_allowlist"
    threshold = float(getattr(config, "ACTION_ROUTER_EXECUTE_MIN_CONFIDENCE", 0.85))
    if float(decision.confidence or 0.0) < threshold:
        return "below_confidence_threshold"
    return None


def _new_router_audit(text: str, context: Optional[dict[str, Any]]) -> _RouterDecisionAudit:
    legacy = ((context or {}).get("legacy") or {}).get("command_match") or {}
    return _RouterDecisionAudit(
        utterance=str(text or ""),
        legacy_command=legacy.get("command_key"),
        legacy_match_type=legacy.get("match_type"),
    )


def _router_audit_note_decision(
    audit: Optional[_RouterDecisionAudit],
    decision: Optional[action_router.ActionDecision],
    *,
    text: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> None:
    if audit is None:
        return
    audit.router_action = decision.action if decision is not None else None
    audit.router_confidence = (
        round(float(decision.confidence or 0.0), 3)
        if decision is not None
        else None
    )
    audit.router_requires_confirmation = (
        bool(decision.requires_confirmation)
        if decision is not None
        else None
    )
    block_reason = _router_execution_block_reason(
        decision,
        text=text,
        context=context,
        dialogue_decision=dialogue_decision,
    )
    audit.allowlist_result = "allowed" if block_reason is None else block_reason


def _router_audit_note_fast_local_action(
    audit: Optional[_RouterDecisionAudit],
    action: str,
    *,
    args: Optional[dict[str, Any]] = None,
    reason: str = "deterministic fast local command",
) -> None:
    """Record the stable action represented by a deterministic fast lane."""
    if audit is None:
        return
    decision = action_router.ActionDecision(
        action=action,
        confidence=1.0,
        args=args or {},
        requires_confirmation=False,
        reason=reason,
    )
    _router_audit_note_decision(audit, decision)


def _router_audit_note_execute_disabled(audit: Optional[_RouterDecisionAudit]) -> None:
    if audit is not None and audit.allowlist_result == "not_run":
        audit.allowlist_result = "execute_disabled_shadow"


def _router_blocked_confirmation_response(
    decision: Optional[action_router.ActionDecision],
    block_reason: Optional[str],
) -> Optional[str]:
    """Return a short spoken response for blocked actions that need consent."""
    if decision is None or block_reason != "requires_confirmation":
        return None
    if decision.action == "vision.snapshot":
        return (
            "I can take a look, but I won't store a scene memory without explicit "
            "confirmation. Say yes, remember this scene if you want that."
        )
    return None


_GENERAL_KNOWLEDGE_REPLY_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+know|tell\s+me|explain|teach\s+me)"
    r"(?:\s+(?:about|regarding|on))?\s+(?P<topic>[^?.!]+)",
    re.IGNORECASE,
)
_PERSONAL_MEMORY_TOPIC_RE = re.compile(
    r"\b(?:me|my|mine|myself|you\s+remember|remember\s+about|"
    r"know\s+about\s+me|know\s+about\s+my)\b",
    re.IGNORECASE,
)
_MEMORY_LEARNING_CLOSURE_RE = re.compile(
    r"\b("
    r"bye|goodbye|good-bye|see you|see ya|talk to you later|talk later|"
    r"catch you later|later|nice speaking|nice talking|nice chatting|"
    r"i'?m\s+going\s+to\s+go|i\s+am\s+going\s+to\s+go|i\s+have\s+to\s+go|"
    r"gotta\s+go"
    r")\b",
    re.IGNORECASE,
)


def _conversation_reply_should_skip_memory_learning(
    text: str,
    decision: Optional[action_router.ActionDecision],
) -> bool:
    """Avoid saving Rex's own general-knowledge answer as a personal fact."""
    if decision is None or decision.action != "conversation.reply":
        return False
    text_clean = str(text or "").strip()
    if not text_clean:
        return False
    if _MEMORY_LEARNING_CLOSURE_RE.search(text_clean):
        return True
    match = _GENERAL_KNOWLEDGE_REPLY_RE.search(text_clean)
    if not match:
        return False
    topic = str(match.group("topic") or "")
    return not _PERSONAL_MEMORY_TOPIC_RE.search(topic)


def _router_nonexecuted_should_suppress_memory_learning(
    decision: Optional[action_router.ActionDecision],
) -> bool:
    """Control/tool routes should not be learned when they fail open to chat."""
    if decision is None or decision.action == "conversation.reply":
        return False
    if decision.action == "identity.introduce_person":
        return False
    return True


def _router_audit_note_legacy_command(
    audit: Optional[_RouterDecisionAudit],
    match: Optional[command_parser.CommandMatch],
) -> None:
    if audit is None or match is None:
        return
    audit.legacy_command = match.command_key
    audit.legacy_match_type = match.match_type


def _router_audit_clear_legacy_command(audit: Optional[_RouterDecisionAudit]) -> None:
    if audit is None:
        return
    audit.legacy_command = None
    audit.legacy_match_type = None


def _legacy_command_blocked_by_dialogue(
    match: Optional[command_parser.CommandMatch],
    dialogue_decision: Optional[dialogue_act.DialogueActDecision],
    text: str = "",
) -> bool:
    """Keep contextual replies from falling through to older command parsers."""
    if match is None or dialogue_decision is None:
        return False
    if dialogue_decision.label != "answer_to_rex":
        return False
    decision = _legacy_command_action_decision(match)
    if decision is not None and _dialogue_allows_action_breakout(
        decision.action,
        text,
        dialogue_decision,
    ):
        return False
    return True


_LEGACY_COMMAND_ACTION_MAP: dict[str, str] = {
    "time_query": "time.query",
    "date_query": "date.query",
    "status_uptime": "status.uptime",
    "vision_describe": "vision.describe_scene",
    "vision_who_am_i": "identity.who_is_speaking",
    "whats_my_name": "identity.who_is_speaking",
    "rename_me": "identity.name_correction",
    "memory_review": "memory.query",
    "memory_forget_fact": "memory.forget_specific",
    "memory_boundary": "memory.recent_discard",
    "forget_specific": "memory.forget_specific",
    "query_play_options": "music.options",
    "dj_stop": "music.stop",
    "dj_skip": "music.skip",
    "dj_play_vibe": "music.play",
    "start_trivia": "game.start",
    "start_i_spy": "game.start",
    "start_20_questions": "game.start",
    "start_jeopardy": "game.start",
    "start_word_association": "game.start",
    "start_game": "game.start",
    "stop_game": "game.stop",
    "sleep": "system.sleep",
    "wake_up": "system.sleep",
    "quiet_mode": "system.sleep",
    "shutdown": "system.shutdown",
}


_INTENT_ACTION_MAP: dict[str, str] = {
    "query_time": "time.query",
    "query_date": "date.query",
    "query_weather": "weather.query",
    "query_capabilities": "status.capabilities",
    "query_uptime": "status.uptime",
    "query_what_do_you_see": "vision.describe_scene",
    "query_who_is_speaking": "identity.who_is_speaking",
    "query_memory": "memory.query",
    "play_music": "music.play",
    "query_music_options": "music.options",
}


def _legacy_command_confidence(match: command_parser.CommandMatch) -> float:
    if match.match_type == "exact":
        return 0.99
    if match.match_type in {"prefix", "pattern", "action_router", "active_game_stop"}:
        return 0.96
    if match.match_type == "fuzzy":
        return 0.74
    return 0.90


def _legacy_command_action_decision(
    match: Optional[command_parser.CommandMatch],
) -> Optional[action_router.ActionDecision]:
    if match is None:
        return None
    action = _LEGACY_COMMAND_ACTION_MAP.get(match.command_key)
    if not action:
        return None
    args = dict(match.args or {})
    if match.command_key == "dj_play_vibe":
        args = {"music_query": args.get("vibe") or ""}
    elif match.command_key == "start_game":
        args = {"game": args.get("game") or ""}
    elif match.command_key.startswith("start_"):
        args = {"game": match.command_key.removeprefix("start_")}
    elif match.command_key == "memory_forget_fact":
        args = {"target": args.get("statement") or ""}
    return action_router.ActionDecision(
        action=action,
        confidence=_legacy_command_confidence(match),
        args=args,
        reason=f"legacy command parser claim: {match.command_key}",
    )


def _dialogue_allows_action_breakout(
    action: Optional[str],
    text: str,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision],
) -> bool:
    """Let clear commands win over a stale conversational reply frame."""
    if not action or dialogue_decision is None:
        return False
    if dialogue_decision.label != "answer_to_rex":
        return False
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if action == "system.sleep":
        return command_parser.is_standalone_sleep_command(cleaned)
    if action == "system.shutdown":
        return command_parser.is_standalone_shutdown_command(cleaned)
    if action == "music.play":
        return bool(_IDLE_DIRECT_MUSIC_RE.search(cleaned))
    if action in {"music.stop", "music.skip"}:
        return bool(re.search(r"^\s*(?:stop|pause|skip)\b", cleaned, re.IGNORECASE))
    return False


def _legacy_command_execution_block_reason(
    match: Optional[command_parser.CommandMatch],
    *,
    text: str,
    context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> Optional[str]:
    """Central turn-policy gate for legacy command parser claims."""
    if match is None:
        return None
    if _legacy_command_blocked_by_dialogue(match, dialogue_decision, text):
        return "blocked_by_dialogue_act"
    if (
        match.match_type == "fuzzy"
        and not bool(getattr(config, "LEGACY_COMMAND_FUZZY_EXECUTE_ENABLED", False))
    ):
        return "legacy_fuzzy_disabled"
    decision = _legacy_command_action_decision(match)
    if decision is None:
        return None
    return action_router.missing_required_evidence_reason(
        text,
        decision,
        context=context,
    )


def _intent_action_decision(intent: str) -> Optional[action_router.ActionDecision]:
    action = _INTENT_ACTION_MAP.get(intent)
    if not action:
        return None
    return action_router.ActionDecision(
        action=action,
        confidence=0.94,
        args={},
        reason=f"deterministic intent claim: {intent}",
    )


def _intent_execution_block_reason(
    intent: str,
    *,
    text: str,
    context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
    router_action: Optional[str] = None,
) -> Optional[str]:
    """Central turn-policy gate for deterministic intent-classifier claims."""
    if not intent or intent == "general":
        return None
    decision = _intent_action_decision(intent)
    if decision is None:
        return None
    # The LLM router already judged this turn a conversational REPAIR (a
    # correction or complaint like "you didn't give me time to answer"). A
    # repair is never a clock/date/weather/data query — so don't let a keyword
    # match in the deterministic fallback hijack it into a tool answer (this is
    # what produced "It's 8:33 PM." in response to a pacing complaint).
    if (router_action or "") == "conversation.repair":
        return "router_classified_repair"
    evidence_reason = action_router.missing_required_evidence_reason(
        text,
        decision,
        context=context,
    )
    if evidence_reason:
        return evidence_reason
    if (
        dialogue_decision is not None
        and dialogue_decision.label == "answer_to_rex"
        and not _dialogue_allows_action_breakout(
            decision.action,
            text,
            dialogue_decision,
        )
    ):
        return "blocked_by_dialogue_act"
    return None


def _router_audit_note_result(
    audit: Optional[_RouterDecisionAudit],
    *,
    completed: Optional[bool] = None,
    handler_error: Optional[str] = None,
    spoken_text: Optional[str] = None,
    spoken_text_present: Optional[bool] = None,
) -> None:
    if audit is None:
        return
    if completed is not None and audit.completed is None:
        audit.completed = bool(completed)
    if handler_error and audit.handler_error is None:
        text = " ".join(str(handler_error).strip().split())
        audit.handler_error = text[:240] if text else None
    if spoken_text_present is not None and audit.spoken_text_present is None:
        audit.spoken_text_present = bool(spoken_text_present)
    elif spoken_text is not None and audit.spoken_text_present is None:
        audit.spoken_text_present = bool(str(spoken_text).strip())
    if audit.completed is None and audit.spoken_text_present is not None:
        audit.completed = audit.spoken_text_present


def _log_router_audit(
    audit: Optional[_RouterDecisionAudit],
    final_executed_path: str,
    *,
    completed: Optional[bool] = None,
    handler_error: Optional[str] = None,
    spoken_text: Optional[str] = None,
    spoken_text_present: Optional[bool] = None,
) -> None:
    if audit is None or audit.emitted:
        return
    if not bool(getattr(config, "ACTION_ROUTER_AUDIT_LOG_ENABLED", True)):
        return
    _router_audit_note_result(
        audit,
        completed=completed,
        handler_error=handler_error,
        spoken_text=spoken_text,
        spoken_text_present=spoken_text_present,
    )
    audit.final_executed_path = final_executed_path
    audit.emitted = True
    payload = {
        "utterance": audit.utterance,
        "router_action": audit.router_action,
        "router_confidence": audit.router_confidence,
        "router_requires_confirmation": audit.router_requires_confirmation,
        "allowlist_result": audit.allowlist_result,
        "legacy_command": audit.legacy_command,
        "legacy_match_type": audit.legacy_match_type,
        "final_executed_path": audit.final_executed_path,
        "completed": audit.completed,
        "handler_error": audit.handler_error,
        "spoken_text_present": audit.spoken_text_present,
    }
    _log.info("[action_router_audit] %s", json.dumps(payload, sort_keys=True))


def _safe_round_score(value: Any) -> Optional[float]:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean_trace_error(error: Any) -> Optional[str]:
    text = " ".join(str(error or "").strip().split())
    return text[:240] if text else None


def _new_character_loop_trace(
    text: str,
    *,
    from_idle_activation: bool,
    turn_start: float,
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    speaker_score: Optional[float],
) -> Optional[_CharacterLoopTrace]:
    if not bool(getattr(config, "CHARACTER_LOOP_TRACE_ENABLED", True)):
        return None
    return _CharacterLoopTrace(
        turn_id=next(_character_loop_turn_ids),
        utterance=str(text or ""),
        heard_text=str(text or ""),
        from_idle_activation=bool(from_idle_activation),
        turn_start=turn_start,
        raw_best_id=_safe_int(raw_best_id),
        raw_best_name=raw_best_name,
        speaker_score=_safe_round_score(speaker_score),
    )


def _infer_identity_resolution_strategy(
    *,
    person_id: Optional[int],
    raw_best_id: Optional[int],
    speaker_score: Optional[float],
    hard_threshold: float,
    soft_threshold: float,
    sticky_accepted: bool,
    ws_identified: Optional[list[Any]],
    recent_engagement: Optional[dict],
    off_camera_unknown: bool,
) -> str:
    if person_id is None:
        return "off_camera_unknown" if off_camera_unknown else "unresolved"
    if sticky_accepted:
        return "voice_soft_session_sticky"

    resolved_id = _safe_int(person_id)
    raw_id = _safe_int(raw_best_id)
    score = float(speaker_score or 0.0)
    if raw_id is not None and resolved_id == raw_id:
        if score >= hard_threshold:
            return "voice_hard_match"
        if score >= soft_threshold:
            return "voice_soft_contextual_match"
        return "voice_face_contextual_match"

    try:
        visible_ids = {
            int(p.get("person_db_id"))
            for p in (ws_identified or [])
            if getattr(p, "get", None) and p.get("person_db_id") is not None
        }
    except Exception:
        visible_ids = set()
    if resolved_id in visible_ids:
        return "visible_worldstate_context"

    recent_id = _safe_int((recent_engagement or {}).get("person_id"))
    if recent_id is not None and recent_id == resolved_id:
        return "recent_engagement_context"
    return "resolved"


def _single_visible_face_voice_override(
    *,
    resolved_person_id: Optional[int],
    ws_person: Optional[dict],
    visible_known_by_id: dict,
    has_unknown_visible_or_recent: bool,
    speaker_score: float = 0.0,
    hard_threshold: Optional[float] = None,
) -> Optional[tuple[int, Optional[str]]]:
    """
    Resolve voice/face disagreement in favor of the only visible known face.

    Speaker ID is useful, but short commands can false-match. When the camera has
    exactly one known face and no other visible/recent unknown, that face is the
    most reliable identity signal for the current turn.
    """
    resolved_id = _safe_int(resolved_person_id)
    if resolved_id is None:
        return None
    if not isinstance(ws_person, dict):
        return None
    ws_pid = _safe_int(ws_person.get("person_db_id"))
    if ws_pid is None or resolved_id == ws_pid:
        return None
    if len(visible_known_by_id or {}) != 1:
        return None
    if ws_pid not in (visible_known_by_id or {}):
        return None
    if bool(has_unknown_visible_or_recent):
        return None
    threshold = float(
        hard_threshold
        if hard_threshold is not None
        else getattr(config, "SPEAKER_ID_SIMILARITY_THRESHOLD", 0.75)
    )
    if float(speaker_score or 0.0) >= threshold:
        return None
    return ws_pid, (ws_person.get("face_id") or ws_person.get("voice_id"))


def _character_loop_note_speaker(
    trace: Optional[_CharacterLoopTrace],
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    speaker_label: Optional[str],
    identity_resolution: str,
    off_camera_unknown: bool,
    visible_known_count: Optional[int],
    has_unknown_visible_or_recent: Optional[bool],
    anonymous_speaker_label: Optional[str] = None,
    anonymous_speaker_match_score: Optional[float] = None,
) -> None:
    if trace is None:
        return
    trace.person_id = _safe_int(person_id)
    trace.person_name = person_name
    trace.speaker_label = speaker_label
    trace.anonymous_speaker_label = anonymous_speaker_label
    trace.anonymous_speaker_match_score = _safe_round_score(anonymous_speaker_match_score)
    trace.identity_resolution = identity_resolution
    trace.off_camera_unknown = bool(off_camera_unknown)
    trace.visible_known_count = (
        _safe_int(visible_known_count) if visible_known_count is not None else None
    )
    trace.has_unknown_visible_or_recent = (
        bool(has_unknown_visible_or_recent)
        if has_unknown_visible_or_recent is not None
        else None
    )


def _character_loop_note_router_audit(
    trace: Optional[_CharacterLoopTrace],
    audit: Optional[_RouterDecisionAudit],
) -> None:
    if trace is None or audit is None:
        return
    trace.router_action = audit.router_action
    trace.router_confidence = audit.router_confidence
    trace.router_requires_confirmation = audit.router_requires_confirmation
    trace.allowlist_result = audit.allowlist_result
    trace.legacy_command = audit.legacy_command
    trace.legacy_match_type = audit.legacy_match_type
    if audit.final_executed_path:
        trace.final_executed_path = audit.final_executed_path
    if audit.completed is not None:
        trace.completed = audit.completed
    if audit.handler_error:
        trace.handler_error = audit.handler_error
    if audit.spoken_text_present is not None:
        trace.spoken_text_present = audit.spoken_text_present


def _log_character_loop_trace(
    trace: Optional[_CharacterLoopTrace],
    *,
    router_audit: Optional[_RouterDecisionAudit] = None,
    final_executed_path: Optional[str] = None,
    completed: Optional[bool] = None,
    handler_error: Optional[str] = None,
    spoken_text: Optional[str] = None,
    spoken_text_present: Optional[bool] = None,
    assistant_asked_question: Optional[bool] = None,
    suppress_memory_learning: Optional[bool] = None,
    repair_move: Optional[dict] = None,
    intent: Optional[str] = None,
) -> None:
    if trace is None or trace.emitted:
        return
    if not bool(getattr(config, "CHARACTER_LOOP_TRACE_ENABLED", True)):
        return

    _character_loop_note_router_audit(trace, router_audit)
    if final_executed_path:
        trace.final_executed_path = final_executed_path
    if completed is not None:
        trace.completed = bool(completed)
    if handler_error:
        trace.handler_error = _clean_trace_error(handler_error)
    if spoken_text_present is not None:
        trace.spoken_text_present = bool(spoken_text_present)
    elif spoken_text is not None:
        trace.spoken_text_present = bool(str(spoken_text).strip())
    if assistant_asked_question is not None:
        trace.assistant_asked_question = bool(assistant_asked_question)
    if suppress_memory_learning is not None:
        trace.suppress_memory_learning = bool(suppress_memory_learning)
    if intent:
        trace.intent = str(intent)
    if repair_move:
        trace.repair_kind = str(repair_move.get("kind") or "unknown")
    if trace.completed is None and trace.spoken_text_present is not None:
        trace.completed = trace.spoken_text_present

    payload = {
        "turn_id": trace.turn_id,
        "utterance": trace.utterance,
        "heard_text": trace.heard_text,
        "duration_ms": int(max(0.0, time.monotonic() - trace.turn_start) * 1000),
        "speaker": {
            "person_id": trace.person_id,
            "name": trace.person_name,
            "label": trace.speaker_label,
            "identity_resolution": trace.identity_resolution,
            "from_idle_activation": trace.from_idle_activation,
            "off_camera_unknown": trace.off_camera_unknown,
            "visible_known_count": trace.visible_known_count,
            "has_unknown_visible_or_recent": trace.has_unknown_visible_or_recent,
            "raw_candidate": {
                "person_id": trace.raw_best_id,
                "name": trace.raw_best_name,
                "score": trace.speaker_score,
            },
            "anonymous_speaker": {
                "label": trace.anonymous_speaker_label,
                "match_score": trace.anonymous_speaker_match_score,
            },
        },
        "interpretation": {
            "router_action": trace.router_action,
            "router_confidence": trace.router_confidence,
            "router_requires_confirmation": trace.router_requires_confirmation,
            "allowlist_result": trace.allowlist_result,
            "legacy_command": trace.legacy_command,
            "legacy_match_type": trace.legacy_match_type,
            "intent": trace.intent,
            "repair_kind": trace.repair_kind,
        },
        "execution": {
            "final_executed_path": trace.final_executed_path,
            "completed": trace.completed,
            "handler_error": trace.handler_error,
            "spoken_text_present": trace.spoken_text_present,
            "assistant_asked_question": trace.assistant_asked_question,
            "suppress_memory_learning": trace.suppress_memory_learning,
        },
        "timing": _ttfs_timing_payload(trace),
    }
    trace.emitted = True
    _log.info("[character_loop] %s", json.dumps(payload, sort_keys=True))


def _router_audit_fast_local_final_path(audit: Optional[_RouterDecisionAudit]) -> str:
    if audit is None:
        return "fast_local_takeover.unknown"
    if audit.router_action and audit.allowlist_result == "allowed":
        return f"fast_local_takeover.{audit.router_action}"
    if audit.legacy_command:
        return f"legacy_command.{audit.legacy_command}"
    if audit.router_action:
        return f"fast_local_takeover.{audit.router_action}"
    return "fast_local_takeover.unknown"


def _chunk_for_vad(audio_chunk: np.ndarray) -> np.ndarray:
    """Apply playback suppression before VAD so background audio does not self-trigger."""
    try:
        return echo_cancel.filter(audio_chunk)
    except Exception:
        return audio_chunk


def _duck_dj_for_speech() -> Optional[float]:
    if not bool(getattr(config, "DJ_DUCK_DURING_SPEECH", True)):
        return None
    try:
        from features import dj as dj_mod
        if not dj_mod.is_playing():
            return None
        original = float(dj_mod.get_volume())
        ducked = min(original, float(getattr(config, "DJ_LISTEN_DUCK_VOLUME", 0.18)))
        if ducked < original:
            dj_mod.set_volume(ducked)
            return original
    except Exception as exc:
        _log.debug("DJ duck failed: %s", exc)
    return None


def _restore_dj_volume(volume: Optional[float]) -> None:
    if volume is None:
        return
    try:
        from features import dj as dj_mod
        if dj_mod.is_playing():
            dj_mod.set_volume(volume)
    except Exception as exc:
        _log.debug("DJ volume restore failed: %s", exc)


def _speak_blocking(
    text: str,
    emotion: str = "neutral",
    priority: int = 1,
    pre_beat_ms: int = 0,
    post_beat_ms_override: int = 0,
    voice_settings: Optional[dict] = None,
    log_text: bool = True,
) -> bool:
    """
    Enqueue text for speech and block until playback finishes, monitoring for
    wake-word interruption.  Returns True on normal completion, False if cut short.

    priority 1 = normal response; priority 2 = urgent acknowledgment.
    Enqueueing drops all waiting items of lower priority and preempts any
    currently-playing item of lower priority.

    `post_beat_ms_override` is used by the empathy delivery layer to extend
    the post-punchline beat for sympathetic modes (listen/support/etc.) where
    we want the line to settle rather than snap into the next exchange. The
    override takes precedence over the default punchline beat when larger.
    """
    if not _can_speak() or not text or not text.strip():
        return True
    text = llm.clean_response_text(text)
    if not text:
        return True

    # Post-punchline beat: a brief silence after a normal-priority response so
    # the line lands. Skipped for urgent acks (priority >= 2), filler, etc.
    post_beat_ms = 0
    if priority == 1:
        beat_min = getattr(config, "POST_PUNCHLINE_BEAT_MS_MIN", 0)
        beat_max = getattr(config, "POST_PUNCHLINE_BEAT_MS_MAX", 0)
        if beat_max > 0 and beat_max >= beat_min:
            post_beat_ms = random.randint(beat_min, beat_max)
    if post_beat_ms_override > 0:
        post_beat_ms = max(post_beat_ms, post_beat_ms_override)
    asked_question = _assistant_asked_question(text)
    if asked_question:
        post_beat_ms = 0

    trace = _current_character_loop_trace.get()
    _mark_first_response_queued(trace, text=text, priority=priority)

    def _on_playback_start() -> None:
        _mark_first_response_audio_started(trace)

    done = speech_queue.enqueue(
        text, emotion, priority=priority,
        pre_beat_ms=pre_beat_ms, post_beat_ms=post_beat_ms,
        voice_settings=voice_settings,
        on_start=_on_playback_start if trace is not None else None,
        log_text=log_text,
    )

    while not done.wait(timeout=0.05):
        if _interrupted.is_set():
            try:
                import sounddevice as sd
                echo_cancel.request_cancel()
                sd.stop()
            except Exception:
                pass
            # Drop our queued item in case it hasn't played yet, then wait for
            # whatever was playing (or our item if it just started) to finish.
            speech_queue.clear_below_priority(priority + 1)
            done.wait(timeout=0.5)
            return False

    # Normal completion — arm the post-TTS handoff. Any spoken Rex line may
    # receive an immediate human reply, so preserve the rolling mic buffer.
    _apply_post_tts_handoff(text, source="blocking")
    return True


def _speak_proactive(
    text: str,
    *,
    emotion: str = "neutral",
    priority: int = 1,
    voice_settings: Optional[dict] = None,
    pre_beat_ms: int = 0,
    post_beat_ms_override: int = 0,
    label: str = "proactive",
) -> bool:
    """Speak a self-initiated line, but yield the floor if the user has started.

    Unlike a direct reply (where the user just spoke), a proactive line is decided
    on a silence timer and then spends ~1-2s in LLM + TTS generation before any
    sound — long enough for the user to begin talking unseen, after which Rex used
    to play right over them. This pre-caches the audio so playback is instant, then
    re-checks the mic immediately before the sound: if the user is already
    speaking, it returns False WITHOUT speaking and the normal loop captures their
    turn from the (un-attenuated) rolling buffer. Falls back to plain blocking
    speech when the guard is disabled or in text-only mode.
    """
    if not text or not text.strip():
        return False
    if bool(getattr(config, "PROACTIVE_SPEECH_YIELD_ENABLED", True)) and not _text_only_mode:
        # Collapse the check->sound gap: get the audio ready first so the mic
        # check below happens right before playback, not ~1s before it.
        try:
            from audio import tts
            tts.ensure_cached(text, voice_settings=voice_settings, emotion=emotion)
        except Exception as exc:
            _log.debug("[interaction] proactive pre-cache failed: %s", exc)
        try:
            if barge_guard.user_speaking_now():
                _log.info(
                    "[interaction] proactive line yielded — user already speaking (%s): %r",
                    label,
                    text,
                )
                return False
        except Exception as exc:
            _log.debug("[interaction] proactive yield check failed: %s", exc)
    completed = _speak_blocking(
        text,
        emotion=emotion,
        priority=priority,
        pre_beat_ms=pre_beat_ms,
        post_beat_ms_override=post_beat_ms_override,
        voice_settings=voice_settings,
    )
    if completed:
        global _last_proactive_line_at
        _last_proactive_line_at = time.monotonic()
    return completed


def _proactive_line_recently_fired(min_gap: Optional[float] = None) -> bool:
    """True if any proactive line played within the last ``min_gap`` seconds, so
    a second one should hold off instead of stacking (one line, then wait)."""
    if min_gap is None:
        min_gap = float(getattr(config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0) or 0.0)
    if min_gap <= 0 or _last_proactive_line_at <= 0:
        return False
    return (time.monotonic() - _last_proactive_line_at) < min_gap


def _post_tts_handoff_policy(text: Optional[str]) -> _PostTtsHandoffPolicy:
    asked_question = _assistant_asked_question(text or "")
    fast_response_expected = _fast_response_handoff_expected(text)
    delay_secs = float(getattr(config, "POST_SPEECH_LISTEN_DELAY_SECS", 0.35))
    # Statements no longer flush by default: a reply that began as Rex finished a
    # statement sits (un-attenuated) in the rolling buffer, and flushing deleted it.
    flush_buffer = bool(getattr(config, "POST_SPEECH_FLUSH_AUDIO_BUFFER", False))
    if fast_response_expected:
        delay_secs = float(
            getattr(config, "POST_QUESTION_LISTEN_DELAY_SECS", delay_secs)
        )
        flush_buffer = bool(
            getattr(config, "POST_QUESTION_FLUSH_AUDIO_BUFFER", False)
        )
    return _PostTtsHandoffPolicy(
        asked_question=asked_question,
        fast_response_expected=fast_response_expected,
        listen_delay_secs=max(0.0, delay_secs),
        flush_buffer=flush_buffer,
    )


def _apply_post_tts_handoff(
    text: Optional[str],
    *,
    source: str = "speech_queue",
) -> _PostTtsHandoffPolicy:
    global _listen_resume_at, _listen_capture_floor_at, _post_tts_flush_needed, _last_speech_at
    global _last_fast_handoff_at
    policy = _post_tts_handoff_policy(text)
    now = time.monotonic()

    # Sticky responsiveness: a streamed reply fires this handoff per sentence and
    # again for the whole reply. If Rex asked a question but his trailing sentence
    # is a statement, that last handoff would flip back to the long flush window
    # and delete the human's immediate answer. Keep the responsive (short,
    # no-flush) window for a brief sticky window after any question handoff,
    # regardless of which handoff lands last.
    sticky_secs = float(getattr(config, "POST_QUESTION_HANDOFF_STICKY_SECS", 0.0) or 0.0)
    if policy.fast_response_expected:
        _last_fast_handoff_at = now
    elif sticky_secs > 0.0 and (now - _last_fast_handoff_at) <= sticky_secs:
        policy = _PostTtsHandoffPolicy(
            asked_question=policy.asked_question,
            fast_response_expected=True,
            listen_delay_secs=min(
                policy.listen_delay_secs,
                float(getattr(config, "POST_QUESTION_LISTEN_DELAY_SECS", 0.12)),
            ),
            flush_buffer=bool(getattr(config, "POST_QUESTION_FLUSH_AUDIO_BUFFER", False)),
        )
        _log.debug(
            "[aec] post-tts handoff kept responsive (sticky question window, source=%s)",
            source,
        )

    _last_speech_at = now
    aec_on = hardware_aec.is_active()
    listen_delay = policy.listen_delay_secs
    if aec_on:
        # Hardware AEC removed Rex's voice from the mic — resume listening almost
        # immediately so a reply landing as he finishes is not missed.
        listen_delay = min(
            listen_delay,
            float(getattr(config, "POST_TTS_LISTEN_DELAY_SECS_AEC", 0.05)),
        )
    _listen_resume_at = now + listen_delay
    if policy.fast_response_expected:
        grace = max(
            0.0,
            float(
                getattr(
                    config,
                    "POST_QUESTION_CAPTURE_PREROLL_GRACE_SECS",
                    getattr(config, "POST_TTS_CAPTURE_PREROLL_GRACE_SECS", 0.0),
                )
                or 0.0
            ),
        )
    else:
        grace = max(
            0.0,
            float(getattr(config, "POST_TTS_CAPTURE_PREROLL_GRACE_SECS", 0.0) or 0.0),
        )
    if aec_on:
        # Safe to reach further back past the handoff to recover an overlapping
        # reply: that audio is hardware-AEC'd, so Rex's tail in it is ~16 dB down.
        grace = max(grace, float(getattr(config, "POST_TTS_CAPTURE_PREROLL_GRACE_SECS_AEC", 0.5)))
    _listen_capture_floor_at = max(0.0, now - grace)
    if policy.flush_buffer:
        stream.flush()
        _post_tts_flush_needed = True
    else:
        _post_tts_flush_needed = False
    try:
        vad.reset_state()
    except Exception as exc:
        _log.debug("[aec] VAD reset after post-TTS handoff failed: %s", exc)
    _log.debug(
        "[aec] post-tts handoff source=%s asked_question=%s fast_response=%s delay=%.3fs flush=%s",
        source,
        policy.asked_question,
        policy.fast_response_expected,
        policy.listen_delay_secs,
        policy.flush_buffer,
    )
    return policy


def _reply_playback_tail_secs(asked_question: bool) -> float:
    """Mic-attenuation tail to hold after a spoken reply ends.

    Questions and statements both invite immediate replies, so both use a short
    tail — long enough to let room echo of Rex's own voice decay, short enough that
    a human reply starting as Rex finishes is detected promptly. Falls back to the
    general post-playback tail if the specific knob is unset.

    When the ReSpeaker Lite's hardware AEC is active it has already removed Rex's
    voice from the mic, so the tail only needs to cover the speaker's mechanical
    settle — shrink it so a reply landing as Rex finishes is not attenuated away.
    """
    if hardware_aec.is_active():
        return float(getattr(config, "POST_PLAYBACK_SUPPRESSION_SECS_AEC", 0.05))
    general = float(getattr(config, "POST_PLAYBACK_SUPPRESSION_SECS", 0.5))
    if asked_question:
        return float(getattr(config, "POST_QUESTION_PLAYBACK_SUPPRESSION_SECS", general))
    return float(getattr(config, "POST_SPEECH_PLAYBACK_SUPPRESSION_SECS", general))


def _end_response_sequence_for_text(text: Optional[str]) -> None:
    policy = _post_tts_handoff_policy(text)
    tail_secs = _reply_playback_tail_secs(policy.fast_response_expected)
    echo_cancel.end_sequence(flush=policy.flush_buffer, tail_secs=tail_secs)


def _arm_post_tts_window(item=None) -> None:
    """Arm the post-TTS deaf window. Registered with speech_queue so it fires
    after every queue item — not just items played via _speak_blocking."""
    text = getattr(item, "text", None)
    _apply_post_tts_handoff(text, source="speech_queue")


def _speak_async(text: str, emotion: str = "neutral") -> None:
    if not _can_speak() or not text:
        return
    text = llm.clean_response_text(text)
    if not text:
        return
    if speech_queue.is_speaking():
        return
    speech_queue.enqueue(text, emotion, priority=0)


def _audio_group_chatter_active() -> bool:
    if not bool(getattr(config, "GROUP_CHATTER_ENABLED", True)):
        return False
    try:
        scene = world_state.get("audio_scene") or {}
    except Exception:
        return False
    until = scene.get("group_chatter_until")
    if until is not None:
        try:
            if time.time() <= float(until):
                return True
            scene["group_chatter_detected"] = False
            scene["group_chatter_until"] = None
            scene["group_chatter_reason"] = None
            world_state.update("audio_scene", scene)
            return False
        except (TypeError, ValueError):
            pass
    return bool(scene.get("group_chatter_detected"))


def _set_audio_group_chatter(reason: str) -> None:
    if not bool(getattr(config, "GROUP_CHATTER_ENABLED", True)):
        return
    try:
        scene = world_state.get("audio_scene") or {}
        hold = float(getattr(config, "GROUP_CHATTER_HOLD_SECS", 6.0))
        scene["group_chatter_detected"] = True
        scene["group_chatter_until"] = time.time() + max(0.0, hold)
        scene["group_chatter_reason"] = reason
        world_state.update("audio_scene", scene)
    except Exception as exc:
        _log.debug("group chatter world_state update failed: %s", exc)


def _voice_turn_label(
    *,
    person_id: Optional[int],
    raw_best_id: Optional[int],
    raw_best_score: float,
) -> str:
    if person_id is not None:
        return f"known:{person_id}"
    candidate_floor = float(getattr(config, "GROUP_CHATTER_VOICE_CANDIDATE_FLOOR", 0.30))
    if raw_best_id is not None and raw_best_score >= candidate_floor:
        return f"candidate:{raw_best_id}"
    return "unknown"


def _note_voice_turn_for_group_chatter(
    *,
    person_id: Optional[int],
    raw_best_id: Optional[int],
    raw_best_score: float,
) -> bool:
    """
    Mark group chatter when the raw voice candidate changes repeatedly.

    Resemblyzer only compares against known prints, so unknown adults may appear
    as low-confidence swings between known candidates. Those swings are exactly
    the useful signal here: not "who is this?", but "there are multiple voices."
    """
    if not bool(getattr(config, "GROUP_CHATTER_ENABLED", True)):
        return False

    now = time.time()
    window = float(getattr(config, "GROUP_CHATTER_VOICE_WINDOW_SECS", 10.0))
    low_conf_max = float(getattr(config, "GROUP_CHATTER_VOICE_LOW_CONF_MAX", 0.62))
    label = _voice_turn_label(
        person_id=person_id,
        raw_best_id=raw_best_id,
        raw_best_score=raw_best_score,
    )
    low_confidence = person_id is None or raw_best_score <= low_conf_max

    _recent_voice_turns.append((now, label, low_confidence))
    cutoff = now - max(0.0, window)
    while _recent_voice_turns and _recent_voice_turns[0][0] < cutoff:
        _recent_voice_turns.popleft()

    labels = [entry[1] for entry in _recent_voice_turns]
    low_conf_count = sum(1 for entry in _recent_voice_turns if entry[2])
    changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    if (
        len(labels) >= int(getattr(config, "GROUP_CHATTER_VOICE_MIN_TURNS", 3))
        and changes >= int(getattr(config, "GROUP_CHATTER_VOICE_MIN_CHANGES", 2))
        and low_conf_count >= 2
    ):
        _set_audio_group_chatter("rapid_voice_changes")
        _log.info(
            "[interaction] group chatter detected from voice changes "
            "(turns=%d changes=%d labels=%s raw_score=%.3f)",
            len(labels), changes, labels[-6:], raw_best_score,
        )
        return True
    return False


def _normalize_voice_embedding(embedding: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-10:
        return None
    return (arr / norm).astype(np.float32)


def _clear_anonymous_speaker_slots() -> None:
    global _anonymous_speaker_next_id
    _anonymous_speaker_slots.clear()
    _anonymous_speaker_next_id = 1


def _retire_anonymous_speaker_slot(
    label: Optional[str],
    *,
    person_id: Optional[int] = None,
    person_name: Optional[str] = None,
) -> None:
    if not label:
        return
    before = len(_anonymous_speaker_slots)
    _anonymous_speaker_slots[:] = [
        slot for slot in _anonymous_speaker_slots if slot.label != label
    ]
    if len(_anonymous_speaker_slots) != before:
        _log.info(
            "[anonymous_speaker] retired label=%s person_id=%s name=%r",
            label,
            person_id,
            person_name,
        )


def _resolve_anonymous_speaker_slot(
    audio_array: np.ndarray,
    *,
    person_id: Optional[int],
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    raw_best_score: float,
) -> tuple[Optional[str], Optional[float]]:
    """Return a session-only label for an unresolved recurring voice."""
    global _anonymous_speaker_next_id
    if person_id is not None:
        return None, None
    if not bool(getattr(config, "ANONYMOUS_SPEAKER_SLOTS_ENABLED", True)):
        return None, None

    try:
        embedding = _normalize_voice_embedding(speaker_id.get_embedding(audio_array))
    except Exception as exc:
        _log.debug("[anonymous_speaker] embedding failed: %s", exc)
        return None, None
    if embedding is None:
        return None, None

    now = time.time()
    threshold = float(getattr(config, "ANONYMOUS_SPEAKER_SLOT_MATCH_THRESHOLD", 0.74))
    sticky_threshold = float(
        getattr(config, "ANONYMOUS_SPEAKER_SLOT_STICKY_THRESHOLD", 0.70)
    )
    best_slot: Optional[_AnonymousSpeakerSlot] = None
    best_score: Optional[float] = None
    for slot in _anonymous_speaker_slots:
        if slot.embedding.shape != embedding.shape:
            continue
        score = float(np.dot(slot.embedding, embedding))
        if best_score is None or score > best_score:
            best_score = score
            best_slot = slot

    raw_id = _safe_int(raw_best_id)
    raw_candidate_sticky = (
        best_slot is not None
        and best_score is not None
        and raw_id is not None
        and best_slot.raw_best_id is not None
        and raw_id == best_slot.raw_best_id
        and best_score >= sticky_threshold
    )
    if (
        best_slot is not None
        and best_score is not None
        and (best_score >= threshold or raw_candidate_sticky)
    ):
        blend_weight = float(min(max(best_slot.turns, 1), 5))
        blended = _normalize_voice_embedding(
            (best_slot.embedding * blend_weight) + embedding
        )
        if blended is not None:
            best_slot.embedding = blended
        best_slot.turns += 1
        best_slot.last_seen_at = now
        best_slot.raw_best_id = _safe_int(raw_best_id)
        best_slot.raw_best_name = raw_best_name
        best_slot.raw_best_score = _safe_round_score(raw_best_score)
        score_out = _safe_round_score(best_score)
        _log.info(
            "[anonymous_speaker] matched label=%s score=%.3f turns=%d raw_candidate=%s/%r raw_score=%.3f reason=%s",
            best_slot.label,
            float(best_score),
            best_slot.turns,
            raw_best_id,
            raw_best_name,
            float(raw_best_score or 0.0),
            "raw_candidate_sticky" if raw_candidate_sticky else "threshold",
        )
        return best_slot.label, score_out

    max_slots = int(getattr(config, "ANONYMOUS_SPEAKER_SLOT_MAX", 8) or 0)
    if max_slots <= 0 or len(_anonymous_speaker_slots) >= max_slots:
        _log.info(
            "[anonymous_speaker] no slot available max=%d best_score=%s raw_candidate=%s/%r",
            max_slots,
            f"{best_score:.3f}" if best_score is not None else None,
            raw_best_id,
            raw_best_name,
        )
        return None, _safe_round_score(best_score)

    label = f"unknown_voice_{_anonymous_speaker_next_id}"
    _anonymous_speaker_next_id += 1
    _anonymous_speaker_slots.append(
        _AnonymousSpeakerSlot(
            label=label,
            embedding=embedding,
            first_seen_at=now,
            last_seen_at=now,
            raw_best_id=_safe_int(raw_best_id),
            raw_best_name=raw_best_name,
            raw_best_score=_safe_round_score(raw_best_score),
        )
    )
    _log.info(
        "[anonymous_speaker] created label=%s raw_candidate=%s/%r raw_score=%.3f best_existing=%s",
        label,
        raw_best_id,
        raw_best_name,
        float(raw_best_score or 0.0),
        f"{best_score:.3f}" if best_score is not None else None,
    )
    return label, None


def _format_current_time_response(now: Optional[datetime] = None) -> str:
    """Return a deterministic clock answer without an LLM round trip."""
    now = now or datetime.now()
    h12 = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    return f"It's {h12}:{now.minute:02d} {ampm}."


def _format_current_date_response(now: Optional[datetime] = None) -> str:
    """Return a deterministic date answer without an LLM round trip."""
    now = now or datetime.now()
    return f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}."


_DATE_QUERY_PAT = re.compile(
    r"\b(date|today|today's|todays|day of week)\b|^\s*what\s+day\b",
    re.IGNORECASE,
)


def _looks_like_date_query(text: str) -> bool:
    return bool(_DATE_QUERY_PAT.search(text or ""))


def _wake_ack() -> None:
    global _last_wake_ack
    pool = list(getattr(config, "WAKE_ACKNOWLEDGMENTS", []) or [])
    if not pool:
        return
    candidates = [line for line in pool if line != _last_wake_ack] or pool
    chosen = random.choice(candidates)
    if getattr(config, "WAKE_ACK_REQUIRE_CACHE", True):
        try:
            from audio import tts
            if not tts.is_cached(chosen):
                _log.warning(
                    "[wake_word] wake ack skipped because TTS is not cached: %r",
                    chosen,
                )
                return
        except Exception as exc:
            _log.debug("[wake_word] wake ack cache check failed: %s", exc)
            return
    _last_wake_ack = chosen
    _speak_blocking(chosen, priority=2)


def _choose_transition_line(config_name: str, fallback: str, last_line: Optional[str]) -> str:
    pool = [
        str(line).strip()
        for line in getattr(config, config_name, []) or []
        if str(line).strip()
    ]
    if not pool:
        return fallback
    candidates = [line for line in pool if line != last_line] or pool
    return random.choice(candidates)


def _sleep_mode_line() -> str:
    global _last_sleep_mode_ack
    line = _choose_transition_line(
        "SLEEP_MODE_ACKNOWLEDGMENTS",
        "Fine. Power nap mode. Wake me when the galaxy gets less weird.",
        _last_sleep_mode_ack,
    )
    _last_sleep_mode_ack = line
    return line


def _wake_from_sleep_line() -> str:
    global _last_wake_from_sleep_ack
    line = _choose_transition_line(
        "WAKE_FROM_SLEEP_ACKNOWLEDGMENTS",
        "I'm up. I was dreaming in binary and somehow still got interrupted.",
        _last_wake_from_sleep_ack,
    )
    _last_wake_from_sleep_ack = line
    return line


def _clear_listening_state_for_sleep() -> None:
    global _listen_resume_at, _listen_capture_floor_at, _post_tts_flush_needed
    try:
        speech_queue.clear_below_priority(999)
    except Exception as exc:
        _log.debug("[sleep_mode] speech queue cleanup failed: %s", exc)
    try:
        stream.flush()
    except Exception as exc:
        _log.debug("[sleep_mode] stream flush failed: %s", exc)
    try:
        vad.reset_state()
    except Exception as exc:
        _log.debug("[sleep_mode] VAD reset failed: %s", exc)
    try:
        _situation_assessor.set_vad_active(False)
    except Exception:
        pass
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass
    _wake_word_fired.clear()
    _interrupted.clear()
    _listen_resume_at = 0.0
    _listen_capture_floor_at = 0.0
    _post_tts_flush_needed = False


def _run_sleep_animation() -> None:
    try:
        from sequences import animations
        animations.sleep()
    except Exception as exc:
        _log.warning("[sleep_mode] sleep animation failed: %s", exc)


def _run_wake_animation() -> None:
    try:
        from sequences import animations
        animations.wake()
    except Exception as exc:
        _log.warning("[sleep_mode] wake animation failed: %s", exc)


def _enter_sleep_mode() -> str:
    resp = _sleep_mode_line()
    _speak_blocking(resp, emotion="sleepy")
    _clear_listening_state_for_sleep()
    state_module.set_state(State.SLEEP)
    _run_sleep_animation()
    return resp


def _wake_from_sleep() -> str:
    _clear_listening_state_for_sleep()
    state_module.set_state(State.ACTIVE)
    _run_wake_animation()
    resp = _wake_from_sleep_line()
    _speak_blocking(resp, emotion="happy", priority=2, pre_beat_ms=80, post_beat_ms_override=150)
    return resp


def _wake_from_quiet() -> None:
    """Exit quiet mode when an explicit wake word gets Rex's attention."""
    state_module.set_state(State.ACTIVE)
    _wake_ack()


def _vad_barge_in_enabled() -> bool:
    return bool(getattr(config, "VAD_BARGE_IN_ENABLED", False))


def _is_interruptible_game_audio_path(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        from features import games as games_mod
        if not games_mod.is_active():
            return False
    except Exception:
        return False
    try:
        path_obj = Path(str(path))
    except Exception:
        return False
    return path_obj.parent.name == "jeopardy"


def _response_wait_active() -> bool:
    try:
        return bool(consciousness.is_waiting_for_response())
    except Exception as exc:
        _log.debug("[wake_word] response-wait check failed: %s", exc)
        return False


def _should_play_active_wake_ack() -> bool:
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False
    if _response_wait_active():
        return False
    return True


_BARE_WAKE_ADDRESS_PAT = re.compile(
    r"^\s*(?:"
    r"hey\s+(?:dj[\s-]*)?r(?:ex|3x)|"
    r"hi\s+(?:dj[\s-]*)?r(?:ex|3x)|"
    r"hello\s+(?:dj[\s-]*)?r(?:ex|3x)|"
    r"yo\s+(?:robot|(?:dj[\s-]*)?r(?:ex|3x))|"
    r"(?:dj[\s-]*)?r(?:ex|3x)"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_PLAIN_IDLE_GREETING_PAT = re.compile(
    r"^\s*(?:hello|hi|hey|yo|you there)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _is_bare_wake_address(text: str) -> bool:
    return bool(_BARE_WAKE_ADDRESS_PAT.match(text or ""))


def _is_sleep_wake_transcript(text: str) -> bool:
    """Return True for explicit transcribed phrases that should exit sleep."""
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    cleaned = re.sub(r"[^a-z0-9]+", " ", raw).strip()
    compact = re.sub(r"[^a-z0-9]", "", raw)
    if compact in {
        "wakeuprex",
        "wakeupr3x",
        "wakeuprx",
        "rexwakeup",
        "r3xwakeup",
        "rxwakeup",
    }:
        return True
    rex_name = r"(?:d\s*j\s+)?(?:rex|r\s*3\s*x|r3x|rx)"
    wake_prefix = r"(?:(?:hey|yo)\s+)?(?:please\s+)?"
    wake_suffix = r"(?:\s+please)?"
    return bool(
        re.fullmatch(rf"{wake_prefix}wake\s+up\s+{rex_name}{wake_suffix}", cleaned)
        or re.fullmatch(rf"{wake_prefix}{rex_name}\s+wake\s+up{wake_suffix}", cleaned)
    )


def _shutdown_requested() -> bool:
    try:
        if _stop_event.is_set():
            return True
    except Exception:
        pass
    try:
        return state_module.get_state() == State.SHUTDOWN
    except Exception:
        return False


def _prefill_wake_ack_cache() -> None:
    """Warm the tiny wake-ack TTS set so wake feedback is instant."""
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        return
    if not getattr(config, "WAKE_ACK_REQUIRE_CACHE", True):
        return
    pool = (
        list(getattr(config, "WAKE_ACKNOWLEDGMENTS", []) or [])
        + list(getattr(config, "SLEEP_MODE_ACKNOWLEDGMENTS", []) or [])
        + list(getattr(config, "WAKE_FROM_SLEEP_ACKNOWLEDGMENTS", []) or [])
    )
    if not pool:
        return
    try:
        from audio import tts
        for line in pool:
            try:
                tts.ensure_cached(line)
            except Exception as exc:
                _log.debug("[wake_word] wake ack cache prefill failed for %r: %s", line, exc)
    except Exception as exc:
        _log.debug("[wake_word] wake ack cache prefill unavailable: %s", exc)


def _prefill_slow_path_ack_cache() -> None:
    """Warm slow-path acknowledgment TTS so live turns never fetch it."""
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        return
    if not bool(getattr(config, "SLOW_PATH_ACK_ENABLED", True)):
        return
    if not bool(getattr(config, "SLOW_PATH_ACK_REQUIRE_CACHE", True)):
        return
    pool = _all_slow_path_ack_lines()
    if not pool:
        return
    try:
        from audio import tts
        for line in pool:
            try:
                tts.ensure_cached(line)
            except Exception as exc:
                _log.debug(
                    "[slow_path_ack] cache prefill failed for %r: %s",
                    line,
                    exc,
                )
    except Exception as exc:
        _log.debug("[slow_path_ack] cache prefill unavailable: %s", exc)


def _interrupt_ack() -> None:
    _speak_blocking(random.choice(config.INTERRUPT_ACKNOWLEDGMENTS), priority=2)


def _speak_filler() -> None:
    """Speak a latency filler line asynchronously, never repeating back-to-back."""
    global _last_filler
    if not getattr(config, "LATENCY_FILLER_ENABLED", True):
        return
    pool = config.LATENCY_FILLER_LINES
    candidates = [l for l in pool if l != _last_filler] or pool
    chosen = random.choice(candidates)
    if getattr(config, "LATENCY_FILLER_REQUIRE_CACHE", True):
        try:
            from audio import tts
            if not tts.is_cached(chosen):
                _log.debug("[interaction] latency filler skipped — not cached: %r", chosen)
                return
        except Exception as exc:
            _log.debug("[interaction] latency filler cache check failed: %s", exc)
            return
    _last_filler = chosen
    speech_queue.enqueue(chosen, "neutral", priority=1, tag="latency_filler")


def _configured_slow_path_ack_lines(kind: str) -> list[str]:
    raw = getattr(config, "SLOW_PATH_ACK_LINES", {}) or {}
    lines: list[str] = []
    if isinstance(raw, dict):
        value = raw.get(kind) or raw.get("default") or []
        if isinstance(value, str):
            lines = [value]
        else:
            lines = [str(line) for line in value if str(line).strip()]
    elif isinstance(raw, str):
        lines = [raw]
    else:
        lines = [str(line) for line in raw if str(line).strip()]
    return [line.strip() for line in lines if line.strip()]


def _all_slow_path_ack_lines() -> list[str]:
    raw = getattr(config, "SLOW_PATH_ACK_LINES", {}) or {}
    lines: list[str] = []
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, str):
                lines.append(value)
            else:
                lines.extend(str(line) for line in value if str(line).strip())
    elif isinstance(raw, str):
        lines.append(raw)
    else:
        lines.extend(str(line) for line in raw if str(line).strip())
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        cleaned = str(line).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _simple_question(text: str) -> bool:
    cleaned = (text or "").strip()
    if "?" not in cleaned:
        return False
    return _word_count(cleaned) <= 8


def _slow_path_ack_expected_slow(kind: str) -> bool:
    try:
        minimum = float(getattr(config, "SLOW_PATH_ACK_MIN_EXPECTED_SECS", 1.5))
    except (TypeError, ValueError):
        minimum = 1.5
    raw = getattr(config, "SLOW_PATH_ACK_EXPECTED_SECS", {}) or {}
    expected = None
    if isinstance(raw, dict):
        expected = raw.get(kind, raw.get("default"))
    else:
        expected = raw
    try:
        return float(expected) >= minimum
    except (TypeError, ValueError):
        return True


def _slow_path_ack_allowed_for_turn(
    kind: str,
    text: str = "",
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> bool:
    if kind != "general":
        return True
    try:
        min_words = int(getattr(config, "SLOW_PATH_ACK_GENERAL_MIN_WORDS", 9) or 0)
    except (TypeError, ValueError):
        min_words = 9
    if min_words > 0 and _word_count(text) < min_words:
        return False
    if (
        not bool(getattr(config, "SLOW_PATH_ACK_GENERAL_ALLOW_SIMPLE_QUESTIONS", False))
        and _simple_question(text)
    ):
        return False
    if dialogue_decision is not None and dialogue_decision.label == "answer_to_rex":
        return False
    return True


def _try_slow_path_ack(
    kind: str,
    *,
    text: str = "",
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
) -> bool:
    """Queue a cached, non-blocking acknowledgment for a known slow path."""
    global _last_slow_path_ack
    if not bool(getattr(config, "SLOW_PATH_ACK_ENABLED", True)):
        return False
    if not _slow_path_ack_allowed_for_turn(kind, text, dialogue_decision):
        return False
    if not _slow_path_ack_expected_slow(kind):
        return False
    trace = _current_character_loop_trace.get()
    if trace is not None and trace.first_response_queued_at is not None:
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or _interrupted.is_set():
        return False

    pool = _configured_slow_path_ack_lines(kind)
    if not pool:
        return False
    candidates = [line for line in pool if line != _last_slow_path_ack] or pool
    chosen = random.choice(candidates)

    audio_suppressed = bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    )
    if audio_suppressed and not bool(getattr(config, "SLOW_PATH_ACK_IN_TEXT_ONLY", False)):
        return False
    if bool(getattr(config, "SLOW_PATH_ACK_REQUIRE_CACHE", True)) and not audio_suppressed:
        try:
            from audio import tts
            if not tts.is_cached(chosen):
                _log.debug(
                    "[slow_path_ack] skipped kind=%s because TTS is not cached: %r",
                    kind,
                    chosen,
                )
                return False
        except Exception as exc:
            _log.debug("[slow_path_ack] cache check failed kind=%s: %s", kind, exc)
            return False

    queued = _mark_first_response_queued(trace, text=chosen, priority=1)

    def _on_playback_start() -> None:
        _mark_first_response_audio_started(trace)

    try:
        speech_queue.enqueue(
            chosen,
            "neutral",
            priority=1,
            tag="slow_path_ack",
            on_start=_on_playback_start if trace is not None else None,
        )
        _last_slow_path_ack = chosen
        _log.info("[slow_path_ack] queued kind=%s text=%r", kind, chosen)
        return True
    except Exception as exc:
        _log.debug("[slow_path_ack] enqueue failed kind=%s: %s", kind, exc)
        return queued


def _start_latency_filler_timer() -> threading.Event:
    """Start a delayed filler timer and return an event that cancels it.

    Filler used to fire immediately for every utterance, including short grief
    flow replies. Delaying it keeps normal turns quiet while still covering
    genuinely slow LLM calls.
    """
    stop = threading.Event()
    if not getattr(config, "LATENCY_FILLER_ENABLED", True):
        stop.set()
        return stop

    delay = float(getattr(config, "LATENCY_FILLER_DELAY_SECS", 1.4))
    if delay <= 0:
        _speak_filler()
        stop.set()
        return stop

    trace = _current_character_loop_trace.get()

    def _timer() -> None:
        if stop.wait(delay):
            return
        if trace is not None and trace.first_response_queued_at is not None:
            return
        if (
            state_module.get_state() == State.ACTIVE
            and not speech_queue.is_speaking()
            and not output_gate.is_busy()
            and not _interrupted.is_set()
        ):
            _speak_filler()

    threading.Thread(target=_timer, daemon=True, name="latency-filler").start()
    return stop


_QUOTED_QUESTION_RE = re.compile(
    r"(?:“[^”]*\?[^”]*”|\"[^\"]*\?[^\"]*\")"
)
_RESPONSE_QUESTION_START_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should|want|wanna|need|got|any|care\s+to)\b",
    re.IGNORECASE,
)
_SHORT_SLOT_QUESTION_RE = re.compile(
    r"\b(?:last\s+name|surname)\b",
    re.IGNORECASE,
)
_IMPERATIVE_RESPONSE_PROMPT_RE = re.compile(
    r"\b(?:give|tell|toss|send|share)\s+(?:me\s+)?(?:your\s+|a\s+)?"
    r"(?:last\s+name|surname)\b|"
    r"\b(?:last\s+name|surname)\s+(?:too|please)\b",
    re.IGNORECASE,
)
def _strip_quoted_questions(text: str) -> str:
    return _QUOTED_QUESTION_RE.sub(
        lambda match: match.group(0).replace("?", ""),
        text or "",
    )


def _assistant_asked_question(text: str) -> bool:
    return _question_expects_response(text)


def _fast_response_handoff_expected(text: Optional[str]) -> bool:
    cleaned = _strip_quoted_questions(str(text or "")).strip()
    if not cleaned:
        return False
    return _question_expects_response(cleaned)


def _question_expects_response(text: str) -> bool:
    cleaned = _strip_quoted_questions(text).strip()
    if not cleaned:
        return False
    if "?" not in cleaned:
        return bool(_IMPERATIVE_RESPONSE_PROMPT_RE.search(cleaned))
    for part in re.split(r"(?<=[.!?])\s+", cleaned):
        if "?" in part and _question_sentence_expects_response(part):
            return True
    return False


def _question_sentence_expects_response(sentence: str) -> bool:
    lowered = (sentence or "").strip().lower()
    if not lowered or "?" not in lowered:
        return False
    if re.search(r"\b(right|okay|ok|huh|yeah)\?\s*$", lowered):
        return False
    if re.search(r"\bwhy\s+(?:risk|bother|would|not)\b.+\bwhen\b", lowered):
        return False
    # Rhetorical "who doesn't / who wouldn't / who hasn't ... ?" — Rex's own
    # flourish, not a real question. Don't arm the no-response quip on these
    # (it fired awkwardly after lines like "who doesn't appreciate a droid...?").
    if re.search(
        r"\bwho\s+(?:doesn'?t|does\s+not|wouldn'?t|would\s+not|hasn'?t|has\s+not|"
        r"isn'?t|ain'?t|can'?t)\b",
        lowered,
    ):
        return False
    # Rhetorical reformulations ("So what you're saying is …?", "So you're
    # telling me …?") restate the human's point for comic effect — they don't
    # expect a literal answer, so they must not arm the no-response quip (the
    # live "No answer. Bold strategy." misfire fired off exactly this shape).
    if re.search(
        r"\bso[, ]+(?:what\s+you'?re\s+saying|you'?re\s+(?:saying|telling)|"
        r"that\s+means|i\s+(?:guess|take\s+it))\b",
        lowered,
    ):
        return False
    if _SHORT_SLOT_QUESTION_RE.search(lowered):
        return True
    words = re.findall(r"[a-z0-9']+", lowered)
    if (
        len(words) <= 4
        and not _RESPONSE_QUESTION_START_RE.search(lowered)
        and not any(word in {"you", "your", "we", "they", "it", "this", "that"} for word in words)
    ):
        return False
    return True


def _question_recovery_cooldown_secs() -> float:
    return max(
        0.0,
        float(getattr(config, "CONVERSATION_NO_RESPONSE_QUIP_SECS", 7.0) or 0.0),
    )


def _should_no_response_recovery_fire(
    *,
    asked_at: float,
    now: Optional[float] = None,
    last_speech_at: Optional[float] = None,
) -> bool:
    if _game_suppresses_conversation():
        return False
    cooldown = _question_recovery_cooldown_secs()
    if cooldown <= 0:
        return False
    current = time.monotonic() if now is None else now
    latest_speech = _last_speech_at if last_speech_at is None else last_speech_at
    return current - asked_at >= cooldown and latest_speech <= asked_at


def _arm_no_response_recovery(
    question_text: str,
    person_id: Optional[int],
) -> None:
    """After Rex asks a question, recover with one quip if nobody answers."""
    global _no_response_recovery_token

    if _game_suppresses_conversation():
        return
    if not _question_expects_response(question_text):
        return
    cooldown = _question_recovery_cooldown_secs()
    if cooldown <= 0:
        return

    asked_at = time.monotonic()
    # Hold the floor: Rex asked a real question, so give the user an actual window
    # to answer before idle banter jumps in with a new prompt.
    global _floor_held_until
    _floor_held_until = asked_at + float(
        getattr(config, "POST_QUESTION_FLOOR_HOLD_SECS", 10.0) or 0.0
    )
    with _no_response_recovery_lock:
        _no_response_recovery_token += 1
        token = _no_response_recovery_token

    def _timer() -> None:
        _stop_event.wait(cooldown)
        if _stop_event.is_set():
            return
        with _no_response_recovery_lock:
            if token != _no_response_recovery_token:
                return
        if not _should_no_response_recovery_fire(asked_at=asked_at):
            return
        if state_module.get_state() != State.ACTIVE:
            return
        if _game_suppresses_conversation():
            return
        if (
            speech_queue.is_speaking()
            or output_gate.is_busy()
            or echo_cancel.is_suppressed()
            or _interrupted.is_set()
        ):
            return
        # Defer to a more useful proactive beat (e.g. an idle-banter question that
        # asks about the user) — don't stack the snarky quip on top of it so the
        # user gets one re-engagement and then real silence to answer into.
        if _proactive_line_recently_fired():
            return

        quips = getattr(config, "CONVERSATION_NO_RESPONSE_QUIPS", None) or [
            "Guess that question landed in the cargo bay."
        ]
        quip = random.choice(list(quips))
        _log.info("[interaction] no-response recovery quip after question=%r", question_text)
        completed = _speak_proactive(
            quip, emotion="neutral", priority=1, label="no_response_recovery"
        )
        if completed:
            # Only treat the question as unanswered once the quip actually plays.
            # If it was aborted because the user just started answering, leave the
            # pending question open so their reply still matches it.
            try:
                if person_id is not None:
                    rel_memory.decline_latest_pending_question(
                        person_id,
                        reason="no response after Rex question",
                    )
            except Exception as exc:
                _log.debug("no-response pending question close failed: %s", exc)
            conv_memory.add_to_transcript("Rex", quip)
            conv_log.log_rex(quip)
            _register_rex_utterance(quip)

    threading.Thread(
        target=_timer,
        daemon=True,
        name="question-no-response-recovery",
    ).start()


def _register_rex_utterance(
    text: str,
    wait_secs: Optional[float] = None,
    *,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    target_person_id: Optional[int] = None,
    target_name: Optional[str] = None,
    expected_reply_types: Optional[list[str]] = None,
    blocked_actions: Optional[list[str]] = None,
) -> None:
    if not text or not text.strip():
        return
    try:
        repair_moves.note_assistant_turn(text)
    except Exception:
        pass
    try:
        comedy_modes.note_spoken_line(text)
    except Exception:
        pass
    try:
        end_thread.note_assistant_turn(text)
    except Exception:
        pass
    try:
        topic_thread.note_assistant_turn(text)
    except Exception:
        pass
    try:
        consciousness.note_rex_utterance(
            text,
            wait_secs=wait_secs,
            source=source,
            topic=topic,
            target_person_id=target_person_id,
            target_name=target_name,
            expected_reply_types=expected_reply_types,
            blocked_actions=blocked_actions,
        )
    except Exception:
        pass


def _primary_session_person_id() -> Optional[int]:
    if len(_session_person_ids) == 1:
        return next(iter(_session_person_ids))
    try:
        people = world_state.get("people", []) or []
    except Exception:
        people = []
    known_visible = [
        p.get("person_db_id")
        for p in people
        if p.get("person_db_id") is not None
    ]
    unique = {int(pid) for pid in known_visible if pid is not None}
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _profile_fact_count(person_id: int) -> int:
    return profile_questions.profile_fact_count(person_id)


def _next_profile_question(person_id: int) -> Optional[dict]:
    return profile_questions.next_profile_question(person_id)


def _format_low_memory_question(person_id: int, question_text: str) -> str:
    text = (question_text or "").strip()
    if not text:
        return ""
    try:
        person = people_memory.get_person(person_id) or {}
    except Exception:
        person = {}
    full_name = str(person.get("name") or "").strip()
    first_name = _first_name_or(full_name, "there")
    template = str(
        getattr(
            config,
            "LOW_MEMORY_IDLE_QUESTION_PREFIX",
            "I don't know you well yet, {name}, {question}",
        )
        or "{question}"
    )
    try:
        formatted = template.format(name=first_name, question=text)
    except Exception:
        formatted = f"I don't know you well yet, {first_name}, {text}"
    return llm.clean_response_text(formatted)


def _maybe_interest_idle_followup(
    *,
    idle_for: float,
    effective_idle_timeout: float,
) -> bool:
    """Give one topic-aware nudge before an active interest thread goes idle."""
    global _session_exchange_count
    if _game_suppresses_conversation():
        return False
    if _directed_context_fresh():
        return False
    if not bool(getattr(config, "INTEREST_IDLE_FOLLOWUP_ENABLED", True)):
        return False
    threshold = float(getattr(config, "INTEREST_IDLE_FOLLOWUP_SECS", 12.0) or 0.0)
    if threshold <= 0 or idle_for < threshold:
        return False
    # Leave at least a little room between the nudge and the hard idle cutoff.
    if idle_for >= max(0.0, effective_idle_timeout - 1.0):
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False
    if _interrupted.is_set():
        return False
    try:
        if end_thread.is_grace_active():
            return False
    except Exception:
        pass

    person_id = _primary_session_person_id()
    if person_id is None:
        return False
    try:
        steering = conversation_steering.build_context(person_id)
    except Exception as exc:
        _log.debug("interest idle follow-up steering context failed: %s", exc)
        return False
    if not steering:
        return False

    key = (person_id, steering.fact_key)
    if key in _interest_idle_followups_spoken:
        return False

    allow_question = True
    try:
        allow_question = question_budget.can_ask("interest_idle_followup")
    except Exception:
        allow_question = True

    max_words = int(getattr(config, "INTEREST_IDLE_FOLLOWUP_MAX_WORDS", 22) or 22)
    question_rule = (
        "Ask one simple, low-pressure question that deepens this topic."
        if allow_question else
        "Do not ask a question; offer one specific opinion or curious observation."
    )
    try:
        line = llm.get_response(
            "Generate ONE short DJ-R3X line to re-engage a person after a quiet "
            f"pause. Active interest: {steering.topic!r}. {question_rule} "
            "Sound genuinely interested in what they told Rex they like. "
            "Use one sentence only. Do not add a second sentence, self-reference, "
            "flight jokes, roasts, or 'I bet...' riffs. "
            "Do not mention the silence, the camera, the room, or the idle timer. "
            f"Maximum {max_words} words. Return only the line.",
            person_id,
        )
    except Exception as exc:
        _log.debug("interest idle follow-up LLM failed: %s", exc)
        return False

    line = llm.clean_response_text(line or "")
    if not line:
        return False
    frame = social_frame.build_frame(
        f"I like {steering.topic}",
        person_id,
        agenda_directive=(
            steering.directive
            + " Primary purpose: after a quiet pause, keep the active interest "
            "thread alive with one compact, topic-aware follow-up."
        ),
    )
    if not allow_question:
        frame.allow_question = False
    frame.max_words = min(frame.max_words, max_words)
    frame.max_sentences = 2 if frame.allow_question else 1
    governed = social_frame.govern_response(line, frame)
    line = governed.text
    if not line:
        return False

    _log.info(
        "[interaction] interest idle follow-up — person_id=%s topic=%r text=%r",
        person_id,
        steering.topic,
        line,
    )
    completed = _speak_proactive(
        line, emotion="curious", priority=1, label="interest_idle_followup"
    )
    if completed:
        _interest_idle_followups_spoken.add(key)
        conv_memory.add_to_transcript("Rex", line)
        conv_log.log_rex(line)
        _register_rex_utterance(line)
        _session_exchange_count += 1
        if "?" in line:
            try:
                rel_memory.save_question_asked(
                    person_id,
                    f"{steering.fact_key}_idle_followup",
                    line,
                    1,
                )
            except Exception as exc:
                _log.debug("interest idle follow-up save_qa failed: %s", exc)
    return completed


def _maybe_low_memory_idle_question(
    *,
    idle_for: float,
    effective_idle_timeout: float,
) -> bool:
    """Ask one profile-building question during a lull for known sparse profiles."""
    if _game_suppresses_conversation():
        return False
    if _directed_context_fresh():
        return False
    if not bool(getattr(config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True)):
        return False
    threshold = float(getattr(config, "LOW_MEMORY_IDLE_QUESTION_SECS", 10.0) or 0.0)
    if threshold <= 0 or idle_for < threshold:
        return False
    if idle_for >= max(0.0, effective_idle_timeout - 1.0):
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False
    if _interrupted.is_set():
        return False
    try:
        if end_thread.is_grace_active():
            return False
    except Exception:
        pass

    person_id = _primary_session_person_id()
    if person_id is None or person_id in _low_memory_idle_questions_spoken:
        return False
    max_facts = int(getattr(config, "LOW_MEMORY_PROFILE_MAX_FACTS", 4) or 4)
    if _profile_fact_count(person_id) > max_facts:
        return False
    try:
        if not question_budget.can_ask("low_memory_idle_question"):
            return False
    except Exception:
        pass

    question = _next_profile_question(person_id)
    if not question:
        return False
    question_text = str(question.get("text") or "").strip()
    if not question_text:
        return False
    spoken_text = _format_low_memory_question(person_id, question_text)
    if not spoken_text:
        return False

    _log.info(
        "[interaction] low-memory idle profile question — person_id=%s fact_count=%d key=%r text=%r",
        person_id,
        _profile_fact_count(person_id),
        question.get("key"),
        spoken_text,
    )
    completed = _speak_proactive(
        spoken_text, emotion="curious", priority=1, label="low_memory_idle_question"
    )
    if completed:
        _low_memory_idle_questions_spoken.add(person_id)
        conv_memory.add_to_transcript("Rex", spoken_text)
        conv_log.log_rex(spoken_text)
        _register_rex_utterance(spoken_text)
        try:
            rel_memory.save_question_asked(
                person_id,
                question["key"],
                spoken_text,
                question.get("depth", 1),
            )
        except Exception as exc:
            _log.debug("low-memory idle question save_qa failed: %s", exc)
    return completed


_IDLE_BANTER_DIRECTIVES = (
    # Even attempts: turn the spotlight on the user.
    "The conversation just went quiet — the user didn't say goodbye, they simply "
    "stopped talking. DRIVE it forward, in character: ask ONE short, specific "
    "question about the user that you're genuinely curious about given what you "
    "know of them (their day, their work, their dog, what they're into), or riff "
    "a question off something you can see. Light and inviting, one or two "
    "sentences. Do not sign off and do not announce the silence.",
    # Odd attempts: Rex volunteers something of his own.
    "Still quiet. Keep the room alive by VOLUNTEERING something of your own, in "
    "character — a strong Rex opinion, a music take, a preference, a memory, or a "
    "dry observation about the room. Give them something worth reacting to. Do "
    "NOT ask a question this time, and do not sign off or announce the silence.",
)


def _governor_enforcing() -> bool:
    """True when the action governor is the single decider for proactive speech
    (ACTION_GOVERNOR_ENFORCE). Interaction-thread proactive paths then SUBMIT a
    candidate (governor.submit_external) instead of speaking inline."""
    try:
        from intelligence.action_governor import governor
        return bool(governor.enforcing)
    except Exception:
        return False


def _maybe_idle_banter(
    *,
    idle_for: float,
    effective_idle_timeout: float,
) -> bool:
    """General proactive filler: when the user simply goes quiet (no goodbye),
    Rex re-engages instead of waiting out the timeout and signing off. Alternates
    between asking about the user and volunteering his own opinion/preference, up
    to IDLE_BANTER_MAX_PER_STRETCH times per silent stretch. Unlike the interest /
    low-memory idle paths, this also fires for well-known, fully-profiled people."""
    global _last_idle_banter_at, _idle_banter_count, _session_exchange_count
    if not bool(getattr(config, "IDLE_BANTER_ENABLED", True)):
        return False
    if _game_suppresses_conversation():
        return False
    if _directed_context_fresh():
        return False
    threshold = max(2.0, float(getattr(config, "IDLE_BANTER_SECS", 8.0) or 8.0))
    if idle_for < threshold:
        return False
    # Rex just asked a real question — hold the floor so the user can actually
    # answer instead of getting a fresh prompt piled on top.
    if time.monotonic() < _floor_held_until:
        return False
    # Leave room before the hard idle cutoff so the outro can still land.
    if idle_for >= max(0.0, effective_idle_timeout - 1.0):
        return False
    if _idle_banter_count >= int(getattr(config, "IDLE_BANTER_MAX_PER_STRETCH", 2) or 0):
        return False
    now = time.monotonic()
    if now - _last_idle_banter_at < float(getattr(config, "IDLE_BANTER_COOLDOWN_SECS", 12.0) or 0.0):
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False
    if _interrupted.is_set():
        return False
    try:
        if end_thread.is_grace_active():
            return False
    except Exception:
        pass

    person_id = _primary_session_person_id()
    if person_id is None:
        return False

    # Lead with SUBSTANCE, not an interview: the first re-engagement after a pause
    # VOLUNTEERS Rex's own current preoccupation (rex_pov); only a later banter in
    # the same silent stretch turns the spotlight back on the user. This is the
    # design intent ("volunteer, don't just interview"), and because a silent
    # stretch usually gets only ONE banter before the user replies, the old
    # ask-first default buried the POV every turn (it never surfaced live). The
    # reply path already carries curiosity about the user; idle banter is the slot
    # where Rex brings his own thing.
    attempt = _idle_banter_count % len(_IDLE_BANTER_DIRECTIVES)
    ask_user = attempt != 0
    directive = _IDLE_BANTER_DIRECTIVES[0 if ask_user else 1]
    if not ask_user:
        # Volunteer attempt: lead with Rex's SPECIFIC current preoccupation (rex_pov)
        # rather than a generic improvised opinion, so idle volunteering matches what
        # he's been bringing up in replies. Falls back to the generic volunteer
        # directive when POV is disabled/empty.
        try:
            pov_text = rex_pov.active_pov_text()
        except Exception:
            pov_text = ""
        if pov_text:
            directive = (
                "Still quiet. Keep the room alive by VOLUNTEERING what's on your mind "
                "right now — " + pov_text + " Say it like a passing thought you want "
                "them to react to. Do NOT ask a question this time, and do not sign off "
                "or announce the silence."
            )
    # The generate + govern + speak + on-spoken bookkeeping, deferred so the action
    # governor can run it ONLY if idle banter wins the tick (enforce mode). Captures
    # `directive`/`ask_user`/`person_id` decided above.
    def _deliver() -> bool:
        global _session_exchange_count
        try:
            line = llm.get_response(
                "You are re-engaging after a quiet pause in an ongoing conversation. "
                + directive
                + " Keep it to one or two short sentences. Return only the line.",
                person_id,
            )
        except Exception as exc:
            _log.debug("idle banter LLM failed: %s", exc)
            return False

        line = llm.clean_response_text(line or "")
        if not line:
            return False
        try:
            frame = social_frame.build_frame(
                "(the user has gone quiet)",
                person_id,
                agenda_directive=(
                    "Primary purpose: after a quiet pause, proactively keep the "
                    "conversation alive. "
                    + ("Ask one short, genuine question about the user."
                       if ask_user else
                       "Volunteer one specific Rex opinion, preference, or observation; "
                       "do not ask a question.")
                ),
            )
            # Asking IS the point of an "ask the user" nudge — force the question
            # through (otherwise the quiet-energy frame strips it and Rex dead-acks).
            frame.allow_question = bool(ask_user)
            if ask_user:
                frame.max_sentences = max(frame.max_sentences, 2)
                frame.max_words = max(frame.max_words, 24)
            governed = social_frame.govern_response(line, frame)
            if governed.text:
                line = governed.text
        except Exception as exc:
            _log.debug("idle banter govern failed: %s", exc)

        completed = _speak_proactive(
            line, emotion="curious", priority=1, label="idle_banter"
        )
        if completed:
            _log.info(
                "[interaction] idle banter — person_id=%s ask_user=%s text=%r",
                person_id, ask_user, line,
            )
            conv_memory.add_to_transcript("Rex", line)
            conv_log.log_rex(line)
            _register_rex_utterance(line)
            _session_exchange_count += 1
        return completed

    if _governor_enforcing():
        # ENFORCE: the governor is the single decider. Arm the idle-banter cooldown
        # NOW so the idle loop doesn't re-submit every tick, then submit a candidate
        # carrying the deferred speak work — only the tick's winner runs; if a
        # higher-priority proactive wins, idle banter quietly yields.
        _last_idle_banter_at = time.monotonic()
        _idle_banter_count += 1
        try:
            from intelligence.action_governor import CandidateMove, governor
            governor.submit_external(CandidateMove(
                source="interaction._maybe_idle_banter",
                purpose="idle_monologue",
                priority=int(getattr(config, "IDLE_BANTER_GOVERNOR_PRIORITY", 50)),
                label="idle_banter",
                speak_fn=_deliver,
                metadata={"topic_key": f"idle_banter:{person_id}"},
            ))
        except Exception as exc:
            _log.debug("idle banter governor submit failed: %s", exc)
        return True

    # LEGACY: speak inline; arm the cooldown only on an actual spoken line.
    completed = _deliver()
    if completed:
        _last_idle_banter_at = time.monotonic()
        _idle_banter_count += 1
    return completed


def _maybe_idle_outro() -> bool:
    """Say one tiny silence-aware line before an active session returns to IDLE."""
    global _idle_outro_spoken
    if _game_suppresses_conversation():
        return False
    if _idle_outro_spoken:
        return False
    if not bool(getattr(config, "IDLE_OUTRO_ENABLED", True)):
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False
    if _interrupted.is_set():
        return False
    pool = list(getattr(config, "IDLE_OUTRO_LINES", []) or [])
    if not pool:
        return False
    line = random.choice(pool)
    line = llm.clean_response_text(line or "")
    if not line:
        return False

    _log.info("[interaction] idle outro before IDLE: %r", line)
    completed = _speak_proactive(line, emotion="neutral", priority=1, label="idle_outro")
    if completed:
        _idle_outro_spoken = True
        conv_memory.add_to_transcript("Rex", line)
        conv_log.log_rex(line)
        _register_rex_utterance(line)
    return completed


def _record_being_discussed(
    *,
    text: str,
    label: str,
    sentiment: str,
    speaker_id: Optional[int],
    speaker_name: Optional[str],
) -> None:
    """Update world_state.social.being_discussed with this overheard mention.

    Increments mentions_in_window when the previous mention was within
    BEING_DISCUSSED_ACTIVE_WINDOW_SECS, otherwise resets the count.
    """
    try:
        social = world_state.get("social") or {}
        bd = social.get("being_discussed") or {}
        now_epoch = time.time()
        last_at = bd.get("last_mention_at")
        window = float(getattr(config, "BEING_DISCUSSED_ACTIVE_WINDOW_SECS", 30.0))
        rolling_window = float(
            getattr(config, "BEING_DISCUSSED_ROLLING_WINDOW_SECS", 60.0)
        )
        if last_at is not None and (now_epoch - float(last_at)) <= rolling_window:
            count = int(bd.get("mentions_in_window") or 0) + 1
        else:
            count = 1

        new_bd = {
            "last_mention_at": now_epoch,
            "last_snippet": text,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "addressee_id": None,
            "label": label,
            "sentiment": sentiment,
            "mentions_in_window": count,
            # Reset chimed_in flag; it's per-active-window so a fresh discussion
            # earns its own chance for a chime-in.
            "chimed_in": False if (now_epoch - float(last_at or 0)) > window else bd.get("chimed_in", False),
        }
        social["being_discussed"] = new_bd
        world_state.update("social", social)
        _log.info(
            "[interaction] being_discussed updated — label=%s sentiment=%s mentions=%d snippet=%r",
            label, sentiment, count, text[:120],
        )
        # Log to conversation transcript so future turns have context.
        try:
            conv_memory.add_to_transcript(
                speaker_name or "Someone",
                f"(overheard, {label}) {text}",
            )
        except Exception:
            pass
    except Exception as exc:
        _log.debug("_record_being_discussed failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Identity enrollment helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalized_name_tokens(candidate: str) -> list[str]:
    return normalized_name_key(candidate).split()


def _is_non_name_candidate(candidate: str) -> bool:
    return normalize_person_name(candidate, allow_single=True) is None


def _normalize_name(candidate: str) -> Optional[str]:
    """
    Normalize a spoken/self-reported name candidate.
    Returns None when the candidate does not look like a usable name.
    """
    return normalize_person_name(candidate, allow_single=True)


def _first_name_or(value: Optional[str], fallback: str = "") -> str:
    parts = str(value or "").strip().split()
    return parts[0] if parts else fallback


def _identity_enrollment_ack(name: str) -> str:
    special = person_specials.special_intro_ack(name)
    if special:
        return special
    return f"Got it, {name}. Nice to meet you."


def _special_intro_prompt(name: str) -> Optional[str]:
    special_context = person_specials.special_prompt_context(name)
    if not special_context:
        return None
    if person_specials.is_jt_volleyball_celebrity(name):
        special_instruction = (
            "acknowledge them by name as a major volleyball celebrity. Include "
            "one affectionate joke about getting old as a volleyball athlete, "
            "with bones or muscles as the absurd target."
        )
    elif person_specials.is_rex_creator(name):
        special_instruction = (
            "acknowledge Bret Benziger by name as Rex's creator and maker. Be "
            "warm, reverent, loyal, and still dryly Rex about being Bret's "
            "high-maintenance droid creation."
        )
    elif person_specials.is_galactic_hair_stylist(name):
        special_instruction = (
            "acknowledge them by name as one of the greatest hair stylists in "
            "the galactic quadrant. Include one affectionate joke about elite "
            "styling skills, legendary blowouts, emergency bang repair, or "
            "frizz surrendering."
        )
    else:
        special_instruction = "acknowledge them by name with the special hook."
    return (
        f"{special_context} You just learned this person's name is {name}. "
        f"In ONE short in-character Rex line, {special_instruction} "
        "No follow-up question."
    )


def _same_person_name(left: Optional[str], right: Optional[str]) -> bool:
    left_norm = _normalize_name(left or "")
    right_norm = _normalize_name(right or "")
    if not left_norm or not right_norm:
        return False
    return left_norm.lower() == right_norm.lower()


def _name_supported_by_user_text(name: Optional[str], text: str) -> bool:
    normalized_name = _normalize_name(name or "")
    if not normalized_name:
        return False
    text_key = normalized_name_key(text or "")
    name_key = normalized_name_key(normalized_name)
    if not text_key or not name_key:
        return False
    if re.search(rf"(?<!\w){re.escape(name_key)}(?!\w)", text_key):
        return True
    if looks_like_initials(normalized_name):
        return name_key.replace(" ", "") in text_key.replace(" ", "")
    return False


def _relationship_supported_by_user_text(
    relationship: Optional[str],
    text: str,
) -> bool:
    rel = _normalize_relationship_word(relationship)
    if not rel:
        return False
    text_plain = _plain_confirmation_text(text)
    if not text_plain:
        return False
    aliases = {
        rel,
        rel.replace("_", " "),
        rel.replace("_", "-"),
        rel.replace("_", ""),
    }
    for raw, normalized in _RELATIONSHIP_WORD_NORMALIZE.items():
        if normalized == rel:
            aliases.add(raw)
    for alias in aliases:
        alias_plain = _plain_confirmation_text(alias)
        if alias_plain and re.search(
            rf"(?<!\w){re.escape(alias_plain)}(?!\w)",
            text_plain,
        ):
            return True
    return False


def _filter_relationship_introduction_evidence(
    parsed: dict,
    user_text: str,
    speaker_name: Optional[str],
    *,
    source: str,
) -> tuple[Optional[str], Optional[str]]:
    name = parsed.get("name")
    relationship = parsed.get("relationship")
    name = name.strip() if isinstance(name, str) else None
    relationship = relationship.strip().lower() if isinstance(relationship, str) else None

    if name and not _name_supported_by_user_text(name, user_text):
        _log.warning(
            "[identity] rejected extracted newcomer name without transcript evidence "
            "source=%s name=%r text=%r parsed=%r",
            source,
            name,
            user_text,
            parsed,
        )
        name = None
    if name and speaker_name and (
        _same_person_name(name, speaker_name)
        or names_are_similar(name, speaker_name)
    ):
        _log.warning(
            "[identity] rejected extracted newcomer name matching speaker "
            "source=%s name=%r speaker=%r text=%r",
            source,
            name,
            speaker_name,
            user_text,
        )
        name = None
    if relationship and not _relationship_supported_by_user_text(
        relationship,
        user_text,
    ):
        _log.warning(
            "[identity] rejected extracted relationship without transcript evidence "
            "source=%s relationship=%r text=%r parsed=%r",
            source,
            relationship,
            user_text,
            parsed,
        )
        relationship = None
    return name, relationship


def _extract_self_identified_name(text: str) -> Optional[str]:
    normalized = (text or "").strip()
    if not normalized:
        return None
    for pattern in _SELF_NAME_PATTERNS:
        match = pattern.search(normalized)
        if match:
            candidate = (match.group(1) or "").strip()
            if _SELF_INTRO_NON_NAME_RE.search(candidate):
                return None
            return _normalize_name(candidate)
    return None


_IDENTITY_PROMPT_ECHO_RE = re.compile(
    r"\bwhat\s+name\s+should\s+i\s+(?:save|say|use|call|file)\s+for\s+you\b",
    re.IGNORECASE,
)


def _prompted_bare_name_text(text: str) -> str:
    """
    Clean a bare reply to Rex's identity prompt.

    Whisper can capture the tail of Rex's own question before the user's answer.
    It also sometimes inserts commas inside a two-token name. In the prompted
    identity path, preserving both name-like chunks is safer than truncating at
    the first punctuation mark.
    """
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""
    if _IDENTITY_PROMPT_ECHO_RE.search(cleaned) and "?" in cleaned:
        tail = cleaned.rsplit("?", 1)[-1].strip()
        if tail:
            cleaned = tail
    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if 1 < len(parts) <= 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", p) for p in parts):
            cleaned = " ".join(parts)
    return cleaned


_RELATIONSHIP_WORD_NORMALIZE = {
    "best friend": "best_friend",
    "co-worker": "coworker",
    "co worker": "coworker",
    "colleague": "coworker",
    "dad": "father",
    "mom": "mother",
    "brother": "sibling",
    "sister": "sibling",
}


def _normalize_relationship_word(value: Optional[str]) -> Optional[str]:
    rel = re.sub(r"\s+", " ", (value or "").strip().lower().replace("-", " "))
    if not rel:
        return None
    return _RELATIONSHIP_WORD_NORMALIZE.get(rel, rel).replace(" ", "_")


def _extract_self_relationship_to_engaged(
    text: str,
    engaged_name: Optional[str],
) -> Optional[str]:
    """
    Parse a self-intro aside like "my name is Jennifer, this is my brother Bret".

    The relationship is from the speaker toward the already-engaged person.
    Gendered sibling labels collapse to "sibling" so Jennifer is not stored as
    Bret's "brother" when she says Bret is her brother.
    """
    if not text or not engaged_name:
        return None
    engaged_first = _first_name_or(engaged_name).lower()
    if not engaged_first:
        return None
    engaged_full = str(engaged_name or "").strip().lower()
    rel_words = (
        "best friend|friend|father|dad|mother|mom|parent|coworker|co-worker|"
        "colleague|boss|supervisor|manager|aunt|uncle|partner|girlfriend|"
        "boyfriend|fiancee|fiance|wife|husband|spouse|sister|brother|sibling|"
        "cousin|roommate|neighbor|neighbour|son|daughter|child"
    )
    patterns = [
        rf"\b(?:this is|that'?s|that is)\s+my\s+(?P<rel>{rel_words})\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
        rf"\b(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\s+is\s+my\s+(?P<rel>{rel_words})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        candidate = _normalize_name(match.group("name") or "")
        candidate_lower = (candidate or "").lower()
        if not candidate_lower:
            continue
        if candidate_lower == engaged_full or _first_name_or(candidate_lower) == engaged_first:
            return _normalize_relationship_word(match.group("rel"))
    return None


def _extract_introduced_name(text: str, allow_bare_name: bool = False) -> Optional[str]:
    """Extract a self-introduced name from speech text."""
    normalized = text.strip()
    if not normalized:
        return None

    for pattern in _NAME_PATTERNS:
        m = pattern.search(normalized)
        if not m:
            continue
        return _normalize_name(m.group(1))

    # After Rex explicitly asks "who are you?", many people reply with only a name.
    if allow_bare_name:
        return _normalize_name(_prompted_bare_name_text(normalized))

    return None


def _prompted_name_reply_needs_confirmation(raw_text: str, name: str) -> bool:
    """Only suspicious prompted-name replies get a yes/no save confirmation."""
    if not name:
        return False
    text = (raw_text or "").strip()
    if not text:
        return False
    if "," in text:
        return True
    if "?" in text and _IDENTITY_PROMPT_ECHO_RE.search(text):
        return True
    return False


def _name_word_count(name: str) -> int:
    return len([part for part in (name or "").split() if part.strip()])


def _is_common_first_name_only(name: str) -> bool:
    if not bool(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_DISAMBIGUATION_ENABLED", True)):
        return False
    normalized = _normalize_name(name or "")
    if not normalized or _name_word_count(normalized) != 1:
        return False
    common = {
        str(item).strip().lower()
        for item in getattr(config, "COMMON_FIRST_NAMES_REQUIRE_LAST_NAME", [])
        if str(item).strip()
    }
    return normalized.lower() in common


def _format_common_first_name_last_name_prompt(first_name: str) -> str:
    prompts = list(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_PROMPTS", []) or [])
    if not prompts:
        prompts = [
            "{first}, how original. Last name too, please, so the memory banks don't get confused."
        ]
    try:
        template = random.choice(prompts)
        return template.format(first=first_name)
    except Exception:
        return f"{first_name}, how original. Last name too, please."


def _extract_last_name_reply(text: str, first_name: str) -> Optional[str]:
    normalized = (text or "").strip()
    if not normalized:
        return None

    patterns = [
        re.compile(r"\bmy\s+last\s+name\s+is\s+(.+)$", re.IGNORECASE),
        re.compile(r"\blast\s+name(?:'s| is)?\s+(.+)$", re.IGNORECASE),
        re.compile(r"\bsurname(?:'s| is)?\s+(.+)$", re.IGNORECASE),
        re.compile(r"\bit'?s\s+(.+)$", re.IGNORECASE),
    ]
    candidate = normalized
    for pattern in patterns:
        match = pattern.search(normalized)
        if match:
            candidate = match.group(1)
            break

    parsed = _normalize_name(candidate)
    if not parsed:
        return None

    parts = parsed.split()
    if len(parts) >= 2 and parts[0].lower() == first_name.lower():
        return " ".join(parts[1:])
    if len(parts) == 1 and parts[0].lower() != first_name.lower():
        return parts[0]
    return None


_LAST_NAME_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:would\s+)?rather\s+not\s+(?:say|tell|share)|"
    r"(?:i\s+)?(?:do\s+not|don't|won't|will\s+not|am\s+not|ain't)\s+"
    r"(?:want\s+to\s+)?(?:say|tell|share|give)\s+(?:you\s+)?(?:my\s+)?(?:last\s+name|surname)|"
    r"(?:i'?m\s+)?not\s+(?:telling|sharing|giving)\s+(?:you\s+)?(?:my\s+)?(?:last\s+name|surname)|"
    r"you\s+(?:do\s+not|don't)\s+need\s+(?:to\s+know\s+)?(?:my\s+)?(?:last\s+name|surname)|"
    r"(?:my\s+)?(?:last\s+name|surname)\s+is\s+(?:private|personal|classified|none\s+of\s+your\s+business)|"
    r"(?:none\s+of\s+your\s+business|that'?s\s+private|that\s+is\s+private|classified)|"
    r"(?:first\s+name\s+only|no\s+last\s+name|skip\s+(?:it|the\s+last\s+name))|"
    r"(?:just|only)\s+{first}\b|"
    r"(?:call\s+me|you\s+can\s+call\s+me)\s+{first}\b"
    r")\b",
    re.IGNORECASE,
)
_LAST_NAME_IMPLICIT_REFUSAL_RE = re.compile(
    r"^\s*(?:"
    r"(?:no+|nope|nah|naw)(?:\s*[,!.]?\s*(?:thanks|thank\s+you)?)?$|"
    r"(?:(?:no+|nope|nah|naw)\s*[,!.]?\s*)?"
    r"(?:(?:i\s+)?(?:am\s+not|ain['’]?t)|i['’]?m\s+not)\s+"
    r"(?:going|gonna)\s+to\b|"
    r"(?:(?:no+|nope|nah|naw)\s*[,!.]?\s*)?"
    r"(?:i\s+)?(?:won['’]?t|will\s+not|do\s+not|don['’]?t)\s+"
    r"(?:(?:want|wanna)\s+to\s+)?(?:say|tell|share|give|do\s+that)\b"
    r")",
    re.IGNORECASE,
)


def _is_last_name_refusal(text: str, first_name: str) -> bool:
    cleaned = (text or "").strip()
    first = re.escape((first_name or "").strip())
    if not cleaned or not first:
        return False
    pattern = _LAST_NAME_REFUSAL_RE.pattern.format(first=first)
    return bool(
        re.search(pattern, cleaned, re.IGNORECASE)
        or _LAST_NAME_IMPLICIT_REFUSAL_RE.search(cleaned)
    )


def _common_first_name_context_fresh(ctx: Optional[dict]) -> bool:
    if not ctx:
        return False
    ttl = float(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS", 30.0))
    return (time.monotonic() - float(ctx.get("asked_at", 0.0))) <= max(1.0, ttl)


def _pending_existing_common_first_name_reply_target(
    text: str,
) -> Optional[tuple[int, str]]:
    """Return the pending known person when a short reply answers their prompt."""
    ctx = _pending_existing_common_first_name
    if not _common_first_name_context_fresh(ctx):
        return None

    first_name = str((ctx or {}).get("first_name") or "").strip()
    person_id = _safe_int((ctx or {}).get("person_id"))
    if person_id is None or not first_name:
        return None

    if _is_last_name_refusal(text, first_name):
        return person_id, first_name
    if _extract_last_name_reply(text, first_name):
        return person_id, first_name
    return None


_LAST_NAME_DECLINED_FACT_KEY = "last_name_declined"


def _remember_last_name_declined(person_id: Optional[int], first_name: str) -> None:
    if person_id is None:
        return
    try:
        facts_memory.add_fact(
            int(person_id),
            "identity",
            _LAST_NAME_DECLINED_FACT_KEY,
            f"{first_name} declined to share a last name",
            "identity_boundary",
            confidence=1.0,
        )
    except Exception as exc:
        _log.debug("last-name decline fact save failed: %s", exc)


def _has_declined_last_name(person_id: Optional[int]) -> bool:
    if person_id is None:
        return False
    try:
        facts = facts_memory.get_facts_by_category(int(person_id), "identity")
    except Exception as exc:
        _log.debug("last-name decline fact check failed: %s", exc)
        return False
    for fact in facts:
        if fact.get("key") != _LAST_NAME_DECLINED_FACT_KEY:
            continue
        value = str(fact.get("value") or "").strip().lower()
        if value:
            return True
    return False


def _known_person_needs_last_name(person_id: Optional[int], person_name: Optional[str]) -> bool:
    if person_id is None:
        return False
    if int(person_id) in _common_first_name_prompted_this_session:
        return False
    normalized = _normalize_name(person_name or "")
    if not normalized or _name_word_count(normalized) != 1:
        return False
    if looks_like_initials(normalized):
        return False
    return not _has_declined_last_name(int(person_id))


def _extract_name_update(text: str) -> Optional[str]:
    """Extract an explicit request/correction to rename the current person."""
    normalized = (text or "").strip()
    if not normalized or not _NAME_CORRECTION_RE.search(normalized):
        return None

    # When a preferred short name is supplied, use it over the formal name.
    call_match = _CALL_ME_NAME_RE.search(normalized)
    if call_match:
        name = _normalize_name(call_match.group(1))
        if name:
            return name

    return _extract_introduced_name(normalized, allow_bare_name=False)


def _single_visible_person_identity() -> tuple[Optional[int], Optional[str]]:
    """Return the only visible known person identity, when unambiguous."""
    try:
        people = world_state.get("people") or []
    except Exception:
        return None, None

    identities: dict[int, Optional[str]] = {}
    for person in people:
        pid = person.get("person_db_id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        identities[pid_int] = (
            person.get("face_id")
            or person.get("voice_id")
            or identities.get(pid_int)
        )

    if len(identities) != 1:
        return None, None
    pid, name = next(iter(identities.items()))
    return pid, name


def _resolve_name_update_target(
    person_id: Optional[int],
    person_name: Optional[str],
) -> tuple[Optional[int], Optional[str]]:
    """Resolve which known person row an explicit name correction should edit."""
    if person_id is not None:
        return int(person_id), person_name
    return _single_visible_person_identity()


def _refresh_world_state_person_name(person_id: int, name: str) -> None:
    """Keep live face/voice labels in sync after a memory rename."""
    try:
        def _apply_rename(people):
            changed = False
            for person in people:
                try:
                    pid = int(person.get("person_db_id"))
                except (TypeError, ValueError):
                    continue
                if pid != int(person_id):
                    continue
                person["face_id"] = name
                person["voice_id"] = name
                changed = True
            return people if changed else None

        world_state.mutate("people", _apply_rename)

        crowd = world_state.get("crowd") or {}
        if crowd.get("dominant_speaker"):
            crowd["dominant_speaker"] = name
            world_state.update("crowd", crowd)
    except Exception as exc:
        _log.debug("world_state name refresh failed: %s", exc)


def _plain_confirmation_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


def _memory_wipe_confirm_ttl() -> float:
    return float(getattr(config, "MEMORY_WIPE_CONFIRM_WINDOW_SECS", 30.0))


def _clear_pending_memory_wipe() -> None:
    global _pending_memory_wipe
    _pending_memory_wipe = None


def _pending_memory_wipe_expired(now: Optional[float] = None) -> bool:
    if _pending_memory_wipe is None:
        return True
    now = time.monotonic() if now is None else now
    asked_at = float(_pending_memory_wipe.get("asked_at") or 0.0)
    if (now - asked_at) <= _memory_wipe_confirm_ttl():
        return False
    _log.info("[memory] pending wipe confirmation expired: %s", _pending_memory_wipe)
    _clear_pending_memory_wipe()
    return True


def _memory_wipe_confirmation_phrase(scope: str) -> str:
    if scope != "all":
        return "yes forget me"
    code = str(getattr(config, "FULL_MEMORY_WIPE_ACCESS_CODE", "") or "").strip()
    return f"confirm full wipe {code}" if code else "confirm full wipe"


def _memory_wipe_confirmation_hint(scope: str) -> str:
    if scope != "all":
        return '"yes forget me"'
    code = str(getattr(config, "FULL_MEMORY_WIPE_ACCESS_CODE", "") or "").strip()
    if code:
        return '"confirm full wipe" followed by the full-wipe access code'
    return '"confirm full wipe"'


def _memory_wipe_confirmation_prefix(scope: str) -> str:
    return "confirm full wipe" if scope == "all" else "yes forget me"


def _arm_memory_wipe_confirmation(
    *,
    scope: str,
    person_id: Optional[int] = None,
    person_name: Optional[str] = None,
    requester_id: Optional[int] = None,
) -> None:
    global _pending_memory_wipe
    _pending_memory_wipe = {
        "scope": scope,
        "person_id": int(person_id) if person_id is not None else None,
        "person_name": person_name,
        "requester_id": int(requester_id) if requester_id is not None else None,
        "asked_at": time.monotonic(),
    }


def _resolve_forget_me_target(
    person_id: Optional[int],
    person_name: Optional[str],
) -> tuple[Optional[int], Optional[str]]:
    if person_id is not None:
        return int(person_id), person_name
    target_id, target_name = _single_visible_person_identity()
    if target_id is not None:
        return int(target_id), target_name
    return None, None


def _scrub_world_state_after_memory_wipe(
    *,
    person_id: Optional[int] = None,
    all_people: bool = False,
) -> None:
    """Drop deleted DB identity labels from the live WorldState snapshot."""
    try:
        def _apply_clear(people):
            changed = False
            for person in people:
                if all_people:
                    should_clear = True
                else:
                    try:
                        should_clear = int(person.get("person_db_id")) == int(person_id)
                    except (TypeError, ValueError):
                        should_clear = False
                if not should_clear:
                    continue
                person["person_db_id"] = None
                person["face_id"] = None
                person["voice_id"] = None
                changed = True
            return people if changed else None

        world_state.mutate("people", _apply_clear)

        crowd = world_state.get("crowd") or {}
        if all_people or crowd.get("dominant_speaker"):
            crowd["dominant_speaker"] = None
            world_state.update("crowd", crowd)

        social = world_state.get("social") or {}
        discussed = social.get("being_discussed")
        if isinstance(discussed, dict) and (
            all_people or discussed.get("speaker_id") == person_id
        ):
            discussed["speaker_id"] = None
            discussed["speaker_name"] = None
            discussed["addressee_id"] = None
            social["being_discussed"] = discussed
            world_state.update("social", social)
    except Exception as exc:
        _log.debug("world_state memory-wipe scrub failed: %s", exc)


def _clear_deleted_person_session_state(person_id: int) -> None:
    pid = int(person_id)
    _session_person_ids.discard(pid)
    _session_person_turn_counts.pop(pid, None)
    _session_forget_terms.pop(pid, None)
    _session_router_control_topics.pop(pid, None)
    _grief_flow_state.pop(pid, None)
    _voice_refreshed_this_session.discard(pid)
    _face_reveal_declined.discard(pid)
    _common_first_name_prompted_this_session.discard(pid)


def _clear_memory_related_pending_state() -> None:
    global _awaiting_followup_event, _pending_offscreen_identify
    global _pending_face_reveal_confirm, _pending_introduction
    global _pending_intro_followup, _pending_intro_voice_capture
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation
    global _identity_prompt_until

    _awaiting_followup_event = None
    _pending_post_greet_relationship[0] = None
    _pending_offscreen_identify = None
    _pending_face_reveal_confirm = None
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
    _pending_common_first_name_identity = None
    _pending_common_first_name_introduction = None
    _pending_existing_common_first_name = None
    _pending_identity_match_confirmation = None
    _pending_prompted_name_confirmation = None
    _identity_prompt_until = 0.0


def _confirmation_speaker_mismatch(
    pending: dict,
    person_id: Optional[int],
) -> bool:
    requester_id = pending.get("requester_id")
    target_id = pending.get("person_id")
    if person_id is None:
        return False
    try:
        current = int(person_id)
    except (TypeError, ValueError):
        return True
    if requester_id is not None and current != int(requester_id):
        return True
    if pending.get("scope") == "person" and target_id is not None and current != int(target_id):
        return True
    return False


def _handle_pending_memory_wipe_confirmation(
    text: str,
    person_id: Optional[int],
) -> Optional[str]:
    """Execute or cancel a pending destructive memory wipe confirmation."""
    if _pending_memory_wipe is None or _pending_memory_wipe_expired():
        return None

    pending = dict(_pending_memory_wipe)
    plain = _plain_confirmation_text(text)
    scope = str(pending.get("scope") or "")
    expected = _plain_confirmation_text(_memory_wipe_confirmation_phrase(scope))
    cancel_phrases = {
        "no",
        "nope",
        "cancel",
        "cancel that",
        "never mind",
        "nevermind",
        "abort",
        "stop",
        "do not",
        "dont",
    }

    if plain in cancel_phrases:
        _clear_pending_memory_wipe()
        resp = "Memory wipe canceled. My memory banks remain annoyingly intact."
        _speak_blocking(resp, emotion="neutral")
        return resp

    if plain != expected:
        _clear_pending_memory_wipe()
        if scope == "all" and plain.startswith(_memory_wipe_confirmation_prefix(scope)):
            resp = "Full memory wipe rejected. Access code missing or incorrect."
            _speak_blocking(resp, emotion="neutral")
            return resp
        return None

    if _confirmation_speaker_mismatch(pending, person_id):
        _clear_pending_memory_wipe()
        resp = (
            "Confirmation rejected. That did not come from the same identity. "
            "Destructive memory wipe canceled."
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    try:
        if scope == "person":
            target_id = pending.get("person_id")
            if target_id is None:
                raise RuntimeError("missing target person_id for memory wipe")
            people_memory.delete_person(int(target_id))
            _clear_deleted_person_session_state(int(target_id))
            _clear_memory_related_pending_state()
            _clear_anonymous_speaker_slots()
            _scrub_world_state_after_memory_wipe(person_id=int(target_id))
            try:
                consciousness.clear_engagement()
            except Exception:
                pass
            conv_memory.clear_transcript()
            name = _first_name_or(pending.get("person_name"))
            address = f"{name}. " if name else ""
            resp = (
                f"{address}Confirmed. I deleted your name, face, voice print, "
                "memories, and conversation records. Clean slate. Very dramatic."
            )
            _log.info("[memory] confirmed person wipe person_id=%s", target_id)
        elif scope == "all":
            people_memory.delete_all_people()
            _session_person_ids.clear()
            _session_person_turn_counts.clear()
            _session_forget_terms.clear()
            _session_router_control_topics.clear()
            _clear_anonymous_speaker_slots()
            _grief_flow_state.clear()
            _voice_refreshed_this_session.clear()
            _face_reveal_declined.clear()
            _common_first_name_prompted_this_session.clear()
            _clear_memory_related_pending_state()
            _scrub_world_state_after_memory_wipe(all_people=True)
            try:
                consciousness.clear_engagement()
            except Exception:
                pass
            conv_memory.clear_transcript()
            resp = (
                "Confirmed. Every person record, face, voice print, relationship, "
                "and conversation memory is gone. My social life has been factory-reset."
            )
            _log.info("[memory] confirmed full people-memory wipe")
        else:
            raise RuntimeError(f"unknown memory wipe scope: {scope!r}")
    except Exception as exc:
        _log.error("[memory] memory wipe failed: %s", exc, exc_info=True)
        resp = "Memory wipe failed. Annoying, but safer than a half-erased brain."

    _clear_pending_memory_wipe()
    _speak_blocking(resp, emotion="neutral")
    return resp


def _handle_name_update_request(
    text: str,
    person_id: Optional[int],
    person_name: Optional[str],
) -> Optional[str]:
    """Apply an explicit "call me / my name is / you got my name wrong" update."""
    new_name = _extract_name_update(text)
    if not new_name:
        return None

    target_id, old_name = _resolve_name_update_target(person_id, person_name)
    old_clean = (old_name or "").strip()
    try:
        existing = people_memory.find_person_by_name(new_name)
    except Exception:
        existing = None
    if existing is not None:
        existing_id = int(existing["id"])
        if target_id is None or (
            existing_id != int(target_id)
            and _known_person_visible_recently(existing_id)
        ):
            target_id = existing_id
            old_name = existing.get("name")
            old_clean = (old_name or "").strip()
        elif target_id is not None and existing_id != int(target_id):
            response = repair_moves.add_better_luck_line(
                f"Got it. {new_name} is already a separate person in my memory, "
                f"so I won't rename {old_clean or 'this speaker'} into {new_name}."
            )
            _log.info(
                "[identity] name correction matched existing person "
                "target_id=%s existing_id=%s new=%r text=%r",
                target_id,
                existing_id,
                new_name,
                text,
            )
            _speak_blocking(response, emotion="neutral")
            return response

    if target_id is None:
        _log.info("[identity] name update had no clear target text=%r", text)
        return None

    if old_clean.lower() == new_name.lower():
        response = f"Already got you as {new_name}."
        _speak_blocking(response)
        return response

    if not people_memory.rename_person(target_id, new_name):
        response = f"I couldn't safely rename that memory to {new_name}."
        _speak_blocking(response)
        return response

    _refresh_world_state_person_name(target_id, new_name)
    _log.info(
        "[identity] renamed person_id=%s old=%r new=%r text=%r",
        target_id,
        old_name,
        new_name,
        text,
    )
    response = _identity_enrollment_ack(new_name)
    if not person_specials.special_intro_ack(new_name):
        response = repair_moves.add_better_luck_line(f"Got it. I'll call you {new_name}.")
    _speak_blocking(response, emotion="happy")
    return response


_IDLE_DIRECT_MUSIC_RE = re.compile(
    r"\b(?:play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue)\b"
    r".{0,60}\b(?:music|song|track|station|playlist|jazz|country|rock|pop|"
    r"classical|lo-fi|lofi|hip\s*hop)\b",
    re.IGNORECASE,
)
_BACKGROUND_CROSSTALK_RE = re.compile(
    r"\b(?:"
    r"came\s+out|powered\s+off|channels?|camera|the\s+other\s+one|"
    r"when\s+you(?:'re|\s+are)\s+talking\s+to\s+it|"
    r"he\s+(?:is|was|has|had|knows)|she\s+(?:is|was|has|had|knows)|"
    r"they\s+(?:are|were|have|had|know)|"
    r"it\s+(?:is|was|has|had|knows)"
    r")\b",
    re.IGNORECASE,
)


def _speech_is_directed_to_rex(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _is_bare_wake_address(cleaned):
        return True
    try:
        if address_mode.contains_rex_keyword(cleaned):
            return True
    except Exception:
        pass
    if command_parser.parse(cleaned) is not None:
        return True
    if _IDLE_DIRECT_MUSIC_RE.search(cleaned):
        return True
    if _MUSIC_PLAY_REQUEST_PAT.search(cleaned) and re.search(
        r"\b(?:music|song|track|station|playlist)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    return False


def _looks_like_background_crosstalk(text: str) -> bool:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    if _speech_is_directed_to_rex(cleaned):
        return False
    if _extract_self_identified_name(cleaned):
        return False
    if _extract_introduced_name(cleaned, allow_bare_name=False):
        return False
    if "?" in cleaned:
        # Questions can be addressed to Rex without a wake word in an active
        # exchange. Keep them in the conversation path unless they are strongly
        # third-person/meta chatter about the system.
        return bool(_BACKGROUND_CROSSTALK_RE.search(cleaned))
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) >= 5 and _BACKGROUND_CROSSTALK_RE.search(cleaned):
        return True
    if len(words) >= 10 and not re.search(r"\b(?:i|me|my|you|your|rex|r3x)\b", cleaned, re.I):
        return True
    return False


def _looks_like_direct_question_to_rex(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or "?" not in cleaned:
        return False
    if _extract_self_identified_name(cleaned):
        return False
    if _extract_introduced_name(cleaned, allow_bare_name=False):
        return False
    return True


def _turn_should_defer_identity_prompts(text: str) -> bool:
    if not bool(getattr(config, "IDENTITY_PROMPT_DEFER_ON_DIRECT_TURN", True)):
        return False
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _is_bare_wake_address(cleaned):
        return False
    if _extract_self_identified_name(cleaned):
        return False
    if _extract_introduced_name(cleaned, allow_bare_name=False):
        return False
    if command_parser.parse(cleaned) is not None:
        return True
    if _speech_is_directed_to_rex(cleaned):
        return True
    if _looks_like_direct_question_to_rex(cleaned):
        return True
    return False


def _utterance_invites_identity_question(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _is_bare_wake_address(cleaned):
        return True
    if _extract_self_identified_name(cleaned):
        return True
    if _extract_introduced_name(cleaned, allow_bare_name=False):
        return True
    if _looks_like_direct_question_to_rex(cleaned):
        return False
    words = _word_count(cleaned)
    return bool(
        words <= 5
        and re.search(
            r"\b(?:hi|hello|hey|yo|rex|r3x|dee\s*jay|dj)\b",
            cleaned,
            re.IGNORECASE,
        )
    )


def _clear_pending_identity_prompts(reason: str) -> bool:
    global _identity_prompt_until, _pending_offscreen_identify, _pending_face_reveal_confirm
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation, _pending_introduction
    global _pending_intro_followup, _pending_intro_voice_capture

    changed = any(
        item is not None
        for item in (
            _pending_offscreen_identify,
            _pending_face_reveal_confirm,
            _pending_common_first_name_identity,
            _pending_common_first_name_introduction,
            _pending_existing_common_first_name,
            _pending_identity_match_confirmation,
            _pending_prompted_name_confirmation,
            _pending_introduction,
            _pending_intro_followup,
            _pending_intro_voice_capture,
        )
    ) or _identity_prompt_until > 0.0

    _identity_prompt_until = 0.0
    _pending_offscreen_identify = None
    _pending_face_reveal_confirm = None
    _pending_common_first_name_identity = None
    _pending_common_first_name_introduction = None
    _pending_existing_common_first_name = None
    _pending_identity_match_confirmation = None
    _pending_prompted_name_confirmation = None
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
    try:
        changed = consciousness.clear_pending_identity_prompts(reason=reason) or changed
    except Exception as exc:
        _log.debug("clear pending consciousness identity prompts failed: %s", exc)
    if changed:
        _log.info("[identity] cleared pending identity prompts reason=%s", reason)
    return changed


def _should_ignore_idle_background_speech(
    *,
    from_idle_activation: bool,
    person_id: Optional[int],
    has_unknown_visible: bool,
    identity_prompt_active: bool,
    text_input: bool = False,
    text: str,
) -> bool:
    """Ignore no-wake IDLE activations that sound like off-camera background speech."""
    if not from_idle_activation:
        return False
    if text_input:
        return False
    if has_unknown_visible or identity_prompt_active:
        return False
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _speech_is_directed_to_rex(cleaned):
        return False
    if person_id is not None:
        if _known_person_visible_recently(person_id):
            return False
        if _looks_like_background_crosstalk(cleaned):
            return True
        words = re.findall(r"[A-Za-z0-9']+", cleaned)
        return len(words) >= 5
    try:
        wake_ready = bool(wake_word.is_ready())
    except Exception:
        wake_ready = True
    if not wake_ready and _PLAIN_IDLE_GREETING_PAT.match(cleaned):
        return False
    return True


def _pending_question_recent_attribution(
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    recent_engagement: Optional[dict],
    raw_best_id: Optional[int],
    speaker_score: float,
    text: str,
) -> tuple[Optional[int], Optional[str], bool]:
    """
    Attribute the next answer to the recently engaged person Rex just questioned.

    This prevents a panned-away camera or weak voice score from derailing a direct
    answer into the off-camera-unknown identity flow.
    """
    if person_id is not None:
        return person_id, person_name, False
    if not recent_engagement:
        return person_id, person_name, False
    if command_parser.parse(text) is not None:
        return person_id, person_name, False

    recent_id = _safe_int(recent_engagement.get("person_id"))
    if recent_id is None:
        return person_id, person_name, False
    pending = _latest_pending_question(recent_id)
    thread_question = ""
    try:
        thread_question = str(
            (topic_thread.snapshot() or {}).get("unresolved_question") or ""
        ).strip()
    except Exception:
        thread_question = ""
    if not pending and not thread_question:
        return person_id, person_name, False

    floor = float(getattr(config, "SPEAKER_ID_PENDING_QA_RECENT_FLOOR", 0.35))
    raw_id = _safe_int(raw_best_id)
    raw_matches_recent = raw_id == recent_id
    no_voice_candidate = raw_id is None
    # A brief false-positive unknown face should not steal the answer to a
    # question Rex just asked the engaged person. If the voice model's top
    # candidate is the asked person above the low pending-QA floor, direct
    # question continuity wins. Without matching voice evidence, unknown faces
    # still block attribution so real newcomers can answer for themselves.
    try:
        unknown_present = _has_unknown_visible_or_recent()
    except Exception:
        unknown_present = True
    if unknown_present and not (raw_matches_recent and speaker_score >= floor):
        return person_id, person_name, False

    if not no_voice_candidate and not (raw_matches_recent and speaker_score >= floor):
        return person_id, person_name, False

    resolved_name = str(
        recent_engagement.get("name")
        or recent_engagement.get("face_id")
        or recent_engagement.get("voice_id")
        or ""
    ).strip() or person_name
    _log.info(
        "[interaction] person resolution: pending-question continuity — "
        "person_id=%s name=%r voice_score=%.3f raw_best=%s source=%s",
        recent_id,
        resolved_name,
        speaker_score,
        raw_best_id,
        "durable_qa" if pending else "topic_thread",
    )
    return recent_id, resolved_name, True


def _single_visible_engaged_continuity_floor(
    *,
    ws_pid: Optional[int],
    raw_best_id: Optional[int],
) -> float:
    broad_floor = float(
        getattr(config, "SPEAKER_ID_SINGLE_VISIBLE_CONTINUITY_FLOOR", 0.45)
    )
    if _safe_int(raw_best_id) == _safe_int(ws_pid):
        match_floor = float(
            getattr(
                config,
                "SPEAKER_ID_SINGLE_VISIBLE_MATCH_FLOOR",
                broad_floor,
            )
        )
        return min(broad_floor, match_floor)
    return broad_floor


def _extract_offscreen_identify_reply(
    text: str,
    speaker_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Parse the engaged person's answer to "who just chimed in?"

    In this slot a bare one-word reply is usually a name. Trust that context
    before letting words like "Joy" drift into ordinary sentiment/content.
    """
    try:
        parsed = llm.extract_relationship_introduction(text, speaker_name)
    except Exception as exc:
        _log.debug("offscreen identify extract error: %s", exc)
        parsed = {"name": None, "relationship": None}

    intro_name = parsed.get("name")
    rel_label = parsed.get("relationship")
    if not intro_name:
        if "?" in (text or ""):
            return None, rel_label
        intro_name = _extract_introduced_name(text, allow_bare_name=True)
    if intro_name and _is_non_name_candidate(intro_name):
        intro_name = None
    return intro_name, rel_label


_THIRD_PERSON_PRESENCE_ANNOUNCEMENT_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2}\s*,?\s+"
    r"(?:(?:is|are|was|were)\s+(?:here|there|around|with\b|talking\b)|"
    r"(?:'s|’s)\s+(?:here|there|around|with\b|talking\b))",
    re.IGNORECASE,
)


def _looks_like_direct_offscreen_identity_answer(
    text: str,
    intro_name: Optional[str],
) -> bool:
    """
    True when an off-camera unknown appears to answer Rex's "who was that?"
    prompt directly with their own name.

    The engaged person may answer with a fuller phrase, but the unknown speaker
    commonly replies with just "JT" or "I'm Joy." Keep this narrow so an
    unrelated off-camera sentence does not get silently enrolled as a person.
    """
    if not intro_name:
        return False
    cleaned = (text or "").strip()
    if not cleaned or "?" in cleaned:
        return False
    if _THIRD_PERSON_PRESENCE_ANNOUNCEMENT_RE.match(cleaned):
        return False
    if _extract_introduced_name(cleaned, allow_bare_name=False):
        return True
    named_phrase = re.search(
        r"\b(?:it(?:'|’)?s|it\s+is|that(?:'|’)?s|that\s+is|"
        r"(?:his|her|their)\s+name\s+is|name(?:'|’)?s)\s+(.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if named_phrase and _same_person_name(named_phrase.group(1), intro_name):
        return True
    return (
        _same_person_name(cleaned, intro_name)
        and _name_word_count(intro_name) <= _NAME_MAX_WORDS
    )


def _is_offscreen_identify_cancel_reply(text: str) -> bool:
    plain = _plain_confirmation_text(text)
    return plain in {
        "cancel",
        "cancel that",
        "drop it",
        "forget it",
        "forget that",
        "ignore it",
        "ignore that",
        "never mind",
        "never mind that",
        "nevermind",
        "nevermind that",
        "skip it",
        "stop",
    }


def _offscreen_identity_enrollment_audio(
    pending_audio: object,
    reply_audio: Optional[np.ndarray],
    *,
    include_reply_audio: bool,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    if isinstance(pending_audio, np.ndarray) and len(pending_audio) > 0:
        arrays.append(pending_audio)
    if (
        include_reply_audio
        and isinstance(reply_audio, np.ndarray)
        and len(reply_audio) > 0
    ):
        arrays.append(reply_audio)
    if len(arrays) > 1:
        return np.concatenate(arrays)
    if arrays:
        return arrays[0]
    return np.zeros(1, dtype=np.float32)


def _audio_duration_secs(audio_array: Optional[np.ndarray]) -> float:
    if not isinstance(audio_array, np.ndarray) or len(audio_array) <= 0:
        return 0.0
    try:
        rate = float(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
    except Exception:
        rate = 16000.0
    if rate <= 0:
        return 0.0
    return float(len(audio_array)) / rate


def _voice_enrollment_sample_allowed(
    audio_array: Optional[np.ndarray],
    *,
    transcript_text: str = "",
    confirmed: bool = False,
) -> tuple[bool, str]:
    min_secs = float(getattr(config, "IDENTITY_VOICE_ENROLL_MIN_AUDIO_SECS", 1.2) or 0.0)
    min_words = int(getattr(config, "IDENTITY_VOICE_ENROLL_MIN_WORDS", 2) or 0)
    duration = _audio_duration_secs(audio_array)
    if duration < min_secs:
        return False, f"audio_too_short:{duration:.2f}s<{min_secs:.2f}s"
    if not confirmed and min_words > 0 and transcript_text:
        words = _word_count(transcript_text)
        if words < min_words:
            return False, f"too_few_words:{words}<{min_words}"
    return True, "ok"


def _safe_enroll_voice(
    person_id: int,
    audio_array: Optional[np.ndarray],
    *,
    transcript_text: str = "",
    source: str,
    confirmed: bool = False,
) -> bool:
    allowed, reason = _voice_enrollment_sample_allowed(
        audio_array,
        transcript_text=transcript_text,
        confirmed=confirmed,
    )
    if not allowed:
        _log.info(
            "[identity] skipped voice enrollment person_id=%s source=%s reason=%s",
            person_id,
            source,
            reason,
        )
        return False
    try:
        return bool(speaker_id.enroll_voice(person_id, audio_array))
    except Exception as exc:
        _log.warning("voice enrollment failed for person_id=%s source=%s: %s", person_id, source, exc)
        return False


def _handle_pending_offscreen_identify_reply(
    text: str,
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    audio_array: Optional[np.ndarray],
    anonymous_speaker_label: Optional[str],
) -> tuple[bool, Optional[str]]:
    """
    Consume a reply to Rex's off-camera "who was that?" question.

    Returns (consumed, response_text). response_text is present when Rex spoke
    an acknowledgement. The answer may come from the engaged person or directly
    from the off-camera unknown with a short self-name reply.
    """
    global _pending_offscreen_identify, _session_exchange_count

    pending = _pending_offscreen_identify
    if pending is None:
        return False, None

    now_mono = time.monotonic()
    ttl = float(getattr(config, "OFFSCREEN_IDENTIFY_WINDOW_SECS", 30.0))
    if (now_mono - pending["asked_at"]) > ttl:
        _log.info("[interaction] off-camera identify window expired — clearing")
        _pending_offscreen_identify = None
        return False, None

    prior_engaged_id = pending.get("prior_engaged_id")
    prior_engaged_name = pending.get("prior_engaged_name") or "friend"
    from_engaged_person = (
        person_id is not None
        and prior_engaged_id is not None
        and int(person_id) == int(prior_engaged_id)
    )

    if _is_offscreen_identify_cancel_reply(text):
        _pending_offscreen_identify = None
        ack_text = "Never mind. Bad sensor read on my end."
        if from_engaged_person:
            completed = _speak_blocking(
                ack_text,
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", ack_text)
            conv_log.log_rex(ack_text)
            _session_exchange_count += 1
            _register_rex_utterance(ack_text)
            repair_moves.mark_handled("misunderstood")
            _log.info(
                "[interaction] off-camera identify cancelled by engaged person "
                "completed=%s text=%r",
                completed,
                text,
            )
            return True, ack_text
        _log.info("[interaction] off-camera identify cancelled text=%r", text)
        return True, None

    intro_name, rel_label = _extract_offscreen_identify_reply(
        text,
        person_name or prior_engaged_name,
    )
    intro_name, rel_label = _filter_relationship_introduction_evidence(
        {"name": intro_name, "relationship": rel_label},
        text,
        person_name or prior_engaged_name,
        source="offscreen_identify",
    )
    from_unknown_self_answer = (
        person_id is None
        and _looks_like_direct_offscreen_identity_answer(text, intro_name)
    )
    if not from_engaged_person and not from_unknown_self_answer:
        return False, None

    if not intro_name:
        # No name in this reply. Drop the pending state so Rex doesn't badger;
        # if the off-camera voice speaks again we'll re-ask.
        _log.info(
            "[interaction] off-camera reply had no name — clearing pending: %r",
            text,
        )
        _pending_offscreen_identify = None
        repair = repair_moves.detect(text) if from_engaged_person else None
        repair_kind = str((repair or {}).get("kind") or "")
        if repair_kind in {
            "clarify",
            "misunderstood",
            "wrong_person",
            "factual",
            "bare_negation",
        }:
            ack_text = "Never mind. Bad sensor read on my end."
            completed = _speak_blocking(
                ack_text,
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", ack_text)
            conv_log.log_rex(ack_text)
            _session_exchange_count += 1
            _register_rex_utterance(ack_text)
            repair_moves.mark_handled(repair_kind)
            _log.info(
                "[interaction] off-camera identify repaired after confused reply "
                "kind=%s completed=%s",
                repair_kind,
                completed,
            )
            return True, ack_text
        return True, None

    new_pid = None
    try:
        new_pid, created = people_memory.find_or_create_person(intro_name)
        if new_pid is not None:
            enroll_audio = _offscreen_identity_enrollment_audio(
                pending.get("audio"),
                audio_array,
                include_reply_audio=from_unknown_self_answer,
            )
            _safe_enroll_voice(
                new_pid,
                enroll_audio,
                transcript_text=text,
                source="offscreen_identify",
                confirmed=from_engaged_person,
            )
            first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
            if created and first_inc > 0:
                people_memory.update_familiarity(new_pid, first_inc)

            if _has_unknown_visible_person():
                try:
                    from vision import camera as _cam_mod
                    from vision import face as _face_mod
                    frame = _cam_mod.capture_still()
                    if frame is not None:
                        if _face_mod.enroll_unknown_face(new_pid, frame):
                            threading.Thread(
                                target=_face_mod.update_appearance,
                                args=(new_pid, frame.copy()),
                                daemon=True,
                                name=f"appearance-enroll-{new_pid}",
                            ).start()
                except Exception as exc:
                    _log.warning(
                        "off-camera visible face enroll failed: %s",
                        exc,
                    )
            _bind_world_state_identity(new_pid, intro_name)
            retired_labels = {
                pending.get("anonymous_speaker_label"),
                anonymous_speaker_label,
            }
            for label in retired_labels:
                _retire_anonymous_speaker_slot(
                    label,
                    person_id=new_pid,
                    person_name=intro_name,
                )

            if rel_label and prior_engaged_id:
                try:
                    from memory import social as _social
                    _social.save_relationship(
                        from_person_id=prior_engaged_id,
                        to_person_id=new_pid,
                        relationship=rel_label,
                        described_by=prior_engaged_id,
                    )
                except Exception as exc:
                    _log.warning("off-camera relationship save failed: %s", exc)

            _log.info(
                "[interaction] off-camera speaker enrolled as %s (person_id=%s)%s",
                intro_name, new_pid,
                f" with relationship {rel_label!r}" if rel_label else "",
            )
            _episodic_person_enrolled(new_pid, intro_name, created=created)
    except Exception as exc:
        _log.error("off-camera identify enrollment failed: %s", exc)

    # Consume whether or not we succeeded — don't retry on the next turn.
    _pending_offscreen_identify = None
    if new_pid is None:
        return True, None

    ack_text = f"Got it: {intro_name}. Welcome to the frequency."
    try:
        introducer_name = _first_name_or(person_name or prior_engaged_name, "friend")
        ack_text = llm.get_response(
            f"You just learned the nearby/off-camera person's "
            f"name is {intro_name}. {introducer_name} introduced "
            f"or named them. In ONE very short in-character Rex line, "
            f"acknowledge {intro_name} by name and welcome "
            f"them. Address {intro_name}, not {introducer_name}. "
            f"Do not ask another question."
        ) or ack_text
    except Exception as exc:
        _log.debug("off-camera identify ack generation failed: %s", exc)
    _speak_blocking(ack_text)
    conv_memory.add_to_transcript("Rex", ack_text)
    conv_log.log_rex(ack_text)
    _session_exchange_count += 1
    _register_rex_utterance(ack_text)
    return True, ack_text


def _ask_visible_unknown_identity_question(
    text: str,
    *,
    recent_engagement: Optional[dict],
    group_chatter_active: bool,
    source: str,
) -> Optional[str]:
    """Ask a visible unknown speaker for their name and open enrollment window."""
    global _identity_prompt_until, _session_exchange_count

    if group_chatter_active:
        _log.info(
            "[interaction] suppressing visible-unknown identity ask during "
            "group chatter source=%s text=%r",
            source,
            text,
        )
        return None

    prior_first = ""
    if recent_engagement and recent_engagement.get("name"):
        prior_first = _first_name_or(recent_engagement.get("name"))
    if prior_first:
        ask_prompt = (
            f"A new face just appeared on camera and an unfamiliar voice "
            f"said: '{text}'. You don't recognize them. You were just "
            f"talking with '{prior_first}'. In ONE short in-character Rex "
            f"line, greet the newcomer, ask their name, AND ask how they "
            f"know {prior_first}. Warm, curious. One line ending in a "
            f"question mark."
        )
    else:
        ask_prompt = (
            f"A new face just appeared on camera and an unfamiliar voice "
            f"said: '{text}'. You don't recognize them. In ONE short "
            f"in-character Rex line, greet them and ask who they are / "
            f"what name to call them. One line ending in a question mark."
        )
    try:
        q_text = llm.get_response(ask_prompt)
    except Exception as exc:
        _log.debug("visible-unknown identity ask error source=%s: %s", source, exc)
        return None
    if not q_text:
        return None

    _speak_blocking(q_text)
    conv_memory.add_to_transcript("Rex", q_text)
    conv_log.log_rex(q_text)
    _session_exchange_count += 1
    _register_rex_utterance(q_text)
    _identity_prompt_until = time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS
    _log.info(
        "[interaction] visible-unknown identity ask source=%s prior=%r: %r",
        source,
        prior_first or None,
        q_text,
    )
    return q_text


def _bare_wake_should_ask_visible_unknown_identity(
    *,
    person_id: Optional[int],
    identity_prompt_active: bool,
    has_unknown_visible_or_recent: bool,
    game_conversation_lock: bool,
) -> bool:
    return (
        person_id is None
        and not identity_prompt_active
        and has_unknown_visible_or_recent
        and not game_conversation_lock
        and _pending_face_reveal_confirm is None
    )


def _has_unknown_visible_person() -> bool:
    """True if WorldState currently includes at least one person without a face match."""
    try:
        people = world_state.get("people")
    except Exception:
        return False
    for person in people:
        if person.get("person_db_id") is not None:
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        if person.get("face_box") or person.get("bounding_box") or person.get("bbox"):
            return True
    return False


def _has_unknown_visible_or_recent() -> bool:
    """True if an unknown face is visible now or within the configured grace window."""
    if _has_unknown_visible_person():
        return True
    try:
        return bool(
            consciousness.unknown_visible_recently(
                float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0))
            )
        )
    except Exception:
        return False


def _other_known_visible_recently(person_id: Optional[int]) -> bool:
    """True if another known participant is visible now or just flickered out."""
    try:
        return bool(
            consciousness.known_visible_recently_except(
                person_id,
                float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0)),
            )
        )
    except Exception:
        return False


def _known_person_visible_recently(person_id: Optional[int]) -> bool:
    try:
        return bool(
            consciousness.person_visible_recently(
                person_id,
                float(getattr(config, "UNKNOWN_FACE_RECENT_GRACE_SECS", 6.0)),
            )
        )
    except Exception:
        return False


def _episodic_person_enrolled(person_id, name, *, created: bool) -> None:
    """Log "I met <name>" to rex.db — only on a genuinely NEW enrollment (created=True),
    so re-binding an existing person doesn't re-log a first meeting. Gated + failure-safe
    (episodes.* no-ops under the test runner / when episodic memory is disabled)."""
    if not created:
        return
    try:
        episodes_memory.record_person_enrolled(
            person_id if isinstance(person_id, int) else None, name,
        )
    except Exception as exc:
        _log.debug("episodic person_enrolled failed: %s", exc)


def _bind_world_state_identity(person_id: int, name: str) -> None:
    """
    Attach the newly enrolled identity to the first unknown world-state person slot.
    """
    try:
        def _apply_bind(people):
            for person in people:
                if person.get("person_db_id") is None or person.get("face_id") is None:
                    person["person_db_id"] = person_id
                    person["face_id"] = name
                    person["voice_id"] = name
                    return people
            return None

        world_state.mutate("people", _apply_bind)

        crowd = world_state.get("crowd")
        crowd["dominant_speaker"] = name
        world_state.update("crowd", crowd)
    except Exception as exc:
        _log.debug("world_state identity bind failed: %s", exc)


def _handle_relationship_reply(
    rel_ctx: dict,
    user_text: str,
    speaker_person_id: Optional[int],
    speaker_person_name: Optional[str],
) -> Optional[str]:
    """
    Process a reply to one of two relationship-flow questions:

    Mode A — "who's this?" asked of the ENGAGED person.
        ctx has engaged_person_id but no newcomer_person_id. The speaker is
        expected to be the engaged person; their reply is parsed for BOTH a
        {name, relationship} and we enroll a brand-new person for the newcomer.

    Mode B — "how do you know <engaged>?" asked of the NEWCOMER after identity
        enrollment.
        ctx has both engaged_person_id AND newcomer_person_id. The speaker is
        expected to be the newcomer (already enrolled). Reply is parsed for a
        {relationship} (the name is already known); the edge is saved from the
        newcomer to the engaged person (speaker's perspective).

    In either mode, if the speaker is plausibly correct and the parse produces
    a relationship label, an edge is stored and the slot is marked handled.
    """
    global _pending_introduction

    engaged_id = rel_ctx.get("engaged_person_id")
    engaged_name = rel_ctx.get("engaged_name") or "friend"
    newcomer_pid_pre = rel_ctx.get("newcomer_person_id")  # Mode B only
    slot_id = rel_ctx.get("slot_id") or ""

    mode_b = newcomer_pid_pre is not None

    # Speaker validity check. If this reply clearly came from a different
    # person than the one Rex asked, don't consume the prompt.
    expected_speaker = newcomer_pid_pre if mode_b else engaged_id
    if (
        speaker_person_id is not None
        and expected_speaker is not None
        and speaker_person_id != expected_speaker
    ):
        _log.info(
            "[interaction] relationship reply came from person_id=%s but Rex asked "
            "person_id=%s (mode_%s) — ignoring",
            speaker_person_id, expected_speaker, "B" if mode_b else "A",
        )
        return None

    # Extract. In both modes the LLM call returns {name, relationship}; in mode
    # B we ignore the name because we already know the speaker.
    try:
        parsed = llm.extract_relationship_introduction(
            user_text,
            speaker_person_name or (engaged_name if not mode_b else "newcomer"),
        )
    except Exception as exc:
        _log.debug("relationship extraction error: %s", exc)
        parsed = {"name": None, "relationship": None}

    name, relationship = _filter_relationship_introduction_evidence(
        parsed,
        user_text,
        speaker_person_name or (engaged_name if not mode_b else "newcomer"),
        source="relationship_prompt_mode_b" if mode_b else "relationship_prompt_mode_a",
    )

    # ── Mode B: just save the edge, no enrollment needed ──────────────────────
    if mode_b:
        if not relationship:
            _log.info(
                "[interaction] mode-B relationship reply had no label — user_text=%r parsed=%r",
                user_text, parsed,
            )
            consciousness.note_relationship_slot_handled(slot_id)
            return None
        try:
            from memory import social as social_memory
            # Speaker (newcomer) is describing their relationship to the engaged
            # person. from_id = newcomer (speaker), to_id = engaged.
            social_memory.save_relationship(
                from_person_id=newcomer_pid_pre,
                to_person_id=engaged_id,
                relationship=relationship,
                described_by=newcomer_pid_pre,
            )
            _log.info(
                "[interaction] mode-B saved edge: person_id=%s --[%s]--> person_id=%s (%s)",
                newcomer_pid_pre, relationship, engaged_id, engaged_name,
            )
        except Exception as exc:
            _log.warning("social.save_relationship failed (mode B): %s", exc)
        consciousness.note_relationship_slot_handled(slot_id)
        return None

    # ── Mode A: enroll newcomer + save edge ────────────────────────────────────
    if not name:
        if relationship:
            introducer_for_pending = engaged_id or speaker_person_id
            if introducer_for_pending is None:
                consciousness.note_relationship_slot_handled(slot_id)
                return None
            _pending_introduction = {
                "introducer_id": introducer_for_pending,
                "introducer_name": engaged_name,
                "relationship": relationship,
                "visible_newcomer": True,
                "created_at": time.monotonic(),
                "asked_at": time.monotonic(),
            }
            consciousness.note_relationship_slot_handled(slot_id)
            try:
                first = _first_name_or(engaged_name, "friend")
                return llm.get_response(
                    f"{first} said the visible newcomer is their {relationship}, "
                    f"but did not give the person's name yet. In ONE short "
                    f"in-character Rex line, ask {first} ONLY for the newcomer's "
                    f"name. Do NOT ask how they are related. Do NOT imply the "
                    f"newcomer is related to Rex."
                )
            except Exception:
                return f"Got the {relationship} part. What name am I filing for them?"

        # Deflection or no name given. Don't badger — mark slot handled so Rex
        # moves on.
        _log.info(
            "[interaction] mode-A relationship reply had no name — user_text=%r parsed=%r",
            user_text, parsed,
        )
        consciousness.note_relationship_slot_handled(slot_id)
        return None

    # Enroll the newcomer. Use enroll_unknown_face so we don't rebind the
    # engaged person's face to the new name.
    new_id = None
    try:
        from vision import camera as camera_mod
        from vision import face as face_mod

        new_id, created = people_memory.find_or_create_person(name)
        if new_id is None:
            _log.error("[interaction] failed to create DB row for newcomer %r", name)
            consciousness.note_relationship_slot_handled(slot_id)
            return None
        if engaged_id is not None and int(new_id) == int(engaged_id):
            _log.warning(
                "[interaction] refusing mode-A self-introduction: parsed newcomer %r resolves to engaged person_id=%s; text=%r",
                name,
                engaged_id,
                user_text,
            )
            consciousness.note_relationship_slot_handled(slot_id)
            return None

        first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
        if created and first_inc > 0:
            people_memory.update_familiarity(new_id, first_inc)

        face_enrolled = False
        frame = camera_mod.capture_still()
        if frame is not None:
            face_enrolled = face_mod.enroll_unknown_face(new_id, frame)
            if face_enrolled:
                threading.Thread(
                    target=face_mod.update_appearance,
                    args=(new_id, frame.copy()),
                    daemon=True,
                    name=f"appearance-enroll-{new_id}",
                ).start()

        if face_enrolled:
            _bind_world_state_identity(new_id, name)
        if engaged_id:
            _store_introduction_memories(
                int(engaged_id),
                engaged_name,
                new_id,
                name,
                relationship,
            )

        _log.info(
            "[interaction] mode-A enrolled newcomer %s (person_id=%s) as %s of %s",
            name, new_id, relationship or "acquaintance", engaged_name,
        )
        _episodic_person_enrolled(new_id, name, created=created)
        introducer_for_ack = engaged_id or speaker_person_id
        if introducer_for_ack is None:
            return f"{_first_name_or(name, 'there')}, welcome. Identity filed under 'better than mystery organic.'"
        return _intro_ack_and_followup(
            int(introducer_for_ack),
            engaged_name,
            new_id,
            name,
            relationship,
            visible_newcomer=True,
        )
    except Exception as exc:
        _log.error("relationship enrollment failed: %s", exc)
        return None
    finally:
        consciousness.note_relationship_slot_handled(slot_id)


def _handle_common_first_name_last_name_reply(
    text: str,
    audio_array: Optional[np.ndarray] = None,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Complete a deferred identity enrollment after Rex asked for a last name.

    Returns (response_text, enrolled_person_id, full_name). response_text may be
    None when the pending context expired or the reply was not a usable last name.
    """
    global _pending_common_first_name_identity, _pending_common_first_name_introduction

    ctx = _pending_common_first_name_identity
    if not _common_first_name_context_fresh(ctx):
        _pending_common_first_name_identity = None
        return None, None, None

    first_name = str(ctx.get("first_name") or "").strip()
    refused_last_name = _is_last_name_refusal(text, first_name)
    if refused_last_name:
        last_name = None
    else:
        last_name = _extract_last_name_reply(text, first_name)
    if not first_name:
        return None, None, None
    if not last_name and not refused_last_name:
        return None, None, None

    full_name = first_name if refused_last_name else f"{first_name} {last_name}"
    stored_audio = ctx.get("audio")
    if (
        isinstance(stored_audio, np.ndarray)
        and len(stored_audio) > 0
        and isinstance(audio_array, np.ndarray)
        and len(audio_array) > 0
    ):
        enroll_audio = np.concatenate([stored_audio, audio_array])
    elif isinstance(stored_audio, np.ndarray) and len(stored_audio) > 0:
        enroll_audio = stored_audio
    elif isinstance(audio_array, np.ndarray) and len(audio_array) > 0:
        enroll_audio = audio_array
    else:
        enroll_audio = np.zeros(1, dtype=np.float32)

    prior_engagement = ctx.get("prior_engagement")
    enrolled_id = _enroll_new_person(
        full_name,
        enroll_audio,
        enroll_unknown_face=bool(prior_engagement),
    )
    _pending_common_first_name_identity = None
    if enrolled_id is None:
        return None, None, None

    _retire_anonymous_speaker_slot(
        ctx.get("anonymous_speaker_label"),
        person_id=enrolled_id,
        person_name=full_name,
    )

    if prior_engagement and prior_engagement.get("person_id") != enrolled_id:
        _pending_post_greet_relationship[0] = {
            "prior_engaged_id": prior_engagement["person_id"],
            "prior_engaged_name": prior_engagement.get("name"),
            "newcomer_person_id": enrolled_id,
            "newcomer_name": full_name,
        }

    if refused_last_name:
        _remember_last_name_declined(enrolled_id, first_name)
        _log.info(
            "[identity] last-name request declined for %r; filing first name only",
            first_name,
        )
        response = f"Fine. Filed as {first_name}. The memory banks will squint and cope."
    else:
        response = f"Filed as {full_name}. The memory banks have stopped panicking."
    return response, enrolled_id, full_name


def _handle_common_first_name_intro_last_name_reply(text: str) -> Optional[str]:
    """Complete an explicit introduction delayed for a common first name."""
    global _pending_common_first_name_introduction

    ctx = _pending_common_first_name_introduction
    if not _common_first_name_context_fresh(ctx):
        _pending_common_first_name_introduction = None
        return None

    first_name = str(ctx.get("first_name") or "").strip()
    refused_last_name = _is_last_name_refusal(text, first_name)
    if refused_last_name:
        last_name = None
    else:
        last_name = _extract_last_name_reply(text, first_name)
    if not first_name:
        return None
    if not last_name and not refused_last_name:
        return None

    full_name = first_name if refused_last_name else f"{first_name} {last_name}"
    introducer_id = int(ctx["introducer_id"])
    introducer_name = str(ctx.get("introducer_name") or "friend")
    relationship = ctx.get("relationship")
    visible_newcomer = bool(ctx.get("visible_newcomer", True))
    subject_kind = str(ctx.get("subject_kind") or "person")

    new_id = _enroll_introduced_person(
        full_name,
        introducer_id,
        introducer_name,
        relationship,
        enroll_visible_face=visible_newcomer,
    )
    _pending_common_first_name_introduction = None
    if new_id is None:
        return None

    if refused_last_name:
        _remember_last_name_declined(new_id, first_name)
        _log.info(
            "[introduction] last-name request declined for %r; filing first name only",
            first_name,
        )
    else:
        _log.info(
            "[introduction] %s introduced %s as %s (person_id=%s) after last-name disambiguation",
            introducer_name,
            full_name,
            relationship or "acquaintance",
            new_id,
        )
    return _intro_ack_and_followup(
        introducer_id,
        introducer_name,
        new_id,
        full_name,
        relationship,
        subject_kind=subject_kind,
        visible_newcomer=visible_newcomer,
    )


def _handle_existing_common_first_name_last_name_reply(text: str) -> Optional[str]:
    """Complete a last-name prompt for a returning person already in memory."""
    global _pending_existing_common_first_name, _pending_identity_match_confirmation

    ctx = _pending_existing_common_first_name
    if not _common_first_name_context_fresh(ctx):
        _pending_existing_common_first_name = None
        return None

    first_name = str(ctx.get("first_name") or "").strip()
    person_id = ctx.get("person_id")
    if not first_name or person_id is None:
        _pending_existing_common_first_name = None
        return None

    refused_last_name = _is_last_name_refusal(text, first_name)
    last_name = None if refused_last_name else _extract_last_name_reply(text, first_name)
    if not refused_last_name and not last_name:
        return None

    _pending_existing_common_first_name = None
    person_id = int(person_id)
    _common_first_name_prompted_this_session.add(person_id)

    if refused_last_name:
        _remember_last_name_declined(person_id, first_name)
        _log.info(
            "[identity] returning person_id=%s declined last-name request; keeping %r",
            person_id,
            first_name,
        )
        return f"Fine. I'll keep you as {first_name}. The filing cabinet is judging both of us."

    full_name = f"{first_name} {last_name}"
    if not people_memory.rename_person(person_id, full_name):
        _log.info(
            "[identity] returning person_id=%s last-name rename blocked target=%r",
            person_id,
            full_name,
        )
        return f"I couldn't safely rename that memory to {full_name}."

    _refresh_world_state_person_name(person_id, full_name)
    _log.info(
        "[identity] returning person_id=%s upgraded common first name %r -> %r",
        person_id,
        first_name,
        full_name,
    )
    return f"Updated: {full_name}. The memory banks have unclenched."


def _should_defer_existing_common_first_name_prompt(text: str) -> bool:
    """Do not hijack a clear command just to ask a known person's last name."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    try:
        match = command_parser.parse(cleaned)
    except Exception:
        match = None
    if match is not None:
        return True
    try:
        intent = intent_classifier._deterministic_label(cleaned)
    except Exception:
        intent = "general"
    return intent != "general"


def _maybe_prompt_existing_common_first_name(
    person_id: Optional[int],
    person_name: Optional[str],
    current_text: str = "",
) -> Optional[str]:
    """Ask returning single-name-only people for a last name once per session."""
    global _pending_existing_common_first_name, _pending_identity_match_confirmation

    if not _known_person_needs_last_name(person_id, person_name):
        return None
    if _should_defer_existing_common_first_name_prompt(current_text):
        _log.info(
            "[identity] deferred last-name prompt for person_id=%s name=%r due to command text=%r",
            person_id,
            person_name,
            current_text,
        )
        return None

    pid = int(person_id)
    min_turns = _last_name_prompt_min_person_turns()
    turn_count = _session_person_turn_counts.get(pid, 0)
    if not _person_ready_for_last_name_prompt(pid):
        _log.info(
            "[identity] deferring last-name prompt for person_id=%s name=%r "
            "until longer conversation turns=%s/%s",
            pid,
            person_name,
            turn_count,
            min_turns,
        )
        return None

    first_name = _normalize_name(person_name or "") or (person_name or "").strip()
    if not first_name:
        return None

    _pending_existing_common_first_name = {
        "person_id": pid,
        "first_name": first_name,
        "asked_at": time.monotonic(),
    }
    _common_first_name_prompted_this_session.add(pid)
    _log.info(
        "[identity] returning common first-name-only person_id=%s name=%r needs last-name disambiguation",
        pid,
        first_name,
    )
    return _format_common_first_name_last_name_prompt(first_name)


def _mark_single_name_for_later_last_name(person_id: Optional[int], person_name: Optional[str]) -> None:
    if person_id is None:
        return
    normalized = _normalize_name(person_name or "")
    if not normalized or _name_word_count(normalized) != 1 or looks_like_initials(normalized):
        return
    _log.info(
        "[identity] person_id=%s filed under single name %r; will ask for last name later",
        person_id,
        normalized,
    )


def _note_session_person_turn(person_id: Optional[int]) -> None:
    pid = _safe_int(person_id)
    if pid is None:
        return
    _session_person_turn_counts[pid] = _session_person_turn_counts.get(pid, 0) + 1


def _last_name_prompt_min_person_turns() -> int:
    configured = getattr(config, "COMMON_FIRST_NAME_LAST_NAME_MIN_PERSON_TURNS", None)
    if configured is None:
        configured = getattr(config, "LONG_CONVERSATION_MIN_EXCHANGES", 5)
    try:
        return max(0, int(configured))
    except (TypeError, ValueError):
        return 5


def _person_ready_for_last_name_prompt(person_id: Optional[int]) -> bool:
    pid = _safe_int(person_id)
    if pid is None:
        return False
    min_turns = _last_name_prompt_min_person_turns()
    if min_turns <= 0:
        return True
    return _session_person_turn_counts.get(pid, 0) >= min_turns


def _identity_match_confirmation_fresh(ctx: Optional[dict]) -> bool:
    if not ctx:
        return False
    ttl = float(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS", 30.0))
    return (time.monotonic() - float(ctx.get("asked_at", 0.0))) <= max(1.0, ttl)


def _identity_confirmation_affirms(text: str) -> bool:
    plain = _plain_confirmation_text(text)
    return plain in {"yes", "yeah", "yep", "correct", "that is me", "that's me", "thats me"} or bool(
        re.search(r"\b(yes|yeah|yep|correct|that'?s me|that is me)\b", text or "", re.IGNORECASE)
    )


def _identity_confirmation_declines(text: str) -> bool:
    plain = _plain_confirmation_text(text)
    return plain in {"no", "nope", "not me", "someone new", "new person"} or bool(
        re.search(r"\b(no|nope|not me|someone new|new person|different person)\b", text or "", re.IGNORECASE)
    )


def _identity_match_prompt(candidate_name: str, existing_name: str, match_type: str) -> str:
    candidate_first = _first_name_or(candidate_name, "that")
    if match_type == "first_name":
        return (
            f"Are you {existing_name}? I'll update my records so I don't create "
            f"another {candidate_first}."
        )
    return f"Did you mean {existing_name}, or is {candidate_name} someone new?"


def _maybe_ask_identity_match_confirmation(
    candidate_name: str,
    audio_array: np.ndarray,
    *,
    prior_engagement: Optional[dict] = None,
    anonymous_speaker_label: Optional[str] = None,
    enroll_unknown_face: bool = False,
    defer_face_enrollment: bool = False,
    source: str = "identity",
) -> bool:
    """Ask before binding a weak name candidate to an existing person."""
    global _pending_identity_match_confirmation, _identity_prompt_until, _session_exchange_count

    try:
        match = people_memory.find_potential_person_match(candidate_name)
    except Exception as exc:
        _log.debug("identity match lookup failed for %r: %s", candidate_name, exc)
        return False
    if not match or match.get("match_type") not in {"first_name", "fuzzy"}:
        return False

    person = match.get("person") or {}
    existing_id = person.get("id")
    existing_name = str(person.get("name") or "").strip()
    clean_candidate = str(match.get("candidate_name") or candidate_name or "").strip()
    if existing_id is None or not existing_name or not clean_candidate:
        return False

    _pending_identity_match_confirmation = {
        "candidate_name": clean_candidate,
        "existing_person_id": int(existing_id),
        "existing_name": existing_name,
        "match_type": match.get("match_type"),
        "audio": audio_array.copy() if isinstance(audio_array, np.ndarray) else np.zeros(1, dtype=np.float32),
        "prior_engagement": prior_engagement,
        "anonymous_speaker_label": anonymous_speaker_label,
        "enroll_unknown_face": bool(enroll_unknown_face),
        "defer_face_enrollment": bool(defer_face_enrollment),
        "asked_at": time.monotonic(),
        "source": source,
    }
    _identity_prompt_until = max(
        _identity_prompt_until,
        time.monotonic() + float(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS", 30.0)),
    )
    prompt = _identity_match_prompt(clean_candidate, existing_name, str(match.get("match_type") or ""))
    _log.info(
        "[identity] asking match confirmation candidate=%r existing=%r type=%s source=%s",
        clean_candidate,
        existing_name,
        match.get("match_type"),
        source,
    )
    _speak_blocking(prompt, emotion="curious", pre_beat_ms=100, post_beat_ms_override=200)
    conv_memory.add_to_transcript("Rex", prompt)
    conv_log.log_rex(prompt)
    _session_exchange_count += 1
    _register_rex_utterance(prompt)
    return True


def _attach_identity_sample_to_person(
    person_id: int,
    person_name: str,
    audio_array: np.ndarray,
    *,
    enroll_unknown_face: bool = False,
    defer_face_enrollment: bool = False,
) -> None:
    _safe_enroll_voice(
        person_id,
        audio_array,
        source="identity_alias_refresh",
        confirmed=True,
    )

    def _enroll_face_biometric() -> None:
        try:
            from vision import camera as camera_mod
            from vision import face as face_mod

            if defer_face_enrollment or enroll_unknown_face:
                frame = camera_mod.capture_current_gaze(
                    settle_secs=float(
                        getattr(config, "IDENTITY_FACE_ENROLL_CURRENT_GAZE_SETTLE_SECS", 0.25)
                        or 0.0
                    )
                )
            else:
                frame = camera_mod.capture_still()
            if frame is None:
                return
            if enroll_unknown_face:
                face_enrolled = face_mod.enroll_unknown_face(person_id, frame)
            else:
                face_enrolled = face_mod.enroll_face(person_id, frame)
            if face_enrolled:
                threading.Thread(
                    target=face_mod.update_appearance,
                    args=(person_id, frame.copy()),
                    daemon=True,
                    name=f"appearance-enroll-{person_id}",
                ).start()
        except Exception as exc:
            _log.warning("face alias refresh failed for person_id=%s: %s", person_id, exc)

    if defer_face_enrollment:
        threading.Thread(
            target=_enroll_face_biometric,
            daemon=True,
            name=f"face-alias-enroll-{person_id}",
        ).start()
    else:
        _enroll_face_biometric()
    _bind_world_state_identity(person_id, person_name)


def _handle_pending_identity_match_confirmation(
    text: str,
    audio_array: np.ndarray,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Consume yes/no after Rex asks whether a weak name matched an existing person."""
    global _pending_identity_match_confirmation, _pending_common_first_name_identity

    ctx = _pending_identity_match_confirmation
    if not _identity_match_confirmation_fresh(ctx):
        _pending_identity_match_confirmation = None
        return None, None, None

    candidate = str(ctx.get("candidate_name") or "").strip()
    existing_id = ctx.get("existing_person_id")
    existing_name = str(ctx.get("existing_name") or "").strip()
    if not candidate or existing_id is None or not existing_name:
        _pending_identity_match_confirmation = None
        return None, None, None

    if _identity_confirmation_affirms(text):
        person_id = int(existing_id)
        stored_audio = ctx.get("audio")
        if (
            isinstance(stored_audio, np.ndarray)
            and len(stored_audio) > 0
            and isinstance(audio_array, np.ndarray)
            and len(audio_array) > 0
        ):
            enroll_audio = np.concatenate([stored_audio, audio_array])
        elif isinstance(stored_audio, np.ndarray) and len(stored_audio) > 0:
            enroll_audio = stored_audio
        elif isinstance(audio_array, np.ndarray) and len(audio_array) > 0:
            enroll_audio = audio_array
        else:
            enroll_audio = np.zeros(1, dtype=np.float32)

        try:
            people_memory.add_alias(
                person_id,
                candidate,
                source=f"confirmed_{ctx.get('match_type') or 'match'}",
            )
        except Exception as exc:
            _log.debug("confirmed alias save failed: %s", exc)
        _attach_identity_sample_to_person(
            person_id,
            existing_name,
            enroll_audio,
            enroll_unknown_face=bool(ctx.get("enroll_unknown_face")),
            defer_face_enrollment=bool(ctx.get("defer_face_enrollment")),
        )
        _retire_anonymous_speaker_slot(
            ctx.get("anonymous_speaker_label"),
            person_id=person_id,
            person_name=existing_name,
        )
        _pending_identity_match_confirmation = None
        response = (
            f"Got it. I'll treat {candidate} as {existing_name} and keep that record tidy."
        )
        return response, person_id, existing_name

    if _identity_confirmation_declines(text):
        _pending_identity_match_confirmation = None
        first_name = _first_name_or(_normalize_name(candidate) or candidate)
        if _name_word_count(candidate) >= 2:
            person_id = _enroll_new_person(
                candidate,
                ctx.get("audio") if isinstance(ctx.get("audio"), np.ndarray) else audio_array,
                enroll_unknown_face=bool(ctx.get("enroll_unknown_face")),
                defer_face_enrollment=bool(ctx.get("defer_face_enrollment")),
            )
            if person_id is not None:
                _mark_single_name_for_later_last_name(person_id, candidate)
                response = f"Got it. I'll save {candidate} as a separate person."
                return response, person_id, candidate
            return None, None, None

        _pending_common_first_name_identity = {
            "first_name": first_name,
            "audio": ctx.get("audio") if isinstance(ctx.get("audio"), np.ndarray) else audio_array.copy(),
            "asked_at": time.monotonic(),
            "prior_engagement": ctx.get("prior_engagement"),
            "anonymous_speaker_label": ctx.get("anonymous_speaker_label"),
        }
        response = (
            f"Got it. What's your last name so I don't mix you up with {existing_name}?"
        )
        return response, None, None

    return None, None, None


def _prompted_name_confirmation_fresh(ctx: Optional[dict]) -> bool:
    if not ctx:
        return False
    ttl = float(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS", 30.0))
    return (time.monotonic() - float(ctx.get("asked_at", 0.0))) <= max(1.0, ttl)


def _maybe_ask_prompted_name_confirmation(
    candidate_name: str,
    audio_array: np.ndarray,
    *,
    raw_text: str,
    prior_engagement: Optional[dict] = None,
    anonymous_speaker_label: Optional[str] = None,
    enroll_unknown_face: bool = False,
    defer_face_enrollment: bool = False,
) -> bool:
    """Ask before saving a prompted name when the transcript looks contaminated."""
    global _pending_prompted_name_confirmation, _identity_prompt_until, _session_exchange_count

    if not _prompted_name_reply_needs_confirmation(raw_text, candidate_name):
        return False

    _pending_prompted_name_confirmation = {
        "candidate_name": candidate_name,
        "raw_text": raw_text,
        "audio": audio_array.copy() if isinstance(audio_array, np.ndarray) else np.zeros(1, dtype=np.float32),
        "prior_engagement": prior_engagement,
        "anonymous_speaker_label": anonymous_speaker_label,
        "enroll_unknown_face": bool(enroll_unknown_face),
        "defer_face_enrollment": bool(defer_face_enrollment),
        "asked_at": time.monotonic(),
    }
    _identity_prompt_until = max(
        _identity_prompt_until,
        time.monotonic() + float(getattr(config, "COMMON_FIRST_NAME_LAST_NAME_WINDOW_SECS", 30.0)),
    )
    prompt = (
        f"I heard {candidate_name}. Do I save that, or did the microphone embarrass itself?"
    )
    _log.info(
        "[identity] asking prompted-name confirmation candidate=%r raw=%r",
        candidate_name,
        raw_text,
    )
    _speak_blocking(prompt, emotion="curious", pre_beat_ms=100, post_beat_ms_override=200)
    conv_memory.add_to_transcript("Rex", prompt)
    conv_log.log_rex(prompt)
    _session_exchange_count += 1
    _register_rex_utterance(prompt)
    return True


def _handle_pending_prompted_name_confirmation(
    text: str,
    audio_array: np.ndarray,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Consume yes/no after confirming a noisy prompted identity transcript."""
    global _pending_prompted_name_confirmation, _identity_prompt_until

    ctx = _pending_prompted_name_confirmation
    if not _prompted_name_confirmation_fresh(ctx):
        _pending_prompted_name_confirmation = None
        return None, None, None

    candidate = str(ctx.get("candidate_name") or "").strip()
    if not candidate:
        _pending_prompted_name_confirmation = None
        return None, None, None

    if _identity_confirmation_affirms(text):
        stored_audio = ctx.get("audio")
        if (
            isinstance(stored_audio, np.ndarray)
            and len(stored_audio) > 0
            and isinstance(audio_array, np.ndarray)
            and len(audio_array) > 0
        ):
            enroll_audio = np.concatenate([stored_audio, audio_array])
        elif isinstance(stored_audio, np.ndarray) and len(stored_audio) > 0:
            enroll_audio = stored_audio
        elif isinstance(audio_array, np.ndarray) and len(audio_array) > 0:
            enroll_audio = audio_array
        else:
            enroll_audio = np.zeros(1, dtype=np.float32)

        person_id = _enroll_new_person(
            candidate,
            enroll_audio,
            enroll_unknown_face=bool(ctx.get("enroll_unknown_face")),
            defer_face_enrollment=bool(ctx.get("defer_face_enrollment")),
        )
        _pending_prompted_name_confirmation = None
        _identity_prompt_until = 0.0
        if person_id is None:
            return None, None, None
        _retire_anonymous_speaker_slot(
            ctx.get("anonymous_speaker_label"),
            person_id=person_id,
            person_name=candidate,
        )
        _mark_single_name_for_later_last_name(person_id, candidate)
        return _identity_enrollment_ack(candidate), person_id, candidate

    if _identity_confirmation_declines(text):
        _pending_prompted_name_confirmation = None
        _identity_prompt_until = time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS
        return (
            "Good. Then give me the name again, cleanly. Apparently the microphone wanted a solo.",
            None,
            None,
        )

    replacement = _extract_introduced_name(text, allow_bare_name=True)
    if replacement and not _prompted_name_reply_needs_confirmation(text, replacement):
        _pending_prompted_name_confirmation = None
        _identity_prompt_until = 0.0
        person_id = _enroll_new_person(
            replacement,
            audio_array,
            enroll_unknown_face=bool(ctx.get("enroll_unknown_face")),
            defer_face_enrollment=bool(ctx.get("defer_face_enrollment")),
        )
        if person_id is None:
            return None, None, None
        _retire_anonymous_speaker_slot(
            ctx.get("anonymous_speaker_label"),
            person_id=person_id,
            person_name=replacement,
        )
        _mark_single_name_for_later_last_name(person_id, replacement)
        return _identity_enrollment_ack(replacement), person_id, replacement

    return None, None, None


def _maybe_auto_refresh_voice(
    person_id: int,
    voice_score: float,
    audio_array: np.ndarray,
) -> None:
    """
    On high-confidence face+voice agreement, append the current audio as an
    additional voice biometric row for this person, up to a per-person cap.

    Runs asynchronously so the enrollment (which recomputes an embedding via
    Resemblyzer) doesn't delay Rex's response. Rate-limited to one refresh per
    person per session.
    """
    if person_id is None:
        return
    if person_id in _voice_refreshed_this_session:
        return
    min_score = float(getattr(config, "AUTO_VOICE_REFRESH_MIN_SCORE", 0.90))
    if voice_score < min_score:
        return
    max_samples = int(getattr(config, "AUTO_VOICE_REFRESH_MAX_SAMPLES", 5))
    current = people_memory.count_biometrics(person_id, "voice")
    if current >= max_samples:
        _voice_refreshed_this_session.add(person_id)  # don't keep checking
        return

    _voice_refreshed_this_session.add(person_id)
    audio_copy = audio_array.copy()

    def _task() -> None:
        try:
            ok = _safe_enroll_voice(
                person_id,
                audio_copy,
                source="auto_voice_refresh",
                confirmed=True,
            )
            if ok:
                new_total = people_memory.count_biometrics(person_id, "voice")
                _log.info(
                    "[interaction] auto-refreshed voice biometric for person_id=%s "
                    "(score=%.3f, now %d sample(s))",
                    person_id, voice_score, new_total,
                )
        except Exception as exc:
            _log.warning("auto-refresh voice enrollment failed: %s", exc)

    threading.Thread(
        target=_task, daemon=True, name=f"auto-voice-refresh-{person_id}"
    ).start()


def _enroll_new_person(
    name: str,
    audio_array: np.ndarray,
    enroll_unknown_face: bool = False,
    defer_face_enrollment: bool = False,
) -> Optional[int]:
    """
    Enroll a brand-new person and attach available voice/face biometrics.
    Returns person_id on success.

    If enroll_unknown_face=True, use face_mod.enroll_unknown_face which picks
    the largest face NOT already matched to an existing known person. Use this
    when a known person is visible alongside the newcomer so we don't rebind
    the known person's face to the new name.
    """
    person_id, created = people_memory.find_or_create_person(name)
    if person_id is None:
        _log.error("failed to enroll new person row for name=%r", name)
        return None

    first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
    if created and first_inc > 0:
        people_memory.update_familiarity(person_id, first_inc)

    _safe_enroll_voice(
        person_id,
        audio_array,
        source="new_person",
        confirmed=True,
    )

    def _enroll_face_biometric() -> None:
        try:
            from vision import camera as camera_mod
            from vision import face as face_mod

            if defer_face_enrollment or enroll_unknown_face:
                frame = camera_mod.capture_current_gaze(
                    settle_secs=float(
                        getattr(config, "IDENTITY_FACE_ENROLL_CURRENT_GAZE_SETTLE_SECS", 0.25)
                        or 0.0
                    )
                )
            else:
                frame = camera_mod.capture_still()
            if frame is not None:
                if enroll_unknown_face:
                    face_enrolled = face_mod.enroll_unknown_face(person_id, frame)
                else:
                    face_enrolled = face_mod.enroll_face(person_id, frame)
                if face_enrolled:
                    # Appearance extraction is useful but non-blocking.
                    threading.Thread(
                        target=face_mod.update_appearance,
                        args=(person_id, frame.copy()),
                        daemon=True,
                        name=f"appearance-enroll-{person_id}",
                    ).start()
        except Exception as exc:
            _log.warning("face enrollment failed for person_id=%s: %s", person_id, exc)

    if defer_face_enrollment:
        threading.Thread(
            target=_enroll_face_biometric,
            daemon=True,
            name=f"face-enroll-{person_id}",
        ).start()
    else:
        _enroll_face_biometric()

    _bind_world_state_identity(person_id, name)
    _log.info("[interaction] enrolled new person: %s (person_id=%s)", name, person_id)
    _episodic_person_enrolled(person_id, name, created=created)
    return person_id


_INTRO_INVERSE_RELATIONSHIP = {
    "father": "child",
    "mother": "child",
    "parent": "child",
    "dad": "child",
    "mom": "child",
    "aunt": "niece_or_nephew",
    "uncle": "niece_or_nephew",
    "boss": "employee",
    "supervisor": "employee",
    "manager": "employee",
    "dog": "owner",
    "cat": "owner",
    "pet": "owner",
}
_INTRO_SYMMETRIC_RELATIONSHIPS = {
    "friend", "best_friend", "partner", "girlfriend", "boyfriend", "wife",
    "husband", "spouse", "sister", "brother", "sibling", "cousin",
    "coworker", "colleague", "roommate", "neighbor", "fiance", "fiancee",
}
_INTRO_SELF_EXPLANATORY_RELATIONSHIPS = {
    "father", "mother", "parent", "dad", "mom", "son", "daughter", "child",
    "aunt", "uncle", "grandfather", "grandmother", "grandparent",
    "brother", "sister", "sibling", "wife", "husband", "spouse",
    "partner", "girlfriend", "boyfriend", "fiance", "fiancee",
}


def _intro_relationship_self_explanatory(relationship: Optional[str]) -> bool:
    rel = (relationship or "").strip().lower().replace(" ", "_")
    return rel in _INTRO_SELF_EXPLANATORY_RELATIONSHIPS


def _intro_relationship_question_instruction(
    relationship: Optional[str],
    introducer_first: str,
    introduced_first: str,
) -> str:
    rel = (relationship or "").strip().lower().replace(" ", "_")
    if rel in {"father", "mother", "parent", "dad", "mom"}:
        return (
            f"Then ask {introduced_first} exactly one easy, playful question "
            f"about {introducer_first}'s origin story or what {introducer_first} "
            f"was like before the current firmware."
        )
    if rel in {"partner", "girlfriend", "boyfriend", "fiance", "fiancee", "wife", "husband", "spouse"}:
        return (
            f"Then ask {introduced_first} exactly one easy, playful question "
            f"about what it is like being in {introducer_first}'s orbit."
        )
    if rel in {"brother", "sister", "sibling"}:
        return (
            f"Then ask {introduced_first} exactly one easy, playful question "
            f"about shared family lore or what {introducer_first} was like growing up."
        )
    if rel in {"aunt", "uncle", "grandfather", "grandmother", "grandparent"}:
        return (
            f"Then ask {introduced_first} exactly one easy question about family "
            f"lore or what {introducer_first} was like from their angle."
        )
    return (
        f"Then ask {introduced_first} exactly one easy question that fits their "
        f"relationship to {introducer_first}."
    )


def _intro_inverse_relationship(relationship: Optional[str]) -> str:
    rel = (relationship or "acquaintance").strip().lower()
    if rel in _INTRO_SYMMETRIC_RELATIONSHIPS:
        return rel
    return _INTRO_INVERSE_RELATIONSHIP.get(rel, "acquaintance")


def _store_introduction_memories(
    introducer_id: int,
    introducer_name: str,
    introduced_id: int,
    introduced_name: str,
    relationship: Optional[str],
) -> None:
    if int(introducer_id) == int(introduced_id):
        _log.warning(
            "[introduction] refusing self-relationship for person_id=%s relationship=%r",
            introducer_id,
            relationship,
        )
        return
    rel = (relationship or "acquaintance").strip().lower()
    inverse = _intro_inverse_relationship(rel)
    try:
        from memory import social as social_memory
        social_memory.save_relationship(
            from_person_id=introducer_id,
            to_person_id=introduced_id,
            relationship=rel,
            described_by=introducer_id,
        )
    except Exception as exc:
        _log.warning("introduction relationship save failed: %s", exc)

    try:
        facts_memory.add_fact(
            introducer_id,
            "relationship",
            f"relationship_to_{introduced_id}",
            f"{introduced_name} is {introducer_name}'s {rel}",
            "explicit_introduction",
            confidence=0.95,
        )
        facts_memory.add_fact(
            introduced_id,
            "relationship",
            f"relationship_to_{introducer_id}",
            f"{introducer_name} is {introduced_name}'s {inverse}",
            "explicit_introduction",
            confidence=0.90,
        )
    except Exception as exc:
        _log.debug("introduction fact save failed: %s", exc)


def _enroll_introduced_person(
    name: str,
    introducer_id: int,
    introducer_name: str,
    relationship: Optional[str],
    *,
    enroll_visible_face: bool = True,
) -> Optional[int]:
    try:
        new_id, created = people_memory.find_or_create_person(name)
    except Exception as exc:
        _log.warning("introduction find_or_create_person failed: %s", exc)
        return None
    if new_id is None:
        return None

    first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
    if created and first_inc > 0:
        people_memory.update_familiarity(new_id, first_inc)

    face_enrolled = False
    if enroll_visible_face:
        try:
            from vision import camera as camera_mod
            from vision import face as face_mod
            frame = camera_mod.capture_still()
            if frame is not None:
                face_enrolled = face_mod.enroll_unknown_face(new_id, frame)
                if face_enrolled:
                    threading.Thread(
                        target=face_mod.update_appearance,
                        args=(new_id, frame.copy()),
                        daemon=True,
                        name=f"appearance-enroll-{new_id}",
                    ).start()
        except Exception as exc:
            _log.warning("introduction face enrollment failed: %s", exc)

    if face_enrolled:
        _bind_world_state_identity(new_id, name)
    _store_introduction_memories(
        introducer_id,
        introducer_name,
        new_id,
        name,
        relationship,
    )
    _episodic_person_enrolled(new_id, name, created=created)
    return new_id


def _store_pet_introduction(
    owner_id: int,
    owner_name: str,
    pet_name: Optional[str],
    relationship: Optional[str],
) -> None:
    species = relationship or "pet"
    value = f"{pet_name} ({species})" if pet_name else species
    key = f"pet_{(pet_name or species).lower().replace(' ', '_')}"
    facts_memory.add_fact(
        owner_id,
        "pet",
        key,
        value,
        "explicit_introduction",
        confidence=0.95,
    )
    _log.info(
        "[introduction] stored pet intro for %s: %s",
        owner_name,
        value,
    )


def _intro_ack_and_followup(
    introducer_id: int,
    introducer_name: str,
    introduced_id: Optional[int],
    introduced_name: str,
    relationship: Optional[str],
    *,
    subject_kind: str = "person",
    visible_newcomer: bool = True,
) -> str:
    global _pending_intro_followup, _pending_intro_voice_capture

    introducer_first = _first_name_or(introducer_name, "there")
    introduced_first = _first_name_or(introduced_name, "there")
    rel_clause = f"{relationship}" if relationship else "guest"
    self_explanatory_relationship = _intro_relationship_self_explanatory(relationship)
    followup_kind = (
        "relationship_color" if self_explanatory_relationship else "connection_story"
    )
    special_context = (
        person_specials.special_prompt_context(introduced_name)
        if visible_newcomer
        else None
    )
    if person_specials.is_jt_volleyball_celebrity(introduced_name):
        special_quip_instruction = (
            f"acknowledge {introduced_first} by name as a major volleyball "
            "celebrity with one affectionate joke about getting old, bones, "
            "or muscles."
        )
    elif person_specials.is_rex_creator(introduced_name):
        special_quip_instruction = (
            "acknowledge Bret Benziger by name as Rex's creator and maker. Be "
            "warm, reverent, loyal, and still dryly Rex about being Bret's "
            "high-maintenance droid creation."
        )
    elif person_specials.is_galactic_hair_stylist(introduced_name):
        special_quip_instruction = (
            f"acknowledge {introduced_first} by name as one of the greatest "
            "hair stylists in the galactic quadrant with one affectionate "
            "joke about legendary styling skills, blowouts, bangs, or frizz."
        )
    else:
        special_quip_instruction = f"acknowledge {introduced_first} by name."

    if special_context and self_explanatory_relationship:
        question_instruction = _intro_relationship_question_instruction(
            relationship,
            introducer_first,
            introduced_first,
        )
        prompt = (
            f"{introducer_first} just explicitly introduced {introduced_first} "
            f"as {introducer_first}'s {rel_clause}. {special_context} "
            f"In one or two short in-character Rex sentences, "
            f"{special_quip_instruction} {question_instruction} Address "
            f"{introduced_first}, not {introducer_first}. Do NOT ask how they "
            "know each other."
        )
    elif special_context:
        prompt = (
            f"{introducer_first} just explicitly introduced {introduced_first} "
            f"as their {rel_clause}. {special_context} In ONE short "
            f"in-character Rex line, {special_quip_instruction} Then ask how "
            f"{introduced_first} and {introducer_first} know each other. Address {introduced_first}, "
            f"not just {introducer_first}."
        )
    elif visible_newcomer and self_explanatory_relationship:
        question_instruction = _intro_relationship_question_instruction(
            relationship,
            introducer_first,
            introduced_first,
        )
        prompt = (
            f"{introducer_first} just explicitly introduced {introduced_first} "
            f"as {introducer_first}'s {rel_clause}. This already explains their "
            f"relationship. In one or two short in-character Rex sentences, acknowledge "
            f"{introduced_first} by name with a funny but friendly quip about "
            f"their connection to {introducer_first}. {question_instruction} "
            f"Address {introduced_first}, not {introducer_first}. Do NOT ask how "
            f"they know each other. Do NOT imply {introduced_first} is related to Rex."
        )
    elif visible_newcomer:
        prompt = (
            f"{introducer_first} just explicitly introduced {introduced_first} "
            f"as their {rel_clause}. This is a social introduction with at least "
            f"three participants. In ONE short in-character Rex line, acknowledge "
            f"{introduced_first} by name with a funny but friendly quip, then ask "
            f"one natural question about how {introduced_first} and {introducer_first} "
            f"know each other or what story Rex is missing. Address {introduced_first}, "
            f"not just {introducer_first}. Keep it warm, conversational, and not mean."
        )
    else:
        prompt = (
            f"{introducer_first} just explicitly introduced {introduced_first} "
            f"as their {rel_clause}, but {introduced_first} is not visible on camera "
            f"right now. In ONE short in-character Rex line, START with exactly "
            f"'Nice to meet you, {introduced_first}.' Then invite {introduced_first} "
            f"to say a quick hello so you can learn their voice. Do not address "
            f"{introducer_first} as if they are the newcomer, and do not ask how "
            f"they know each other yet."
        )
    try:
        text = llm.get_response(prompt) or ""
    except Exception as exc:
        _log.debug("intro ack generation failed: %s", exc)
        text = ""
    if not text:
        special_ack = person_specials.special_intro_ack(introduced_name)
        if special_ack and self_explanatory_relationship:
            text = (
                f"{special_ack} What should I know about {introducer_first} "
                "from your side of the evidence locker?"
            )
        elif special_ack:
            text = (
                f"{special_ack} How did you end up in {introducer_first}'s orbit?"
            )
        elif self_explanatory_relationship:
            text = (
                f"{introduced_first}, welcome. So you're {introducer_first}'s "
                f"{rel_clause}; suddenly several mysteries have useful context. "
                f"What should I know about {introducer_first} from your side of the evidence locker?"
            )
        else:
            text = f"{introduced_first}, welcome to the frequency. How did you end up in {introducer_first}'s orbit?"
    if not visible_newcomer and not text.lower().startswith("nice to meet you"):
        text = (
            f"Nice to meet you, {introduced_first}. Give me a quick hello so "
            f"I can file your voice somewhere more useful than 'mystery guest.'"
        )

    if introduced_id is not None and subject_kind == "person":
        pending = {
            "introducer_id": introducer_id,
            "introducer_name": introducer_name,
            "introduced_id": introduced_id,
            "introduced_name": introduced_name,
            "relationship": relationship,
            "followup_kind": followup_kind,
            "asked_at": time.monotonic(),
        }
        if not visible_newcomer:
            _pending_intro_voice_capture = dict(pending)
        else:
            _pending_intro_followup = pending
        try:
            consciousness.note_person_greeted_this_session(introduced_id)
        except Exception:
            pass
    return text


def _intro_voice_capture_fresh(ctx: Optional[dict]) -> bool:
    if not ctx:
        return False
    ttl = float(getattr(config, "INTRO_VOICE_CAPTURE_WINDOW_SECS", 45.0))
    return (time.monotonic() - float(ctx.get("asked_at") or 0.0)) <= ttl


def _intro_voice_text_sounds_like_newcomer(text: str, name: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False
    if re.search(r"\b(no|nope|not now|not here|later|wait|hold on|can't|cannot)\b", cleaned):
        return False
    first = _first_name_or(name).lower()
    if first and re.search(rf"\b(this is|that is|that's)\s+{re.escape(first)}\b", cleaned):
        return False
    if re.search(r"\b(he|she|they)\s+(is|isn't|was|wasn't|can't|cannot|will|won't)\b", cleaned):
        return False
    if re.search(r"\b(hi|hello|hey|yo|nice to meet|what'?s up|i'?m|i am|my name is)\b", cleaned):
        return True
    words = re.findall(r"[a-z']+", cleaned)
    return 0 < len(words) <= 8


def _bind_intro_visible_face_if_present(person_id: int, name: str) -> None:
    if not _has_unknown_visible_person():
        return
    try:
        from vision import camera as _cam_mod
        from vision import face as _face_mod
        frame = _cam_mod.capture_still()
        if frame is None:
            return
        if _face_mod.enroll_unknown_face(person_id, frame):
            threading.Thread(
                target=_face_mod.update_appearance,
                args=(person_id, frame.copy()),
                daemon=True,
                name=f"appearance-enroll-{person_id}",
            ).start()
            _bind_world_state_identity(person_id, name)
    except Exception as exc:
        _log.warning("introduction visible face bind failed: %s", exc)


def _handle_intro_voice_capture(
    text: str,
    audio_array: np.ndarray,
    person_id: Optional[int],
    raw_best_id: Optional[int],
    speaker_score: float,
) -> Optional[str]:
    global _pending_intro_voice_capture, _pending_intro_followup

    ctx = _pending_intro_voice_capture
    if ctx is None:
        return None
    if not _intro_voice_capture_fresh(ctx):
        _log.info("[introduction] voice capture window expired for %s", ctx.get("introduced_name"))
        _pending_intro_voice_capture = None
        return None

    introduced_id = int(ctx["introduced_id"])
    introduced_name = ctx.get("introduced_name") or "the newcomer"
    introducer_id = ctx.get("introducer_id")
    introducer_name = ctx.get("introducer_name") or "the introducer"
    relationship = ctx.get("relationship")
    self_explanatory_relationship = _intro_relationship_self_explanatory(relationship)
    followup_kind = (
        "relationship_color" if self_explanatory_relationship else "connection_story"
    )

    if person_id == introduced_id:
        _pending_intro_voice_capture = None
        followup = dict(ctx)
        followup["followup_kind"] = followup_kind
        followup["asked_at"] = time.monotonic()
        _pending_intro_followup = followup
        return None

    hard_threshold = float(config.SPEAKER_ID_SIMILARITY_THRESHOLD)
    looks_like_newcomer = _intro_voice_text_sounds_like_newcomer(text, introduced_name)
    accepted_unknown = person_id is None
    weak_introducer_match = (
        person_id == introducer_id
        and raw_best_id == introducer_id
        and speaker_score < hard_threshold
        and looks_like_newcomer
    )
    if not accepted_unknown and not weak_introducer_match:
        return None

    if not looks_like_newcomer and not accepted_unknown:
        return None

    ok = _safe_enroll_voice(
        introduced_id,
        audio_array,
        transcript_text=text,
        source="intro_voice_capture",
        confirmed=False,
    )
    if not ok:
        ctx["asked_at"] = time.monotonic()
        first = _first_name_or(introduced_name)
        return (
            f"{first}, my voice scanner got a mouthful of static. Give me one "
            f"more sentence so I can stop calling you theoretical."
        )

    _pending_intro_voice_capture = None
    followup = dict(ctx)
    followup["followup_kind"] = followup_kind
    followup["asked_at"] = time.monotonic()
    _pending_intro_followup = followup
    _bind_intro_visible_face_if_present(introduced_id, introduced_name)
    _log.info(
        "[introduction] enrolled voice for introduced person %r (person_id=%s)",
        introduced_name,
        introduced_id,
    )
    try:
        _session_person_ids.add(introduced_id)
        consciousness.mark_engagement(introduced_id)
        consciousness.note_person_spoke(introduced_id)
    except Exception:
        pass
    try:
        conv_memory.add_to_transcript(introduced_name, text)
        conv_log.log_heard(introduced_name, text)
        print(
            f"[VOICE] Known voice detected: {introduced_name} (person_id={introduced_id})",
            flush=True,
        )
        print(f"[HEARD] {introduced_name}: {text}", flush=True)
        _log.info(
            "[introduction] voice capture speech segment — speaker=%r person_id=%s text=%r",
            introduced_name,
            introduced_id,
            text,
        )
    except Exception as exc:
        _log.debug("intro voice capture transcript log failed: %s", exc)
    try:
        topic_thread.note_user_turn(text, introduced_id)
        user_energy.note_user_turn(text, introduced_id)
    except Exception as exc:
        _log.debug("intro voice capture turn tracking failed: %s", exc)

    intro_first = _first_name_or(introducer_name, "there")
    introduced_first = _first_name_or(introduced_name, "there")
    try:
        if self_explanatory_relationship:
            question_instruction = _intro_relationship_question_instruction(
                relationship,
                intro_first,
                introduced_first,
            )
            return llm.get_response(
                f"{introduced_first} just responded after {intro_first} introduced "
                f"them as {intro_first}'s {relationship}. You successfully stored "
                f"{introduced_first}'s voice print. In one or two short "
                f"in-character Rex sentences, acknowledge {introduced_first} by "
                f"name with a friendly quip. {question_instruction} Do NOT ask "
                f"how they know each other. Do NOT imply they are related to Rex."
            )
        return llm.get_response(
            f"{introduced_first} just responded after {intro_first} introduced them, "
            f"and you successfully stored {introduced_first}'s voice print. In ONE "
            f"short in-character Rex line, acknowledge {introduced_first} by name "
            f"with a light quip, then ask how {introduced_first} and {intro_first} "
            f"know each other."
        )
    except Exception as exc:
        _log.debug("intro voice capture ack generation failed: %s", exc)
    if self_explanatory_relationship:
        return (
            f"Got it, {introduced_first}. Voice filed. What should I know about "
            f"{intro_first} from your side of the evidence locker?"
        )
    return f"Got it, {introduced_first}. Voice filed. So how did you and {intro_first} get tangled up?"


def _handle_intro_followup_answer(text: str) -> Optional[str]:
    global _pending_intro_followup

    ctx = _pending_intro_followup
    if not introductions.followup_fresh(ctx):
        _pending_intro_followup = None
        return None
    if not introductions.should_capture_followup(text):
        return None

    _pending_intro_followup = None
    intro_id = int(ctx["introducer_id"])
    introduced_id = int(ctx["introduced_id"])
    if intro_id == introduced_id:
        _log.warning(
            "[introduction] refusing intro followup for self relationship person_id=%s text=%r",
            intro_id,
            text,
        )
        return None
    intro_name = ctx.get("introducer_name") or "the introducer"
    introduced_name = ctx.get("introduced_name") or "the newcomer"
    relationship = ctx.get("relationship")
    followup_kind = ctx.get("followup_kind") or "connection_story"
    detail = text.strip()

    try:
        if followup_kind == "relationship_color":
            rel_phrase = f" ({relationship})" if relationship else ""
            fact_key_a = f"intro_note_{introduced_id}"
            fact_key_b = f"intro_note_{intro_id}"
            source = "relationship_color_followup"
            value = f"{intro_name} introduced {introduced_name}{rel_phrase}: {detail}"
        else:
            fact_key_a = f"connection_story_{introduced_id}"
            fact_key_b = f"connection_story_{intro_id}"
            source = "introduction_followup"
            value = f"{intro_name} and {introduced_name}: {detail}"
        facts_memory.add_fact(
            intro_id,
            "relationship",
            fact_key_a,
            value,
            source,
            confidence=0.90,
        )
        facts_memory.add_fact(
            introduced_id,
            "relationship",
            fact_key_b,
            value,
            source,
            confidence=0.90,
        )
        _log.info(
            "[introduction] stored intro followup (%s) for person_id=%s and person_id=%s: %r",
            followup_kind,
            intro_id,
            introduced_id,
            detail,
        )
    except Exception as exc:
        _log.debug("intro followup fact save failed: %s", exc)

    try:
        if followup_kind == "relationship_color":
            return llm.get_response(
                f"You asked a light introduction question after learning "
                f"{introduced_name} is {intro_name}'s {relationship or 'person'}. "
                f"They answered: '{detail}'. In ONE short Rex line, acknowledge "
                f"the detail with a light quip. No new question."
            )
        return llm.get_response(
            f"You just learned how {intro_name} and {introduced_name} know each "
            f"other: '{detail}'. In ONE short Rex line, acknowledge the story "
            f"with a light quip. No new question."
        )
    except Exception as exc:
        _log.debug("intro followup ack generation failed: %s", exc)
    return "Noted. Another organic relationship filed under suspicious but charming."


def _resolve_existing_visible_introduced_person(
    name: Optional[str],
    introducer_id: Optional[int],
) -> Optional[tuple[int, str]]:
    """Return a known visible/recent person matching an introduced name."""
    candidate_name = _normalize_name(name or "")
    if not candidate_name:
        return None
    try:
        existing = people_memory.find_person_by_name(candidate_name)
    except Exception as exc:
        _log.debug("introduction existing-person lookup failed: %s", exc)
        return None
    if not existing:
        return None
    try:
        existing_id = int(existing["id"])
    except Exception:
        return None
    if introducer_id is not None and existing_id == int(introducer_id):
        return None
    if not _known_person_visible_recently(existing_id):
        return None
    existing_name = _normalize_name(str(existing.get("name") or "")) or candidate_name
    return existing_id, existing_name


def _handle_introduction_parse(
    parsed: introductions.IntroductionParse,
    *,
    introducer_id: int,
    introducer_name: str,
    visible_newcomer: bool = True,
) -> Optional[str]:
    global _pending_introduction, _pending_common_first_name_introduction

    if parsed.subject_kind == "pet":
        _store_pet_introduction(
            introducer_id,
            introducer_name,
            parsed.name,
            parsed.relationship,
        )
        pet_name = parsed.name or f"your {parsed.relationship or 'pet'}"
        try:
            return llm.get_response(
                f"{introducer_name} just introduced {pet_name}, their "
                f"{parsed.relationship or 'pet'}. In ONE short friendly Rex "
                f"line, acknowledge the pet by name/species with a tiny quip. "
                f"No follow-up question."
            )
        except Exception:
            return f"{pet_name}, welcome. Finally, a civilized member of the party."

    if not parsed.name:
        _pending_introduction = {
            "introducer_id": introducer_id,
            "introducer_name": introducer_name,
            "relationship": parsed.relationship,
            "visible_newcomer": visible_newcomer,
            "created_at": time.monotonic(),
            "asked_at": time.monotonic(),
        }
        rel_hint = f" your {parsed.relationship}" if parsed.relationship else ""
        try:
            if parsed.relationship:
                relationship_label = str(parsed.relationship).replace("_", " ")
                intro_clause = (
                    f"this visible newcomer is {introducer_name}'s {relationship_label}"
                    if visible_newcomer
                    else f"they want you to meet their {relationship_label}"
                )
                return llm.get_response(
                    f"{introducer_name} said {intro_clause}, "
                    f"but did not give the person's name yet. In ONE short "
                    f"in-character Rex line, ask {introducer_name} ONLY for the "
                    f"newcomer's name. Do NOT ask how they are related. Do NOT "
                    f"imply the newcomer is related to Rex."
                )
            return llm.get_response(
                f"{introducer_name} said they want to introduce you to{rel_hint} "
                f"someone, and an unknown face is visible, but you do not have "
                f"the person's name yet. In ONE short in-character Rex line, "
                f"ask {introducer_name} for the newcomer's name and relationship. "
                f"Make it feel like a real introduction, not a form."
            )
        except Exception:
            if parsed.relationship:
                return f"Got the {parsed.relationship} part. What name am I filing for them?"
            return "Fine, I see the mystery organic. What name and relationship am I filing under?"

    existing_intro = _resolve_existing_visible_introduced_person(
        parsed.name,
        introducer_id,
    )
    if existing_intro is not None:
        existing_id, existing_name = existing_intro
        _pending_introduction = None
        _store_introduction_memories(
            introducer_id,
            introducer_name,
            existing_id,
            existing_name,
            parsed.relationship,
        )
        _log.info(
            "[introduction] matched existing visible/recent person %s "
            "(person_id=%s) as %s of %s",
            existing_name,
            existing_id,
            parsed.relationship or "acquaintance",
            introducer_name,
        )
        return _intro_ack_and_followup(
            introducer_id,
            introducer_name,
            existing_id,
            existing_name,
            parsed.relationship,
            subject_kind=parsed.subject_kind,
            visible_newcomer=True,
        )

    new_id = _enroll_introduced_person(
        parsed.name,
        introducer_id,
        introducer_name,
        parsed.relationship,
        enroll_visible_face=visible_newcomer,
    )
    if new_id is None:
        return None
    _pending_introduction = None
    _mark_single_name_for_later_last_name(new_id, parsed.name)
    _log.info(
        "[introduction] %s introduced %s as %s (person_id=%s)",
        introducer_name,
        parsed.name,
        parsed.relationship or "acquaintance",
        new_id,
    )
    return _intro_ack_and_followup(
        introducer_id,
        introducer_name,
        new_id,
        parsed.name,
        parsed.relationship,
        subject_kind=parsed.subject_kind,
        visible_newcomer=visible_newcomer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent transcription + speaker identification
# ─────────────────────────────────────────────────────────────────────────────

def _process_audio(
    audio_array: np.ndarray,
) -> tuple[str, Optional[int], Optional[str], float]:
    """
    Run transcription and speaker ID simultaneously in two threads.

    Returns (transcribed_text, raw_best_id, raw_best_name, raw_best_score).
    The speaker values are the RAW top voice-ID candidate — NOT threshold-
    filtered. Callers apply hard/soft thresholds themselves so they can also
    consult session context (recent engagement) for identity continuity.
    """
    text_box: list[str] = [""]
    speaker_box: list = [None, None, 0.0]  # [person_id, name, score]

    def _transcribe() -> None:
        text_box[0] = transcription.transcribe(audio_array)

    def _identify() -> None:
        pid, name, score = speaker_id.identify_speaker_raw(audio_array)
        speaker_box[0] = pid
        speaker_box[1] = name
        speaker_box[2] = score

    t1 = threading.Thread(target=_transcribe, daemon=True, name="transcription")
    t2 = threading.Thread(target=_identify, daemon=True, name="speaker-id")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return text_box[0] or "", speaker_box[0], speaker_box[1], speaker_box[2]


# ─────────────────────────────────────────────────────────────────────────────
# Turn completion repair
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_prompt_incomplete_turn() -> bool:
    """Ask a tiny repair question when a held fragment times out."""
    global _session_exchange_count
    if _game_suppresses_conversation():
        turn_completion.clear_stale_prompted()
        return False
    if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
        return False

    pending = turn_completion.mark_prompt_due()
    if pending is None:
        turn_completion.clear_stale_prompted()
        return False

    prompt = pending.prompt
    if not prompt:
        return False

    speaker_label = "user"
    heard_name: Optional[str] = None
    try:
        if (
            pending.raw_best_name
            and pending.raw_best_score >= float(getattr(config, "SPEAKER_ID_SOFT_THRESHOLD", 0.60))
        ):
            speaker_label = pending.raw_best_name
            heard_name = pending.raw_best_name
    except Exception:
        pass
    try:
        conv_memory.add_to_transcript(speaker_label, pending.text)
        conv_log.log_heard(heard_name, pending.text)
    except Exception as exc:
        _log.debug("turn completion partial transcript log failed: %s", exc)
    print(f"[HEARD] {speaker_label}: {pending.text}", flush=True)
    _log.info(
        "[turn_completion] prompting for completion — speaker=%r text=%r prompt=%r",
        speaker_label,
        pending.text,
        prompt,
    )

    completed = _speak_proactive(
        prompt,
        emotion="neutral",
        pre_beat_ms=100,
        post_beat_ms_override=150,
        label="incomplete_turn_prompt",
    )
    if completed:
        conv_memory.add_to_transcript("Rex", prompt)
        conv_log.log_rex(prompt)
        _session_exchange_count += 1
        try:
            wait_secs = float(
                getattr(config, "INCOMPLETE_TURN_PROMPT_REPLY_WINDOW_SECS", 10.0)
            )
        except Exception:
            wait_secs = 10.0
        _register_rex_utterance(prompt, wait_secs=wait_secs)
    return True


def _arm_visible_unknown_identity_followup(
    person_id: Optional[int],
    *,
    source: str,
) -> None:
    """
    When Rex has just asked about a visible mystery guest from the normal LLM
    path, open the same parsing window used by proactive relationship prompts.

    Without this, a natural bare reply like "Alex" after "who's this?" can fall
    through as ordinary chat instead of enrolling the visible newcomer.
    """
    unknown_slot = None
    try:
        for p in world_state.get("people") or []:
            if p.get("person_db_id") is None:
                unknown_slot = str(p.get("id") or "unknown_visible")
                break
    except Exception:
        unknown_slot = None
    if not unknown_slot:
        return

    if person_id is None:
        global _identity_prompt_until
        _identity_prompt_until = max(
            _identity_prompt_until,
            time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS,
        )
        _log.info(
            "[interaction] armed visible-unknown self-identity followup source=%s slot=%s",
            source,
            unknown_slot,
        )
        return

    engaged_name = None
    try:
        row = people_memory.get_person(person_id)
        engaged_name = row.get("name") if row else None
    except Exception:
        engaged_name = None
    try:
        consciousness.set_relationship_prompt_context({
            "engaged_person_id": person_id,
            "engaged_name": engaged_name,
            "slot_id": unknown_slot,
            "asked_at": time.monotonic(),
        })
        _log.info(
            "[interaction] armed visible-unknown relationship followup "
            "source=%s engaged_id=%s slot=%s",
            source,
            person_id,
            unknown_slot,
        )
    except Exception as exc:
        _log.debug("visible unknown relationship followup arm failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Speech accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _speech_preroll_secs() -> float:
    preroll = float(getattr(config, "SPEECH_PREROLL_SECS", 0.45) or 0.0)
    if _response_wait_active():
        preroll = max(
            preroll,
            float(getattr(config, "POST_QUESTION_SPEECH_PREROLL_SECS", preroll) or 0.0),
        )
    return max(0.0, preroll)


def _speech_capture_secs(speech_start_mono: float, finished_mono: Optional[float] = None) -> float:
    """
    Return how much rolling audio to send to Whisper for this speech segment.

    Question answers get extra pre-roll, but never from before the latest
    post-TTS handoff floor. Otherwise Rex's own question can be captured along
    with the human's answer.
    """
    finished = time.monotonic() if finished_mono is None else float(finished_mono)
    start_at = float(speech_start_mono) - _speech_preroll_secs()
    if _listen_capture_floor_at > 0.0:
        start_at = max(start_at, _listen_capture_floor_at)
    duration = max(0.0, finished - start_at)
    return min(duration, float(config.AUDIO_BUFFER_SECONDS))


def _accumulate_speech(
    speech_start_mono: float,
    *,
    allowed_states: tuple[State, ...] = (State.ACTIVE,),
) -> Optional[np.ndarray]:
    """
    Poll VAD until config.SILENCE_TIMEOUT_SECS of sustained silence, then
    return the full speech segment captured from the rolling buffer.

    speech_start_mono is the monotonic timestamp when speech was first detected
    in the main loop — used to calculate how much of the buffer to grab.
    """
    silence_timeout = config.SILENCE_TIMEOUT_SECS
    silence_elapsed = 0.0

    while not _stop_event.is_set():
        if state_module.get_state() not in allowed_states:
            return None

        chunk = stream.get_audio_chunk(_CHUNK_SECS)
        if len(chunk) == 0:
            _stop_event.wait(_CHUNK_SECS)
            continue

        is_speech = vad.is_speech(_chunk_for_vad(chunk))
        _situation_assessor.set_vad_active(is_speech)
        if is_speech:
            silence_elapsed = 0.0
        else:
            silence_elapsed += _CHUNK_SECS
            elapsed = time.monotonic() - speech_start_mono
            min_duration = getattr(config, "MIN_SPEECH_DURATION_SECS", 0.0)
            if silence_elapsed >= silence_timeout and elapsed >= min_duration:
                break

        _stop_event.wait(_CHUNK_SECS)

    if _stop_event.is_set():
        return None

    # Grab the full segment from the rolling buffer. Add pre-roll so soft starts
    # before the first VAD-positive chunk are not clipped, clamped to the latest
    # post-TTS handoff so Rex's own question is not transcribed as user speech.
    capture_secs = _speech_capture_secs(speech_start_mono)
    return stream.get_audio_chunk(capture_secs)


def _wake_from_sleep_if_transcribed(audio_segment: Optional[np.ndarray]) -> bool:
    """Wake from sleep when a transcribed sleep utterance is an explicit wake phrase."""
    global _last_speech_at
    if audio_segment is None or len(audio_segment) == 0:
        return False
    try:
        text = transcription.transcribe(audio_segment).strip()
    except Exception as exc:
        _log.debug("[sleep_mode] wake transcript failed: %s", exc)
        return False
    if not text:
        return False
    if not _is_sleep_wake_transcript(text):
        _log.info("[sleep_mode] ignored sleeping speech text=%r", text)
        return False
    _log.info("[sleep_mode] waking from transcribed phrase text=%r", text)
    _last_speech_at = time.monotonic()
    _wake_from_sleep()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# LLM streaming to TTS
# ─────────────────────────────────────────────────────────────────────────────

def _stream_llm_response(
    text: str,
    person_id: Optional[int],
    answered_question: Optional[dict] = None,
    turn_start: Optional[float] = None,
) -> str:
    """Collect the full LLM response, then speak it in a single TTS call.

    Collecting before speaking keeps AEC suppression as one continuous window
    per response. At max_tokens=150 the added latency is negligible.
    """
    # Fire the surprise classifier in parallel with the main LLM stream so
    # its result is (usually) ready by the time the response text is in hand.
    # If it returns is_surprising=True before we speak, we prepend a brief
    # silent beat — Rex's "...huh, didn't see that coming" pause.
    surprise_result: dict[str, bool] = {"value": False}

    def _classify() -> None:
        try:
            surprise_result["value"] = llm.classify_surprise(text)
        except Exception as exc:
            _log.debug("surprise classify thread error: %s", exc)

    surprise_thread = threading.Thread(
        target=_classify, daemon=True, name="surprise-classify"
    )
    surprise_thread.start()

    filler_stop = _start_latency_filler_timer()
    try:
        turn_plan = conversation_agenda.build_turn_plan(
            text,
            person_id,
            answered_question=answered_question,
        )
        agenda_directive = turn_plan.directive
        frame = social_frame.build_frame(
            text,
            person_id,
            answered_question=answered_question,
            agenda_directive=agenda_directive,
            turn_plan=turn_plan,
        )
        comedy_mode = comedy_modes.select_mode(
            text,
            person_id,
            frame=frame,
            agenda_directive=agenda_directive,
        )
        agenda_directive = "\n".join([
            agenda_directive,
            social_frame.build_directive(frame),
            comedy_modes.build_directive(comedy_mode),
        ])
        _log.info("[agenda] %s", agenda_directive.replace("\n", " | "))
        if (
            getattr(config, "LLM_STREAMING_TTS_ENABLED", True)
            and not bool(
                getattr(config, "NO_AUDIO_MODE", False)
                or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
            )
            and not _interrupted.is_set()
        ):
            return _stream_and_speak_sentences(
                text,
                person_id,
                frame,
                comedy_mode,
                agenda_directive,
                surprise_result,
                turn_start,
                filler_stop,
            )
        llm_started = time.monotonic()
        full_text = llm.get_response(
            text,
            person_id,
            agenda_directive=agenda_directive,
        )
        if turn_start is not None:
            _latency_log(turn_start, "llm_response", llm_started)
    finally:
        filler_stop.set()

    pre_beat_ms = 0
    delivery_emotion = "neutral"
    delivery_post_beat_ms = 0
    delivery_voice_settings: Optional[dict] = None
    if full_text and full_text.strip() and not _interrupted.is_set():
        try:
            governed = social_frame.govern_response(full_text, frame)
            full_text = governed.text
        except Exception as exc:
            _log.debug("social frame governor failed: %s", exc)
        try:
            full_text = comedy_modes.polish_response(
                full_text,
                comedy_mode,
                allow_roast=frame.allow_roast,
            )
        except Exception as exc:
            _log.debug("comedy polish failed: %s", exc)

        # Brief join — classifier usually finishes before or alongside the
        # main response since its prompt is tiny. Cap the wait so a slow
        # classifier never delays Rex perceptibly.
        surprise_thread.join(timeout=0.3)
        if surprise_result["value"]:
            try:
                _play_event_body_beat("emotion.surprise")
            except Exception as exc:
                _log.debug("[interaction] surprise body beat skipped: %s", exc)
            delivery_emotion = "surprised"
            beat_min = getattr(config, "SURPRISE_PAUSE_MS_MIN", 500)
            beat_max = getattr(config, "SURPRISE_PAUSE_MS_MAX", 1000)
            if beat_max >= beat_min > 0:
                pre_beat_ms = random.randint(beat_min, beat_max)
                _log.info("[interaction] surprise beat: %d ms", pre_beat_ms)

        # Empathy-mode delivery shaping: sympathetic posture + pacing for
        # listen/support/etc. without changing the cached TTS audio (cache key
        # is text+voice+model — emotion only drives LEDs and body bias).
        try:
            overrides = empathy.get_delivery_overrides(person_id)
        except Exception as exc:
            _log.debug("empathy.get_delivery_overrides error: %s", exc)
            overrides = None
        if overrides:
            if overrides.get("emotion") and not surprise_result["value"]:
                delivery_emotion = overrides["emotion"]
            extra_pre = int(overrides.get("pre_beat_ms") or 0)
            if extra_pre > 0:
                # Take the larger of surprise-beat vs empathy-mode beat — they
                # serve the same "let it land" purpose and shouldn't stack.
                pre_beat_ms = max(pre_beat_ms, extra_pre)
            delivery_post_beat_ms = int(overrides.get("post_beat_ms") or 0)
            delivery_voice_settings = overrides.get("voice_settings")
            _log.info(
                "[empathy] delivery shaping: mode=%s emotion=%s pre=%dms post=%dms voice=%s",
                overrides.get("mode"), delivery_emotion,
                pre_beat_ms, delivery_post_beat_ms,
                delivery_voice_settings if delivery_voice_settings else "default",
            )

        speak_started = time.monotonic()
        completed = _speak_blocking(
            full_text,
            emotion=delivery_emotion,
            pre_beat_ms=pre_beat_ms,
            post_beat_ms_override=delivery_post_beat_ms,
            voice_settings=delivery_voice_settings,
        )
        if turn_start is not None:
            _latency_log(turn_start, "tts_playback_complete", speak_started)
        if completed and frame.purpose == "identity" and "?" in full_text:
            _arm_visible_unknown_identity_followup(
                person_id,
                source="agenda_llm_identity",
            )
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# Streaming response → TTS (speak sentence-by-sentence as the LLM generates)
# ─────────────────────────────────────────────────────────────────────────────

# A sentence boundary: a run of . ! ? (plus any closing quotes/brackets)
# followed by whitespace. Trailing punctuation at the very end of the buffer is
# treated as still-in-progress, since more tokens may be coming.
_STREAM_SENTENCE_BOUNDARY_RE = re.compile(r'[.!?]+["\')\]’”]*\s')
_STREAM_TAIL_TERMINAL_RE = re.compile(r'[.!?…]["\')\]’”]*$')

# A tail ending in an ellipsis ("…" or "...") — the model trailing off.
_ELLIPSIS_TAIL_RE = re.compile(r'(?:…|\.{2,})["\')\]’”]*\s*$')

# Connector / article / auxiliary words that leave an UNFINISHED thought when an
# ellipsis trail-off ends on them ("…the excitement of…", "…I mean, I've…"). Such
# a tail passes the terminal-punctuation check (… is terminal) but the ellipsis is
# stripped before TTS, so what gets spoken is a bare dangling fragment that lands
# as a hard cut-off. Deliberately excludes words that can legitimately END a
# sentence ("this", "to", "is") so only genuinely-dangling trail-offs are dropped.
_DANGLING_TAIL_WORDS = frozenset({
    "a", "an", "the", "of", "with", "for", "into", "about", "than", "and", "or",
    "but", "my", "your", "his", "her", "their", "its", "our",
    "i'm", "i've", "you're", "we're", "they're", "it's", "he's", "she's",
})


def _tail_is_speakable(tail: str) -> bool:
    """A leftover stream tail is spoken only if it's a finished sentence. An
    unpunctuated remainder means the model was cut mid-sentence ("What's the
    deal", "Glad to") — speaking it would emit a truncated fragment, so drop it.
    A tail the model trailed off on mid-clause with an ELLIPSIS ("…the excitement
    of…") also reads as a hard cut-off once TTS strips the ellipsis, so drop it
    when it ends on a dangling connector word. A '.'/'!'/'?' sentence is a finished
    thought even if it ends on "this"/"to" — keep it."""
    t = (tail or "").strip()
    if not t or not _STREAM_TAIL_TERMINAL_RE.search(t):
        return False
    if _ELLIPSIS_TAIL_RE.search(t):
        words = re.findall(r"[a-z']+", t.lower())
        if words and words[-1] in _DANGLING_TAIL_WORDS:
            return False
    return True


def _split_stream_sentences(buffer: str, min_chars: int) -> tuple[list[str], str]:
    """Pull complete sentences out of a growing stream buffer.

    Returns (complete_sentences, remainder). A finished sentence shorter than
    min_chars is merged into the following one, which keeps tiny fragments,
    initials, abbreviations, and decimals (e.g. "Dr.", "3.") from splitting into
    choppy one-word bursts.
    """
    sentences: list[str] = []
    pending_start = 0
    floor = max(1, int(min_chars or 1))
    for match in _STREAM_SENTENCE_BOUNDARY_RE.finditer(buffer):
        candidate = buffer[pending_start:match.end()].strip()
        if len(candidate) >= floor:
            sentences.append(candidate)
            pending_start = match.end()
    return sentences, buffer[pending_start:]


def _complete_sentence_prefix(text: str) -> str:
    """Trim `text` to its last COMPLETE sentence, dropping a trailing mid-sentence
    fragment (the model got cut off by max_tokens). Uses min_chars=1 so a short but
    finished sentence ("Wow indeed!") is recognized rather than merged into the
    truncated tail. Returns "" when nothing complete remains. Used by the streaming
    safety-net fallback so it never re-emits a cut-off fragment ("Wow indeed! I")."""
    t = (text or "").strip()
    if not t:
        return ""
    sentences, remainder = _split_stream_sentences(t, 1)
    remainder = remainder.strip()
    if remainder and _tail_is_speakable(remainder):
        sentences.append(remainder)
    return " ".join(s for s in sentences if s).strip()


def _streaming_surprise_beat(surprise_result: dict) -> int:
    """Pre-beat (ms) before the first sentence when the utterance is surprising.

    Reads the in-flight surprise classifier result without blocking — if it has
    not resolved by the time the first sentence is ready, the beat is skipped so
    the first word is never delayed.
    """
    if not surprise_result.get("value"):
        return 0
    beat_min = getattr(config, "SURPRISE_PAUSE_MS_MIN", 500)
    beat_max = getattr(config, "SURPRISE_PAUSE_MS_MAX", 1000)
    if beat_max >= beat_min > 0:
        return random.randint(beat_min, beat_max)
    return 0


def _prepare_stream_sentence(sentence: str, frame, comedy_mode) -> str:
    """Govern + polish a single streamed sentence; "" means skip it."""
    governed = social_frame.govern_stream_sentence(sentence, frame)
    if not governed:
        return ""
    try:
        governed = comedy_modes.polish_stream_sentence(
            governed, comedy_mode, allow_roast=frame.allow_roast
        )
    except Exception as exc:
        _log.debug("[interaction] stream polish failed: %s", exc)
    governed = llm.clean_response_text(governed or "")
    return governed.strip() if governed else ""


def _prefetch_stream_audio(
    text: str, emotion: str, voice_settings: Optional[dict]
) -> None:
    """Warm the TTS cache for an upcoming sentence so playback doesn't gap."""
    def _run() -> None:
        try:
            from audio import tts
            tts.ensure_cached(text, voice_settings=voice_settings, emotion=emotion)
        except Exception as exc:
            _log.debug("[interaction] stream prefetch failed: %s", exc)

    threading.Thread(target=_run, daemon=True, name="tts-prefetch").start()


def _await_streamed_speech(
    done_events: list[threading.Event], priority: int
) -> bool:
    """Block until every queued sentence finishes; honor wake-word interruption.

    Mirrors _speak_blocking's interruption handling across multiple queued
    items: on interrupt, cut current playback and drop the remaining sentences.
    Returns True on normal completion, False if cut short.
    """
    for done in done_events:
        while not done.wait(timeout=0.05):
            if _interrupted.is_set():
                try:
                    import sounddevice as sd
                    echo_cancel.request_cancel()
                    sd.stop()
                except Exception:
                    pass
                speech_queue.clear_below_priority(priority + 1)
                done.wait(timeout=0.5)
                return False
    return True


def _stream_and_speak_sentences(
    user_text: str,
    person_id: Optional[int],
    frame,
    comedy_mode,
    agenda_directive: str,
    surprise_result: dict,
    turn_start: Optional[float],
    filler_stop: threading.Event,
) -> str:
    """Stream the LLM reply and speak it sentence-by-sentence.

    The first sentence is queued the instant it is generated (the latency win);
    later sentences queue behind it in order through the single speech queue, so
    Rex never overlaps himself. Returns the full spoken text.
    """
    trace = _current_character_loop_trace.get()
    priority = 1
    min_chars = int(getattr(config, "LLM_STREAMING_MIN_SENTENCE_CHARS", 12) or 1)
    prefetch_enabled = bool(getattr(config, "LLM_STREAMING_PREFETCH_ENABLED", True))

    # Person-based delivery shaping — independent of the reply text, so resolve
    # once and apply to every sentence (emotion / voice settings) or to the right
    # edge (pre-beat on the first line, post-beat after the last).
    delivery_emotion = "neutral"
    delivery_voice_settings: Optional[dict] = None
    empathy_pre_beat_ms = 0
    empathy_post_beat_ms = 0
    try:
        overrides = empathy.get_delivery_overrides(person_id)
    except Exception as exc:
        _log.debug("empathy.get_delivery_overrides error: %s", exc)
        overrides = None
    if overrides:
        if overrides.get("emotion"):
            delivery_emotion = overrides["emotion"]
        empathy_pre_beat_ms = int(overrides.get("pre_beat_ms") or 0)
        empathy_post_beat_ms = int(overrides.get("post_beat_ms") or 0)
        delivery_voice_settings = overrides.get("voice_settings")

    spoken: list[str] = []
    done_events: list[threading.Event] = []
    state = {"first": True, "spoke_question": False, "emotion": delivery_emotion}
    llm_started = time.monotonic()

    def _consume(raw_sentence: str) -> None:
        prepared = _prepare_stream_sentence(raw_sentence, frame, comedy_mode)
        if not prepared:
            return
        # One-question cap across the whole reply (govern keeps it per sentence).
        if social_frame.is_question_sentence(prepared):
            if state["spoke_question"]:
                return
            state["spoke_question"] = True

        pre_beat_ms = empathy_pre_beat_ms
        on_start = None
        if state["first"]:
            if surprise_result.get("value"):
                state["emotion"] = "surprised"
                surprise_beat = _streaming_surprise_beat(surprise_result)
                if surprise_beat > 0:
                    pre_beat_ms = max(pre_beat_ms, surprise_beat)
                    try:
                        _play_event_body_beat("emotion.surprise")
                    except Exception as exc:
                        _log.debug("[interaction] surprise beat skipped: %s", exc)
            _mark_first_response_queued(trace, text=prepared, priority=priority)
            if trace is not None:
                on_start = lambda: _mark_first_response_audio_started(trace)
            filler_stop.set()
            if turn_start is not None:
                _latency_log(turn_start, "llm_first_sentence", llm_started)
        elif prefetch_enabled:
            _prefetch_stream_audio(prepared, state["emotion"], delivery_voice_settings)

        done = speech_queue.enqueue(
            prepared,
            state["emotion"],
            priority=priority,
            pre_beat_ms=pre_beat_ms,
            post_beat_ms=0,
            voice_settings=delivery_voice_settings,
            on_start=on_start,
            log_text=False,  # the whole turn is logged once below
        )
        done_events.append(done)
        spoken.append(prepared)
        state["first"] = False

    buffer = ""
    raw_chunks: list[str] = []
    try:
        for chunk in llm.stream_response(
            user_text, person_id, agenda_directive=agenda_directive
        ):
            if _interrupted.is_set():
                break
            raw_chunks.append(chunk)
            buffer += chunk
            ready, buffer = _split_stream_sentences(buffer, min_chars)
            for sentence in ready:
                _consume(sentence)
    except Exception as exc:
        _log.error("[interaction] streaming LLM error: %s", exc)
    finally:
        filler_stop.set()

    if not _interrupted.is_set():
        tail = buffer.strip()
        if tail and _tail_is_speakable(tail):
            _consume(tail)
        elif tail:
            _log.info(
                "[interaction] dropped incomplete stream tail (mid-sentence cut): %r",
                tail,
            )

    # Safety net: if the model produced text but every sentence was governed away
    # (e.g. an all-questions reply under a no-questions frame), fall back to
    # whole-reply governance — which yields a frame fallback line — so Rex is
    # never left silent. Rare: general frames never drop sentences here.
    if not spoken and not _interrupted.is_set():
        # Trim a trailing mid-sentence fragment first — otherwise a max_tokens
        # truncation ("Wow indeed! I") gets re-emitted as a cut-off line. A short
        # finished sentence the splitter's min-chars merge skipped ("Wow indeed!",
        # 11<12 chars) is recovered here; nothing complete → stay silent.
        raw_full = _complete_sentence_prefix("".join(raw_chunks))
        if raw_full:
            fallback = ""
            try:
                fallback = social_frame.govern_response(raw_full, frame).text
                fallback = comedy_modes.polish_response(
                    fallback, comedy_mode, allow_roast=frame.allow_roast
                )
                fallback = llm.clean_response_text(fallback or "")
            except Exception as exc:
                _log.debug("[interaction] stream fallback failed: %s", exc)
            if fallback:
                filler_stop.set()
                _speak_blocking(
                    fallback,
                    emotion=delivery_emotion,
                    voice_settings=delivery_voice_settings,
                )
                return fallback

    full_text = " ".join(part for part in spoken if part).strip()
    if full_text:
        # Log the whole reply once (file + GUI), not one entry per sentence.
        try:
            conv_log.log_rex(full_text)
        except Exception as exc:
            _log.debug("[interaction] stream conversation log failed: %s", exc)
    if turn_start is not None:
        _latency_log(turn_start, "llm_response", llm_started)

    completed = _await_streamed_speech(done_events, priority)

    if full_text:
        if completed:
            post_beat_ms = empathy_post_beat_ms
            if not _assistant_asked_question(full_text):
                beat_min = getattr(config, "POST_PUNCHLINE_BEAT_MS_MIN", 0)
                beat_max = getattr(config, "POST_PUNCHLINE_BEAT_MS_MAX", 0)
                if beat_max > 0 and beat_max >= beat_min:
                    post_beat_ms = max(post_beat_ms, random.randint(beat_min, beat_max))
            if post_beat_ms > 0:
                time.sleep(post_beat_ms / 1000.0)
        _apply_post_tts_handoff(full_text, source="streaming")
        if (
            completed
            and getattr(frame, "purpose", None) == "identity"
            and "?" in full_text
        ):
            _arm_visible_unknown_identity_followup(
                person_id, source="agenda_llm_identity"
            )

    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# Command execution
# ─────────────────────────────────────────────────────────────────────────────

def _silent_command_response(label: str) -> str:
    return f"{_SILENT_COMMAND_PREFIX}{label}"


def _is_silent_command_response(text: Optional[str]) -> bool:
    return isinstance(text, str) and text.startswith(_SILENT_COMMAND_PREFIX)


def _plain_directed_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9'\s]", " ", (text or "").lower()).split())


def _directed_context_fresh(now: Optional[float] = None) -> bool:
    now = time.monotonic() if now is None else now
    last_at = float(_directed_look_context.get("last_at") or 0.0)
    ttl = float(getattr(config, "DIRECTED_LOOK_CONTEXT_WINDOW_SECS", 25.0))
    return last_at > 0.0 and (now - last_at) <= ttl


def _reset_directed_look_context() -> None:
    _directed_look_context.update({
        "started_at": 0.0,
        "last_at": 0.0,
        "last_direction": None,
        "bare_count": 0,
        "asked_clarify": False,
        "target_hint": "",
        "target_kind": None,
        "greeted_key": None,
    })


def _note_directed_look_context(
    *,
    direction: str,
    target_hint: str = "",
    target_kind: Optional[str] = None,
    found: bool = False,
) -> None:
    now = time.monotonic()
    if found:
        _reset_directed_look_context()
        return
    if not _directed_context_fresh(now):
        _reset_directed_look_context()
        _directed_look_context["started_at"] = now
    elif not _directed_look_context.get("started_at"):
        _directed_look_context["started_at"] = now
    _directed_look_context["last_at"] = now
    _directed_look_context["last_direction"] = direction
    if target_hint:
        _directed_look_context["target_hint"] = target_hint
    if target_kind:
        _directed_look_context["target_kind"] = target_kind


def _directed_target_kind(target_hint: str) -> str:
    clean = _plain_directed_text(target_hint)
    if not clean:
        return "scene"
    if clean in {"this", "that", "here", "there", "this thing", "that thing"}:
        return "scene"
    if clean in {"me", "me silly", "me obviously", "my face", "myself"}:
        return "person"
    if re.match(r"^(?:the|a|an)\s+", clean):
        return "object"
    if re.match(
        r"^(?:my|your)\s+(?:wife|husband|spouse|partner|girlfriend|boyfriend|"
        r"mother|father|mom|dad|sister|brother|sibling|friend|kid|child|son|daughter)\b",
        clean,
    ):
        return "person"
    if clean in {"person", "face", "someone", "somebody", "human", "people"}:
        return "person"
    if re.match(r"^(?:my|your)\s+", clean):
        return "object"
    maybe_name = _normalize_name(target_hint)
    if maybe_name is not None:
        try:
            if people_memory.find_person_by_name(maybe_name):
                return "person"
        except Exception as exc:
            _log.debug("directed target person lookup failed: %s", exc)
    if maybe_name is not None and _name_word_count(maybe_name) == 1:
        return "person"
    return "object"


def _target_name_from_hint(target_hint: str, person_name: Optional[str]) -> Optional[str]:
    clean = _plain_directed_text(target_hint)
    if clean in {"me", "me silly", "me obviously", "my face", "myself"}:
        return person_name
    return _normalize_name(target_hint)


def _directional_search_sequence(
    start_direction: str,
    *,
    max_attempts: int,
) -> list[str]:
    start = (start_direction or "current").strip().lower()
    if start == "other_way":
        start = "current"
    if start not in {"current", "left", "right", "up", "down", "center"}:
        start = "current"

    if start == "down":
        base = ["down", "left", "right", "center"]
    elif start == "up":
        base = ["up", "left", "right", "center"]
    elif start == "left":
        base = ["left", "right", "down", "up", "center"]
    elif start == "right":
        base = ["right", "left", "down", "up", "center"]
    elif start == "center":
        base = ["center", "left", "right", "down", "up"]
    else:
        base = ["left", "right", "down", "up", "center"]

    ordered: list[str] = []
    for direction in base:
        if direction not in ordered:
            ordered.append(direction)
    return ordered[:max(1, int(max_attempts))]


def _move_and_capture_gaze(
    direction: str,
    *,
    target_hint: str = "",
):
    from sequences import animations
    from vision import camera as camera_mod

    suspend_secs = (
        float(getattr(config, "DIRECTED_LOOK_SETTLE_SECS", 0.22))
        + 0.5
    )
    consciousness.suspend_face_tracking(suspend_secs)
    actual_direction = animations.directed_look_pose(direction, target=target_hint)
    frame = camera_mod.capture_current_gaze(settle_secs=0.06)
    try:
        consciousness.resume_face_tracking()
    except Exception:
        pass
    return actual_direction, frame


def _visible_known_face_candidate(target_name: Optional[str] = None) -> Optional[dict]:
    try:
        people = world_state.get("people") or []
    except Exception:
        return None
    candidates: list[dict] = []
    for person in people:
        pid = person.get("person_db_id")
        if pid is None:
            continue
        if person.get("face_visible") is False or person.get("face_missing"):
            continue
        name = str(
            person.get("face_id")
            or person.get("voice_id")
            or person.get("name")
            or ""
        ).strip()
        if target_name and name and not _same_person_name(name, target_name):
            continue
        candidates.append({
            "person_id": pid,
            "name": name,
            "known": True,
        })
    if not candidates:
        return None
    return candidates[0]


def _detect_faces_in_gaze(frame) -> list[dict]:
    if frame is None:
        return []
    try:
        from vision import face as face_mod
        detections = face_mod.detect_faces(frame) or []
    except Exception as exc:
        _log.debug("directed face scan failed: %s", exc)
        return []

    results: list[dict] = []
    for det in detections:
        box = det.get("bounding_box")
        area = 0
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                area = int(float(box[2]) * float(box[3]))
            except Exception:
                area = 0
        person_record = None
        try:
            from vision import face as face_mod
            person_record = face_mod.identify_face(det.get("encoding"))
        except Exception as exc:
            _log.debug("directed face identify failed: %s", exc)
        results.append({
            "box": box,
            "area": area,
            "person_id": (person_record or {}).get("id"),
            "name": (person_record or {}).get("name"),
            "known": person_record is not None,
        })
    results.sort(key=lambda item: item.get("area") or 0, reverse=True)
    return results


def _face_matches_target(face: dict, target_name: Optional[str]) -> bool:
    if not target_name:
        return True
    face_name = str(face.get("name") or "").strip()
    if not face_name:
        return True
    return _same_person_name(face_name, target_name)


def _directed_face_greeting(face: dict, fallback_name: Optional[str]) -> str:
    name = str(face.get("name") or fallback_name or "").strip()
    first = _first_name_or(name)
    if first:
        return f"Oh hi, {first}."
    return "Oh hi. There you are."


def _greet_directed_face_once(face: dict, fallback_name: Optional[str]) -> Optional[str]:
    key = face.get("person_id") or face.get("name") or fallback_name or "unknown_face"
    if key and _directed_look_context.get("greeted_key") == key:
        return _silent_command_response("directed_look.face_already_found")
    _directed_look_context["greeted_key"] = key
    response = _directed_face_greeting(face, fallback_name)
    _speak_blocking(response, emotion="happy", pre_beat_ms=60, post_beat_ms_override=150)
    _reset_directed_look_context()
    return response


def _scan_for_person(
    *,
    target_name: Optional[str],
    start_direction: str,
    person_name: Optional[str],
) -> str:
    max_attempts = int(getattr(config, "DIRECTED_LOOK_FACE_SEARCH_MAX_ATTEMPTS", 5))
    for direction in _directional_search_sequence(start_direction, max_attempts=max_attempts):
        actual_direction, frame = _move_and_capture_gaze(direction, target_hint=target_name or "person")
        for face in _detect_faces_in_gaze(frame):
            if _face_matches_target(face, target_name):
                _note_directed_look_context(
                    direction=actual_direction,
                    target_hint=target_name or "person",
                    target_kind="person",
                    found=True,
                )
                return _greet_directed_face_once(face, target_name or person_name) or ""
        _log.info(
            "[directed_look] person search miss direction=%s target=%r",
            actual_direction,
            target_name,
        )
    response = (
        f"I am looking for {target_name}, but no face lock yet. "
        "Try another direction before my dignity files a complaint."
        if target_name
        else "I am looking, but no face lock yet. Directional hints accepted, regrettably."
    )
    _speak_blocking(response, emotion="curious")
    return response


def _search_for_object(
    *,
    target_hint: str,
    start_direction: str,
    raw_text: str,
    person_id: Optional[int],
) -> str:
    from vision import scene as vision_scene

    max_attempts = int(getattr(config, "DIRECTED_LOOK_OBJECT_SEARCH_MAX_ATTEMPTS", 5))
    final_analysis: dict = {}
    final_direction = start_direction
    for direction in _directional_search_sequence(start_direction, max_attempts=max_attempts):
        actual_direction, frame = _move_and_capture_gaze(direction, target_hint=target_hint)
        final_direction = actual_direction
        if frame is None:
            continue
        analysis = vision_scene.analyze_directed_attention(
            frame,
            direction=actual_direction,
            utterance=raw_text,
            target_hint=target_hint,
        )
        final_analysis = analysis or final_analysis
        if _analysis_found_target(analysis, target_hint):
            prompt = (
                f"The person asked you to look for {target_hint!r}. "
                f"You found it while looking {_directed_look_label(actual_direction)}. "
                f"Vision analysis:\n{json.dumps(analysis, ensure_ascii=False)}\n\n"
                "Reply as Rex in one concise snarky observation based only on the visible details. "
                "Max 28 words. Do not mention JSON, APIs, cameras, screenshots, or image analysis."
            )
            response = llm.get_response(prompt, person_id) or _fallback_directed_look_response(analysis)
            _speak_blocking(response)
            _reset_directed_look_context()
            return llm.clean_response_text(response)
        _log.info(
            "[directed_look] object search miss target=%r direction=%s",
            target_hint,
            actual_direction,
        )

    if final_analysis:
        _log.info(
            "[directed_look] object search exhausted target=%r last_direction=%s analysis=%r",
            target_hint,
            final_direction,
            final_analysis.get("target_summary"),
        )
    response = _not_found_visual_response(target_hint)
    _speak_blocking(response)
    return response


def _analyze_directed_view_once(
    *,
    frame,
    direction: str,
    target_hint: str,
    raw_text: str,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    if frame is None:
        response = (
            "I looked, but my photoreceptors came back empty. "
            "Very dramatic. Terrible evidence."
        )
        _speak_blocking(response)
        return response
    from vision import scene as vision_scene

    analysis = vision_scene.analyze_directed_attention(
        frame,
        direction=direction,
        utterance=raw_text,
        target_hint=target_hint,
    )
    if not analysis:
        response = "I looked, but the visual intel is thin. Atmospheric, yes. Useful, no."
        _speak_blocking(response)
        return response

    speaker = person_name or "the person"
    prompt = (
        f"{speaker} asked you to physically look {_directed_look_label(direction)}. "
        f"The original request was: {raw_text!r}. "
        "You moved your head/visor, took a fresh look, and got this vision analysis:\n"
        f"{json.dumps(analysis, ensure_ascii=False)}\n\n"
        "Reply as Rex with one concise roast-style observation or opinion based ONLY "
        "on the analysis. Max 35 words. If the target is a person, child, pet, or "
        "possible introduction, acknowledge them warmly with a harmless quip and, "
        "only if useful, ask who they are. Do not mention JSON, APIs, cameras, "
        "screenshots, or image analysis."
    )
    response = llm.get_response(prompt, person_id) or _fallback_directed_look_response(analysis)
    _speak_blocking(response)
    return llm.clean_response_text(response)


def _clarify_directed_search() -> str:
    _directed_look_context["asked_clarify"] = True
    response = "What am I looking for?"
    _speak_blocking(response, emotion="curious")
    return response


def _pending_directed_search_reply(
    text: str,
    *,
    person_id: Optional[int],
    person_name: Optional[str],
) -> Optional[str]:
    if not _directed_context_fresh():
        return None
    if not _directed_look_context.get("asked_clarify"):
        return None

    clean = _plain_directed_text(text)
    if clean in {"me", "me silly", "me obviously", "me dummy", "me droid"}:
        try:
            recent = consciousness.get_recent_engagement() or {}
        except Exception:
            recent = {}
        target_name = person_name or recent.get("name")
        return _scan_for_person(
            target_name=target_name,
            start_direction=str(_directed_look_context.get("last_direction") or "current"),
            person_name=person_name,
        )

    look_match = command_parser.parse(text)
    if look_match is not None and look_match.command_key == "directed_look":
        args = look_match.args or {}
        target_hint = str(args.get("target_hint") or "").strip()
        if target_hint:
            kind = _directed_target_kind(target_hint)
            if kind == "person":
                return _scan_for_person(
                    target_name=_target_name_from_hint(target_hint, person_name),
                    start_direction=str(args.get("direction") or _directed_look_context.get("last_direction") or "current"),
                    person_name=person_name,
                )
            return _search_for_object(
                target_hint=target_hint,
                start_direction=str(args.get("direction") or _directed_look_context.get("last_direction") or "current"),
                raw_text=text,
                person_id=person_id,
            )
        return None

    kind = _directed_target_kind(text)
    if kind == "object":
        return _search_for_object(
            target_hint=text.strip(),
            start_direction=str(_directed_look_context.get("last_direction") or "current"),
            raw_text=text,
            person_id=person_id,
        )
    if kind == "person":
        maybe_name = _extract_introduced_name(text, allow_bare_name=True)
        return _scan_for_person(
            target_name=maybe_name,
            start_direction=str(_directed_look_context.get("last_direction") or "current"),
            person_name=person_name,
        )

    return None


def _directed_look_label(direction: str) -> str:
    return {
        "left": "to your left",
        "right": "to your right",
        "up": "upward",
        "down": "downward",
        "center": "straight ahead",
        "current": "at what they were showing you",
    }.get((direction or "current").lower(), "at what they were showing you")


def _fallback_directed_look_response(analysis: dict) -> str:
    summary = (analysis or {}).get("target_summary") or ""
    roast = (analysis or {}).get("roast_angle") or ""
    if summary and roast:
        return f"{summary} {roast}"
    if summary:
        return summary
    return "I looked, but my photoreceptors found mostly mystery and questionable staging."


def _analysis_found_target(analysis: dict, target_hint: str) -> bool:
    if not analysis:
        return False
    if not target_hint:
        return True
    if isinstance(analysis.get("target_visible"), bool):
        return bool(analysis.get("target_visible"))
    confidence = str(analysis.get("confidence") or "").lower()
    subject_type = str(analysis.get("subject_type") or "").lower()
    if confidence == "high" and subject_type not in {"", "unknown"}:
        return True
    haystack = " ".join([
        str(analysis.get("target_summary") or ""),
        str(analysis.get("roast_angle") or ""),
        " ".join(str(x) for x in analysis.get("notable_details") or []),
    ]).lower()
    target_words = [
        word for word in re.findall(r"[a-z0-9']+", target_hint.lower())
        if len(word) > 2 and word not in {"this", "that", "the", "my", "your"}
    ]
    return bool(target_words and any(word in haystack for word in target_words))


def _directed_search_directions(start_direction: str) -> list[str]:
    configured = list(getattr(
        config,
        "DIRECTED_LOOK_SEARCH_DIRECTIONS",
        ["current", "left", "right", "down", "up"],
    ))
    ordered: list[str] = []
    for direction in [start_direction, *configured]:
        direction = (direction or "current").strip().lower()
        if direction == "other_way":
            continue
        if direction not in {"current", "left", "right", "up", "down", "center"}:
            continue
        if direction not in ordered:
            ordered.append(direction)
    max_attempts = int(getattr(config, "DIRECTED_LOOK_MAX_SEARCH_ATTEMPTS", 4))
    return ordered[:max(1, max_attempts)]


def _not_found_visual_response(target_hint: str) -> str:
    target = (target_hint or "that").strip()
    target = re.sub(r"^(?:this|that|the|my|your)\s+", "", target, flags=re.IGNORECASE).strip()
    target = target or "that"
    return (
        f"I don't see the {target} you're talking about. "
        "Either it's hiding, or my vision sensors have entered their tragic poet era."
    )


def _execute_directed_look_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
    raw_text: str,
) -> str:
    direction = (args.get("direction") or "current").strip().lower()
    target_hint = (args.get("target_hint") or "").strip()
    search_target = bool(args.get("search_target") and target_hint)
    target_kind = _directed_target_kind(target_hint) if target_hint else None
    bare_directional = not target_hint and not search_target
    search_start_direction = direction
    if (
        direction == "current"
        and target_hint
        and _directed_context_fresh()
        and _directed_look_context.get("last_direction")
    ):
        search_start_direction = str(_directed_look_context.get("last_direction"))

    try:
        if target_hint and (search_target or target_kind in {"person", "object"}):
            if target_kind == "person":
                return _scan_for_person(
                    target_name=_target_name_from_hint(target_hint, person_name),
                    start_direction=search_start_direction,
                    person_name=person_name,
                )
            return _search_for_object(
                target_hint=target_hint,
                start_direction=search_start_direction,
                raw_text=raw_text,
                person_id=person_id,
            )

        actual_direction, frame = _move_and_capture_gaze(direction, target_hint=target_hint)

        if bare_directional:
            visible_face = _visible_known_face_candidate(person_name)
            if visible_face:
                return (
                    _greet_directed_face_once(visible_face, person_name)
                    or _silent_command_response("directed_look.face_found")
                )

        if target_hint and target_kind == "scene":
            return _analyze_directed_view_once(
                frame=frame,
                direction=actual_direction,
                target_hint=target_hint,
                raw_text=raw_text,
                person_id=person_id,
                person_name=person_name,
            )

        _note_directed_look_context(
            direction=actual_direction,
            target_hint=target_hint,
            target_kind=target_kind,
        )

        if bare_directional:
            # Commit to the gaze he was told to take instead of immediately
            # scanning the room / drifting his head back up to level.
            try:
                consciousness.hold_directed_gaze(actual_direction)
            except Exception as exc:
                _log.debug("directed gaze hold failed: %s", exc)
            _directed_look_context["bare_count"] = int(
                _directed_look_context.get("bare_count") or 0
            ) + 1
            clarify_after = int(getattr(config, "DIRECTED_LOOK_CLARIFY_AFTER_COMMANDS", 3))
            if (
                _directed_look_context["bare_count"] >= max(1, clarify_after)
                and not _directed_look_context.get("asked_clarify")
            ):
                return _clarify_directed_search()
            return _silent_command_response(f"directed_look.{actual_direction}")
    except Exception as exc:
        _log.debug("directed look command failed: %s", exc)
        resp = "I tried to look, but my neck servos and photoreceptors are staging a tiny rebellion."
        _speak_blocking(resp)
        return resp

    resp = "Moved. Give me a target if this is supposed to become detective work."
    _speak_blocking(resp)
    return resp


def _execute_wave_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    target = (args.get("target") or "").strip()
    if not target or target.lower() == "them":
        target = person_name or "them"

    try:
        from sequences import animations
        threading.Thread(
            target=animations.arm_wave,
            daemon=True,
            name="command_wave",
        ).start()
    except Exception as exc:
        _log.debug("wave command motion failed: %s", exc)

    prompt = (
        f"You are physically waving your right arm to {target}. "
        "Give one short Rex-style line, max 16 words. "
        "Do not mention servos unless joking very lightly."
    )
    resp = llm.get_response(prompt, person_id) or f"Fine, {target}, consider yourself waved at."
    _speak_blocking(resp, emotion="happy")
    return llm.clean_response_text(resp)


_MEMORY_SELF_REFS = {"i", "me", "my", "myself"}
_MEMORY_NAME_RE = re.compile(r"^[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}$")
_MEMORY_PERSON_NAME_CHUNK = r"[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}"
_MEMORY_RELATION_WORDS = (
    r"best\s+friend|partner|spouse|wife|husband|girlfriend|boyfriend|"
    r"fianc[eé]e?|father|dad|mother|mom|parent|son|daughter|child|"
    r"brother|sister|sibling|friend|boss|manager|supervisor|employee|"
    r"coworker|co[-\s]?worker|colleague|roommate|neighbor|neighbour"
)
_DIRECT_RELATIONSHIP_PATTERNS = (
    re.compile(
        rf"\bmy\s+(?P<rel>{_MEMORY_RELATION_WORDS})\s+"
        rf"(?:is\s+)?(?:named|called)\s+(?P<name>{_MEMORY_PERSON_NAME_CHUNK})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bmy\s+(?P<rel>{_MEMORY_RELATION_WORDS})'?s\s+name\s+is\s+"
        rf"(?P<name>{_MEMORY_PERSON_NAME_CHUNK})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?P<name>{_MEMORY_PERSON_NAME_CHUNK})\s+is\s+my\s+"
        rf"(?P<rel>{_MEMORY_RELATION_WORDS})\b",
        re.IGNORECASE,
    ),
)
_NAMED_PERSON_FACT_STATEMENT_RE = re.compile(
    rf"^\s*(?P<name>{_MEMORY_PERSON_NAME_CHUNK})\s+"
    r"(?P<verb>is|has|works(?:\s+as|\s+in|\s+for)?)\s+"
    r"(?P<detail>[^.?!]{2,160})",
)
_JOB_DETAIL_RE = re.compile(
    r"\b("
    r"administrator|admin|analyst|architect|assistant|developer|director|"
    r"doctor|editor|engineer|manager|nurse|programmer|teacher|technician|"
    r"writer|works?\s+(?:as|in|for)"
    r")\b",
    re.IGNORECASE,
)


def _memory_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_") or "detail"


def _clean_memory_person_name(value: str) -> str:
    name = " ".join(str(value or "").strip(" .?!,;:").split())
    name = re.split(
        r"\b(?:and|but|who|that|which|please|thanks|thank\s+you)\b",
        name,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .?!,;:")
    return name if _MEMORY_NAME_RE.match(name) else ""


def _learn_direct_memory_from_user_text(
    text: str,
    person_id: Optional[int],
) -> None:
    """Store obvious relationship/person facts immediately, without an LLM pass."""
    if person_id is None:
        return
    _learn_direct_relationship_statement(text, int(person_id))
    _learn_named_person_fact_statement(text, int(person_id))


def _learn_direct_relationship_statement(text: str, person_id: int) -> None:
    for pattern in _DIRECT_RELATIONSHIP_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        rel = _normalize_relationship_word(match.group("rel")) or ""
        name = _clean_memory_person_name(match.group("name"))
        if not rel or not name:
            return
        other_id, _created = people_memory.find_or_create_person(name)
        if other_id is None or int(other_id) == int(person_id):
            return
        social_memory.save_relationship(
            from_person_id=int(person_id),
            to_person_id=int(other_id),
            relationship=rel,
            described_by=int(person_id),
        )
        facts_memory.add_fact(
            int(person_id),
            "family",
            f"{rel}_name",
            name,
            source="explicit",
            confidence=0.98,
            importance=0.9,
            decay_rate="permanent",
        )
        _record_recent_memory_candidate(
            person_id,
            kind="relationship",
            target=f"{rel} {name}",
            label=f"{rel.replace('_', ' ')} {name}",
        )
        _log.info(
            "[memory] learned direct relationship person_id=%s rel=%s name=%r other_id=%s",
            person_id,
            rel,
            name,
            other_id,
        )
        return


def _learn_named_person_fact_statement(text: str, person_id: int) -> None:
    match = _NAMED_PERSON_FACT_STATEMENT_RE.match(text or "")
    if not match:
        return
    name = _clean_memory_person_name(match.group("name"))
    if not name:
        return
    row = people_memory.find_person_by_name(name)
    if not row:
        return
    target_id = int(row["id"])
    if target_id == int(person_id):
        return
    verb = " ".join(str(match.group("verb") or "").lower().split())
    detail = " ".join(str(match.group("detail") or "").strip(" .?!").split())
    if not detail or detail.lower().startswith(("my ", "your ")):
        return
    value = re.sub(r"^(?:a|an|the)\s+", "", detail, flags=re.IGNORECASE).strip()
    category = "job" if verb.startswith("work") or _JOB_DETAIL_RE.search(detail) else "other"
    key = "job_title" if category == "job" else _memory_key(verb)
    facts_memory.add_fact(
        target_id,
        category,
        key,
        value,
        source="explicit",
        confidence=0.95,
        importance=0.8 if category == "job" else 0.55,
    )
    _record_recent_memory_candidate(
        target_id,
        kind="fact",
        target=f"{category} {key} {value}",
        label=f"{name}: {value}",
    )
    _log.info(
        "[memory] learned named-person fact target_id=%s name=%r key=%s value=%r",
        target_id,
        name,
        key,
        value,
    )


def _record_recent_memory_candidate(
    person_id: Optional[int],
    *,
    kind: str,
    target: str,
    label: str,
) -> None:
    if person_id is None or not target:
        return
    _recent_memory_candidates.append({
        "person_id": int(person_id),
        "kind": kind,
        "target": target,
        "label": label or target,
        "ts": time.time(),
    })


def _register_forget_terms(person_id: Optional[int], result: forgetting.ForgetResult) -> None:
    if person_id is None:
        return
    if result.terms:
        _session_forget_terms.setdefault(int(person_id), set()).update(result.terms)


def _extract_memory_statement_target(
    statement: str,
    person_id: Optional[int],
    person_name: Optional[str],
) -> tuple[Optional[int], Optional[str], str, bool]:
    """
    Resolve a spoken memory statement to a person.

    Returns (target_id, target_name, detail, explicitly_named). Self references
    resolve to the engaged person; another person's memory requires a clear name.
    """
    text = (statement or "").strip()
    if not text:
        return None, None, "", False

    if re.match(r"(?i)^(?:i|i'm|im|my|me)\b", text):
        return person_id, person_name, text, False

    possessive = re.match(
        r"^([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})'s\s+(.+)$",
        text,
    )
    if possessive:
        name = possessive.group(1).strip()
        detail = possessive.group(2).strip()
        row = people_memory.find_person_by_name(name)
        return (
            int(row["id"]) if row else None,
            (row.get("name") if row else name),
            detail,
            True,
        )

    subject = re.match(
        r"^([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})\s+"
        r"(likes|loves|hates|dislikes|prefers|avoids|is|has|works|wants|plays|collects)\b\s*(.*)$",
        text,
    )
    if subject:
        name = subject.group(1).strip()
        verb = subject.group(2).strip()
        rest = subject.group(3).strip()
        row = people_memory.find_person_by_name(name)
        return (
            int(row["id"]) if row else None,
            (row.get("name") if row else name),
            f"{verb} {rest}".strip(),
            True,
        )

    return person_id, person_name, text, False


def _resolve_review_target(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    target = (args.get("target") or "").strip()
    if args.get("self_ref") or target.lower() in _MEMORY_SELF_REFS:
        if person_id is None:
            return None, None, "I need to know who you are before I can review your memory."
        return person_id, person_name, None

    row = people_memory.find_person_by_name(target)
    if not row:
        return None, None, f"I don't have a clear memory record for {target}."
    return int(row["id"]), row.get("name"), None


def _execute_memory_review_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    target_id, _target_name, error = _resolve_review_target(args, person_id, person_name)
    if error:
        _speak_blocking(error)
        return error
    resp = person_summary.summarize_for_review(
        int(target_id),
        include_sensitive=bool(args.get("include_sensitive")),
    )
    _speak_blocking(resp, emotion="neutral")
    return resp


def _execute_memory_forget_fact_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    statement = (args.get("statement") or "").strip()
    target_id, target_name, detail, named = _extract_memory_statement_target(
        statement,
        person_id,
        person_name,
    )
    if target_id is None:
        who = target_name or "that person"
        resp = f"I need a clearer person match before I delete {who}'s memory."
        _speak_blocking(resp)
        return resp
    if named is False and person_id is None:
        resp = "I need to know whose memory this belongs to before I delete it."
        _speak_blocking(resp)
        return resp

    target = " ".join(part for part in [detail, statement] if part).strip()
    result = forgetting.forget_memory_detail(int(target_id), target)
    _register_forget_terms(target_id, result)
    _log.info(
        "[memory] local forget person_id=%s named=%s statement=%r deleted=%s",
        target_id,
        named,
        statement,
        result.deleted,
    )
    if result.total_deleted <= 0:
        resp = f"I couldn't find a stored fact or preference matching {detail or statement}."
    else:
        resp = f"Deleted that memory for {target_name or 'them'}. Clean, deliberate, logged."
    _speak_blocking(resp, emotion="neutral")
    return resp


def _execute_memory_boundary_command(
    person_id: Optional[int],
) -> str:
    candidate = None
    for item in reversed(_recent_memory_candidates):
        if person_id is None or int(item.get("person_id")) == int(person_id):
            candidate = item
            break
    if not candidate:
        resp = "Nothing recent to discard. My memory banks are innocent this time."
        _speak_blocking(resp)
        return resp

    target_id = int(candidate["person_id"])
    result = forgetting.forget_memory_detail(target_id, candidate.get("target") or "")
    _register_forget_terms(target_id, result)
    try:
        _recent_memory_candidates.remove(candidate)
    except ValueError:
        pass
    _log.info(
        "[memory] discarded recent candidate person_id=%s kind=%s target=%r deleted=%s",
        target_id,
        candidate.get("kind"),
        candidate.get("target"),
        result.deleted,
    )
    resp = f"Understood. I discarded the recent memory about {candidate.get('label') or 'that'}."
    _speak_blocking(resp, emotion="neutral")
    return resp


def _execute_memory_correct_fact_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    correction = (args.get("correction") or "").strip()
    call_me = re.match(r"(?i)^(?:please\s+)?call\s+me\s+(.+)$", correction)
    if call_me:
        new_name = call_me.group(1).strip(" .!?")
        if person_id is None:
            resp = "I need to know who I'm renaming first."
            _speak_blocking(resp)
            return resp
        if not people_memory.rename_person(int(person_id), new_name):
            resp = f"I couldn't safely rename that memory to {new_name}."
            _speak_blocking(resp)
            return resp
        facts_memory.apply_fact_correction(
            int(person_id),
            "name",
            new_name,
            category="identity",
            importance=0.95,
            decay_rate="permanent",
        )
        _refresh_world_state_person_name(int(person_id), new_name)
        _log.info("[memory] corrected name person_id=%s new=%r", person_id, new_name)
        resp = repair_moves.add_better_luck_line(
            f"Corrected. Bret-level bureaucracy complete: I will call you {new_name}."
        )
        _speak_blocking(resp, emotion="happy")
        return resp

    target_id, target_name, detail, named = _extract_memory_statement_target(
        correction,
        person_id,
        person_name,
    )
    last_name = re.match(r"(?i)^last\s+name\s+is\s+([A-Za-z][A-Za-z' -]+)$", detail)
    if last_name and target_id is not None:
        surname = last_name.group(1).strip()
        first = _first_name_or(target_name or correction)
        full_name = f"{first} {surname}".strip()
        if people_memory.rename_person(int(target_id), full_name):
            facts_memory.apply_fact_correction(
                int(target_id),
                "last_name",
                surname,
                category="identity",
                importance=0.95,
                decay_rate="permanent",
            )
            _refresh_world_state_person_name(int(target_id), full_name)
            _log.info("[memory] corrected last name person_id=%s new=%r", target_id, full_name)
            resp = repair_moves.add_better_luck_line(
                f"Corrected. {first}'s last name is {surname}. "
                "Logged with actual confidence for once."
            )
        else:
            resp = f"I couldn't safely update that name to {full_name}."
        _speak_blocking(resp, emotion="neutral")
        return resp

    generic = re.match(r"(?i)^(.+?)\s+is\s+(.+)$", detail)
    if target_id is not None and generic:
        key = _memory_key(generic.group(1))
        value = generic.group(2).strip(" .!?")
        facts_memory.apply_fact_correction(
            int(target_id),
            key,
            value,
            category="other",
            importance=0.75,
        )
        _log.info(
            "[memory] corrected fact person_id=%s key=%r value=%r named=%s",
            target_id,
            key,
            value,
            named,
        )
        resp = repair_moves.add_better_luck_line(
            f"Corrected. I now have {target_name or 'them'} as {generic.group(1)}: {value}."
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    resp = "I heard the correction, but I need one clear fact to update."
    _speak_blocking(resp)
    return resp


def _store_simple_memory_statement(
    target_id: int,
    detail: str,
    *,
    source: str = "explicit",
) -> str:
    text = (detail or "").strip(" .!?")
    called = re.match(r"(?i)^hates\s+being\s+called\s+(.+)$", text)
    if called:
        nickname = called.group(1).strip(" .!?")
        preferences_memory.upsert_preference(
            int(target_id),
            "interaction",
            "boundary",
            f"not_called_{_memory_key(nickname)}",
            f"Do not call them {nickname}.",
            confidence=1.0,
            importance=0.98,
            source=source,
        )
        return f"not being called {nickname}"

    like = re.match(r"(?i)^(likes|loves|hates|dislikes|prefers|avoids)\s+(.+)$", text)
    if like:
        verb = like.group(1).lower()
        thing = like.group(2).strip()
        pref_type = {
            "likes": "likes",
            "loves": "likes",
            "hates": "dislikes",
            "dislikes": "dislikes",
            "prefers": "prefers",
            "avoids": "avoids",
        }[verb]
        domain = "music" if "music" in thing.lower() or thing.lower() in {"country", "country music"} else "general"
        preferences_memory.upsert_preference(
            int(target_id),
            domain,
            pref_type,
            _memory_key(thing),
            thing,
            confidence=0.95,
            importance=0.85 if pref_type in {"dislikes", "avoids"} else 0.7,
            source=source,
        )
        return thing

    generic = re.match(r"(?i)^(.+?)\s+is\s+(.+)$", text)
    if generic:
        key = _memory_key(generic.group(1))
        value = generic.group(2).strip()
        facts_memory.add_fact(
            int(target_id),
            "other",
            key,
            value,
            source=source,
            confidence=0.95,
        )
        return f"{generic.group(1)} is {value}"

    facts_memory.add_fact(
        int(target_id),
        "other",
        _memory_key(text[:40]),
        text,
        source=source,
        confidence=0.95,
        importance=0.35,
        decay_rate="fast",
    )
    return text


def _execute_memory_remember_fact_command(
    args: dict,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    statement = (args.get("statement") or "").strip()
    target_id, target_name, detail, named = _extract_memory_statement_target(
        statement,
        person_id,
        person_name,
    )
    if target_id is None and named and target_name and _MEMORY_NAME_RE.match(target_name):
        target_id, _created = people_memory.find_or_create_person(target_name)
    if target_id is None:
        resp = "I need a clearer person before I store that memory."
        _speak_blocking(resp)
        return resp
    stored_label = _store_simple_memory_statement(int(target_id), detail)
    _log.info(
        "[memory] remembered explicit statement person_id=%s named=%s statement=%r",
        target_id,
        named,
        statement,
    )
    resp = f"Remembered: {target_name or 'they'} - {stored_label}. Trustworthy memory, minimal theatrics."
    _speak_blocking(resp, emotion="happy")
    return resp


def _execute_command(
    match: command_parser.CommandMatch,
    person_id: Optional[int],
    person_name: Optional[str],
    raw_text: str,
) -> str:
    """
    Dispatch a locally matched command, speak the response, and return the
    response text for transcript logging.
    """
    key = match.command_key
    args = match.args
    updater = person_name or "unknown"

    def _say(prompt: str, emotion: str = "neutral") -> str:
        resp = llm.get_response(prompt, person_id)
        _speak_blocking(resp, emotion)
        return resp

    def _target_for_user(target: str) -> str:
        cleaned = (target or "").strip()
        if re.match(r"(?i)^my\b", cleaned):
            return re.sub(r"(?i)^my\b", "your", cleaned, count=1)
        return cleaned

    # ── Physical attention / movement ─────────────────────────────────────────
    if key == "directed_look":
        return _execute_directed_look_command(args, person_id, person_name, raw_text)

    if key == "wave_to":
        return _execute_wave_command(args, person_id, person_name)

    if key == "query_play_options":
        resp = "I can play music or games, which would you like?"
        _speak_blocking(resp)
        return resp

    # ── Time & date ────────────────────────────────────────────────────────────
    if key == "time_query":
        resp = _format_current_time_response()
        _speak_blocking(resp)
        return resp

    if key == "date_query":
        resp = _format_current_date_response()
        _speak_blocking(resp)
        return resp

    # ── State transitions ──────────────────────────────────────────────────────
    if key == "sleep":
        return _enter_sleep_mode()

    if key == "wake_up":
        if state_module.get_state() == State.SLEEP:
            return _wake_from_sleep()
        if state_module.get_state() == State.QUIET:
            state_module.set_state(State.ACTIVE)
            return _say(
                "You just exited quiet mode after being told to resume. "
                "One short in-character line."
            )
        state_module.set_state(State.ACTIVE)
        return _say("You just woke up. One short in-character wake-up line.")

    if key == "quiet_mode":
        resp = llm.get_response(
            "You are entering quiet mode and won't speak until told to resume. "
            "One brief in-character acknowledgment.",
            person_id,
        )
        _speak_blocking(resp)
        state_module.set_state(State.QUIET)
        return resp

    if key == "shutdown":
        resp = llm.get_response(
            "You are powering all the way down now (not just napping). Your "
            "always-listening helper stays awake and will boot you back up when "
            "someone says 'wake up Rex'. Give one final, in-character sign-off "
            "line — a little dramatic, like a pilot powering down the cockpit.",
            person_id,
        )
        _speak_blocking(resp)
        state_module.set_state(State.SHUTDOWN)
        return resp

    # ── Memory ─────────────────────────────────────────────────────────────────
    if key == "memory_review":
        return _execute_memory_review_command(args, person_id, person_name)

    if key == "memory_forget_fact":
        return _execute_memory_forget_fact_command(args, person_id, person_name)

    if key == "memory_boundary":
        return _execute_memory_boundary_command(person_id)

    if key == "memory_correct_fact":
        return _execute_memory_correct_fact_command(args, person_id, person_name)

    if key == "memory_remember_fact":
        return _execute_memory_remember_fact_command(args, person_id, person_name)

    if key == "forget_specific":
        target = (args.get("target") or "").strip()
        if person_id is None:
            resp = "I need to know who I'm forgetting that for first."
            _speak_blocking(resp)
            return resp
        result = forgetting.forget_specific_memory(person_id, target)
        if result.terms:
            existing = _session_forget_terms.setdefault(int(person_id), set())
            existing.update(result.terms)
        _log.info(
            "[memory] targeted forget person_id=%s target=%r terms=%s deleted=%s",
            person_id,
            target,
            sorted(result.terms),
            result.deleted,
        )
        if result.total_deleted <= 0:
            resp = f"I couldn't find a stored memory matching {_target_for_user(target)}."
        else:
            resp = f"Okay. I forgot the stored memory I found about {_target_for_user(target)}."
        _speak_blocking(resp, emotion="neutral")
        return resp

    if key == "forget_me":
        target_id, target_name = _resolve_forget_me_target(person_id, person_name)
        if target_id is None:
            resp = "I don't have any record of you to forget."
            _speak_blocking(resp)
            return resp
        _arm_memory_wipe_confirmation(
            scope="person",
            person_id=target_id,
            person_name=target_name,
            requester_id=target_id,
        )
        resp = (
            "Memory wipe armed. Are you sure? Say \"yes forget me\" to confirm "
            "and I'll delete your name, face, voice print, and everything I "
            "remember about you. Very dramatic. Very irreversible."
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    if key == "forget_everyone":
        _arm_memory_wipe_confirmation(
            scope="all",
            requester_id=person_id,
        )
        resp = (
            "Full memory wipe armed. Are you absolutely sure? Say "
            f"{_memory_wipe_confirmation_hint('all')} to delete every person, "
            "face, voice print, relationship, and conversation memory. This is "
            "the big red button, just with worse branding."
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    if key == "whats_my_name":
        if person_name:
            return _say(
                f"The person you're talking to is named {person_name}. "
                f"Tell them their name in one in-character line."
            )
        return _say(
            "You don't know this person's name. Tell them you don't know who they are, in character."
        )

    if key == "rename_me":
        response = _handle_name_update_request(raw_text, person_id, person_name)
        return response or ""

    # ── Status ─────────────────────────────────────────────────────────────────
    if key == "status_uptime":
        sys_state = interoception.get_system_state()
        up = sys_state.get("uptime_seconds", 0)
        return _say(
            f"Your uptime is {up // 3600}h {(up % 3600) // 60}m. "
            f"State your uptime in one in-character line."
        )

    # ── Personality ────────────────────────────────────────────────────────────
    if key == "set_personality":
        param = args.get("param", "")
        value = args.get("value", 50)
        if not param:
            return ""
        old, new = personality.set_param(param, int(value), updated_by=updater)
        resp = personality.generate_acknowledgment(param, old, new)
        _speak_blocking(resp)
        return resp

    if key == "query_personality":
        param = args.get("param", "")
        if not param:
            return ""
        val = personality.get_param(param)
        return _say(
            f"Someone asked what your {param} level is. It is currently {val}/100. "
            f"Answer in one in-character line."
        )

    # ── DJ controls ────────────────────────────────────────────────────────────
    if key in ("dj_stop", "dj_skip", "volume_up", "volume_down", "dj_play_vibe"):
        try:
            from features import dj as dj_mod
            if key == "dj_stop":
                dj_mod.stop()
                return _say("The music just stopped. One in-character line.")
            if key == "dj_skip":
                dj_mod.skip()
                return _say("Skipping to the next track. One in-character line.")
            if key == "volume_up":
                dj_mod.volume_up()
                return _say("Turning the volume up. One short in-character line.")
            if key == "volume_down":
                dj_mod.volume_down()
                return _say("Turning the volume down. One short in-character line.")
            if key == "dj_play_vibe":
                vibe = args.get("vibe", "")
                track = dj_mod.handle_request(vibe)
                response = _say(
                    f"Playing something with vibe '{vibe}'. Announce it in one in-character line."
                )
                if track is not None:
                    _start_dj_after_response(track)
                return response
        except Exception as exc:
            _log.debug("DJ command error: %s", exc)

    # ── Games ──────────────────────────────────────────────────────────────────
    if key in (
        "start_trivia", "start_i_spy", "start_20_questions",
        "start_jeopardy", "start_word_association", "start_game", "stop_game",
    ):
        try:
            from features import games as games_mod
            if key == "start_trivia":
                resp = games_mod.start_trivia(person_id)
                _speak_blocking(resp)
                return resp
            if key == "start_i_spy":
                resp = games_mod.start_game("i_spy", person_id)
                _speak_blocking(resp)
                return resp
            if key == "start_20_questions":
                resp = games_mod.start_game("20_questions", person_id)
                _speak_blocking(resp)
                return resp
            if key == "start_jeopardy":
                resp = games_mod.start_game("jeopardy", person_id)
                _speak_blocking(resp)
                return resp
            if key == "start_word_association":
                resp = games_mod.start_game("word_association", person_id)
                _speak_blocking(resp)
                return resp
            if key == "start_game":
                game = args.get("game", "")
                resp = games_mod.start_game(game, person_id)
                _speak_blocking(resp)
                return resp
            if key == "stop_game":
                if not games_mod.is_active():
                    try:
                        from features import dj as dj_mod
                        if dj_mod.is_playing():
                            dj_mod.stop()
                            return _say("The music just stopped. One in-character line.")
                    except Exception as exc:
                        _log.debug("DJ fallback for stop_game command failed: %s", exc)
                resp = games_mod.stop_game(person_id)
                _speak_blocking(resp)
                return resp
        except Exception as exc:
            _log.debug("Games command error: %s", exc)

    # ── Vision ─────────────────────────────────────────────────────────────────
    if key == "vision_describe":
        desc = ""
        try:
            from vision import scene as vision_scene
            desc = vision_scene.describe_scene()
        except Exception as exc:
            _log.debug("vision describe error: %s", exc)
        return _say(
            f"You were asked what you see. Scene analysis: '{desc or 'nothing notable'}'. "
            f"Describe it in character in one or two lines."
        )

    if key == "vision_who_am_i":
        if person_name:
            return _say(
                f"Someone asked if you know who they are. You do — they're {person_name}. "
                f"Confirm in one in-character line."
            )
        return _say(
            "Someone asked if you know who they are. You don't recognize them. "
            "Respond in one in-character line."
        )

    # ── Fallback for any unhandled key ─────────────────────────────────────────
    return _stream_llm_response(raw_text, person_id)


# ─────────────────────────────────────────────────────────────────────────────
# Post-response processing
# ─────────────────────────────────────────────────────────────────────────────

def _forgotten_terms_for_person(person_id: Optional[int]) -> set[str]:
    if person_id is None:
        return set()
    return set(_session_forget_terms.get(int(person_id), set()))


def _filter_forgotten_transcript(
    transcript: list[dict],
    person_id: Optional[int],
) -> list[dict]:
    terms = _forgotten_terms_for_person(person_id)
    if not terms:
        return transcript
    return [
        turn for turn in transcript
        if not forgetting.text_matches_terms(turn.get("text") or "", terms)
    ]


def _extracted_memory_allowed(payload: dict, person_id: Optional[int]) -> bool:
    terms = _forgotten_terms_for_person(person_id)
    if not terms:
        return True
    return not forgetting.fact_or_event_matches(payload, terms)


def _conversation_exchange_count() -> int:
    """Transcript length so far — the cadence clock (~2 lines per back-and-forth).

    Mirrors how rex_pov / the conversation arc clock themselves; shrinks at a
    session reset, which `_memory_followup_cadence_allows` uses to self-reset."""
    try:
        return len(conv_memory.get_session_transcript() or [])
    except Exception:
        return 0


def _memory_followup_cadence_allows() -> bool:
    """Moderate clamp: may a proactive memory follow-up ("how did X go?") fire now?

    Fires at most one per conversational lull — requires a minimum gap in
    transcript exchanges AND a seconds cooldown since the last follow-up, and
    stays quiet when the arc reads the room as flat. The spacing is also what lets
    Rex's own POV / idle banter occasionally win the proactive slot instead of
    every gap becoming another interview question. Self-resets when the transcript
    shrinks (new session)."""
    global _last_followup_exchange, _last_followup_at, _fired_followup_event_ids
    try:
        import config
        min_gap = int(getattr(config, "FOLLOWUP_MIN_GAP_EXCHANGES", 5) or 0)
        cooldown = float(getattr(config, "FOLLOWUP_COOLDOWN_SECS", 60.0) or 0.0)
        suppress_flat = bool(getattr(config, "FOLLOWUP_SUPPRESS_WHEN_FLAT", True))
    except Exception:
        min_gap, cooldown, suppress_flat = 5, 60.0, True

    now_exchange = _conversation_exchange_count()
    if now_exchange < _last_followup_exchange:  # transcript shrank → new session
        _last_followup_exchange = -(10**9)
        _last_followup_at = 0.0
        _fired_followup_event_ids = set()

    if suppress_flat:
        try:
            if topic_thread.arc_reads_flat():
                return False
        except Exception:
            pass
    if now_exchange - _last_followup_exchange < min_gap:
        return False
    if cooldown > 0.0 and _last_followup_at > 0.0 and (
        time.monotonic() - _last_followup_at < cooldown
    ):
        return False
    return True


def _note_memory_followup_fired(event_id) -> None:
    """Record that a proactive memory follow-up just fired (cadence + anti-repeat)."""
    global _last_followup_exchange, _last_followup_at
    _last_followup_exchange = _conversation_exchange_count()
    _last_followup_at = time.monotonic()
    try:
        if event_id is not None:
            _fired_followup_event_ids.add(int(event_id))
    except Exception:
        pass


def _post_response(
    user_text: str,
    person_id: Optional[int],
    person_name: Optional[str] = None,
    *,
    assistant_asked_question: bool = False,
    pre_classified_insult: bool = False,
    pre_classified_compliment: bool = False,
    suppress_memory_learning: bool = False,
) -> None:
    """
    Run after every response in ACTIVE state. Sentiment, facts, follow-up,
    and interoception are handled here. Sentiment/facts run in a background
    thread; follow-up delivery and interoception run in the calling thread.
    """
    # ── Follow-up delivery (sync — spoken as part of this turn) ───────────────
    global _awaiting_followup_event

    if _game_suppresses_conversation():
        try:
            interoception.record_interaction()
        except Exception as exc:
            _log.debug("post_response interoception error: %s", exc)
        return

    suppress_stale_followup = False
    try:
        suppress_stale_followup = events_memory.looks_like_cancellation(user_text)
    except Exception:
        suppress_stale_followup = False
    # A reply that an event didn't happen ("I didn't end up going") shouldn't be
    # answered by immediately pivoting to a follow-up about a DIFFERENT remembered
    # event — that reads as not listening. Hold the queue for this turn.
    if not suppress_stale_followup:
        try:
            if _followup_event_did_not_happen(user_text):
                suppress_stale_followup = True
        except Exception:
            pass
    try:
        visible_known_count = sum(
            1
            for p in (world_state.get("people") or [])
            if p.get("person_db_id") is not None
        )
        if visible_known_count > 1:
            suppress_stale_followup = True
    except Exception:
        pass

    if person_id is not None and not suppress_stale_followup:
        try:
            followups = consciousness.get_pending_followup(person_id)
            if followups:
                # Never follow up twice on the same event in a session. The in-memory
                # queue can get re-seeded from the DB in the window before a "didn't
                # happen" reply is written, which otherwise re-raises a resolved event
                # (e.g. re-asking about a trip the user already said they skipped).
                followups = [
                    event
                    for event in followups
                    if int(event.get("id") or -1) not in _fired_followup_event_ids
                ]
            if not followups:
                pass
            elif assistant_asked_question:
                # If Rex just asked a question, do not immediately ask another one.
                # Re-queue and wait for the person's reply first.
                for event in followups:
                    consciousness.set_pending_followup(person_id, event)
            elif not _memory_followup_cadence_allows():
                # Moderate cadence clamp: at most one follow-up per conversational
                # lull, and none while the room reads flat. Re-queue everything so
                # nothing is lost — it fires in a later lull, which also leaves the
                # proactive slot open for Rex's own POV / idle banter.
                for event in followups:
                    consciousness.set_pending_followup(person_id, event)
            else:
                # Ask one follow-up at a time; keep the rest queued.
                event = followups[0]
                for leftover in followups[1:]:
                    consciousness.set_pending_followup(person_id, leftover)

                event_name = event.get("event_name", "that thing you mentioned")
                resp = llm.get_response(
                    f"You're following up on something this person mentioned before: "
                    f"'{event_name}'. Ask how it went in one short Rex-style line.",
                    person_id,
                )
                if resp:
                    conv_memory.add_to_transcript("Rex", resp)
                    conv_log.log_rex(resp)
                    completed = _speak_blocking(resp)
                    if completed:
                        _register_rex_utterance(
                            resp,
                            source="memory_followup",
                            topic=event_name,
                            target_person_id=person_id,
                            target_name=person_name,
                            expected_reply_types=[
                                "status_update",
                                "cancel_event",
                                "dismissal",
                            ],
                            blocked_actions=[
                                "identity.name_correction",
                                "identity.introduce_person",
                            ],
                        )
                        _awaiting_followup_event = None
                        event_id = event.get("id")
                        # Record the fire for cadence spacing + per-session
                        # anti-repeat (so this event is never re-raised).
                        _note_memory_followup_fired(event_id)
                        if event_id is not None:
                            _awaiting_followup_event = {
                                "person_id": person_id,
                                "event_id": int(event_id),
                                "event_name": event_name,
                            }
                        try:
                            consciousness.note_memory_hint(resp, person_id)
                        except Exception as exc:
                            _log.debug("note follow-up memory hint failed: %s", exc)
                    else:
                        consciousness.set_pending_followup(person_id, event)
                else:
                    consciousness.set_pending_followup(person_id, event)
        except Exception as exc:
            _log.debug("post_response follow-up error: %s", exc)

    # ── Interoception (sync — just a counter update, no I/O) ──────────────────
    try:
        interoception.record_interaction()
    except Exception as exc:
        _log.debug("post_response interoception error: %s", exc)

    memory_recent_transcript = None
    if person_id is not None and not suppress_memory_learning:
        try:
            _learn_direct_memory_from_user_text(user_text, person_id)
        except Exception as exc:
            _log.debug("direct memory learning error: %s", exc)
        try:
            transcript = conv_memory.get_session_transcript()
            recent = transcript[-10:] if len(transcript) >= 10 else transcript
            memory_recent_transcript = list(_filter_forgotten_transcript(recent, person_id))
        except Exception as exc:
            _log.debug("post_response memory transcript snapshot error: %s", exc)
            memory_recent_transcript = []

    # ── Sentiment + facts (async — both are LLM calls) ────────────────────────
    def _background() -> None:
        # Sentiment analysis → anger / relationship updates
        try:
            sentiment = llm.analyze_sentiment(user_text)

            if sentiment.get("is_insult"):
                # Skip if layer 1 already counted this turn so we don't
                # double-bump anger or antagonism for one utterance.
                if not pre_classified_insult:
                    personality.increment_anger(person_id)
                    if person_id is not None:
                        people_memory.apply_relationship_increment(person_id, "insult_mild")
                    _set_body_mood("offended", source="layer2_insult")

            elif sentiment.get("is_apology"):
                personality.decrement_anger()
                if person_id is not None:
                    people_memory.apply_relationship_increment(person_id, "sincere_apology")

            if sentiment.get("is_compliment"):
                if person_id is not None:
                    people_memory.apply_relationship_increment(person_id, "compliment")
                # Layer-2 pleased reaction for subtler praise the keyword pre-check
                # missed; skip when layer 1 already reacted this turn.
                if not pre_classified_compliment:
                    try:
                        _play_event_body_beat("compliment.detected")
                        _set_body_mood("proud", source="layer2_compliment")
                    except Exception as exc:
                        _log.debug("[interaction] layer-2 compliment beat skipped: %s", exc)

        except Exception as exc:
            _log.debug("post_response sentiment error: %s", exc)

        if person_id is not None and not suppress_memory_learning:
            try:
                friendship_patterns.learn_from_turn(person_id, user_text)
            except Exception as exc:
                _log.debug("friendship pattern learning error: %s", exc)

        # Fact extraction from recent transcript
        if person_id is not None and memory_recent_transcript is not None:
            try:
                recent = list(memory_recent_transcript)
                new_facts = llm.extract_facts(person_id, recent, person_name=person_name)
                saved_count = 0
                for fact in new_facts:
                    if (
                        fact.get("key")
                        and fact.get("value")
                        and _extracted_memory_allowed(fact, person_id)
                    ):
                        facts_memory.add_fact(
                            person_id,
                            fact.get("category", "other"),
                            fact["key"],
                            fact["value"],
                            source="explicit",
                            confidence=0.95,
                        )
                        _record_recent_memory_candidate(
                            person_id,
                            kind="fact",
                            target=" ".join([
                                str(fact.get("category") or ""),
                                str(fact.get("key") or ""),
                                str(fact.get("value") or ""),
                            ]),
                            label=str(fact.get("value") or fact.get("key") or "that fact"),
                        )
                        saved_count += 1
                _log.info(
                    "[interaction] facts extracted=%d saved=%d for person_id=%s",
                    len(new_facts), saved_count, person_id,
                )
            except Exception as exc:
                _log.debug("post_response fact extraction error: %s", exc)

            # Typed preference extraction — supplements generic facts.
            try:
                recent = list(memory_recent_transcript)
                new_preferences = llm.extract_preferences(
                    person_id,
                    recent,
                    person_name=person_name,
                )
                saved_preferences = 0
                for pref in new_preferences:
                    if (
                        pref.get("domain")
                        and pref.get("preference_type")
                        and pref.get("key")
                        and _extracted_memory_allowed(pref, person_id)
                    ):
                        if pref.get("preference_type") == "boundary":
                            pref["importance"] = max(
                                float(pref.get("importance") or 0.0),
                                0.95,
                            )
                        preferences_memory.upsert_preference(
                            person_id,
                            pref["domain"],
                            pref["preference_type"],
                            pref["key"],
                            pref.get("value") or "",
                            confidence=float(pref.get("confidence") or 1.0),
                            importance=float(pref.get("importance") or 0.5),
                            source=pref.get("source") or "explicit",
                        )
                        _record_recent_memory_candidate(
                            person_id,
                            kind="preference",
                            target=" ".join([
                                str(pref.get("domain") or ""),
                                str(pref.get("preference_type") or ""),
                                str(pref.get("key") or ""),
                                str(pref.get("value") or ""),
                            ]),
                            label=str(pref.get("value") or pref.get("key") or "that preference"),
                        )
                        saved_preferences += 1
                _log.info(
                    "[interaction] preferences extracted=%d saved=%d for person_id=%s",
                    len(new_preferences), saved_preferences, person_id,
                )
            except Exception as exc:
                _log.debug("post_response preference extraction error: %s", exc)

            # Typed interest extraction — durable conversation hooks separate
            # from generic facts.
            try:
                recent = list(memory_recent_transcript)
                new_interests = llm.extract_interests(
                    person_id,
                    recent,
                    person_name=person_name,
                )
                saved_interests = 0
                for interest in new_interests:
                    if (
                        interest.get("name")
                        and _extracted_memory_allowed(interest, person_id)
                    ):
                        interests_memory.upsert_interest(
                            person_id,
                            interest["name"],
                            interest.get("category") or "hobby",
                            interest.get("interest_strength") or "medium",
                            confidence=float(interest.get("confidence") or 1.0),
                            source=interest.get("source") or "explicit",
                            notes=interest.get("notes") or "",
                            associated_people=interest.get("associated_people") or "",
                            associated_stories=interest.get("associated_stories") or "",
                        )
                        _record_recent_memory_candidate(
                            person_id,
                            kind="interest",
                            target=" ".join([
                                str(interest.get("name") or ""),
                                str(interest.get("category") or ""),
                                str(interest.get("notes") or ""),
                            ]),
                            label=str(interest.get("name") or "that interest"),
                        )
                        saved_interests += 1
                _log.info(
                    "[interaction] interests extracted=%d saved=%d for person_id=%s",
                    len(new_interests), saved_interests, person_id,
                )
            except Exception as exc:
                _log.debug("post_response interest extraction error: %s", exc)

            # Event extraction → person_events table for follow-ups + small talk
            try:
                recent = list(memory_recent_transcript)
                new_events = llm.extract_events(person_id, recent, person_name=person_name)
                saved_events = 0
                if new_events:
                    existing = events_memory.get_upcoming_events(person_id) or []
                    existing_keys = {
                        ((e.get("event_name") or "").strip().lower(), e.get("event_date"))
                        for e in existing
                    }
                    for ev in new_events:
                        if not _extracted_memory_allowed(ev, person_id):
                            continue
                        key = (ev["event_name"].strip().lower(), ev.get("event_date"))
                        if key in existing_keys:
                            continue
                        events_memory.add_event(
                            person_id,
                            ev["event_name"],
                            ev.get("event_date"),
                            ev.get("event_notes", ""),
                        )
                        existing_keys.add(key)
                        saved_events += 1
                if saved_events:
                    _log.info(
                        "[interaction] events extracted=%d saved=%d for person_id=%s",
                        len(new_events), saved_events, person_id,
                    )
            except Exception as exc:
                _log.debug("post_response event extraction error: %s", exc)

    threading.Thread(target=_background, daemon=True, name="post-response-bg").start()


# ─────────────────────────────────────────────────────────────────────────────
# Session memory consolidation
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_memory_float(value: object, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_memory_source(value: object, default: str = "explicit") -> str:
    source = str(value or default).strip().lower()
    return source if source in {"explicit", "inferred", "observed", "corrected"} else default


def _normalize_decay_rate(value: object, default: str = "normal") -> str:
    decay = str(value or default).strip().lower()
    return decay if decay in {"fast", "normal", "permanent"} else default


def _existing_memory_snapshot(person_id: int) -> dict:
    """Build a compact existing-memory snapshot for consolidation prompts."""
    try:
        person = people_memory.get_person(person_id) or {}
    except Exception:
        person = {}
    try:
        facts = facts_memory.get_prompt_worthy_facts(person_id, limit=16)
    except Exception:
        facts = []
    try:
        preferences = preferences_memory.get_preferences_for_prompt(person_id, limit=16)
    except Exception:
        preferences = []
    try:
        interests = interests_memory.get_interests_for_prompt(person_id, limit=16)
    except Exception:
        interests = []
    try:
        events = events_memory.get_open_events(person_id)
    except Exception:
        events = []
    try:
        relationships = social_memory.get_all_involving(person_id)
    except Exception:
        relationships = []
    try:
        emotional = emotional_events.get_active_events(person_id, limit=8)
    except Exception:
        emotional = []
    return {
        "person": {
            "id": person.get("id"),
            "name": person.get("name"),
            "nickname": person.get("nickname"),
            "friendship_tier": person.get("friendship_tier"),
        },
        "facts": [
            {
                "category": f.get("category"),
                "key": f.get("key"),
                "value": f.get("value"),
                "confidence": f.get("confidence"),
                "source": f.get("source"),
                "importance": f.get("importance"),
                "decay_rate": f.get("decay_rate"),
            }
            for f in facts
        ],
        "preferences": preferences,
        "interests": interests,
        "events": events,
        "relationships": relationships,
        "emotional_events": emotional,
    }


def _payload_allowed_against_terms(payload: dict, terms: set[str]) -> bool:
    if not terms:
        return True
    return not forgetting.fact_or_event_matches(payload, terms)


def _store_consolidated_facts(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        key = str(item.get("key") or item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        source = _normalize_memory_source(item.get("source"))
        category = str(item.get("category") or "other").strip().lower()
        if source == "corrected":
            facts_memory.apply_fact_correction(
                person_id,
                key,
                value,
                category=category,
                importance=_clamp_memory_float(item.get("importance"), 0.9),
                decay_rate=_normalize_decay_rate(item.get("decay_rate"), "normal"),
            )
            counts["updated"] += 1
        else:
            facts_memory.add_fact(
                person_id,
                category,
                key,
                value,
                source=source,
                confidence=_clamp_memory_float(item.get("confidence"), 0.95),
                importance=_clamp_memory_float(item.get("importance"), 0.5),
                decay_rate=_normalize_decay_rate(item.get("decay_rate"), "normal"),
            )
            counts["stored"] += 1


def _store_consolidated_preferences(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    valid_types = {"likes", "dislikes", "prefers", "avoids", "boundary"}
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        domain = str(item.get("domain") or item.get("category") or "general").strip().lower()
        pref_type = str(item.get("preference_type") or item.get("type") or "").strip().lower()
        key = str(item.get("key") or item.get("name") or "").strip()
        if pref_type not in valid_types or not key:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        importance = _clamp_memory_float(item.get("importance"), 0.5)
        if pref_type == "boundary":
            importance = max(importance, 0.95)
        preferences_memory.upsert_preference(
            person_id,
            domain,
            pref_type,
            key,
            str(item.get("value") or "").strip(),
            confidence=_clamp_memory_float(item.get("confidence"), 0.95),
            importance=importance,
            source=_normalize_memory_source(item.get("source")),
        )
        counts["stored"] += 1


def _store_consolidated_interests(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        name = str(item.get("name") or item.get("key") or "").strip()
        if not name:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        interests_memory.upsert_interest(
            person_id,
            name,
            str(item.get("category") or "hobby").strip().lower(),
            str(item.get("interest_strength") or "medium").strip().lower(),
            confidence=_clamp_memory_float(item.get("confidence"), 0.95),
            source=_normalize_memory_source(item.get("source")),
            notes=str(item.get("notes") or item.get("rationale") or "").strip(),
            associated_people=str(item.get("associated_people") or "").strip(),
            associated_stories=str(item.get("associated_stories") or "").strip(),
        )
        counts["stored"] += 1


def _store_consolidated_events(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    try:
        existing = events_memory.get_open_events(person_id) or []
    except Exception:
        existing = []
    existing_keys = {
        ((e.get("event_name") or "").strip().lower(), e.get("event_date"))
        for e in existing
    }
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        name = str(item.get("event_name") or item.get("name") or item.get("value") or "").strip()
        if not name:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        event_date = item.get("event_date")
        if event_date in ("", "null", "None"):
            event_date = None
        key = (name.lower(), event_date)
        if key in existing_keys:
            counts["updated"] += 1
            continue
        events_memory.add_event(
            person_id,
            name,
            event_date,
            str(item.get("event_notes") or item.get("rationale") or "").strip(),
        )
        existing_keys.add(key)
        counts["stored"] += 1


def _store_consolidated_relationships(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        other_name = str(item.get("other_person_name") or item.get("name") or "").strip()
        relationship = str(item.get("relationship") or item.get("value") or "").strip()
        if not other_name or not relationship:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        other_id, _created = people_memory.find_or_create_person(other_name)
        if other_id is None:
            counts["requires_review"] += 1
            continue
        direction = str(item.get("direction") or "current_to_other").strip().lower()
        if direction == "other_to_current":
            social_memory.save_relationship(int(other_id), person_id, relationship, described_by=person_id)
        else:
            social_memory.save_relationship(person_id, int(other_id), relationship, described_by=person_id)
        counts["stored"] += 1


def _store_consolidated_emotional_events(
    person_id: int,
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        desc = str(item.get("description") or item.get("value") or "").strip()
        if not desc:
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        sensitivity = item.get("sensitivity_decay_days")
        try:
            sensitivity = int(sensitivity) if sensitivity is not None else None
        except (TypeError, ValueError):
            sensitivity = None
        try:
            valence = max(-1.0, min(1.0, float(item.get("valence") or -0.5)))
        except (TypeError, ValueError):
            valence = -0.5
        emotional_events.add_event(
            person_id,
            str(item.get("category") or "other").strip().lower(),
            desc,
            valence=valence,
            sensitivity_decay_days=sensitivity,
            person_invited_topic=bool(item.get("person_invited_topic", True)),
            loss_subject=item.get("loss_subject"),
            loss_subject_kind=item.get("loss_subject_kind"),
            loss_subject_name=item.get("loss_subject_name"),
        )
        counts["stored"] += 1


def _apply_consolidated_corrections(
    person_id: int,
    person_name: Optional[str],
    items: list,
    terms: set[str],
    counts: dict[str, int],
) -> None:
    for item in items:
        if not isinstance(item, dict):
            counts["requires_review"] += 1
            continue
        if not _payload_allowed_against_terms(item, terms):
            counts["skipped"] += 1
            continue
        target = str(item.get("target") or item.get("type") or "fact").strip().lower()
        key = str(item.get("key") or item.get("name") or target).strip()
        value = str(item.get("value") or "").strip()
        if not value:
            counts["requires_review"] += 1
            continue
        if target == "identity" and key in {"name", "display_name", "nickname"}:
            if people_memory.rename_person(person_id, value):
                _refresh_world_state_person_name(person_id, value)
                facts_memory.apply_fact_correction(
                    person_id,
                    "name",
                    value,
                    category="identity",
                    importance=0.95,
                    decay_rate="permanent",
                )
                counts["updated"] += 1
            else:
                counts["requires_review"] += 1
            continue
        if target == "preference":
            _store_consolidated_preferences(person_id, [dict(item, source="corrected")], terms, counts)
            continue
        if target == "interest":
            _store_consolidated_interests(person_id, [dict(item, source="corrected")], terms, counts)
            continue
        facts_memory.apply_fact_correction(
            person_id,
            key,
            value,
            category=str(item.get("category") or "other").strip().lower(),
            importance=_clamp_memory_float(item.get("importance"), 0.9),
            decay_rate=_normalize_decay_rate(item.get("decay_rate"), "normal"),
        )
        counts["updated"] += 1


def _write_consolidated_memory(
    person_id: int,
    person_name: Optional[str],
    consolidated: dict,
    forgotten_terms: set[str],
) -> dict[str, int]:
    counts = {"stored": 0, "updated": 0, "skipped": 0, "requires_review": 0}
    _apply_consolidated_corrections(
        person_id,
        person_name,
        consolidated.get("corrections") or [],
        forgotten_terms,
        counts,
    )
    _store_consolidated_facts(person_id, consolidated.get("stable_facts") or [], forgotten_terms, counts)
    _store_consolidated_preferences(person_id, consolidated.get("preferences") or [], forgotten_terms, counts)
    _store_consolidated_interests(person_id, consolidated.get("interests") or [], forgotten_terms, counts)
    _store_consolidated_relationships(person_id, consolidated.get("relationships") or [], forgotten_terms, counts)
    _store_consolidated_events(person_id, consolidated.get("events") or [], forgotten_terms, counts)
    _store_consolidated_emotional_events(
        person_id,
        consolidated.get("emotional_events") or [],
        forgotten_terms,
        counts,
    )
    discarded = consolidated.get("discarded_noise") or []
    counts["skipped"] += len(discarded) if isinstance(discarded, list) else 0
    return counts


def _consolidate_session_memories(
    person_id: int,
    person_name: Optional[str],
    person_transcript: list[dict],
    forgotten_terms: set[str],
) -> bool:
    """Run bounded session-end memory consolidation. Returns True when completed."""
    if not bool(getattr(config, "MEMORY_CONSOLIDATION_ENABLED", True)):
        return False
    min_exchanges = int(getattr(config, "MEMORY_CONSOLIDATION_MIN_SESSION_EXCHANGES", 3))
    human_turns = [
        t for t in person_transcript
        if str(t.get("speaker") or "").lower() not in {"rex", "dj-r3x", "djr3x"}
    ]
    if _session_exchange_count < min_exchanges and len(human_turns) < min_exchanges:
        _log.info(
            "[memory_consolidation] skipped person_id=%s exchanges=%s human_turns=%s min=%s",
            person_id,
            _session_exchange_count,
            len(human_turns),
            min_exchanges,
        )
        return True

    result_box: dict[str, object] = {"completed": False}

    def _run() -> None:
        try:
            existing = _existing_memory_snapshot(person_id)
            consolidated = llm.consolidate_session_memories(
                person_id,
                person_transcript,
                person_name=person_name,
                existing_memories=existing,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
            counts = _write_consolidated_memory(
                person_id,
                person_name,
                consolidated,
                forgotten_terms,
            )
            result_box["completed"] = True
            result_box["counts"] = counts
            _log.info(
                "[memory_consolidation] person_id=%s stored=%d updated=%d "
                "skipped_as_noise=%d requires_review=%d",
                person_id,
                counts["stored"],
                counts["updated"],
                counts["skipped"],
                counts["requires_review"],
            )
        except Exception as exc:
            result_box["completed"] = True
            result_box["error"] = str(exc)
            _log.error(
                "[memory_consolidation] failed person_id=%s: %s",
                person_id,
                exc,
                exc_info=True,
            )

    worker = threading.Thread(
        target=_run,
        daemon=True,
        name=f"memory-consolidation-{person_id}",
    )
    worker.start()
    timeout = float(getattr(config, "MEMORY_CONSOLIDATION_TIMEOUT_SECS", 12.0))
    worker.join(timeout=max(1.0, timeout))
    if worker.is_alive():
        _log.warning(
            "[memory_consolidation] timed out after %.1fs for person_id=%s; continuing teardown",
            timeout,
            person_id,
        )
        return True
    return bool(result_box.get("completed"))


# ─────────────────────────────────────────────────────────────────────────────
# Session teardown
# ─────────────────────────────────────────────────────────────────────────────

def _end_session() -> None:
    """
    Called on ACTIVE → IDLE transition. Generates and persists a session summary,
    updates visit records and familiarity, then clears in-memory session state.
    """
    global _session_exchange_count, _identity_prompt_until, _awaiting_followup_event
    global _idle_outro_spoken
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation

    transcript = conv_memory.get_session_transcript()
    if not transcript:
        _session_exchange_count = 0
        _session_person_ids.clear()
        _session_person_turn_counts.clear()
        _session_forget_terms.clear()
        _session_router_control_topics.clear()
        _interest_idle_followups_spoken.clear()
        _low_memory_idle_questions_spoken.clear()
        _recent_memory_candidates.clear()
        _clear_anonymous_speaker_slots()
        _idle_outro_spoken = False
        try:
            topic_thread.clear()
        except Exception:
            pass
        try:
            # Persist the preoccupation (+ anti-repeat set) BEFORE wiping it, so it
            # carries across visits; the next session restores it via load_persisted().
            rex_pov.persist()
            rex_pov.clear()
        except Exception:
            pass
        try:
            dialogue_act.clear()
        except Exception:
            pass
        try:
            user_energy.clear()
        except Exception:
            pass
        try:
            question_budget.clear()
        except Exception:
            pass
        try:
            repair_moves.clear()
        except Exception:
            pass
        try:
            end_thread.clear()
        except Exception:
            pass
        try:
            turn_completion.clear()
        except Exception:
            pass
        _identity_prompt_until = 0.0
        _awaiting_followup_event = None
        _pending_introduction = None
        _pending_intro_followup = None
        _pending_intro_voice_capture = None
        _pending_common_first_name_identity = None
        _pending_common_first_name_introduction = None
        _pending_existing_common_first_name = None
        _pending_identity_match_confirmation = None
        _pending_prompted_name_confirmation = None
        _clear_pending_memory_wipe()
        _common_first_name_prompted_this_session.clear()
        try:
            consciousness.clear_response_wait()
            consciousness.clear_engagement()
        except Exception:
            pass
        return

    for person_id in list(_session_person_ids):
        try:
            person_row = people_memory.get_person(person_id)
            person_name = person_row.get("name") if person_row else None
            person_transcript = _filter_forgotten_transcript(transcript, person_id)

            if person_transcript:
                # Prefer the conversation arc — it's already computed each turn,
                # richer (mood/landed-flopped/open threads), and reusing it saves an
                # extra session-summary OpenAI call. Guards: only a 1:1 session (the
                # arc summarizes the whole conversation, not per-person), AND only
                # when the arc's recent-window actually covered the session — for a
                # long session the windowed arc would drop early content, so fall
                # back to the full-transcript generate_session_summary.
                arc_window = int(getattr(config, "CONVERSATION_ARC_CONTEXT_LINES", 12))
                arc_fields = (
                    topic_thread.arc_persistence_fields()
                    if len(_session_person_ids) == 1 and len(transcript) <= arc_window
                    else None
                )
                if arc_fields:
                    summary, emotion_tone, topics = arc_fields
                else:
                    summary = llm.generate_session_summary(person_id, person_transcript)
                    emotion_tone, topics = "neutral", ""
                if summary:
                    conv_memory.save_conversation(
                        person_id,
                        summary,
                        emotion_tone=emotion_tone,
                        topics=topics,
                    )

            forgotten_terms = _forgotten_terms_for_person(person_id)
            consolidation_completed = False
            if person_transcript:
                consolidation_completed = _consolidate_session_memories(
                    person_id,
                    person_name,
                    person_transcript,
                    forgotten_terms,
                )

            # Full-transcript extraction fallback — catches facts the
            # per-exchange rolling window may have missed when consolidation is
            # disabled, below threshold, or times out/fails before writing.
            if person_transcript and not consolidation_completed:
                try:
                    end_facts = llm.extract_facts(person_id, person_transcript, person_name=person_name)
                    saved = 0
                    for fact in end_facts:
                        if (
                            fact.get("key")
                            and fact.get("value")
                            and _extracted_memory_allowed(fact, person_id)
                        ):
                            facts_memory.add_fact(
                                person_id,
                                fact.get("category", "other"),
                                fact["key"],
                                fact["value"],
                                source="explicit",
                                confidence=0.95,
                            )
                            _record_recent_memory_candidate(
                                person_id,
                                kind="fact",
                                target=" ".join([
                                    str(fact.get("category") or ""),
                                    str(fact.get("key") or ""),
                                    str(fact.get("value") or ""),
                                ]),
                                label=str(fact.get("value") or fact.get("key") or "that fact"),
                            )
                            saved += 1
                    _log.info(
                        "[interaction] session-end facts extracted=%d saved=%d for person_id=%s (%s)",
                        len(end_facts), saved, person_id, person_name,
                    )
                except Exception as exc:
                    _log.error("session-end fact extraction error for person_id=%s: %s", person_id, exc)

                try:
                    end_preferences = llm.extract_preferences(
                        person_id,
                        person_transcript,
                        person_name=person_name,
                    )
                    saved_preferences = 0
                    for pref in end_preferences:
                        if (
                            pref.get("domain")
                            and pref.get("preference_type")
                            and pref.get("key")
                            and _extracted_memory_allowed(pref, person_id)
                        ):
                            if pref.get("preference_type") == "boundary":
                                pref["importance"] = max(
                                    float(pref.get("importance") or 0.0),
                                    0.95,
                                )
                            preferences_memory.upsert_preference(
                                person_id,
                                pref["domain"],
                                pref["preference_type"],
                                pref["key"],
                                pref.get("value") or "",
                                confidence=float(pref.get("confidence") or 1.0),
                                importance=float(pref.get("importance") or 0.5),
                                source=pref.get("source") or "explicit",
                            )
                            _record_recent_memory_candidate(
                                person_id,
                                kind="preference",
                                target=" ".join([
                                    str(pref.get("domain") or ""),
                                    str(pref.get("preference_type") or ""),
                                    str(pref.get("key") or ""),
                                    str(pref.get("value") or ""),
                                ]),
                                label=str(pref.get("value") or pref.get("key") or "that preference"),
                            )
                            saved_preferences += 1
                    _log.info(
                        "[interaction] session-end preferences extracted=%d saved=%d for person_id=%s (%s)",
                        len(end_preferences), saved_preferences, person_id, person_name,
                    )
                except Exception as exc:
                    _log.error("session-end preference extraction error for person_id=%s: %s", person_id, exc)

                try:
                    end_interests = llm.extract_interests(
                        person_id,
                        person_transcript,
                        person_name=person_name,
                    )
                    saved_interests = 0
                    for interest in end_interests:
                        if (
                            interest.get("name")
                            and _extracted_memory_allowed(interest, person_id)
                        ):
                            interests_memory.upsert_interest(
                                person_id,
                                interest["name"],
                                interest.get("category") or "hobby",
                                interest.get("interest_strength") or "medium",
                                confidence=float(interest.get("confidence") or 1.0),
                                source=interest.get("source") or "explicit",
                                notes=interest.get("notes") or "",
                                associated_people=interest.get("associated_people") or "",
                                associated_stories=interest.get("associated_stories") or "",
                            )
                            _record_recent_memory_candidate(
                                person_id,
                                kind="interest",
                                target=" ".join([
                                    str(interest.get("name") or ""),
                                    str(interest.get("category") or ""),
                                    str(interest.get("notes") or ""),
                                ]),
                                label=str(interest.get("name") or "that interest"),
                            )
                            saved_interests += 1
                    _log.info(
                        "[interaction] session-end interests extracted=%d saved=%d for person_id=%s (%s)",
                        len(end_interests), saved_interests, person_id, person_name,
                    )
                except Exception as exc:
                    _log.error("session-end interest extraction error for person_id=%s: %s", person_id, exc)

            # update_visit increments visit_count, last_seen, and applies the
            # return_visit familiarity increment defined in config.
            people_memory.update_visit(person_id)

            # Extra familiarity increment for a sufficiently long conversation
            if _session_exchange_count >= config.LONG_CONVERSATION_MIN_EXCHANGES:
                people_memory.update_familiarity(
                    person_id,
                    config.FAMILIARITY_INCREMENTS.get("long_conversation", 0.02),
                )
        except Exception as exc:
            _log.error("session end error for person_id=%s: %s", person_id, exc)

    conv_memory.clear_transcript()
    _session_forget_terms.clear()
    _session_router_control_topics.clear()
    _interest_idle_followups_spoken.clear()
    _low_memory_idle_questions_spoken.clear()
    _recent_memory_candidates.clear()
    _clear_anonymous_speaker_slots()
    _idle_outro_spoken = False
    try:
        topic_thread.clear()
    except Exception:
        pass
    try:
        dialogue_act.clear()
    except Exception:
        pass
    try:
        user_energy.clear()
    except Exception:
        pass
    try:
        question_budget.clear()
    except Exception:
        pass
    try:
        repair_moves.clear()
    except Exception:
        pass
    try:
        end_thread.clear()
    except Exception:
        pass
    try:
        turn_completion.clear()
    except Exception:
        pass
    _session_exchange_count = 0
    _session_person_ids.clear()
    _session_person_turn_counts.clear()
    _identity_prompt_until = 0.0
    _awaiting_followup_event = None
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
    _pending_common_first_name_identity = None
    _pending_common_first_name_introduction = None
    _pending_existing_common_first_name = None
    _pending_identity_match_confirmation = None
    _pending_prompted_name_confirmation = None
    _clear_pending_memory_wipe()
    _common_first_name_prompted_this_session.clear()
    _voice_refreshed_this_session.clear()
    _face_reveal_declined.clear()
    _grief_flow_state.clear()
    try:
        consciousness.clear_response_wait()
        consciousness.clear_engagement()
    except Exception:
        pass
    _log.info("[interaction] session ended — summary saved, transcript cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Curiosity routine
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic patterns for detecting when Rex's free-form response has already
# raised a QUESTION_POOL topic. Used so the curiosity routine doesn't follow
# up with a near-duplicate question on a later turn.
_POOL_TOPIC_PATTERNS: dict[str, list[re.Pattern]] = {
    "hometown": [re.compile(r"\bwhere (?:are|were|you)\s+(?:are\s+)?(?:you\s+)?from\b", re.I),
                 re.compile(r"\bhometown\b", re.I),
                 re.compile(r"\bwhere.*\bborn\b", re.I)],
    "job": [re.compile(r"\bwhat do you do\b", re.I),
            re.compile(r"\bprofessionally\b", re.I),
            re.compile(r"\bfor a living\b", re.I),
            re.compile(r"\boccupation\b", re.I),
            re.compile(r"\bwhat.*\b(?:job|career)\b", re.I)],
    "favorite_movie": [re.compile(r"\bfavorite\s+(?:movie|film)\b", re.I)],
    "favorite_music": [re.compile(r"\bfavorite\s+(?:music|band|artist|song)\b", re.I),
                       re.compile(r"\bwhat\s+(?:kind\s+of\s+)?music\b", re.I)],
    "how_found_rex": [re.compile(r"\bhow\s+did\s+you\s+(?:find|end\s+up)\b", re.I)],
    "hobbies": [re.compile(r"\bhobb(?:y|ies)\b", re.I),
                re.compile(r"\bfor\s+fun\b", re.I),
                re.compile(r"\bfree\s+time\b", re.I)],
    "travel": [re.compile(r"\binteresting\s+place\b", re.I),
               re.compile(r"\bplaces?\s+you'?ve\s+been\b", re.I)],
    "proudest_moment": [re.compile(r"\bproud(?:est)?\s+of\b", re.I)],
    "biggest_challenge": [re.compile(r"\bhardest\s+thing\b", re.I),
                          re.compile(r"\bbiggest\s+challenge\b", re.I)],
    "obsession": [re.compile(r"\bobsess(?:ed|ion)\b", re.I)],
    "relationships": [re.compile(r"\bmost\s+important\s+person\b", re.I)],
    "values": [re.compile(r"\bbelieve\s+in\b", re.I)],
    "fears": [re.compile(r"\bkeeps?\s+you\s+up\b", re.I),
              re.compile(r"\bafraid\s+of\b", re.I)],
    "life_changing": [re.compile(r"\bchanged\s+you\b", re.I),
                      re.compile(r"\blife[- ]changing\b", re.I)],
    "regret": [re.compile(r"\bdo\s+differently\b", re.I),
               re.compile(r"\bregret\b", re.I)],
    "meaning_of_life": [re.compile(r"\bpoint\s+of\s+all\s+this\b", re.I),
                        re.compile(r"\bmeaning\s+of\s+life\b", re.I)],
    "free_will": [re.compile(r"\bfree\s+will\b", re.I),
                  re.compile(r"\breal\s+choices\b", re.I)],
    "consciousness": [re.compile(r"\bnot\s+be\s+conscious\b", re.I)],
    "good_life": [re.compile(r"\blife\s+worth\s+living\b", re.I)],
}

def _question_blocked_by_boundary(person_id: Optional[int], question: dict) -> bool:
    return profile_questions.question_blocked_by_boundary(person_id, question)


def _record_pool_topics_in_response(response_text: str, person_id: int) -> None:
    """Mark any QUESTION_POOL topics raised by Rex's response as already-asked,
    so the curiosity routine won't re-ask them on a later turn."""
    try:
        asked = rel_memory.get_asked_question_keys(person_id)
    except Exception as exc:
        _log.debug("record_pool_topics: get_asked error: %s", exc)
        return

    pool_by_key = {q["key"]: q for q in config.QUESTION_POOL}
    for key, patterns in _POOL_TOPIC_PATTERNS.items():
        if key in asked:
            continue
        if not any(p.search(response_text) for p in patterns):
            continue
        pool_q = pool_by_key.get(key)
        if pool_q is None:
            continue
        try:
            rel_memory.save_question_asked(
                person_id,
                key,
                response_text.strip(),
                pool_q.get("depth", 1),
            )
            _log.info(
                "[interaction] pool topic %r marked asked from Rex's response", key
            )
        except Exception as exc:
            _log.debug("record_pool_topics: save_qa error: %s", exc)


def _grief_flow_active(person_id: Optional[int]) -> bool:
    """True when this person has an unfinished grief flow within TTL."""
    if person_id is None:
        return False
    state = _grief_flow_state.get(person_id)
    if not state:
        return False
    if (time.monotonic() - state["started_at"]) > _GRIEF_FLOW_TTL_SECS:
        _grief_flow_state.pop(person_id, None)
        return False
    return True


def _grief_flow_clear(person_id: Optional[int]) -> None:
    if person_id is not None:
        _grief_flow_state.pop(person_id, None)


def _force_grief_empathy(person_id: Optional[int], reason: str = "structured grief flow") -> None:
    if person_id is None:
        return
    try:
        empathy.force_mode(
            person_id,
            "listen",
            affect="sad",
            needs="vent",
            sensitivity="heavy",
            invitation=True,
            confidence=1.0,
            reason=reason,
        )
    except Exception as exc:
        _log.debug("force grief empathy failed: %s", exc)


_AFFIRM_PAT = re.compile(
    r"\b(yes|yeah|yep|sure|okay|ok|please|alright|i guess|fine|"
    r"i'?d like to|i would|let'?s|talk about (it|him|her|them))\b",
    re.IGNORECASE,
)
_DECLINE_PAT = re.compile(
    r"\b(no|nope|not really|don'?t want to|rather not|maybe later|"
    r"not now|skip|drop it|leave it|let'?s not|"
    # "muzzle" is Rex's own word for the keep-the-music-off option, so a reply
    # like "let's keep it muzzled for now" must read as a decline even though it
    # opens with the affirmative-sounding "let's".
    r"muzzle[ds]?|keep it (?:off|down|quiet|muzzled|silent)|"
    r"keep (?:the )?(?:jukebox|music) (?:off|down|quiet|muzzled|silent)|"
    r"hold off|no music|keep quiet|stay quiet)\b",
    re.IGNORECASE,
)
_MUSIC_PLAY_REQUEST_PAT = re.compile(
    r"\b(play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue|turn\s+on)\b",
    re.IGNORECASE,
)
_EMOTIONAL_BOUNDARY_PAT = re.compile(
    r"\b("
    r"i'?d rather not|i would rather not|rather not talk|"
    r"don'?t want to talk|do not want to talk|"
    r"i told you i didn'?t want to talk about (?:this|that|it)|"
    r"i told you i did not want to talk about (?:this|that|it)|"
    r"let'?s change (the )?subject|change (the )?subject|"
    r"talk about something else|something else please|"
    r"don'?t ask me (about (that|it) )?again|do not ask me (about (that|it) )?again|"
    r"please don'?t ask|please do not ask|"
    r"stop asking|stop bringing (that|it) up|"
    r"can we not|not talk about (that|it)|"
    r"don'?t talk about (this|that|it)( anymore| again)?|"
    r"do not talk about (this|that|it)( anymore| again)?|"
    r"stop talking about (this|that|it)|"
    r"don'?t bring (this|that|it) up( anymore| again)?|"
    r"do not bring (this|that|it) up( anymore| again)?|"
    r"don'?t mention (this|that|it)( anymore| again)?|"
    r"do not mention (this|that|it)( anymore| again)?|"
    r"forget about (this|that|it)|"
    r"drop it|leave it alone|no more check-?ins?"
    r")\b",
    re.IGNORECASE,
)
_EMOTIONAL_TOPIC_BOUNDARY_PAT = re.compile(
    r"\b("
    r"don'?t talk about (this|that|it)( anymore| again)?|"
    r"do not talk about (this|that|it)( anymore| again)?|"
    r"stop talking about (this|that|it)|"
    r"don'?t bring (this|that|it) up( anymore| again)?|"
    r"do not bring (this|that|it) up( anymore| again)?|"
    r"don'?t mention (this|that|it)( anymore| again)?|"
    r"do not mention (this|that|it)( anymore| again)?|"
    r"forget about (this|that|it)"
    r")\b",
    re.IGNORECASE,
)


def _classify_consent(text: str) -> Optional[bool]:
    """Heuristic yes/no classifier for the consent step. None when ambiguous."""
    if not text:
        return None
    if _DECLINE_PAT.search(text):
        return False
    if _AFFIRM_PAT.search(text):
        return True
    return None


def _handle_emotional_checkin_boundary(
    person_id: Optional[int],
    text: str,
) -> Optional[str]:
    """
    Honor consent boundaries after Rex checks in on a recent hard event.

    If the person asks not to discuss it, mute proactive check-ins for that
    specific emotional event. The memory remains stored; Rex just stops leading
    with it unless the person brings it up again.
    """
    if person_id is None or not text:
        return None
    if not _EMOTIONAL_BOUNDARY_PAT.search(text):
        return None

    reason = text.strip()[:240]
    topic_boundary = _EMOTIONAL_TOPIC_BOUNDARY_PAT.search(text) is not None
    released_checkin_hold = False
    try:
        muted = emotional_events.mute_recent_checkin_for_person(
            person_id,
            reason=reason,
            window_minutes=int(getattr(config, "EMOTIONAL_CHECKIN_BOUNDARY_WINDOW_MINUTES", 20)),
        )
        if muted is None and (_grief_flow_active(person_id) or topic_boundary):
            muted = emotional_events.mute_latest_active_negative_for_person(
                person_id,
                reason=reason,
            )
        if muted is not None or topic_boundary:
            try:
                released_checkin_hold = consciousness.note_emotional_checkin_boundary(person_id)
            except Exception as exc:
                _log.debug("release emotional check-in visual hold failed: %s", exc)
        if muted is None:
            if not released_checkin_hold:
                return None
            muted = {"id": None, "category": "recent_checkin"}
        _grief_flow_clear(person_id)
        _log.info(
            "[empathy] muted proactive emotional check-ins for person_id=%s "
            "event_id=%s category=%s due to boundary reply: %r",
            person_id,
            muted.get("id"),
            muted.get("category"),
            text,
        )
        try:
            empathy.force_mode(
                person_id,
                "brief",
                affect="neutral",
                needs="boundary",
                sensitivity="heavy",
                invitation=False,
                confidence=1.0,
                reason="person set emotional check-in boundary",
            )
        except Exception:
            pass
        _apply_topic_boundary_side_effects(person_id, text)
        return "Understood. I won't bring it up again unless you do."
    except Exception as exc:
        _log.debug("emotional check-in boundary handler failed: %s", exc)
        return None


def _handle_router_emotional_boundary(
    person_id: Optional[int],
    text: str,
    *,
    topic_hint: Optional[str] = None,
) -> Optional[str]:
    """Fallback boundary action when the action router catches user intent."""
    if person_id is None:
        return None
    try:
        released = False
        muted = emotional_events.mute_latest_active_negative_for_person(
            person_id,
            reason=text.strip()[:240],
        )
        try:
            released = consciousness.note_emotional_checkin_boundary(person_id)
        except Exception as exc:
            _log.debug("router boundary release hold failed: %s", exc)

        topic = (topic_hint or "").strip() or _boundary_fallback_topic() or "current topic"
        applied = boundary_memory.apply_detected_boundary(
            person_id,
            {
                "action": "add",
                "behavior": "mention",
                "topic": topic,
                "description": f"Do not bring up {topic} unless the person does.",
                "source_text": text.strip(),
            },
        )
        _grief_flow_clear(person_id)
        try:
            empathy.force_mode(
                person_id,
                "brief",
                affect="neutral",
                needs="boundary",
                sensitivity="heavy",
                invitation=False,
                confidence=1.0,
                reason="action router detected emotional boundary",
            )
        except Exception:
            pass
        _log.info(
            "[action_router] executed emotional.boundary person_id=%s muted_event=%s "
            "saved_boundary=%s released_hold=%s text=%r",
            person_id,
            (muted or {}).get("id"),
            (applied or {}).get("id"),
            released,
            text,
        )
        return "Got it. I won't bring it up again unless you do."
    except Exception as exc:
        _log.debug("router emotional boundary handler failed: %s", exc)
        return None


def _router_arg_text(
    decision: Optional[action_router.ActionDecision],
    *keys: str,
) -> str:
    if decision is None:
        return ""
    for key in keys:
        value = decision.args.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _visible_known_name_for_intent() -> Optional[str]:
    try:
        for p in world_state.get("people") or []:
            if p.get("person_db_id") is not None and p.get("face_id"):
                return str(p["face_id"])
    except Exception:
        pass
    return None


def _router_system_command(text: str, decision: action_router.ActionDecision) -> Optional[str]:
    mode = _router_arg_text(decision, "mode", "state", "target")
    haystack = f"{text} {mode}".lower()
    if any(word in haystack for word in ("shutdown", "shut down", "power off", "turn off")):
        return "shutdown"
    if any(word in haystack for word in ("quiet", "mute", "silent")):
        return "quiet_mode"
    if "wake" in haystack:
        return "wake_up"
    if not command_parser.is_standalone_sleep_command(text):
        return None
    return "sleep"


def _play_performance_body_beat(beat: str) -> None:
    from sequences import animations
    animations.play_body_beat(beat)


def _play_event_body_beat(event: str, **context) -> Optional[str]:
    beat = performance_output.execute_body_beat_event(
        event,
        play_body_beat=_play_performance_body_beat,
        **context,
    )
    if not beat:
        _log.debug("[performance] no body beat played for event=%s context=%s", event, context)
    return beat


def _set_body_mood(mood: str, *, source: str = "") -> None:
    """Set Rex's sustained body mood (drives his posture/visor between turns). Gated +
    failure-safe; no-op if the body-mood feature is disabled."""
    try:
        body_mood.set_mood(mood, source=source)
    except Exception as exc:
        _log.debug("[interaction] set_body_mood(%s) failed: %s", mood, exc)


_PLAN_ROUTER_ACTIONS = {
    "humor.tell_joke",
    "humor.roast",
    "humor.free_bit",
    "performance.dj_bit",
    "performance.body_beat",
    "performance.mood_pose",
}


def _handle_router_performance_action(
    decision: action_router.ActionDecision,
    text: str,
    person_id: Optional[int],
    router_audit: Optional[_RouterDecisionAudit] = None,
) -> Optional[str]:
    if decision.action not in _PLAN_ROUTER_ACTIONS:
        _router_audit_note_result(
            router_audit,
            handler_error="unsupported_performance_action",
        )
        return None
    plan = performance_plan.plan_for_action(
        decision.action,
        user_text=text,
        args=decision.args,
    )
    if plan is None:
        _router_audit_note_result(
            router_audit,
            handler_error="missing_performance_plan",
        )
        return None
    output = performance_output.execute_plan(
        plan,
        generate_text=lambda prompt: llm.get_response(prompt, person_id),
        speak_text=_speak_blocking,
        play_body_beat=_play_performance_body_beat,
        clean_text=llm.clean_response_text,
    )
    if output.generation_failed:
        _log.debug("performance action generation failed; used fallback text")
    if output.body_beat_failed:
        _log.debug("performance action body beat failed; speech still delivered")
    result_errors = []
    if output.generation_failed:
        result_errors.append("generation_failed")
    if output.body_beat_failed:
        result_errors.append("body_beat_failed")
    _router_audit_note_result(
        router_audit,
        completed=output.completed,
        handler_error=",".join(result_errors) if result_errors else None,
        spoken_text=output.text,
    )
    _log.info(
        "[action_router] executed %s person_id=%s delivery=%s memory_policy=%s text=%r",
        output.action,
        person_id,
        output.delivery_style,
        output.memory_policy,
        text,
    )
    return output.text


def _handle_router_character_preference(
    decision: action_router.ActionDecision,
    text: str,
    router_audit: Optional[_RouterDecisionAudit] = None,
) -> str:
    """Answer a question about Rex's own tastes with a stable line and motion."""
    try:
        reply = rex_preferences.answer_preference_query(text, decision.args)
    except Exception as exc:
        _log.debug("Rex preference response failed: %s", exc)
        reply = rex_preferences.PreferenceReply(
            text="Mmm. Preference circuits are recalibrating.",
            emotion="curious",
            body_beat="thinking_tilt",
            stance="unknown",
            topic="unknown",
        )

    body_beat_failed = False
    if reply.body_beat:
        try:
            _play_performance_body_beat(reply.body_beat)
        except Exception as exc:
            body_beat_failed = True
            _log.debug("Rex preference body beat failed: %s", exc)

    completed = _speak_blocking(
        reply.text,
        emotion=reply.emotion,
        pre_beat_ms=reply.pre_beat_ms,
        post_beat_ms_override=reply.post_beat_ms,
    )
    _router_audit_note_result(
        router_audit,
        completed=completed,
        handler_error="body_beat_failed" if body_beat_failed else None,
        spoken_text=reply.text,
    )
    _log.info(
        "[character] Rex preference reply topic=%r stance=%s beat=%s text=%r",
        reply.topic,
        reply.stance,
        reply.body_beat,
        reply.text,
    )
    return reply.text


def _router_repair_move(
    text: str,
    decision: action_router.ActionDecision,
) -> dict:
    repair = repair_moves.detect(text)
    if repair is not None:
        return repair
    repair_kind = (
        _router_arg_text(decision, "repair_kind", "kind", "type")
        or "misunderstood"
    )
    correction = _router_arg_text(decision, "correction", "corrected_text", "clarification")
    return {
        "kind": repair_kind,
        "severity": "medium",
        "correction": correction,
        "user_text": text,
    }


def _handle_router_identity_name_correction(
    text: str,
    decision: action_router.ActionDecision,
    person_id: Optional[int],
    person_name: Optional[str],
) -> str:
    """Execute a current-speaker name correction, or ask for the missing name."""
    response = _handle_name_update_request(text, person_id, person_name)
    if response:
        return response

    new_name = _normalize_name(
        _router_arg_text(decision, "name", "new_name", "person_name")
    )
    if new_name:
        response = _handle_name_update_request(
            f"call me {new_name}",
            person_id,
            person_name,
        )
        if response:
            return response

    resp = repair_moves.add_better_luck_line(
        "I caught the identity correction, but I need the right name."
    )
    _speak_blocking(resp, emotion="neutral")
    return resp


def _handle_router_takeover_action(
    decision: Optional[action_router.ActionDecision],
    text: str,
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    raw_best_score: float,
    router_context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
    router_audit: Optional[_RouterDecisionAudit] = None,
) -> Optional[str]:
    """Execute router-owned actions that map to stable local handlers."""
    if not _router_decision_executable(
        decision,
        text=text,
        context=router_context,
        dialogue_decision=dialogue_decision,
    ):
        return None

    action = decision.action
    if action == "conversation.reply":
        return None

    if action == "conversation.repair":
        repair = _router_repair_move(text, decision)
        _log.info(
            "[action_router] executing conversation.repair kind=%s person_id=%s text=%r",
            repair.get("kind"),
            person_id,
            text,
        )
        return _generate_repair_response(person_id, text, repair)

    if action == "identity.name_correction":
        _log.info(
            "[action_router] executing identity.name_correction person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_router_identity_name_correction(
            text,
            decision,
            person_id,
            person_name,
        )

    if action in _PLAN_ROUTER_ACTIONS:
        return _handle_router_performance_action(
            decision,
            text,
            person_id,
            router_audit=router_audit,
        )

    if action == "character.preference_query":
        return _handle_router_character_preference(
            decision,
            text,
            router_audit=router_audit,
        )

    if action == "memory.forget_specific":
        target = _router_arg_text(decision, "target", "topic", "memory")
        if not target:
            target = forgetting.extract_specific_forget_target(text)
        if not target:
            return None
        _log.info(
            "[action_router] executing memory.forget_specific person_id=%s target=%r text=%r",
            person_id,
            target,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch(
                "forget_specific",
                "action_router",
                {"target": target},
            ),
            person_id,
            person_name,
            text,
        )

    if action == "memory.recent_discard":
        _log.info(
            "[action_router] executing memory.recent_discard person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_memory_boundary_command(person_id)

    if action == "identity.who_is_speaking":
        _log.info(
            "[action_router] executing identity.who_is_speaking person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent(
            "query_who_is_speaking",
            text,
            person_id,
            raw_best_id=raw_best_id,
            raw_best_name=raw_best_name,
            raw_best_score=raw_best_score,
            visible_known_name=_visible_known_name_for_intent(),
        )

    if action == "game.start":
        game = _router_arg_text(decision, "game", "game_name", "target")
        _log.info(
            "[action_router] executing game.start person_id=%s game=%r text=%r",
            person_id,
            game,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch(
                "start_game",
                "action_router",
                {"game": game or text},
            ),
            person_id,
            person_name,
            text,
        )

    if action == "game.stop":
        _log.info(
            "[action_router] executing game.stop person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch("stop_game", "action_router", {}),
            person_id,
            person_name,
            text,
        )

    if action == "game.answer":
        try:
            from features import games as games_mod
            if not games_mod.is_active():
                return None
            _log.info(
                "[action_router] executing game.answer person_id=%s text=%r",
                person_id,
                text,
            )
            resp = games_mod.handle_input(text, person_id)
            completed = _speak_blocking(resp)
            if completed:
                games_mod.on_response_spoken()
                after_audio = games_mod.consume_pending_audio_after_response()
                if after_audio and not _interrupted.is_set():
                    speech_queue.enqueue_audio_file(
                        after_audio,
                        priority=1,
                        tag="game:after_audio",
                    )
            return resp
        except Exception as exc:
            _log.debug("router game.answer failed: %s", exc)
            _router_audit_note_result(
                router_audit,
                handler_error=f"game_answer_failed:{type(exc).__name__}",
            )
            return None

    if action == "music.play":
        _log.info(
            "[action_router] executing music.play person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("play_music", text, person_id)

    if action == "music.stop":
        _log.info(
            "[action_router] executing music.stop person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch("dj_stop", "action_router", {}),
            person_id,
            person_name,
            text,
        )

    if action == "music.skip":
        _log.info(
            "[action_router] executing music.skip person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch("dj_skip", "action_router", {}),
            person_id,
            person_name,
            text,
        )

    if action == "music.options":
        _log.info(
            "[action_router] executing music.options person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_music_options", text, person_id)

    if action == "vision.describe_scene":
        _log.info(
            "[action_router] executing vision.describe_scene person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_what_do_you_see", text, person_id)

    if action == "memory.query":
        _log.info(
            "[action_router] executing memory.query person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent(
            "query_memory",
            text,
            person_id,
            raw_best_id=raw_best_id,
            raw_best_name=raw_best_name,
            raw_best_score=raw_best_score,
            visible_known_name=_visible_known_name_for_intent(),
        )

    if action == "time.query":
        _log.info(
            "[action_router] executing time.query person_id=%s text=%r",
            person_id,
            text,
        )
        if _looks_like_date_query(text):
            return _handle_classified_intent("query_date", text, person_id)
        return _handle_classified_intent("query_time", text, person_id)

    if action == "date.query":
        _log.info(
            "[action_router] executing date.query person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_date", text, person_id)

    if action == "weather.query":
        _log.info(
            "[action_router] executing weather.query person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_weather", text, person_id)

    if action == "status.capabilities":
        _log.info(
            "[action_router] executing status.capabilities person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_capabilities", text, person_id)

    if action == "status.uptime":
        _log.info(
            "[action_router] executing status.uptime person_id=%s text=%r",
            person_id,
            text,
        )
        return _handle_classified_intent("query_uptime", text, person_id)

    if action == "system.shutdown":
        if not command_parser.is_standalone_shutdown_command(text):
            _log.info(
                "[action_router] ignored non-standalone system.shutdown candidate person_id=%s text=%r",
                person_id,
                text,
            )
            return None
        _log.info(
            "[action_router] executing system.shutdown person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch("shutdown", "action_router", {}),
            person_id,
            person_name,
            text,
        )

    if action == "system.sleep":
        key = _router_system_command(text, decision)
        if key is None:
            _log.info(
                "[action_router] ignored non-standalone system.sleep candidate person_id=%s text=%r",
                person_id,
                text,
            )
            return None
        _log.info(
            "[action_router] executing system.sleep mapped_key=%s person_id=%s text=%r",
            key,
            person_id,
            text,
        )
        return _execute_command(
            command_parser.CommandMatch(key, "action_router", {}),
            person_id,
            person_name,
            text,
        )

    return None


def _handle_fast_local_takeover(
    text: str,
    *,
    person_id: Optional[int],
    person_name: Optional[str],
    router_context: Optional[dict[str, Any]] = None,
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None,
    router_audit: Optional[_RouterDecisionAudit] = None,
) -> Optional[str]:
    """Handle obvious local control commands before the blocking router call."""
    humor_decision = action_router.classify_explicit_humor(text)
    if humor_decision is not None:
        _router_audit_note_decision(router_audit, humor_decision)
    if _router_decision_executable(humor_decision):
        return _handle_router_performance_action(
            humor_decision,
            text,
            person_id,
            router_audit=router_audit,
        )
    performance_decision = action_router.classify_explicit_performance(text)
    if performance_decision is not None:
        _router_audit_note_decision(router_audit, performance_decision)
    if _router_decision_executable(performance_decision):
        return _handle_router_performance_action(
            performance_decision,
            text,
            person_id,
            router_audit=router_audit,
        )

    directed_followup = _pending_directed_search_reply(
        text,
        person_id=person_id,
        person_name=person_name,
    )
    if directed_followup is not None:
        _router_audit_note_fast_local_action(
            router_audit,
            "vision.directed_search",
            reason="pending directed-look search follow-up",
        )
        return directed_followup

    try:
        match = command_parser.parse(text)
    except Exception as exc:
        _log.debug("fast local command parse failed: %s", exc)
        match = None
    if match is None:
        return None
    _router_audit_note_legacy_command(router_audit, match)
    legacy_block_reason = _legacy_command_execution_block_reason(
        match,
        text=text,
        context=router_context,
        dialogue_decision=dialogue_decision,
    )
    if legacy_block_reason:
        _log.info(
            "[turn_policy] blocked fast legacy command=%s reason=%s text=%r",
            match.command_key,
            legacy_block_reason,
            text,
        )
        _router_audit_clear_legacy_command(router_audit)
        return None

    key = match.command_key
    if key == "directed_look":
        _log.info(
            "[action_router] fast_lane legacy_command=directed_look "
            "person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(match, person_id, person_name, text)

    if key == "forget_specific":
        target = str((match.args or {}).get("target") or "").strip()
        if not target:
            return None
        _router_audit_note_fast_local_action(
            router_audit,
            "memory.forget_specific",
            args={"target": target},
            reason="legacy fast local forget command",
        )
        _log.info(
            "[action_router] fast_lane action=memory.forget_specific "
            "person_id=%s target=%r text=%r",
            person_id,
            target,
            text,
        )
        return _execute_command(match, person_id, person_name, text)

    if key == "memory_boundary":
        _router_audit_note_fast_local_action(
            router_audit,
            "memory.recent_discard",
            args={"scope": "recent"},
            reason="legacy fast local recent-memory discard command",
        )
        _log.info(
            "[action_router] fast_lane action=memory.recent_discard "
            "person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_memory_boundary_command(person_id)

    if key == "rename_me":
        _router_audit_note_fast_local_action(
            router_audit,
            "identity.name_correction",
            args=match.args or {},
            reason="legacy fast local current-speaker name correction",
        )
        _log.info(
            "[action_router] fast_lane action=identity.name_correction "
            "person_id=%s text=%r",
            person_id,
            text,
        )
        return _execute_command(match, person_id, person_name, text)

    if key == "stop_game":
        _router_audit_note_fast_local_action(
            router_audit,
            "game.stop",
            reason="legacy fast local stop-game command",
        )
        try:
            from features import games as games_mod
            resp = games_mod.stop_game_fast(person_id)
        except Exception as exc:
            _log.debug("fast game stop failed: %s", exc)
            return None
        _log.info(
            "[action_router] fast_lane action=game.stop person_id=%s text=%r",
            person_id,
            text,
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    if key == "dj_stop":
        _router_audit_note_fast_local_action(
            router_audit,
            "music.stop",
            reason="legacy fast local stop-music command",
        )
        try:
            from features import dj as dj_mod
            was_playing = bool(dj_mod.is_playing())
            dj_mod.stop()
        except Exception as exc:
            _log.debug("fast music stop failed: %s", exc)
            return None
        resp = "Music stopped." if was_playing else "No music is playing."
        _log.info(
            "[action_router] fast_lane action=music.stop person_id=%s "
            "active_music=%s text=%r",
            person_id,
            was_playing,
            text,
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    if key in {"time_query", "date_query"}:
        action = "time.query" if key == "time_query" else "date.query"
        _router_audit_note_fast_local_action(
            router_audit,
            action,
            reason=f"legacy fast local {key} command",
        )
        resp = (
            _format_current_time_response()
            if key == "time_query"
            else _format_current_date_response()
        )
        _log.info(
            "[action_router] fast_lane action=%s person_id=%s text=%r",
            action,
            person_id,
            text,
        )
        _speak_blocking(resp, emotion="neutral")
        return resp

    return None


_MEMORY_HINT_PAT = re.compile(
    r"\b(remember|told me|you said|plan|planned|schedule|trip|congrats|"
    r"how'?s|how is|how did|did .* go|survive|ready for)\b",
    re.IGNORECASE,
)
_CANCEL_CONTEXT_STOPWORDS = {
    "any", "anymore", "going", "gonna", "not", "now", "more", "the", "this",
    "that", "there", "where", "what", "when", "with", "you", "your", "i'm",
    "im",
}


def _recent_rex_memory_hint() -> str:
    """Return the last Rex line when it looks like a memory callback."""
    try:
        transcript = conv_memory.get_session_transcript()
    except Exception:
        return ""
    for turn in reversed(transcript[-8:]):
        if (turn.get("speaker") or "").lower() != "rex":
            continue
        text = (turn.get("text") or "").strip()
        if text and _MEMORY_HINT_PAT.search(text):
            return text
        return ""
    try:
        text = consciousness.get_last_memory_hint()
    except Exception:
        text = ""
    if text and _MEMORY_HINT_PAT.search(text):
        return text
    return ""


def _has_specific_memory_token(text: str) -> bool:
    tokens = [
        t.lower().strip("'")
        for t in re.findall(r"[A-Za-z0-9']+", text or "")
    ]
    return any(len(t) >= 3 and t not in _CANCEL_CONTEXT_STOPWORDS for t in tokens)


def _cancel_stale_event_memory(
    person_id: Optional[int],
    text: str,
    *,
    event_hint: Optional[dict] = None,
) -> list[str]:
    """
    Apply corrections like "I'm not going anymore" to planned/social memories.

    Returns short labels for canceled or muted memories. Empty means the text
    did not look like a cancellation or nothing matched safely.
    """
    if person_id is None:
        try:
            person_id = consciousness.get_last_memory_hint_target()
        except Exception:
            person_id = None
    if person_id is None or not events_memory.looks_like_cancellation(text):
        return []

    labels: list[str] = []
    event_hint_name = ""
    effective_event_hint = event_hint
    if event_hint:
        event_hint_name = (event_hint.get("event_name") or "").strip()
    elif not _has_specific_memory_token(text):  # generic "I'm not going anymore"
        event_hint_name = _recent_rex_memory_hint()
        if event_hint_name:
            effective_event_hint = {"event_name": event_hint_name}

    try:
        canceled = events_memory.cancel_matching_events(
            int(person_id),
            text,
            event_hint=effective_event_hint,
        )
    except Exception as exc:
        _log.debug("event cancellation matching failed: %s", exc)
        canceled = []

    if canceled:
        try:
            ids = {int(ev["id"]) for ev in canceled if ev.get("id") is not None}
            consciousness.drop_pending_followups(int(person_id), ids)
        except Exception as exc:
            _log.debug("drop canceled pending followups failed: %s", exc)
        for ev in canceled:
            name = (ev.get("event_name") or "").strip()
            if name:
                labels.append(name)

    try:
        muted = emotional_events.mute_matching_positive_events(
            int(person_id),
            text,
            reason=text,
            event_hint=event_hint_name,
        )
    except Exception as exc:
        _log.debug("positive emotional event mute failed: %s", exc)
        muted = []
    for ev in muted:
        desc = (ev.get("description") or "").strip()
        if desc:
            labels.append(desc)

    # Keep order but remove duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(label)
    if unique:
        _log.info(
            "[memory] canceled/staled event memory for person_id=%s: %s",
            person_id,
            "; ".join(unique),
        )
    return unique


def _event_cancellation_ack(labels: list[str], person_id: Optional[int]) -> str:
    del person_id
    label = labels[0] if labels else "that plan"
    return f"Got it - {label} is no longer on the flight plan."


def _handle_conversation_boundary(
    person_id: Optional[int],
    text: str,
) -> Optional[str]:
    """Store durable conversational boundaries like "don't ask about work"."""
    if person_id is None or not text:
        return None
    try:
        detected = boundary_memory.detect_boundary(
            text,
            fallback_topic=_boundary_fallback_topic(),
        )
        if not detected:
            return None
        applied = boundary_memory.apply_detected_boundary(person_id, detected)
        if not applied:
            return None
        topic = applied.get("topic") or "that"
        topic_phrase = "how you're doing" if topic == "how are you" else topic
        action = applied.get("action")
        behavior = applied.get("behavior") or "mention"
        if action == "clear":
            return f"Got it. {topic_phrase} is back on the table, cautiously."
        _apply_topic_boundary_side_effects(person_id, text)
        if behavior == "roast":
            return f"Noted. I won't roast you about {topic_phrase}."
        if behavior == "ask":
            return f"Got it. I won't ask about {topic_phrase} unless you bring it up."
        return f"Understood. I won't bring up {topic_phrase} unless you do."
    except Exception as exc:
        _log.debug("conversation boundary handler failed: %s", exc)
        return None


def _apply_topic_boundary_side_effects(person_id: Optional[int], text: str) -> None:
    """Stop proactive continuation when the human closes or rejects a topic."""
    try:
        conversation_steering.clear(person_id)
    except Exception as exc:
        _log.debug("clear conversation steering after boundary failed: %s", exc)
    try:
        topic_thread.clear()
    except Exception as exc:
        _log.debug("clear topic thread after boundary failed: %s", exc)
    try:
        end_thread.note_user_turn(text, person_id)
    except Exception as exc:
        _log.debug("end-thread boundary grace failed: %s", exc)


def _boundary_fallback_topic() -> Optional[str]:
    """Best current topic for generic boundaries like 'don't ask me that again'."""
    if _pending_face_reveal_confirm is not None:
        return "face"
    if _pending_offscreen_identify is not None:
        return "identity"
    if (
        _pending_introduction is not None
        or _pending_intro_followup is not None
        or _pending_intro_voice_capture is not None
        or _pending_common_first_name_identity is not None
        or _pending_common_first_name_introduction is not None
        or _pending_existing_common_first_name is not None
        or _pending_identity_match_confirmation is not None
        or _pending_prompted_name_confirmation is not None
    ):
        return "introductions"
    try:
        pending_rel = consciousness.get_pending_relationship_context()
        if pending_rel:
            return "relationships"
    except Exception:
        pass
    try:
        thread = topic_thread.snapshot() or {}
        if thread.get("label"):
            return thread.get("label")
    except Exception:
        pass
    return None


def _dismiss_pending_consent_prompts(person_id: Optional[int], reason: str) -> None:
    """Close optional pending prompts when a person sets a boundary or declines."""
    global _pending_face_reveal_confirm, _pending_offscreen_identify
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation

    if person_id is not None:
        try:
            declined = rel_memory.decline_latest_pending_question(person_id, reason=reason)
            if declined:
                _log.info(
                    "[boundaries] declined pending Q&A person_id=%s key=%r reason=%r",
                    person_id,
                    declined.get("question_key"),
                    reason,
                )
        except Exception as exc:
            _log.debug("decline pending Q&A failed: %s", exc)

    if _pending_face_reveal_confirm is not None:
        if person_id is None or _pending_face_reveal_confirm.get("person_id") == person_id:
            _face_reveal_declined.add(_pending_face_reveal_confirm["person_id"])
            _pending_face_reveal_confirm = None
    if _pending_offscreen_identify is not None:
        _pending_offscreen_identify = None
    if _pending_introduction is not None:
        _pending_introduction = None
    if _pending_intro_followup is not None:
        _pending_intro_followup = None
    if _pending_intro_voice_capture is not None:
        _pending_intro_voice_capture = None
    if _pending_common_first_name_identity is not None:
        _pending_common_first_name_identity = None
    if _pending_common_first_name_introduction is not None:
        _pending_common_first_name_introduction = None
    if _pending_existing_common_first_name is not None:
        if (
            person_id is None
            or _pending_existing_common_first_name.get("person_id") == person_id
        ):
            try:
                _common_first_name_prompted_this_session.add(
                    int(_pending_existing_common_first_name["person_id"])
                )
            except Exception:
                pass
            _pending_existing_common_first_name = None
    if _pending_identity_match_confirmation is not None:
        _pending_identity_match_confirmation = None
    if _pending_prompted_name_confirmation is not None:
        _pending_prompted_name_confirmation = None


def _generate_repair_response(person_id: Optional[int], text: str, repair: dict) -> str:
    """Generate one concise recovery when the human flags a conversational miss."""
    if repair.get("kind") == "pronoun":
        _maybe_store_pronoun_repair(person_id, text)
    prompt = repair_moves.build_prompt(repair)
    try:
        response = llm.get_response(prompt, person_id)
    except Exception as exc:
        _log.debug("repair response generation failed: %s", exc)
        response = ""
    response = (response or "").strip()
    if not response:
        response = repair_moves.fallback_response(repair)
    if repair_moves.should_use_better_luck_line(repair):
        response = repair_moves.add_better_luck_line(response)
    _play_event_body_beat("repair", repair_kind=str(repair.get("kind") or ""))
    _speak_blocking(
        response,
        emotion="neutral",
        pre_beat_ms=150,
        post_beat_ms_override=300,
    )
    repair_moves.mark_handled(repair.get("kind") or "")
    _log.info(
        "[repair] handled kind=%s severity=%s correction=%r user=%r response=%r",
        repair.get("kind"),
        repair.get("severity"),
        repair.get("correction"),
        text,
        response,
    )
    return response


def _maybe_store_pronoun_repair(person_id: Optional[int], text: str) -> None:
    """Persist explicit pronoun corrections for future group-cast prompts."""
    cleaned = (text or "").strip()
    if not cleaned:
        return

    target_id = None
    pronouns = None

    self_match = _PRONOUN_SELF_RE.search(cleaned)
    if self_match and person_id is not None:
        target_id = person_id
        pronouns = self_match.group(1)

    if target_id is None:
        named_match = _PRONOUN_NAMED_RE.search(cleaned)
        if named_match:
            name = named_match.group(1)
            if name.lower() not in {"he", "she", "they", "i", "my"}:
                try:
                    row = people_memory.find_person_by_name(name)
                    if row and row.get("id") is not None:
                        target_id = int(row["id"])
                        pronouns = named_match.group(2)
                except Exception as exc:
                    _log.debug("pronoun repair name lookup failed: %s", exc)

    if target_id is None:
        third_match = _PRONOUN_THIRD_PERSON_RE.search(cleaned)
        if third_match:
            candidates = []
            try:
                for p in world_state.get("people") or []:
                    pid = p.get("person_db_id")
                    if pid is not None and pid != person_id:
                        candidates.append(int(pid))
            except Exception:
                candidates = []
            if len(candidates) == 1:
                target_id = candidates[0]
                pronouns = third_match.group(1)

    if target_id is None or not pronouns:
        return

    normalized = pronouns.lower()
    if not _PRONOUN_VALUE_RE.fullmatch(normalized):
        return
    try:
        facts_memory.apply_fact_correction(
            int(target_id),
            "pronouns",
            normalized,
            category="identity",
        )
        _log.info(
            "[repair] stored pronoun correction person_id=%s pronouns=%s",
            target_id,
            normalized,
        )
    except Exception as exc:
        _log.debug("pronoun repair fact save failed: %s", exc)


def _apply_local_sensitive_topic_prepass(
    person_id: Optional[int],
    text: str,
) -> Optional[dict]:
    """Cache a same-turn safety mode before the async empathy classifier returns."""
    result = empathy.classify_local_sensitivity(text)
    if not result:
        return None

    person_row = None
    if person_id is not None:
        try:
            person_row = people_memory.get_person(person_id)
        except Exception:
            person_row = None

    child_in_scene = False
    try:
        ws_people = world_state.get("people") or []
        child_in_scene = any(
            p.get("age_estimate") == "child" for p in ws_people
        )
    except Exception:
        pass

    mode_pack = empathy.select_mode(
        result,
        person=person_row,
        child_in_scene=child_in_scene,
        person_id=person_id,
    )
    empathy.record(person_id, result, mode_pack)
    _log.info(
        "[empathy] local sensitive prepass affect=%s sens=%s crisis=%s "
        "event=%s → mode=%s",
        result.get("affect"),
        result.get("topic_sensitivity"),
        result.get("crisis"),
        (result.get("event") or {}).get("category"),
        mode_pack.get("mode"),
    )
    return result


def _merge_with_local_sensitive_prepass(
    result: Optional[dict],
    local_result: Optional[dict],
) -> Optional[dict]:
    """Preserve same-turn safety if the slower classifier comes back weaker."""
    if not result or not local_result:
        return result

    rank = {"none": 0, "mild": 1, "heavy": 2}
    result_rank = rank.get((result.get("topic_sensitivity") or "none").lower(), 0)
    local_rank = rank.get((local_result.get("topic_sensitivity") or "none").lower(), 0)
    if bool(local_result.get("crisis")) and not bool(result.get("crisis")):
        return local_result
    if result_rank < local_rank:
        return local_result

    local_event = local_result.get("event")
    if local_event and not result.get("event"):
        merged = dict(result)
        merged["event"] = local_event
        merged["source"] = result.get("source") or "llm_with_local_sensitive_event"
        return merged
    return result


def _maybe_start_grief_flow(
    person_id: Optional[int],
    empathy_event: Optional[dict],
) -> Optional[str]:
    """Return Rex's first-line response if a grief flow should kick off, else None.

    Conditions: person_id known, no active flow already, empathy classified
    a fresh event whose category indicates loss AND has an identifiable
    loss_subject. The structured response replaces the LLM-streamed one for
    this turn; subsequent turns from the same person are routed through
    _continue_grief_flow until the flow closes.
    """
    if person_id is None or empathy_event is None:
        return None
    if _grief_flow_active(person_id):
        return None
    category = (empathy_event.get("category") or "").lower()
    if category not in _GRIEF_FLOW_KEYWORDS:
        return None
    subject = (empathy_event.get("loss_subject") or "").strip().lower()
    if not subject:
        return None
    subject_kind = (empathy_event.get("loss_subject_kind") or "other").lower()
    name = (empathy_event.get("loss_subject_name") or "").strip() or None

    _grief_flow_state[person_id] = {
        "step": "awaiting_consent",
        "subject": subject,
        "subject_kind": subject_kind,
        "name": name,
        "started_at": time.monotonic(),
    }
    _force_grief_empathy(person_id, "grief flow started")

    your_subject = f"your {subject}"
    if name:
        return (
            f"I'm so sorry to hear about {name}. That's a lot. "
            f"Would you like to talk about {'them' if subject_kind != 'pet' else 'them'}?"
        )
    return (
        f"I'm sorry to hear about {your_subject}. "
        f"Would you like to talk about {'them' if subject_kind != 'pet' else 'them'}?"
    )


def _continue_grief_flow(person_id: int, text: str) -> Optional[str]:
    """Advance the grief flow one step. Returns Rex's response or None to fall
    back to the normal LLM path (with empathy directive still applied)."""
    state = _grief_flow_state.get(person_id)
    if not state:
        return None
    step = state.get("step")
    subject = state["subject"]
    subject_kind = state["subject_kind"]
    _force_grief_empathy(person_id, f"grief flow active: {step}")

    if step == "awaiting_consent":
        consent = _classify_consent(text)
        if consent is False:
            _grief_flow_clear(person_id)
            return (
                "Of course. I'm not going anywhere — say the word "
                "if you ever want to."
            )
        if consent is True:
            if state.get("name"):
                # Name was already in the original utterance — skip the name
                # ask, go straight to an open question that invites them to
                # share. The "I'm sorry" condolence already happened on the
                # first turn, so don't repeat it.
                state["step"] = "awaiting_description"
                return f"What were they like?"
            state["step"] = "awaiting_name"
            return f"What was your {subject}'s name?"
        # Ambiguous — don't push. Treat as decline-soft.
        _grief_flow_clear(person_id)
        return (
            "No pressure. I'm here if you want to circle back to it."
        )

    if step == "awaiting_name":
        try:
            name = llm.extract_name_from_reply(text)
        except Exception as exc:
            _log.debug("extract_name_from_reply failed: %s", exc)
            name = None
        if name:
            state["name"] = name
            state["step"] = "awaiting_description"
            # Persist the name onto the most recent emotional event so future
            # sessions and the system-prompt recall layer can use it.
            try:
                rows = _db_fetch_latest_emo_event(person_id)
                if rows:
                    new_desc = rows[0]["description"]
                    if name.lower() not in (new_desc or "").lower():
                        new_desc = f"{new_desc} (name: {name})"
                        _db_update_emo_description(rows[0]["id"], new_desc)
            except Exception as exc:
                _log.debug("emotional event name persist failed: %s", exc)
            return (
                f"I'm so sorry to hear {name} passed. That's got to be "
                f"difficult. What were they like?"
            )
        # No name extracted — gracefully proceed without one.
        state["step"] = "awaiting_description"
        return (
            "I'm so sorry. What were they like?"
        )

    if step == "awaiting_description":
        # Hand back to the normal LLM with the empathy directive intact —
        # the response now has rich context (name, transcript, mode=listen)
        # so the LLM can naturally validate and stay present.
        _grief_flow_clear(person_id)
        return None

    # Unknown step → bail safely.
    _grief_flow_clear(person_id)
    return None


def _db_fetch_latest_emo_event(person_id: int) -> list[dict]:
    from memory import database as _db
    rows = _db.fetchall(
        "SELECT id, description FROM person_emotional_events "
        "WHERE person_id = ? ORDER BY id DESC LIMIT 1",
        (person_id,),
    )
    return [dict(r) for r in rows]


def _db_update_emo_description(event_id: int, new_description: str) -> None:
    from memory import database as _db
    _db.execute(
        "UPDATE person_emotional_events SET description = ? WHERE id = ?",
        (new_description, int(event_id)),
    )


def _curiosity_check(
    response_text: str,
    user_text: str,
    person_id: Optional[int],
    person_name: Optional[str],
) -> Optional[str]:
    """
    After Rex gives a response with no question mark, optionally ask a follow-up.

    Priority:
      1. If the response already contains '?', do nothing.
      2. Roll against CURIOSITY_QUESTION_PROBABILITY — skip on failure.
      3. Try the depth-appropriate question pool for this person.
      4. Fall back to a short contextual LLM question.

    Speaks the question as a separate TTS line (within the caller's AEC sequence)
    and returns the text, or None if nothing was spoken.
    """
    if "?" in response_text:
        if person_id is not None:
            _record_pool_topics_in_response(response_text, person_id)
        return None

    # Empathy gate — never fire a separate snarky follow-up question when the
    # active mode is sympathetic. The curiosity step uses a stand-alone LLM
    # call with no system-prompt scaffolding, so left ungated it produces
    # tone-deaf lines like "guess you'll need a new best friend" right after
    # someone mentions their pet died. Only allow curiosity in modes where
    # snark is appropriate.
    _CURIOSITY_OK_MODES = {"default", "amplify", "lift", "gentle_probe"}
    try:
        cached = empathy.peek(person_id)
    except Exception:
        cached = None
    if cached:
        active_mode = (cached.get("mode") or {}).get("mode", "default")
        if active_mode not in _CURIOSITY_OK_MODES:
            _log.info(
                "[interaction] curiosity_check suppressed — empathy mode=%s "
                "(person_id=%s)",
                active_mode, person_id,
            )
            return None

    try:
        if end_thread.is_grace_active():
            _log.info(
                "[interaction] curiosity_check suppressed — end-of-thread grace "
                "(person_id=%s)",
                person_id,
            )
            return None
    except Exception as exc:
        _log.debug("end-of-thread curiosity check failed: %s", exc)

    try:
        if not question_budget.can_ask("curiosity_followup"):
            _log.info(
                "[interaction] curiosity_check suppressed — question budget full "
                "(person_id=%s)",
                person_id,
            )
            return None
    except Exception as exc:
        _log.debug("question budget curiosity check failed: %s", exc)

    if random.random() >= config.CURIOSITY_QUESTION_PROBABILITY:
        return None

    question_text: Optional[str] = None
    pool_question: Optional[dict] = None

    # Interest steering wins over the generic question pool. If someone just
    # opened a hobby/skill thread, the add-on question should deepen *that*
    # instead of suddenly asking where they are from like a clipboard with LEDs.
    if person_id is not None:
        try:
            steering = conversation_steering.build_context(person_id)
        except Exception as exc:
            _log.debug("curiosity_check steering context failed: %s", exc)
            steering = None
        if steering:
            try:
                question_text = llm.get_response(
                    "Generate ONE short DJ-R3X follow-up question about this "
                    f"person's interest in {steering.topic!r}. It should ask "
                    "about their skill, process, tools, taste, or favorite part. "
                    "Optional: include one tiny subject-specific tidbit if you "
                    "are confident. Funny and snarky is fine, but do not roast "
                    "their competence. Return only the line.",
                    person_id,
                )
                if question_text:
                    pool_question = {
                        "key": f"{steering.fact_key}_followup",
                        "text": question_text,
                        "depth": 1,
                    }
                    try:
                        interests_memory.mark_interest_asked(
                            person_id,
                            steering.topic,
                        )
                    except Exception as exc:
                        _log.debug("curiosity_check mark active interest asked failed: %s", exc)
            except Exception as exc:
                _log.debug("curiosity_check steering question error: %s", exc)

    # Known durable interests are deeper hooks, not basic intake questions.
    if person_id is not None and not question_text:
        try:
            hooks = interests_memory.get_interest_hooks(person_id)
        except Exception as exc:
            _log.debug("curiosity_check interest hooks load error: %s", exc)
            hooks = []
        if hooks:
            hook = hooks[0]
            interest_name = hook.get("name") or ""
            note = hook.get("notes") or hook.get("associated_stories") or ""
            try:
                question_text = llm.get_response(
                    "Generate ONE short DJ-R3X follow-up question about this "
                    f"known interest: {interest_name!r}. "
                    f"Known context: {note!r}. "
                    "Do not ask whether they like it; Rex already knows that. "
                    "Ask a deeper, specific follow-up about what they are making, "
                    "playing, practicing, learning, comparing, or what changed "
                    "since they last mentioned it. Funny and snarky is fine, "
                    "but do not roast their competence. Return only the line.",
                    person_id,
                )
                if question_text:
                    pool_question = {
                        "key": f"interest_{re.sub(r'[^a-z0-9]+', '_', interest_name.lower()).strip('_')}_deep_followup",
                        "text": question_text,
                        "depth": 1,
                    }
                    interests_memory.mark_interest_asked(person_id, interest_name)
            except Exception as exc:
                _log.debug("curiosity_check known interest question error: %s", exc)

    # Try question pool first — structured, depth-gated, no extra LLM call
    if person_id is not None and not question_text:
        try:
            person = people_memory.get_person(person_id)
            tier = (person.get("friendship_tier", "stranger") if person else "stranger")
            max_depth = config.TIER_MAX_DEPTH.get(tier, 1)
            asked = rel_memory.get_asked_question_keys(person_id)

            # Load all known facts once so we can skip questions whose topic is
            # already covered — e.g. don't ask about job if a job fact exists.
            known_fact_keys: set[str] = set()
            known_fact_categories: set[str] = set()
            try:
                for fact in facts_memory.get_facts(person_id):
                    known_fact_keys.add(fact["key"])
                    known_fact_categories.add(fact["category"])
            except Exception as exc:
                _log.debug("curiosity_check facts load error: %s", exc)

            for candidate in config.QUESTION_POOL:
                if candidate["depth"] > max_depth:
                    continue
                if candidate["key"] in asked:
                    continue
                # Skip if Rex already knows something about this topic
                q_key = candidate["key"]
                if q_key in known_fact_keys or q_key in known_fact_categories:
                    _log.debug(
                        "curiosity_check: skipping %r — fact already recorded", q_key
                    )
                    continue
                if q_key in {"hobbies", "favorite_music", "favorite_movie"}:
                    try:
                        if interests_memory.get_interests_for_prompt(person_id, limit=1):
                            _log.debug(
                                "curiosity_check: skipping %r — interests already recorded",
                                q_key,
                            )
                            continue
                    except Exception as exc:
                        _log.debug("curiosity_check interest skip failed: %s", exc)
                if _question_blocked_by_boundary(person_id, candidate):
                    _log.info(
                        "[boundaries] curiosity_check suppressed question=%r for person_id=%s",
                        q_key,
                        person_id,
                    )
                    continue
                question_text = candidate.get("text", "")
                pool_question = candidate
                break
        except Exception as exc:
            _log.debug("curiosity_check pool error: %s", exc)

    # LLM fallback — contextual question when pool is empty or person is unknown
    if not question_text:
        if person_id is not None:
            try:
                if boundary_memory.is_blocked(person_id, "ask", "questions"):
                    _log.info(
                        "[boundaries] curiosity_check LLM fallback suppressed — questions blocked person_id=%s",
                        person_id,
                    )
                    return None
            except Exception as exc:
                _log.debug("curiosity fallback boundary check failed: %s", exc)
        try:
            question_text = llm.generate_curiosity_question(response_text, user_text, person_id)
        except Exception as exc:
            _log.debug("curiosity_check LLM error: %s", exc)

    if not question_text or not question_text.strip():
        return None

    _speak_blocking(question_text)
    _log.info("[interaction] curiosity question spoken: %r", question_text)

    # Record pool question as asked immediately so get_next_question() won't
    # return it again in this session. Answer is empty — updated later if the
    # person responds; what matters now is the key is in the DB.
    if pool_question is not None and person_id is not None:
        try:
            rel_memory.save_question_asked(
                person_id,
                pool_question["key"],
                question_text,
                pool_question.get("depth", 1),
            )
        except Exception as exc:
            _log.debug("curiosity_check save_qa error: %s", exc)

    return question_text


def _maybe_capture_pending_qa(
    person_id: Optional[int],
    text: str,
    *,
    identity_prompt_active: bool = False,
) -> Optional[dict]:
    """
    If Rex asked this person a curiosity question on the previous turn, treat
    this utterance as the answer and store it before generating the next reply.
    """
    if person_id is None:
        return None
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if identity_prompt_active:
        return None
    if _pending_offscreen_identify is not None or _pending_face_reveal_confirm is not None:
        return None
    if _awaiting_followup_event is not None:
        return None
    if command_parser.parse(cleaned) is not None:
        return None
    if _EMOTIONAL_BOUNDARY_PAT.search(cleaned):
        return None
    if _looks_like_incomplete_pending_answer(cleaned):
        _log.info(
            "[interaction] pending Q&A capture held — incomplete fragment %r",
            cleaned,
        )
        return None
    # If the user is asking a new question instead of answering Rex's last one,
    # leave the pending question open. Real conversations branch sometimes.
    if "?" in cleaned:
        return None
    try:
        answered = rel_memory.answer_latest_pending_question(person_id, cleaned)
        if answered:
            _log.info(
                "[interaction] captured answer to pending Q&A — person_id=%s "
                "key=%r answer=%r",
                person_id,
                answered.get("question_key"),
                cleaned[:160],
            )
            if answered.get("question_key") == "startup_conversation_steering":
                try:
                    conversation_steering.note_bare_interest_answer(
                        person_id,
                        cleaned,
                        source="startup_steering_answer",
                    )
                except Exception as exc:
                    _log.debug("startup steering answer capture failed: %s", exc)
            elif conversation_steering.is_interest_seed_question(
                answered.get("question_key")
            ):
                # Answer to a profile interest question (obsession/hobbies/music/
                # …) — seed it as the active topic so Rex deepens it with real
                # curiosity instead of a throwaway quip.
                try:
                    conversation_steering.seed_from_answer(
                        person_id,
                        cleaned,
                        answered.get("question_key"),
                    )
                except Exception as exc:
                    _log.debug("interest-seed answer capture failed: %s", exc)
        return answered
    except Exception as exc:
        _log.debug("pending Q&A capture failed: %s", exc)
        return None


_INCOMPLETE_PENDING_ANSWER_PAT = re.compile(
    r"^\s*(?:"
    r"i\s+(?:like|love|want|think|mean|prefer|would|was|am|have|had|feel)|"
    r"i'?d\s+(?:like|love|rather)|"
    r"my|the|it(?:'?s| is)?|because|when|what i"
    r")\s*$",
    re.IGNORECASE,
)


def _looks_like_incomplete_pending_answer(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return bool(_INCOMPLETE_PENDING_ANSWER_PAT.match(cleaned))


def _looks_like_startup_steering_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        phrase in q
        for phrase in (
            "what topic",
            "what corner",
            "what mission",
            "what problem",
            "what are you up to",
            "what do you want to talk",
            "what should we talk",
            "what are we discussing",
            "discussing first",
            "what's been rolling around",
            "what’s been rolling around",
            "been rolling around in your mind",
            "in your mind today",
            "what's on your mind",
            "what’s on your mind",
            "what is on your mind",
        )
    )


def _maybe_capture_topic_thread_answer(
    person_id: Optional[int],
    text: str,
    *,
    identity_prompt_active: bool = False,
) -> Optional[dict]:
    """Fallback answer capture when Rex's latest question lives only in thread state."""
    if person_id is None or identity_prompt_active:
        return None
    cleaned = (text or "").strip()
    if not cleaned or "?" in cleaned:
        return None
    if command_parser.parse(cleaned) is not None:
        return None
    try:
        snap = topic_thread.snapshot() or {}
        question = str(snap.get("unresolved_question") or "").strip()
    except Exception:
        question = ""
    if not question or not _looks_like_startup_steering_question(question):
        return None
    try:
        ctx = conversation_steering.note_bare_interest_answer(
            person_id,
            cleaned,
            source="startup_thread_answer",
        )
    except Exception as exc:
        _log.debug("topic-thread steering answer capture failed: %s", exc)
        ctx = None
    if ctx is None:
        answered = {
            "question_key": "startup_conversation_steering_reply",
            "question_text": question,
            "answer_text": cleaned,
            "depth_level": 1,
            "source": "topic_thread",
        }
        _log.info(
            "[interaction] captured topic-thread reply without storing interest — "
            "person_id=%s key=%r answer=%r",
            person_id,
            answered["question_key"],
            cleaned[:160],
        )
        return answered
    answered = {
        "question_key": "startup_conversation_steering",
        "question_text": question,
        "answer_text": cleaned,
        "depth_level": 1,
        "source": "topic_thread",
    }
    _log.info(
        "[interaction] captured topic-thread answer — person_id=%s "
        "key=%r answer=%r",
        person_id,
        answered["question_key"],
        cleaned[:160],
    )
    return answered


def _latest_pending_question(person_id: Optional[int]) -> Optional[dict]:
    if person_id is None:
        return None
    try:
        return rel_memory.get_latest_pending_question(person_id)
    except Exception as exc:
        _log.debug("pending Q&A lookup failed: %s", exc)
        return None


def _store_music_preference_fact(person_id: Optional[int], preference: str) -> None:
    if person_id is None:
        return
    cleaned = (preference or "").strip()
    if not cleaned:
        return
    try:
        facts_memory.add_fact(
            person_id,
            "preference",
            "favorite_music",
            cleaned,
            "pending_qa:favorite_music",
            confidence=0.95,
        )
    except Exception as exc:
        _log.debug("favorite_music fact save failed: %s", exc)


_MUSIC_PREFERENCE_LEADING_PATTERNS = (
    re.compile(
        r"^\s*i\s+(?:really\s+|kind\s+of\s+|sort\s+of\s+)?"
        r"(?:like|love|enjoy|prefer|am\s+into|am\s+interested\s+in|listen\s+to)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*i['’]?m\s+(?:really\s+|kind\s+of\s+|sort\s+of\s+)?"
        r"(?:into|a\s+fan\s+of)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*my\s+(?:favorite|favourite)\s+"
        r"(?:kind|type|style|genre)?\s*(?:of\s+)?music\s+(?:is|would\s+be|has\s+to\s+be)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:probably|mostly|usually|mainly|generally)?\s*"
        r"(?:i\s+)?(?:would\s+say|guess|think)\s+",
        re.IGNORECASE,
    ),
)


def _normalize_music_preference_answer(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.strip(" \t\r\n\"'“”‘’")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()

    changed = True
    while changed:
        changed = False
        for pattern in _MUSIC_PREFERENCE_LEADING_PATTERNS:
            updated = pattern.sub("", cleaned, count=1).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True

    cleaned = re.sub(r"^(?:some|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+(?:music|stuff|songs|tracks|artists|bands)$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned or (text or "").strip().rstrip(".!?")


def _music_offer_reply(preference: str) -> str:
    cleaned = (preference or "").strip().rstrip(".!?")
    if not cleaned:
        cleaned = "that"
    return f"Good to know. Want me to play some {cleaned}, or keep the jukebox muzzled?"


def _music_offer_play_response(track, preference: str) -> str:
    name = getattr(track, "name", "") or preference
    return f"Yes detected. Spinning {name} now."


def _handle_pending_music_offer_reply(
    person_id: Optional[int],
    text: str,
) -> Optional[str]:
    """Handle yes/no replies to Rex's follow-up music offer."""
    global _pending_music_offer
    offer = _pending_music_offer
    if not offer:
        return None

    ttl = float(getattr(config, "MUSIC_OFFER_REPLY_WINDOW_SECS", 25.0))
    if time.monotonic() - float(offer.get("asked_at") or 0.0) > ttl:
        _pending_music_offer = None
        return None

    if offer.get("person_id") is not None and person_id != offer.get("person_id"):
        return None

    consent = _classify_consent(text)
    if consent is None:
        return None

    preference = str(offer.get("music_query") or "").strip()
    _pending_music_offer = None

    if consent is False:
        resp = "Got it. I logged the taste and spared the room a soundtrack ambush."
        _speak_blocking(resp, emotion="neutral")
        return resp

    try:
        from features import dj as dj_mod
        track = dj_mod.handle_request(preference)
    except Exception as exc:
        _log.debug("pending music offer lookup failed: %s", exc)
        track = None

    if track is None:
        resp = f"I logged {preference} as your taste, but I can't find a station for it yet."
        _speak_blocking(resp, emotion="neutral")
        return resp

    resp = _music_offer_play_response(track, preference)
    _speak_blocking(resp, emotion="happy")
    _start_dj_after_response(track)
    return resp


def _handle_pending_music_preference_answer(
    person_id: Optional[int],
    text: str,
    *,
    pending_question: Optional[dict],
    identity_prompt_active: bool = False,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Consume answers to Rex's music-preference question before the action router.

    A bare genre/artist/style is a memory answer. Rex asks for explicit consent
    before starting playback, so "classical music" cannot accidentally become
    a DJ command.
    """
    global _pending_music_offer
    if person_id is None or not pending_question:
        return None, None
    if pending_question.get("question_key") != "favorite_music":
        return None, None
    if identity_prompt_active:
        return None, None

    cleaned = (text or "").strip()
    if not cleaned or "?" in cleaned:
        return None, None
    preference = _normalize_music_preference_answer(cleaned)
    if not preference:
        return None, None

    answered = _maybe_capture_pending_qa(
        person_id,
        preference,
        identity_prompt_active=identity_prompt_active,
    )
    if not answered:
        return None, None

    _store_music_preference_fact(person_id, preference)

    if _MUSIC_PLAY_REQUEST_PAT.search(cleaned):
        return None, answered

    _pending_music_offer = {
        "person_id": person_id,
        "music_query": preference,
        "asked_at": time.monotonic(),
    }
    resp = _music_offer_reply(preference)
    _speak_blocking(resp, emotion="neutral")
    return resp, answered


# ─────────────────────────────────────────────────────────────────────────────
# Intent-routed local responses (LLM fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_classified_intent(
    intent: str,
    raw_text: str,
    person_id: Optional[int],
    *,
    raw_best_id: Optional[int] = None,
    raw_best_name: Optional[str] = None,
    raw_best_score: float = 0.0,
    visible_known_name: Optional[str] = None,
) -> Optional[str]:
    """Answer a classified intent locally with real data, in Rex's voice.

    Returns the spoken response text, or None if the intent should fall
    through to the normal LLM path (i.e. 'general' or unhandled).

    The raw_best_* kwargs carry the UNFILTERED top voice-ID candidate for
    this utterance (no threshold applied). Used by query_who_is_speaking to
    produce confidence-aware responses. visible_known_name is the name of a
    currently-visible identified face if any.
    """
    def _say(prompt: str) -> str:
        resp = llm.get_response(prompt, person_id)
        _speak_blocking(resp)
        return resp

    tool_scope = (
        "Stay strictly on the user's requested tool answer. Do not mention "
        "unrelated memories, grief, death, sensitive events, visual observations, "
        "or personal facts unless the user asked for them in this turn."
    )

    if intent == "query_what_do_you_see":
        _try_slow_path_ack("vision")
    elif intent == "query_memory":
        _try_slow_path_ack("memory")

    if intent == "query_time":
        resp = _format_current_time_response()
        _speak_blocking(resp)
        return resp

    if intent == "query_date":
        resp = _format_current_date_response()
        _speak_blocking(resp)
        return resp

    if intent == "query_weather":
        from awareness.chronoception import refresh_weather
        w = refresh_weather()
        temp = w.get("temp_f")
        desc = w.get("description") or w.get("condition") or "unknown"
        location = getattr(config, "WEATHER_LOCATION", "the local area")
        _log.info("[interaction] weather fetched for %r: %s", location, w)
        if temp is None:
            return _say(
                "The user asked about the weather but your weather feed is offline "
                "right now. Tell them you can't reach the weather service in one "
                f"Rex-style line. Do NOT make up a temperature or conditions. {tool_scope}"
            )
        return _say(
            f"The real current weather in {location} is exactly {temp}°F and "
            f"{desc}. Tell the user the weather in one Rex-style line. "
            f"You MUST state the temperature exactly as given ({temp}°F) and the "
            f"conditions ({desc}) — do not round, do not invent different numbers, "
            f"do not substitute different conditions. {tool_scope}"
        )

    if intent == "query_games":
        from features import games as games_mod
        game_list = ", ".join(games_mod.available_game_names()) or "none right now"
        return _say(
            f"The user asked what games you can play. Your actual game list: {game_list}. "
            f"Tell them in one Rex-style line. Be brief. {tool_scope}"
        )

    if intent == "query_capabilities":
        capabilities = (
            "hold a real conversation and remember people across visits; "
            "recognize faces and voices; describe what you see through your camera; "
            "play games (Trivia, Jeopardy, I Spy, 20 Questions, Word Association); "
            "DJ music with skip / stop / volume control; "
            "tell time, date, and weather; "
            "track your own mood, anger, and uptime"
        )
        return _say(
            f"The user asked what you can do. Your real capabilities: {capabilities}. "
            f"Summarize in one or two Rex-style lines. Be brief. {tool_scope}"
        )

    if intent == "query_uptime":
        try:
            self_state = world_state.get("self_state")
            up = int(self_state.get("uptime_seconds", 0) or 0)
        except Exception:
            up = 0
        hours = up // 3600
        minutes = (up % 3600) // 60
        return _say(
            f"Tell the user your uptime is exactly {hours} hours and {minutes} minutes "
            f"in Rex character. Be brief. {tool_scope}"
        )

    if intent == "query_what_do_you_see":
        desc = ""
        try:
            from vision import scene as vision_scene
            desc = vision_scene.describe_scene()
        except Exception as exc:
            _log.debug("intent vision describe error: %s", exc)
        return _say(
            f"You were asked what you see. Scene analysis: '{desc or 'nothing notable'}'. "
            f"Describe it in character in one or two lines."
        )

    if intent == "query_music_options":
        try:
            stations = getattr(config, "RADIO_STATIONS", []) or []
            seen: list[str] = []
            for s in stations:
                vibes = s.get("vibes") or []
                if vibes:
                    primary = str(vibes[0]).strip().lower()
                    if primary and primary not in seen:
                        seen.append(primary)
            genre_list = ", ".join(seen) if seen else "a variety of genres"
        except Exception as exc:
            _log.debug("query_music_options enumeration error: %s", exc)
            genre_list = "a variety of genres"
        return _say(
            f"The user asked what kind of music you can play. Your real radio "
            f"genre buckets are: {genre_list}. You can also play local tracks by "
            f"title, artist, or vibe. Answer in ONE short Rex-style line — list "
            f"the genres tersely (comma-separated is fine), no preamble, no fluff."
        )

    if intent == "query_memory":
        try:
            target = memory_query.resolve_target(raw_text, person_id)
            if target.ambiguous_names:
                names = ", ".join(target.ambiguous_names[:5])
                return _say(
                    f"The user asked a memory question ({raw_text!r}), but their "
                    f"relationship phrase matched multiple people: {names}. Ask "
                    f"which one they mean in ONE short Rex-style line. Do not guess."
                )
            if not target.resolved:
                if target.detail == "relationship_query_without_current_person":
                    rel = target.relation_label or "relationship"
                    return _say(
                        f"The user asked about their {rel}, but you do not know who "
                        f"is speaking with enough confidence. In ONE short Rex-style "
                        f"line, say you need to know who they are first before pulling "
                        f"that relationship memory. Do not invent a name."
                    )
                if target.detail == "no_relationship_match":
                    rel = target.relation_label or "relationship"
                    return _say(
                        f"The user asked about their {rel}, but no matching "
                        f"relationship is stored in memory. In ONE short Rex-style "
                        f"line, say you do not have that relationship saved yet. "
                        f"Do not invent a person."
                    )
                if target.detail == "no_person_match" and target.name:
                    return _say(
                        f"The user asked what you know about {target.name!r}, but "
                        f"there is no matching person in memory. In ONE short "
                        f"Rex-style line, say you do not have memory for that person "
                        f"yet. Do not invent facts."
                    )
                return _say(
                    f"The user asked a memory question ({raw_text!r}), but you "
                    f"cannot resolve who it is about. In ONE short Rex-style line, "
                    f"ask them to name the person. Do not invent."
                )

            context = memory_query.build_context(target, person_id)
            if not context.has_memory:
                return _say(
                    f"The user asked a memory question ({raw_text!r}). You resolved "
                    f"the target as {target.name or 'that person'}, but only the "
                    f"person record exists and no facts, relationships, events, or "
                    f"conversation summaries were found. Say that honestly in ONE "
                    f"short Rex-style line."
                )
            return _say(memory_query.build_response_prompt(raw_text, context))
        except Exception as exc:
            _log.debug("query_memory intent failed: %s", exc)
            return _say(
                "The user asked a memory question, but your memory lookup errored. "
                "Tell them briefly in one Rex-style line that the memory banks "
                "hiccuped and you cannot pull it up right now."
            )

    if intent == "play_music":
        try:
            from features import dj as dj_mod
            track = dj_mod.handle_request(raw_text)
        except Exception as exc:
            _log.debug("play_music intent dj error: %s", exc)
            track = None
        if track is None:
            return _say(
                f"The user asked you to play music ('{raw_text}') but no matching "
                f"track or station was found. Tell them you couldn't find anything "
                f"matching that in one in-character Rex line."
            )
        response = _say(
            f"You're about to play '{track.name}' ({track.description}) in response "
            f"to: '{raw_text}'. Announce it in one short in-character Rex line."
        )
        _start_dj_after_response(track)
        return response

    if intent == "query_who_is_speaking":
        # Build a confidence-aware prompt. Priority order:
        #   1. Face visible + identified → confident by face
        #   2. Voice score >= hard threshold → confident by voice
        #   3. Voice score >= a "maybe" floor with a plausible candidate →
        #      Rex expresses uncertainty with the candidate name
        #   4. Nothing above the floor → Rex honestly says he doesn't know
        hard = float(config.SPEAKER_ID_SIMILARITY_THRESHOLD)
        # "Maybe" floor: voice scores in [0.50, hard) deserve a tentative guess.
        maybe_floor = float(getattr(config, "SPEAKER_ID_MAYBE_FLOOR", 0.50))

        candidate_name = raw_best_name
        candidate_score = raw_best_score

        if visible_known_name:
            # Face wins — tell them with confidence.
            return _say(
                f"Someone asked who's speaking. You can actually SEE them on "
                f"camera and you recognize them as {visible_known_name}. "
                f"In one short in-character Rex line, confirm their identity — "
                f"warm but dry. Address them by name. One line only."
            )

        if candidate_name and candidate_score >= hard:
            return _say(
                f"Someone asked who's speaking. You recognize the voice with high "
                f"confidence (score {candidate_score:.2f}) as {candidate_name}. "
                f"In ONE short in-character Rex line, confirm their identity. "
                f"Be direct, address them by name."
            )

        if candidate_name and candidate_score >= maybe_floor:
            return _say(
                f"Someone asked who's speaking. You have a PARTIAL voice match "
                f"(confidence score {candidate_score:.2f} out of 1.0) for "
                f"{candidate_name} — it might be them, but you're not sure. "
                f"In ONE short in-character Rex line, voice that uncertainty out "
                f"loud: say you're not sure but it could be {candidate_name}. "
                f"Keep the hedging audible — 'I'm not positive' / 'could be' / "
                f"'pretty sure but my sensors aren't certain'. One line only."
            )

        # Nothing plausible — honest unknown.
        return _say(
            f"Someone asked who's speaking but no voice print matched (top "
            f"similarity was only {candidate_score:.2f}) and you don't see their "
            f"face. In ONE short in-character Rex line, admit honestly that you "
            f"don't recognize the voice — NO guessing, NO roast about forget"
            f"tability unless it comes naturally. Just 'no idea, who's asking?' "
            f"in Rex's voice. One line only."
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Crosstalk / addressee triage
# ─────────────────────────────────────────────────────────────────────────────
# With an always-on mic, Rex overhears people talking to each OTHER (a partner in
# another room, the user answering someone else). Perfect addressee detection is
# impossible, so this is deliberately HIGH PRECISION: only the clearest
# "talking to another person" cues, and never when Rex is addressed — so it can
# never make Rex ignore plausibly-directed speech.

_REX_ADDRESS_TOKEN_RE = re.compile(
    r"\b(rex|r3x|r-?3x|dj|droid|robot|jukebox)\b",
    re.IGNORECASE,
)
# "love you" / "love you too" at a clause edge — affection aimed at a person, not
# a DJ droid. The edge anchor avoids requests like "I'd love you to play music".
_AFFECTION_TO_PERSON_RE = re.compile(
    r"\blove\s+(?:you|ya|u)(?:\s+too)?\s*(?:[,.!?]|$)",
    re.IGNORECASE,
)
# "babe" is essentially never a common noun in a living room → partner vocative
# anywhere in the line.
_PARTNER_VOCATIVE_ANYWHERE_RE = re.compile(r"\bbabe\b", re.IGNORECASE)
# Other endearments only count as vocatives when comma-set-off or at a sentence
# edge, so "pass the honey" / "what a sweetheart of a deal" don't trip it.
_PARTNER_VOCATIVE_EDGE_RE = re.compile(
    r"(?:^|,)\s*(?:honey|hon|sweetie|sweetheart|darling)\s*(?:[,.!?]|$)",
    re.IGNORECASE,
)


def _looks_like_third_party_crosstalk(text: str) -> bool:
    """Best-effort: True when an utterance clearly addresses another person, not
    Rex (partner endearments, "love you too"). High precision, low recall — it
    must never fire on plausibly Rex-directed speech, so unsure cases return False.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _REX_ADDRESS_TOKEN_RE.search(cleaned):
        return False
    if _AFFECTION_TO_PERSON_RE.search(cleaned):
        return True
    if _PARTNER_VOCATIVE_ANYWHERE_RE.search(cleaned):
        return True
    if _PARTNER_VOCATIVE_EDGE_RE.search(cleaned):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Speech segment processing
# ─────────────────────────────────────────────────────────────────────────────

def _handle_speech_segment(
    audio_array: np.ndarray,
    *,
    from_idle_activation: bool = False,
    transcribed_text: Optional[str] = None,
    raw_best_id_override: Optional[int] = None,
    raw_best_name_override: Optional[str] = None,
    speaker_score_override: float = 0.0,
    text_input: bool = False,
) -> None:
    """Full processing pipeline for one detected speech segment in ACTIVE state."""
    global _session_exchange_count, _identity_prompt_until, _awaiting_followup_event
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_offscreen_identify, _pending_face_reveal_confirm

    turn_start = time.monotonic()
    answered_question: Optional[dict] = None
    assistant_asked_question = False
    game_after_audio_path: Optional[str] = None
    event_cancellation_ack: Optional[str] = None
    repair_move: Optional[dict] = None
    router_decision: Optional[action_router.ActionDecision] = None
    router_audit: Optional[_RouterDecisionAudit] = None
    router_block_reason: Optional[str] = None
    dialogue_decision: Optional[dialogue_act.DialogueActDecision] = None
    character_trace: Optional[_CharacterLoopTrace] = None
    match: Optional[command_parser.CommandMatch] = None
    response_text: Optional[str] = None
    intent: Optional[str] = None
    completed: Optional[bool] = None
    handler_error: Optional[str] = None
    final_executed_path: Optional[str] = None
    suppress_memory_learning = False
    handled_active_game_turn = False
    used_agenda_llm = False
    used_classified_intent = False
    trace_context_token: Optional[contextvars.Token] = None

    if _shutdown_requested():
        return

    # Randomised pre-response pause — prevents Rex from feeling instant/robotic
    delay_started = time.monotonic()
    delay_ms = random.randint(config.REACTION_DELAY_MS_MIN, config.REACTION_DELAY_MS_MAX)
    time.sleep(delay_ms / 1000.0)
    if _shutdown_requested():
        return
    _latency_log(turn_start, "reaction_delay", delay_started)

    # Once Rex starts speaking in response, hold AEC suppression open across any
    # related output as one continuous window. Do not start it before
    # transcription; that made the logs look like user speech was captured
    # while playback suppression was active and encouraged premature filler.
    sequence_started = False
    try:
        # Concurrent transcription + speaker identification, unless the GUI
        # supplied text directly.
        process_started = time.monotonic()
        if transcribed_text is not None:
            text = str(transcribed_text or "").strip()
            raw_best_id = raw_best_id_override
            raw_best_name = raw_best_name_override
            speaker_score = float(speaker_score_override or 0.0)
            transcript_ready_at = time.monotonic()
            _latency_log(turn_start, "text_input", process_started)
        else:
            text, raw_best_id, raw_best_name, speaker_score = _process_audio(audio_array)
            transcript_ready_at = time.monotonic()
            _latency_log(turn_start, "transcribe_and_speaker_id", process_started)

        if _shutdown_requested():
            final_executed_path = "ignored.shutdown"
            completed = False
            return
        if not text:
            # Empty/blank transcript — a false VAD trigger or fully-filtered audio
            # (e.g. a Whisper hallucination loop). Nothing was actually said to Rex.
            # If we only flipped to ACTIVE because of this segment, drop straight back
            # to IDLE instead of camping in ACTIVE until the 45s conversation idle
            # timeout. Mirrors the crosstalk path below.
            final_executed_path = "ignored.empty_transcript"
            completed = False
            if from_idle_activation:
                try:
                    state_module.set_state(State.IDLE)
                except Exception:
                    pass
            return

        character_trace = _new_character_loop_trace(
            text,
            from_idle_activation=from_idle_activation,
            turn_start=turn_start,
            raw_best_id=raw_best_id,
            raw_best_name=raw_best_name,
            speaker_score=speaker_score,
        )
        character_trace.transcript_ready_at = transcript_ready_at
        trace_context_token = _current_character_loop_trace.set(character_trace)

        heard_log_text = text

        # Best-effort crosstalk suppression: if this utterance clearly addresses
        # another person (partner endearment, "love you too") and never names Rex,
        # treat it as overheard couple-talk, not a turn directed at Rex.
        if (
            not text_input
            and bool(getattr(config, "CROSSTALK_SUPPRESSION_ENABLED", True))
            and not _game_suppresses_conversation()
            and _looks_like_third_party_crosstalk(text)
        ):
            _log.info(
                "[interaction] suppressed overheard crosstalk (not addressed to Rex): %r",
                text,
            )
            final_executed_path = "ignored.crosstalk"
            completed = False
            if from_idle_activation:
                # Don't camp in ACTIVE on speech that was not even meant for Rex.
                try:
                    state_module.set_state(State.IDLE)
                except Exception:
                    pass
            return

        if not _game_suppresses_conversation():
            completion = turn_completion.consume_continuation(
                text=text,
                audio_array=audio_array,
                raw_best_id=raw_best_id,
                raw_best_name=raw_best_name,
                raw_best_score=speaker_score,
            )
            if completion and completion.get("action") == "merge":
                text = completion["text"]
                merged_audio = completion.get("audio_array")
                if merged_audio is not None:
                    audio_array = merged_audio
                raw_best_id = completion.get("raw_best_id")
                raw_best_name = completion.get("raw_best_name")
                speaker_score = float(completion.get("raw_best_score") or 0.0)
                if completion.get("was_prompted"):
                    heard_log_text = completion.get("continuation_text") or heard_log_text
                else:
                    heard_log_text = text
            else:
                signal = turn_completion.classify(text)
                if signal is not None:
                    pending = turn_completion.hold(
                        text=text,
                        audio_array=audio_array,
                        raw_best_id=raw_best_id,
                        raw_best_name=raw_best_name,
                        raw_best_score=speaker_score,
                        signal=signal,
                    )
                    try:
                        consciousness.begin_response_wait(
                            max(0.5, pending.hold_until - time.monotonic()) + 0.5
                        )
                    except Exception as exc:
                        _log.debug("turn completion response-wait failed: %s", exc)
                    final_executed_path = "turn_completion.hold"
                    completed = False
                    return
        else:
            try:
                turn_completion.clear_stale_prompted()
            except Exception:
                pass

        if character_trace is not None:
            character_trace.utterance = str(text or "")
            character_trace.heard_text = str(heard_log_text or "")
            character_trace.raw_best_id = _safe_int(raw_best_id)
            character_trace.raw_best_name = raw_best_name
            character_trace.speaker_score = _safe_round_score(speaker_score)

        specific_forget_target_for_turn = forgetting.extract_specific_forget_target(text)

        echo_cancel.start_sequence()
        sequence_started = True

        # ── Voice acceptance: hard + session-sticky soft threshold ─────────────
        # Short/noisy utterances cause wild score variance (0.60–0.85 for the
        # same person within a single conversation). Mirror human identity
        # continuity: once someone is confirmed this session, accept subsequent
        # low-score utterances from the same person as long as they clear the
        # softer floor. New speakers still need the hard threshold because
        # their voice won't match the engaged person.
        hard_threshold = float(config.SPEAKER_ID_SIMILARITY_THRESHOLD)
        soft_threshold = float(getattr(config, "SPEAKER_ID_SOFT_THRESHOLD", 0.60))
        try:
            _recent_engaged = consciousness.get_recent_engagement()
        except Exception:
            _recent_engaged = None

        person_id: Optional[int] = None
        person_name: Optional[str] = None
        sticky_accepted = False
        identity_resolution_override: Optional[str] = None
        if raw_best_id is not None and speaker_score >= hard_threshold:
            person_id = raw_best_id
            person_name = raw_best_name
        elif (
            raw_best_id is not None
            and speaker_score >= soft_threshold
            and _recent_engaged is not None
            and raw_best_id == _recent_engaged.get("person_id")
        ):
            person_id = raw_best_id
            person_name = raw_best_name or _recent_engaged.get("name")
            sticky_accepted = True
            _log.info(
                "[interaction] voice soft-accept under session stickiness — "
                "person_id=%s name=%r score=%.3f (hard=%.2f, soft=%.2f)",
                person_id, person_name, speaker_score, hard_threshold, soft_threshold,
            )
        # else: person_id stays None — truly unknown voice.

        # ── World-state person resolution ──────────────────────────────────────
        # Consciousness runs face ID independently; if speaker ID missed or is absent,
        # fall back to whoever consciousness already identified in world_state.people.
        # Only use this when exactly one identified person is visible (unambiguous).
        ws_identified = []
        try:
            ws_people = world_state.get("people")
            ws_identified = [p for p in ws_people if p.get("person_db_id") is not None]
            ws_person = ws_identified[0] if len(ws_identified) == 1 else None
        except Exception:
            ws_identified = []
            ws_person = None

        # Detect "off-camera unknown voice": speaker-ID found no match AND nobody
        # unknown is visible AND the visible engaged person isn't the speaker
        # (because their voice print would have matched). Treat person_id=None
        # as genuinely unknown rather than mis-attributing to the visible engaged
        # person — this is the hook for Rex's "who's that speaking?" behavior.
        off_camera_unknown = False
        recent_engagement = None
        try:
            recent_engagement = consciousness.get_recent_engagement()
        except Exception:
            recent_engagement = None

        visible_known_by_id = {}
        try:
            visible_known_by_id = {
                int(p["person_db_id"]): p
                for p in ws_identified
                if p.get("person_db_id") is not None
            }
        except Exception:
            visible_known_by_id = {}

        if text_input and person_id is None:
            recent_id = _safe_int((recent_engagement or {}).get("person_id"))
            if (
                recent_id is not None
                and (not visible_known_by_id or recent_id in visible_known_by_id)
            ):
                vis = visible_known_by_id.get(recent_id, {})
                person_id = recent_id
                person_name = (
                    (recent_engagement or {}).get("name")
                    or vis.get("face_id")
                    or vis.get("voice_id")
                )
                _log.info(
                    "[interaction] person resolution: GUI text input recent engagement — "
                    "person_id=%s name=%r",
                    person_id,
                    person_name,
                )
            elif len(visible_known_by_id) == 1:
                person_id, vis = next(iter(visible_known_by_id.items()))
                person_name = vis.get("face_id") or vis.get("voice_id")
                _log.info(
                    "[interaction] person resolution: GUI text input single visible person — "
                    "person_id=%s name=%r",
                    person_id,
                    person_name,
                )

        pending_qa_recent_attribution = False
        if person_id is None and not text_input:
            person_id, person_name, pending_qa_recent_attribution = (
                _pending_question_recent_attribution(
                    person_id=person_id,
                    person_name=person_name,
                    recent_engagement=recent_engagement,
                    raw_best_id=raw_best_id,
                    speaker_score=speaker_score,
                    text=text,
                )
            )

        # Multi-person scenes need a gentler fallback than "unseen stranger."
        # If the top voice candidate is one of the visible known faces, accept
        # it at a weaker floor. If not, keep conversational continuity with the
        # recently engaged visible person instead of derailing into "who said
        # that?" when two known faces are already in frame.
        if person_id is None and len(visible_known_by_id) >= 2:
            multi_floor = float(getattr(config, "SPEAKER_ID_MULTI_VISIBLE_FLOOR", 0.50))
            recent_floor = float(getattr(config, "SPEAKER_ID_MULTI_VISIBLE_RECENT_FLOOR", 0.45))
            if raw_best_id in visible_known_by_id and speaker_score >= multi_floor:
                vis = visible_known_by_id[int(raw_best_id)]
                person_id = int(raw_best_id)
                person_name = raw_best_name or vis.get("face_id") or vis.get("voice_id")
                _log.info(
                    "[interaction] person resolution: multi-visible voice+face attribution — "
                    "person_id=%s name=%r voice_score=%.3f (floor=%.2f)",
                    person_id, person_name, speaker_score, multi_floor,
                )
            else:
                recent_id = (recent_engagement or {}).get("person_id")
                if (
                    recent_id in visible_known_by_id
                    and not _has_unknown_visible_or_recent()
                    and not _other_known_visible_recently(recent_id)
                    # Require the weak voice match to actually point at the recent
                    # person before pinning the turn on them.
                    and _safe_int(raw_best_id) == _safe_int(recent_id)
                    and speaker_score >= recent_floor
                ):
                    vis = visible_known_by_id[int(recent_id)]
                    person_id = int(recent_id)
                    person_name = (
                        (recent_engagement or {}).get("name")
                        or vis.get("face_id")
                        or vis.get("voice_id")
                    )
                    _log.info(
                        "[interaction] person resolution: multi-visible recent speaker fallback — "
                        "person_id=%s name=%r voice_score=%.3f (floor=%.2f, raw_best=%s)",
                        person_id, person_name, speaker_score, recent_floor, raw_best_id,
                    )

        if ws_person is not None:
            ws_pid = ws_person.get("person_db_id")
            ws_name = ws_person.get("face_id") or ws_person.get("voice_id")
            if person_id is None:
                unknown_visible = _has_unknown_visible_or_recent()
                # Speaker-ID missed. Only fall back to ws_person if they are NOT
                # the engaged person — otherwise we'd be claiming the engaged
                # person spoke when the voice didn't actually match them. That
                # scenario is exactly the off-camera unknown case.
                engaged_is_visible = (
                    recent_engagement is not None
                    and recent_engagement.get("person_id") == ws_pid
                )
                # Engaged-and-visible attribution: if the best voice candidate
                # IS the engaged + visible person, even at sub-soft-threshold
                # score, attribute to them. Face presence + soft voice match
                # stack together: each is a weak signal alone, both together
                # are stronger than the soft threshold on voice alone. This
                # prevents "off-camera unknown" misfires when a known speaker's
                # voice happens to score just under the soft floor on a noisy
                # utterance.
                eng_visible_floor = float(
                    getattr(config, "SPEAKER_ID_ENGAGED_VISIBLE_FLOOR", 0.50)
                )
                if (
                    engaged_is_visible
                    and raw_best_id == ws_pid
                    and speaker_score >= eng_visible_floor
                ):
                    person_id = ws_pid
                    person_name = ws_name
                    _log.info(
                        "[interaction] person resolution: engaged+visible attribution — "
                        "person_id=%s name=%r voice_score=%.3f (floor=%.2f)",
                        person_id, person_name, speaker_score, eng_visible_floor,
                    )
                elif (
                    engaged_is_visible
                    and not unknown_visible
                    and not _other_known_visible_recently(ws_pid)
                ):
                    single_visible_continuity_floor = (
                        _single_visible_engaged_continuity_floor(
                            ws_pid=ws_pid,
                            raw_best_id=raw_best_id,
                        )
                    )
                    if (
                        len(visible_known_by_id) == 1
                        # Only attribute to the engaged person if the top voice
                        # candidate is actually them. Otherwise a second person
                        # speaking in a one-on-one frame gets their words pinned
                        # on the engaged person at a sub-threshold score.
                        and _safe_int(raw_best_id) == _safe_int(ws_pid)
                        and speaker_score >= single_visible_continuity_floor
                    ):
                        person_id = ws_pid
                        person_name = ws_name
                        _log.info(
                            "[interaction] person resolution: single visible engaged "
                            "continuity — person_id=%s name=%r voice_score=%.3f "
                            "(floor=%.2f, raw_best=%s)",
                            person_id,
                            person_name,
                            speaker_score,
                            single_visible_continuity_floor,
                            raw_best_id,
                        )
                    # Grief-flow override: Rex just asked this engaged person
                    # a direct question and is awaiting their reply. Voice-ID
                    # can score just below the engaged-visible floor on short
                    # utterances (a single name like "Joe", or noisy audio),
                    # but face match + top-candidate match is plenty of
                    # evidence in this context. Don't divert to off-camera
                    # handling and lose the grief flow's turn.
                    elif _grief_flow_active(ws_pid) and (
                        raw_best_id == ws_pid
                    ):
                        grief_floor = float(
                            getattr(config, "SPEAKER_ID_GRIEF_FLOW_FLOOR", 0.30)
                        )
                        if speaker_score >= grief_floor:
                            person_id = ws_pid
                            person_name = ws_name
                            _log.info(
                                "[interaction] person resolution: grief flow active for "
                                "engaged+visible person %r — attributing despite voice "
                                "score %.3f below engaged+visible floor (grief floor=%.2f)",
                                ws_name, speaker_score, grief_floor,
                            )
                        else:
                            off_camera_unknown = True
                            _log.info(
                                "[interaction] person resolution: speaker-ID missed while engaged "
                                "person %r is visible and no recent unknown face — treating as off-camera "
                                "unknown voice",
                                ws_name,
                            )
                    else:
                        off_camera_unknown = True
                        _log.info(
                            "[interaction] person resolution: speaker-ID missed while engaged "
                            "person %r is visible and no recent unknown face — treating as off-camera "
                            "unknown voice",
                            ws_name,
                        )
                elif engaged_is_visible and unknown_visible:
                    _log.info(
                        "[interaction] person resolution: speaker-ID missed while engaged "
                        "person %r is visible but a newcomer is/was recently visible — "
                        "leaving speaker unknown",
                        ws_name,
                    )
                else:
                    person_id = ws_pid
                    person_name = ws_name
                    _log.info(
                        "[interaction] person resolution: worldstate match — person_id=%s name=%r",
                        person_id, person_name,
                    )
            elif person_id == ws_pid:
                _log.info(
                    "[interaction] person resolution: both agreed — person_id=%s name=%r score=%.3f",
                    person_id, person_name, speaker_score,
                )
                # Auto-refresh voice biometric: when face AND voice both confirm
                # the same person with HIGH voice confidence, append this audio
                # as an additional voice-print row (up to a per-person cap). Over
                # time this builds a more robust multi-sample representation
                # without manual re-enrollment.
                try:
                    _maybe_auto_refresh_voice(person_id, speaker_score, audio_array)
                except Exception as exc:
                    _log.debug("auto voice-refresh skip: %s", exc)
            else:
                face_override = _single_visible_face_voice_override(
                    resolved_person_id=person_id,
                    ws_person=ws_person,
                    visible_known_by_id=visible_known_by_id,
                    has_unknown_visible_or_recent=_has_unknown_visible_or_recent(),
                    speaker_score=speaker_score,
                    hard_threshold=hard_threshold,
                )
                if face_override is not None:
                    prior_person_id = person_id
                    prior_person_name = person_name
                    person_id, person_name = face_override
                    sticky_accepted = False
                    identity_resolution_override = "visible_face_over_voice_conflict"
                    _log.info(
                        "[interaction] person resolution: single visible face overrides "
                        "conflicting voice-ID — person_id=%s name=%r raw_voice=%s/%r "
                        "score=%.3f",
                        person_id,
                        person_name,
                        prior_person_id,
                        prior_person_name,
                        speaker_score,
                    )
                else:
                    _log.info(
                        "[interaction] person resolution: speaker_id match (score=%.3f) vs worldstate "
                        "(ws_pid=%s) — keeping speaker_id result person_id=%s name=%r",
                        speaker_score, ws_pid, person_id, person_name,
                    )
        elif person_id is not None:
            _log.info(
                "[interaction] person resolution: speaker_id match — person_id=%s name=%r score=%.3f",
                person_id, person_name, speaker_score,
            )
        else:
            # Neither face-ID nor speaker-ID matched anyone. If a known person
            # was engaged recently but isn't visible right now, and no unknown
            # face is visible either, this is still an off-camera unknown voice.
            if (
                recent_engagement
                and not _has_unknown_visible_or_recent()
                and not _other_known_visible_recently((recent_engagement or {}).get("person_id"))
                and len(ws_identified) < 2
            ):
                off_camera_unknown = True
                _log.info(
                    "[interaction] person resolution: no face, no voice match, "
                    "but %r was engaged recently — off-camera unknown voice",
                    recent_engagement.get("name"),
                )
            elif recent_engagement and len(ws_identified) >= 2:
                _log.info(
                    "[interaction] person resolution: ambiguous low-score voice "
                    "while multiple known faces visible — not treating as off-camera unknown"
                )
            # Else: falls through to normal enrollment logic below

        pending_last_name_target = _pending_existing_common_first_name_reply_target(text)
        if pending_last_name_target is not None:
            pending_person_id, pending_first_name = pending_last_name_target
            if person_id != pending_person_id:
                _log.info(
                    "[interaction] person resolution: pending last-name reply attributed "
                    "to person_id=%s name=%r despite voice candidate=%s/%r score=%.3f "
                    "prior_person_id=%s",
                    pending_person_id,
                    pending_first_name,
                    raw_best_id,
                    raw_best_name,
                    speaker_score,
                    person_id,
                )
            person_id = pending_person_id
            person_name = pending_first_name
            off_camera_unknown = False
            identity_resolution_override = "pending_last_name_reply"

        game_conversation_lock = _game_suppresses_conversation()
        identity_prompt_consumed = False
        if not game_conversation_lock:
            try:
                identity_prompt_consumed = bool(consciousness.consume_identity_prompt_request())
            except Exception as exc:
                _log.debug("identity prompt request consume failed: %s", exc)
        if identity_prompt_consumed:
            _identity_prompt_until = max(
                _identity_prompt_until,
                time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS,
            )
        identity_prompt_active = bool(
            identity_prompt_consumed or time.monotonic() <= _identity_prompt_until
        )
        has_unknown_visible_now = _has_unknown_visible_person()
        has_unknown_visible_or_recent = has_unknown_visible_now or _has_unknown_visible_or_recent()
        anonymous_speaker_label: Optional[str] = None
        anonymous_speaker_match_score: Optional[float] = None
        speaker_label_for_turn = person_name or anonymous_speaker_label or "user"
        identity_resolution_for_turn = (
            identity_resolution_override
            or _infer_identity_resolution_strategy(
                person_id=person_id,
                raw_best_id=raw_best_id,
                speaker_score=speaker_score,
                hard_threshold=hard_threshold,
                soft_threshold=soft_threshold,
                sticky_accepted=sticky_accepted,
                ws_identified=ws_identified,
                recent_engagement=recent_engagement,
                off_camera_unknown=off_camera_unknown,
            )
        )
        _character_loop_note_speaker(
            character_trace,
            person_id=person_id,
            person_name=person_name,
            speaker_label=speaker_label_for_turn,
            identity_resolution=identity_resolution_for_turn,
            off_camera_unknown=off_camera_unknown,
            visible_known_count=len(visible_known_by_id),
            has_unknown_visible_or_recent=has_unknown_visible_or_recent,
            anonymous_speaker_label=anonymous_speaker_label,
            anonymous_speaker_match_score=anonymous_speaker_match_score,
        )
        voice_group_chatter = _note_voice_turn_for_group_chatter(
            person_id=person_id,
            raw_best_id=raw_best_id,
            raw_best_score=speaker_score,
        )
        group_chatter_active = voice_group_chatter or _audio_group_chatter_active()
        if _should_ignore_idle_background_speech(
            from_idle_activation=from_idle_activation,
            person_id=person_id,
            has_unknown_visible=has_unknown_visible_now,
            identity_prompt_active=identity_prompt_active,
            text_input=text_input,
            text=text,
        ) and not game_conversation_lock:
            _log.info(
                "[interaction] ignoring no-wake IDLE speech from unrecognized/off-camera "
                "source text=%r raw_best=%s score=%.3f visible_known=%d",
                text,
                raw_best_id,
                speaker_score,
                len(visible_known_by_id),
            )
            try:
                state_module.set_state(State.IDLE)
            except Exception:
                pass
            final_executed_path = "ignored.idle_background_speech"
            completed = False
            return

        if (
            group_chatter_active
            and person_id is None
            and not identity_prompt_active
            and command_parser.parse(text) is None
            and not game_conversation_lock
        ):
            _log.info(
                "[interaction] ignoring unknown speech during group chatter "
                "text=%r raw_best=%s score=%.3f",
                text,
                raw_best_id,
                speaker_score,
            )
            final_executed_path = "ignored.group_chatter"
            completed = False
            return

        if not text_input:
            try:
                consciousness.note_speaker_gaze_intent(
                    person_id,
                    unknown_voice=(person_id is None or off_camera_unknown),
                    reason="off_camera_unknown" if off_camera_unknown else "speech",
                )
            except Exception as exc:
                _log.debug("speaker gaze intent note failed: %s", exc)

        anonymous_speaker_label, anonymous_speaker_match_score = _resolve_anonymous_speaker_slot(
            audio_array,
            person_id=person_id,
            raw_best_id=raw_best_id,
            raw_best_name=raw_best_name,
            raw_best_score=speaker_score,
        )
        if anonymous_speaker_label:
            speaker_label_for_turn = anonymous_speaker_label
            _character_loop_note_speaker(
                character_trace,
                person_id=person_id,
                person_name=person_name,
                speaker_label=speaker_label_for_turn,
                identity_resolution=identity_resolution_for_turn,
                off_camera_unknown=off_camera_unknown,
                visible_known_count=len(visible_known_by_id),
                has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                anonymous_speaker_label=anonymous_speaker_label,
                anonymous_speaker_match_score=anonymous_speaker_match_score,
            )

        # Any non-empty user utterance means we should stop waiting for a reply.
        try:
            consciousness.clear_response_wait()
        except Exception:
            pass

        # User spoke — any queued presence reaction about this person is now stale.
        # Drop it so Rex doesn't narrate someone's "departure" immediately after they spoke.
        try:
            ws_people_now = world_state.get("people")
            for p in ws_people_now:
                pid = p.get("person_db_id")
                if pid is not None:
                    speech_queue.drop_by_tag(f"presence:{pid}")
                slot_id = p.get("id")
                if slot_id:
                    speech_queue.drop_by_tag(f"presence:{slot_id}")
        except Exception:
            pass

        # Consciousness asked an unknown person for their name. Its reply request
        # is consumed before the no-wake IDLE filter above so immediate name
        # replies are not mistaken for background speech.

        relationship_prompt_consumed = False
        prompted_identity_ack_text: Optional[str] = None
        heard_turn_recorded = False

        def _record_heard_turn_once() -> None:
            nonlocal heard_turn_recorded
            if heard_turn_recorded:
                return
            heard_turn_recorded = True
            speaker_label = person_name or anonymous_speaker_label or "user"
            heard_speaker_name = person_name or anonymous_speaker_label
            conv_memory.add_to_transcript(speaker_label, heard_log_text)
            conv_log.log_heard(heard_speaker_name, heard_log_text)
            print(f"[HEARD] {speaker_label}: {heard_log_text}", flush=True)
            _log.info(
                "[interaction] speech segment — speaker=%r person_id=%s text=%r",
                speaker_label, person_id, text,
            )
            _note_session_person_turn(person_id)

        if _turn_should_defer_identity_prompts(text):
            _clear_pending_identity_prompts("direct_turn")

        identity_match_response, identity_match_person_id, identity_match_name = (None, None, None)
        if not game_conversation_lock:
            identity_match_response, identity_match_person_id, identity_match_name = (
                _handle_pending_identity_match_confirmation(text, audio_array)
            )
        if identity_match_response:
            if identity_match_person_id is not None:
                person_id = identity_match_person_id
                person_name = identity_match_name
                _session_person_ids.add(identity_match_person_id)
                _retire_anonymous_speaker_slot(
                    anonymous_speaker_label,
                    person_id=person_id,
                    person_name=person_name,
                )
                _character_loop_note_speaker(
                    character_trace,
                    person_id=person_id,
                    person_name=person_name,
                    speaker_label=person_name or "user",
                    identity_resolution="identity_match_confirmation",
                    off_camera_unknown=False,
                    visible_known_count=len(visible_known_by_id),
                    has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                )
            _identity_prompt_until = 0.0
            _record_heard_turn_once()
            response_text = identity_match_response
            final_executed_path = "identity.match_confirmation"
            _speak_blocking(
                identity_match_response,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", identity_match_response)
            conv_log.log_rex(identity_match_response)
            _session_exchange_count += 1
            _register_rex_utterance(identity_match_response)
            return

        prompted_name_confirm_response, prompted_name_confirm_id, prompted_name_confirm_name = (None, None, None)
        if not game_conversation_lock:
            prompted_name_confirm_response, prompted_name_confirm_id, prompted_name_confirm_name = (
                _handle_pending_prompted_name_confirmation(text, audio_array)
            )
        if prompted_name_confirm_response:
            if prompted_name_confirm_id is not None:
                person_id = prompted_name_confirm_id
                person_name = prompted_name_confirm_name
                _session_person_ids.add(prompted_name_confirm_id)
                _character_loop_note_speaker(
                    character_trace,
                    person_id=person_id,
                    person_name=person_name,
                    speaker_label=person_name or "user",
                    identity_resolution="prompted_identity_confirmation",
                    off_camera_unknown=False,
                    visible_known_count=len(visible_known_by_id),
                    has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                )
            _record_heard_turn_once()
            response_text = prompted_name_confirm_response
            final_executed_path = "identity.prompted_name_confirmation"
            _speak_blocking(
                prompted_name_confirm_response,
                emotion="happy" if prompted_name_confirm_id is not None else "curious",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", prompted_name_confirm_response)
            conv_log.log_rex(prompted_name_confirm_response)
            _session_exchange_count += 1
            _register_rex_utterance(prompted_name_confirm_response)
            return

        common_name_response, common_name_person_id, common_name_full = (None, None, None)
        if not game_conversation_lock:
            common_name_response, common_name_person_id, common_name_full = (
                _handle_common_first_name_last_name_reply(text, audio_array)
            )
        if common_name_response:
            if common_name_person_id is not None:
                person_id = common_name_person_id
                person_name = common_name_full
                _session_person_ids.add(common_name_person_id)
                _retire_anonymous_speaker_slot(
                    anonymous_speaker_label,
                    person_id=person_id,
                    person_name=person_name,
                )
                _character_loop_note_speaker(
                    character_trace,
                    person_id=person_id,
                    person_name=person_name,
                    speaker_label=person_name or "user",
                    identity_resolution="common_first_name_disambiguation",
                    off_camera_unknown=False,
                    visible_known_count=len(visible_known_by_id),
                    has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                )
            _identity_prompt_until = 0.0
            _record_heard_turn_once()
            response_text = common_name_response
            final_executed_path = "identity.common_first_name_reply"
            _speak_blocking(
                common_name_response,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", common_name_response)
            conv_log.log_rex(common_name_response)
            _session_exchange_count += 1
            _register_rex_utterance(common_name_response)
            return

        common_intro_response = (
            _handle_common_first_name_intro_last_name_reply(text)
            if not game_conversation_lock
            else None
        )
        if common_intro_response:
            _record_heard_turn_once()
            response_text = common_intro_response
            final_executed_path = "identity.common_intro_reply"
            _speak_blocking(
                common_intro_response,
                emotion="happy",
                pre_beat_ms=150,
                post_beat_ms_override=300,
            )
            conv_memory.add_to_transcript("Rex", common_intro_response)
            conv_log.log_rex(common_intro_response)
            _session_exchange_count += 1
            _register_rex_utterance(common_intro_response)
            return

        existing_common_name_response = (
            _handle_existing_common_first_name_last_name_reply(text)
            if not game_conversation_lock
            else None
        )
        if existing_common_name_response:
            _record_heard_turn_once()
            response_text = existing_common_name_response
            final_executed_path = "identity.existing_common_first_name_reply"
            _speak_blocking(
                existing_common_name_response,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", existing_common_name_response)
            conv_log.log_rex(existing_common_name_response)
            _session_exchange_count += 1
            _register_rex_utterance(existing_common_name_response)
            return

        self_identified_name = _extract_self_identified_name(text)
        if (
            self_identified_name
            and person_id is not None
            and has_unknown_visible_or_recent
            and not _same_person_name(self_identified_name, person_name)
            and not game_conversation_lock
        ):
            _log.info(
                "[identity] explicit self-introduction %r conflicts with weak/current "
                "speaker label %r while newcomer is visible/recent — treating as newcomer",
                self_identified_name,
                person_name,
            )
            person_id = None
            person_name = None
            off_camera_unknown = False

        if (
            self_identified_name
            and person_id is None
            and has_unknown_visible_or_recent
            and not game_conversation_lock
        ):
            prior_engagement = recent_engagement
            if _maybe_ask_identity_match_confirmation(
                self_identified_name,
                audio_array,
                prior_engagement=prior_engagement,
                anonymous_speaker_label=anonymous_speaker_label,
                enroll_unknown_face=has_unknown_visible_now or bool(prior_engagement),
                source="self_introduction",
            ):
                return

            enrolled_id = _enroll_new_person(
                self_identified_name,
                audio_array,
                enroll_unknown_face=has_unknown_visible_now or bool(prior_engagement),
            )
            if enrolled_id is not None:
                person_id = enrolled_id
                person_name = self_identified_name
                _retire_anonymous_speaker_slot(
                    anonymous_speaker_label,
                    person_id=person_id,
                    person_name=person_name,
                )
                _character_loop_note_speaker(
                    character_trace,
                    person_id=person_id,
                    person_name=person_name,
                    speaker_label=person_name or "user",
                    identity_resolution="self_identified_enrollment",
                    off_camera_unknown=False,
                    visible_known_count=len(visible_known_by_id),
                    has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                )
                _identity_prompt_until = 0.0
                _pending_offscreen_identify = None
                try:
                    rel_ctx = consciousness.consume_relationship_prompt_request()
                    if rel_ctx is not None:
                        consciousness.note_relationship_slot_handled(str(rel_ctx.get("slot_id") or ""))
                except Exception:
                    pass

                relationship = _extract_self_relationship_to_engaged(
                    text,
                    (prior_engagement or {}).get("name"),
                )
                if relationship and prior_engagement and prior_engagement.get("person_id") != enrolled_id:
                    try:
                        from memory import social as social_memory
                        social_memory.save_relationship(
                            from_person_id=enrolled_id,
                            to_person_id=int(prior_engagement["person_id"]),
                            relationship=relationship,
                            described_by=enrolled_id,
                        )
                    except Exception as exc:
                        _log.warning("self-intro relationship save failed: %s", exc)
                elif prior_engagement and prior_engagement.get("person_id") != enrolled_id:
                    _pending_post_greet_relationship[0] = {
                        "prior_engaged_id": prior_engagement["person_id"],
                        "prior_engaged_name": prior_engagement.get("name"),
                        "newcomer_person_id": enrolled_id,
                        "newcomer_name": self_identified_name,
                    }
                _mark_single_name_for_later_last_name(enrolled_id, self_identified_name)

                try:
                    consciousness.mark_engagement(enrolled_id)
                    consciousness.note_person_spoke(enrolled_id)
                    consciousness.note_person_greeted_this_session(enrolled_id)
                except Exception:
                    pass

                speaker_label = person_name or "user"
                conv_memory.add_to_transcript(speaker_label, text)
                conv_log.log_heard(person_name, text)
                print(f"[HEARD] {speaker_label}: {text}", flush=True)
                _log.info(
                    "[interaction] visible newcomer self-introduction — speaker=%r person_id=%s text=%r relationship_to_prior=%r",
                    speaker_label,
                    person_id,
                    text,
                    relationship,
                )

                first = _first_name_or(self_identified_name, self_identified_name)
                prior_first = _first_name_or((prior_engagement or {}).get("name"))
                special_intro_ack = person_specials.special_intro_ack(
                    self_identified_name
                )
                if special_intro_ack:
                    ack_text = special_intro_ack
                elif relationship and prior_first:
                    rel_words = relationship.replace("_", " ")
                    ack_text = (
                        f"{first}. Filed. Relationship to {prior_first}: {rel_words}. That explains at least "
                        "three suspicious data points."
                    )
                else:
                    ack_text = f"{first}. Filed under 'new biological, probably trouble.'"
                try:
                    special_prompt = _special_intro_prompt(self_identified_name)
                    ack_text = llm.get_response(
                        special_prompt
                        or (
                            f"You just learned a visible newcomer's name is {self_identified_name}. "
                            f"{('They said their relationship to ' + prior_first + ' is ' + relationship + '.') if relationship and prior_first else ''} "
                            f"In ONE short in-character Rex line, acknowledge them by name. "
                            f"Do not ask another question."
                        )
                    ) or ack_text
                except Exception as exc:
                    _log.debug("self-intro ack generation failed: %s", exc)
                _speak_blocking(ack_text, emotion="happy", pre_beat_ms=100, post_beat_ms_override=200)
                conv_memory.add_to_transcript("Rex", ack_text)
                conv_log.log_rex(ack_text)
                _session_exchange_count += 1
                _register_rex_utterance(ack_text)
                return

        # If the engaged person answers an unknown-face/off-camera moment with
        # an actual introduction ("this is my dad, Jeff"), let the dedicated
        # introduction flow consume it before generic identity handling. Speaker
        # ID often misses the introducer here, so fall back to recent engagement.
        intro_introducer_id = person_id
        intro_introducer_name = person_name
        if intro_introducer_id is None and recent_engagement is not None:
            recent_id = recent_engagement.get("person_id")
            try:
                intro_introducer_id = int(recent_id) if recent_id is not None else None
            except (TypeError, ValueError):
                intro_introducer_id = None
            intro_introducer_name = recent_engagement.get("name")
        if intro_introducer_id is not None and not game_conversation_lock:
            has_unknown_for_intro = has_unknown_visible_now
            parsed_intro = introductions.detect(
                text,
                has_unknown_face=has_unknown_for_intro,
            )
            if parsed_intro.is_introduction and (
                parsed_intro.subject_kind == "pet"
                or has_unknown_for_intro
                or bool(parsed_intro.name)
                or bool(parsed_intro.relationship)
            ):
                intro_response = _handle_introduction_parse(
                    parsed_intro,
                    introducer_id=intro_introducer_id,
                    introducer_name=intro_introducer_name or f"person_{intro_introducer_id}",
                    visible_newcomer=has_unknown_for_intro,
                )
                off_camera_unknown = False
                rel_ctx_for_intro = consciousness.consume_relationship_prompt_request()
                if rel_ctx_for_intro is not None:
                    relationship_prompt_consumed = True
                    consciousness.note_relationship_slot_handled(
                        str(rel_ctx_for_intro.get("slot_id") or "")
                    )
                if intro_response:
                    _record_heard_turn_once()
                    _speak_blocking(
                        intro_response,
                        emotion="happy",
                        pre_beat_ms=150,
                        post_beat_ms_override=300,
                    )
                    conv_memory.add_to_transcript("Rex", intro_response)
                    conv_log.log_rex(intro_response)
                    _session_exchange_count += 1
                    _register_rex_utterance(intro_response)
                    return

        # Consciousness asked the ENGAGED person about an unknown newcomer.
        # Try to extract {name, relationship}, enroll the newcomer using the
        # current unknown face, and save the relationship edge.
        rel_ctx = (
            consciousness.consume_relationship_prompt_request()
            if not game_conversation_lock
            else None
        )
        relationship_prompt_consumed = relationship_prompt_consumed or rel_ctx is not None
        if rel_ctx is not None:
            relationship_response = _handle_relationship_reply(
                rel_ctx, text, person_id, person_name
            )
            if relationship_response:
                _record_heard_turn_once()
                _speak_blocking(
                    relationship_response,
                    emotion="happy",
                    pre_beat_ms=150,
                    post_beat_ms_override=300,
                )
                conv_memory.add_to_transcript("Rex", relationship_response)
                conv_log.log_rex(relationship_response)
                _session_exchange_count += 1
                _register_rex_utterance(relationship_response)
                return

        intro_voice_response = _handle_intro_voice_capture(
            text,
            audio_array,
            person_id,
            raw_best_id,
            speaker_score,
        )
        if intro_voice_response:
            _record_heard_turn_once()
            _speak_blocking(
                intro_voice_response,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", intro_voice_response)
            conv_log.log_rex(intro_voice_response)
            _session_exchange_count += 1
            _register_rex_utterance(intro_voice_response)
            return

        intro_followup_response = _handle_intro_followup_answer(text)
        if intro_followup_response:
            _record_heard_turn_once()
            _speak_blocking(
                intro_followup_response,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", intro_followup_response)
            conv_log.log_rex(intro_followup_response)
            _session_exchange_count += 1
            _register_rex_utterance(intro_followup_response)
            return

        if _pending_introduction is not None:
            if introductions.context_fresh(_pending_introduction):
                expected_id = _pending_introduction.get("introducer_id")
                if person_id is None or expected_id is None or person_id == expected_id:
                    parsed_intro = introductions.parse_pending_answer(
                        text,
                        default_relationship=_pending_introduction.get("relationship"),
                    )
                    if parsed_intro.is_introduction:
                        intro_response = _handle_introduction_parse(
                            parsed_intro,
                            introducer_id=int(_pending_introduction["introducer_id"]),
                            introducer_name=_pending_introduction.get("introducer_name") or (person_name or "friend"),
                            visible_newcomer=bool(_pending_introduction.get("visible_newcomer", True)),
                        )
                        if intro_response:
                            _record_heard_turn_once()
                            _speak_blocking(
                                intro_response,
                                emotion="happy",
                                pre_beat_ms=150,
                                post_beat_ms_override=300,
                            )
                            conv_memory.add_to_transcript("Rex", intro_response)
                            conv_log.log_rex(intro_response)
                            _session_exchange_count += 1
                            _register_rex_utterance(intro_response)
                            return
            else:
                _pending_introduction = None

        if person_id is not None and not relationship_prompt_consumed:
            has_unknown_for_intro = has_unknown_visible_now
            parsed_intro = introductions.detect(
                text,
                has_unknown_face=has_unknown_for_intro,
            )
            if parsed_intro.is_introduction and (
                parsed_intro.subject_kind == "pet"
                or has_unknown_for_intro
                or bool(parsed_intro.name)
                or bool(parsed_intro.relationship)
            ):
                intro_response = _handle_introduction_parse(
                    parsed_intro,
                    introducer_id=person_id,
                    introducer_name=person_name or f"person_{person_id}",
                    visible_newcomer=has_unknown_for_intro,
                )
                if intro_response:
                    _record_heard_turn_once()
                    _speak_blocking(
                        intro_response,
                        emotion="happy",
                        pre_beat_ms=150,
                        post_beat_ms_override=300,
                    )
                    conv_memory.add_to_transcript("Rex", intro_response)
                    conv_log.log_rex(intro_response)
                    _session_exchange_count += 1
                    _register_rex_utterance(intro_response)
                    return

        existing_common_name_prompt = _maybe_prompt_existing_common_first_name(
            person_id,
            person_name,
            current_text=text,
        )
        if existing_common_name_prompt:
            _record_heard_turn_once()
            _speak_blocking(
                existing_common_name_prompt,
                emotion="curious",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", existing_common_name_prompt)
            conv_log.log_rex(existing_common_name_prompt)
            _session_exchange_count += 1
            _register_rex_utterance(existing_common_name_prompt)
            return

        # ── Address-mode classification ────────────────────────────────────────
        # If the utterance MENTIONS Rex but is not addressed TO him (e.g.
        # "say hi to Rex", "Rex is so fun"), do not generate a reply. Record
        # the mention to world_state.social.being_discussed so the consciousness
        # loop can decide whether to chime in. Skipped when Rex is in a pending
        # identity / face-reveal flow (those handlers expect this turn's text
        # as the user's reply).
        try:
            _addr_identity_window_active = identity_prompt_active
            if (
                getattr(config, "ADDRESS_MODE_ENABLED", True)
                and address_mode.contains_rex_keyword(text)
                and _pending_offscreen_identify is None
                and _pending_face_reveal_confirm is None
                and not _addr_identity_window_active
                # parser handles obvious commands directly; treat as direct.
                and command_parser.parse(text) is None
            ):
                ctx_bits: list[str] = []
                try:
                    crowd = world_state.get("crowd") or {}
                    cnt = crowd.get("count")
                    if isinstance(cnt, int):
                        ctx_bits.append(f"{cnt} people present")
                    dom = crowd.get("dominant_speaker")
                    if dom:
                        ctx_bits.append(f"dominant speaker slot {dom}")
                except Exception:
                    pass
                if person_name:
                    ctx_bits.append(f"speaker is {person_name}")
                addr = address_mode.classify(text, context="; ".join(ctx_bits))
                _log.info(
                    "[interaction] address_mode=%s sentiment=%s rule=%s text=%r",
                    addr.label, addr.sentiment, addr.rule, text[:120],
                )
                if addr.label in (
                    address_mode.ADDRESS_REFERENTIAL,
                    address_mode.ADDRESS_INSTRUCTIONAL,
                ):
                    _record_being_discussed(
                        text=text,
                        label=addr.label,
                        sentiment=addr.sentiment,
                        speaker_id=person_id,
                        speaker_name=person_name,
                    )
                    return
                # ADDRESS_UNRELATED → fall through to LLM; the model will
                # handle the coincidental reference naturally.
        except Exception as exc:
            _log.debug("[interaction] address_mode error (continuing): %s", exc)

        # Rex previously asked "who's that speaking?" about an off-camera unknown
        # voice. If this reply came from the engaged person and names the
        # off-camera speaker, enroll their voice (using the STORED audio of the
        # original utterance, not this engaged-person audio) and save a
        # relationship edge if the engaged person also stated one. The
        # off-camera person may also answer directly with a bare name like "JT".
        pending_offscreen_snapshot = (
            dict(_pending_offscreen_identify)
            if isinstance(_pending_offscreen_identify, dict)
            else {}
        )
        had_pending_offscreen_identify = bool(pending_offscreen_snapshot)
        if had_pending_offscreen_identify:
            _record_heard_turn_once()
        consumed_offscreen_identify, _offscreen_ack = _handle_pending_offscreen_identify_reply(
            text,
            person_id=person_id,
            person_name=person_name,
            audio_array=audio_array,
            anonymous_speaker_label=anonymous_speaker_label,
        )
        if consumed_offscreen_identify:
            answered_offscreen_question = {
                "question_key": "off_camera_identity",
                "question_text": str(
                    pending_offscreen_snapshot.get("question_text")
                    or "Who was that off-camera?"
                ),
                "answer_text": text,
                "depth_level": 1,
                "source": "offscreen_identify",
            }
            try:
                topic_thread.note_user_turn(
                    text,
                    person_id,
                    answered_question=answered_offscreen_question,
                )
            except Exception as exc:
                _log.debug("topic thread off-camera reply update failed: %s", exc)
            try:
                question_budget.note_user_turn(
                    text,
                    person_id,
                    answered_question=answered_offscreen_question,
                )
            except Exception as exc:
                _log.debug("question budget off-camera reply update failed: %s", exc)
            try:
                end_thread.note_user_turn(
                    text,
                    person_id,
                    answered_question=answered_offscreen_question,
                )
            except Exception as exc:
                _log.debug("end-thread off-camera reply update failed: %s", exc)
            response_text = _offscreen_ack
            completed = bool(_offscreen_ack)
            final_executed_path = "identity.offscreen_identify_reply"
            return

        # Face-reveal confirmation handler: Rex asked "is that you, X?" or
        # "are you on my left or my right?" — parse this reply and, if the
        # person (or the engaged person) confirms, bind the face to X and fire
        # the surprise+roast reaction.
        if _pending_face_reveal_confirm is not None:
            pending_fr = _pending_face_reveal_confirm
            now_mono = time.monotonic()
            fr_ttl = float(getattr(config, "FACE_REVEAL_CONFIRM_WINDOW_SECS", 30.0))
            if (now_mono - pending_fr["asked_at"]) > fr_ttl:
                _log.info("[interaction] face-reveal confirm window expired — clearing")
                _pending_face_reveal_confirm = None
            else:
                try:
                    fr_parsed = llm.extract_face_reveal_answer(text)
                except Exception as exc:
                    _log.debug("face-reveal extract error: %s", exc)
                    fr_parsed = {"intent": None}
                fr_intent = fr_parsed.get("intent")
                candidate_slots = pending_fr.get("candidates", [])

                chosen_encoding = None
                if pending_fr.get("mode") == "binary" and fr_intent in ("yes", "no"):
                    if fr_intent == "yes" and candidate_slots:
                        chosen_encoding = candidate_slots[0]["encoding"]
                    elif fr_intent == "no":
                        _face_reveal_declined.add(pending_fr["person_id"])
                        _log.info(
                            "[interaction] face-reveal declined for person_id=%s — won't re-ask this session",
                            pending_fr["person_id"],
                        )
                        _pending_face_reveal_confirm = None
                elif pending_fr.get("mode") == "lateral" and fr_intent in ("left", "right"):
                    # "left" from Rex's POV = smaller x in camera frame.
                    sorted_by_x = sorted(candidate_slots, key=lambda c: c["x"])
                    if fr_intent == "left" and sorted_by_x:
                        chosen_encoding = sorted_by_x[0]["encoding"]
                    elif fr_intent == "right" and sorted_by_x:
                        chosen_encoding = sorted_by_x[-1]["encoding"]

                if chosen_encoding is not None:
                    try:
                        people_memory.add_biometric(
                            pending_fr["person_id"], "face", chosen_encoding
                        )
                        _log.info(
                            "[interaction] face-reveal: bound face to person_id=%s name=%r (mode=%s intent=%s)",
                            pending_fr["person_id"], pending_fr["name"],
                            pending_fr["mode"], fr_intent,
                        )
                        # Fire the surprise+roast reaction asynchronously via
                        # the normal response path: skip normal response for
                        # this turn and speak a custom reveal line instead.
                        reveal_name = pending_fr["name"] or "you"
                        first_reveal = _first_name_or(reveal_name, "you")
                        try:
                            reveal_text = llm.get_response(
                                f"You just confirmed the face of someone you previously only "
                                f"knew by voice. Their name is {first_reveal}. You are seeing "
                                f"their face for the FIRST time. In ONE short in-character "
                                f"Rex line, react with surprise at what they look like compared "
                                f"to the mental picture you had from their voice — AND slip in "
                                f"a light roast about the mismatch. Address {first_reveal} by "
                                f"name. One line only."
                            )
                            if reveal_text:
                                _speak_blocking(reveal_text)
                                conv_memory.add_to_transcript("Rex", reveal_text)
                                conv_log.log_rex(reveal_text)
                                _register_rex_utterance(reveal_text)
                                _log.info(
                                    "[interaction] face-reveal reaction: %r", reveal_text,
                                )
                        except Exception as exc:
                            _log.debug("face-reveal reaction error: %s", exc)
                    except Exception as exc:
                        _log.error("face-reveal binding failed: %s", exc)
                    _pending_face_reveal_confirm = None
                    # Skip normal LLM response; the reveal line IS the response.
                    return

        # If we don't recognize the speaker but there is an unknown person visible,
        # attempt self-identification enrollment from this utterance.
        #
        # Also: when Rex has *just asked* an unknown person for their name
        # (_identity_prompt_until > now) and an unknown face is visible, force
        # the enrollment path even if speaker-ID wrongly matched the voice to
        # an existing known person. Without this, a newcomer whose voice is not
        # yet enrolled gets collapsed into the nearest known voice print and
        # their reply never triggers enrollment.
        now_mono = time.monotonic()
        identity_prompt_active = now_mono <= _identity_prompt_until
        has_unknown = _has_unknown_visible_or_recent()
        should_attempt_enroll = (
            (person_id is None and (has_unknown or identity_prompt_active))
            or (identity_prompt_active and has_unknown)
        )
        if should_attempt_enroll:
            allow_bare = identity_prompt_active
            intro_name = _extract_introduced_name(text, allow_bare_name=allow_bare)
            if intro_name:
                # Distinguish "newcomer self-introducing" from "known person
                # describing the newcomer." When the speaker already resolves
                # to a known person AND an unknown face is visible, the
                # utterance is the known person telling Rex who the new face
                # is — so we must NOT bind the speaker's voice to the new
                # person's identity and we must NOT reattribute this turn's
                # speech to the newcomer.
                describing_newcomer = (
                    person_id is not None and has_unknown
                )
                prior_engagement = None
                try:
                    prior_engagement = consciousness.get_recent_engagement()
                except Exception:
                    prior_engagement = None

                if describing_newcomer:
                    # Face-only enrollment for the unknown face. Try to extract
                    # a relationship label from the same utterance; save an
                    # edge if found. Speaker identity (person_id/person_name)
                    # is preserved.
                    parsed_name = intro_name
                    relationship: Optional[str] = None
                    try:
                        parsed = llm.extract_relationship_introduction(
                            text, person_name or "friend"
                        )
                        parsed_name_from_llm, relationship = (
                            _filter_relationship_introduction_evidence(
                                parsed,
                                text,
                                person_name or "friend",
                                source="describe_newcomer",
                            )
                        )
                        parsed_name = parsed_name_from_llm or intro_name
                    except Exception as exc:
                        _log.debug("describe-newcomer extract error: %s", exc)

                    new_id: Optional[int] = None
                    try:
                        new_id, created = people_memory.find_or_create_person(parsed_name)
                    except Exception as exc:
                        _log.warning("describe-newcomer find_or_create_person failed: %s", exc)

                    if new_id is not None:
                        first_inc = config.FAMILIARITY_INCREMENTS.get(
                            "first_enrollment", 0.0
                        )
                        if created and first_inc > 0:
                            people_memory.update_familiarity(new_id, first_inc)
                        try:
                            from vision import camera as _cam_mod
                            from vision import face as _face_mod
                            frame = _cam_mod.capture_still()
                            if frame is not None:
                                if _face_mod.enroll_unknown_face(new_id, frame):
                                    threading.Thread(
                                        target=_face_mod.update_appearance,
                                        args=(new_id, frame.copy()),
                                        daemon=True,
                                        name=f"appearance-enroll-{new_id}",
                                    ).start()
                        except Exception as exc:
                            _log.warning(
                                "describe-newcomer face enroll failed: %s", exc
                            )
                        if relationship:
                            try:
                                from memory import social as social_memory
                                social_memory.save_relationship(
                                    from_person_id=person_id,
                                    to_person_id=new_id,
                                    relationship=relationship,
                                    described_by=person_id,
                                )
                            except Exception as exc:
                                _log.warning(
                                    "describe-newcomer save_relationship failed: %s",
                                    exc,
                                )
                        _log.info(
                            "[interaction] describe-newcomer enrollment: "
                            "speaker=%r (person_id=%s) named the unknown face "
                            "as %r (person_id=%s, relationship=%s); voice NOT "
                            "rebound to newcomer.",
                            person_name, person_id, parsed_name, new_id,
                            relationship or "—",
                        )
                        _episodic_person_enrolled(new_id, parsed_name, created=created)
                        _identity_prompt_until = 0.0
                else:
                    if _maybe_ask_prompted_name_confirmation(
                        intro_name,
                        audio_array,
                        raw_text=text,
                        prior_engagement=prior_engagement,
                        anonymous_speaker_label=anonymous_speaker_label,
                        enroll_unknown_face=bool(prior_engagement),
                        defer_face_enrollment=identity_prompt_active,
                    ):
                        return

                    if _maybe_ask_identity_match_confirmation(
                        intro_name,
                        audio_array,
                        prior_engagement=prior_engagement,
                        anonymous_speaker_label=anonymous_speaker_label,
                        enroll_unknown_face=bool(prior_engagement),
                        defer_face_enrollment=identity_prompt_active,
                        source="prompted_identity",
                    ):
                        return

                    # Newcomer self-introduction. Existing flow.
                    enrolled_id = _enroll_new_person(
                        intro_name,
                        audio_array,
                        enroll_unknown_face=bool(prior_engagement),
                        defer_face_enrollment=identity_prompt_active,
                    )
                    if enrolled_id is not None:
                        if person_id is not None and person_id != enrolled_id:
                            _log.info(
                                "[interaction] enrollment overrode speaker-ID mismatch: "
                                "speaker_id said person_id=%s, identity prompt replied %r → new person_id=%s",
                                person_id, intro_name, enrolled_id,
                            )
                        person_id = enrolled_id
                        person_name = intro_name
                        _retire_anonymous_speaker_slot(
                            anonymous_speaker_label,
                            person_id=person_id,
                            person_name=person_name,
                        )
                        _character_loop_note_speaker(
                            character_trace,
                            person_id=person_id,
                            person_name=person_name,
                            speaker_label=person_name or "user",
                            identity_resolution="prompted_identity_enrollment",
                            off_camera_unknown=False,
                            visible_known_count=len(visible_known_by_id),
                            has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                        )
                        _identity_prompt_until = 0.0
                        prompted_identity_ack_text = _identity_enrollment_ack(intro_name)

                        # Chain into a relationship follow-up if we were just
                        # engaged with someone else — set a flag for the post-
                        # response hook to ask "how do you know <name>?"
                        if prior_engagement and prior_engagement.get("person_id") != enrolled_id:
                            _pending_post_greet_relationship[0] = {
                                "prior_engaged_id": prior_engagement["person_id"],
                                "prior_engaged_name": prior_engagement.get("name"),
                                "newcomer_person_id": enrolled_id,
                                "newcomer_name": intro_name,
                            }
                        _mark_single_name_for_later_last_name(enrolled_id, intro_name)

        # If Rex had an outstanding event follow-up question, treat this utterance
        # as the outcome and close the loop in memory.
        if _awaiting_followup_event:
            repair_move = repair_moves.detect(text)
            followup_repair = (
                repair_move is not None
                and repair_move.get("kind") in {
                    "clarify",
                    "factual",
                    "misunderstood",
                    "bare_negation",
                    "repeat",
                }
            )
            pending_pid = _awaiting_followup_event.get("person_id")
            pending_event_id = _awaiting_followup_event.get("event_id")
            pending_event_name = _awaiting_followup_event.get("event_name")
            pid_matches = (
                pending_pid is None
                or person_id is None
                or pending_pid == person_id
            )
            # A reply that the event never happened ("I never went / didn't go")
            # resolves the follow-up even if it parses as a repair — otherwise Rex
            # keeps re-asking about something the user says they didn't do.
            did_not_happen = _followup_event_did_not_happen(text)
            # Don't badger: after a bounded number of unresolved (repair) turns,
            # give up so the follow-up stops being re-injected into the agenda.
            holds = int(_awaiting_followup_event.get("holds") or 0)
            hold_cap = int(getattr(config, "FOLLOWUP_MAX_HELD_OPEN_TURNS", 1) or 0)
            give_up = followup_repair and holds >= hold_cap
            if pending_event_id is not None and pid_matches and (
                did_not_happen or give_up or not followup_repair
            ):
                try:
                    if not did_not_happen and events_memory.looks_like_cancellation(text):
                        target_pid = person_id if person_id is not None else pending_pid
                        event_hint = {
                            "id": int(pending_event_id),
                            "event_name": pending_event_name or "",
                        }
                        labels = _cancel_stale_event_memory(
                            target_pid,
                            text,
                            event_hint=event_hint,
                        )
                        if labels:
                            event_cancellation_ack = _event_cancellation_ack(
                                labels,
                                target_pid,
                            )
                    else:
                        # Real answer, an "I didn't go", or a give-up: close the
                        # loop in memory so this event stops surfacing.
                        events_memory.mark_followed_up(int(pending_event_id), text.strip())
                    _awaiting_followup_event = None
                    if did_not_happen or give_up:
                        _log.info(
                            "[interaction] follow-up resolved without an outcome — "
                            "event_id=%s reason=%s text=%r",
                            pending_event_id,
                            "did_not_happen" if did_not_happen else "gave_up_after_holds",
                            text,
                        )
                except Exception as exc:
                    _log.debug("follow-up outcome write failed: %s", exc)
            elif followup_repair:
                _awaiting_followup_event["holds"] = holds + 1
                _log.info(
                    "[interaction] follow-up outcome held open because user requested repair — "
                    "event_id=%s kind=%s holds=%d text=%r",
                    pending_event_id,
                    repair_move.get("kind"),
                    holds + 1,
                    text,
                )

        if person_id is not None:
            _session_person_ids.add(person_id)
            voice_name = person_name or f"person_{person_id}"
            print(f"[VOICE] Known voice detected: {voice_name} (person_id={person_id})", flush=True)
            # Mark this person as Rex's current conversational partner. Consciousness
            # uses this to suppress presence reactions about them while engaged.
            try:
                consciousness.mark_engagement(person_id)
                consciousness.note_person_spoke(person_id)
            except Exception:
                pass
        else:
            unknown_voice_label = anonymous_speaker_label or "unknown"
            print(f"[VOICE] Unknown voice detected: {unknown_voice_label}", flush=True)

        _record_heard_turn_once()

        if prompted_identity_ack_text:
            try:
                consciousness.note_person_greeted_this_session(person_id)
            except Exception:
                pass
            completed = _speak_blocking(
                prompted_identity_ack_text,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            response_text = prompted_identity_ack_text
            final_executed_path = "identity.prompted_enrollment_ack"
            suppress_memory_learning = True
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", prompted_identity_ack_text)
            conv_log.log_rex(prompted_identity_ack_text)
            _session_exchange_count += 1
            _register_rex_utterance(prompted_identity_ack_text)
            return

        memory_wipe_response = _handle_pending_memory_wipe_confirmation(text, person_id)
        if memory_wipe_response:
            response_text = memory_wipe_response
            final_executed_path = "memory.pending_wipe_confirmation"
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", memory_wipe_response)
            conv_log.log_rex(memory_wipe_response)
            _session_exchange_count += 1
            _register_rex_utterance(memory_wipe_response)
            return

        if _is_bare_wake_address(text):
            if _bare_wake_should_ask_visible_unknown_identity(
                person_id=person_id,
                identity_prompt_active=(time.monotonic() <= _identity_prompt_until),
                has_unknown_visible_or_recent=has_unknown_visible_or_recent,
                game_conversation_lock=game_conversation_lock,
            ):
                identity_question = _ask_visible_unknown_identity_question(
                    text,
                    recent_engagement=recent_engagement,
                    group_chatter_active=group_chatter_active,
                    source="bare_wake_address",
                )
                if identity_question:
                    final_executed_path = "identity_prompt.visible_unknown_wake"
                    completed = True
                    response_text = identity_question
                    return

            _log.info("[wake_word] transcribed wake address fast-ack text=%r", text)
            _wake_ack()
            final_executed_path = "wake_address.ack"
            completed = True
            response_text = "(wake acknowledgment)"
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            return

        name_update_decision = None
        try:
            explicit_name_decision = action_router.classify_explicit_control(text)
            if (
                explicit_name_decision is not None
                and explicit_name_decision.action == "identity.name_correction"
            ):
                name_update_decision = explicit_name_decision
        except Exception as exc:
            _log.debug("pre-router identity correction classification failed: %s", exc)

        name_update_response = _handle_name_update_request(
            text,
            person_id,
            person_name,
        )
        if name_update_response:
            if name_update_decision is not None:
                try:
                    name_update_context = _action_router_context(
                        text,
                        person_id=person_id,
                        person_name=person_name,
                        raw_best_id=raw_best_id,
                        raw_best_name=raw_best_name,
                        speaker_score=speaker_score,
                        recent_engagement=recent_engagement,
                        off_camera_unknown=off_camera_unknown,
                        identity_prompt_active=identity_prompt_active,
                    )
                    name_update_audit = _new_router_audit(text, name_update_context)
                    _router_audit_note_decision(name_update_audit, name_update_decision)
                    _log_router_audit(
                        name_update_audit,
                        "fast_local_takeover.identity.name_correction",
                        spoken_text=name_update_response,
                    )
                    router_audit = name_update_audit
                except Exception as exc:
                    _log.debug("pre-router identity correction audit failed: %s", exc)
            response_text = name_update_response
            final_executed_path = "fast_local_takeover.identity.name_correction"
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", name_update_response)
            conv_log.log_rex(name_update_response)
            _session_exchange_count += 1
            _register_rex_utterance(name_update_response)
            return

        # Active games get first claim on ordinary utterances after explicit
        # corrections/stop commands. Identity prompts otherwise steal roster
        # answers like "Will, Jen, Daniel, and Bret" as mystery voices.
        try:
            from features import games as games_mod
            if games_mod.is_active():
                game_match = command_parser.parse(text)
                command_key = game_match.command_key if game_match is not None else None
                normalized_game_text = " ".join(text.lower().strip().split())
                if game_match is None and normalized_game_text in {"quit", "end", "end game", "quit game"}:
                    command_key = "stop_game"
                elif (
                    command_key == "dj_stop"
                    and normalized_game_text in {"stop", "quit", "end", "stop playing"}
                ):
                    command_key = "stop_game"
                elif command_key == "dj_skip" and normalized_game_text == "skip":
                    command_key = None

                game_escape_commands = {
                    "stop_game", "sleep", "shutdown", "quiet_mode", "wake_up",
                    "dj_stop", "dj_skip", "volume_up", "volume_down",
                }
                if command_key not in game_escape_commands:
                    game_response = games_mod.handle_input(text, person_id, audio_array)
                    completed = _speak_blocking(game_response)
                    response_text = game_response
                    final_executed_path = "game.active_turn.early"
                    suppress_memory_learning = True
                    if completed:
                        games_mod.on_response_spoken()
                        game_after_audio_path = games_mod.consume_pending_audio_after_response()
                    conv_memory.add_to_transcript("Rex", game_response)
                    conv_log.log_rex(game_response)
                    _session_exchange_count += 1
                    _register_rex_utterance(game_response)
                    assistant_asked_question = _assistant_asked_question(game_response)
                    if sequence_started:
                        echo_cancel.end_sequence(
                            flush=False,
                            tail_secs=_reply_playback_tail_secs(assistant_asked_question),
                        )
                        sequence_started = False
                        if game_after_audio_path and not _interrupted.is_set():
                            speech_queue.enqueue_audio_file(
                                game_after_audio_path,
                                priority=1,
                                tag="game:after_audio",
                            )
                    try:
                        interoception.record_interaction()
                    except Exception as exc:
                        _log.debug("active game interoception error: %s", exc)
                    return
        except Exception as exc:
            _log.debug("early active game routing failed: %s", exc)

        music_offer_response = _handle_pending_music_offer_reply(person_id, text)
        if music_offer_response:
            response_text = music_offer_response
            final_executed_path = "music.pending_offer_reply"
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", music_offer_response)
            conv_log.log_rex(music_offer_response)
            _session_exchange_count += 1
            _register_rex_utterance(music_offer_response)
            return

        pending_question_for_turn = _latest_pending_question(person_id)
        music_pref_response, answered_question = _handle_pending_music_preference_answer(
            person_id,
            text,
            pending_question=pending_question_for_turn,
            identity_prompt_active=identity_prompt_active,
        )
        if music_pref_response:
            response_text = music_pref_response
            final_executed_path = "music.pending_preference_answer"
            try:
                if answered_question:
                    topic_thread.note_answered_question(answered_question)
            except Exception as exc:
                _log.debug("topic thread music preference answer update failed: %s", exc)
            try:
                question_budget.note_user_turn(
                    text,
                    person_id,
                    answered_question=answered_question,
                )
            except Exception as exc:
                _log.debug("question budget music preference update failed: %s", exc)
            try:
                end_thread.note_user_turn(
                    text,
                    person_id,
                    answered_question=answered_question,
                )
            except Exception as exc:
                _log.debug("end thread music preference update failed: %s", exc)
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", music_pref_response)
            conv_log.log_rex(music_pref_response)
            _session_exchange_count += 1
            _register_rex_utterance(music_pref_response)
            return

        if answered_question is None and pending_question_for_turn:
            answered_question = _maybe_capture_pending_qa(
                person_id,
                text,
                identity_prompt_active=identity_prompt_active,
            )
        if answered_question is None:
            answered_question = _maybe_capture_topic_thread_answer(
                person_id,
                text,
                identity_prompt_active=identity_prompt_active,
            )

        router_context = None
        try:
            router_context = _action_router_context(
                text,
                person_id=person_id,
                person_name=person_name,
                raw_best_id=raw_best_id,
                raw_best_name=raw_best_name,
                speaker_score=speaker_score,
                recent_engagement=recent_engagement,
                off_camera_unknown=off_camera_unknown,
                identity_prompt_active=identity_prompt_active,
            )
            router_audit = _new_router_audit(text, router_context)
            dialogue_decision = dialogue_act.classify(
                text,
                router_context,
                person_id=person_id,
            )
            router_context["dialogue_act"] = dialogue_decision.as_context()
            _log.info(
                "[dialogue_act] label=%s conf=%.2f reason=%s frame_source=%s "
                "frame_topic=%r skip_router=%s",
                dialogue_decision.label,
                float(dialogue_decision.confidence or 0.0),
                dialogue_decision.reason,
                dialogue_decision.frame.source if dialogue_decision.frame else None,
                dialogue_decision.frame.topic if dialogue_decision.frame else "",
                dialogue_decision.skip_action_router,
            )

            fast_takeover_response = None
            if not dialogue_decision.skip_action_router:
                fast_takeover_response = _handle_fast_local_takeover(
                    text,
                    person_id=person_id,
                    person_name=person_name,
                    router_context=router_context,
                    dialogue_decision=dialogue_decision,
                    router_audit=router_audit,
                )
            if fast_takeover_response is not None:
                silent_command = _is_silent_command_response(fast_takeover_response)
                response_text = None if silent_command else fast_takeover_response
                final_executed_path = _router_audit_fast_local_final_path(router_audit)
                _log_router_audit(
                    router_audit,
                    final_executed_path,
                    completed=True,
                    spoken_text=None if silent_command else fast_takeover_response,
                    spoken_text_present=False if silent_command else None,
                )
                if final_executed_path and (
                    "directed_look" in final_executed_path
                    or "vision.directed_search" in final_executed_path
                ):
                    suppress_memory_learning = True
                _dismiss_pending_consent_prompts(person_id, text)
                try:
                    consciousness.clear_response_wait()
                except Exception:
                    pass
                if not silent_command:
                    conv_memory.add_to_transcript("Rex", fast_takeover_response)
                    conv_log.log_rex(fast_takeover_response)
                    _session_exchange_count += 1
                    _register_rex_utterance(fast_takeover_response)
                return
            if dialogue_decision.skip_action_router:
                router_decision = action_router.ActionDecision(
                    action="conversation.reply",
                    confidence=dialogue_decision.confidence,
                    args={"dialogue_act": dialogue_decision.label},
                    reason=f"{dialogue_decision.reason}; conversation owns ambiguity",
                )
                _router_audit_note_decision(
                    router_audit,
                    router_decision,
                    text=text,
                    context=router_context,
                    dialogue_decision=dialogue_decision,
                )
                _log.info(
                    "[action_router] skipped blocking router — dialogue_act=%s reason=%s",
                    dialogue_decision.label,
                    dialogue_decision.reason,
                )
            elif bool(getattr(config, "ACTION_ROUTER_EXECUTE_ENABLED", False)):
                router_started = time.monotonic()
                router_decision = action_router.decide(text, router_context)
                _latency_log(turn_start, "action_router", router_started)
                action_router.log_decision(router_decision, router_context, mode="execute")
                _router_audit_note_decision(
                    router_audit,
                    router_decision,
                    text=text,
                    context=router_context,
                    dialogue_decision=dialogue_decision,
                )
            else:
                _router_audit_note_execute_disabled(router_audit)
                action_router.start_shadow_decision(text, router_context)
            router_block_reason = _router_execution_block_reason(
                router_decision,
                text=text,
                context=router_context,
                dialogue_decision=dialogue_decision,
            )
            if router_block_reason is None:
                suppress_memory_learning = True
            elif _conversation_reply_should_skip_memory_learning(text, router_decision):
                suppress_memory_learning = True
                _log.info(
                    "[memory] suppressing extraction for general-knowledge router reply text=%r",
                    text,
                )
            elif _router_nonexecuted_should_suppress_memory_learning(router_decision):
                suppress_memory_learning = True
                _log.info(
                    "[action_router] not executing action=%s confidence=%.2f reason=%s",
                    router_decision.action,
                    float(router_decision.confidence or 0.0),
                    router_block_reason,
                )
        except Exception as exc:
            _log.debug("action router shadow start failed: %s", exc)
        blocked_router_response = _router_blocked_confirmation_response(
            router_decision,
            router_block_reason,
        )
        if blocked_router_response:
            completed = _speak_blocking(blocked_router_response, emotion="neutral")
            response_text = blocked_router_response
            final_executed_path = (
                f"router_confirmation.{router_decision.action if router_decision else 'unknown'}"
            )
            _log_router_audit(
                router_audit,
                final_executed_path,
                completed=completed,
                spoken_text=blocked_router_response,
            )
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", blocked_router_response)
            conv_log.log_rex(blocked_router_response)
            _session_exchange_count += 1
            _register_rex_utterance(blocked_router_response)
            return
        try:
            topic_thread.note_user_turn(
                text, person_id, answered_question=answered_question
            )
        except Exception as exc:
            _log.debug("topic thread user update failed: %s", exc)
        try:
            user_energy.note_user_turn(
                text, person_id, answered_question=answered_question
            )
        except Exception as exc:
            _log.debug("user energy text update failed: %s", exc)

        boundary_person_id = person_id
        if boundary_person_id is None and _EMOTIONAL_BOUNDARY_PAT.search(text):
            try:
                boundary_person_id = consciousness.get_last_memory_hint_target()
            except Exception:
                boundary_person_id = None
            if boundary_person_id is None and recent_engagement and not _has_unknown_visible_or_recent():
                try:
                    boundary_person_id = int(recent_engagement.get("person_id"))
                except (TypeError, ValueError):
                    boundary_person_id = None
            if boundary_person_id is not None:
                _log.info(
                    "[interaction] emotional boundary attributed to recent memory target "
                    "person_id=%s despite uncertain speaker",
                    boundary_person_id,
                )

        boundary_response = _handle_emotional_checkin_boundary(boundary_person_id, text)
        if boundary_response:
            _dismiss_pending_consent_prompts(boundary_person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            completed = _speak_blocking(
                boundary_response,
                emotion="neutral",
                pre_beat_ms=200,
                post_beat_ms_override=300,
            )
            response_text = boundary_response
            final_executed_path = "legacy.emotional_checkin_boundary"
            _log_router_audit(
                router_audit,
                final_executed_path,
                completed=completed,
                spoken_text=boundary_response,
            )
            conv_memory.add_to_transcript("Rex", boundary_response)
            conv_log.log_rex(boundary_response)
            _session_exchange_count += 1
            _register_rex_utterance(boundary_response)
            return

        preference_response = _handle_conversation_boundary(person_id, text)
        if preference_response:
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            completed = _speak_blocking(
                preference_response,
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            response_text = preference_response
            final_executed_path = "legacy.conversation_boundary"
            _log_router_audit(
                router_audit,
                final_executed_path,
                completed=completed,
                spoken_text=preference_response,
            )
            conv_memory.add_to_transcript("Rex", preference_response)
            conv_log.log_rex(preference_response)
            _session_exchange_count += 1
            _register_rex_utterance(preference_response)
            return

        router_takeover_response = _handle_router_takeover_action(
            router_decision,
            text,
            person_id=person_id,
            person_name=person_name,
            raw_best_id=raw_best_id,
            raw_best_name=raw_best_name,
            raw_best_score=speaker_score,
            router_context=router_context,
            dialogue_decision=dialogue_decision,
            router_audit=router_audit,
        )
        if router_takeover_response:
            response_text = router_takeover_response
            final_executed_path = (
                f"router_takeover.{router_decision.action if router_decision else 'unknown'}"
            )
            _log_router_audit(
                router_audit,
                final_executed_path,
                spoken_text=router_takeover_response,
            )
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            conv_memory.add_to_transcript("Rex", router_takeover_response)
            conv_log.log_rex(router_takeover_response)
            _session_exchange_count += 1
            _register_rex_utterance(router_takeover_response)
            return
        if (
            _router_decision_executable(
                router_decision,
                text=text,
                context=router_context,
                dialogue_decision=dialogue_decision,
            )
            and router_decision.action not in {"conversation.reply", "emotional.boundary"}
        ):
            _router_audit_note_result(
                router_audit,
                handler_error="router_takeover_no_response",
            )

        if (
            _router_decision_executable(
                router_decision,
                text=text,
                context=router_context,
                dialogue_decision=dialogue_decision,
            )
            and router_decision.action == "emotional.boundary"
        ):
            router_boundary_person_id = boundary_person_id or person_id
            if router_boundary_person_id is None and recent_engagement and not _has_unknown_visible_or_recent():
                try:
                    router_boundary_person_id = int(recent_engagement.get("person_id"))
                except (TypeError, ValueError):
                    router_boundary_person_id = None
            boundary_response = _handle_router_emotional_boundary(
                router_boundary_person_id,
                text,
                topic_hint=(
                    _session_router_control_topics.get(int(router_boundary_person_id))
                    if router_boundary_person_id is not None
                    else None
                ),
            )
            if boundary_response:
                _dismiss_pending_consent_prompts(router_boundary_person_id, text)
                try:
                    consciousness.clear_response_wait()
                except Exception:
                    pass
                completed = _speak_blocking(
                    boundary_response,
                    emotion="neutral",
                    pre_beat_ms=200,
                    post_beat_ms_override=300,
                )
                response_text = boundary_response
                final_executed_path = "router_takeover.emotional.boundary"
                _log_router_audit(
                    router_audit,
                    final_executed_path,
                    completed=completed,
                    spoken_text=boundary_response,
                )
                conv_memory.add_to_transcript("Rex", boundary_response)
                conv_log.log_rex(boundary_response)
                _session_exchange_count += 1
                _register_rex_utterance(boundary_response)
                return

        if answered_question is None:
            answered_question = _maybe_capture_pending_qa(
                person_id,
                text,
                identity_prompt_active=identity_prompt_active,
            )
        if answered_question is None:
            answered_question = _maybe_capture_topic_thread_answer(
                person_id,
                text,
                identity_prompt_active=identity_prompt_active,
            )
        try:
            if answered_question:
                topic_thread.note_answered_question(answered_question)
        except Exception as exc:
            _log.debug("topic thread answer update failed: %s", exc)
        try:
            question_budget.note_user_turn(
                text,
                person_id,
                answered_question=answered_question,
            )
        except Exception as exc:
            _log.debug("question budget user update failed: %s", exc)
        try:
            closure = end_thread.note_user_turn(
                text,
                person_id,
                answered_question=answered_question,
            )
            if closure:
                _log.info(
                    "[end_thread] closure cue detected — reason=%s text=%r",
                    closure.get("reason"),
                    text,
                )
        except Exception as exc:
            _log.debug("end-of-thread user update failed: %s", exc)

        # Face-reveal prompt: speaker-ID matched a known person with HIGH
        # confidence, they have NO face biometric yet (voice-only enrollment),
        # and at least one unknown face is visible. Rex asks confirmation
        # instead of auto-binding — yes/no for one unknown, left/right for two.
        if (
            person_id is not None
            and person_id != (recent_engagement or {}).get("person_id")
            and person_id not in _face_reveal_declined
            and _pending_face_reveal_confirm is None
            and speaker_score >= float(getattr(config, "FACE_REVEAL_MIN_SCORE", 0.80))
            and not people_memory.has_face_biometric(person_id)
        ):
            face_reveal_blocked = False
            try:
                if (
                    boundary_memory.is_blocked(person_id, "ask", "face")
                    or boundary_memory.is_blocked(person_id, "ask", "identity")
                ):
                    face_reveal_blocked = True
                    _face_reveal_declined.add(person_id)
                    _log.info(
                        "[boundaries] face-reveal skipped — face/identity ask blocked for person_id=%s",
                        person_id,
                    )
            except Exception as exc:
                _log.debug("face-reveal boundary check failed: %s", exc)
            if face_reveal_blocked:
                pass
            else:
                # Collect unknown faces currently visible with x-position and encoding.
                try:
                    from vision import camera as _cam_mod
                    from vision import face as _face_mod
                    frame = _cam_mod.get_frame()
                    unknown_candidates: list[dict] = []
                    if frame is not None:
                        detected = _face_mod.detect_faces(frame)
                        for det in detected:
                            if _face_mod.identify_face(det["encoding"]) is None:
                                x, _y, w, _h = det["bounding_box"]
                                unknown_candidates.append({
                                    "slot_id": f"face_{x}_{_y}",
                                    "encoding": det["encoding"],
                                    "x": x + w // 2,
                                })
                except Exception as exc:
                    _log.debug("face-reveal candidate capture error: %s", exc)
                    unknown_candidates = []

                if len(unknown_candidates) == 1:
                    mode = "binary"
                    name_for_prompt = _first_name_or(person_name, "you")
                    ask_prompt = (
                        f"You recognize this speaker's VOICE as '{name_for_prompt}' — but "
                        f"you've never seen their face before. Now there's one unfamiliar "
                        f"face in view and it's almost certainly them. In ONE short "
                        f"in-character Rex line, ask {name_for_prompt} to confirm that "
                        f"IS them — express surprise at what they look like compared to "
                        f"the voice you'd imagined. End with a clear yes/no question."
                    )
                elif len(unknown_candidates) == 2:
                    mode = "lateral"
                    name_for_prompt = _first_name_or(person_name, "you")
                    ask_prompt = (
                        f"You recognize this speaker's VOICE as '{name_for_prompt}' but "
                        f"two unfamiliar faces are in view and you can't tell which is "
                        f"{name_for_prompt}. In ONE short in-character Rex line, ask "
                        f"{name_for_prompt} whether they are the one on YOUR LEFT or on "
                        f"YOUR RIGHT — phrase it from your perspective. End with a clear "
                        f"'left or right?' question."
                    )
                else:
                    mode = None
                    _log.info(
                        "[interaction] face-reveal skipped — %d unknown candidates (need 1 or 2)",
                        len(unknown_candidates),
                    )

                if mode is not None:
                    _pending_face_reveal_confirm = {
                        "person_id": person_id,
                        "name": person_name,
                        "mode": mode,
                        "candidates": unknown_candidates,
                        "asked_at": time.monotonic(),
                    }
                    try:
                        q_text = llm.get_response(ask_prompt)
                        if q_text:
                            _speak_blocking(q_text)
                            conv_memory.add_to_transcript("Rex", q_text)
                            conv_log.log_rex(q_text)
                            _register_rex_utterance(q_text)
                            _log.info(
                                "[interaction] face-reveal asked (mode=%s, person_id=%s): %r",
                                mode, person_id, q_text,
                            )
                            # Skip normal LLM response — this turn IS the reveal question.
                            return
                        else:
                            _pending_face_reveal_confirm = None
                    except Exception as exc:
                        _log.debug("face-reveal ask error: %s", exc)
                        _pending_face_reveal_confirm = None

        # Off-camera unknown voice: if the person-resolution pass flagged this
        # utterance as coming from someone we can neither see nor voice-ID, fire
        # Rex's "who's that speaking?" question and store the audio so the
        # engaged person's next reply can enroll the unknown's voice.
        # Skip if a pending identify is already in flight or this is a local command.
        if (
            off_camera_unknown
            and _pending_offscreen_identify is None
            and command_parser.parse(text) is None
        ):
            if _looks_like_background_crosstalk(text):
                _log.info(
                    "[interaction] ignoring off-camera background chatter "
                    "text=%r raw_best=%s score=%.3f",
                    text,
                    raw_best_id,
                    speaker_score,
                )
                final_executed_path = "ignored.off_camera_background_chatter"
                completed = False
                return
            if not _utterance_invites_identity_question(text):
                _log.info(
                    "[interaction] deferring off-camera identity ask; turn should get "
                    "normal handling text=%r raw_best=%s score=%.3f",
                    text,
                    raw_best_id,
                    speaker_score,
                )
                off_camera_unknown = False
            if not off_camera_unknown:
                pass
            elif group_chatter_active:
                _log.info(
                    "[interaction] suppressing off-camera identity ask during "
                    "group chatter; treating utterance as background text=%r",
                    text,
                )
                return
            elif person_id is not None:
                try:
                    if boundary_memory.is_blocked(person_id, "ask", "identity"):
                        _log.info(
                            "[boundaries] off-camera identity ask skipped for person_id=%s",
                            person_id,
                        )
                        off_camera_unknown = False
                except Exception as exc:
                    _log.debug("off-camera identity boundary check failed: %s", exc)
            if not off_camera_unknown:
                pass
            else:
                engaged_name_local = (
                    recent_engagement.get("name") if recent_engagement else None
                ) or ""
                first_name_local = _first_name_or(engaged_name_local, "friend")
                _pending_offscreen_identify = {
                    "audio": audio_array.copy(),
                    "asked_at": time.monotonic(),
                    "prior_engaged_id": (recent_engagement or {}).get("person_id"),
                    "prior_engaged_name": engaged_name_local,
                    "overheard_text": text,
                    "anonymous_speaker_label": anonymous_speaker_label,
                }
                try:
                    q_text = llm.get_response(
                        f"You just heard an UNFAMILIAR voice but you cannot see who said it "
                        f"(no face in view). They said: '{text}'. Your friend "
                        f"'{first_name_local}' is here with you and you were talking "
                        f"with them. In ONE short in-character Rex line, ask who that "
                        f"was off-camera — curious, slightly wary. Address "
                        f"{first_name_local} naturally. Examples: "
                        f"'Who's that, {first_name_local}? I can't see them.', "
                        f"'Hold up — who just chimed in back there?', "
                        f"'Someone's lurking off-camera, {first_name_local}. Friend of yours?' "
                        f"One line ending in a question mark."
                    )
                    if q_text:
                        _pending_offscreen_identify["question_text"] = q_text
                        _speak_blocking(q_text)
                        conv_memory.add_to_transcript("Rex", q_text)
                        conv_log.log_rex(q_text)
                        _register_rex_utterance(q_text)
                        _log.info(
                            "[interaction] off-camera unknown — Rex asked: %r (overheard: %r)",
                            q_text, text,
                        )
                except Exception as exc:
                    _log.debug("off-camera identify ask error: %s", exc)
                # Skip normal LLM response — we don't want to treat the unknown
                # utterance as something Rex should respond to directly. The engaged
                # person's next reply gets consumed by the _pending_offscreen_identify
                # handler above.
                return
        # Newcomer-on-camera: an unknown face is visible AND the speaker's voice
        # didn't match anyone AND this utterance didn't already enroll them via
        # the self-introduction path above. Covers the "camera handoff" case
        # (e.g. Bret hands the device to a friend who says "Hi Rex"): without
        # this branch, the intent classifier hijacks the utterance into
        # query_who_is_speaking and Rex says "no idea, who's asking?" with no
        # follow-up. Here we proactively ask their name (and how they know the
        # recently-engaged person, if any) and open the identity-reply window
        # so the next utterance triggers enrollment + the post-greet
        # relationship chain.
        if (
            person_id is None
            and not identity_prompt_active
            and _has_unknown_visible_person()
            and _pending_face_reveal_confirm is None
            and command_parser.parse(text) is None
        ):
            if _looks_like_background_crosstalk(text):
                _log.info(
                    "[interaction] ignoring visible-unknown background chatter "
                    "text=%r raw_best=%s score=%.3f",
                    text,
                    raw_best_id,
                    speaker_score,
                )
                final_executed_path = "ignored.visible_unknown_background_chatter"
                completed = False
                return
            if not _utterance_invites_identity_question(text):
                _log.info(
                    "[interaction] deferring visible-unknown identity ask; turn should "
                    "get normal handling text=%r raw_best=%s score=%.3f",
                    text,
                    raw_best_id,
                    speaker_score,
                )
            else:
                q_text = _ask_visible_unknown_identity_question(
                    text,
                    recent_engagement=recent_engagement,
                    group_chatter_active=group_chatter_active,
                    source="newcomer_on_camera",
                )
                if group_chatter_active:
                    return
                if q_text:
                    return

        # Empathy classification — kicked off in parallel so the affect/mode
        # directive is ready by the time the LLM-fallback path assembles its
        # system prompt. Result is cached per-person for future turns even if
        # this turn is handled deterministically by the command parser.
        _empathy_thread = None
        active_grief_for_turn = person_id is not None and _grief_flow_active(person_id)
        local_sensitive_result = None
        if getattr(config, "EMPATHY_ENABLED", True) and not active_grief_for_turn:
            local_sensitive_result = _apply_local_sensitive_topic_prepass(person_id, text)
            _audio_for_prosody = audio_array
            def _run_empathy() -> None:
                try:
                    prosody_features = None
                    if getattr(config, "EMPATHY_PROSODY_ENABLED", True):
                        try:
                            prosody_features = prosody.analyze(
                                _audio_for_prosody,
                                sample_rate=config.AUDIO_SAMPLE_RATE,
                                transcript_text=text,
                            )
                        except Exception as exc:
                            _log.debug("prosody.analyze failed: %s", exc)
                            prosody_features = None
                    face_mood = None
                    if person_id is not None and getattr(config, "EMPATHY_FACE_MOOD_MISMATCH_ENABLED", True):
                        try:
                            face_mood = consciousness.get_cached_mood(
                                person_id,
                                max_age_secs=float(
                                    getattr(config, "EMPATHY_FACE_MOOD_CACHE_MAX_AGE_SECS", 180.0)
                                ),
                            )
                        except Exception as exc:
                            _log.debug("cached face mood lookup failed: %s", exc)
                            face_mood = None
                    result = empathy.classify_affect(
                        text,
                        prosody_features=prosody_features,
                        face_mood=face_mood,
                    )
                    if not result:
                        return
                    result = _merge_with_local_sensitive_prepass(
                        result,
                        local_sensitive_result,
                    )
                    try:
                        user_energy.note_user_turn(
                            text,
                            person_id,
                            prosody_features=prosody_features,
                            affect_result=result,
                        )
                    except Exception as exc:
                        _log.debug("user energy prosody/empathy update failed: %s", exc)
                    person_row = None
                    if person_id is not None:
                        try:
                            person_row = people_memory.get_person(person_id)
                        except Exception:
                            person_row = None
                    child_in_scene = False
                    try:
                        ws_people = world_state.get("people") or []
                        child_in_scene = any(
                            p.get("age_estimate") == "child" for p in ws_people
                        )
                    except Exception:
                        pass
                    mode_pack = empathy.select_mode(
                        result, person=person_row, child_in_scene=child_in_scene,
                        person_id=person_id,
                    )
                    empathy.record(person_id, result, mode_pack)
                    _prosody = result.get("prosody") or {}
                    _log.info(
                        "[empathy] affect=%s needs=%s sens=%s invite=%s "
                        "conf=%.2f prosody=%s → mode=%s (%s)",
                        result.get("affect"), result.get("needs"),
                        result.get("topic_sensitivity"),
                        result.get("invitation"), result.get("confidence", 0.0),
                        _prosody.get("tag") or "—",
                        mode_pack.get("mode"), mode_pack.get("reason"),
                    )
                    ev = result.get("event")
                    if ev and person_id is not None:
                        if specific_forget_target_for_turn:
                            _log.info(
                                "[empathy] skipped emotional memory write for targeted "
                                "forget request person_id=%s target=%r",
                                person_id,
                                specific_forget_target_for_turn,
                            )
                            return
                        if suppress_memory_learning:
                            _log.info(
                                "[empathy] skipped emotional memory write because "
                                "this turn was handled as memory/control intent "
                                "person_id=%s: %s",
                                person_id,
                                ev.get("description"),
                            )
                            return
                        if not _extracted_memory_allowed(ev, person_id):
                            _log.info(
                                "[empathy] skipped emotional memory write matching "
                                "forgotten terms for person_id=%s: %s",
                                person_id,
                                ev.get("description"),
                            )
                            return
                        try:
                            row_id = emotional_events.add_event(
                                person_id,
                                category=ev.get("category", "other"),
                                description=ev.get("description", ""),
                                valence=ev.get("valence", -0.5),
                                person_invited_topic=bool(result.get("invitation")),
                                loss_subject=ev.get("loss_subject"),
                                loss_subject_kind=ev.get("loss_subject_kind"),
                                loss_subject_name=ev.get("loss_subject_name"),
                            )
                            if row_id:
                                _log.info(
                                    "[empathy] stored emotional event id=%s "
                                    "category=%s for person_id=%s: %s",
                                    row_id, ev.get("category"), person_id,
                                    ev.get("description"),
                                )
                        except Exception as exc:
                            _log.debug("emotional_events.add_event failed: %s", exc)
                except Exception as exc:
                    _log.warning("empathy classification thread error: %s", exc, exc_info=True)

            _empathy_thread = threading.Thread(
                target=_run_empathy, daemon=True, name="empathy-classify",
            )
            _empathy_thread.start()

        match = None
        response_text = None
        used_agenda_llm = False
        used_classified_intent = False
        routing_text = text

        if response_text is None:
            if event_cancellation_ack is None:
                cancel_person_id = person_id
                if cancel_person_id is None:
                    try:
                        cancel_person_id = consciousness.get_last_memory_hint_target()
                    except Exception:
                        cancel_person_id = None
                labels = _cancel_stale_event_memory(cancel_person_id, text)
                if (
                    not labels
                    and _router_decision_executable(
                        router_decision,
                        text=text,
                        context=router_context,
                        dialogue_decision=dialogue_decision,
                    )
                    and router_decision.action == "event.cancel"
                ):
                    event_hint = (
                        router_decision.args.get("event_hint")
                        or router_decision.args.get("target")
                        or router_decision.args.get("plan")
                        or ""
                    )
                    if event_hint:
                        labels = _cancel_stale_event_memory(
                            cancel_person_id,
                            text,
                            event_hint={"event_name": str(event_hint)},
                        )
                    if not labels:
                        labels = [str(event_hint).strip() or "that plan"]
                        _log.info(
                            "[action_router] executed event.cancel without matching "
                            "stored row person_id=%s label=%r text=%r",
                            cancel_person_id,
                            labels[0],
                            text,
                        )
                if labels:
                    if cancel_person_id is not None:
                        try:
                            _session_router_control_topics[int(cancel_person_id)] = labels[0]
                        except (TypeError, ValueError):
                            pass
                    event_cancellation_ack = _event_cancellation_ack(labels, cancel_person_id)
            if event_cancellation_ack:
                suppress_memory_learning = True
                completed = _speak_blocking(event_cancellation_ack, emotion="neutral")
                _router_audit_note_result(
                    router_audit,
                    completed=completed,
                    spoken_text=event_cancellation_ack,
                )
                response_text = event_cancellation_ack
                used_agenda_llm = True
                final_executed_path = "router.event_cancel_ack"

        if repair_move is None:
            repair_move = repair_moves.detect(text)
        if (
            repair_move
            and active_grief_for_turn
            and repair_move.get("kind") in {"misheard", "misunderstood"}
            and repair_move.get("correction")
        ):
            routing_text = repair_move["correction"]
            repair_moves.mark_handled(repair_move.get("kind") or "")
            _log.info(
                "[repair] applying corrected text to active grief flow: %r",
                routing_text,
            )
        elif repair_move:
            response_text = _generate_repair_response(person_id, text, repair_move)
            used_agenda_llm = True
            final_executed_path = f"repair.{repair_move.get('kind') or 'unknown'}"

        # Layer-1 insult pre-check: keyword match → bump anger BEFORE the LLM
        # call so this turn's system prompt reflects the new escalation level.
        # Layer 2 (llm.analyze_sentiment in _post_response) skips its own
        # increment when this flag is set so we never double-count.
        pre_classified_insult = False
        pre_classified_compliment = False
        if response_text is None and personality.is_obvious_insult(text):
            new_level = personality.increment_anger(person_id)
            pre_classified_insult = True
            try:
                _play_event_body_beat("insult.detected")
                _set_body_mood("offended", source="layer1_insult")
            except Exception as exc:
                _log.debug("[interaction] insult body beat skipped: %s", exc)
            if person_id is not None:
                people_memory.apply_relationship_increment(person_id, "insult_mild")
            _log.info(
                "[interaction] layer-1 insult detected — anger now %d", new_level,
            )
        elif response_text is None and personality.is_obvious_compliment(text):
            # Layer-1 compliment pre-check: immediate pleased body language on this turn.
            # Layer 2 (llm.analyze_sentiment) skips its own beat when this flag is set so
            # we never double-react; subtle praise still gets caught there.
            pre_classified_compliment = True
            try:
                _play_event_body_beat("compliment.detected")
                _set_body_mood("proud", source="layer1_compliment")
            except Exception as exc:
                _log.debug("[interaction] compliment body beat skipped: %s", exc)
            _log.info("[interaction] layer-1 compliment detected")

        # Active grief flow must consume short replies before command/intent
        # routing. A name like "Tom Foster" can otherwise be misclassified as
        # "who is speaking?" and fall back to roast-first Rex.
        if response_text is None and active_grief_for_turn:
            grief_response = _continue_grief_flow(person_id, routing_text)
            if grief_response:
                _log.info(
                    "[interaction] grief flow → person_id=%s step=%s text=%r",
                    person_id,
                    (_grief_flow_state.get(person_id) or {}).get("step"),
                    grief_response,
                )
                grief_emotion = "sad"
                grief_pre_ms = 600
                grief_post_ms = 600
                grief_voice = None
                try:
                    ov = empathy.get_delivery_overrides(person_id)
                    if ov:
                        if ov.get("emotion"):
                            grief_emotion = ov["emotion"]
                        grief_pre_ms = max(grief_pre_ms, int(ov.get("pre_beat_ms") or 0))
                        grief_post_ms = max(grief_post_ms, int(ov.get("post_beat_ms") or 0))
                        grief_voice = ov.get("voice_settings")
                except Exception as exc:
                    _log.debug("grief flow delivery override error: %s", exc)
                completed = _speak_blocking(
                    grief_response,
                    emotion=grief_emotion,
                    pre_beat_ms=grief_pre_ms,
                    post_beat_ms_override=grief_post_ms,
                    voice_settings=grief_voice,
                )
                _router_audit_note_result(
                    router_audit,
                    completed=completed,
                    spoken_text=grief_response,
                )
                response_text = grief_response
                final_executed_path = "grief_flow.active_turn"

        # Command parser → local dispatch or LLM fallback. Skip local routing
        # for active grief flow turns, including the open-ended description
        # reply where _continue_grief_flow returns None and hands off to LLM.
        if response_text is None and not active_grief_for_turn:
            match = command_parser.parse(text)
            try:
                from features import games as games_mod
                if games_mod.is_active():
                    command_key = match.command_key if match is not None else None
                    normalized_game_text = " ".join(text.lower().strip().split())
                    if match is None and normalized_game_text in {
                        "stop", "quit", "end", "stop game", "end game", "quit game",
                        "stop the game",
                    }:
                        match = command_parser.CommandMatch("stop_game", "active_game_stop", {})
                        command_key = "stop_game"
                    elif (
                        command_key == "dj_stop"
                        and normalized_game_text in {"stop", "quit", "end", "stop playing"}
                    ):
                        match = command_parser.CommandMatch("stop_game", "active_game_stop", {})
                        command_key = "stop_game"
                    elif command_key == "dj_skip" and normalized_game_text == "skip":
                        command_key = None

                    game_escape_commands = {
                        "stop_game", "sleep", "shutdown", "quiet_mode", "wake_up",
                        "dj_stop", "dj_skip", "volume_up", "volume_down",
                    }
                    if command_key not in game_escape_commands:
                        response_text = games_mod.handle_input(text, person_id, audio_array)
                        completed = _speak_blocking(response_text)
                        _router_audit_note_result(
                            router_audit,
                            completed=completed,
                            spoken_text=response_text,
                        )
                        if completed:
                            games_mod.on_response_spoken()
                            after_audio = games_mod.consume_pending_audio_after_response()
                            if after_audio and not _interrupted.is_set():
                                game_after_audio_path = after_audio
                        used_agenda_llm = True
                        handled_active_game_turn = True
                        suppress_memory_learning = True
                        final_executed_path = "game.active_turn"
                        match = None
            except Exception as exc:
                _log.debug("active game routing failed: %s", exc)
                _router_audit_note_result(
                    router_audit,
                    handler_error=f"active_game_routing_failed:{type(exc).__name__}",
                )
            legacy_block_reason = _legacy_command_execution_block_reason(
                match,
                text=text,
                context=router_context,
                dialogue_decision=dialogue_decision,
            )
            if legacy_block_reason:
                _log.info(
                    "[turn_policy] blocked legacy command=%s reason=%s text=%r",
                    match.command_key,
                    legacy_block_reason,
                    text,
                )
                _router_audit_clear_legacy_command(router_audit)
                match = None
            else:
                _router_audit_note_legacy_command(router_audit, match)
        if match is not None:
            _router_audit_note_legacy_command(router_audit, match)
            response_text = _execute_command(match, person_id, person_name, text)
            final_executed_path = f"legacy_command.{match.command_key}"
            if match.command_key in {
                "forget_specific",
                "forget_me",
                "forget_everyone",
                "memory_review",
                "memory_forget_fact",
                "memory_correct_fact",
                "memory_remember_fact",
                "memory_boundary",
            }:
                suppress_memory_learning = True
        elif response_text is None:
            if getattr(config, "INTENT_CLASSIFIER_ENABLED", False) and not active_grief_for_turn:
                try:
                    intent_started = time.monotonic()
                    intent = intent_classifier.classify_deterministic(text)
                    _latency_log(turn_start, "intent_classifier", intent_started)
                except Exception as exc:
                    _log.debug("intent classification error: %s", exc)
                    intent = "general"
                intent_block_reason = _intent_execution_block_reason(
                    intent,
                    text=text,
                    context=router_context,
                    dialogue_decision=dialogue_decision,
                    router_action=(
                        router_decision.action if router_decision is not None else None
                    ),
                )
                if intent_block_reason:
                    _log.info(
                        "[turn_policy] blocked intent=%s reason=%s text=%r",
                        intent,
                        intent_block_reason,
                        text,
                    )
                    intent = "general"
                _log.info("[interaction] intent classifier: %s", intent)
                if intent != "general":
                    used_classified_intent = True
                    # Pass raw voice-match data and currently-visible identified
                    # face so handlers like query_who_is_speaking can answer with
                    # real biometric ground truth instead of LLM hallucination.
                    _visible_name = None
                    try:
                        for p in world_state.get("people") or []:
                            if p.get("person_db_id") is not None and p.get("face_id"):
                                _visible_name = p["face_id"]
                                break
                    except Exception:
                        pass
                    response_text = _handle_classified_intent(
                        intent, text, person_id,
                        raw_best_id=raw_best_id,
                        raw_best_name=raw_best_name,
                        raw_best_score=speaker_score,
                        visible_known_name=_visible_name,
                    )
                    if response_text is not None:
                        final_executed_path = f"intent_classifier.{intent}"
            if response_text is None:
                if _empathy_thread is not None:
                    _empathy_join_timeout = float(getattr(
                        config, "EMPATHY_CLASSIFY_JOIN_TIMEOUT_SECS", 4.0,
                    ))
                    empathy_wait_started = time.monotonic()
                    _empathy_thread.join(timeout=_empathy_join_timeout)
                    _latency_log(turn_start, "empathy_join", empathy_wait_started)
                    if _empathy_thread.is_alive():
                        _log.warning(
                            "[empathy] classification thread did not finish "
                            "within %.1fs — grief flow / mode directive may "
                            "be missing from this turn",
                            _empathy_join_timeout,
                        )
                # Grief flow — intercept the LLM path with a structured
                # condolence/consent/name walk when the empathy classifier
                # has detected (or is mid-conversation about) a loss.
                grief_response = None
                if person_id is not None:
                    if _grief_flow_active(person_id):
                        grief_response = _continue_grief_flow(person_id, text)
                    else:
                        cached = empathy.peek(person_id)
                        ev = (
                            (cached.get("result") or {}).get("event")
                            if cached else None
                        )
                        grief_response = _maybe_start_grief_flow(person_id, ev)
                if grief_response:
                    _log.info(
                        "[interaction] grief flow → person_id=%s step=%s text=%r",
                        person_id,
                        (_grief_flow_state.get(person_id) or {}).get("step"),
                        grief_response,
                    )
                    # Speak the structured response with the empathy delivery
                    # envelope (sympathetic LEDs/voice, longer pre/post beats).
                    grief_emotion = "sad"
                    grief_pre_ms = 600
                    grief_post_ms = 600
                    grief_voice = None
                    try:
                        ov = empathy.get_delivery_overrides(person_id)
                        if ov:
                            if ov.get("emotion"):
                                grief_emotion = ov["emotion"]
                            grief_pre_ms = max(grief_pre_ms, int(ov.get("pre_beat_ms") or 0))
                            grief_post_ms = max(grief_post_ms, int(ov.get("post_beat_ms") or 0))
                            grief_voice = ov.get("voice_settings")
                    except Exception as exc:
                        _log.debug("grief flow delivery override error: %s", exc)
                    completed = _speak_blocking(
                        grief_response,
                        emotion=grief_emotion,
                        pre_beat_ms=grief_pre_ms,
                        post_beat_ms_override=grief_post_ms,
                        voice_settings=grief_voice,
                    )
                    _router_audit_note_result(
                        router_audit,
                        completed=completed,
                        spoken_text=grief_response,
                    )
                    response_text = grief_response
                    final_executed_path = "grief_flow.llm_intercept"
                # If the grief flow handled this turn, do NOT also fire the
                # LLM streaming path — that would speak a second response on
                # top of the grief line. Fall through to LLM only when grief
                # didn't intercept.
                if response_text is None:
                    # If there are unacknowledged emotional events for this
                    # person the system prompt is about to fire the
                    # ACKNOWLEDGE-ON-RETURN directive — mark them acknowledged
                    # now so it doesn't repeat across this person's subsequent
                    # turns this session.
                    if person_id is not None:
                        try:
                            try:
                                ws_now = world_state.snapshot()
                                crowd_count = int((ws_now.get("crowd") or {}).get("count", 1) or 1)
                            except Exception:
                                crowd_count = 1
                            suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
                            unack = [
                                ev for ev in emotional_events.get_active_events(person_id, limit=3)
                                if not ev.get("last_acknowledged_at")
                                and emotional_events.can_surface_event(ev)
                                and not (
                                    suppress_in_crowd
                                    and crowd_count > 1
                                    and emotional_events.is_heavy_event(ev)
                                )
                            ]
                            if unack:
                                for ev in unack:
                                    emotional_events.mark_acknowledged(int(ev["id"]))
                                _log.info(
                                    "[empathy] marked %d emotional event(s) acknowledged "
                                    "for person_id=%s on this turn",
                                    len(unack), person_id,
                                )
                        except Exception as exc:
                            _log.debug("emotional event ack error: %s", exc)
                    _try_slow_path_ack(
                        "general",
                        text=text,
                        dialogue_decision=dialogue_decision,
                    )
                    response_text = _stream_llm_response(
                        text,
                        person_id,
                        answered_question=answered_question,
                        turn_start=turn_start,
                    )
                    used_agenda_llm = True
                    final_executed_path = "llm.stream"

        if response_text:
            _log_router_audit(
                router_audit,
                final_executed_path or "response_text.unknown",
                spoken_text=response_text,
            )
            conv_memory.add_to_transcript("Rex", response_text)
            conv_log.log_rex(response_text)
            _session_exchange_count += 1
            _register_rex_utterance(response_text)
            assistant_asked_question = _assistant_asked_question(response_text)
            question_recovery_text = response_text if assistant_asked_question else ""
            try:
                end_thread.mark_closure_spoken()
            except Exception:
                pass
        else:
            question_recovery_text = ""

        # Post-greeting relationship hook: we just enrolled a newcomer while a
        # different known person was recently engaged. Ask them how they know
        # each other, and open a relationship-prompt window so the next reply
        # gets parsed and saved as an edge.
        post_greet_fired = False
        pending_rel = _pending_post_greet_relationship[0]
        if pending_rel and not _interrupted.is_set() and person_id == pending_rel.get("newcomer_person_id"):
            _pending_post_greet_relationship[0] = None
            prior_name = pending_rel.get("prior_engaged_name") or "the other person"
            newcomer_name = pending_rel.get("newcomer_name") or "you"
            prior_first = _first_name_or(prior_name, "them")
            newcomer_first = _first_name_or(newcomer_name, "there")
            try:
                q_text = llm.get_response(
                    f"You just learned a new person named '{newcomer_first}' is here. "
                    f"Your friend '{prior_first}' is also present (you were just talking "
                    f"with them). In ONE short in-character Rex line, ask {newcomer_first} "
                    f"how they know {prior_first}. Warm, curious, one line ending in a "
                    f"question mark. Use names naturally."
                )
                if q_text:
                    _speak_blocking(q_text)
                    conv_memory.add_to_transcript("Rex", q_text)
                    conv_log.log_rex(q_text)
                    _register_rex_utterance(q_text)
                    assistant_asked_question = True
                    question_recovery_text = q_text
                    post_greet_fired = True
                    try:
                        consciousness.set_relationship_prompt_context({
                            "engaged_person_id": pending_rel["prior_engaged_id"],
                            "engaged_name": pending_rel.get("prior_engaged_name"),
                            "newcomer_person_id": pending_rel["newcomer_person_id"],
                            "slot_id": f"post_greet_{pending_rel['newcomer_person_id']}",
                            "asked_at": time.monotonic(),
                        })
                    except Exception as exc:
                        _log.debug("set_relationship_prompt_context failed: %s", exc)
                    _log.info(
                        "[interaction] post-greet relationship ask fired for "
                        "newcomer=%s prior=%s",
                        newcomer_name, prior_name,
                    )
            except Exception as exc:
                _log.debug("post-greet relationship ask error: %s", exc)

        # Curiosity routine — skip if we just asked a relationship question.
        if (
            match is None
            and response_text
            and not _interrupted.is_set()
            and not post_greet_fired
            and not used_agenda_llm
            and not used_classified_intent
        ):
            curiosity_q = _curiosity_check(response_text, text, person_id, person_name)
            if curiosity_q:
                conv_memory.add_to_transcript("Rex", curiosity_q)
                conv_log.log_rex(curiosity_q)
                _register_rex_utterance(curiosity_q)
                assistant_asked_question = True
                question_recovery_text = curiosity_q
        elif (
            used_agenda_llm
            and response_text
            and person_id is not None
            and "?" in response_text
        ):
            _record_pool_topics_in_response(response_text, person_id)

        # Release playback suppression before slower post-response memory work.
        # Otherwise an immediate human reply can land in the rolling buffer and
        # then get flushed when the API-backed fact/sentiment pass finishes.
        if sequence_started:
            echo_cancel.end_sequence(
                flush=False,
                tail_secs=_reply_playback_tail_secs(assistant_asked_question),
            )
            sequence_started = False
            if game_after_audio_path and not _interrupted.is_set():
                speech_queue.enqueue_audio_file(
                    game_after_audio_path,
                    priority=1,
                    tag="game:after_audio",
                )
                game_after_audio_path = None

        if assistant_asked_question and question_recovery_text and not handled_active_game_turn:
            _arm_no_response_recovery(question_recovery_text, person_id)

        if handled_active_game_turn:
            try:
                interoception.record_interaction()
            except Exception as exc:
                _log.debug("active game interoception error: %s", exc)
        else:
            _post_response(
                text,
                person_id,
                person_name,
                assistant_asked_question=assistant_asked_question,
                pre_classified_insult=pre_classified_insult,
                pre_classified_compliment=pre_classified_compliment,
                suppress_memory_learning=suppress_memory_learning,
            )
    except Exception as exc:
        handler_error = f"unhandled:{type(exc).__name__}"
        final_executed_path = final_executed_path or "exception.unhandled"
        raise
    finally:
        if sequence_started:
            _end_response_sequence_for_text(response_text)
        try:
            _log_character_loop_trace(
                character_trace,
                router_audit=router_audit,
                final_executed_path=final_executed_path,
                completed=completed,
                handler_error=handler_error,
                spoken_text=response_text,
                assistant_asked_question=assistant_asked_question,
                suppress_memory_learning=suppress_memory_learning,
                repair_move=repair_move,
                intent=intent,
            )
        finally:
            if trace_context_token is not None:
                _current_character_loop_trace.reset(trace_context_token)


def submit_text(
    text: str,
    *,
    person_id: Optional[int] = None,
    person_name: Optional[str] = None,
) -> bool:
    """Process GUI/CLI text input through the normal interaction pipeline."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    with _text_input_lock:
        current_state = state_module.get_state()
        if current_state in (State.SLEEP, State.SHUTDOWN):
            return False
        from_idle_activation = current_state != State.ACTIVE
        if current_state == State.IDLE:
            state_module.set_state(State.ACTIVE)

        global _last_speech_at
        _last_speech_at = time.monotonic()
        _begin_user_turn()
        try:
            _handle_speech_segment(
                np.zeros(1, dtype=np.float32),
                from_idle_activation=from_idle_activation,
                transcribed_text=cleaned,
                raw_best_id_override=person_id,
                raw_best_name_override=person_name,
                speaker_score_override=1.0 if person_id is not None else 0.0,
                text_input=True,
            )
            _last_speech_at = time.monotonic()
            return True
        finally:
            _end_user_turn()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def _loop() -> None:
    global _last_speech_at, _listen_resume_at, _post_tts_flush_needed

    idle_timeout = config.CONVERSATION_IDLE_TIMEOUT_SECS
    _last_speech_at = time.monotonic()

    while not _stop_event.is_set():
        current_state = state_module.get_state()

        # ── SHUTDOWN ────────────────────────────────────────────────────────────
        if current_state == State.SHUTDOWN:
            break

        # ── QUIET — stay silent until an explicit wake word exits quiet mode ───
        if current_state == State.QUIET:
            if _wake_word_fired.is_set():
                _wake_word_fired.clear()
                _last_speech_at = time.monotonic()
                _wake_from_quiet()
            _stop_event.wait(0.1)
            continue

        # ── SLEEP — any active wake model can bring Rex back online ───────────
        if current_state == State.SLEEP:
            try:
                _situation_assessor.set_vad_active(False)
            except Exception:
                pass
            if _wake_word_fired.is_set():
                _wake_word_fired.clear()
                with _wake_lock:
                    model = _last_wake_word
                if model:
                    _log.info("[wake_word] waking from sleep via model=%s", model)
                    _last_speech_at = time.monotonic()
                    _wake_from_sleep()
                    continue
            if not bool(getattr(config, "SLEEP_TRANSCRIBED_WAKE_FALLBACK_ENABLED", True)):
                _stop_event.wait(0.05)
                continue
            if time.monotonic() < _listen_resume_at:
                _stop_event.wait(_CHUNK_SECS)
                continue
            chunk = stream.get_audio_chunk(_CHUNK_SECS)
            if len(chunk) == 0:
                _stop_event.wait(_CHUNK_SECS)
                continue
            _sleep_speech = vad.is_speech(_chunk_for_vad(chunk))
            _situation_assessor.set_vad_active(_sleep_speech)
            if not _sleep_speech:
                _stop_event.wait(_CHUNK_SECS)
                continue
            _log.info("[sleep_mode] speech detected while asleep — checking wake phrase")
            speech_start = time.monotonic()
            audio_segment = _accumulate_speech(
                speech_start,
                allowed_states=(State.SLEEP,),
            )
            _wake_from_sleep_if_transcribed(audio_segment)
            _stop_event.wait(0.05)
            continue

        # ── IDLE — wake word first; optional direct speech activation ──────────
        if current_state == State.IDLE:
            if _wake_word_fired.is_set():
                _wake_word_fired.clear()
                # A wake word during music is a deliberate barge-in: stop the
                # track so Rex can hear and answer, then go ACTIVE and ack.
                if _dj_suppresses_conversation():
                    _stop_dj_for_wake()
                state_module.set_state(State.ACTIVE)
                _last_speech_at = time.monotonic()
                _wake_ack()
                continue

            # Never let Rex's own playback in IDLE self-trigger the interaction
            # loop into ACTIVE.
            listen_during_dj = (
                bool(getattr(config, "IDLE_LISTEN_DURING_DJ_PLAYBACK", True))
                and _dj_is_playing()
            )
            if _dj_suppresses_conversation() and not listen_during_dj:
                try:
                    _situation_assessor.set_vad_active(False)
                except Exception:
                    pass
                _last_speech_at = time.monotonic()
                _stop_event.wait(0.1)
                continue
            if (
                speech_queue.is_speaking()
                or output_gate.is_busy()
                or (echo_cancel.is_suppressed() and not listen_during_dj)
            ):
                _stop_event.wait(0.05)
                continue

            # Optional hands-free behavior: allow normal speech to activate Rex
            # directly from IDLE without a wake word.
            if not getattr(config, "IDLE_LISTEN_WITHOUT_WAKE_WORD", False):
                _stop_event.wait(0.05)
                continue
            try:
                if consciousness.is_identity_prompt_in_flight():
                    _stop_event.wait(0.05)
                    continue
            except Exception:
                pass

            if time.monotonic() < _listen_resume_at:
                _stop_event.wait(_CHUNK_SECS)
                continue

            chunk = stream.get_audio_chunk(_CHUNK_SECS)
            if len(chunk) == 0:
                _stop_event.wait(_CHUNK_SECS)
                continue
            _idle_speech = vad.is_speech(_chunk_for_vad(chunk))
            _situation_assessor.set_vad_active(_idle_speech)
            if not _idle_speech:
                _stop_event.wait(_CHUNK_SECS)
                continue

            # First detected speech after the post-TTS window may be the human's
            # immediate answer to Rex's question. If cleanup already happened,
            # just clear the marker; do not discard this VAD-positive chunk.
            global _post_tts_flush_needed
            if _post_tts_flush_needed:
                _post_tts_flush_needed = False

            _log.info("[interaction] speech detected in IDLE — activating without wake word")
            state_module.set_state(State.ACTIVE)
            speech_start = time.monotonic()
            _last_speech_at = speech_start
            _begin_user_turn()
            _dj_restore_volume = _duck_dj_for_speech()
            try:
                audio_segment = _accumulate_speech(speech_start)
                if audio_segment is None or len(audio_segment) == 0:
                    continue

                _last_speech_at = time.monotonic()
                _handle_speech_segment(audio_segment, from_idle_activation=True)
            finally:
                _restore_dj_volume(_dj_restore_volume)
                _end_user_turn()
            continue

        # ── ACTIVE — full continuous listening ─────────────────────────────────

        if _wake_word_fired.is_set():
            _wake_word_fired.clear()
            _last_speech_at = time.monotonic()
            # A wake word during music is a deliberate barge-in: stop the track
            # so Rex can hear the follow-up command instead of swallowing the
            # wake word (which left the user no way to shut the music off).
            if _dj_suppresses_conversation():
                _stop_dj_for_wake()
            if _should_play_active_wake_ack():
                _wake_ack()
            else:
                _log.info(
                    "[wake_word] active wake ack suppressed — busy or waiting for response"
                )
            continue

        if _dj_suppresses_conversation():
            try:
                _situation_assessor.set_vad_active(False)
            except Exception:
                pass
            _last_speech_at = time.monotonic()
            _stop_event.wait(0.1)
            continue

        # Idle timeout → end session and return to IDLE
        effective_idle_timeout = idle_timeout
        try:
            from features import games as games_mod
            if games_mod.is_active():
                effective_idle_timeout = max(
                    effective_idle_timeout,
                    float(getattr(config, "ACTIVE_GAME_IDLE_TIMEOUT_SECS", effective_idle_timeout)),
                )
        except Exception:
            pass
        idle_for = time.monotonic() - _last_speech_at
        if _maybe_interest_idle_followup(
            idle_for=idle_for,
            effective_idle_timeout=effective_idle_timeout,
        ):
            continue
        if _maybe_low_memory_idle_question(
            idle_for=idle_for,
            effective_idle_timeout=effective_idle_timeout,
        ):
            continue
        if _maybe_idle_banter(
            idle_for=idle_for,
            effective_idle_timeout=effective_idle_timeout,
        ):
            continue

        if time.monotonic() < _listen_resume_at:
            _situation_assessor.set_vad_active(False)
            _stop_event.wait(_CHUNK_SECS)
            continue

        direct_audio_path = None
        try:
            direct_audio_path = speech_queue.current_audio_path()
        except Exception:
            direct_audio_path = None
        if speech_queue.is_speaking() or output_gate.is_busy():
            interruptible_audio = (
                direct_audio_path is not None
                and _is_interruptible_game_audio_path(direct_audio_path)
            )
            if not interruptible_audio and not _vad_barge_in_enabled():
                _situation_assessor.set_vad_active(False)
                _stop_event.wait(_CHUNK_SECS)
                continue

        if idle_for >= effective_idle_timeout:
            _log.info("[interaction] conversation idle timeout — returning to IDLE")
            _maybe_idle_outro()
            _end_session()
            state_module.set_state(State.IDLE)
            continue

        # Read a small audio chunk and run VAD
        chunk = stream.get_audio_chunk(_CHUNK_SECS)
        if len(chunk) == 0:
            _stop_event.wait(_CHUNK_SECS)
            continue

        _active_speech = vad.is_speech(_chunk_for_vad(chunk))
        _situation_assessor.set_vad_active(_active_speech)
        if not _active_speech:
            if _maybe_prompt_incomplete_turn():
                _last_speech_at = time.monotonic()
            _stop_event.wait(_CHUNK_SECS)
            continue

        # First detected speech after the post-TTS window may be the human's
        # immediate answer. If cleanup already happened, just clear the marker;
        # keep this chunk and start accumulating.
        if _post_tts_flush_needed:
            _post_tts_flush_needed = False

        # ── Speech detected ────────────────────────────────────────────────────
        speech_start = time.monotonic()
        _last_speech_at = speech_start
        _begin_user_turn()
        try:
            from features import games as games_mod
            if games_mod.is_active():
                speech_queue.drop_by_tag("game:after_audio")
        except Exception:
            pass

        # Mid-speech interruption: stop TTS, acknowledge, flush the mic buffer of
        # Rex's voice tail, then WAIT for a fresh VAD rising edge before
        # accumulating. Without this, the rolling buffer still holds ~seconds of
        # Rex's own voice which Whisper concatenates onto the user's utterance.
        if speech_queue.is_speaking() or output_gate.is_busy():
            if _is_interruptible_game_audio_path(direct_audio_path):
                _interrupted.set()
                try:
                    import sounddevice as sd
                    echo_cancel.request_cancel()
                    sd.stop()
                except Exception:
                    pass
                time.sleep(0.1)
                try:
                    echo_cancel.clear_suppression_tail()
                except Exception:
                    pass
                _interrupted.clear()
                speech_start = time.monotonic()
                _last_speech_at = speech_start
                _log.info(
                    "[interaction] interrupted game audio for player speech: %s",
                    direct_audio_path,
                )
            elif not _vad_barge_in_enabled():
                global _last_vad_barge_in_suppressed_log_at
                now = time.monotonic()
                if now - _last_vad_barge_in_suppressed_log_at >= 2.0:
                    _last_vad_barge_in_suppressed_log_at = now
                    _log.info(
                        "[interaction] VAD barge-in suppressed while Rex is speaking; use wake word to interrupt"
                    )
                _end_user_turn()
                _stop_event.wait(_CHUNK_SECS)
                continue
            _interrupted.set()
            try:
                import sounddevice as sd
                echo_cancel.request_cancel()
                sd.stop()
            except Exception:
                pass
            # Brief settle so the worker can clean up its finally block
            time.sleep(0.1)
            _interrupted.clear()
            if direct_audio_path:
                # Non-speech clips/music beds are interruptible: keep the user's
                # current utterance instead of saying "yeah?" and forcing a repeat.
                _log.info("[interaction] direct audio interrupted by user speech: %s", direct_audio_path)
            else:
                _interrupt_ack()

                # Drop the polluted buffer and re-arm. The next user utterance must
                # trigger VAD again; the original speech_start is discarded.
                stream.flush()
                _listen_resume_at = time.monotonic() + config.POST_SPEECH_LISTEN_DELAY_SECS
                _post_tts_flush_needed = True
                _end_user_turn()
                continue

        _dj_restore_volume = _duck_dj_for_speech()
        try:
            # Accumulate the full utterance
            audio_segment = _accumulate_speech(speech_start)
            if audio_segment is None or len(audio_segment) == 0:
                continue

            _last_speech_at = time.monotonic()
            _handle_speech_segment(audio_segment)
        finally:
            _restore_dj_volume(_dj_restore_volume)
            _end_user_turn()


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start(*, text_only: bool = False) -> None:
    """Start the wake word detector and the continuous interaction loop."""
    global _thread, _identity_prompt_until, _awaiting_followup_event
    global _listen_resume_at, _listen_capture_floor_at, _post_tts_flush_needed
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation
    global _text_only_mode

    if _thread and _thread.is_alive():
        _log.warning("[interaction] already running")
        return

    _text_only_mode = bool(text_only)
    _stop_event.clear()
    _wake_word_fired.clear()
    _interrupted.clear()
    _session_person_ids.clear()
    _session_person_turn_counts.clear()
    _session_router_control_topics.clear()
    _clear_anonymous_speaker_slots()
    _interest_idle_followups_spoken.clear()
    _identity_prompt_until = 0.0
    _listen_resume_at = 0.0
    _listen_capture_floor_at = 0.0
    _post_tts_flush_needed = False
    _awaiting_followup_event = None
    _pending_common_first_name_identity = None
    _pending_common_first_name_introduction = None
    _pending_existing_common_first_name = None
    _pending_identity_match_confirmation = None
    _pending_prompted_name_confirmation = None
    _clear_pending_memory_wipe()
    _common_first_name_prompted_this_session.clear()
    topic_thread.clear()
    dialogue_act.clear()
    user_energy.clear()
    question_budget.clear()
    repair_moves.clear()
    end_thread.clear()
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

    speech_queue.register_on_item_done(_arm_post_tts_window)
    if _text_only_mode:
        _log.info("[interaction] started in text-only mode; wake word and mic loop disabled")
        return

    wake_word.start(_on_wake_word)
    threading.Thread(
        target=_prefill_wake_ack_cache,
        daemon=True,
        name="wake-ack-cache-prefill",
    ).start()
    threading.Thread(
        target=_prefill_slow_path_ack_cache,
        daemon=True,
        name="slow-path-ack-cache-prefill",
    ).start()
    if not wake_word.is_ready():
        _log.warning(
            "[interaction] wake word unavailable — entering ACTIVE fallback so speech can still be processed"
        )
        if state_module.get_state() == State.IDLE:
            state_module.set_state(State.ACTIVE)

    _thread = threading.Thread(
        target=_loop, daemon=True, name="interaction-loop"
    )
    _thread.start()
    _log.info("[interaction] started")


def stop() -> None:
    """Stop the interaction loop and wake word detector, waiting for clean exit."""
    global _thread, _awaiting_followup_event, _identity_prompt_until
    global _listen_resume_at, _listen_capture_floor_at, _post_tts_flush_needed
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture
    global _pending_common_first_name_identity, _pending_common_first_name_introduction
    global _pending_existing_common_first_name, _pending_identity_match_confirmation
    global _pending_prompted_name_confirmation
    global _text_only_mode

    _stop_event.set()
    if not _text_only_mode:
        wake_word.stop()

    if _thread:
        _thread.join(timeout=3.0)
        if _thread.is_alive():
            _log.warning("[interaction] loop thread did not stop cleanly")
        _thread = None

    _awaiting_followup_event = None
    _identity_prompt_until = 0.0
    _listen_resume_at = 0.0
    _listen_capture_floor_at = 0.0
    _post_tts_flush_needed = False
    _session_person_turn_counts.clear()
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
    _pending_common_first_name_identity = None
    _pending_common_first_name_introduction = None
    _pending_existing_common_first_name = None
    _pending_identity_match_confirmation = None
    _pending_prompted_name_confirmation = None
    _clear_pending_memory_wipe()
    _common_first_name_prompted_this_session.clear()
    _recent_voice_turns.clear()
    _clear_anonymous_speaker_slots()
    topic_thread.clear()
    user_energy.clear()
    question_budget.clear()
    repair_moves.clear()
    end_thread.clear()
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

    _text_only_mode = False
    _log.info("[interaction] stopped")
