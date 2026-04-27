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
import json
import random
import re
import threading
import time
from typing import Optional

import numpy as np

import config
import state as state_module
from state import State
from audio import stream, vad, wake_word, transcription, speaker_id
from audio import speech_queue, output_gate
from audio import echo_cancel
from audio import prosody
from intelligence import command_parser, llm, personality
from intelligence import consciousness
from intelligence import intent_classifier
from intelligence import empathy
from intelligence import conversation_agenda
from intelligence import topic_thread
from intelligence import user_energy
from intelligence import question_budget
from intelligence import repair_moves
from intelligence import end_thread
from intelligence import introductions
from intelligence import social_frame
from intelligence import turn_completion
from intelligence import friendship_patterns
from memory import facts as facts_memory
from memory import conversations as conv_memory
from memory import people as people_memory
from memory import events as events_memory
from memory import relationships as rel_memory
from memory import emotional_events
from memory import boundaries as boundary_memory
from awareness import interoception
from awareness import address_mode
from awareness.situation import assessor as _situation_assessor
from world_state import world_state
from utils import conv_log

_log = logging.getLogger(__name__)

# ── VAD chunk size ─────────────────────────────────────────────────────────────
# 32 ms at 16 kHz — matches the sounddevice blocksize and Silero's preferred size.
_CHUNK_SECS = 0.032


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None

# Wake word signaling ─────────────────────────────────────────────────────────
_wake_word_fired = threading.Event()
_last_wake_word: Optional[str] = None
_wake_lock = threading.Lock()

# Interruption signal: set when wake word fires while Rex is speaking.
# The _speak_blocking() watchdog calls sd.stop() when it sees this set.
_interrupted = threading.Event()

# Session tracking ────────────────────────────────────────────────────────────
_session_person_ids: set[int] = set()
_session_exchange_count: int = 0
_last_speech_at: float = 0.0  # monotonic timestamp of most recent speech chunk

# Anti-repeat for latency filler lines
_last_filler: Optional[str] = None

# Monotonic deadline before which VAD speech-onset detections are discarded.
# Set at the end of each TTS utterance; prevents Rex's own voice tail from
# immediately triggering a new speech segment.
_listen_resume_at: float = 0.0

# When True, the main loop flushes the buffer once after the post-TTS delay
# expires to drop any room-decay audio accumulated during the silent wait.
_post_tts_flush_needed: bool = False

# Time window (set by consciousness) where a short bare-name reply is accepted.
_identity_prompt_until: float = 0.0

_IDENTITY_REPLY_WINDOW_SECS = 45.0


def _begin_user_turn() -> None:
    """Suppress proactive speech while the interaction loop handles a user turn."""
    try:
        _situation_assessor.set_interaction_busy(True)
    except Exception:
        pass
    try:
        # User speech makes queued background/presence chatter stale. Drop any
        # waiting non-urgent speech before it can start mid-answer.
        speech_queue.clear_below_priority(2)
    except Exception:
        pass


def _end_user_turn() -> None:
    try:
        _situation_assessor.set_interaction_busy(False)
    except Exception:
        pass
_NAME_MAX_WORDS = 3

# If a follow-up question ("how did X go?") is outstanding, this stores the
# event row so the next user utterance can be captured as the outcome.
_awaiting_followup_event: Optional[dict] = None

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
# when Bret says "this is my partner JT", this tracks the intro as a social
# event, saves the relationship, and asks one natural "how did you meet?" beat.
_pending_introduction: Optional[dict] = None
_pending_intro_followup: Optional[dict] = None
_pending_intro_voice_capture: Optional[dict] = None

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

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi am\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi['’]m\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bim\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bthis is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bcall me\s+(.+)$", re.IGNORECASE),
]

_NAME_STOPWORDS = {
    "again",
    "back",
    "fine",
    "good",
    "great",
    "here",
    "okay",
    "ok",
    "ready",
    "sorry",
    "there",
}


# ─────────────────────────────────────────────────────────────────────────────
# Wake word callback (called from wake_word's background daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

def _on_wake_word(model_name: str) -> None:
    global _last_wake_word
    with _wake_lock:
        _last_wake_word = model_name

    if speech_queue.is_speaking():
        _interrupted.set()

    _wake_word_fired.set()


# ─────────────────────────────────────────────────────────────────────────────
# Speech helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_speak() -> bool:
    return state_module.get_state() not in (State.QUIET, State.SHUTDOWN)


def _dj_is_playing() -> bool:
    try:
        from features import dj as dj_mod
        return bool(dj_mod.is_playing())
    except Exception:
        return False


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

    done = speech_queue.enqueue(
        text, emotion, priority=priority,
        pre_beat_ms=pre_beat_ms, post_beat_ms=post_beat_ms,
        voice_settings=voice_settings,
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

    # Normal completion — block new speech-onset detections briefly and discard
    # any mic audio that captured Rex's voice tail during playback. If Rex just
    # asked a direct question, use a much shorter handoff so immediate answers
    # do not lose their first syllables.
    global _listen_resume_at, _post_tts_flush_needed
    delay_secs = config.POST_SPEECH_LISTEN_DELAY_SECS
    if asked_question:
        delay_secs = float(
            getattr(config, "POST_QUESTION_LISTEN_DELAY_SECS", delay_secs)
        )
    _listen_resume_at = time.monotonic() + delay_secs
    stream.flush()
    _post_tts_flush_needed = True
    return True


def _arm_post_tts_window() -> None:
    """Arm the post-TTS deaf window. Registered with speech_queue so it fires
    after every queue item — not just items played via _speak_blocking."""
    global _listen_resume_at, _post_tts_flush_needed
    _listen_resume_at = time.monotonic() + config.POST_SPEECH_LISTEN_DELAY_SECS
    stream.flush()
    _post_tts_flush_needed = True


def _speak_async(text: str, emotion: str = "neutral") -> None:
    if not _can_speak() or not text:
        return
    text = llm.clean_response_text(text)
    if not text:
        return
    if speech_queue.is_speaking():
        return
    speech_queue.enqueue(text, emotion, priority=0)


def _wake_ack() -> None:
    pool = config.WAKE_ACKNOWLEDGMENTS
    _speak_blocking(random.choice(pool), priority=2)


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

    def _timer() -> None:
        if stop.wait(delay):
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


def _assistant_asked_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and ("?" in cleaned)


def _register_rex_utterance(text: str, wait_secs: Optional[float] = None) -> None:
    if not text or not text.strip():
        return
    try:
        repair_moves.note_assistant_turn(text)
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
        consciousness.note_rex_utterance(text, wait_secs=wait_secs)
    except Exception:
        pass


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

def _normalize_name(candidate: str) -> Optional[str]:
    """
    Normalize a spoken/self-reported name candidate.
    Returns None when the candidate does not look like a usable name.
    """
    text = candidate.strip()
    if not text:
        return None

    # Keep the first clause only ("my name is Bret, nice to meet you").
    text = re.split(r"[,.!?;:]", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s+", " ", text)

    tokens = []
    for raw in text.split(" "):
        token = re.sub(r"[^A-Za-z'\-]", "", raw).strip("'-")
        if token:
            tokens.append(token)

    if not tokens or len(tokens) > _NAME_MAX_WORDS:
        return None

    if len(tokens) == 1 and tokens[0].lower() in _NAME_STOPWORDS:
        return None

    if any(t.lower() in {"i", "im", "i'm", "me", "my", "name"} for t in tokens):
        return None

    # If Whisper returned all lowercase, title-case for storage/readback.
    if all(t.islower() for t in tokens):
        tokens = [t.capitalize() for t in tokens]

    return " ".join(tokens)


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
        return _normalize_name(normalized)

    return None


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
        intro_name = _extract_introduced_name(text, allow_bare_name=True)
    return intro_name, rel_label


def _has_unknown_visible_person() -> bool:
    """True if WorldState currently includes at least one person without a face match."""
    try:
        people = world_state.get("people")
    except Exception:
        return False
    return any(p.get("face_id") is None for p in people)


def _bind_world_state_identity(person_id: int, name: str) -> None:
    """
    Attach the newly enrolled identity to the first unknown world-state person slot.
    """
    try:
        people = world_state.get("people")
        changed = False
        for person in people:
            if person.get("person_db_id") is None or person.get("face_id") is None:
                person["person_db_id"] = person_id
                person["face_id"] = name
                person["voice_id"] = name
                changed = True
                break
        if changed:
            world_state.update("people", people)

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

    name = parsed.get("name")
    relationship = parsed.get("relationship")

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
                first = (engaged_name or "friend").split()[0]
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

        new_id = people_memory.enroll_person(name)
        if new_id is None:
            _log.error("[interaction] failed to create DB row for newcomer %r", name)
            consciousness.note_relationship_slot_handled(slot_id)
            return None

        first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
        if first_inc > 0:
            people_memory.update_familiarity(new_id, first_inc)

        frame = camera_mod.capture_still()
        if frame is not None:
            face_mod.enroll_unknown_face(new_id, frame)
            threading.Thread(
                target=face_mod.update_appearance,
                args=(new_id, frame.copy()),
                daemon=True,
                name=f"appearance-enroll-{new_id}",
            ).start()

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
        introducer_for_ack = engaged_id or speaker_person_id
        if introducer_for_ack is None:
            return f"{name.split()[0]}, welcome. Identity filed under 'better than mystery organic.'"
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
            ok = speaker_id.enroll_voice(person_id, audio_copy)
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
) -> Optional[int]:
    """
    Enroll a brand-new person and attach available voice/face biometrics.
    Returns person_id on success.

    If enroll_unknown_face=True, use face_mod.enroll_unknown_face which picks
    the largest face NOT already matched to an existing known person. Use this
    when a known person is visible alongside the newcomer so we don't rebind
    the known person's face to the new name.
    """
    person_id = people_memory.enroll_person(name)
    if person_id is None:
        _log.error("failed to enroll new person row for name=%r", name)
        return None

    first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
    if first_inc > 0:
        people_memory.update_familiarity(person_id, first_inc)

    try:
        speaker_id.enroll_voice(person_id, audio_array)
    except Exception as exc:
        _log.warning("voice enrollment failed for person_id=%s: %s", person_id, exc)

    try:
        from vision import camera as camera_mod
        from vision import face as face_mod

        frame = camera_mod.capture_still()
        if frame is not None:
            if enroll_unknown_face:
                face_mod.enroll_unknown_face(person_id, frame)
            else:
                face_mod.enroll_face(person_id, frame)
            # Appearance extraction is useful but non-blocking.
            threading.Thread(
                target=face_mod.update_appearance,
                args=(person_id, frame.copy()),
                daemon=True,
                name=f"appearance-enroll-{person_id}",
            ).start()
    except Exception as exc:
        _log.warning("face enrollment failed for person_id=%s: %s", person_id, exc)

    _bind_world_state_identity(person_id, name)
    _log.info("[interaction] enrolled new person: %s (person_id=%s)", name, person_id)
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
        new_id = people_memory.enroll_person(name)
    except Exception as exc:
        _log.warning("introduction enroll_person failed: %s", exc)
        return None
    if new_id is None:
        return None

    first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
    if first_inc > 0:
        people_memory.update_familiarity(new_id, first_inc)

    if enroll_visible_face:
        try:
            from vision import camera as camera_mod
            from vision import face as face_mod
            frame = camera_mod.capture_still()
            if frame is not None:
                face_mod.enroll_unknown_face(new_id, frame)
                threading.Thread(
                    target=face_mod.update_appearance,
                    args=(new_id, frame.copy()),
                    daemon=True,
                    name=f"appearance-enroll-{new_id}",
                ).start()
        except Exception as exc:
            _log.warning("introduction face enrollment failed: %s", exc)

        _bind_world_state_identity(new_id, name)
    _store_introduction_memories(
        introducer_id,
        introducer_name,
        new_id,
        name,
        relationship,
    )
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

    introducer_first = (introducer_name or "there").split()[0]
    introduced_first = (introduced_name or "there").split()[0]
    rel_clause = f"{relationship}" if relationship else "guest"
    self_explanatory_relationship = _intro_relationship_self_explanatory(relationship)
    followup_kind = (
        "relationship_color" if self_explanatory_relationship else "connection_story"
    )

    if visible_newcomer and self_explanatory_relationship:
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
        if self_explanatory_relationship:
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
    first = (name or "").split()[0].lower()
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

    ok = speaker_id.enroll_voice(introduced_id, audio_array)
    if not ok:
        ctx["asked_at"] = time.monotonic()
        first = introduced_name.split()[0]
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

    intro_first = (introducer_name or "there").split()[0]
    introduced_first = (introduced_name or "there").split()[0]
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


def _handle_introduction_parse(
    parsed: introductions.IntroductionParse,
    *,
    introducer_id: int,
    introducer_name: str,
    visible_newcomer: bool = True,
) -> Optional[str]:
    global _pending_introduction

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
                return llm.get_response(
                    f"{introducer_name} said this visible newcomer is{rel_hint}, "
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

    completed = _speak_blocking(
        prompt,
        emotion="neutral",
        pre_beat_ms=100,
        post_beat_ms_override=150,
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


# ─────────────────────────────────────────────────────────────────────────────
# Speech accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_speech(speech_start_mono: float) -> Optional[np.ndarray]:
    """
    Poll VAD until config.SILENCE_TIMEOUT_SECS of sustained silence, then
    return the full speech segment captured from the rolling buffer.

    speech_start_mono is the monotonic timestamp when speech was first detected
    in the main loop — used to calculate how much of the buffer to grab.
    """
    silence_timeout = config.SILENCE_TIMEOUT_SECS
    silence_elapsed = 0.0

    while not _stop_event.is_set():
        if state_module.get_state() != State.ACTIVE:
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

    # Grab the full segment from the rolling buffer.
    # Add 0.3 s pre-buffer so we capture the very start of the utterance.
    duration = time.monotonic() - speech_start_mono + 0.3
    capture_secs = min(duration, config.AUDIO_BUFFER_SECONDS)
    return stream.get_audio_chunk(capture_secs)


# ─────────────────────────────────────────────────────────────────────────────
# LLM streaming to TTS
# ─────────────────────────────────────────────────────────────────────────────

def _stream_llm_response(
    text: str,
    person_id: Optional[int],
    answered_question: Optional[dict] = None,
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
        agenda_directive = conversation_agenda.build_turn_directive(
            text,
            person_id,
            answered_question=answered_question,
        )
        frame = social_frame.build_frame(
            text,
            person_id,
            answered_question=answered_question,
            agenda_directive=agenda_directive,
        )
        agenda_directive = "\n".join([
            agenda_directive,
            social_frame.build_directive(frame),
        ])
        _log.info("[agenda] %s", agenda_directive.replace("\n", " | "))
        full_text = llm.get_response(
            text,
            person_id,
            agenda_directive=agenda_directive,
        )
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

        # Brief join — classifier usually finishes before or alongside the
        # main response since its prompt is tiny. Cap the wait so a slow
        # classifier never delays Rex perceptibly.
        surprise_thread.join(timeout=0.3)
        if surprise_result["value"]:
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
            if overrides.get("emotion"):
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

        _speak_blocking(
            full_text,
            emotion=delivery_emotion,
            pre_beat_ms=pre_beat_ms,
            post_beat_ms_override=delivery_post_beat_ms,
            voice_settings=delivery_voice_settings,
        )
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# Command execution
# ─────────────────────────────────────────────────────────────────────────────

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
    actual_direction = direction
    analysis: dict = {}

    try:
        from sequences import animations
        from vision import camera as camera_mod
        from vision import scene as vision_scene

        suspend_base = (
            float(getattr(config, "DIRECTED_LOOK_SETTLE_SECS", 0.65))
            + float(getattr(config, "CAMERA_POSE_SETTLE_SECS", 0.5))
            + 3.0
        )
        directions = _directed_search_directions(direction) if search_target else [direction]
        for attempt_direction in directions:
            consciousness.suspend_face_tracking(suspend_base)
            actual_direction = animations.directed_look_pose(
                attempt_direction,
                target=target_hint,
            )
            frame = camera_mod.capture_current_gaze(settle_secs=0.12)
            if frame is None:
                resp = (
                    "I looked, but my photoreceptors came back empty. "
                    "Very dramatic. Terrible evidence."
                )
                _speak_blocking(resp)
                return resp

            analysis = vision_scene.analyze_directed_attention(
                frame,
                direction=actual_direction,
                utterance=raw_text,
                target_hint=target_hint,
            )
            if not search_target or _analysis_found_target(analysis, target_hint):
                break
            _log.info(
                "[interaction] directed look search miss — target=%r direction=%s",
                target_hint,
                actual_direction,
            )
    except Exception as exc:
        _log.debug("directed look command failed: %s", exc)
        resp = "I tried to look, but my neck servos and photoreceptors are staging a tiny rebellion."
        _speak_blocking(resp)
        return resp

    if not analysis:
        resp = "I looked, but the visual intel is thin. Atmospheric, yes. Useful, no."
        _speak_blocking(resp)
        return resp
    if search_target and not _analysis_found_target(analysis, target_hint):
        resp = _not_found_visual_response(target_hint)
        _speak_blocking(resp)
        return resp

    speaker = person_name or "the person"
    prompt = (
        f"{speaker} asked you to physically look {_directed_look_label(actual_direction)}. "
        f"The original request was: {raw_text!r}. "
        "You moved your head/visor, took a fresh look, and got this vision analysis:\n"
        f"{json.dumps(analysis, ensure_ascii=False)}\n\n"
        "Reply as Rex with one concise roast-style observation or opinion based ONLY "
        "on the analysis. Max 35 words. If the target is a person, child, pet, or "
        "possible introduction, acknowledge them warmly with a harmless quip and, "
        "only if useful, ask who they are. Do not mention JSON, APIs, cameras, "
        "screenshots, or image analysis."
    )
    resp = llm.get_response(prompt, person_id) or _fallback_directed_look_response(analysis)
    _speak_blocking(resp)
    return llm.clean_response_text(resp)


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
        from awareness.chronoception import get_time_context
        ctx = get_time_context()
        h = ctx.get("hour", 0)
        minute = ctx.get("minute", 0)
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return _say(
            f"The exact current time is {h12}:{minute:02d} {ampm}. "
            f"Tell the person the time in one in-character line. "
            f"You MUST state both the hour and the minute exactly as given "
            f"({h12}:{minute:02d}) — do not round to the hour, do not omit the minute."
        )

    if key == "date_query":
        from awareness.chronoception import get_time_context
        ctx = get_time_context()
        return _say(
            f"Today is {ctx.get('day_of_week', 'unknown')}, time of day is "
            f"{ctx.get('time_of_day', 'unknown')}. "
            f"State the date in one in-character line."
        )

    # ── State transitions ──────────────────────────────────────────────────────
    if key == "sleep":
        resp = llm.get_response(
            "You are entering sleep mode. Deliver one short in-character sleep line.", person_id
        )
        _speak_blocking(resp)
        state_module.set_state(State.SLEEP)
        return resp

    if key == "wake_up":
        state_module.set_state(State.ACTIVE)
        return _say("You just woke from sleep. One short in-character wake-up line.")

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
            "You are about to shut down. One final in-character shutdown line.", person_id
        )
        _speak_blocking(resp)
        state_module.set_state(State.SHUTDOWN)
        return resp

    # ── Memory ─────────────────────────────────────────────────────────────────
    if key == "forget_me":
        if person_id is None:
            resp = "I don't have any record of you to forget."
            _speak_blocking(resp)
            return resp
        return _say(
            "Someone asked you to forget them. Ask for confirmation in one Rex-style line: "
            "say you'll forget everything — name, face, all of it — and tell them to say "
            "'yes forget me' to confirm."
        )

    if key == "forget_everyone":
        return _say(
            "Someone asked you to wipe your entire memory of all people. "
            "Ask for explicit confirmation in one Rex-style line: say it deletes every person, "
            "every face, every name, every conversation, and they must say 'confirm full wipe'."
        )

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
        name = args.get("name", "").strip()
        if not name:
            return ""
        if person_id is not None:
            from memory import database as _db
            _db.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
        return _say(
            f"The person you're talking to just told you to call them '{name}'. "
            f"Acknowledge in one in-character line."
        )

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
                dj_mod.play_by_vibe(vibe)
                return _say(
                    f"Playing something with vibe '{vibe}'. Announce it in one in-character line."
                )
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

def _post_response(
    user_text: str,
    person_id: Optional[int],
    person_name: Optional[str] = None,
    *,
    assistant_asked_question: bool = False,
    pre_classified_insult: bool = False,
) -> None:
    """
    Run after every response in ACTIVE state. Sentiment, facts, follow-up,
    and interoception are handled here. Sentiment/facts run in a background
    thread; follow-up delivery and interoception run in the calling thread.
    """
    # ── Follow-up delivery (sync — spoken as part of this turn) ───────────────
    global _awaiting_followup_event

    if person_id is not None:
        try:
            followups = consciousness.get_pending_followup(person_id)
            if followups:
                # If Rex just asked a question, do not immediately ask another one.
                # Re-queue and wait for the person's reply first.
                if assistant_asked_question:
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
                            _register_rex_utterance(resp)
                            _awaiting_followup_event = None
                            event_id = event.get("id")
                            if event_id is not None:
                                _awaiting_followup_event = {
                                    "person_id": person_id,
                                    "event_id": int(event_id),
                                }
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
                        people_memory.update_relationship_scores(person_id, antagonism=+0.03)

            elif sentiment.get("is_apology"):
                personality.decrement_anger()
                if person_id is not None:
                    people_memory.update_relationship_scores(person_id, antagonism=-0.02)

            if sentiment.get("is_compliment") and person_id is not None:
                people_memory.update_relationship_scores(person_id, warmth=+0.02)

        except Exception as exc:
            _log.debug("post_response sentiment error: %s", exc)

        if person_id is not None:
            try:
                friendship_patterns.learn_from_turn(person_id, user_text)
            except Exception as exc:
                _log.debug("friendship pattern learning error: %s", exc)

        # Fact extraction from recent transcript
        if person_id is not None:
            try:
                transcript = conv_memory.get_session_transcript()
                # Last 10 entries (~5 exchanges) — wider window than before so
                # facts mentioned a few turns back are still in scope.
                recent = transcript[-10:] if len(transcript) >= 10 else transcript
                new_facts = llm.extract_facts(person_id, recent, person_name=person_name)
                saved_count = 0
                for fact in new_facts:
                    if fact.get("key") and fact.get("value"):
                        facts_memory.add_fact(
                            person_id,
                            fact.get("category", "other"),
                            fact["key"],
                            fact["value"],
                            source="stated",
                            confidence=0.9,
                        )
                        saved_count += 1
                _log.info(
                    "[interaction] facts extracted=%d saved=%d for person_id=%s",
                    len(new_facts), saved_count, person_id,
                )
            except Exception as exc:
                _log.debug("post_response fact extraction error: %s", exc)

            # Event extraction → person_events table for follow-ups + small talk
            try:
                transcript = conv_memory.get_session_transcript()
                recent = transcript[-10:] if len(transcript) >= 10 else transcript
                new_events = llm.extract_events(person_id, recent, person_name=person_name)
                saved_events = 0
                if new_events:
                    existing = events_memory.get_upcoming_events(person_id) or []
                    existing_keys = {
                        ((e.get("event_name") or "").strip().lower(), e.get("event_date"))
                        for e in existing
                    }
                    for ev in new_events:
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
# Session teardown
# ─────────────────────────────────────────────────────────────────────────────

def _end_session() -> None:
    """
    Called on ACTIVE → IDLE transition. Generates and persists a session summary,
    updates visit records and familiarity, then clears in-memory session state.
    """
    global _session_exchange_count, _identity_prompt_until, _awaiting_followup_event
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture

    transcript = conv_memory.get_session_transcript()
    if not transcript:
        _session_exchange_count = 0
        _session_person_ids.clear()
        try:
            topic_thread.clear()
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

            summary = llm.generate_session_summary(person_id, transcript)
            if summary:
                conv_memory.save_conversation(
                    person_id,
                    summary,
                    emotion_tone="neutral",
                    topics="",
                )

            # Full-transcript fact extraction at session end — catches facts the
            # per-exchange rolling window may have missed.
            try:
                end_facts = llm.extract_facts(person_id, transcript, person_name=person_name)
                saved = 0
                for fact in end_facts:
                    if fact.get("key") and fact.get("value"):
                        facts_memory.add_fact(
                            person_id,
                            fact.get("category", "other"),
                            fact["key"],
                            fact["value"],
                            source="stated",
                            confidence=0.9,
                        )
                        saved += 1
                _log.info(
                    "[interaction] session-end facts extracted=%d saved=%d for person_id=%s (%s)",
                    len(end_facts), saved, person_id, person_name,
                )
            except Exception as exc:
                _log.error("session-end fact extraction error for person_id=%s: %s", person_id, exc)

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
    try:
        topic_thread.clear()
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
    _identity_prompt_until = 0.0
    _awaiting_followup_event = None
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
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


_QUESTION_BOUNDARY_TOPICS = {
    "hometown": "hometown",
    "job": "work",
    "favorite_movie": "movies",
    "favorite_music": "music",
    "how_found_rex": "rex",
    "hobbies": "hobbies",
    "travel": "travel",
    "proudest_moment": "personal history",
    "biggest_challenge": "personal history",
    "obsession": "interests",
    "relationships": "relationships",
    "values": "values",
    "fears": "fears",
    "life_changing": "personal history",
    "regret": "regret",
    "meaning_of_life": "philosophy",
    "free_will": "philosophy",
    "consciousness": "philosophy",
    "good_life": "philosophy",
}


def _question_blocked_by_boundary(person_id: Optional[int], question: dict) -> bool:
    if person_id is None or not question:
        return False
    topic = _QUESTION_BOUNDARY_TOPICS.get(question.get("key") or "")
    if not topic:
        return False
    try:
        return (
            boundary_memory.is_blocked(person_id, "ask", topic)
            or boundary_memory.is_blocked(person_id, "mention", topic)
            or boundary_memory.is_blocked(person_id, "ask", "questions")
        )
    except Exception as exc:
        _log.debug("question boundary check failed: %s", exc)
        return False


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
    r"not now|skip|drop it|leave it)\b",
    re.IGNORECASE,
)
_EMOTIONAL_BOUNDARY_PAT = re.compile(
    r"\b("
    r"i'?d rather not|i would rather not|rather not talk|"
    r"don'?t want to talk|do not want to talk|"
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
        return "Understood. I won't bring it up again unless you do."
    except Exception as exc:
        _log.debug("emotional check-in boundary handler failed: %s", exc)
        return None


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
        if behavior == "roast":
            return f"Noted. I won't roast you about {topic_phrase}."
        if behavior == "ask":
            return f"Got it. I won't ask about {topic_phrase} unless you bring it up."
        return f"Understood. I won't bring up {topic_phrase} unless you do."
    except Exception as exc:
        _log.debug("conversation boundary handler failed: %s", exc)
        return None


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


def _generate_repair_response(person_id: Optional[int], text: str, repair: dict) -> str:
    """Generate one concise recovery when the human flags a conversational miss."""
    prompt = repair_moves.build_prompt(repair)
    try:
        response = llm.get_response(prompt, person_id)
    except Exception as exc:
        _log.debug("repair response generation failed: %s", exc)
        response = ""
    response = (response or "").strip()
    if not response:
        response = repair_moves.fallback_response(repair)
    _speak_blocking(
        response,
        emotion="neutral",
        pre_beat_ms=150,
        post_beat_ms_override=300,
    )
    repair_moves.mark_handled()
    _log.info(
        "[repair] handled kind=%s severity=%s correction=%r user=%r response=%r",
        repair.get("kind"),
        repair.get("severity"),
        repair.get("correction"),
        text,
        response,
    )
    return response


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

    # Try question pool first — structured, depth-gated, no extra LLM call
    if person_id is not None:
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
            question_text = llm.generate_curiosity_question(response_text, user_text)
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
        return answered
    except Exception as exc:
        _log.debug("pending Q&A capture failed: %s", exc)
        return None


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

    if intent == "query_time":
        from datetime import datetime
        now = datetime.now()
        h12 = now.hour % 12 or 12
        ampm = "AM" if now.hour < 12 else "PM"
        return _say(
            f"Tell the user the current time is exactly {h12}:{now.minute:02d} {ampm} "
            f"in Rex character. Be brief. You MUST state both the hour and the minute "
            f"exactly as given ({h12}:{now.minute:02d}) — do not round to the hour, "
            f"do not omit the minute."
        )

    if intent == "query_weather":
        from awareness.chronoception import fetch_weather
        w = fetch_weather()
        temp = w.get("temp_f")
        desc = w.get("description") or w.get("condition") or "unknown"
        location = getattr(config, "WEATHER_LOCATION", "the local area")
        _log.info("[interaction] weather fetched for %r: %s", location, w)
        if temp is None:
            return _say(
                "The user asked about the weather but your weather feed is offline "
                "right now. Tell them you can't reach the weather service in one "
                "Rex-style line. Do NOT make up a temperature or conditions."
            )
        return _say(
            f"The real current weather in {location} is exactly {temp}°F and "
            f"{desc}. Tell the user the weather in one Rex-style line. "
            f"You MUST state the temperature exactly as given ({temp}°F) and the "
            f"conditions ({desc}) — do not round, do not invent different numbers, "
            f"do not substitute different conditions."
        )

    if intent == "query_games":
        from features import games as games_mod
        game_list = ", ".join(games_mod.available_game_names()) or "none right now"
        return _say(
            f"The user asked what games you can play. Your actual game list: {game_list}. "
            f"Tell them in one Rex-style line. Be brief."
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
            f"Summarize in one or two Rex-style lines. Be brief."
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
            f"in Rex character. Be brief."
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
        try:
            dj_mod.play(track)
        except Exception as exc:
            _log.debug("play_music intent play error: %s", exc)
            return _say(
                "You tried to play music but the playback system errored. Tell the "
                "user briefly in one Rex line."
            )
        return _say(
            f"You're now playing '{track.name}' ({track.description}) in response "
            f"to: '{raw_text}'. Announce it in one short in-character Rex line."
        )

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
# Speech segment processing
# ─────────────────────────────────────────────────────────────────────────────

def _handle_speech_segment(audio_array: np.ndarray) -> None:
    """Full processing pipeline for one detected speech segment in ACTIVE state."""
    global _session_exchange_count, _identity_prompt_until, _awaiting_followup_event
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture

    answered_question: Optional[dict] = None
    assistant_asked_question = False
    game_after_audio_path: Optional[str] = None

    # Randomised pre-response pause — prevents Rex from feeling instant/robotic
    delay_ms = random.randint(config.REACTION_DELAY_MS_MIN, config.REACTION_DELAY_MS_MAX)
    time.sleep(delay_ms / 1000.0)

    # Once Rex starts speaking in response, hold AEC suppression open across any
    # related output as one continuous window. Do not start it before
    # transcription; that made the logs look like user speech was captured
    # while playback suppression was active and encouraged premature filler.
    sequence_started = False
    try:
        # Concurrent transcription + speaker identification
        text, raw_best_id, raw_best_name, speaker_score = _process_audio(audio_array)

        if not text:
            return

        heard_log_text = text
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
                return

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
        try:
            ws_people = world_state.get("people")
            ws_identified = [p for p in ws_people if p.get("person_db_id") is not None]
            ws_person = ws_identified[0] if len(ws_identified) == 1 else None
        except Exception:
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

        if ws_person is not None:
            ws_pid = ws_person.get("person_db_id")
            ws_name = ws_person.get("face_id") or ws_person.get("voice_id")
            if person_id is None:
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
                elif engaged_is_visible and not _has_unknown_visible_person():
                    # Grief-flow override: Rex just asked this engaged person
                    # a direct question and is awaiting their reply. Voice-ID
                    # can score just below the engaged-visible floor on short
                    # utterances (a single name like "Joe", or noisy audio),
                    # but face match + top-candidate match is plenty of
                    # evidence in this context. Don't divert to off-camera
                    # handling and lose the grief flow's turn.
                    grief_floor = float(
                        getattr(config, "SPEAKER_ID_GRIEF_FLOW_FLOOR", 0.30)
                    )
                    if (
                        _grief_flow_active(ws_pid)
                        and raw_best_id == ws_pid
                        and speaker_score >= grief_floor
                    ):
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
                            "person %r is visible and no unknown face — treating as off-camera "
                            "unknown voice",
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
            if recent_engagement and not _has_unknown_visible_person():
                off_camera_unknown = True
                _log.info(
                    "[interaction] person resolution: no face, no voice match, "
                    "but %r was engaged recently — off-camera unknown voice",
                    recent_engagement.get("name"),
                )
            # Else: falls through to normal enrollment logic below

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

        # Consciousness asked an unknown person for their name. Open a short window
        # where single/short name replies are treated as enrollment input.
        if consciousness.consume_identity_prompt_request():
            _identity_prompt_until = max(
                _identity_prompt_until,
                time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS,
            )

        relationship_prompt_consumed = False

        # If the engaged person answers the unknown-face prompt with an actual
        # introduction ("this is my dad, Jeff"), let the dedicated introduction
        # flow consume it before the generic relationship extractor. That path
        # gives a welcome line, binds the visible face, and avoids asking the
        # already-answered "how do you know them?" question again.
        if person_id is not None:
            has_unknown_for_intro = _has_unknown_visible_person()
            parsed_intro = introductions.detect(
                text,
                has_unknown_face=has_unknown_for_intro,
            )
            if parsed_intro.is_introduction and (
                parsed_intro.subject_kind == "pet"
                or has_unknown_for_intro
                or bool(parsed_intro.name)
            ):
                intro_response = _handle_introduction_parse(
                    parsed_intro,
                    introducer_id=person_id,
                    introducer_name=person_name or f"person_{person_id}",
                    visible_newcomer=has_unknown_for_intro,
                )
                rel_ctx_for_intro = consciousness.consume_relationship_prompt_request()
                if rel_ctx_for_intro is not None:
                    relationship_prompt_consumed = True
                    consciousness.note_relationship_slot_handled(
                        str(rel_ctx_for_intro.get("slot_id") or "")
                    )
                if intro_response:
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
        rel_ctx = consciousness.consume_relationship_prompt_request()
        relationship_prompt_consumed = relationship_prompt_consumed or rel_ctx is not None
        if rel_ctx is not None:
            relationship_response = _handle_relationship_reply(
                rel_ctx, text, person_id, person_name
            )
            if relationship_response:
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
            has_unknown_for_intro = _has_unknown_visible_person()
            parsed_intro = introductions.detect(
                text,
                has_unknown_face=has_unknown_for_intro,
            )
            if parsed_intro.is_introduction and (
                parsed_intro.subject_kind == "pet"
                or has_unknown_for_intro
                or bool(parsed_intro.name)
            ):
                intro_response = _handle_introduction_parse(
                    parsed_intro,
                    introducer_id=person_id,
                    introducer_name=person_name or f"person_{person_id}",
                    visible_newcomer=has_unknown_for_intro,
                )
                if intro_response:
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

        # ── Address-mode classification ────────────────────────────────────────
        # If the utterance MENTIONS Rex but is not addressed TO him (e.g.
        # "say hi to Rex", "Rex is so fun"), do not generate a reply. Record
        # the mention to world_state.social.being_discussed so the consciousness
        # loop can decide whether to chime in. Skipped when Rex is in a pending
        # identity / face-reveal flow (those handlers expect this turn's text
        # as the user's reply).
        global _pending_offscreen_identify, _pending_face_reveal_confirm
        try:
            _addr_identity_window_active = time.monotonic() <= _identity_prompt_until
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
        # relationship edge if the engaged person also stated one.
        if _pending_offscreen_identify is not None:
            pending = _pending_offscreen_identify
            now_mono = time.monotonic()
            ttl = float(getattr(config, "OFFSCREEN_IDENTIFY_WINDOW_SECS", 30.0))
            prior_engaged_id = pending.get("prior_engaged_id")

            if (now_mono - pending["asked_at"]) > ttl:
                _log.info("[interaction] off-camera identify window expired — clearing")
                _pending_offscreen_identify = None
            elif person_id is not None and person_id == prior_engaged_id:
                # The engaged person is answering Rex's "who's that?" question.
                # This context makes a bare reply like "Joy" a name, not a mood.
                intro_name, rel_label = _extract_offscreen_identify_reply(
                    text,
                    person_name or "friend",
                )

                if intro_name:
                    new_pid = None
                    try:
                        new_pid = people_memory.enroll_person(intro_name)
                        if new_pid is not None:
                            # Enroll the VOICE from the off-camera audio we stored
                            # when Rex asked — not the engaged person's audio.
                            speaker_id.enroll_voice(new_pid, pending["audio"])
                            first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
                            if first_inc > 0:
                                people_memory.update_familiarity(new_pid, first_inc)

                            if _has_unknown_visible_person():
                                try:
                                    from vision import camera as _cam_mod
                                    from vision import face as _face_mod
                                    frame = _cam_mod.capture_still()
                                    if frame is not None:
                                        _face_mod.enroll_unknown_face(new_pid, frame)
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
                    except Exception as exc:
                        _log.error("off-camera identify enrollment failed: %s", exc)
                    # Consume whether or not we succeeded — don't retry on next turn.
                    _pending_offscreen_identify = None
                    if new_pid is not None:
                        ack_text = (
                            f"Got it: {intro_name}. Welcome to the frequency."
                        )
                        try:
                            introducer_name = (person_name or "the engaged person").split()[0]
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
                        return
                    return
                else:
                    # No name in this reply. Drop the pending state so Rex doesn't
                    # badger; if the off-camera voice speaks again we'll re-ask.
                    _log.info(
                        "[interaction] off-camera reply had no name — clearing pending: %r",
                        text,
                    )
                    _pending_offscreen_identify = None

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
                        first_reveal = reveal_name.split()[0]
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
        has_unknown = _has_unknown_visible_person()
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
                        parsed_name = parsed.get("name") or intro_name
                        relationship = parsed.get("relationship")
                    except Exception as exc:
                        _log.debug("describe-newcomer extract error: %s", exc)

                    new_id: Optional[int] = None
                    try:
                        new_id = people_memory.enroll_person(parsed_name)
                    except Exception as exc:
                        _log.warning("describe-newcomer enroll_person failed: %s", exc)

                    if new_id is not None:
                        first_inc = config.FAMILIARITY_INCREMENTS.get(
                            "first_enrollment", 0.0
                        )
                        if first_inc > 0:
                            people_memory.update_familiarity(new_id, first_inc)
                        try:
                            from vision import camera as _cam_mod
                            from vision import face as _face_mod
                            frame = _cam_mod.capture_still()
                            if frame is not None:
                                _face_mod.enroll_unknown_face(new_id, frame)
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
                        _identity_prompt_until = 0.0
                else:
                    # Newcomer self-introduction. Existing flow.
                    enrolled_id = _enroll_new_person(
                        intro_name,
                        audio_array,
                        enroll_unknown_face=bool(prior_engagement),
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
                        _identity_prompt_until = 0.0

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

        # If Rex had an outstanding event follow-up question, treat this utterance
        # as the outcome and close the loop in memory.
        if _awaiting_followup_event:
            pending_pid = _awaiting_followup_event.get("person_id")
            pending_event_id = _awaiting_followup_event.get("event_id")
            pid_matches = (
                pending_pid is None
                or person_id is None
                or pending_pid == person_id
            )
            if pending_event_id is not None and pid_matches:
                try:
                    events_memory.mark_followed_up(int(pending_event_id), text.strip())
                    _awaiting_followup_event = None
                except Exception as exc:
                    _log.debug("follow-up outcome write failed: %s", exc)

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
            print("[VOICE] Unknown voice detected", flush=True)

        speaker_label = person_name or "user"
        conv_memory.add_to_transcript(speaker_label, heard_log_text)
        conv_log.log_heard(person_name, heard_log_text)
        print(f"[HEARD] {speaker_label}: {heard_log_text}", flush=True)
        _log.info(
            "[interaction] speech segment — speaker=%r person_id=%s text=%r",
            speaker_label, person_id, text,
        )
        try:
            topic_thread.note_user_turn(text, person_id)
        except Exception as exc:
            _log.debug("topic thread user update failed: %s", exc)
        try:
            user_energy.note_user_turn(text, person_id)
        except Exception as exc:
            _log.debug("user energy text update failed: %s", exc)

        boundary_response = _handle_emotional_checkin_boundary(person_id, text)
        if boundary_response:
            _dismiss_pending_consent_prompts(person_id, text)
            try:
                consciousness.clear_response_wait()
            except Exception:
                pass
            _speak_blocking(
                boundary_response,
                emotion="neutral",
                pre_beat_ms=200,
                post_beat_ms_override=300,
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
            _speak_blocking(
                preference_response,
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            conv_memory.add_to_transcript("Rex", preference_response)
            conv_log.log_rex(preference_response)
            _session_exchange_count += 1
            _register_rex_utterance(preference_response)
            return

        answered_question = _maybe_capture_pending_qa(
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
                    name_for_prompt = (person_name or "").split()[0] or "you"
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
                    name_for_prompt = (person_name or "").split()[0] or "you"
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
            if person_id is not None:
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
                first_name_local = engaged_name_local.split()[0] if engaged_name_local else "friend"
                _pending_offscreen_identify = {
                    "audio": audio_array.copy(),
                    "asked_at": time.monotonic(),
                    "prior_engaged_id": (recent_engagement or {}).get("person_id"),
                    "prior_engaged_name": engaged_name_local,
                    "overheard_text": text,
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
            prior_first = ""
            if recent_engagement and recent_engagement.get("name"):
                prior_first = recent_engagement["name"].split()[0]
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
                if q_text:
                    _speak_blocking(q_text)
                    conv_memory.add_to_transcript("Rex", q_text)
                    conv_log.log_rex(q_text)
                    _register_rex_utterance(q_text)
                    _identity_prompt_until = time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS
                    _log.info(
                        "[interaction] newcomer-on-camera — Rex asked name "
                        "(prior=%r): %r",
                        prior_first or None, q_text,
                    )
                    return
            except Exception as exc:
                _log.debug("newcomer-on-camera ask error: %s", exc)

        # Empathy classification — kicked off in parallel so the affect/mode
        # directive is ready by the time the LLM-fallback path assembles its
        # system prompt. Result is cached per-person for future turns even if
        # this turn is handled deterministically by the command parser.
        _empathy_thread = None
        active_grief_for_turn = person_id is not None and _grief_flow_active(person_id)
        if getattr(config, "EMPATHY_ENABLED", True) and not active_grief_for_turn:
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

        repair_move = repair_moves.detect(text)
        if (
            repair_move
            and active_grief_for_turn
            and repair_move.get("kind") in {"misheard", "misunderstood"}
            and repair_move.get("correction")
        ):
            routing_text = repair_move["correction"]
            repair_moves.mark_handled()
            _log.info(
                "[repair] applying corrected text to active grief flow: %r",
                routing_text,
            )
        elif repair_move:
            response_text = _generate_repair_response(person_id, text, repair_move)
            used_agenda_llm = True

        # Layer-1 insult pre-check: keyword match → bump anger BEFORE the LLM
        # call so this turn's system prompt reflects the new escalation level.
        # Layer 2 (llm.analyze_sentiment in _post_response) skips its own
        # increment when this flag is set so we never double-count.
        pre_classified_insult = bool(repair_move)
        if response_text is None and personality.is_obvious_insult(text):
            new_level = personality.increment_anger(person_id)
            pre_classified_insult = True
            if person_id is not None:
                people_memory.update_relationship_scores(person_id, antagonism=+0.03)
            _log.info(
                "[interaction] layer-1 insult detected — anger now %d", new_level,
            )

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
                _speak_blocking(
                    grief_response,
                    emotion=grief_emotion,
                    pre_beat_ms=grief_pre_ms,
                    post_beat_ms_override=grief_post_ms,
                    voice_settings=grief_voice,
                )
                response_text = grief_response

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
                    if match is None and normalized_game_text in {"quit", "end", "end game", "quit game"}:
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
                        response_text = games_mod.handle_input(text, person_id)
                        completed = _speak_blocking(response_text)
                        if completed:
                            games_mod.on_response_spoken()
                            after_audio = games_mod.consume_pending_audio_after_response()
                            if after_audio and not _interrupted.is_set():
                                game_after_audio_path = after_audio
                        used_agenda_llm = True
                        match = None
            except Exception as exc:
                _log.debug("active game routing failed: %s", exc)
        if match is not None:
            response_text = _execute_command(match, person_id, person_name, text)
        elif response_text is None:
            if getattr(config, "INTENT_CLASSIFIER_ENABLED", False) and not active_grief_for_turn:
                try:
                    intent = intent_classifier.classify(text)
                except Exception as exc:
                    _log.debug("intent classification error: %s", exc)
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
            if response_text is None:
                if _empathy_thread is not None:
                    _empathy_join_timeout = float(getattr(
                        config, "EMPATHY_CLASSIFY_JOIN_TIMEOUT_SECS", 4.0,
                    ))
                    _empathy_thread.join(timeout=_empathy_join_timeout)
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
                    _speak_blocking(
                        grief_response,
                        emotion=grief_emotion,
                        pre_beat_ms=grief_pre_ms,
                        post_beat_ms_override=grief_post_ms,
                        voice_settings=grief_voice,
                    )
                    response_text = grief_response
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
                    response_text = _stream_llm_response(
                        text,
                        person_id,
                        answered_question=answered_question,
                    )
                    used_agenda_llm = True

        if response_text:
            conv_memory.add_to_transcript("Rex", response_text)
            conv_log.log_rex(response_text)
            _session_exchange_count += 1
            _register_rex_utterance(response_text)
            assistant_asked_question = _assistant_asked_question(response_text)
            try:
                end_thread.mark_closure_spoken()
            except Exception:
                pass

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
            prior_first = prior_name.split()[0] if prior_name else "them"
            newcomer_first = newcomer_name.split()[0] if newcomer_name else "there"
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
            question_tail = None
            if assistant_asked_question:
                question_tail = float(
                    getattr(
                        config,
                        "POST_QUESTION_PLAYBACK_SUPPRESSION_SECS",
                        getattr(config, "POST_PLAYBACK_SUPPRESSION_SECS", 0.5),
                    )
                )
            echo_cancel.end_sequence(flush=False, tail_secs=question_tail)
            sequence_started = False
            if game_after_audio_path and not _interrupted.is_set():
                speech_queue.enqueue_audio_file(
                    game_after_audio_path,
                    priority=1,
                    tag="game:after_audio",
                )
                game_after_audio_path = None

        _post_response(
            text,
            person_id,
            person_name,
            assistant_asked_question=assistant_asked_question,
            pre_classified_insult=pre_classified_insult,
        )
    finally:
        if sequence_started:
            echo_cancel.end_sequence()


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

        # ── QUIET — discard everything including wake word events ──────────────
        if current_state == State.QUIET:
            _wake_word_fired.clear()
            _stop_event.wait(0.1)
            continue

        # ── SLEEP — only wakeuprex fires the callback; gate transition here ────
        if current_state == State.SLEEP:
            if _wake_word_fired.is_set():
                _wake_word_fired.clear()
                with _wake_lock:
                    model = _last_wake_word
                # wake_word.py only routes 'wakeuprex' to the callback in SLEEP;
                # the check is defensive in case that ever changes.
                if model == "wakeuprex":
                    state_module.set_state(State.ACTIVE)
                    _last_speech_at = time.monotonic()
                    _wake_ack()
            _stop_event.wait(0.05)
            continue

        # ── IDLE — wake word first; optional direct speech activation ──────────
        if current_state == State.IDLE:
            if _wake_word_fired.is_set():
                _wake_word_fired.clear()
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

            chunk = stream.get_audio_chunk(_CHUNK_SECS)
            if len(chunk) == 0:
                _stop_event.wait(_CHUNK_SECS)
                continue
            _idle_speech = vad.is_speech(_chunk_for_vad(chunk))
            _situation_assessor.set_vad_active(_idle_speech)
            if not _idle_speech:
                _stop_event.wait(_CHUNK_SECS)
                continue

            if time.monotonic() < _listen_resume_at:
                _stop_event.wait(_CHUNK_SECS)
                continue

            # First detected speech after the post-TTS window may be the human's
            # immediate answer to Rex's question. The queue callback already
            # flushed playback tail when TTS ended, so do not discard this VAD
            # positive chunk.
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
                _handle_speech_segment(audio_segment)
            finally:
                _restore_dj_volume(_dj_restore_volume)
                _end_user_turn()
            continue

        # ── ACTIVE — full continuous listening ─────────────────────────────────

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
        if time.monotonic() - _last_speech_at >= effective_idle_timeout:
            _log.info("[interaction] conversation idle timeout — returning to IDLE")
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

        # Discard speech onset during the post-TTS deaf window so Rex's own
        # voice tail (or the first 0.8 s of reverb decay) cannot self-trigger.
        if time.monotonic() < _listen_resume_at:
            _stop_event.wait(_CHUNK_SECS)
            continue

        # First detected speech after the post-TTS window may be the human's
        # immediate answer. Playback tail was already flushed by the TTS-done
        # callback, so keep this chunk and start accumulating.
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
        direct_audio_path = None
        try:
            direct_audio_path = speech_queue.current_audio_path()
        except Exception:
            direct_audio_path = None
        if speech_queue.is_speaking() or output_gate.is_busy():
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

def start() -> None:
    """Start the wake word detector and the continuous interaction loop."""
    global _thread, _identity_prompt_until, _awaiting_followup_event

    if _thread and _thread.is_alive():
        _log.warning("[interaction] already running")
        return

    _stop_event.clear()
    _wake_word_fired.clear()
    _interrupted.clear()
    _session_person_ids.clear()
    _identity_prompt_until = 0.0
    _awaiting_followup_event = None
    topic_thread.clear()
    user_energy.clear()
    question_budget.clear()
    repair_moves.clear()
    end_thread.clear()
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

    speech_queue.register_on_item_done(_arm_post_tts_window)
    wake_word.start(_on_wake_word)
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
    global _pending_introduction, _pending_intro_followup, _pending_intro_voice_capture

    _stop_event.set()
    wake_word.stop()

    if _thread:
        _thread.join(timeout=3.0)
        if _thread.is_alive():
            _log.warning("[interaction] loop thread did not stop cleanly")
        _thread = None

    _awaiting_followup_event = None
    _identity_prompt_until = 0.0
    _pending_introduction = None
    _pending_intro_followup = None
    _pending_intro_voice_capture = None
    topic_thread.clear()
    user_energy.clear()
    question_budget.clear()
    repair_moves.clear()
    end_thread.clear()
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

    _log.info("[interaction] stopped")
