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
from intelligence import command_parser, llm, personality
from intelligence import consciousness
from intelligence import intent_classifier
from memory import facts as facts_memory
from memory import conversations as conv_memory
from memory import people as people_memory
from memory import events as events_memory
from memory import relationships as rel_memory
from awareness import interoception
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

# Per-session set of (person_id) we've already attempted a face reveal for
# but were declined ("no") — so Rex doesn't keep re-asking the same question.
_face_reveal_declined: set[int] = set()

# Per-session set of person_ids whose voice biometric was auto-refreshed this
# session. Cap at one refresh per person per session so we don't spam new
# biometric rows when someone speaks a lot.
_voice_refreshed_this_session: set[int] = set()

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


def _speak_blocking(
    text: str,
    emotion: str = "neutral",
    priority: int = 1,
    pre_beat_ms: int = 0,
) -> bool:
    """
    Enqueue text for speech and block until playback finishes, monitoring for
    wake-word interruption.  Returns True on normal completion, False if cut short.

    priority 1 = normal response; priority 2 = urgent acknowledgment.
    Enqueueing drops all waiting items of lower priority and preempts any
    currently-playing item of lower priority.
    """
    if not _can_speak() or not text or not text.strip():
        return True

    # Post-punchline beat: a brief silence after a normal-priority response so
    # the line lands. Skipped for urgent acks (priority >= 2), filler, etc.
    post_beat_ms = 0
    if priority == 1:
        beat_min = getattr(config, "POST_PUNCHLINE_BEAT_MS_MIN", 0)
        beat_max = getattr(config, "POST_PUNCHLINE_BEAT_MS_MAX", 0)
        if beat_max > 0 and beat_max >= beat_min:
            post_beat_ms = random.randint(beat_min, beat_max)

    done = speech_queue.enqueue(
        text, emotion, priority=priority,
        pre_beat_ms=pre_beat_ms, post_beat_ms=post_beat_ms,
    )

    while not done.wait(timeout=0.05):
        if _interrupted.is_set():
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            # Drop our queued item in case it hasn't played yet, then wait for
            # whatever was playing (or our item if it just started) to finish.
            speech_queue.clear_below_priority(priority + 1)
            done.wait(timeout=0.5)
            return False

    # Normal completion — block new speech-onset detections briefly and discard
    # any mic audio that captured Rex's voice tail during playback.
    global _listen_resume_at, _post_tts_flush_needed
    _listen_resume_at = time.monotonic() + config.POST_SPEECH_LISTEN_DELAY_SECS
    stream.flush()
    _post_tts_flush_needed = True
    return True


def _speak_async(text: str, emotion: str = "neutral") -> None:
    if not _can_speak() or not text:
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
    pool = config.LATENCY_FILLER_LINES
    candidates = [l for l in pool if l != _last_filler] or pool
    chosen = random.choice(candidates)
    _last_filler = chosen
    _speak_async(chosen)


def _assistant_asked_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and ("?" in cleaned)


def _register_rex_utterance(text: str, wait_secs: Optional[float] = None) -> None:
    if not text or not text.strip():
        return
    try:
        consciousness.note_rex_utterance(text, wait_secs=wait_secs)
    except Exception:
        pass


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
) -> None:
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
        return

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
            return
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
        return

    # ── Mode A: enroll newcomer + save edge ────────────────────────────────────
    if not name:
        # Deflection or no name given. Don't badger — mark slot handled so Rex
        # moves on. Relationship could still be saved if name later surfaces.
        _log.info(
            "[interaction] mode-A relationship reply had no name — user_text=%r parsed=%r",
            user_text, parsed,
        )
        consciousness.note_relationship_slot_handled(slot_id)
        return

    # Enroll the newcomer. Use enroll_unknown_face so we don't rebind the
    # engaged person's face to the new name.
    try:
        from vision import camera as camera_mod
        from vision import face as face_mod

        new_id = people_memory.enroll_person(name)
        if new_id is None:
            _log.error("[interaction] failed to create DB row for newcomer %r", name)
            consciousness.note_relationship_slot_handled(slot_id)
            return

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

        # Save the edge (engaged speaker → newcomer, per speaker's perspective).
        if engaged_id and relationship:
            try:
                from memory import social as social_memory
                social_memory.save_relationship(
                    from_person_id=engaged_id,
                    to_person_id=new_id,
                    relationship=relationship,
                    described_by=engaged_id,
                )
            except Exception as exc:
                _log.warning("social.save_relationship failed: %s", exc)

        _log.info(
            "[interaction] mode-A enrolled newcomer %s (person_id=%s) as %s of %s",
            name, new_id, relationship or "acquaintance", engaged_name,
        )
    except Exception as exc:
        _log.error("relationship enrollment failed: %s", exc)
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

        is_speech = vad.is_speech(chunk)
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

def _stream_llm_response(text: str, person_id: Optional[int]) -> str:
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

    full_text = llm.get_response(text, person_id)

    pre_beat_ms = 0
    if full_text and full_text.strip() and not _interrupted.is_set():
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
        _speak_blocking(full_text, pre_beat_ms=pre_beat_ms)
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# Command execution
# ─────────────────────────────────────────────────────────────────────────────

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
    if key in ("start_trivia", "start_game", "stop_game"):
        try:
            from features import games as games_mod
            if key == "start_trivia":
                games_mod.start_trivia()
                return _say("Starting trivia now. Introduce it in one in-character line.")
            if key == "start_game":
                game = args.get("game", "")
                games_mod.start_game(game)
                return _say(f"Starting '{game}'. Introduce it in one in-character line.")
            if key == "stop_game":
                games_mod.stop_game()
                return _say("Game stopped. One in-character line.")
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

    transcript = conv_memory.get_session_transcript()
    if not transcript:
        _session_exchange_count = 0
        _session_person_ids.clear()
        _identity_prompt_until = 0.0
        _awaiting_followup_event = None
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
    _session_exchange_count = 0
    _session_person_ids.clear()
    _identity_prompt_until = 0.0
    _awaiting_followup_event = None
    _voice_refreshed_this_session.clear()
    _face_reveal_declined.clear()
    try:
        consciousness.clear_response_wait()
        consciousness.clear_engagement()
    except Exception:
        pass
    _log.info("[interaction] session ended — summary saved, transcript cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Curiosity routine
# ─────────────────────────────────────────────────────────────────────────────

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
        return None

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
            answered = rel_memory.get_answered_question_keys(person_id)

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
                if candidate["key"] in answered:
                    continue
                # Skip if Rex already knows something about this topic
                q_key = candidate["key"]
                if q_key in known_fact_keys or q_key in known_fact_categories:
                    _log.debug(
                        "curiosity_check: skipping %r — fact already recorded", q_key
                    )
                    continue
                question_text = candidate.get("text", "")
                pool_question = candidate
                break
        except Exception as exc:
            _log.debug("curiosity_check pool error: %s", exc)

    # LLM fallback — contextual question when pool is empty or person is unknown
    if not question_text:
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
            rel_memory.save_qa(
                person_id,
                pool_question["key"],
                question_text,
                "",
                pool_question.get("depth", 1),
            )
        except Exception as exc:
            _log.debug("curiosity_check save_qa error: %s", exc)

    return question_text


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
        # Surface the actual aliases users would say, not the internal keys.
        from features import games as games_mod
        seen: list[str] = []
        for alias, key in games_mod._GAME_ALIASES.items():
            label = {
                "i_spy": "I Spy",
                "20_questions": "20 Questions",
                "trivia": "Trivia",
                "word_association": "Word Association",
            }.get(key)
            if label and label not in seen:
                seen.append(label)
        game_list = ", ".join(seen) if seen else "none right now"
        return _say(
            f"The user asked what games you can play. Your actual game list: {game_list}. "
            f"Tell them in one Rex-style line. Be brief."
        )

    if intent == "query_capabilities":
        capabilities = (
            "hold a real conversation and remember people across visits; "
            "recognize faces and voices; describe what you see through your camera; "
            "play games (Trivia, I Spy, 20 Questions, Word Association); "
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

    # Randomised pre-response pause — prevents Rex from feeling instant/robotic
    delay_ms = random.randint(config.REACTION_DELAY_MS_MIN, config.REACTION_DELAY_MS_MAX)
    time.sleep(delay_ms / 1000.0)

    # Hold AEC suppression open across filler + response as one continuous window.
    # Individual tts.speak() calls will not turn suppression off mid-sequence;
    # end_sequence() in the finally block does the final flush and tail.
    echo_cancel.start_sequence()
    try:
        # Speak filler asynchronously while transcription + speaker ID run
        _speak_filler()

        # Concurrent transcription + speaker identification
        text, raw_best_id, raw_best_name, speaker_score = _process_audio(audio_array)

        if not text:
            return

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
                if engaged_is_visible and not _has_unknown_visible_person():
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

        # Consciousness asked the ENGAGED person about an unknown newcomer.
        # Try to extract {name, relationship}, enroll the newcomer using the
        # current unknown face, and save the relationship edge.
        rel_ctx = consciousness.consume_relationship_prompt_request()
        if rel_ctx is not None:
            _handle_relationship_reply(rel_ctx, text, person_id, person_name)

        # Rex previously asked "who's that speaking?" about an off-camera unknown
        # voice. If this reply came from the engaged person and names the
        # off-camera speaker, enroll their voice (using the STORED audio of the
        # original utterance, not this engaged-person audio) and save a
        # relationship edge if the engaged person also stated one.
        global _pending_offscreen_identify
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
                # Use the LLM to extract BOTH a name AND a relationship label.
                try:
                    parsed = llm.extract_relationship_introduction(
                        text, person_name or "friend"
                    )
                except Exception as exc:
                    _log.debug("offscreen identify extract error: %s", exc)
                    parsed = {"name": None, "relationship": None}
                intro_name = parsed.get("name")
                rel_label = parsed.get("relationship")

                if intro_name:
                    try:
                        new_pid = people_memory.enroll_person(intro_name)
                        if new_pid is not None:
                            # Enroll the VOICE from the off-camera audio we stored
                            # when Rex asked — not the engaged person's audio.
                            speaker_id.enroll_voice(new_pid, pending["audio"])
                            first_inc = config.FAMILIARITY_INCREMENTS.get("first_enrollment", 0.0)
                            if first_inc > 0:
                                people_memory.update_familiarity(new_pid, first_inc)

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
        global _pending_face_reveal_confirm
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
                # Use the face-aware enrollment path when we have a known-person
                # context to preserve; otherwise normal _enroll_new_person is fine.
                prior_engagement = None
                try:
                    prior_engagement = consciousness.get_recent_engagement()
                except Exception:
                    prior_engagement = None

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
            except Exception:
                pass
        else:
            print("[VOICE] Unknown voice detected", flush=True)

        speaker_label = person_name or "user"
        conv_memory.add_to_transcript(speaker_label, text)
        conv_log.log_heard(person_name, text)
        print(f"[HEARD] {speaker_label}: {text}", flush=True)
        _log.info(
            "[interaction] speech segment — speaker=%r person_id=%s text=%r",
            speaker_label, person_id, text,
        )

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
        # Skip if a pending identify is already in flight (from a previous turn).
        if off_camera_unknown and _pending_offscreen_identify is None:
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

        # Layer-1 insult pre-check: keyword match → bump anger BEFORE the LLM
        # call so this turn's system prompt reflects the new escalation level.
        # Layer 2 (llm.analyze_sentiment in _post_response) skips its own
        # increment when this flag is set so we never double-count.
        pre_classified_insult = False
        if personality.is_obvious_insult(text):
            new_level = personality.increment_anger(person_id)
            pre_classified_insult = True
            if person_id is not None:
                people_memory.update_relationship_scores(person_id, antagonism=+0.03)
            _log.info(
                "[interaction] layer-1 insult detected — anger now %d", new_level,
            )

        # Command parser → local dispatch or LLM fallback
        match = command_parser.parse(text)
        if match is not None:
            response_text = _execute_command(match, person_id, person_name, text)
        else:
            response_text = None
            if getattr(config, "INTENT_CLASSIFIER_ENABLED", False):
                try:
                    intent = intent_classifier.classify(text)
                except Exception as exc:
                    _log.debug("intent classification error: %s", exc)
                    intent = "general"
                _log.info("[interaction] intent classifier: %s", intent)
                if intent != "general":
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
                response_text = _stream_llm_response(text, person_id)

        assistant_asked_question = False
        if response_text:
            conv_memory.add_to_transcript("Rex", response_text)
            conv_log.log_rex(response_text)
            _session_exchange_count += 1
            _register_rex_utterance(response_text)
            assistant_asked_question = _assistant_asked_question(response_text)

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
        ):
            curiosity_q = _curiosity_check(response_text, text, person_id, person_name)
            if curiosity_q:
                conv_memory.add_to_transcript("Rex", curiosity_q)
                conv_log.log_rex(curiosity_q)
                _register_rex_utterance(curiosity_q)
                assistant_asked_question = True

        _post_response(
            text,
            person_id,
            person_name,
            assistant_asked_question=assistant_asked_question,
            pre_classified_insult=pre_classified_insult,
        )
    finally:
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
            if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
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
            _idle_speech = vad.is_speech(chunk)
            _situation_assessor.set_vad_active(_idle_speech)
            if not _idle_speech:
                _stop_event.wait(_CHUNK_SECS)
                continue

            if time.monotonic() < _listen_resume_at:
                _stop_event.wait(_CHUNK_SECS)
                continue

            # First chunk after the post-TTS window: flush decay audio and skip.
            global _post_tts_flush_needed
            if _post_tts_flush_needed:
                _post_tts_flush_needed = False
                stream.flush()
                _stop_event.wait(_CHUNK_SECS)
                continue

            _log.info("[interaction] speech detected in IDLE — activating without wake word")
            state_module.set_state(State.ACTIVE)
            speech_start = time.monotonic()
            _last_speech_at = speech_start

            audio_segment = _accumulate_speech(speech_start)
            if audio_segment is None or len(audio_segment) == 0:
                continue

            _last_speech_at = time.monotonic()
            _handle_speech_segment(audio_segment)
            continue

        # ── ACTIVE — full continuous listening ─────────────────────────────────

        # Idle timeout → end session and return to IDLE
        if time.monotonic() - _last_speech_at >= idle_timeout:
            _log.info("[interaction] conversation idle timeout — returning to IDLE")
            _end_session()
            state_module.set_state(State.IDLE)
            continue

        # Read a small audio chunk and run VAD
        chunk = stream.get_audio_chunk(_CHUNK_SECS)
        if len(chunk) == 0:
            _stop_event.wait(_CHUNK_SECS)
            continue

        _active_speech = vad.is_speech(chunk)
        _situation_assessor.set_vad_active(_active_speech)
        if not _active_speech:
            _stop_event.wait(_CHUNK_SECS)
            continue

        # Discard speech onset during the post-TTS deaf window so Rex's own
        # voice tail (or the first 0.8 s of reverb decay) cannot self-trigger.
        if time.monotonic() < _listen_resume_at:
            _stop_event.wait(_CHUNK_SECS)
            continue

        # First chunk after the post-TTS window: flush decay audio and skip.
        if _post_tts_flush_needed:
            _post_tts_flush_needed = False
            stream.flush()
            _stop_event.wait(_CHUNK_SECS)
            continue

        # ── Speech detected ────────────────────────────────────────────────────
        speech_start = time.monotonic()
        _last_speech_at = speech_start

        # Mid-speech interruption: stop TTS, acknowledge, flush the mic buffer of
        # Rex's voice tail, then WAIT for a fresh VAD rising edge before
        # accumulating. Without this, the rolling buffer still holds ~seconds of
        # Rex's own voice which Whisper concatenates onto the user's utterance.
        if speech_queue.is_speaking() or output_gate.is_busy() or echo_cancel.is_suppressed():
            _interrupted.set()
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            # Brief settle so the worker can clean up its finally block
            time.sleep(0.1)
            _interrupted.clear()
            _interrupt_ack()

            # Drop the polluted buffer and re-arm. The next user utterance must
            # trigger VAD again; the original speech_start is discarded.
            stream.flush()
            _listen_resume_at = time.monotonic() + config.POST_SPEECH_LISTEN_DELAY_SECS
            _post_tts_flush_needed = True
            continue

        # Accumulate the full utterance
        audio_segment = _accumulate_speech(speech_start)
        if audio_segment is None or len(audio_segment) == 0:
            continue

        _last_speech_at = time.monotonic()
        _handle_speech_segment(audio_segment)


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
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

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

    _stop_event.set()
    wake_word.stop()

    if _thread:
        _thread.join(timeout=3.0)
        if _thread.is_alive():
            _log.warning("[interaction] loop thread did not stop cleanly")
        _thread = None

    _awaiting_followup_event = None
    _identity_prompt_until = 0.0
    try:
        consciousness.clear_response_wait()
    except Exception:
        pass

    _log.info("[interaction] stopped")
