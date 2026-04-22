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
from audio import tts, output_gate
from intelligence import command_parser, llm, personality
from intelligence import consciousness
from memory import facts as facts_memory
from memory import conversations as conv_memory
from memory import people as people_memory
from awareness import interoception
from world_state import world_state

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

# Time window (set by consciousness) where a short bare-name reply is accepted.
_identity_prompt_until: float = 0.0

_IDENTITY_REPLY_WINDOW_SECS = 45.0
_NAME_MAX_WORDS = 3

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

    if tts.is_speaking():
        _interrupted.set()

    _wake_word_fired.set()


# ─────────────────────────────────────────────────────────────────────────────
# Speech helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_speak() -> bool:
    return state_module.get_state() not in (State.QUIET, State.SHUTDOWN)


def _speak_blocking(text: str, emotion: str = "neutral") -> bool:
    """
    Speak text synchronously, monitoring for wake-word interruption.
    Returns True if playback completed normally, False if a wake word cut it short.
    """
    if not _can_speak() or not text or not text.strip():
        return True

    done = threading.Event()

    def _task() -> None:
        try:
            tts.speak(text, emotion)
        finally:
            done.set()

    t = threading.Thread(target=_task, daemon=True, name="tts-speak")
    t.start()

    while not done.wait(timeout=0.05):
        if _interrupted.is_set():
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            done.wait(timeout=0.5)  # let tts.speak() clean up its finally block
            return False

    # Normal completion — block new speech-onset detections briefly and discard
    # any mic audio that captured Rex's voice tail during playback.
    global _listen_resume_at
    _listen_resume_at = time.monotonic() + config.POST_SPEECH_LISTEN_DELAY_SECS
    stream.flush()
    return True


def _speak_async(text: str, emotion: str = "neutral") -> None:
    if not _can_speak() or not text:
        return
    if tts.is_speaking() or output_gate.is_busy():
        return
    threading.Thread(
        target=tts.speak, args=(text, emotion), daemon=True, name="tts-async"
    ).start()


def _wake_ack() -> None:
    pool = config.WAKE_ACKNOWLEDGMENTS
    _speak_blocking(random.choice(pool))


def _interrupt_ack() -> None:
    _speak_blocking(random.choice(config.INTERRUPT_ACKNOWLEDGMENTS))


def _speak_filler() -> None:
    """Speak a latency filler line asynchronously, never repeating back-to-back."""
    global _last_filler
    pool = config.LATENCY_FILLER_LINES
    candidates = [l for l in pool if l != _last_filler] or pool
    chosen = random.choice(candidates)
    _last_filler = chosen
    _speak_async(chosen)


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


def _enroll_new_person(name: str, audio_array: np.ndarray) -> Optional[int]:
    """
    Enroll a brand-new person and attach available voice/face biometrics.
    Returns person_id on success.
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
) -> tuple[str, Optional[int], Optional[str]]:
    """
    Run transcription and speaker ID simultaneously in two threads.
    Returns (transcribed_text, person_id, person_name).
    """
    text_box: list[str] = [""]
    speaker_box: list = [None, None]  # [person_id, name]

    def _transcribe() -> None:
        text_box[0] = transcription.transcribe(audio_array)

    def _identify() -> None:
        pid, name, _score = speaker_id.identify_speaker(audio_array)
        speaker_box[0] = pid
        speaker_box[1] = name

    t1 = threading.Thread(target=_transcribe, daemon=True, name="transcription")
    t2 = threading.Thread(target=_identify, daemon=True, name="speaker-id")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return text_box[0] or "", speaker_box[0], speaker_box[1]


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

        if vad.is_speech(chunk):
            silence_elapsed = 0.0
        else:
            silence_elapsed += _CHUNK_SECS
            if silence_elapsed >= silence_timeout:
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
    """
    Stream GPT-4o-mini response chunks, speaking at sentence boundaries.
    Stops early if _interrupted is set. Returns the full response string.
    """
    all_chunks: list[str] = []
    sentence_buf: list[str] = []

    for chunk in llm.stream_response(text, person_id):
        if _interrupted.is_set():
            break

        sentence_buf.append(chunk)
        all_chunks.append(chunk)

        combined = "".join(sentence_buf)
        if combined.rstrip().endswith((".", "?", "!", "...", "—")):
            sentence = combined.strip()
            if sentence:
                if not _speak_blocking(sentence):
                    break  # interrupted mid-stream
            sentence_buf.clear()

    # Speak any trailing text if not interrupted
    if sentence_buf and not _interrupted.is_set():
        remaining = "".join(sentence_buf).strip()
        if remaining:
            _speak_blocking(remaining)

    return "".join(all_chunks)


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
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return _say(
            f"The current time is {h12}:{ctx.get('minute', 0):02d} {ampm}. "
            f"Tell the person the time in one in-character line."
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

def _post_response(user_text: str, person_id: Optional[int]) -> None:
    """
    Run after every response in ACTIVE state. Sentiment, facts, follow-up,
    and interoception are handled here. Sentiment/facts run in a background
    thread; follow-up delivery and interoception run in the calling thread.
    """
    # ── Follow-up delivery (sync — spoken as part of this turn) ───────────────
    if person_id is not None:
        try:
            followups = consciousness.get_pending_followup(person_id)
            if followups:
                for event in followups:
                    event_name = event.get("event_name", "that thing you mentioned")
                    resp = llm.get_response(
                        f"You're following up on something this person mentioned before: "
                        f"'{event_name}'. Ask how it went in one short Rex-style line.",
                        person_id,
                    )
                    if resp:
                        conv_memory.add_to_transcript("Rex", resp)
                        _speak_blocking(resp)
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
                # Feed the last 6 entries (3 exchanges) to keep the prompt tight
                recent = transcript[-6:] if len(transcript) >= 6 else transcript
                new_facts = llm.extract_facts(person_id, recent)
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
    global _session_exchange_count, _identity_prompt_until

    transcript = conv_memory.get_session_transcript()
    if not transcript:
        _session_exchange_count = 0
        _session_person_ids.clear()
        _identity_prompt_until = 0.0
        return

    for person_id in list(_session_person_ids):
        try:
            summary = llm.generate_session_summary(person_id, transcript)
            if summary:
                conv_memory.save_conversation(
                    person_id,
                    summary,
                    emotion_tone="neutral",
                    topics="",
                )

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
    _log.info("[interaction] session ended — summary saved, transcript cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Speech segment processing
# ─────────────────────────────────────────────────────────────────────────────

def _handle_speech_segment(audio_array: np.ndarray) -> None:
    """Full processing pipeline for one detected speech segment in ACTIVE state."""
    global _session_exchange_count, _identity_prompt_until

    # Randomised pre-response pause — prevents Rex from feeling instant/robotic
    delay_ms = random.randint(config.REACTION_DELAY_MS_MIN, config.REACTION_DELAY_MS_MAX)
    time.sleep(delay_ms / 1000.0)

    # Speak filler asynchronously while transcription + speaker ID run
    _speak_filler()

    # Concurrent transcription + speaker identification
    text, person_id, person_name = _process_audio(audio_array)

    if not text:
        return

    # Consciousness asked an unknown person for their name. Open a short window
    # where single/short name replies are treated as enrollment input.
    if consciousness.consume_identity_prompt_request():
        _identity_prompt_until = max(
            _identity_prompt_until,
            time.monotonic() + _IDENTITY_REPLY_WINDOW_SECS,
        )

    # If we don't recognize the speaker but there is an unknown person visible,
    # attempt self-identification enrollment from this utterance.
    now_mono = time.monotonic()
    should_attempt_enroll = (
        person_id is None
        and (_has_unknown_visible_person() or now_mono <= _identity_prompt_until)
    )
    if should_attempt_enroll:
        allow_bare = now_mono <= _identity_prompt_until
        intro_name = _extract_introduced_name(text, allow_bare_name=allow_bare)
        if intro_name:
            enrolled_id = _enroll_new_person(intro_name, audio_array)
            if enrolled_id is not None:
                person_id = enrolled_id
                person_name = intro_name
                _identity_prompt_until = 0.0

    if person_id is not None:
        _session_person_ids.add(person_id)
        voice_name = person_name or f"person_{person_id}"
        print(f"[VOICE] Known voice detected: {voice_name} (person_id={person_id})", flush=True)
    else:
        print("[VOICE] Unknown voice detected", flush=True)

    speaker_label = person_name or "user"
    conv_memory.add_to_transcript(speaker_label, text)
    print(f"[HEARD] {speaker_label}: {text}", flush=True)
    _log.info(
        "[interaction] speech segment — speaker=%r person_id=%s text=%r",
        speaker_label, person_id, text,
    )

    # Command parser → local dispatch or LLM fallback
    match = command_parser.parse(text)
    if match is not None:
        response_text = _execute_command(match, person_id, person_name, text)
    else:
        response_text = _stream_llm_response(text, person_id)

    if response_text:
        conv_memory.add_to_transcript("Rex", response_text)
        _session_exchange_count += 1

    _post_response(text, person_id)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def _loop() -> None:
    global _last_speech_at

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

            # Optional hands-free behavior: allow normal speech to activate Rex
            # directly from IDLE without a wake word.
            if not getattr(config, "IDLE_LISTEN_WITHOUT_WAKE_WORD", False):
                _stop_event.wait(0.05)
                continue

            chunk = stream.get_audio_chunk(_CHUNK_SECS)
            if len(chunk) == 0:
                _stop_event.wait(_CHUNK_SECS)
                continue
            if not vad.is_speech(chunk):
                _stop_event.wait(_CHUNK_SECS)
                continue

            if time.monotonic() < _listen_resume_at:
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

        if not vad.is_speech(chunk):
            _stop_event.wait(_CHUNK_SECS)
            continue

        # Discard speech onset during the post-TTS deaf window so Rex's own
        # voice tail (or the first 0.8 s of reverb decay) cannot self-trigger.
        if time.monotonic() < _listen_resume_at:
            _stop_event.wait(_CHUNK_SECS)
            continue

        # ── Speech detected ────────────────────────────────────────────────────
        speech_start = time.monotonic()
        _last_speech_at = speech_start

        # Mid-speech interruption: stop TTS, acknowledge, then listen
        if tts.is_speaking():
            _interrupted.set()
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            # Brief settle so tts.speak() can clean up its finally block
            time.sleep(0.1)
            _interrupted.clear()
            _interrupt_ack()

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
    global _thread, _identity_prompt_until

    if _thread and _thread.is_alive():
        _log.warning("[interaction] already running")
        return

    _stop_event.clear()
    _wake_word_fired.clear()
    _interrupted.clear()
    _session_person_ids.clear()
    _identity_prompt_until = 0.0

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
    global _thread

    _stop_event.set()
    wake_word.stop()

    if _thread:
        _thread.join(timeout=5.0)
        if _thread.is_alive():
            _log.warning("[interaction] loop thread did not stop cleanly")
        _thread = None

    _log.info("[interaction] stopped")
