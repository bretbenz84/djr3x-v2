"""
ElevenLabs TTS with SHA-256 file cache.

speak() is synchronous and blocks until audio has finished playing.
Callers are responsible for running it in a thread when non-blocking behaviour
is needed.

Cache strategy
──────────────
Before every API call, a SHA-256 of (text + voice_id + model_id + voice_settings) is
computed and checked against assets/audio/tts_cache/{hash}.mp3. The voice_settings
component is the resolved expressive ElevenLabs settings (stability / similarity_boost
/ style / use_speaker_boost / speed), so lines that differ only by emotion-derived
voice settings get distinct cache files; it folds in as an empty string when there is
no override, so default-mode lines keep their pre-existing cache entries. On a hit the
file is played
directly — no API call is made. On a miss the ElevenLabs streaming response is
collected, written to the cache file, then played from disk. Writing then reading
from disk (rather than decoding from a BytesIO) keeps the MP3 decode path
identical for both cache hits and misses.

Echo cancellation
─────────────────
set_playing(True/False) is called on audio.echo_cancel so the mic suppression
activates for the duration of playback. The call to set_playing(False) is in a
finally block and fires unconditionally — even if sounddevice raises — so the
suppression cannot be left permanently active.

Mouth LEDs + servo speech motion
────────────────────────────────
A daemon thread iterates through the audio array in TTS_LED_UPDATE_INTERVAL_SECS
chunks, computes the RMS of each chunk, and calls leds_head.speak_level(brightness)
at ~30 fps during playback. The same brightness value is offered to the servo
layer for throttled speech-reactive head/arm motion. Hardware calls are no-ops
when the corresponding device is disabled.
"""

import hashlib
import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

import config
import state as state_module
from audio import echo_cancel
from audio import output_gate
from hardware import leds_head, leds_chest, servos
from intelligence import emotion_orchestrator
from sequences import animations
from state import State
from utils import conv_log

logger = logging.getLogger(__name__)

_speaking = False
_speaking_lock = threading.Lock()


def _normalize_for_speech(text: str) -> str:
    """Expand compact forms that ElevenLabs tends to pronounce badly."""
    spoken = " ".join((text or "").split())
    replacements = [
        (r"\bWWII\b", "World War Two"),
        (r"\bWW2\b", "World War Two"),
        (r"\bWorld War II\b", "World War Two"),
        (r"\bWWI\b", "World War One"),
        (r"\bWW1\b", "World War One"),
        (r"\bWorld War I\b", "World War One"),
    ]
    for pattern, replacement in replacements:
        spoken = re.sub(pattern, replacement, spoken, flags=re.IGNORECASE)
    return spoken


# ── Public API ────────────────────────────────────────────────────────────────

def is_speaking() -> bool:
    """Return True while audio is actively playing."""
    with _speaking_lock:
        return _speaking


def _is_shutdown_state() -> bool:
    try:
        return state_module.is_state(State.SHUTDOWN)
    except Exception:
        return False


def prewarm() -> None:
    """Play 100ms of silence to force audio device initialization before first TTS.

    Holds the output gate so prewarm waits for any startup clip to finish
    before opening the output device — back-to-back sd.play() calls during
    device init can cause sd.wait() to return early on the first real TTS call.
    """
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        logger.info("[tts] audio output prewarm skipped — audio suppressed")
        return
    with output_gate.hold("tts-prewarm") as acquired:
        if not acquired:
            return
        try:
            import sounddevice as sd
            silence = np.zeros(int(44100 * 0.1), dtype=np.float32)
            sd.play(silence, samplerate=44100, blocksize=2048)
            sd.wait()
            logger.info("[tts] audio output device pre-warmed")
        except Exception as exc:
            logger.warning("[tts] prewarm failed (non-fatal): %s", exc)


def _resolve_voice_settings(
    emotion: str, override: Optional[dict]
) -> Optional[dict]:
    """Resolve the ElevenLabs voice_settings for a line.

    An explicit override (empathy/grief delivery shaping) always wins. Otherwise
    the settings are derived from the emotion frame's voice_style so normal lines
    carry Rex's expressive baseline instead of the voice clone's flat defaults.
    """
    if override:
        return override
    try:
        return emotion_orchestrator.voice_settings_for_emotion(emotion)
    except Exception as exc:
        logger.debug("[tts] voice settings resolution failed: %s", exc)
        return None


# ── Eleven v3 audio tags ─────────────────────────────────────────────────────
# Tags shape delivery at synthesis; they must NEVER reach the transcript / log / memory / GUI.
_AUDIO_TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z '\-]*)\]")


def strip_audio_tags(text: Optional[str]) -> str:
    """Remove [audio tags] from text — use anywhere Rex's line is stored or displayed, so a v3
    delivery tag never leaks into the transcript/log/memory."""
    if not text:
        return text or ""
    return re.sub(r"\s{2,}", " ", _AUDIO_TAG_RE.sub("", text)).strip()


def _v3_tags_active() -> bool:
    return (
        str(getattr(config, "TTS_MODEL_ID", "")).strip() == "eleven_v3"
        and bool(getattr(config, "TTS_V3_AUDIO_TAGS_ENABLED", False))
    )


def resolve_audio_tag(emotion: Optional[str] = None, comedy_mode: Optional[str] = None) -> Optional[str]:
    """Deterministic affect -> v3 tag. comedy_mode (Rex's comedic STANCE — where sarcasm/mischief
    live) wins; else the reply emotion. Returns a whitelisted bare tag word, or None for
    neutral/sincere/unknown affect (never tag a serious moment)."""
    by_mode = getattr(config, "TTS_V3_TAG_BY_COMEDY_MODE", {}) or {}
    by_emo = getattr(config, "TTS_V3_TAG_BY_EMOTION", {}) or {}
    tag = by_mode.get((comedy_mode or "").strip().lower()) or by_emo.get((emotion or "").strip().lower())
    whitelist = getattr(config, "TTS_V3_TAG_WHITELIST", set()) or set()
    return tag if (tag and tag in whitelist) else None


def _apply_audio_tags(
    spoken_text: str,
    emotion: Optional[str],
    comedy_mode: Optional[str],
    voice_settings: Optional[dict],
) -> Tuple[str, Optional[dict]]:
    """Return (text-for-ElevenLabs, voice_settings) with a v3 audio tag applied. Used by BOTH speak
    and ensure_cached so their cache keys match. No-op unless v3 tags are active. Keeps only
    whitelisted inline tags (model-emitted, a later phase), else prepends the affect-mapped leading
    tag, and PINS stability to Creative on any tagged line (high stability mutes tags per the docs)."""
    if not _v3_tags_active():
        return spoken_text, voice_settings
    whitelist = getattr(config, "TTS_V3_TAG_WHITELIST", set()) or set()
    text = _AUDIO_TAG_RE.sub(
        lambda m: m.group(0) if m.group(1).strip().lower() in whitelist else "", spoken_text
    )
    has_tag = bool(_AUDIO_TAG_RE.search(text))
    if not has_tag:
        tag = resolve_audio_tag(emotion, comedy_mode)
        if tag:
            text = f"[{tag}] {text.lstrip()}"
            has_tag = True
    if has_tag:
        stability = float(getattr(config, "TTS_V3_TAG_STABILITY", 0.35))
        voice_settings = {**(voice_settings or {}), "stability": stability}
    return re.sub(r"\s{2,}", " ", text).strip(), voice_settings


def speak(
    text: str,
    emotion: str = "neutral",
    voice_settings: Optional[dict] = None,
    on_playback_start: Optional[Callable[[], None]] = None,
    post_playback_tail_secs: Optional[float] = None,
    flush_on_playback_stop: Optional[bool] = None,
    log_text: bool = True,
    comedy_mode: Optional[str] = None,
    suppress_audio_tag: bool = False,
) -> None:
    """Convert text to speech and play it, blocking until playback finishes.

    On cache hit: plays the cached MP3 with no API call.
    On cache miss: calls ElevenLabs streaming API, saves to cache, then plays.

    `voice_settings` explicitly overrides ElevenLabs voice parameters (stability,
    style, similarity_boost, use_speaker_boost, speed) — used by the empathy/grief
    delivery layer. When omitted, expressive settings are derived from `emotion`
    (see config.TTS_VOICE_SETTINGS_*), so normal lines are no longer rendered with
    the clone's flat defaults. The resolved settings are folded into the cache key,
    so each (text, emotion) take caches separately.
    """
    if not text or not text.strip():
        return
    spoken_text = _normalize_for_speech(text)
    print(f"[TTS] {spoken_text}", flush=True)
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        if log_text:
            try:
                conv_log.log_rex(spoken_text)
            except Exception as exc:
                logger.debug("[tts] conversation log write failed: %s", exc)
        if on_playback_start is not None:
            try:
                on_playback_start()
            except Exception:
                pass
        logger.info("[tts] audio suppressed — emitted text only")
        return

    # Pre-turn gaze aversion: arm the "look away to think" beat (longer + glance-up
    # for a complex reply, short to-the-side for a simple one) just before the audio
    # is fetched/played, so the head breaks contact while "thinking" and returns as
    # Rex starts to speak. No-op when the gaze feature is off or no head is attached.
    try:
        from intelligence import gaze_engine
        gaze_engine.note_about_to_speak(spoken_text)
    except Exception:
        pass

    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    # synth_text may carry a leading v3 audio tag ([sarcastic] …); spoken_text stays CLEAN for the
    # conversation log below, so tags reach ElevenLabs only, never the transcript. suppress_audio_tag
    # is set for the 2nd+ sentences of a streamed reply so the leading tag lands once, not per sentence.
    if suppress_audio_tag:
        synth_text = spoken_text
    else:
        synth_text, voice_settings = _apply_audio_tags(spoken_text, emotion, comedy_mode, voice_settings)
    cache_file = _cache_path(synth_text, voice_id, model_id, voice_settings)

    if cache_file.exists():
        logger.info("[tts] cache hit: %s", cache_file.name)
    else:
        logger.info(
            "[tts] cache miss — calling ElevenLabs API%s",
            f" (voice_settings={_summarize_settings(voice_settings)})"
            if voice_settings else "",
        )
        audio_bytes = _fetch_from_api(synth_text, voice_id, model_id, voice_settings)
        if not audio_bytes:
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(audio_bytes)
        logger.info("[tts] saved to cache: %s", cache_file.name)

    audio, samplerate = _read_audio(cache_file)
    if audio is None or len(audio) == 0:
        logger.error("[tts] audio decode produced empty array — skipping playback")
        return
    audio = _trim_trailing_silence(audio, samplerate)

    if log_text:
        try:
            conv_log.log_rex(spoken_text)
        except Exception as exc:
            logger.debug("[tts] conversation log write failed: %s", exc)

    _play(
        audio,
        samplerate,
        emotion,
        on_playback_start=on_playback_start,
        post_playback_tail_secs=post_playback_tail_secs,
        flush_on_playback_stop=flush_on_playback_stop,
    )


def _trim_trailing_silence(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """Remove TTS tail padding that keeps AEC active after Rex sounds done."""
    if not bool(getattr(config, "TTS_TRIM_TRAILING_SILENCE_ENABLED", True)):
        return audio
    if audio is None or audio.size == 0 or samplerate <= 0:
        return audio

    threshold = max(
        0.0,
        float(getattr(config, "TTS_TRIM_TRAILING_SILENCE_THRESHOLD", 0.003) or 0.0),
    )
    if threshold <= 0.0:
        return audio

    window_ms = max(
        1.0,
        float(getattr(config, "TTS_TRIM_TRAILING_SILENCE_WINDOW_MS", 20) or 20),
    )
    padding_ms = max(
        0.0,
        float(getattr(config, "TTS_TRIM_TRAILING_SILENCE_PADDING_MS", 40) or 0),
    )
    window = max(1, int(samplerate * window_ms / 1000.0))
    padding = max(0, int(samplerate * padding_ms / 1000.0))

    mono = audio
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    last_voice_sample = -1
    for end in range(mono.size, 0, -window):
        start = max(0, end - window)
        chunk = mono[start:end]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        peak = float(np.max(np.abs(chunk)))
        if rms >= threshold or peak >= threshold * 2.0:
            last_voice_sample = end
            break

    if last_voice_sample < 0:
        return audio

    trim_at = min(audio.shape[0], last_voice_sample + padding)
    # Keep tiny trims alone; they are not worth perturbing playback timing.
    if audio.shape[0] - trim_at < int(samplerate * 0.08):
        return audio

    trimmed = audio[:trim_at].copy()
    removed_ms = (audio.shape[0] - trim_at) * 1000.0 / float(samplerate)
    logger.info("[tts] trimmed %.0fms trailing silence", removed_ms)
    return trimmed


# ── Internal: playback ────────────────────────────────────────────────────────

def _play(
    audio: np.ndarray,
    samplerate: int,
    emotion: str,
    *,
    on_playback_start: Optional[Callable[[], None]] = None,
    post_playback_tail_secs: Optional[float] = None,
    flush_on_playback_stop: Optional[bool] = None,
) -> None:
    global _speaking

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("[tts] sounddevice not installed — cannot play audio")
        return

    with output_gate.hold("tts") as acquired:
        if not acquired:
            logger.debug("[tts] playback skipped — output gate busy")
            return

        with _speaking_lock:
            _speaking = True

        stop_event = threading.Event()
        led_thread = threading.Thread(
            target=_drive_leds,
            args=(audio, samplerate, stop_event),
            daemon=True,
            name="tts-leds",
        )

        # Hold AEC suppression for at least the audio's actual duration. A
        # CoreAudio glitch can cause sd.wait() to return early while audio is
        # still buffered for playback — without this guard, set_playing(False)
        # fires immediately, the mic unmutes, and Rex hears himself and triggers
        # an interrupt-ack ("what?") mid-sentence.
        expected_duration = len(audio) / float(samplerate)
        play_started_at = time.monotonic()
        emotion_frame = emotion_orchestrator.frame_for_speech(emotion)
        led_emotion = emotion_frame.led_style
        emotion_orchestrator.publish_frame(
            emotion_frame,
            ttl_secs=max(2.0, expected_duration + 2.0),
        )

        try:
            try:
                animations.speech_activity_start()
                servos.begin_speech_motion(emotion_frame)
            except Exception as exc:
                logger.debug("[tts] speech servo start failed: %s", exc)
            leds_head.speak(led_emotion)
            # Re-assert the eyes ON at the emotion colour every turn. The mouth
            # SPEAK command never touches the eyes, and the serial link is lossy
            # during speech, so without this the eyes ride entirely on a single
            # easily-dropped post-speech re-arm. The heartbeat keeps them lit
            # between turns; this guarantees they are lit for the turn itself.
            leds_head.ensure_eyes_on(led_emotion)
            leds_chest.speak(led_emotion)
            echo_cancel.set_playing(True)
            led_thread.start()
            if on_playback_start is not None:
                try:
                    on_playback_start()
                except Exception:
                    pass
            sd.play(audio, samplerate, blocksize=2048)
            sd.wait()
        except Exception as exc:
            logger.error("[tts] playback error: %s", exc)
        finally:
            elapsed = time.monotonic() - play_started_at
            remaining = expected_duration - elapsed
            if remaining > 0.05 and not echo_cancel.was_canceled():
                logger.warning(
                    "[tts] sd.wait() returned %.2fs early (likely CoreAudio glitch) — "
                    "holding suppression for the remaining %.2fs",
                    remaining, remaining,
                )
                time.sleep(remaining)
            stop_event.set()
            if led_thread.is_alive():
                led_thread.join(timeout=1.0)
            shutdown_now = _is_shutdown_state()
            try:
                if shutdown_now:
                    leds_head.off()
                else:
                    leds_head.speak_stop()
            except Exception as exc:
                logger.warning("[tts] head LED cleanup failed: %s", exc)
            try:
                if shutdown_now:
                    leds_chest.off()
                else:
                    leds_chest.active()
            except Exception as exc:
                logger.debug("[tts] chest LED cleanup failed: %s", exc)
            try:
                servos.end_speech_motion()
            except Exception as exc:
                logger.debug("[tts] speech servo stop failed: %s", exc)
            try:
                animations.speech_activity_stop()
            except Exception as exc:
                logger.debug("[tts] speech activity clear failed: %s", exc)
            echo_cancel.set_playing(
                False,
                tail_secs=post_playback_tail_secs,
                flush=flush_on_playback_stop,
            )
            with _speaking_lock:
                _speaking = False


def _drive_leds(
    audio: np.ndarray, samplerate: int, stop_event: threading.Event
) -> None:
    """Iterate audio in fixed chunks, driving mouth LED brightness from RMS."""
    interval = config.TTS_LED_UPDATE_INTERVAL_SECS
    chunk_len = max(1, int(samplerate * interval))
    min_delta = int(getattr(config, "HEAD_LED_SPEAK_LEVEL_MIN_DELTA", 8))
    last_sent = -1

    for i in range(0, len(audio), chunk_len):
        if stop_event.is_set():
            break
        chunk = audio[i : i + chunk_len]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        brightness = min(255, int(rms * config.TTS_LED_BRIGHTNESS_SCALE))
        # Only push a new mouth level when it changes meaningfully (or hits 0):
        # the per-frame flood overlaps the Arduino's interrupt-off show() windows
        # and is the main source of dropped commands. The mouth keeps animating
        # off the last level, so fewer writes don't freeze it.
        if (
            last_sent < 0
            or abs(brightness - last_sent) >= min_delta
            or (brightness == 0 and last_sent != 0)
        ):
            leds_head.speak_level(brightness)
            last_sent = brightness
        servos.speech_reactive_move(brightness / 255.0)
        stop_event.wait(timeout=interval)


# ── Internal: cache & decode ──────────────────────────────────────────────────

def _settings_cache_token(voice_settings: Optional[dict]) -> str:
    """Stable token to fold into the cache hash. Empty when no override —
    preserves the existing cache for default-mode lines.
    """
    if not voice_settings:
        return ""
    keys = ("stability", "similarity_boost", "style", "use_speaker_boost", "speed")
    parts = []
    for k in keys:
        if k in voice_settings and voice_settings[k] is not None:
            parts.append(f"{k}={voice_settings[k]}")
    return "|".join(parts)


def _summarize_settings(voice_settings: Optional[dict]) -> str:
    if not voice_settings:
        return "default"
    return ", ".join(
        f"{k}={v}" for k, v in voice_settings.items() if v is not None
    )


def _cache_path(
    text: str,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[dict] = None,
) -> Path:
    settings_token = _settings_cache_token(voice_settings)
    digest = hashlib.sha256(
        f"{text}{voice_id}{model_id}{settings_token}".encode("utf-8")
    ).hexdigest()
    return Path(config.TTS_CACHE_DIR) / f"{digest}.mp3"


def is_cached(
    text: str,
    voice_settings: Optional[dict] = None,
    emotion: str = "neutral",
) -> bool:
    """Return True if this text already has cached audio for the active voice.

    `emotion` must match what the line will be spoken with so the cache key
    lines up with the expressive voice settings used at playback time.
    """
    if not text or not text.strip():
        return False
    spoken_text = _normalize_for_speech(text)
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    return _cache_path(
        spoken_text,
        config.ELEVENLABS_VOICE_ID,
        config.TTS_MODEL_ID,
        voice_settings,
    ).exists()


def ensure_cached(
    text: str,
    voice_settings: Optional[dict] = None,
    emotion: str = "neutral",
    comedy_mode: Optional[str] = None,
    suppress_audio_tag: bool = False,
) -> bool:
    """Ensure text has a cached TTS file without playing it.

    `emotion` (and `comedy_mode`) must match what the line will be spoken with so the prefilled file
    lands under the same cache key the live turn looks up — including any v3 audio tag + its pinned
    stability, applied identically here and in speak().
    """
    if not text or not text.strip():
        return False
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        logger.info("[tts] cache prefill skipped — audio suppressed")
        return False
    spoken_text = _normalize_for_speech(text)
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    if suppress_audio_tag:
        synth_text = spoken_text
    else:
        synth_text, voice_settings = _apply_audio_tags(spoken_text, emotion, comedy_mode, voice_settings)
    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    cache_file = _cache_path(synth_text, voice_id, model_id, voice_settings)
    if cache_file.exists():
        return True

    logger.info("[tts] cache prefill miss — calling ElevenLabs API for %r", synth_text)
    audio_bytes = _fetch_from_api(synth_text, voice_id, model_id, voice_settings)
    if not audio_bytes:
        return False
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(audio_bytes)
    logger.info("[tts] prefilled cache: %s", cache_file.name)
    return True


def _fetch_from_api(
    text: str,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[dict] = None,
) -> Optional[bytes]:
    try:
        import apikeys
        from elevenlabs import ElevenLabs, VoiceSettings

        client = ElevenLabs(api_key=apikeys.ELEVENLABS_API_KEY)
        kwargs = {
            "voice_id": voice_id,
            "text": text,
            "model_id": model_id,
        }
        if voice_settings:
            kwargs["voice_settings"] = VoiceSettings(
                **{k: v for k, v in voice_settings.items() if v is not None}
            )
        chunks = client.text_to_speech.stream(**kwargs)
        data = b"".join(chunks)
        if not data:
            logger.error("[tts] ElevenLabs returned empty audio stream")
            return None
        return data
    except Exception as exc:
        logger.error("[tts] ElevenLabs API error: %s", exc)
        return None


def _read_audio(path: Path) -> Tuple[Optional[np.ndarray], int]:
    try:
        import soundfile as sf

        audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), samplerate
    except Exception as exc:
        logger.error("[tts] failed to decode %s: %s", path.name, exc)
        return None, 0
