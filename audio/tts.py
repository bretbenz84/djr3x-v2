"""
ElevenLabs TTS with SHA-256 file cache.

speak() is synchronous and blocks until audio has finished playing.
Callers are responsible for running it in a thread when non-blocking behaviour
is needed.

Cache strategy
──────────────
Before every API call, a SHA-256 of (text + voice_id + model_id) is computed and
checked against assets/audio/tts_cache/{hash}.mp3. On a hit the file is played
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
from typing import Optional, Tuple

import numpy as np

import config
from audio import echo_cancel
from audio import output_gate
from hardware import leds_head, leds_chest, servos
from sequences import animations

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


def prewarm() -> None:
    """Play 100ms of silence to force audio device initialization before first TTS.

    Holds the output gate so prewarm waits for any startup clip to finish
    before opening the output device — back-to-back sd.play() calls during
    device init can cause sd.wait() to return early on the first real TTS call.
    """
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


def speak(
    text: str,
    emotion: str = "neutral",
    voice_settings: Optional[dict] = None,
) -> None:
    """Convert text to speech and play it, blocking until playback finishes.

    On cache hit: plays the cached MP3 with no API call.
    On cache miss: calls ElevenLabs streaming API, saves to cache, then plays.

    `voice_settings` overrides ElevenLabs voice parameters (stability, style,
    similarity_boost, use_speaker_boost). When provided AND non-empty, it is
    folded into the cache key so the alternate take is cached separately —
    the default cache (voice_settings=None) is unaffected, so existing cached
    lines stay valid for normal-mode delivery.
    """
    if not text or not text.strip():
        return
    spoken_text = _normalize_for_speech(text)
    print(f"[TTS] {spoken_text}", flush=True)

    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    cache_file = _cache_path(spoken_text, voice_id, model_id, voice_settings)

    if cache_file.exists():
        logger.info("[tts] cache hit: %s", cache_file.name)
    else:
        logger.info(
            "[tts] cache miss — calling ElevenLabs API%s",
            f" (voice_settings={_summarize_settings(voice_settings)})"
            if voice_settings else "",
        )
        audio_bytes = _fetch_from_api(spoken_text, voice_id, model_id, voice_settings)
        if not audio_bytes:
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(audio_bytes)
        logger.info("[tts] saved to cache: %s", cache_file.name)

    audio, samplerate = _read_audio(cache_file)
    if audio is None or len(audio) == 0:
        logger.error("[tts] audio decode produced empty array — skipping playback")
        return

    _play(audio, samplerate, emotion)


# ── Internal: playback ────────────────────────────────────────────────────────

def _play(audio: np.ndarray, samplerate: int, emotion: str) -> None:
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

        try:
            try:
                animations.speech_activity_start()
                servos.begin_speech_motion(emotion)
            except Exception as exc:
                logger.debug("[tts] speech servo start failed: %s", exc)
            leds_head.speak(emotion)
            leds_chest.speak(emotion)
            echo_cancel.set_playing(True)
            led_thread.start()
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
            leds_head.speak_stop()
            leds_chest.active()
            try:
                servos.end_speech_motion()
            except Exception as exc:
                logger.debug("[tts] speech servo stop failed: %s", exc)
            try:
                animations.speech_activity_stop()
            except Exception as exc:
                logger.debug("[tts] speech activity clear failed: %s", exc)
            echo_cancel.set_playing(False)
            with _speaking_lock:
                _speaking = False


def _drive_leds(
    audio: np.ndarray, samplerate: int, stop_event: threading.Event
) -> None:
    """Iterate audio in fixed chunks, driving mouth LED brightness from RMS."""
    interval = config.TTS_LED_UPDATE_INTERVAL_SECS
    chunk_len = max(1, int(samplerate * interval))

    for i in range(0, len(audio), chunk_len):
        if stop_event.is_set():
            break
        chunk = audio[i : i + chunk_len]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        brightness = min(255, int(rms * config.TTS_LED_BRIGHTNESS_SCALE))
        leds_head.speak_level(brightness)
        servos.speech_reactive_move(brightness / 255.0)
        stop_event.wait(timeout=interval)


# ── Internal: cache & decode ──────────────────────────────────────────────────

def _settings_cache_token(voice_settings: Optional[dict]) -> str:
    """Stable token to fold into the cache hash. Empty when no override —
    preserves the existing cache for default-mode lines.
    """
    if not voice_settings:
        return ""
    keys = ("stability", "similarity_boost", "style", "use_speaker_boost")
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
) -> bool:
    """Return True if this text already has cached audio for the active voice."""
    if not text or not text.strip():
        return False
    spoken_text = _normalize_for_speech(text)
    return _cache_path(
        spoken_text,
        config.ELEVENLABS_VOICE_ID,
        config.TTS_MODEL_ID,
        voice_settings,
    ).exists()


def ensure_cached(
    text: str,
    voice_settings: Optional[dict] = None,
) -> bool:
    """Ensure text has a cached TTS file without playing it."""
    if not text or not text.strip():
        return False
    spoken_text = _normalize_for_speech(text)
    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    cache_file = _cache_path(spoken_text, voice_id, model_id, voice_settings)
    if cache_file.exists():
        return True

    logger.info("[tts] cache prefill miss — calling ElevenLabs API for %r", spoken_text)
    audio_bytes = _fetch_from_api(spoken_text, voice_id, model_id, voice_settings)
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
