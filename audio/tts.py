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
The audio is walked in TTS_LED_UPDATE_INTERVAL_SECS chunks, each chunk's RMS
becomes a brightness, and leds_head.speak_level(brightness) drives the mouth at
~30 fps. The same brightness value is offered to the servo layer for throttled
speech-reactive head/arm motion. Hardware calls are no-ops when the corresponding
device is disabled.

Those levels are clocked off the AUDIBLE timeline, not the write one: the device
holds a full output buffer of not-yet-heard audio (0.95 s normally on this Mac,
3.18 s while the clone deep buffer is armed — i.e. every impersonation), and
driving the mouth at write time put the whole animation that far ahead of the
voice. _MouthPacer holds each level, and the mouth-on trigger itself, until its
audio actually reaches the room; the buffered _play() path gets the same offset
via _drive_leds(start_delay=...). Kill switch: TTS_MOUTH_SYNC_ENABLED.
"""

import hashlib
import io
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

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

def spoken_form(text: str) -> str:
    """The exact string the local engine will synthesize for ``text`` —
    normalized and audio-tag-free (Qwen would read [tags] aloud).

    Callers that pre-start a take (features/impersonation.py) must key on THIS,
    not on the raw text, or the player looks up a key that was never parked and
    silently re-renders from scratch.
    """
    return strip_audio_tags(_normalize_for_speech(text))


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
            sd.play(silence, samplerate=44100, **playback_stream_kwargs())
            sd.wait()
            logger.info("[tts] audio output device pre-warmed")
        except Exception as exc:
            logger.warning("[tts] prewarm failed (non-fatal): %s", exc)


_boot_deep_buffer_active = False


def set_boot_deep_buffer(active: bool) -> None:
    """Arm/disarm the boot-window deep playback buffer. While armed, every
    playback stream asks PortAudio for an EXPLICIT multi-hundred-ms host buffer
    instead of the symbolic 'high' preset — on macOS CoreAudio 'high' is only a
    few tens of ms, which the model-preload GIL bursts blow straight through
    (field 2026-08-02: startup filler line stuttering again despite blocksize
    4096 + 'high'). Disarmed after the preloads so conversational replies keep
    their low time-to-first-sound."""
    global _boot_deep_buffer_active
    _boot_deep_buffer_active = bool(active)
    logger.info("[tts] boot deep-buffer playback %s",
                "armed" if active else "disarmed")


def playback_stream_kwargs() -> dict:
    """Deep-buffer playback params (audio QoS): a bigger blocksize + high host-buffer
    latency lets playback survive GIL stalls from heavy work elsewhere (model preloads
    at boot) without the mid-sentence stutter. Shared by every sd.play() call site.
    AUDIO_PLAYBACK_LATENCY may be 'low'/'high' or a numeric string of seconds."""
    latency: "str | float" = str(
        getattr(config, "AUDIO_PLAYBACK_LATENCY", "high") or "high"
    )
    try:
        latency = float(latency)
    except ValueError:
        pass
    blocksize = int(getattr(config, "AUDIO_PLAYBACK_BLOCKSIZE", 4096))
    if _boot_deep_buffer_active:
        latency = float(getattr(config, "AUDIO_PLAYBACK_BOOT_LATENCY_SECS", 1.0))
        blocksize = max(
            blocksize, int(getattr(config, "AUDIO_PLAYBACK_BOOT_BLOCKSIZE", 8192))
        )
    elif _clone_deep_buffer_needed():
        # Same failure as the boot window, mid-session: the local clone engine
        # loading/warming/rendering (an unprompted impression behind an ordinary
        # reply) is exactly the Metal+GIL burst that blows through the symbolic
        # 'high' host buffer — field 2026-08-19: the ElevenLabs reply stuttered
        # for the full 16.7s Jimmy Carter render. Costs ~a second of extra
        # time-to-first-sound only on streams opened during that window.
        latency = float(getattr(config, "AUDIO_PLAYBACK_CLONE_LATENCY_SECS", 1.2))
        blocksize = max(
            blocksize, int(getattr(config, "AUDIO_PLAYBACK_CLONE_BLOCKSIZE", 8192))
        )
    return {"blocksize": blocksize, "latency": latency}


def _clone_deep_buffer_needed() -> bool:
    """True when local clone work is running or imminent, so a playback stream
    opened NOW should carry an explicit deep host buffer. Two signals: the engine
    is actually busy (load/warmup/generation), or an unprompted impression is
    pending (its script call is about to hand the engine a take — the reply that
    covers the render usually opens its stream inside that gap)."""
    if not bool(getattr(config, "AUDIO_PLAYBACK_CLONE_DEEP_BUFFER_ENABLED", True)):
        return False
    try:
        from audio import local_tts
        if local_tts.engine_busy():
            return True
    except Exception:
        pass
    try:
        from features import organic_impersonation
        return organic_impersonation.has_pending()
    except Exception:
        return False


def _resolve_voice_settings(
    emotion: str, override: Optional[dict]
) -> Optional[dict]:
    """Resolve the ElevenLabs voice_settings for a line.

    An explicit override (empathy/grief delivery shaping) always wins. Otherwise
    the settings are derived from the emotion frame's voice_style so normal lines
    carry Rex's expressive baseline instead of the voice clone's flat defaults.

    On eleven_v3, stability is finally pinned to one fixed preset (see
    _pin_v3_stability): the per-emotion/per-comedy stability deltas were tuned for
    v2's continuous knob and make v3 sound like a different voice each sentence.
    """
    if override:
        return _pin_v3_stability(override)
    try:
        return _pin_v3_stability(emotion_orchestrator.voice_settings_for_emotion(emotion))
    except Exception as exc:
        logger.debug("[tts] voice settings resolution failed: %s", exc)
        return _pin_v3_stability(None)   # still pin v3 stability even if emotion resolution fails


def _pin_v3_stability(voice_settings: Optional[dict]) -> Optional[dict]:
    """On eleven_v3, force `stability` to the single configured preset (config.TTS_V3_STABILITY)
    so Rex's voice is consistent line to line. No-op on other models, or when TTS_V3_STABILITY is
    None. This is the ONE choke point every synthesis path passes through."""
    if str(getattr(config, "TTS_MODEL_ID", "")).strip() != "eleven_v3":
        return voice_settings
    fixed = getattr(config, "TTS_V3_STABILITY", None)
    if fixed is None:
        return voice_settings
    return {**(voice_settings or {}), "stability": float(fixed)}


# ── Eleven v3 audio tags ─────────────────────────────────────────────────────
# Tags shape delivery at synthesis; they must NEVER reach the transcript / log / memory / GUI.
# Pattern + strip live in utils.audio_tags (shared with conv_log); re-exported here as the
# established public API.
from utils.audio_tags import AUDIO_TAG_RE as _AUDIO_TAG_RE, strip_audio_tags  # noqa: E402


def _v3_tags_active() -> bool:
    return (
        str(getattr(config, "TTS_MODEL_ID", "")).strip() == "eleven_v3"
        and bool(getattr(config, "TTS_V3_AUDIO_TAGS_ENABLED", False))
    )


def llm_inline_tag_rule() -> str:
    """The compact system-prompt rule that lets the reply LLM place inline v3 delivery tags
    mid-reply, or "" when tags can't land (non-v3 model, kill switch off, or
    TTS_V3_LLM_INLINE_TAGS_ENABLED off). Lives here so the offered palette stays in lockstep
    with the synthesis whitelist — a tag the prompt suggests is always one synthesis keeps.
    Vocalization-only tags the model places poorly (snorts/exhales) are not offered."""
    if not _v3_tags_active():
        return ""
    if not bool(getattr(config, "TTS_V3_LLM_INLINE_TAGS_ENABLED", False)):
        return ""
    whitelist = getattr(config, "TTS_V3_TAG_WHITELIST", set()) or set()
    offered = [
        t for t in ("excited", "curious", "sarcastic", "mischievously",
                    "laughs", "sighs", "whispers")
        if t in whitelist
    ]
    if not offered:
        return ""
    palette = " ".join(f"[{t}]" for t in offered)
    return (
        "Voice delivery tags: you may place AT MOST one bracketed tag inside a reply, "
        "immediately before the words whose delivery genuinely shifts — exactly one of: "
        f"{palette}. Use one only when the beat clearly calls for it (a tease, a reveal, "
        "mock-drama, a weary sigh, a shift to hushed conspiracy); most replies need none. "
        "Never tag a sincere or serious moment, and never invent tags outside that list."
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


def _sanitize_inline_tags(text: str) -> str:
    """Make inline [audio tags] in caller-supplied text safe for synthesis. Inline tags may now
    arrive legitimately mid-sentence — authored on canned seam lines (repair_moves recovery tags)
    or emitted by the lean brain. When v3 tags are ACTIVE: keep whitelisted tags, drop the rest,
    and cap how many survive (TTS_V3_INLINE_TAG_CAP, earliest win) so an over-eager LLM can't
    turn a reply into a laugh track. When INACTIVE (non-v3 model, or the kill switch is off):
    strip ALL tags — v2/turbo would read the brackets aloud, and the kill switch must actually
    kill delivery tags, not let inline ones ride."""
    if not _AUDIO_TAG_RE.search(text):
        return text
    if not _v3_tags_active():
        return strip_audio_tags(text)
    whitelist = getattr(config, "TTS_V3_TAG_WHITELIST", set()) or set()
    cap = int(getattr(config, "TTS_V3_INLINE_TAG_CAP", 2) or 0)
    kept = {"n": 0}

    def _keep(m: re.Match) -> str:
        if m.group(1).strip().lower() not in whitelist:
            return ""
        if cap > 0 and kept["n"] >= cap:
            return ""
        kept["n"] += 1
        return m.group(0)

    return re.sub(r"\s{2,}", " ", _AUDIO_TAG_RE.sub(_keep, text)).strip()


def _apply_audio_tags(
    spoken_text: str,
    emotion: Optional[str],
    comedy_mode: Optional[str],
    voice_settings: Optional[dict],
    suppress_leading: bool = False,
) -> Tuple[str, Optional[dict]]:
    """Return (text-for-ElevenLabs, voice_settings) with v3 audio tags applied. Used by BOTH speak
    and ensure_cached so their cache keys match. Inline tags are ALWAYS sanitized (whitelisted
    survive on v3, everything is stripped otherwise — see _sanitize_inline_tags); the affect-mapped
    LEADING tag is then prepended unless `suppress_leading` (2nd+ chunks of a streamed reply — the
    reply's one leading tag rode chunk 1) or an inline tag already carries the delivery. Stability
    is NOT touched here — it is pinned globally by _pin_v3_stability to the Natural preset, which
    still lets tags land (only HIGH/Robust stability mutes them)."""
    text = _sanitize_inline_tags(spoken_text)
    if not _v3_tags_active():
        return text, voice_settings
    if not suppress_leading and not _AUDIO_TAG_RE.search(text):
        tag = resolve_audio_tag(emotion, comedy_mode)
        if tag:
            text = f"[{tag}] {text.lstrip()}"
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
    previous_text: Optional[str] = None,
    voice_ref: Optional[object] = None,
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

    `voice_ref` (a local_tts.VoiceRef) forces on-device synthesis in THAT voice —
    used by the impersonation feature to clone an arbitrary person. ElevenLabs
    cannot do this, so a voice_ref always routes to the local engine (tags
    stripped, no caching). Independent of `voice_ref`, the local engine also
    renders Rex's own voice when --local-tts mode is on or the ElevenLabs breaker
    is open.
    """
    if not text or not text.strip():
        return
    spoken_text = _normalize_for_speech(text)
    # Callers may pass text carrying inline [audio tags] (authored seam lines, LLM-emitted);
    # clean_text is what the transcript/GUI get — tags reach ElevenLabs only.
    clean_text = strip_audio_tags(spoken_text)
    print(f"[TTS] {spoken_text}", flush=True)
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        if log_text:
            try:
                conv_log.log_rex(clean_text)
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

    # ── Backend dispatch ─────────────────────────────────────────────────────
    # An explicit voice_ref (impersonation) always routes local. Otherwise Rex's
    # own voice goes local when --local-tts mode is on or the ElevenLabs breaker
    # is open. Local synthesizes clean_text (Qwen would read [audio tags] aloud).
    local_ref = voice_ref if voice_ref is not None else (
        _rex_local_ref() if _use_local_backend() else None
    )
    if local_ref is not None:
        try:
            if _speak_local(
                clean_text, local_ref, emotion,
                on_playback_start=on_playback_start,
                post_playback_tail_secs=post_playback_tail_secs,
                flush_on_playback_stop=flush_on_playback_stop,
                log_text=log_text,
            ):
                return
        except Exception as exc:
            logger.warning("[tts] local backend error (%s)", exc)
        if voice_ref is not None:
            # Explicit impersonation voice failed — do NOT speak the parody in
            # Rex's ElevenLabs voice; the caller covers the miss.
            logger.warning("[tts] impersonation voice synth failed — skipping line")
            return
        # Rex's own voice failed locally → fall through to ElevenLabs (best effort).

    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    # synth_text may carry v3 audio tags (a prepended leading tag and/or inline mid-sentence ones);
    # clean_text is what the conversation log below gets, so tags reach ElevenLabs only, never the
    # transcript. suppress_audio_tag is set for the 2nd+ chunks of a streamed reply so the leading
    # tag lands once, not per sentence — inline tags in those chunks are still sanitized and kept.
    synth_text, voice_settings = _apply_audio_tags(
        spoken_text, emotion, comedy_mode, voice_settings, suppress_leading=suppress_audio_tag
    )
    cache_file = _cache_path(synth_text, voice_id, model_id, voice_settings, previous_text)
    # Streamed takes cache as WAV next to the buffered path's MP3 — honor both.
    wav_sibling = cache_file.with_suffix(".wav")

    if cache_file.exists():
        logger.info("[tts] cache hit: %s", cache_file.name)
    elif wav_sibling.exists():
        logger.info("[tts] cache hit: %s", wav_sibling.name)
        cache_file = wav_sibling
    else:
        if bool(getattr(config, "TTS_STREAMING_PLAYBACK_ENABLED", True)):
            handled = False
            try:
                handled = _speak_streaming(
                    synth_text, clean_text, voice_id, model_id, voice_settings,
                    previous_text, emotion, cache_file,
                    on_playback_start=on_playback_start,
                    post_playback_tail_secs=post_playback_tail_secs,
                    flush_on_playback_stop=flush_on_playback_stop,
                    log_text=log_text,
                )
            except Exception as exc:
                logger.warning("[tts] streaming path error (%s) — buffered fallback", exc)
            if handled:
                return
            # The streaming attempt may have just opened the breaker on a
            # network-level failure. Do NOT re-dial the same dead endpoint on the
            # buffered path — that is what turned one 25 s stall into 51.7 s of
            # dead air (field 2026-08-20 20:31). Straight to the local voice.
            # If local isn't installed, fall through and try anyway; the request is
            # bounded by TTS_API_TIMEOUT_SECS now, so the cost is capped.
            if _api_circuit_open() and _speak_local_fallback(
                clean_text, emotion,
                on_playback_start=on_playback_start,
                post_playback_tail_secs=post_playback_tail_secs,
                flush_on_playback_stop=flush_on_playback_stop,
                log_text=log_text,
                reason="ElevenLabs unreachable",
            ):
                return
        logger.info(
            "[tts] cache miss — calling ElevenLabs API%s",
            f" (voice_settings={_summarize_settings(voice_settings)})"
            if voice_settings else "",
        )
        audio_bytes = _fetch_from_api(synth_text, voice_id, model_id, voice_settings, previous_text)
        if not audio_bytes:
            # ElevenLabs failed (network / quota / error) and the streaming path
            # above already failed too. Rather than drop the line, keep Rex
            # talking in his on-device voice (and the breaker, opened by
            # _fetch_from_api, routes the rest of the reply straight to local).
            _speak_local_fallback(
                clean_text, emotion,
                on_playback_start=on_playback_start,
                post_playback_tail_secs=post_playback_tail_secs,
                flush_on_playback_stop=flush_on_playback_stop,
                log_text=log_text,
                reason="ElevenLabs unavailable",
            )
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
            conv_log.log_rex(clean_text)
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


def _ends_hot(audio: np.ndarray, samplerate: int) -> bool:
    """True when the take ends at speech level instead of decaying to silence —
    the signature of a generation truncated mid-word. A natural ElevenLabs take
    tails off to RMS ~0.0002-0.003; truncated ones have been measured ending at
    0.02+. Checks the FINAL window only (an intra-word dip can't false-positive)."""
    threshold = float(getattr(config, "TTS_HOT_END_RMS", 0.010) or 0.0)
    if threshold <= 0.0 or audio is None or audio.size == 0 or samplerate <= 0:
        return False
    window = max(1, int(samplerate * 0.030))
    tail = audio[-window:]
    return float(np.sqrt(np.mean(tail.astype(np.float64) ** 2))) >= threshold


# ── Internal: playback ────────────────────────────────────────────────────────

def _stream_pcm_samplerate() -> int:
    """Sample rate implied by the configured ElevenLabs PCM stream format."""
    fmt = str(getattr(config, "TTS_STREAM_PCM_FORMAT", "pcm_22050"))
    try:
        return int(fmt.split("_", 1)[1])
    except (IndexError, ValueError):
        return 22050


# ── Shared streamed-playback scaffolding ──────────────────────────────────────
# Factored out of _speak_streaming so the local (Qwen3-TTS) path in _speak_local
# reuses the exact same LED/servo/AEC/mouth-drive behavior. Both streamed backends
# share these; the buffered _play() path keeps its own (it drives LEDs from a
# separate thread and guards sd.wait() early-return, which streaming doesn't).

def _stream_output_latency(stream) -> float:
    """PortAudio's REPORTED output latency for this stream, in seconds.

    Not the value we asked for — the CoreAudio host rounds the request up hard
    (measured on the robot Mac 2026-08-20: 'high'/4096 → 0.95 s, the clone deep
    buffer 1.2/8192 → 3.18 s, the boot buffer 2.5/8192 → 6.16 s). This is real
    audio sitting between write() and the speaker, not bookkeeping: stream.stop()
    drains for exactly this long no matter how little was written, while
    stream.abort() — which throws the buffer away — returns in 0.16 s.
    """
    try:
        return max(0.0, float(stream.latency))
    except Exception:
        return 0.0


def _playing_stream_latency(sd) -> float:
    """Reported output latency of the stream sd.play() just opened. Separate from
    _stream_output_latency because sd.get_stream() itself raises when no stream is
    live (a disabled/absent output device), and a missing latency must degrade to
    "no delay", never to a crash on the playback path."""
    try:
        return _stream_output_latency(sd.get_stream())
    except Exception:
        return 0.0


def _mouth_on(led_emotion: str) -> None:
    """The visible mouth-on trigger (head SPEAK: + chest), split out of
    _begin_speech so the playback paths can withhold it until the audio is
    actually audible. See _MouthPacer."""
    leds_head.speak(led_emotion)
    leds_chest.speak(led_emotion)


class _MouthPacer:
    """Applies mouth levels on the AUDIBLE timeline instead of the write one.

    Every playback path set a chunk's brightness immediately before writing that
    chunk to the device, but the device holds _stream_output_latency() seconds of
    audio downstream — so the mouth ran that far AHEAD of the voice: it started
    animating ~1 s before any sound on an ordinary line and ~3.2 s early during an
    impersonation (the clone deep buffer is armed for exactly that window), then
    froze at the last level for the final ~3.2 s. Owner report 2026-08-20: "the
    mouth LEDs start animating as if they're speaking before he actually speaks."

    Levels are queued against a deadline on the audio timeline —
    t0 + latency + (audio scheduled so far) — rather than "now + latency", so a
    line SHORTER than the buffer (whose writes all return at once) still animates
    spread across its real duration instead of firing in one burst. The mouth-on
    trigger is withheld until the first level comes due, so the head starts
    talking exactly when the room hears him.

    Below TTS_MOUTH_SYNC_MIN_LATENCY_SECS — or with TTS_MOUTH_SYNC_ENABLED off —
    this is a pass-through: no thread, no queue, the pre-2026-08-20 behavior.
    """

    __slots__ = ("_delay", "_sr", "_t0", "_cursor", "_q", "_cv", "_thread",
                 "_stop", "_on_start", "_started", "_min_delta", "_last_led")

    def __init__(self, latency_secs: float, samplerate: int, on_start: Optional[Callable[[], None]]):
        self._sr = float(samplerate or 1) or 1.0
        self._on_start = on_start
        self._started = False
        self._min_delta = int(getattr(config, "HEAD_LED_SPEAK_LEVEL_MIN_DELTA", 8))
        self._last_led = -1
        self._cursor = 0.0
        self._q: "deque[tuple[float, int]]" = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        self._t0 = time.monotonic()
        enabled = bool(getattr(config, "TTS_MOUTH_SYNC_ENABLED", True))
        floor = float(getattr(config, "TTS_MOUTH_SYNC_MIN_LATENCY_SECS", 0.08))
        self._delay = max(0.0, float(latency_secs or 0.0)) if enabled else 0.0
        if self._delay < floor:
            self._delay = 0.0
            self._fire_start()          # pass-through: light the mouth right now
            return
        logger.debug("[tts] mouth synced to a %.2fs output buffer", self._delay)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tts-mouth-pacer",
        )
        self._thread.start()

    # ── internals ────────────────────────────────────────────────────────────
    def _fire_start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception as exc:
                logger.debug("[tts] deferred mouth-on failed: %s", exc)

    def _apply(self, brightness: int) -> None:
        """Throttled by HEAD_LED_SPEAK_LEVEL_MIN_DELTA, same as the inline path —
        the per-frame flood overlaps the Arduino's interrupt-off show() windows."""
        if self._last_led < 0 or abs(brightness - self._last_led) >= self._min_delta or (
            brightness == 0 and self._last_led != 0
        ):
            try:
                leds_head.speak_level(brightness)
                servos.speech_reactive_move(brightness / 255.0)
            except Exception:
                pass
            self._last_led = brightness

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._q:
                    self._cv.wait(0.2)
                if self._stop:
                    return
                due, level = self._q[0]
                now = time.monotonic()
                if due > now:
                    self._cv.wait(due - now)
                    continue            # re-check: a close() may have arrived
                self._q.popleft()
            self._fire_start()
            self._apply(level)

    # ── public ───────────────────────────────────────────────────────────────
    def push(self, samples: "np.ndarray") -> None:
        """Schedule one chunk's mouth level, and advance the audio cursor by its
        length. Callers pass exactly what they hand to stream.write()."""
        n = int(getattr(samples, "size", 0) or 0)
        rms = float(np.sqrt(np.mean(samples ** 2))) if n else 0.0
        brightness = min(255, int(rms * config.TTS_LED_BRIGHTNESS_SCALE))
        if self._thread is None:
            self._apply(brightness)
            return
        with self._cv:
            self._q.append((self._t0 + self._delay + self._cursor, brightness))
            self._cursor += n / self._sr
            self._cv.notify()

    @property
    def delay(self) -> float:
        """Seconds a level is held before it reaches the mouth — i.e. how far the
        device is behind the writes. 0.0 in pass-through mode."""
        return self._delay

    def close(self, *, canceled: bool = False) -> None:
        """Stop the pacer BEFORE the caller's LED restore. On barge-in the device
        buffer is thrown away by stream.abort(), so anything still queued is for
        audio nobody will hear — dropping it also keeps a stale SPEAK_LEVEL from
        landing after SPEAK_STOP, which the next SPEAK: would inherit (the head
        firmware does not reset the level on a new utterance)."""
        with self._cv:
            self._stop = True
            self._q.clear()
            self._cv.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        if canceled:
            return
        # A line shorter than the buffer latency can end before its first level
        # came due. The mouth never lit, so leave it dark rather than flashing it
        # on at teardown — _end_speech's speak_stop is a no-op on a dark mouth.


def _begin_speech(emotion: str, ttl_secs: float, *, defer_mouth: bool = False):
    """Playback prologue: publish the emotion frame, start servo/animation speech
    motion, light eyes/mouth/chest, and arm AEC suppression. Returns
    (emotion_frame, led_emotion).

    defer_mouth leaves the mouth-on trigger (_mouth_on) to a _MouthPacer, so it
    fires when the audio is audible rather than up to ~3.2 s earlier. Everything
    else stays here on purpose: the speech-activity flag and begin_speech_motion
    are the "Rex owns the head now" claim (delaying them lets idle wander grab the
    neck mid-line), and AEC suppression must be armed before any sound reaches the
    room, not after."""
    emotion_frame = emotion_orchestrator.frame_for_speech(emotion)
    led_emotion = emotion_frame.led_style
    emotion_orchestrator.publish_frame(emotion_frame, ttl_secs=ttl_secs)
    try:
        animations.speech_activity_start()
        servos.begin_speech_motion(emotion_frame)
    except Exception as exc:
        logger.debug("[tts] speech servo start failed: %s", exc)
    if not defer_mouth:
        leds_head.speak(led_emotion)
    leds_head.ensure_eyes_on(led_emotion)
    if not defer_mouth:
        leds_chest.speak(led_emotion)
    echo_cancel.set_playing(True)
    return emotion_frame, led_emotion


def _led_chunks(samples: np.ndarray, sr: int) -> Iterator[np.ndarray]:
    """Split one generator chunk into LED-update-sized pieces.

    The streaming paths drive the mouth once per chunk they receive, which is
    fine when chunks arrive every ~0.3 s but not when one chunk IS the whole
    line. A clone take renders as a single unit (LOCAL_TTS_TAKE_WHOLE_CLIP), so
    stream() hands playback one ~12 s array: the mouth got a single RMS for the
    entire bit and the head sat still through it, and the 12 s blocking write
    could not be barged in on. Re-slicing here restores per-frame mouth motion
    and gives barge-in a check every ~33 ms.
    """
    step = max(1, int(sr * float(getattr(config, "TTS_LED_UPDATE_INTERVAL_SECS", 0.033))))
    if samples.size <= step:
        yield samples
        return
    for i in range(0, samples.size, step):
        piece = samples[i:i + step]
        if piece.size:
            yield piece


def _end_speech(
    stream,
    post_playback_tail_secs: Optional[float],
    flush_on_playback_stop: Optional[bool],
    *,
    pacer: Optional["_MouthPacer"] = None,
    canceled: bool = False,
) -> None:
    """Playback epilogue for the streamed paths: close the stream, restore
    LEDs/servo/animation, release AEC suppression, clear the speaking flag.

    The pacer is stopped FIRST: its thread is still holding queued mouth levels,
    and one landing after speak_stop() would leave a stale SPEAK_LEVEL that the
    next SPEAK: inherits (the head firmware does not clear the level on a new
    utterance, so the mouth would open at the previous line's amplitude)."""
    global _speaking
    if pacer is not None:
        pacer.close(canceled=canceled)
    if stream is not None:
        try:
            stream.close()
        except Exception:
            pass
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


def _speak_streaming(
    synth_text: str,
    spoken_text: str,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[dict],
    previous_text: Optional[str],
    emotion: str,
    cache_file: Path,
    *,
    on_playback_start: Optional[Callable[[], None]] = None,
    post_playback_tail_secs: Optional[float] = None,
    flush_on_playback_stop: Optional[bool] = None,
    log_text: bool = True,
) -> bool:
    """Cache-miss synthesis with STREAMING playback: audio starts the moment the
    first PCM bytes arrive from ElevenLabs (~0.3-0.6s) instead of after the full
    generation is buffered (~1.5-2s for a conversational sentence — the dominant
    remaining latency stage, measured 2026-07-06). Returns True when the line was
    handled (played, or legitimately skipped); False = caller should fall back to
    the buffered path.

    Parity with _play(): output gate, speaking flag, emotion frame, LEDs/servos,
    AEC suppression, barge-in (polls echo_cancel.was_canceled() between chunk
    writes — the interrupters set it right before sd.stop(), which only affects
    sd.play streams, so the poll IS the cancel path here). Mouth LEDs are driven
    inline from each written chunk's RMS (write-paced ≈ playback-paced within the
    buffer latency), replacing the array-based _drive_leds thread. The full take
    is accumulated and cached as WAV in the background for future cache hits.
    """
    global _speaking
    try:
        import sounddevice as sd
        from elevenlabs import VoiceSettings
    except ImportError as exc:
        logger.debug("[tts] streaming unavailable (%s)", exc)
        return False

    samplerate = _stream_pcm_samplerate()
    client = _get_el_client()
    kwargs = {
        "voice_id": voice_id,
        "text": synth_text,
        "model_id": model_id,
        "output_format": str(getattr(config, "TTS_STREAM_PCM_FORMAT", "pcm_22050")),
    }
    if voice_settings:
        kwargs["voice_settings"] = VoiceSettings(
            **{k: v for k, v in voice_settings.items() if v is not None}
        )
    seed = _v3_seed(model_id)
    if seed is not None:
        kwargs["seed"] = seed
    prev = _stitch_previous_text(previous_text, model_id)
    if prev:
        kwargs["previous_text"] = prev

    requested_at = time.monotonic()
    try:
        chunk_iter = iter(client.text_to_speech.stream(**kwargs))
        first_chunk = next(chunk_iter, None)
    except Exception as exc:
        # Discriminate the same way warmup_api() does: an ApiError is a completed
        # round-trip (quota, 4xx) and the buffered path may still be worth a try,
        # but a NETWORK-level failure means the endpoint is unreachable. Recording
        # it here is what stops speak() paying a second full timeout on the very
        # same dead endpoint (field 2026-08-20: 25 s + 26 s for one two-word line).
        if type(exc).__name__ != "ApiError":
            _note_api_failure()
        logger.warning("[tts] streaming request failed (%s) — buffered fallback", exc)
        return False
    if not first_chunk:
        logger.warning("[tts] streaming returned no audio — buffered fallback")
        return False
    logger.info(
        "[tts] streaming first audio bytes in %.2fs", time.monotonic() - requested_at
    )
    _note_api_success()   # a completed streaming round-trip clears the fallback breaker

    if log_text:
        try:
            conv_log.log_rex(spoken_text)
        except Exception as exc:
            logger.debug("[tts] conversation log write failed: %s", exc)

    with output_gate.hold("tts", timeout=_gate_timeout()) as acquired:
        if not acquired:
            _log_gate_timeout("streamed playback")
            return True   # handled: deliberately skipped, same as _play()

        with _speaking_lock:
            _speaking = True
        pcm_carry = b""
        all_samples: list[np.ndarray] = []
        canceled = False
        play_started_at = time.monotonic()
        stream = None
        pacer = None
        try:
            _, led_emotion = _begin_speech(emotion, ttl_secs=8.0, defer_mouth=True)
            stream = sd.OutputStream(
                samplerate=samplerate, channels=1, dtype="float32",
                **playback_stream_kwargs(),
            )
            stream.start()
            # Mouth-on and every level from here run on the audible timeline, one
            # output-latency behind the writes (see _MouthPacer). Opened AFTER
            # stream.start() because only a started stream reports its real latency.
            pacer = _MouthPacer(
                _stream_output_latency(stream), samplerate,
                lambda: _mouth_on(led_emotion),
            )
            if on_playback_start is not None:
                try:
                    on_playback_start()
                except Exception:
                    pass

            chunk = first_chunk
            while chunk is not None:
                if echo_cancel.was_canceled():
                    canceled = True
                    break
                raw = pcm_carry + chunk
                usable = len(raw) - (len(raw) % 2)   # int16 alignment
                pcm_carry = raw[usable:]
                if usable:
                    samples = (
                        np.frombuffer(raw[:usable], dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    all_samples.append(samples)
                    # Mouth drive, re-sliced to LED-frame size so a fat network
                    # chunk still animates at ~30 fps; the pacer holds each level
                    # until its audio is actually audible.
                    for piece in _led_chunks(samples, samplerate):
                        pacer.push(piece)
                    stream.write(samples)   # blocks on buffer space — natural pacing
                chunk = next(chunk_iter, None)

            if canceled:
                stream.abort()
            else:
                # Push the final speech samples fully through the device before
                # teardown: write a short zero pad so the tail can't be clipped by
                # the host buffer at stop() (CoreAudio has been observed dropping
                # the last ~latency window despite stop()'s documented drain), then
                # stop() waits for the buffered audio to finish playing.
                pad_ms = float(getattr(config, "TTS_STREAM_END_PAD_MS", 200.0) or 0.0)
                if pad_ms > 0:
                    try:
                        pad = np.zeros(int(samplerate * pad_ms / 1000.0),
                                       dtype=np.float32)
                        # Through the pacer too, so the mouth closes on the last
                        # audible word instead of holding the final level through
                        # the buffer drain.
                        pacer.push(pad)
                        stream.write(pad)
                    except Exception:
                        pass
                stream.stop()
        except Exception as exc:
            logger.error("[tts] streamed playback error: %s", exc)
            # Audio may have partially played; do NOT fall back (would double-speak).
        finally:
            _end_speech(stream, post_playback_tail_secs, flush_on_playback_stop,
                        pacer=pacer, canceled=canceled)

        logger.info(
            "[tts] streamed playback %s in %.2fs",
            "canceled" if canceled else "done",
            time.monotonic() - play_started_at,
        )

    # Cache the complete take as WAV — synchronous on purpose: the write is
    # milliseconds, and a daemon thread got killed at process exit before the file
    # landed (observed live: the second identical line re-streamed instead of
    # cache-hitting).
    if all_samples and not canceled:
        try:
            import soundfile as sf
            full = np.concatenate(all_samples)
            if _ends_hot(full, samplerate):
                # The stream delivered audio that ends at speech level with no
                # decay — the generation itself was truncated mid-word (observed
                # live 2026-07-06: cached take ended at RMS 0.023). Caching it
                # would make the clipped ending PERMANENT for this line; skip so
                # the next utterance re-rolls a full take.
                logger.warning(
                    "[tts] streamed take ends hot (speech-level RMS at tail) — "
                    "truncated generation? NOT caching %s", cache_file.stem[:16],
                )
                return True
            trimmed = _trim_trailing_silence(full, samplerate)
            wav_path = cache_file.with_suffix(".wav")
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), trimmed, samplerate)
            logger.info("[tts] saved streamed take to cache: %s", wav_path.name)
        except Exception as exc:
            logger.debug("[tts] streamed cache write failed: %s", exc)
    return True


# ── Output-gate acquisition ───────────────────────────────────────────────────
# Every TTS acquire is BOUNDED. The gate holder calls into CoreAudio while
# holding it (sound_effects' gated path plays inside its own hold), so a wedged
# USB audio device strands the gate on a thread that will never release it —
# and an unbounded acquire here made Rex permanently MUTE on top of permanently
# deaf. Field 2026-08-18: the impersonation "thinking" chirp wedged mid-clip,
# and the finished Jimmy Carter take then waited 91 s on a gate nobody would
# ever hand back. Timing out costs one dropped line; not timing out costs the
# rest of the session.

def _gate_timeout() -> float:
    return max(1.0, float(getattr(config, "TTS_OUTPUT_GATE_TIMEOUT_SECS", 30.0)))


def _log_gate_timeout(what: str) -> None:
    logger.warning(
        "[tts] %s dropped — waited %.0fs for the output gate, still held by %r "
        "(%.0fs). Suspect a wedged audio device.",
        what, _gate_timeout(), output_gate.active_source(), output_gate.held_secs(),
    )


# ── ElevenLabs → local fallback circuit breaker ───────────────────────────────
# When ElevenLabs fails (network down / quota exhausted / API error), Rex keeps
# talking in his on-device voice instead of going silent. The breaker holds the
# fallback for LOCAL_TTS_FALLBACK_HOLD_SECS so the rest of a reply doesn't pay a
# multi-second API timeout per sentence; the hold naturally expires and the next
# line probes ElevenLabs again (a success clears it early).
_api_down_until = 0.0
_api_breaker_lock = threading.Lock()


def _api_circuit_open() -> bool:
    """True while the fallback breaker is holding (recent ElevenLabs failure)."""
    if not bool(getattr(config, "LOCAL_TTS_FALLBACK_ENABLED", True)):
        return False
    with _api_breaker_lock:
        return time.monotonic() < _api_down_until


def _note_api_failure() -> None:
    """Record an ElevenLabs failure; open the breaker for the configured hold."""
    global _api_down_until
    if not bool(getattr(config, "LOCAL_TTS_FALLBACK_ENABLED", True)):
        return
    hold = float(getattr(config, "LOCAL_TTS_FALLBACK_HOLD_SECS", 120.0))
    with _api_breaker_lock:
        was_open = time.monotonic() < _api_down_until
        _api_down_until = time.monotonic() + hold
    if not was_open:
        logger.warning(
            "[tts] ElevenLabs down — holding on Rex's local voice for %.0fs", hold
        )


def _note_api_success() -> None:
    """Any successful ElevenLabs round-trip clears the breaker."""
    global _api_down_until
    with _api_breaker_lock:
        was_open = time.monotonic() < _api_down_until
        _api_down_until = 0.0
    if was_open:
        logger.info("[tts] ElevenLabs recovered — resuming Rex's primary voice")


_local_ref_missing_warned = False


def _rex_local_ref():
    """Rex's local voice reference, or None if the model/ref isn't installed.
    A None while the local backend is WANTED is exactly the silent-ElevenLabs-
    fallback failure the dev mac hit — warn loudly (once per run), never quietly."""
    global _local_ref_missing_warned
    try:
        from audio import local_tts
        if not local_tts.is_available():
            return None
        ref = local_tts.rex_voice_ref()
        if ref is None and not _local_ref_missing_warned:
            _local_ref_missing_warned = True
            logger.warning(
                "[tts] local voice wanted but Rex's reference clip is missing "
                "(%s) — falling back to ElevenLabs",
                local_tts.unavailable_reason(require_rex_ref=True),
            )
        return ref
    except Exception:
        return None


def _speak_local_fallback(
    clean_text: str,
    emotion: str,
    *,
    on_playback_start=None,
    post_playback_tail_secs=None,
    flush_on_playback_stop=None,
    log_text: bool = True,
    reason: str = "ElevenLabs unavailable",
) -> bool:
    """Speak this line in Rex's ON-DEVICE voice because ElevenLabs is not usable.

    Returns True when the line was actually spoken. False means the local engine
    is not installed or failed, and the caller should decide whether to keep
    trying ElevenLabs or drop the line."""
    if not _use_local_backend():
        return False
    fallback_ref = _rex_local_ref()
    if fallback_ref is None:
        return False
    logger.info("[tts] %s — speaking locally", reason)
    try:
        return bool(_speak_local(
            clean_text, fallback_ref, emotion,
            on_playback_start=on_playback_start,
            post_playback_tail_secs=post_playback_tail_secs,
            flush_on_playback_stop=flush_on_playback_stop,
            log_text=log_text,
        ))
    except Exception as exc:
        logger.warning("[tts] local fallback error (%s)", exc)
        return False


def _use_local_backend() -> bool:
    """True when the local Qwen3-TTS engine should render Rex's OWN voice this
    turn: --local-tts mode is on (or the ElevenLabs circuit breaker is open) and
    the model is installed."""
    offline = False
    try:
        from intelligence import connectivity
        offline = connectivity.is_offline()
    except Exception:
        offline = False
    if not (
        bool(getattr(config, "LOCAL_TTS_MODE", False)) or _api_circuit_open()
        or offline
    ):
        return False
    try:
        from audio import local_tts
        return local_tts.is_available()
    except Exception:
        return False


def _speak_local(
    clean_text: str,
    voice_ref,
    emotion: str,
    *,
    on_playback_start: Optional[Callable[[], None]] = None,
    post_playback_tail_secs: Optional[float] = None,
    flush_on_playback_stop: Optional[bool] = None,
    log_text: bool = True,
) -> bool:
    """Synthesize `clean_text` on-device in `voice_ref`'s voice and play it with
    full parity (output gate, AEC, mouth LEDs, servo motion, barge-in). Returns
    True when handled (played or deliberately skipped); False = caller should fall
    back to ElevenLabs.

    `clean_text` must already be audio-tag-free — Qwen would read [tags] aloud.
    Rex's own voice (label 'rex') is cached as WAV keyed on the local backend so
    repeat lines are instant; impersonation voices are never cached (one-off).
    """
    global _speaking
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("[tts] sounddevice not installed — cannot play local audio")
        return False
    try:
        from audio import local_tts
    except Exception as exc:
        logger.warning("[tts] local TTS engine unavailable (%s)", exc)
        return False

    sr = local_tts.sample_rate()
    # Only Rex's own voice is ever cacheable (impersonation takes are one-off), and
    # only when the local cache is enabled — off by default so --local-tts testing
    # always hears freshly synthesized audio.
    cacheable = (
        getattr(voice_ref, "label", "") == "rex"
        and bool(getattr(config, "LOCAL_TTS_CACHE_ENABLED", False))
    )

    # Cache hit (Rex voice only) → play the stored WAV through the buffered path.
    cache_file = None
    if cacheable:
        cache_file = _cache_path(
            clean_text, f"local:{voice_ref.label}",
            str(getattr(config, "LOCAL_TTS_MODEL_ID", "qwen-tts")),
        )
        wav_path = cache_file.with_suffix(".wav")
        if wav_path.exists():
            logger.info("[tts] local cache hit: %s", wav_path.name)
            audio, samplerate = _read_audio(wav_path)
            if audio is not None and len(audio):
                if log_text:
                    try:
                        conv_log.log_rex(clean_text)
                    except Exception as exc:
                        logger.debug("[tts] conversation log write failed: %s", exc)
                _play(
                    audio, samplerate, emotion,
                    on_playback_start=on_playback_start,
                    post_playback_tail_secs=post_playback_tail_secs,
                    flush_on_playback_stop=flush_on_playback_stop,
                )
                return True

    # A CLONED voice plays from a sentence pipeline: sentence 1 starts playing
    # the moment it's rendered while sentence 2 generates behind it, so the room
    # waits one sentence of synthesis instead of the whole take — and each unit
    # is fully buffered before it plays, so nothing stutters mid-sentence (field
    # 2026-08-01: chunk-level streaming starved on a 12s parody line). The
    # impersonation flow starts the take behind Rex's intro line and parks it;
    # any other cloned line starts one here. Rex's own voice keeps the
    # chunk-level stream — his lines are short and latency matters.
    is_clone = getattr(voice_ref, "label", "") != "rex"
    take = local_tts.pop_take(clean_text, voice_ref) if is_clone else None
    if take is None and is_clone and bool(getattr(config, "LOCAL_TTS_TAKE_PIPELINE", True)):
        take = local_tts.Take(clean_text, voice_ref)

    # Kill switch: pipeline off → render the clone whole and play it like a
    # cache hit (the pre-pipeline behavior).
    if take is None and is_clone and bool(getattr(config, "LOCAL_TTS_CLONE_FULL_BUFFER", True)):
        try:
            full_audio, full_sr = local_tts.synthesize(clean_text, voice_ref)
        except Exception as exc:
            logger.warning("[tts] clone full-buffer synthesis failed (%s) — streaming", exc)
            full_audio, full_sr = None, sr
        if full_audio is not None and len(full_audio):
            logger.info("[tts] local buffered take: %.1fs audio (voice=%s)",
                        len(full_audio) / float(full_sr), getattr(voice_ref, "label", "?"))
            if log_text:
                try:
                    conv_log.log_rex(clean_text)
                except Exception as exc:
                    logger.debug("[tts] conversation log write failed: %s", exc)
            _play(
                full_audio, full_sr, emotion,
                on_playback_start=on_playback_start,
                post_playback_tail_secs=post_playback_tail_secs,
                flush_on_playback_stop=flush_on_playback_stop,
            )
            return True

    front_pad = np.zeros(
        int(sr * float(getattr(config, "LOCAL_TTS_FRONT_PAD_MS", 150)) / 1000.0),
        dtype=np.float32,
    )
    preroll_samples = int(sr * float(getattr(config, "LOCAL_TTS_PREROLL_SEC", 0.25)))
    requested_at = time.monotonic()

    # The generator holds local_tts._generate_lock for its whole lifetime, so it
    # MUST be closed on every exit path (including barge-in mid-stream) or the next
    # synthesis deadlocks. gen.close() raises GeneratorExit inside it, releasing it.
    # (A take's stream() closes the take instead — same contract, and it stops the
    # background renderer so an interrupted bit doesn't keep synthesizing.)
    gen = take.stream() if take is not None else local_tts.generate_stream(clean_text, voice_ref)
    try:
        # Prime the pre-roll cushion (or drain fully if the line is short) so the
        # output stream never underruns waiting on the first model chunk.
        buffered: list[np.ndarray] = []
        buffered_n = 0
        canceled = False
        try:
            for chunk in gen:
                if echo_cancel.was_canceled():
                    canceled = True
                    break
                buffered.append(chunk)
                buffered_n += chunk.size
                if buffered_n >= preroll_samples:
                    break
        except Exception as exc:
            logger.warning("[tts] local synth failed to start (%s)", exc)
            return False
        if not buffered:
            logger.warning("[tts] local synth produced no audio")
            return False

        if log_text:
            try:
                conv_log.log_rex(clean_text)
            except Exception as exc:
                logger.debug("[tts] conversation log write failed: %s", exc)

        with output_gate.hold("tts", timeout=_gate_timeout()) as acquired:
            if not acquired:
                _log_gate_timeout("local playback")
                return True   # handled: deliberately skipped, same as _play()

            with _speaking_lock:
                _speaking = True
            # A whole-clip clone take is already fully buffered here, so its real
            # length is known and the emotion frame is given enough TTL to cover
            # it. The flat 8 s expired partway through a ~13 s impersonation,
            # dropping the speech pose and lights before the bit had finished.
            # ...plus the output buffer, which the mouth now waits out too (the
            # pacer holds each level one full latency, so the pose and lights must
            # outlive that as well).
            buffered_secs = sum(int(c.size) for c in buffered) / float(sr or 1)
            _, led_emotion = _begin_speech(
                emotion, ttl_secs=max(8.0, buffered_secs + 6.0), defer_mouth=True,
            )

            all_samples: list[np.ndarray] = list(buffered)
            ttfa_logged = False
            play_started_at = time.monotonic()
            stream = None
            pacer = None
            try:
                stream = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="float32",
                    **playback_stream_kwargs(),
                )
                stream.start()
                # Only a started stream reports its real output latency. The clone
                # deep buffer makes this ~2.9 s at 24 kHz, which is exactly how far
                # ahead of the voice the mouth used to run on an impersonation.
                pacer = _MouthPacer(
                    _stream_output_latency(stream), sr, lambda: _mouth_on(led_emotion),
                )
                if on_playback_start is not None:
                    try:
                        on_playback_start()
                    except Exception:
                        pass
                if not canceled:
                    # Through the pacer as well: the front pad is 150 ms of real
                    # device time and the mouth must not open across it.
                    pacer.push(front_pad)
                    stream.write(front_pad)
                for samples in buffered:
                    if canceled:
                        break
                    # Re-sliced to LED-frame size: one clone take arrives as a
                    # single ~12 s array, and writing it whole froze the mouth and
                    # the head for the entire line (see _led_chunks).
                    for piece in _led_chunks(samples, sr):
                        if echo_cancel.was_canceled():
                            canceled = True
                            break
                        pacer.push(piece)
                        stream.write(piece)
                        if not ttfa_logged:
                            # Logged after the FIRST written piece. It used to sit
                            # after the whole first buffered unit, so a one-unit
                            # 8 s clone take read as "first audio in 8.24s" while
                            # audio had actually been playing the entire time
                            # (2026-08-19 log forensics went down that hole).
                            logger.info(
                                "[tts] local first audio in %.2fs (+%.2fs output "
                                "buffer before it is audible)",
                                time.monotonic() - requested_at,
                                pacer.delay,
                            )
                            ttfa_logged = True
                if not canceled:
                    for samples in gen:
                        if canceled:
                            break
                        all_samples.append(samples)
                        for piece in _led_chunks(samples, sr):
                            if echo_cancel.was_canceled():
                                canceled = True
                                break
                            pacer.push(piece)
                            stream.write(piece)

                if canceled:
                    stream.abort()
                else:
                    pad_ms = float(getattr(config, "TTS_STREAM_END_PAD_MS", 200.0) or 0.0)
                    if pad_ms > 0:
                        try:
                            pad = np.zeros(int(sr * pad_ms / 1000.0), dtype=np.float32)
                            pacer.push(pad)   # closes the mouth on the last word
                            stream.write(pad)
                        except Exception:
                            pass
                    stream.stop()
            except Exception as exc:
                logger.error("[tts] local playback error: %s", exc)
            finally:
                _end_speech(stream, post_playback_tail_secs, flush_on_playback_stop,
                            pacer=pacer, canceled=canceled)

            logger.info(
                "[tts] local playback %s in %.2fs (backend=local, voice=%s)",
                "canceled" if canceled else "done",
                time.monotonic() - play_started_at,
                getattr(voice_ref, "label", "?"),
            )

        # Cache the full take (Rex voice only) for future hits.
        if cacheable and cache_file is not None and all_samples and not canceled:
            try:
                import soundfile as sf
                full = _trim_trailing_silence(np.concatenate(all_samples), sr)
                wav_out = cache_file.with_suffix(".wav")
                wav_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav_out), full, sr)
                logger.info("[tts] saved local take to cache: %s", wav_out.name)
            except Exception as exc:
                logger.debug("[tts] local cache write failed: %s", exc)
        return True
    finally:
        try:
            gen.close()
        except Exception:
            pass


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

    with output_gate.hold("tts", timeout=_gate_timeout()) as acquired:
        if not acquired:
            _log_gate_timeout("playback")
            return

        with _speaking_lock:
            _speaking = True

        stop_event = threading.Event()
        led_thread: Optional[threading.Thread] = None

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
            # Re-assert the eyes ON at the emotion colour every turn. The mouth
            # SPEAK command never touches the eyes, and the serial link is lossy
            # during speech, so without this the eyes ride entirely on a single
            # easily-dropped post-speech re-arm. The heartbeat keeps them lit
            # between turns; this guarantees they are lit for the turn itself.
            leds_head.ensure_eyes_on(led_emotion)
            echo_cancel.set_playing(True)
            if on_playback_start is not None:
                try:
                    on_playback_start()
                except Exception:
                    pass
            sd.play(audio, samplerate, **playback_stream_kwargs())
            # The mouth (head SPEAK: + chest) and the RMS walk both start one
            # output latency from now, when the room actually hears this. sd.play
            # returns immediately, so the stream is live and reporting by here.
            led_thread = threading.Thread(
                target=_drive_leds,
                args=(audio, samplerate, stop_event,
                      _playing_stream_latency(sd),
                      lambda: _mouth_on(led_emotion)),
                daemon=True,
                name="tts-leds",
            )
            led_thread.start()
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
            if led_thread is not None and led_thread.is_alive():
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
    audio: np.ndarray, samplerate: int, stop_event: threading.Event,
    start_delay: float = 0.0,
    on_start: Optional[Callable[[], None]] = None,
) -> None:
    """Iterate audio in fixed chunks, driving mouth LED brightness from RMS.

    start_delay is the playback stream's reported output latency. This thread
    walks the array on a wall clock that starts when it starts, but sd.play()
    hands the audio to a device holding that much buffer — so without the wait
    the mouth ran the whole line ~1 s early (~3.2 s during an impersonation,
    where the clone deep buffer is armed) and never resynced. Waiting it out
    first, and holding the mouth-on trigger until then, lines the two up. See
    _MouthPacer for the same fix on the streamed paths.
    """
    interval = config.TTS_LED_UPDATE_INTERVAL_SECS
    chunk_len = max(1, int(samplerate * interval))
    min_delta = int(getattr(config, "HEAD_LED_SPEAK_LEVEL_MIN_DELTA", 8))
    last_sent = -1

    if start_delay > 0.0 and stop_event.wait(timeout=start_delay):
        return   # canceled/finished before the first sound — never light the mouth
    if on_start is not None:
        try:
            on_start()
        except Exception as exc:
            logger.debug("[tts] deferred mouth-on failed: %s", exc)

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


def _v3_seed(model_id: str) -> Optional[int]:
    """Fixed RNG seed for eleven_v3 so an identical request is reproducible (and the cache is
    deterministic). NOTE: seed does NOT keep DIFFERENT sentences sounding alike — that is what
    _stitch_previous_text is for. None on other models, or when TTS_V3_SEED is unset."""
    if str(model_id).strip() != "eleven_v3":
        return None
    seed = getattr(config, "TTS_V3_SEED", None)
    return int(seed) if seed is not None else None


def _stitch_previous_text(previous_text: Optional[str], model_id: str) -> str:
    """The text that came before this line, for ElevenLabs request stitching — so v3 continues one
    performance across a reply's per-sentence requests instead of re-rolling the voice each call.
    Capped to the last N chars (the near context carries the continuity). "" when disabled, not v3,
    or empty — must be computed identically here and in the request so cache keys line up."""
    if not previous_text:
        return ""
    # eleven_v3 REJECTS previous_text (HTTP 400 "unsupported_model") — verified against the live API.
    # So NEVER send it on v3; doing so drops the sentence. v3 consistency instead comes from
    # whole-reply synthesis (LLM_STREAMING_TTS_ENABLED off → a reply is ONE generation). Stitching
    # stays available for models that DO support it (v2 / turbo), should we ever stream on those.
    if str(model_id).strip() == "eleven_v3":
        return ""
    if not bool(getattr(config, "TTS_V3_STITCH_ENABLED", True)):
        return ""
    cap = int(getattr(config, "TTS_V3_STITCH_MAX_CHARS", 400))
    # Normalize like the spoken text so the conditioning context matches what was actually said
    # (e.g. "WWII" -> "World War Two"), not the raw transcript form.
    text = _normalize_for_speech(str(previous_text)).strip()
    return text[-cap:] if cap > 0 else text


def _cache_path(
    text: str,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[dict] = None,
    previous_text: Optional[str] = None,
) -> Path:
    settings_token = _settings_cache_token(voice_settings)
    seed = _v3_seed(model_id)
    seed_token = f"|seed={seed}" if seed is not None else ""
    prev = _stitch_previous_text(previous_text, model_id)
    prev_token = f"|prev={prev}" if prev else ""
    digest = hashlib.sha256(
        f"{text}{voice_id}{model_id}{settings_token}{seed_token}{prev_token}".encode("utf-8")
    ).hexdigest()
    return Path(config.TTS_CACHE_DIR) / f"{digest}.mp3"


def _local_cache_wav(clean_text: str) -> Path:
    """Cache WAV path for Rex's OWN voice on the local backend (impersonation
    voices are never cached). Keyed on backend + model, distinct from ElevenLabs."""
    return _cache_path(
        clean_text, "local:rex",
        str(getattr(config, "LOCAL_TTS_MODEL_ID", "qwen-tts")),
    ).with_suffix(".wav")


def is_cached(
    text: str,
    voice_settings: Optional[dict] = None,
    emotion: str = "neutral",
    comedy_mode: Optional[str] = None,
) -> bool:
    """Return True if this text already has cached audio for the active voice.

    `emotion` (and `comedy_mode`) must match what the line will be spoken with so
    the cache key lines up with the expressive voice settings and any v3 audio tag
    used at playback time.
    """
    if not text or not text.strip():
        return False
    # In local mode Rex's takes cache under a different (backend) key; audio tags
    # are stripped for Qwen, so the key is the plain normalized text. When the local
    # cache is disabled (the default), nothing is ever cached → always "not cached".
    if _use_local_backend():
        if not bool(getattr(config, "LOCAL_TTS_CACHE_ENABLED", False)):
            return False
        return _local_cache_wav(strip_audio_tags(_normalize_for_speech(text))).exists()
    spoken_text = _normalize_for_speech(text)
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    synth_text, voice_settings = _apply_audio_tags(
        spoken_text, emotion, comedy_mode, voice_settings
    )
    cache_file = _cache_path(
        synth_text,
        config.ELEVENLABS_VOICE_ID,
        config.TTS_MODEL_ID,
        voice_settings,
    )
    return cache_file.exists() or cache_file.with_suffix(".wav").exists()


def ensure_cached(
    text: str,
    voice_settings: Optional[dict] = None,
    emotion: str = "neutral",
    comedy_mode: Optional[str] = None,
    suppress_audio_tag: bool = False,
    previous_text: Optional[str] = None,
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
    # Local mode: prefill via the on-device engine (never touch ElevenLabs).
    if _use_local_backend():
        return _ensure_cached_local(strip_audio_tags(_normalize_for_speech(text)))
    spoken_text = _normalize_for_speech(text)
    voice_settings = _resolve_voice_settings(emotion, voice_settings)
    synth_text, voice_settings = _apply_audio_tags(
        spoken_text, emotion, comedy_mode, voice_settings, suppress_leading=suppress_audio_tag
    )
    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    cache_file = _cache_path(synth_text, voice_id, model_id, voice_settings, previous_text)
    if cache_file.exists():
        return True

    logger.info("[tts] cache prefill miss — calling ElevenLabs API for %r", synth_text)
    audio_bytes = _fetch_from_api(synth_text, voice_id, model_id, voice_settings, previous_text)
    if not audio_bytes:
        return False
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(audio_bytes)
    logger.info("[tts] prefilled cache: %s", cache_file.name)
    return True


def _ensure_cached_local(clean_text: str) -> bool:
    """Prefill Rex's local-voice WAV cache for `clean_text` (already tag-stripped)
    without playing. Used at startup so the first --local-tts line is instant. No-op
    when the local cache is disabled (the default) — every line synthesizes fresh."""
    if not bool(getattr(config, "LOCAL_TTS_CACHE_ENABLED", False)):
        return False
    wav = _local_cache_wav(clean_text)
    if wav.exists():
        return True
    try:
        from audio import local_tts
        ref = local_tts.rex_voice_ref()
        if ref is None or not local_tts.is_available():
            return False
        audio, sr = local_tts.synthesize(clean_text, ref)
        if audio is None or not len(audio):
            return False
        audio = _trim_trailing_silence(audio, sr)
        import soundfile as sf
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), audio, sr)
        logger.info("[tts] prefilled local cache: %s", wav.name)
        return True
    except Exception as exc:
        logger.debug("[tts] local cache prefill failed: %s", exc)
        return False


_el_client = None
_el_client_lock = threading.Lock()


def _get_el_client():
    """One shared ElevenLabs client for the process. A fresh client per call paid a
    full TLS handshake on EVERY sentence (part of the ~1.0-1.4s per-generation cost
    and most of the 2.7-5.6s first-turn outlier, measured 2026-07-06). The SDK's
    underlying httpx pool keeps the connection alive between turns."""
    global _el_client
    if _el_client is not None:
        return _el_client
    with _el_client_lock:
        if _el_client is None:
            import apikeys
            from elevenlabs import ElevenLabs
            # The SDK default is 240 s — not a budget for a conversational line.
            # See config.TTS_API_TIMEOUT_SECS for the field incident.
            _el_client = ElevenLabs(
                api_key=apikeys.ELEVENLABS_API_KEY,
                timeout=float(getattr(config, "TTS_API_TIMEOUT_SECS", 8.0)),
            )
    return _el_client


def warmup_api() -> bool:
    """Open the ElevenLabs TLS connection at startup so the session's FIRST spoken
    reply doesn't pay the cold handshake. Mirrors action_router.warmup() for the
    OpenAI pool. The key is TTS-only scoped, so the metadata probe 401s — that is
    FINE: an HTTP error still means a completed round-trip over a now-open pooled
    connection (the whole point); only a network-level failure counts as cold."""
    try:
        client = _get_el_client()
        try:
            client.user.get()   # any HTTP response = connection opened
        except Exception as exc:
            if type(exc).__name__ != "ApiError":
                raise             # network-level problem — genuinely cold
        logger.info("[tts] ElevenLabs connection warmed")
        _note_api_success()      # a completed round-trip clears the fallback breaker
        return True
    except Exception as exc:
        logger.debug("[tts] ElevenLabs warmup failed (non-fatal): %s", exc)
        return False


def _fetch_from_api(
    text: str,
    voice_id: str,
    model_id: str,
    voice_settings: Optional[dict] = None,
    previous_text: Optional[str] = None,
) -> Optional[bytes]:
    try:
        from elevenlabs import VoiceSettings

        client = _get_el_client()
        kwargs = {
            "voice_id": voice_id,
            "text": text,
            "model_id": model_id,
        }
        if voice_settings:
            kwargs["voice_settings"] = VoiceSettings(
                **{k: v for k, v in voice_settings.items() if v is not None}
            )
        seed = _v3_seed(model_id)
        if seed is not None:
            kwargs["seed"] = seed   # reproducible identical requests + deterministic cache
        prev = _stitch_previous_text(previous_text, model_id)
        if prev:
            kwargs["previous_text"] = prev   # stitch: continue one performance across the reply
        chunks = client.text_to_speech.stream(**kwargs)
        data = b"".join(chunks)
        if not data:
            logger.error("[tts] ElevenLabs returned empty audio stream")
            _note_api_failure()
            return None
        _note_api_success()
        return data
    except Exception as exc:
        logger.error("[tts] ElevenLabs API error: %s", exc)
        _note_api_failure()
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
