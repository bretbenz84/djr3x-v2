"""
Simple playback-suppression AEC.

True acoustic echo cancellation requires sample-accurate latency matching between
the reference signal and the microphone input, which is fragile without dedicated
hardware. The approach here is intentionally simpler: when Rex is playing audio,
mic input is attenuated by AEC_SUPPRESSION_FACTOR so his own voice cannot bleed
into transcription. The reference buffer (add_reference) is accepted but unused —
it exists so TTS/playback modules can call it without caring whether full AEC is
wired up.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_playing = False
_suppress_until: float = 0.0  # monotonic deadline for post-playback tail suppression
_sequence_active: bool = False  # when True, set_playing(False) is deferred until end_sequence()
_playback_canceled: bool = False  # set by request_cancel() right before sd.stop()
# When audio output ACTUALLY last stopped (monotonic), stamped even when the
# sequence hold swallows the set_playing(False) itself. The post-TTS capture floor
# anchors here instead of on the queue callback, which runs 0.5-1.5s later (cache
# save, sequence bookkeeping) — words spoken in that lag were clean, buffered, and
# clipped (field 2026-08-06 00:10: "I know, am I right?" → HEARD "Am I right?").
_last_real_playback_end: float = 0.0
# Deadman for the sequence hold. _sequence_active pins suppression across the gaps
# between a reply's segments, but it is released by the speech queue draining — so
# anything that wedges INSIDE tts holds it open forever. Field 2026-08-20 20:31: an
# ElevenLabs network hang held it 57 s and Rex was deaf to everything, including
# "shut down". These track when the hold last had REAL audio under it.
_segment_active: bool = False        # between set_playing(True) and its (False)
_sequence_idle_since: float = 0.0    # monotonic; 0.0 = a segment is playing
_sequence_idle_released: bool = False


def _sequence_hold_expired_locked() -> bool:
    """True when the sequence hold is open but nothing has actually played for
    AEC_SEQUENCE_IDLE_RELEASE_SECS. Caller must hold _lock."""
    if not _sequence_active or _segment_active or _sequence_idle_since <= 0.0:
        return False
    cap = float(getattr(config, "AEC_SEQUENCE_IDLE_RELEASE_SECS", 25.0) or 0.0)
    if cap <= 0.0:
        return False
    return (time.monotonic() - _sequence_idle_since) > cap


def _suppressing_locked() -> bool:
    """The one place that decides whether the mic is attenuated right now.
    Caller must hold _lock."""
    if _sequence_hold_expired_locked():
        return False
    return _playing or time.monotonic() < _suppress_until


def _warn_if_hold_expired(where: str) -> None:
    """Log the deadman firing once per sequence — silence here would make the next
    'why did Rex go deaf' investigation as hard as the last one."""
    global _sequence_idle_released
    with _lock:
        if not _sequence_hold_expired_locked() or _sequence_idle_released:
            return
        _sequence_idle_released = True
        held = time.monotonic() - _sequence_idle_since
    logger.warning(
        "[aec] sequence hold open %.1fs with no audio under it — releasing "
        "suppression so the mic is not deaf (%s). Something upstream of playback "
        "is wedged; the sequence itself stays open.",
        held, where,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def start_sequence() -> None:
    """Begin a multi-segment playback sequence.

    Suppression is activated immediately and held active across all segments until
    end_sequence() is called. set_playing(False) calls from individual TTS segments
    are ignored — no mid-sequence flush or tail suppression fires.
    """
    global _playing, _suppress_until, _sequence_active, _playback_canceled
    global _segment_active, _sequence_idle_since, _sequence_idle_released
    with _lock:
        _sequence_active = True
        _playing = True
        _suppress_until = 0.0
        _playback_canceled = False
        # Nothing is playing YET — the deadman clock starts here, so a turn whose
        # audio never arrives at all is covered, not just one that stalls midway.
        _segment_active = False
        _sequence_idle_since = time.monotonic()
        _sequence_idle_released = False
    logger.info("[aec] sequence started — suppression held across segments")


def end_sequence(flush: bool = True, tail_secs: Optional[float] = None) -> None:
    """End the playback sequence and apply normal post-playback tail suppression."""
    global _playing, _suppress_until, _sequence_active
    global _segment_active, _sequence_idle_since, _sequence_idle_released
    tail = (
        config.POST_PLAYBACK_SUPPRESSION_SECS
        if tail_secs is None
        else max(0.0, float(tail_secs))
    )
    with _lock:
        _sequence_active = False
        _playing = False
        _segment_active = False
        _sequence_idle_since = 0.0
        _sequence_idle_released = False
        _suppress_until = time.monotonic() + tail
        if flush:
            from audio import stream as _stream
            _stream.flush()
    logger.info(
        "[aec] sequence ended — suppression stopped, %.1fs tail active",
        tail,
    )


def last_playback_ended_at() -> float:
    """Monotonic time audio output ACTUALLY last stopped (0.0 = never).

    Stamped at the real end of every playback segment — including segments whose
    set_playing(False) the sequence hold swallows — so callers can anchor timing
    on the sound instead of on the queue bookkeeping that runs ~a second later."""
    with _lock:
        return _last_real_playback_end


def set_playing(
    is_playing: bool,
    *,
    tail_secs: Optional[float] = None,
    flush: Optional[bool] = None,
) -> None:
    """Called by TTS and playback modules when audio output starts or stops."""
    global _playing, _suppress_until, _playback_canceled, _last_real_playback_end
    global _segment_active, _sequence_idle_since, _sequence_idle_released
    with _lock:
        _segment_active = bool(is_playing)
        if is_playing:
            # Real audio is under the hold again — disarm the deadman and re-arm it
            # for the NEXT gap.
            _sequence_idle_since = 0.0
            _sequence_idle_released = False
        elif _sequence_active:
            _sequence_idle_since = time.monotonic()
        if not is_playing and _sequence_active:
            # Mid-sequence: suppress the turn-off so the next segment sees no gap —
            # but STAMP the real end regardless: if this turns out to be the final
            # segment, this timestamp is when Rex genuinely went quiet, and the
            # capture floor needs it (the sequence callback runs noticeably later).
            _last_real_playback_end = time.monotonic()
            return
        changed = _playing != is_playing
        _playing = is_playing
        if not is_playing:
            _last_real_playback_end = time.monotonic()
            tail = (
                config.POST_PLAYBACK_SUPPRESSION_SECS
                if tail_secs is None
                else max(0.0, float(tail_secs))
            )
            should_flush = True if flush is None else bool(flush)
            # Keep suppression active for a short tail so any of Rex's voice
            # that has already bled into the mic buffer is still attenuated.
            _suppress_until = time.monotonic() + tail
            if should_flush and not _playback_canceled:
                # Drop accumulated mic audio so Whisper never sees Rex's own voice.
                from audio import stream as _stream
                _stream.flush()
        else:
            # Playback starting — cancel any leftover tail from a previous run
            # and clear the cancel flag so this segment's duration guard is armed.
            _suppress_until = 0.0
            _playback_canceled = False

    if changed:
        if is_playing:
            logger.info("[aec] suppression started — playback active")
        else:
            logger.info(
                "[aec] suppression stopped — playback ended, %.1fs tail active",
                (
                    config.POST_PLAYBACK_SUPPRESSION_SECS
                    if tail_secs is None
                    else max(0.0, float(tail_secs))
                ),
            )


def add_reference(audio_array: np.ndarray) -> None:
    """Accept a reference signal from a playback module.

    No-op in the suppression model — retained so callers need no conditional logic
    if a future upgrade wires in true AEC.
    """


def filter(audio_array: np.ndarray) -> np.ndarray:
    """Return audio_array with suppression applied if playback is active or in tail."""
    _warn_if_hold_expired("mic filter")
    with _lock:
        suppressing = _suppressing_locked()
    if suppressing:
        return audio_array * config.AEC_SUPPRESSION_FACTOR
    return audio_array


def is_suppressed() -> bool:
    """Return True if mic input is currently being suppressed (including tail)."""
    _warn_if_hold_expired("is_suppressed")
    with _lock:
        return _suppressing_locked()


def clear_suppression_tail() -> None:
    """Clear post-playback attenuation without flushing the microphone buffer."""
    global _suppress_until
    with _lock:
        if not _playing:
            _suppress_until = 0.0


def request_cancel() -> None:
    """Mark the current playback as deliberately stopped.

    Call this immediately before sd.stop() so tts._play() can distinguish a
    user-initiated interrupt (skip the duration guard) from a CoreAudio glitch
    that returns sd.wait() early (hold suppression for the full audio duration).
    """
    global _playback_canceled
    with _lock:
        _playback_canceled = True


def was_canceled() -> bool:
    """True if request_cancel() has been called since the current playback started."""
    with _lock:
        return _playback_canceled
