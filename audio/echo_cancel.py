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

import numpy as np

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_playing = False
_suppress_until: float = 0.0  # monotonic deadline for post-playback tail suppression
_sequence_active: bool = False  # when True, set_playing(False) is deferred until end_sequence()


# ── Public API ────────────────────────────────────────────────────────────────

def start_sequence() -> None:
    """Begin a multi-segment playback sequence.

    Suppression is activated immediately and held active across all segments until
    end_sequence() is called. set_playing(False) calls from individual TTS segments
    are ignored — no mid-sequence flush or tail suppression fires.
    """
    global _playing, _suppress_until, _sequence_active
    with _lock:
        _sequence_active = True
        _playing = True
        _suppress_until = 0.0
    logger.info("[aec] sequence started — suppression held across segments")


def end_sequence() -> None:
    """End the playback sequence and apply normal post-playback tail suppression."""
    global _playing, _suppress_until, _sequence_active
    with _lock:
        _sequence_active = False
        _playing = False
        _suppress_until = time.monotonic() + config.POST_PLAYBACK_SUPPRESSION_SECS
        from audio import stream as _stream
        _stream.flush()
    logger.info(
        "[aec] sequence ended — suppression stopped, %.1fs tail active",
        config.POST_PLAYBACK_SUPPRESSION_SECS,
    )


def set_playing(is_playing: bool) -> None:
    """Called by TTS and playback modules when audio output starts or stops."""
    global _playing, _suppress_until
    with _lock:
        if not is_playing and _sequence_active:
            # Mid-sequence: suppress the turn-off so the next segment sees no gap.
            return
        changed = _playing != is_playing
        _playing = is_playing
        if not is_playing:
            # Keep suppression active for a short tail so any of Rex's voice
            # that has already bled into the mic buffer is still attenuated.
            _suppress_until = time.monotonic() + config.POST_PLAYBACK_SUPPRESSION_SECS
            # Drop accumulated mic audio so Whisper never sees Rex's own voice.
            from audio import stream as _stream
            _stream.flush()
        else:
            # Playback starting — cancel any leftover tail from a previous run.
            _suppress_until = 0.0

    if changed:
        if is_playing:
            logger.info("[aec] suppression started — playback active")
        else:
            logger.info(
                "[aec] suppression stopped — playback ended, %.1fs tail active",
                config.POST_PLAYBACK_SUPPRESSION_SECS,
            )


def add_reference(audio_array: np.ndarray) -> None:
    """Accept a reference signal from a playback module.

    No-op in the suppression model — retained so callers need no conditional logic
    if a future upgrade wires in true AEC.
    """


def filter(audio_array: np.ndarray) -> np.ndarray:
    """Return audio_array with suppression applied if playback is active or in tail."""
    with _lock:
        suppressing = _playing or time.monotonic() < _suppress_until
    if suppressing:
        return audio_array * config.AEC_SUPPRESSION_FACTOR
    return audio_array


def is_suppressed() -> bool:
    """Return True if mic input is currently being suppressed (including tail)."""
    with _lock:
        return _playing or time.monotonic() < _suppress_until
