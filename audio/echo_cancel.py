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

import numpy as np

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_playing = False


# ── Public API ────────────────────────────────────────────────────────────────

def set_playing(is_playing: bool) -> None:
    """Called by TTS and playback modules when audio output starts or stops."""
    global _playing
    with _lock:
        changed = _playing != is_playing
        _playing = is_playing

    if changed:
        if is_playing:
            logger.info("[aec] suppression started — playback active")
        else:
            logger.info("[aec] suppression stopped — playback ended")


def add_reference(audio_array: np.ndarray) -> None:
    """Accept a reference signal from a playback module.

    No-op in the suppression model — retained so callers need no conditional logic
    if a future upgrade wires in true AEC.
    """


def filter(audio_array: np.ndarray) -> np.ndarray:
    """Return audio_array with suppression applied if playback is active."""
    with _lock:
        suppressing = _playing
    if suppressing:
        return audio_array * config.AEC_SUPPRESSION_FACTOR
    return audio_array


def is_suppressed() -> bool:
    """Return True if mic input is currently being suppressed."""
    with _lock:
        return _playing
