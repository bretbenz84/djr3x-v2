"""
Continuous microphone capture via sounddevice.

Opens the mic once and never closes it. Audio is written into a rolling circular
buffer by a non-blocking sounddevice callback. Callers read from the buffer via
get_audio_chunk() or get_full_buffer().

If AUDIO_DEVICE_INDEX is not set in .env the module initialises as a no-op and
all read functions return empty arrays.
"""

import logging
import math
from collections import deque

import numpy as np

import config
from utils.config_loader import AUDIO_DEVICE_INDEX

_log = logging.getLogger(__name__)

# Fixed frames per callback invocation. 512 samples at 16 kHz = 32 ms, matching
# Silero VAD's preferred chunk size and keeping the callback very fast.
_BLOCKSIZE = 512

# Number of chunks the deque holds so total capacity == AUDIO_BUFFER_SECONDS.
_MAXLEN: int = math.ceil(config.AUDIO_SAMPLE_RATE * config.AUDIO_BUFFER_SECONDS / _BLOCKSIZE)

_buf: deque = deque(maxlen=_MAXLEN)
_stream = None  # sounddevice.InputStream, or None when disabled
_input_channels: int = 1  # actual device channels; set during start()


# ── Callback ──────────────────────────────────────────────────────────────────

def _callback(indata, frames, time_info, status):  # noqa: ANN001
    if status:
        _log.warning("sounddevice status: %s", status)
    # Non-blocking: only append, never block or acquire locks.
    # CPython's GIL makes deque.append thread-safe for this single-writer pattern.
    # Mix stereo → mono by averaging channels so both capsules contribute.
    if _input_channels > 1:
        _buf.append(indata.mean(axis=1).copy())
    else:
        _buf.append(indata[:, 0].copy())


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def start() -> None:
    """Open the microphone and begin filling the rolling buffer."""
    global _stream, _input_channels

    if AUDIO_DEVICE_INDEX is None:
        _log.warning(
            "AUDIO_DEVICE_INDEX not set in .env — audio stream disabled. "
            "Wake word, VAD, transcription, and speaker ID will not function."
        )
        return

    if _stream is not None and _stream.active:
        return

    _input_channels = getattr(config, "AUDIO_INPUT_CHANNELS", config.AUDIO_CHANNELS)

    try:
        import sounddevice as sd

        _stream = sd.InputStream(
            device=AUDIO_DEVICE_INDEX,
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=_input_channels,
            dtype="float32",
            blocksize=_BLOCKSIZE,
            callback=_callback,
        )
        _stream.start()
        _log.info(
            "Audio stream started — device %d, %d Hz, %d-ch input → mono, %ds buffer.",
            AUDIO_DEVICE_INDEX,
            config.AUDIO_SAMPLE_RATE,
            _input_channels,
            config.AUDIO_BUFFER_SECONDS,
        )
    except Exception as exc:
        _log.error("Failed to open audio stream: %s", exc)
        _stream = None


def stop() -> None:
    """Stop and close the microphone stream."""
    global _stream

    if _stream is None:
        return
    try:
        _stream.stop()
        _stream.close()
    except Exception as exc:
        _log.warning("Error closing audio stream: %s", exc)
    finally:
        _stream = None
        _log.info("Audio stream stopped.")


# ── Buffer reads ──────────────────────────────────────────────────────────────

def get_full_buffer() -> np.ndarray:
    """Return a copy of all audio currently in the rolling buffer as a 1-D float32 array."""
    chunks = list(_buf)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def get_audio_chunk(seconds: float) -> np.ndarray:
    """Return the last `seconds` of audio from the buffer as a 1-D float32 array.

    If the buffer contains less than `seconds` of audio, all available audio is
    returned rather than padding with silence.
    """
    samples_needed = int(seconds * config.AUDIO_SAMPLE_RATE)
    audio = get_full_buffer()
    if len(audio) >= samples_needed:
        return audio[-samples_needed:]
    return audio


def flush() -> None:
    """Discard all audio currently in the rolling buffer.

    Called after TTS playback to prevent Rex's own voice tail from being
    picked up as speech onset on the next listening pass.
    """
    _buf.clear()
