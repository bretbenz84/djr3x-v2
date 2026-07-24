"""
Voice Activity Detection using the Silero VAD model.

Safe to import even when torch or the model weights are unavailable — all
functions degrade gracefully (is_speech returns False, get_speech_segments
returns an empty list) and the failure is logged rather than raised.
"""

import logging
import threading

import numpy as np

import config

_log = logging.getLogger(__name__)

_model = None
_get_speech_timestamps = None
_loaded = False
# The Silero model is a single stateful object. The interaction loop streams
# is_speech() from its own thread while the proactive-yield guard may call
# get_speech_segments() from another (consciousness); serialize all model access
# so a concurrent reset/inference can't corrupt the streaming state.
_lock = threading.Lock()


def _load() -> None:
    global _model, _get_speech_timestamps, _loaded
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        _model = load_silero_vad()
        _get_speech_timestamps = get_speech_timestamps
        _loaded = True
        _log.info("Silero VAD model loaded from installed silero_vad package.")
    except Exception as exc:
        _log.error("Failed to load Silero VAD model — speech detection disabled: %s", exc)
        _loaded = False


_load()


def _threshold() -> float:
    """Effective Silero probability threshold for this machine.

    The ReSpeaker robot listens far-field, where soft-onset phonemes ("wh" in
    "what's") sit below the 0.5 default for the first chunks — the VAD fires
    late and the LEADING words are clipped (see VAD_THRESHOLD_AEC in config).
    hardware_aec.is_active() is the single gate for robot-only audio behavior;
    it caches after first call, so this is cheap on the streaming path.
    """
    try:
        from audio import hardware_aec

        if hardware_aec.is_active():
            return float(getattr(config, "VAD_THRESHOLD_AEC", config.VAD_THRESHOLD))
    except Exception as exc:
        _log.debug("hardware_aec check failed, using base VAD threshold: %s", exc)
    return float(config.VAD_THRESHOLD)


# ── Public API ────────────────────────────────────────────────────────────────

def is_speech(audio_chunk: np.ndarray) -> bool:
    """Return True if `audio_chunk` contains speech above VAD_THRESHOLD.

    Intended for streaming use on short chunks (~32 ms at 16 kHz). The model
    maintains internal state across consecutive calls so it uses temporal context
    when deciding whether a chunk is speech.
    """
    if not _loaded:
        return False
    try:
        import torch

        tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        with _lock, torch.no_grad():
            prob: float = _model(tensor, config.AUDIO_SAMPLE_RATE).item()
        return prob >= _threshold()
    except Exception as exc:
        _log.warning("VAD inference error: %s", exc)
        return False


def reset_state() -> None:
    """Reset the streaming VAD context before a fresh listening window."""
    if not _loaded:
        return
    try:
        with _lock:
            _model.reset_states()
    except Exception as exc:
        _log.debug("VAD reset_state error: %s", exc)


def get_speech_segments(audio_array: np.ndarray) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) pairs for every speech region in `audio_array`.

    Resets the model's internal state before processing so each call to this
    function is independent of the streaming is_speech state.
    """
    if not _loaded:
        return []
    try:
        import torch

        tensor = torch.from_numpy(audio_array.astype(np.float32))
        with _lock:
            # Reset stateful context so this batch call is self-contained, then
            # restore it after so a concurrent streaming is_speech() isn't left
            # mid-reset.
            _model.reset_states()
            segments = _get_speech_timestamps(
                tensor,
                _model,
                threshold=_threshold(),
                sampling_rate=config.AUDIO_SAMPLE_RATE,
            )
            _model.reset_states()
        return [
            (seg["start"] / config.AUDIO_SAMPLE_RATE, seg["end"] / config.AUDIO_SAMPLE_RATE)
            for seg in segments
        ]
    except Exception as exc:
        _log.warning("VAD segmentation error: %s", exc)
        return []
