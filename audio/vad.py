"""
Voice Activity Detection using the Silero VAD model.

Safe to import even when torch or the model weights are unavailable — all
functions degrade gracefully (is_speech returns False, get_speech_segments
returns an empty list) and the failure is logged rather than raised.
"""

import logging

import numpy as np

import config

_log = logging.getLogger(__name__)

_model = None
_get_speech_timestamps = None
_loaded = False


def _load() -> None:
    global _model, _get_speech_timestamps, _loaded
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils
        _model = model
        _get_speech_timestamps = get_speech_timestamps
        _loaded = True
        _log.info("Silero VAD model loaded.")
    except Exception as exc:
        _log.error("Failed to load Silero VAD model — speech detection disabled: %s", exc)
        _loaded = False


_load()


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
        with torch.no_grad():
            prob: float = _model(tensor, config.AUDIO_SAMPLE_RATE).item()
        return prob >= config.VAD_THRESHOLD
    except Exception as exc:
        _log.warning("VAD inference error: %s", exc)
        return False


def get_speech_segments(audio_array: np.ndarray) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) pairs for every speech region in `audio_array`.

    Resets the model's internal state before processing so each call to this
    function is independent of the streaming is_speech state.
    """
    if not _loaded:
        return []
    try:
        import torch

        # Reset stateful context so this batch call is self-contained.
        _model.reset_states()

        tensor = torch.from_numpy(audio_array.astype(np.float32))
        segments = _get_speech_timestamps(
            tensor,
            _model,
            threshold=config.VAD_THRESHOLD,
            sampling_rate=config.AUDIO_SAMPLE_RATE,
        )
        return [
            (seg["start"] / config.AUDIO_SAMPLE_RATE, seg["end"] / config.AUDIO_SAMPLE_RATE)
            for seg in segments
        ]
    except Exception as exc:
        _log.warning("VAD segmentation error: %s", exc)
        return []
