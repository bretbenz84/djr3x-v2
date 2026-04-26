"""
audio/prosody.py — Local voice-prosody analysis for emotional intelligence.

Pure numpy + scipy. No model, no API call, no extra dependencies. Computes a
small set of acoustic features from the same audio array Whisper just
transcribed and reduces them to:
  - a (valence, arousal) estimate
  - a short human-readable tag the LLM can use as evidence

Features extracted:
  - RMS energy (mean over voiced frames)
  - Voiced ratio (fraction of frames above an energy floor)
  - Pitch (autocorrelation-based F0, mean + std over voiced frames)
  - Speech rate (words per second, from the supplied transcript)

Mapping rules are deliberately conservative — voice features alone are noisy.
The fusion in intelligence/empathy.py treats this as one signal among many.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import config

_log = logging.getLogger(__name__)


# Frame configuration — ~32ms frames at 16kHz, 50% overlap.
_FRAME_MS = 32
_HOP_MS = 16

# Pitch search range for human voice (Hz).
_PITCH_MIN_HZ = 80
_PITCH_MAX_HZ = 400

# Voiced frame gate: frames whose RMS is below this fraction of the segment
# peak RMS are treated as silence/unvoiced.
_VOICED_FLOOR_FRAC = 0.15

# Below this voiced-frame count we don't trust the analysis.
_MIN_VOICED_FRAMES = 6


def _frame_signal(x: np.ndarray, sr: int) -> np.ndarray:
    """Split x into overlapping frames. Returns shape (n_frames, frame_len)."""
    frame_len = max(1, int(sr * _FRAME_MS / 1000))
    hop = max(1, int(sr * _HOP_MS / 1000))
    if x.size < frame_len:
        return np.empty((0, frame_len), dtype=x.dtype)
    n = 1 + (x.size - frame_len) // hop
    idx = (
        np.arange(frame_len)[None, :]
        + (np.arange(n) * hop)[:, None]
    )
    return x[idx]


def _autocorr_pitch(frame: np.ndarray, sr: int) -> Optional[float]:
    """Estimate F0 via autocorrelation peak inside the human-voice range.

    Returns None if no clear pitched component is detected.
    """
    if frame.size < 64:
        return None
    f = frame - frame.mean()
    if not np.any(f):
        return None
    # Hann window reduces edge bias.
    f = f * np.hanning(f.size)
    ac = np.correlate(f, f, mode="full")
    ac = ac[ac.size // 2:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]

    min_lag = max(1, int(sr / _PITCH_MAX_HZ))
    max_lag = min(ac.size - 1, int(sr / _PITCH_MIN_HZ))
    if max_lag <= min_lag:
        return None
    region = ac[min_lag:max_lag]
    if region.size == 0:
        return None
    peak = int(np.argmax(region)) + min_lag
    # Reject weak peaks — likely no voicing in this frame.
    if ac[peak] < 0.30:
        return None
    return float(sr) / float(peak)


def analyze(
    audio_array: np.ndarray,
    sample_rate: Optional[int] = None,
    transcript_text: Optional[str] = None,
) -> Optional[dict]:
    """Compute prosody features. Returns dict or None if input too short."""
    if audio_array is None or audio_array.size == 0:
        return None
    sr = int(sample_rate or config.AUDIO_SAMPLE_RATE)
    x = audio_array.astype(np.float32)
    # Normalize from int16 if needed.
    if x.size and np.max(np.abs(x)) > 1.5:
        x = x / 32768.0

    duration_s = x.size / float(sr)
    if duration_s < 0.25:
        return None

    frames = _frame_signal(x, sr)
    if frames.shape[0] == 0:
        return None

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    if peak_rms <= 0.0:
        return None
    voiced_mask = rms >= (peak_rms * _VOICED_FLOOR_FRAC)
    voiced_count = int(np.sum(voiced_mask))
    voiced_ratio = voiced_count / float(rms.size)

    if voiced_count < _MIN_VOICED_FRAMES:
        return {
            "duration_s": duration_s,
            "rms_mean": float(np.mean(rms)),
            "voiced_ratio": voiced_ratio,
            "pitch_mean_hz": None,
            "pitch_std_hz": None,
            "speech_rate_wps": None,
            "valence": 0.0,
            "arousal": 0.0,
            "confidence": 0.1,
            "tag": "voice: too short / too quiet to read prosody",
        }

    pitches = []
    for i, is_v in enumerate(voiced_mask):
        if not is_v:
            continue
        f0 = _autocorr_pitch(frames[i], sr)
        if f0 is not None:
            pitches.append(f0)

    pitch_mean = float(np.mean(pitches)) if pitches else None
    pitch_std = float(np.std(pitches)) if len(pitches) >= 2 else None
    rms_mean_voiced = float(np.mean(rms[voiced_mask]))

    speech_rate_wps = None
    if transcript_text:
        words = [w for w in transcript_text.split() if w.strip()]
        if words and duration_s > 0:
            speech_rate_wps = len(words) / duration_s

    valence, arousal, tag = _map_to_affect(
        rms_mean_voiced=rms_mean_voiced,
        peak_rms=peak_rms,
        voiced_ratio=voiced_ratio,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        speech_rate_wps=speech_rate_wps,
    )

    confidence = _confidence_from_coverage(
        voiced_count=voiced_count,
        n_pitches=len(pitches),
        duration_s=duration_s,
    )

    return {
        "duration_s": round(duration_s, 2),
        "rms_mean": round(rms_mean_voiced, 4),
        "voiced_ratio": round(voiced_ratio, 2),
        "pitch_mean_hz": round(pitch_mean, 1) if pitch_mean else None,
        "pitch_std_hz": round(pitch_std, 1) if pitch_std else None,
        "speech_rate_wps": round(speech_rate_wps, 2) if speech_rate_wps else None,
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "confidence": round(confidence, 2),
        "tag": tag,
    }


def _map_to_affect(
    rms_mean_voiced: float,
    peak_rms: float,
    voiced_ratio: float,
    pitch_mean: Optional[float],
    pitch_std: Optional[float],
    speech_rate_wps: Optional[float],
) -> tuple[float, float, str]:
    """Reduce features to (valence, arousal, tag)."""
    cues = []

    # Arousal cues
    arousal = 0.0
    if speech_rate_wps is not None:
        if speech_rate_wps >= 3.5:
            arousal += 0.4
            cues.append("fast pace")
        elif speech_rate_wps <= 1.5:
            arousal -= 0.4
            cues.append("slow pace")
    if pitch_std is not None and pitch_mean:
        std_ratio = pitch_std / max(1.0, pitch_mean)
        if std_ratio >= 0.18:
            arousal += 0.25
            cues.append("variable pitch")
        elif std_ratio <= 0.06:
            arousal -= 0.25
            cues.append("flat pitch")
    if rms_mean_voiced >= 0.10:
        arousal += 0.2
        cues.append("loud")
    elif rms_mean_voiced <= 0.02:
        arousal -= 0.2
        cues.append("quiet")

    # Valence cues — much weaker on prosody alone
    valence = 0.0
    if pitch_mean is not None:
        if pitch_mean >= 220:
            valence += 0.15
        elif pitch_mean <= 110:
            valence -= 0.15
    if voiced_ratio <= 0.35:
        valence -= 0.15
        cues.append("long pauses")
    if rms_mean_voiced <= 0.02 and (
        speech_rate_wps is not None and speech_rate_wps <= 1.8
    ):
        # Quiet + slow is a strong sad/withdrawn signal
        valence -= 0.25

    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    tag = "voice: " + (", ".join(cues) if cues else "neutral prosody")
    return valence, arousal, tag


def _confidence_from_coverage(
    voiced_count: int, n_pitches: int, duration_s: float,
) -> float:
    """How much we trust this analysis based on how much usable signal we got."""
    if duration_s < 0.5:
        return 0.2
    base = min(1.0, voiced_count / 40.0)
    pitch_bonus = min(0.3, n_pitches / 60.0)
    return min(1.0, 0.3 + base * 0.5 + pitch_bonus)
