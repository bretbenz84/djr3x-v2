"""Software acoustic echo suppression so the wake word can be heard over Rex.

Why this exists
───────────────
Rex's own TTS/DJ playback bleeds through the speakers into the mic and acoustically
MASKS a spoken wake word: in live runs "hey rex" scored ~0 while Rex talked and 0.99
the instant he stopped. The ReSpeaker Lite has hardware AEC, but it only works when
playback is routed through the ReSpeaker itself — not possible in this robot's wiring
— so we cancel in software on the host instead.

Approach (and why this one)
───────────────────────────
We know EXACTLY what Rex is playing (the digital signal handed to sounddevice), so we
use it as the echo reference. The mic (ReSpeaker, 16 kHz) and the output device run on
INDEPENDENT clocks that drift, which defeats a sample-locked adaptive filter (NLMS).
So instead of cancelling sample-by-sample we do reference-based SPECTRAL SUPPRESSION:
  1. Capture each played buffer (push_reference), resample to 16 kHz, and place it on a
     monotonic-clock timeline spanning the real time it will play.
  2. For each mic chunk, read the time-aligned reference and refine the residual lag by
     cross-correlation (re-estimated continuously, so slow clock drift is tracked).
  3. Learn the per-frequency echo gain (mic/ref) during echo-dominant frames and
     spectrally subtract it from the mic magnitude, keeping the mic phase. Double-talk
     (the user speaking) freezes the gain so we don't learn the user's voice as echo.

This is magnitude-domain, so it tolerates phase error / drift far better than NLMS, at
the cost of some artifacts — which the wake-word model is robust to.

Safety
──────
process() is a NO-OP passthrough whenever Rex isn't playing (no recent reference
energy) or anything fails, so it can never degrade normal wake detection while Rex is
quiet. It only ever touches the during-playback case, which is otherwise fully masked.
Gated by config.AEC_SOFTWARE_ENABLED.
"""

import logging
import threading
import time

import numpy as np

import config

logger = logging.getLogger(__name__)

_RATE = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))

# Internal DSP constants (config-overridable).
_REF_BUFFER_SECS = float(getattr(config, "AEC_REF_BUFFER_SECS", 6.0))
_MAX_DELAY_SECS = float(getattr(config, "AEC_MAX_DELAY_SECS", 0.4))   # widest echo path we align to
_DELAY_REFINE_INTERVAL_SECS = float(getattr(config, "AEC_DELAY_REFINE_INTERVAL_SECS", 0.25))
_OVERSUBTRACTION = float(getattr(config, "AEC_OVERSUBTRACTION", 1.6))  # >1 = subtract a bit extra
_SPECTRAL_FLOOR = float(getattr(config, "AEC_SPECTRAL_FLOOR", 0.10))   # never null a bin fully
_GAIN_EMA = float(getattr(config, "AEC_GAIN_EMA", 0.15))               # echo-gain learning rate
_DOUBLETALK_RATIO = float(getattr(config, "AEC_DOUBLETALK_RATIO", 2.5))  # mic≫echo ⇒ freeze learning
_REF_ACTIVE_RMS = float(getattr(config, "AEC_REF_ACTIVE_RMS", 0.0015))  # ref louder than this ⇒ "Rex playing"
_DIAG_INTERVAL_SECS = float(getattr(config, "AEC_DIAG_INTERVAL_SECS", 2.0))


def _clock() -> float:
    return time.monotonic()


class _RefTimeline:
    """Monotonic-clock-indexed 16 kHz mono ring of what Rex has played.

    Each pushed buffer is laid down across the real wall-clock span it will play
    ([push_time, push_time + duration]), with silence filling any gaps, so a later
    read at a given clock time returns what the speaker was actually emitting then.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap = max(1, int(_REF_BUFFER_SECS * _RATE))
        self._buf = np.zeros(self._cap, dtype=np.float32)
        # Wall-clock time corresponding to the sample just PAST the end of the buffer.
        self._end_time: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._end_time = None

    def push(self, samples: np.ndarray, play_start: float) -> None:
        n = int(samples.shape[0])
        if n <= 0:
            return
        with self._lock:
            if self._end_time is None:
                self._end_time = play_start
            gap = play_start - self._end_time
            if gap > 0:
                pad = min(self._cap, int(gap * _RATE))
                if pad:
                    self._buf = np.roll(self._buf, -pad)
                    self._buf[-pad:] = 0.0
                self._end_time = play_start
            # Append the clip (rolling). Clips longer than the buffer keep only the tail.
            if n >= self._cap:
                self._buf[:] = samples[-self._cap:]
            else:
                self._buf = np.roll(self._buf, -n)
                self._buf[-n:] = samples
            self._end_time += n / _RATE

    def read(self, end_time: float, count: int) -> np.ndarray:
        """Return `count` reference samples ending at wall-clock `end_time`."""
        out = np.zeros(count, dtype=np.float32)
        with self._lock:
            if self._end_time is None:
                return out
            # Index in _buf of the sample at `end_time` (end of the buffer == _end_time).
            end_idx = self._cap - int(round((self._end_time - end_time) * _RATE))
            start_idx = end_idx - count
            if end_idx <= 0 or start_idx >= self._cap:
                return out
            src_lo = max(0, start_idx)
            src_hi = min(self._cap, end_idx)
            dst_lo = src_lo - start_idx
            out[dst_lo:dst_lo + (src_hi - src_lo)] = self._buf[src_lo:src_hi]
        return out


def _resample_to_rate(audio: np.ndarray, sr_in: int) -> np.ndarray:
    """Mix to mono float32 and resample to the mic rate. Uses soxr, else scipy."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr_in == _RATE:
        return a
    try:
        import soxr
        return np.asarray(soxr.resample(a, sr_in, _RATE), dtype=np.float32)
    except Exception:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr_in), int(_RATE))
            return np.asarray(resample_poly(a, _RATE // g, sr_in // g), dtype=np.float32)
        except Exception as exc:
            logger.debug("[aec] resample failed (%s); dropping reference frame", exc)
            return np.zeros(0, dtype=np.float32)


class _SoftwareAEC:
    def __init__(self) -> None:
        self._timeline = _RefTimeline()
        self._echo_gain: np.ndarray | None = None   # per-rfft-bin learned echo magnitude gain
        self._delay_samples = int(0.05 * _RATE)      # current mic↔ref lag estimate
        self._last_refine_at = 0.0
        self._last_diag_at = 0.0
        self._mic_hist = np.zeros(int(0.5 * _RATE), dtype=np.float32)
        self._erle_ema = 0.0
        self._lock = threading.Lock()

    # ── Reference capture (called from the playback hook) ──────────────────────
    def push_reference(self, audio: np.ndarray, samplerate: int) -> None:
        if not _enabled():
            return
        try:
            res = _resample_to_rate(audio, int(samplerate))
            if res.size:
                self._timeline.push(res, _clock())
        except Exception as exc:
            logger.debug("[aec] push_reference failed: %s", exc)

    def reset(self) -> None:
        self._timeline.reset()
        with self._lock:
            self._echo_gain = None

    # ── Delay tracking ─────────────────────────────────────────────────────────
    def _refine_delay(self, mic_recent: np.ndarray, now: float) -> None:
        max_delay = int(_MAX_DELAY_SECS * _RATE)
        # Reference window covering [now - max_delay - len(mic), now].
        ref_win = self._timeline.read(now, mic_recent.size + max_delay)
        if ref_win.size < mic_recent.size + 8 or float(np.sqrt(np.mean(ref_win ** 2))) < _REF_ACTIVE_RMS:
            return
        # The mic aligns to some lag d in [0, max_delay] into the past of ref_win's end.
        m = mic_recent - np.mean(mic_recent)
        r = ref_win - np.mean(ref_win)
        corr = np.correlate(r, m, mode="valid")  # length max_delay+1
        if corr.size == 0:
            return
        best = int(np.argmax(np.abs(corr)))
        # corr index 0 ⇒ mic aligns to the OLDEST slice ⇒ largest delay; flip to delay-from-now.
        delay = max(0, (corr.size - 1) - best)
        with self._lock:
            self._delay_samples = int(0.5 * self._delay_samples + 0.5 * delay)

    # ── Per-chunk processing ─────────────────────────────────────────────────────
    def process(self, mic_chunk: np.ndarray) -> np.ndarray:
        if not _enabled():
            return mic_chunk
        try:
            return self._process(np.asarray(mic_chunk, dtype=np.float32))
        except Exception as exc:
            logger.debug("[aec] process failed; passthrough: %s", exc)
            return mic_chunk

    def _process(self, mic: np.ndarray) -> np.ndarray:
        now = _clock()
        n = mic.size
        # Keep a short mic history for delay refinement.
        self._mic_hist = np.concatenate((self._mic_hist, mic))[-int(0.5 * _RATE):]

        with self._lock:
            delay = self._delay_samples
        ref = self._timeline.read(now - delay / _RATE, n)

        ref_rms = float(np.sqrt(np.mean(ref ** 2))) if ref.size else 0.0
        if ref_rms < _REF_ACTIVE_RMS:
            # Rex isn't playing (or no aligned reference) — passthrough, untouched.
            return mic

        if now - self._last_refine_at >= _DELAY_REFINE_INTERVAL_SECS:
            self._last_refine_at = now
            self._refine_delay(self._mic_hist, now)

        M = np.fft.rfft(mic)
        R = np.fft.rfft(ref)
        mic_mag = np.abs(M)
        ref_mag = np.abs(R)
        eps = 1e-9

        if self._echo_gain is None or self._echo_gain.shape != mic_mag.shape:
            self._echo_gain = (mic_mag / (ref_mag + eps)).astype(np.float32)

        # Double-talk check: if the mic carries far more energy than the predicted
        # echo, the user is talking — freeze gain learning so we don't cancel them.
        predicted = self._echo_gain * ref_mag
        mic_e = float(np.sum(mic_mag ** 2)) + eps
        pred_e = float(np.sum(predicted ** 2)) + eps
        double_talk = mic_e > _DOUBLETALK_RATIO * pred_e
        if not double_talk:
            inst = mic_mag / (ref_mag + eps)
            self._echo_gain = ((1 - _GAIN_EMA) * self._echo_gain + _GAIN_EMA * inst).astype(np.float32)

        echo_est = self._echo_gain * ref_mag
        gain = (mic_mag - _OVERSUBTRACTION * echo_est) / (mic_mag + eps)
        gain = np.clip(gain, _SPECTRAL_FLOOR, 1.0)
        clean = np.fft.irfft(gain * M, n=n).astype(np.float32)

        # ERLE diagnostic (echo reduction in dB), rate-limited.
        out_e = float(np.sum(clean ** 2)) + eps
        erle = 10.0 * np.log10(mic_e / out_e)
        self._erle_ema = 0.9 * self._erle_ema + 0.1 * erle
        if now - self._last_diag_at >= _DIAG_INTERVAL_SECS:
            self._last_diag_at = now
            logger.info(
                "[aec] active — ERLE=%.1f dB delay=%.0fms double_talk=%s ref_rms=%.4f",
                self._erle_ema, 1000.0 * delay / _RATE, double_talk, ref_rms,
            )
        return clean


_aec = _SoftwareAEC()


def _enabled() -> bool:
    return bool(getattr(config, "AEC_SOFTWARE_ENABLED", True))


# ── Public API ────────────────────────────────────────────────────────────────
def push_reference(audio: np.ndarray, samplerate: int) -> None:
    """Feed a buffer Rex is about to play as the echo reference (from the playback hook)."""
    _aec.push_reference(audio, samplerate)


def process(mic_chunk: np.ndarray) -> np.ndarray:
    """Return the mic chunk with Rex's echo suppressed (passthrough when he's quiet)."""
    return _aec.process(mic_chunk)


def reset() -> None:
    """Clear reference + learned echo state (e.g. on sleep/shutdown)."""
    _aec.reset()
