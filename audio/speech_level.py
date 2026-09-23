"""Incremental speech leveling, independent of devices, synthesis and config.

Measure K-weighted 400 ms windows on a 20 ms clock, not network chunks. A short pre-roll
sets gain before the first word; a rolling estimate then follows delivery slowly.
The same processor can consume a whole cached take or arbitrary stream chunks.
Raw cache samples stay untouched, so replay never compounds gain.
"""

from collections import deque
import math

import numpy as np
from scipy.signal import resample_poly, sosfilt


def _k_weighting(samplerate: int) -> np.ndarray:
    """BS.1770's two 48 kHz biquads, mapped to this sample rate.

    Coefficients: ITU-R BS.1770, Annex 1, Tables 1 and 2.
    https://www.itu.int/rec/R-REC-BS.1770
    The bilinear substitution keeps the underlying analogue response when the
    cache is 44.1 kHz and the API PCM is 22.05 kHz.
    """
    ratio = samplerate / 48000.0
    u, v = np.array([1 - ratio, 1 + ratio]), np.array([1 + ratio, 1 - ratio])
    def convert(c):
        return c[0] * np.convolve(v, v) + c[1] * np.convolve(u, v) + c[2] * np.convolve(u, u)
    sections = []
    for b, a in [
        ([1.53512485958697, -2.69169618940638, 1.19839281085285],
         [1.0, -1.69065929318241, 0.73248077421585]),
        ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]),
    ]:
        numerator, denominator = convert(b), convert(a)
        sections.append(np.r_[numerator, denominator] / denominator[0])
    return np.array(sections)


class SpeechLeveler:
    def __init__(
        self, samplerate: int, *, target_lufs: float = -20.0,
        max_gain_db: float = 26.0, peak_dbfs: float = -3.0,
        preroll_ms: float = 300.0,
    ):
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        self.samplerate = samplerate
        self.frame_size = max(1, round(samplerate * 0.02))
        self.target_lufs = float(target_lufs)
        self.max_gain_db = max(0.0, float(max_gain_db))
        self.ceiling = 10.0 ** (min(-1.0, float(peak_dbfs)) / 20.0)
        self.preroll_frames = max(1, math.ceil(preroll_ms / 20.0))
        self._pending = np.empty(0, dtype=np.float32)
        self._initial: list[np.ndarray] = []
        self._powers: deque[float] = deque(maxlen=150)  # three seconds
        self._started = False
        self._have_voice = False
        self._gain_db = 0.0
        self._finished = False
        self._sos = _k_weighting(samplerate)
        self._filter_state = np.zeros((2, 2))
        self._limit_pending = None
        self._limit_cap = 1.0
        self._limit_gain = 1.0

    def _power(self, frame: np.ndarray) -> float:
        weighted, self._filter_state = sosfilt(self._sos, frame, zi=self._filter_state)
        return float(np.mean(weighted ** 2))

    def _estimate(self) -> float | None:
        powers = np.asarray(self._powers)
        if not powers.size:
            return None
        # Overlapping 400 ms windows make the estimate less sensitive to
        # individual phonemes. Gate silence and windows >10 LU below the mean.
        size = min(20, len(powers))
        blocks = np.convolve(powers, np.ones(size) / size, mode="valid")
        active = blocks[blocks > max(1e-6, float(blocks.mean()) * 0.1)]
        if not active.size:
            return None
        level = -0.691 + 10.0 * math.log10(float(active.mean()))
        return float(np.clip(self.target_lufs - level, -24.0, self.max_gain_db))

    def _render(self, frame: np.ndarray, *, adapt: bool) -> np.ndarray:
        previous = self._gain_db
        if adapt:
            power = self._power(frame)
            self._powers.append(power)
            # Freeze across quiet gaps, rather than increasing gain into noise.
            if power > max(1e-6, max(self._powers) * 0.01):
                desired = self._estimate()
                if desired is not None:
                    if not self._have_voice:
                        self._gain_db = previous = desired
                        self._have_voice = True
                    else:
                        # Attenuate faster than boosting; preserve word emphasis.
                        tau = 0.15 if desired < previous else 0.6
                        alpha = -math.expm1(-len(frame) / self.samplerate / tau)
                        self._gain_db += alpha * (desired - previous)
        gains = np.linspace(previous, self._gain_db, len(frame) + 1)[1:]
        output = frame.astype(np.float64) * np.power(10.0, gains / 20.0)
        return output.astype(np.float32)

    def _limit(self, frame: np.ndarray | None) -> np.ndarray:
        # Look one frame ahead, inspecting interpolated peaks rather than just
        # sample peaks. Ramp DOWN before the transient; release over 100 ms.
        # Both ends of the gain ramp are <= this frame's cap, so its samples
        # stay bounded without clipping/flattening the waveform.
        cap = 1.0
        if frame is not None and frame.size:
            peak = float(np.max(np.abs(resample_poly(frame, 4, 1))))
            cap = min(1.0, self.ceiling / max(peak, 1e-12))
        output = np.empty(0, dtype=np.float32)
        if self._limit_pending is not None:
            prior = self._limit_pending
            release = -math.expm1(-len(prior) / self.samplerate / 0.1)
            end = min(self._limit_cap, cap, self._limit_gain + (1 - self._limit_gain) * release)
            gain = np.linspace(self._limit_gain, end, len(prior) + 1)[1:]
            output = (prior * gain).astype(np.float32)
            self._limit_gain = end
        else:
            self._limit_gain = cap
        self._limit_pending, self._limit_cap = frame, cap
        return output

    def _start(self) -> list[np.ndarray]:
        self._started = True
        desired = self._estimate()
        if desired is not None:
            self._gain_db = desired
            self._have_voice = True
        output = [self._render(frame, adapt=False) for frame in self._initial]
        self._initial.clear()
        return output

    def process(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        """Return available mono float32 audio; call once with final=True at EOF.

        Buffers the pre-roll, a partial frame and one limiter frame. Output sample count
        including the final flush equals input count, with no added padding.
        """
        if self._finished:
            raise ValueError("speech leveler already finished")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("speech leveler requires mono audio")
        # Do not allow malformed samples to poison the running estimate.
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.concatenate((self._pending, samples))
        end = len(data) if final else len(data) // self.frame_size * self.frame_size
        output = []
        for offset in range(0, end, self.frame_size):
            frame = data[offset:min(offset + self.frame_size, end)]
            if not self._started:
                self._initial.append(frame.copy())
                self._powers.append(self._power(frame))
                if len(self._initial) >= self.preroll_frames:
                    output.extend(self._start())
            else:
                output.append(self._render(frame, adapt=True))
        self._pending = data[end:].copy()
        if final:
            self._finished = True
            if not self._started:
                output.extend(self._start())
        limited = [self._limit(frame) for frame in output]
        if final:
            limited.append(self._limit(None))
        return np.concatenate(limited) if limited else np.empty(0, dtype=np.float32)


def level_audio(samples: np.ndarray, samplerate: int, **kwargs) -> np.ndarray:
    """Level a cached take with exactly the same algorithm used by a stream."""
    return SpeechLeveler(samplerate, **kwargs).process(samples, final=True)
