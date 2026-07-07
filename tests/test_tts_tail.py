"""
TTS tail integrity (owner report 2026-07-06: "TTS is being cut off at the end").

Two distinct causes, two guards:
- Truncated GENERATION: some streamed ElevenLabs takes end mid-word at speech-level
  RMS (measured 0.021-0.027 in the live cache; a natural tail decays to ~0.0002).
  _ends_hot detects this so the take is never cached — a bad roll plays once
  instead of becoming the permanent cached rendition of that line.
- Over-eager tail trim: 40ms padding after the last supra-threshold window shaved
  breathy word-final decays. Padding is now 120ms.
"""

import unittest
from unittest import mock

import numpy as np

import config
from audio import tts


def _tone(secs: float, sr: int = 22050, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, secs, int(sr * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class EndsHotTest(unittest.TestCase):
    SR = 22050

    def test_truncated_take_ends_hot(self):
        # Speech-level audio right up to the last sample — the live failure shape.
        self.assertTrue(tts._ends_hot(_tone(1.0), self.SR))

    def test_natural_decay_is_not_hot(self):
        audio = _tone(1.0)
        fade = np.linspace(1.0, 0.0, int(self.SR * 0.25)) ** 2
        audio[-fade.size:] *= fade.astype(np.float32)
        tail = np.zeros(int(self.SR * 0.10), dtype=np.float32)  # trailing silence
        self.assertFalse(tts._ends_hot(np.concatenate([audio, tail]), self.SR))

    def test_intra_word_dip_cannot_false_positive(self):
        # Only the FINAL window is checked — loud audio earlier is irrelevant.
        audio = np.concatenate([_tone(1.0), np.zeros(int(self.SR * 0.05), np.float32)])
        self.assertFalse(tts._ends_hot(audio, self.SR))

    def test_zero_threshold_disables(self):
        with mock.patch.object(config, "TTS_HOT_END_RMS", 0.0, create=True):
            self.assertFalse(tts._ends_hot(_tone(1.0), self.SR))

    def test_empty_audio_is_safe(self):
        self.assertFalse(tts._ends_hot(np.zeros(0, np.float32), self.SR))


class TrimPaddingTest(unittest.TestCase):
    SR = 22050

    def test_trim_keeps_padding_after_last_voice(self):
        # 1s tone + 1s silence; trim should cut to ~tone end + padding.
        audio = np.concatenate([_tone(1.0), np.zeros(self.SR, np.float32)])
        with (
            mock.patch.object(config, "TTS_TRIM_TRAILING_SILENCE_ENABLED", True),
            mock.patch.object(config, "TTS_TRIM_TRAILING_SILENCE_PADDING_MS", 120),
        ):
            trimmed = tts._trim_trailing_silence(audio, self.SR)
        kept_silence = trimmed.size - self.SR   # samples kept beyond the tone
        self.assertGreaterEqual(kept_silence, int(self.SR * 0.10))  # >=100ms cushion
        self.assertLess(trimmed.size, audio.size)                   # still trims

    def test_soft_decay_survives_trim(self):
        # A word-final decay below the RMS threshold must stay within the padding.
        audio = _tone(1.0)
        fade = np.linspace(1.0, 0.0, int(self.SR * 0.08)).astype(np.float32) * 0.008
        decay = fade * np.sin(
            2 * np.pi * 220.0 * np.linspace(0, 0.08, fade.size)).astype(np.float32)
        full = np.concatenate([audio, decay, np.zeros(self.SR, np.float32)])
        with (
            mock.patch.object(config, "TTS_TRIM_TRAILING_SILENCE_ENABLED", True),
            mock.patch.object(config, "TTS_TRIM_TRAILING_SILENCE_PADDING_MS", 120),
        ):
            trimmed = tts._trim_trailing_silence(full, self.SR)
        # The 80ms decay fits inside the 120ms padding after the last loud window.
        self.assertGreaterEqual(trimmed.size, audio.size + decay.size)


if __name__ == "__main__":
    unittest.main()
