import unittest
from unittest import mock

import numpy as np


class SoftwareAECTests(unittest.TestCase):
    def setUp(self):
        from audio import aec
        self.aec = aec
        aec.reset()
        # Deterministic, controllable clock so reference placement + reads align.
        self._t = [0.0]
        self._clock_patch = mock.patch.object(aec, "_clock", side_effect=lambda: self._t[0])
        self._clock_patch.start()

    def tearDown(self):
        self._clock_patch.stop()
        self.aec.reset()

    def _aligned_mic(self, n):
        """The exact reference slice the AEC will read this tick (pure echo)."""
        core = self.aec._aec
        delay = core._delay_samples
        return core._timeline.read(self._t[0] - delay / 16000.0, n)

    def test_passthrough_when_disabled(self):
        with mock.patch.object(self.aec.config, "AEC_SOFTWARE_ENABLED", False):
            x = (np.random.RandomState(0).randn(1280).astype(np.float32) * 0.1)
            out = self.aec.process(x)
        np.testing.assert_array_equal(out, x)

    def test_passthrough_when_rex_quiet(self):
        # No reference pushed → no echo to cancel → mic returned untouched.
        x = (np.random.RandomState(1).randn(1280).astype(np.float32) * 0.1)
        out = self.aec.process(x)
        np.testing.assert_array_equal(out, x)

    def test_suppresses_aligned_echo(self):
        rng = np.random.RandomState(2)
        ref = (rng.randn(3 * 16000).astype(np.float32) * 0.2)  # 3s of "Rex playing"
        self._t[0] = 0.0
        self.aec.push_reference(ref, 16000)

        n = 1280
        last_in = last_out = None
        for k in range(14):
            self._t[0] = 0.4 + k * 0.08          # stays within the 3s reference span
            mic = self._aligned_mic(n)            # pure echo (mic == aligned reference)
            last_in, last_out = mic, self.aec.process(mic)

        in_rms = float(np.sqrt(np.mean(last_in ** 2)))
        out_rms = float(np.sqrt(np.mean(last_out ** 2)))
        self.assertGreater(in_rms, 1e-4)
        self.assertLess(out_rms, 0.5 * in_rms)    # echo meaningfully suppressed

    def test_double_talk_preserves_near_end(self):
        rng = np.random.RandomState(3)
        ref = (rng.randn(3 * 16000).astype(np.float32) * 0.05)  # quiet echo
        self._t[0] = 0.0
        self.aec.push_reference(ref, 16000)

        n = 1280
        for k in range(8):                        # converge echo gain on echo-only
            self._t[0] = 0.4 + k * 0.08
            self.aec.process(self._aligned_mic(n))

        # Loud near-end (user) on top of the quiet echo → double-talk.
        self._t[0] += 0.08
        echo = self._aligned_mic(n)
        near = (np.sin(2 * np.pi * 300.0 * np.arange(n) / 16000.0).astype(np.float32) * 0.3)
        out = self.aec.process(echo + near)

        out_rms = float(np.sqrt(np.mean(out ** 2)))
        near_rms = float(np.sqrt(np.mean(near ** 2)))
        # The user's voice must survive — not be crushed to the spectral floor.
        self.assertGreater(out_rms, 0.3 * near_rms)


if __name__ == "__main__":
    unittest.main()
