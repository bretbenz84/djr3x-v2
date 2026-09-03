"""reSpeaker Flex XVF3800 (6-channel USB build) channel handling.

The Flex is NOT a stereo pair: channel 0 is the AGC'd Conference output, 1 is
the ASR beam, 2-5 are raw capsules. Both the main capture path
(audio/stream.py) and the supervisor's wake-word listener must read exactly
the configured channel and never average the six together. No hardware is
touched here — every test drives the pure helpers with synthetic frames.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("rex_supervisor", _REPO / "rex_supervisor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(n: int = 64, channels: int = 6) -> np.ndarray:
    """(n, channels) where channel c is the constant value c+1 — trivially identifiable."""
    return np.tile(np.arange(1, channels + 1, dtype=np.float32), (n, 1))


class SupervisorChannelSelectTest(unittest.TestCase):
    def setUp(self):
        self.sup = _load_supervisor()

    def test_env_channel_parsing(self):
        f = self.sup._aec_input_channel
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_AEC_INPUT_CHANNEL", None)
            self.assertEqual(f({"AUDIO_AEC_INPUT_CHANNEL": "1"}), 1)
            self.assertEqual(f({"AUDIO_AEC_INPUT_CHANNEL": "0"}), 0)
            self.assertIsNone(f({"AUDIO_AEC_INPUT_CHANNEL": ""}))
            self.assertIsNone(f({"AUDIO_AEC_INPUT_CHANNEL": "-1"}))
            self.assertIsNone(f({}))
            self.assertIsNone(f({"AUDIO_AEC_INPUT_CHANNEL": "nope"}))

    def test_process_env_overrides_dotenv(self):
        with mock.patch.dict(os.environ, {"AUDIO_AEC_INPUT_CHANNEL": "3"}):
            self.assertEqual(self.sup._aec_input_channel({"AUDIO_AEC_INPUT_CHANNEL": "1"}), 3)

    def test_selected_channel_is_read_verbatim(self):
        mono = self.sup._frames_to_mono(_frames(), 1)
        self.assertEqual(mono.shape, (64,))
        self.assertTrue(np.all(mono == 2.0))  # channel 1 carries the value 2

    def test_blank_channel_mixes_like_before(self):
        mono = self.sup._frames_to_mono(_frames(), None)
        self.assertTrue(np.allclose(mono, 3.5))  # mean of 1..6

    def test_out_of_range_channel_falls_back_to_mix(self):
        # A Lite (2-ch) with a stale Flex setting must not crash the listener.
        mono = self.sup._frames_to_mono(_frames(channels=2), 1)
        self.assertTrue(np.all(mono == 2.0))
        mono = self.sup._frames_to_mono(_frames(channels=2), 5)
        self.assertTrue(np.allclose(mono, 1.5))

    def test_mono_stream_passthrough(self):
        one = np.arange(10, dtype=np.float32).reshape(-1, 1)
        self.assertTrue(np.array_equal(self.sup._frames_to_mono(one, 1), np.arange(10)))


class StreamCallbackChannelSelectTest(unittest.TestCase):
    """audio/stream.py must pick the configured column off a 6-ch callback block."""

    def setUp(self):
        from audio import stream
        self.stream = stream
        self._saved = (stream._input_channels, stream._aec_channel, stream._input_gain)
        stream.flush()

    def tearDown(self):
        s = self.stream
        s._input_channels, s._aec_channel, s._input_gain = self._saved
        s.flush()

    def test_callback_reads_only_the_asr_channel(self):
        s = self.stream
        s._input_channels, s._aec_channel, s._input_gain = 6, 1, 1.0
        s._callback(_frames(512), 512, None, None)
        buf = s.get_full_buffer()
        self.assertEqual(buf.shape, (512,))
        self.assertTrue(np.all(buf == 2.0))

    def test_callback_mixes_when_unset(self):
        s = self.stream
        s._input_channels, s._aec_channel, s._input_gain = 6, None, 1.0
        s._callback(_frames(512), 512, None, None)
        self.assertTrue(np.allclose(s.get_full_buffer(), 3.5))


class MicCheckClassificationTest(unittest.TestCase):
    """tools/mic_check.py channel identification + ERLE math, hardware-free."""

    @classmethod
    def setUpClass(cls):
        from tools import mic_check
        cls.mc = mic_check

    @staticmethod
    def _flex_like():
        # ch0/ch1 correlated (both processed), ch2-5 correlated (raw capsules),
        # the two groups independent — what the real 2026-09-02 capture showed.
        corr = np.eye(6)
        corr[0, 1] = corr[1, 0] = 0.96
        for a in range(2, 6):
            for b in range(2, 6):
                if a != b:
                    corr[a, b] = 0.96
        stats = [{"ch": c, "rms_dbfs": lvl, "peak_dbfs": lvl + 12, "clip": 0.0}
                 for c, lvl in enumerate([-26.0, -47.0, -52.0, -52.0, -53.0, -53.0])]
        return stats, corr

    def test_flex_layout_recommends_asr_channel(self):
        stats, corr = self._flex_like()
        res = self.mc._classify_channels(stats, corr)
        self.assertEqual(res["recommend"], 1)
        self.assertIn([0, 1], res["clusters"])
        self.assertIn([2, 3, 4, 5], res["clusters"])
        self.assertIn("Flex", res["verdict"])

    def test_lite_identical_pair_recommends_mix(self):
        stats = [{"ch": 0, "rms_dbfs": -40.0, "peak_dbfs": -20.0, "clip": 0.0},
                 {"ch": 1, "rms_dbfs": -40.1, "peak_dbfs": -20.0, "clip": 0.0}]
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        res = self.mc._classify_channels(stats, corr)
        self.assertIsNone(res["recommend"])
        self.assertIn("IDENTICAL", res["verdict"])

    def test_silent_reference_channel_is_skipped(self):
        stats = [{"ch": 0, "rms_dbfs": -40.0, "peak_dbfs": -20.0, "clip": 0.0},
                 {"ch": 1, "rms_dbfs": -95.0, "peak_dbfs": -90.0, "clip": 0.0}]
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        res = self.mc._classify_channels(stats, corr)
        self.assertEqual(res["recommend"], 0)

    def test_erle_windows_measure_raw_minus_processed(self):
        sr = self.mc.SR
        n = 3 * sr
        rng = np.random.default_rng(0)
        raw = rng.standard_normal(n).astype(np.float32) * 0.1        # -20 dBFS
        rec = np.zeros((n, 6), dtype=np.float32)
        rec[:, 0] = raw * 0.01                                       # 40 dB down
        rec[:, 1] = raw * 0.1                                        # 20 dB down
        for c in range(2, 6):
            rec[:, c] = raw
        rows = self.mc._erle_windows(rec, [0, 1], [2, 3, 4, 5], 1.0)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertAlmostEqual(r["erle_ch0"], 40.0, delta=0.2)
            self.assertAlmostEqual(r["erle_ch1"], 20.0, delta=0.2)
            self.assertAlmostEqual(r["raw_dbfs"], -20.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
