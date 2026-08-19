"""The shared output gate must never mute Rex permanently.

Field 2026-08-18: an impersonation's "thinking" chirp wedged inside its
CoreAudio play call WHILE HOLDING the gate. The Jimmy Carter take finished
rendering 17 s later and then waited on a gate nobody would ever hand back —
no parody, no outro, no audio of any kind for the rest of the turn. Every TTS
acquire is bounded now: timing out costs one dropped line, not the session.
"""

import threading
import time
import unittest
from unittest import mock

import config
from audio import output_gate, tts


class OutputGateTimeoutTest(unittest.TestCase):
    def setUp(self):
        self._timeout = getattr(config, "TTS_OUTPUT_GATE_TIMEOUT_SECS", 30.0)
        config.TTS_OUTPUT_GATE_TIMEOUT_SECS = 0.2

    def tearDown(self):
        config.TTS_OUTPUT_GATE_TIMEOUT_SECS = self._timeout

    def test_gate_reports_how_long_the_holder_has_had_it(self):
        self.assertEqual(output_gate.held_secs(), 0.0, "idle gate holds nothing")
        with output_gate.hold("sound-effects"):
            time.sleep(0.05)
            self.assertGreater(output_gate.held_secs(), 0.0)
            self.assertEqual(output_gate.active_source(), "sound-effects")
        self.assertEqual(output_gate.held_secs(), 0.0, "released gate holds nothing")

    def test_tts_gives_up_on_a_gate_that_is_never_handed_back(self):
        started = threading.Event()
        release = threading.Event()

        def wedged_holder():
            with output_gate.hold("sound-effects"):
                started.set()
                release.wait(5.0)      # stands in for a hung CoreAudio play

        holder = threading.Thread(target=wedged_holder, daemon=True)
        holder.start()
        self.assertTrue(started.wait(2.0), "holder never took the gate")

        try:
            t0 = time.monotonic()
            with output_gate.hold("tts", timeout=tts._gate_timeout()) as acquired:
                self.assertFalse(acquired, "TTS must not think it holds a wedged gate")
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 2.0, "the acquire must be bounded, not forever")
            self.assertGreaterEqual(elapsed, 0.15, "it should still wait its timeout")
        finally:
            release.set()
            holder.join(timeout=2.0)

    def test_the_timeout_names_the_holder_so_the_log_is_actionable(self):
        with output_gate.hold("sound-effects"):
            with mock.patch.object(tts.logger, "warning") as warn:
                tts._log_gate_timeout("local playback")
        self.assertTrue(warn.called, "a dropped line must be logged loudly")
        rendered = warn.call_args[0][0] % warn.call_args[0][1:]
        self.assertIn("local playback", rendered)
        self.assertIn("sound-effects", rendered, "the culprit must be named")

    def test_a_free_gate_is_still_acquired_normally(self):
        with output_gate.hold("tts", timeout=tts._gate_timeout()) as acquired:
            self.assertTrue(acquired)


if __name__ == "__main__":
    unittest.main()
