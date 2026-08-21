"""One dead network must not cost two full timeouts, and must never make Rex deaf.

Field 2026-08-20 20:31 (logs/djr3x-2026-08-20-20-06-34.log L7255-L7388): the reply
"Backing up." was queued 8 ms after the transcript. The ElevenLabs streaming request
hung on the kernel's TCP timer for ~25 s; `_speak_streaming` returned False without
recording the failure, so `speak()` re-dialled the SAME dead endpoint on the buffered
path and paid another ~26 s. First audio landed 51.7 s after queue (session median:
7 ms). Meanwhile the AEC sequence hold — released by the speech queue draining, which
could not happen — held mic suppression for 57 s, so the wake-word detector reported
suppressed the whole time and Rex could not even hear "shut down".

Three independent caps, any one of which bounds the damage; all three are tested here.
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

import config
from audio import echo_cancel, tts


class ApiError(Exception):
    """Stands in for elevenlabs.core.ApiError — discrimination is by class NAME,
    the same way warmup_api() does it, so this is a faithful double."""


class ClientTimeoutTests(unittest.TestCase):
    def test_client_is_constructed_with_an_explicit_budget(self):
        """The SDK default is 240 s. A conversational line must never be able to
        cost that."""
        captured = {}

        class FakeEL:
            def __init__(self, **kw):
                captured.update(kw)

        fake_apikeys = mock.Mock(ELEVENLABS_API_KEY="k")
        fake_module = mock.Mock(ElevenLabs=FakeEL)
        with (
            mock.patch.dict("sys.modules", {"elevenlabs": fake_module,
                                            "apikeys": fake_apikeys}),
            mock.patch.object(tts, "_el_client", None),
        ):
            tts._get_el_client()
        self.assertIn("timeout", captured, "no timeout passed — SDK default is 240s")
        self.assertEqual(captured["timeout"], float(config.TTS_API_TIMEOUT_SECS))
        self.assertLessEqual(captured["timeout"], 60.0)


class StreamingFailureOpensBreakerTests(unittest.TestCase):
    def setUp(self):
        tts._note_api_success()          # breaker closed
        self.addCleanup(tts._note_api_success)

    def _stream_raising(self, exc):
        client = mock.Mock()
        client.text_to_speech.stream.side_effect = exc
        return client

    def test_network_failure_opens_the_breaker(self):
        with (
            mock.patch.object(tts, "_get_el_client",
                              return_value=self._stream_raising(OSError("[Errno 60] Operation timed out"))),
            mock.patch.object(tts, "output_gate"),
        ):
            handled = tts._speak_streaming(
                "hi", "hi", "v", "m", None, None, "neutral",
                tts.Path("/nonexistent/x.mp3"),
            )
        self.assertFalse(handled, "a failed stream must report unhandled")
        self.assertTrue(tts._api_circuit_open(),
                        "network failure did not arm the fallback breaker")

    def test_api_error_does_not_open_the_breaker(self):
        """A quota/4xx is a COMPLETED round-trip — the endpoint is reachable, so the
        buffered path is still worth a try and the local voice should not take over
        for the next two minutes."""
        with (
            mock.patch.object(tts, "_get_el_client",
                              return_value=self._stream_raising(ApiError("quota"))),
            mock.patch.object(tts, "output_gate"),
        ):
            tts._speak_streaming("hi", "hi", "v", "m", None, None, "neutral",
                                 tts.Path("/nonexistent/x.mp3"))
        self.assertFalse(tts._api_circuit_open(),
                         "an HTTP-level ApiError must not trip the breaker")


class NoSecondTimeoutTests(unittest.TestCase):
    """The expensive half: once the breaker is open, speak() must not re-dial."""

    def setUp(self):
        tts._note_api_success()
        self.addCleanup(tts._note_api_success)

    def test_speak_goes_local_instead_of_paying_a_second_timeout(self):
        spoke_locally = []
        streamed = []

        def fake_streaming(*a, **kw):
            streamed.append(1)
            tts._note_api_failure()      # what a network failure now does
            return False

        # NOT a constant True: with the breaker closed at the top of speak() the
        # turn must genuinely take the ElevenLabs path first, or this test would
        # pass on the pre-existing local dispatch and prove nothing.
        with (
            mock.patch.object(tts, "_speak_streaming", side_effect=fake_streaming),
            mock.patch.object(tts, "_fetch_from_api") as fetch,
            mock.patch.object(tts, "_use_local_backend",
                              side_effect=tts._api_circuit_open),
            mock.patch.object(tts, "_rex_local_ref", return_value=object()),
            mock.patch.object(tts, "_speak_local",
                              side_effect=lambda *a, **kw: spoke_locally.append(1) or True),
        ):
            tts.speak("Backing up.", "neutral")

        self.assertEqual(len(streamed), 1, "never reached the streaming path")
        fetch.assert_not_called()
        self.assertEqual(len(spoke_locally), 1, "line was dropped instead of spoken")

    def test_falls_through_to_the_api_when_the_local_voice_is_missing(self):
        """No local engine installed → still try ElevenLabs rather than drop the
        line. The request is bounded by TTS_API_TIMEOUT_SECS, so the cost is capped."""
        def fake_streaming(*a, **kw):
            tts._note_api_failure()
            return False

        with (
            mock.patch.object(tts, "_speak_streaming", side_effect=fake_streaming),
            mock.patch.object(tts, "_fetch_from_api", return_value=b"") as fetch,
            mock.patch.object(tts, "_use_local_backend", return_value=False),
        ):
            tts.speak("Backing up.", "neutral")
        fetch.assert_called_once()


class AecSequenceDeadmanTests(unittest.TestCase):
    """A wedge upstream of playback must not leave the mic attenuated forever."""

    def setUp(self):
        echo_cancel.end_sequence(flush=False, tail_secs=0.0)
        self.addCleanup(echo_cancel.end_sequence, False, 0.0)

    def test_hold_with_no_audio_under_it_releases(self):
        with mock.patch.object(config, "AEC_SEQUENCE_IDLE_RELEASE_SECS", 0.3):
            with mock.patch.object(echo_cancel, "_stream", create=True):
                echo_cancel.start_sequence()
            self.assertTrue(echo_cancel.is_suppressed(), "hold should suppress at first")
            time.sleep(0.45)
            self.assertFalse(echo_cancel.is_suppressed(),
                             "sequence hold stayed pinned with no audio under it")
            quiet = echo_cancel.filter(np.ones(8, dtype=np.float32))
            self.assertTrue(np.allclose(quiet, 1.0), "mic still attenuated after release")

    def test_real_audio_disarms_the_deadman(self):
        """A legitimately long segment — a 13 s impersonation take — must keep its
        suppression for the whole line."""
        with mock.patch.object(config, "AEC_SEQUENCE_IDLE_RELEASE_SECS", 0.3):
            with mock.patch.object(echo_cancel, "_stream", create=True):
                echo_cancel.start_sequence()
            echo_cancel.set_playing(True)
            time.sleep(0.45)
            self.assertTrue(echo_cancel.is_suppressed(),
                            "suppression released while audio was actually playing")

    def test_gap_between_segments_rearms_and_expires(self):
        with mock.patch.object(config, "AEC_SEQUENCE_IDLE_RELEASE_SECS", 0.3):
            with mock.patch.object(echo_cancel, "_stream", create=True):
                echo_cancel.start_sequence()
                echo_cancel.set_playing(True)
                echo_cancel.set_playing(False)          # segment 1 done, hold swallows it
                self.assertTrue(echo_cancel.is_suppressed(), "gap must stay suppressed")
                time.sleep(0.45)
                self.assertFalse(echo_cancel.is_suppressed(),
                                 "deadman did not re-arm for the inter-segment gap")

    def test_disabled_by_config(self):
        with mock.patch.object(config, "AEC_SEQUENCE_IDLE_RELEASE_SECS", 0.0):
            with mock.patch.object(echo_cancel, "_stream", create=True):
                echo_cancel.start_sequence()
            time.sleep(0.15)
            self.assertTrue(echo_cancel.is_suppressed(),
                            "cap of 0 must mean the old unbounded behaviour")


if __name__ == "__main__":
    unittest.main()
