"""The mouth must ride the AUDIBLE timeline, not the write timeline.

Field report 2026-08-20: "the mouth LEDs start animating as if they're speaking
before he actually speaks", noticed on impersonations. Cause: every playback path
computed a chunk's brightness immediately before handing that chunk to the device,
but the device holds its reported output latency worth of audio downstream. On the
robot Mac PortAudio reports 0.95 s for the normal 'high'/4096 setting and 3.18 s
while AUDIO_PLAYBACK_CLONE_LATENCY_SECS is armed — which is exactly the
impersonation window (audio/tts._clone_deep_buffer_needed) — so the mouth ran up to
~3.2 s ahead of the voice and then froze for the last ~3.2 s of the line.

These lock in the pacing contract rather than the numbers.
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

import config
from audio import tts


SR = 22050
PIECE = int(SR * 0.033)


def _loud(i: int, n: int = PIECE) -> np.ndarray:
    """A syllable-like envelope, so consecutive frames differ by more than
    HEAD_LED_SPEAK_LEVEL_MIN_DELTA and are not throttled away."""
    amp = 0.02 + 0.28 * abs(np.sin(i * 0.9))
    return (np.sin(np.linspace(0, 40, n)) * amp).astype(np.float32)


class MouthPacerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self._lock = threading.Lock()

        def _level(b):
            with self._lock:
                self.events.append(("level", time.monotonic(), b))

        def _on_start():
            with self._lock:
                self.events.append(("mouth_on", time.monotonic(), None))

        self._on_start = _on_start
        patches = [
            mock.patch.object(tts.leds_head, "speak_level", side_effect=_level),
            mock.patch.object(tts.servos, "speech_reactive_move"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _drain(self, pacer, secs):
        deadline = time.monotonic() + secs
        while time.monotonic() < deadline:
            time.sleep(0.01)
        pacer.close()

    def _of(self, kind):
        with self._lock:
            return [e for e in self.events if e[0] == kind]

    def test_mouth_opens_one_output_buffer_after_the_first_write(self):
        """The whole bug in one assertion: writing does not open the mouth —
        the audio becoming audible does."""
        latency = 0.5
        pacer = tts._MouthPacer(latency, SR, self._on_start)
        t0 = time.monotonic()
        for i in range(20):
            pacer.push(_loud(i))
        # Every push has returned; nothing is audible yet, so the mouth is dark.
        self.assertEqual(self._of("mouth_on"), [], "mouth opened at write time")
        self._drain(pacer, latency + 1.0)
        opened = self._of("mouth_on")
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(opened[0][1] - t0, latency, delta=0.15)

    def test_levels_span_the_line_even_when_it_is_shorter_than_the_buffer(self):
        """A short line's writes all return at once. Scheduling on the audio
        timeline (not "now + latency") keeps it animating across its own
        duration instead of firing as one burst when the buffer clears."""
        latency, line_secs = 1.0, 0.5
        pacer = tts._MouthPacer(latency, SR, self._on_start)
        pushes = int(line_secs / 0.033)
        started = time.monotonic()
        for i in range(pushes):
            pacer.push(_loud(i))
        self.assertLess(time.monotonic() - started, 0.2, "writes were not instant")
        self._drain(pacer, latency + line_secs + 0.4)
        levels = self._of("level")
        self.assertGreater(len(levels), pushes // 2)
        spread = levels[-1][1] - levels[0][1]
        self.assertGreater(spread, line_secs * 0.6, f"burst, not paced ({spread:.2f}s)")

    def test_barge_in_drops_queued_levels(self):
        """stream.abort() throws the device buffer away, so queued levels are for
        audio nobody will hear. One landing after SPEAK_STOP would leave a stale
        SPEAK_LEVEL that the next SPEAK: inherits."""
        pacer = tts._MouthPacer(2.0, SR, self._on_start)
        for i in range(30):
            pacer.push(_loud(i))
        pacer.close(canceled=True)
        time.sleep(0.2)
        self.assertEqual(self.events, [], "queued mouth levels survived a barge-in")

    def test_shallow_buffer_is_a_pass_through(self):
        """Below TTS_MOUTH_SYNC_MIN_LATENCY_SECS the lead is under three LED
        frames — not worth a thread."""
        pacer = tts._MouthPacer(0.01, SR, self._on_start)
        self.assertIsNone(pacer._thread)
        self.assertEqual(pacer.delay, 0.0)
        self.assertEqual(len(self._of("mouth_on")), 1, "pass-through must open at once")
        pacer.push(_loud(0))
        self.assertEqual(len(self._of("level")), 1)
        pacer.close()

    def test_kill_switch_restores_write_paced_behaviour(self):
        with mock.patch.object(config, "TTS_MOUTH_SYNC_ENABLED", False):
            pacer = tts._MouthPacer(3.2, SR, self._on_start)
            self.assertIsNone(pacer._thread)
            self.assertEqual(pacer.delay, 0.0)
            self.assertEqual(len(self._of("mouth_on")), 1)
            pacer.close()


class DriveLedsStartDelayTests(unittest.TestCase):
    """_play()'s buffered path has the same lead: its LED thread walks the array
    on a wall clock while sd.play() hands the audio to the same deep buffer."""

    def test_start_delay_holds_the_mouth_and_is_cancellable(self):
        stop_event = threading.Event()
        opened = []
        with (
            mock.patch.object(tts.leds_head, "speak_level"),
            mock.patch.object(tts.servos, "speech_reactive_move"),
        ):
            t0 = time.monotonic()
            thread = threading.Thread(
                target=tts._drive_leds,
                args=(np.zeros(SR, dtype=np.float32), SR, stop_event, 0.4,
                      lambda: opened.append(time.monotonic())),
                daemon=True,
            )
            thread.start()
            time.sleep(0.15)
            self.assertEqual(opened, [], "mouth opened before the audio was audible")
            thread.join(timeout=2.0)
            self.assertEqual(len(opened), 1)
            self.assertAlmostEqual(opened[0] - t0, 0.4, delta=0.15)

    def test_cancel_during_the_delay_never_opens_the_mouth(self):
        stop_event = threading.Event()
        opened = []
        with (
            mock.patch.object(tts.leds_head, "speak_level"),
            mock.patch.object(tts.servos, "speech_reactive_move"),
        ):
            thread = threading.Thread(
                target=tts._drive_leds,
                args=(np.zeros(SR, dtype=np.float32), SR, stop_event, 1.0,
                      lambda: opened.append(time.monotonic())),
                daemon=True,
            )
            thread.start()
            time.sleep(0.1)
            stop_event.set()          # barge-in before the first sound
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(opened, [], "mouth lit for audio that was aborted")


class BeginSpeechDeferTests(unittest.TestCase):
    def test_defer_mouth_withholds_only_the_mouth(self):
        """The speech-activity flag and begin_speech_motion are the "Rex owns the
        head now" claim — delaying those would let idle wander take the neck
        mid-line — and AEC must be armed before any sound reaches the room."""
        with (
            mock.patch.object(tts.animations, "speech_activity_start") as activity,
            mock.patch.object(tts.servos, "begin_speech_motion") as servo_start,
            mock.patch.object(tts.leds_head, "speak") as head_speak,
            mock.patch.object(tts.leds_head, "ensure_eyes_on") as eyes,
            mock.patch.object(tts.leds_chest, "speak") as chest_speak,
            mock.patch.object(tts.echo_cancel, "set_playing") as aec,
        ):
            tts._begin_speech("neutral", ttl_secs=8.0, defer_mouth=True)
            head_speak.assert_not_called()
            chest_speak.assert_not_called()
            activity.assert_called_once()
            servo_start.assert_called_once()
            eyes.assert_called_once()
            aec.assert_called_once_with(True)

    def test_default_still_lights_the_mouth_inline(self):
        with (
            mock.patch.object(tts.animations, "speech_activity_start"),
            mock.patch.object(tts.servos, "begin_speech_motion"),
            mock.patch.object(tts.leds_head, "speak") as head_speak,
            mock.patch.object(tts.leds_head, "ensure_eyes_on"),
            mock.patch.object(tts.leds_chest, "speak") as chest_speak,
            mock.patch.object(tts.echo_cancel, "set_playing"),
        ):
            tts._begin_speech("neutral", ttl_secs=8.0)
            head_speak.assert_called_once()
            chest_speak.assert_called_once()


if __name__ == "__main__":
    unittest.main()
