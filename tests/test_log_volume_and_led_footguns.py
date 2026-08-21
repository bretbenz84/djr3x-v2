"""Log volume that hid the signal, and two LED calls that move the mouth silently.

The 2026-08-20 run was 7515 lines. Two patterns accounted for 20% of it: 773
identical `zone_block front` motion events (10.3%) and 742 `pose_face_guard`
phantom-face drops (9.9%). Neither was mis-logged — both subsystems were behaving
correctly — but at 99 events/min a genuine FLAP is invisible inside the noise, and
the flap rate is exactly the first thing the phantom-front-block work says to
check. The logs exist for post-run analysis; this is them getting in its own way.
"""

import time
import unittest
from unittest import mock

import config
from hardware import motion


class MotionEventCollapsingTests(unittest.TestCase):
    def setUp(self):
        motion._last_event_key = ""
        motion._last_event_log_at = 0.0
        motion._event_repeat_count = 0

    def _block(self, direction="front"):
        return {"type": "event", "event": "zone_block", "blocked_dir": direction}

    def test_a_standing_obstacle_logs_once(self):
        with mock.patch.object(motion._log, "info") as info:
            for _ in range(50):
                motion._log_motion_event(self._block())
            self.assertEqual(info.call_count, 1,
                             "the firmware re-announcing a persistent condition "
                             "must not produce a line per frame")

    def test_an_edge_is_never_swallowed(self):
        """The whole point: a real transition has to stay visible."""
        lines = []
        with mock.patch.object(motion._log, "info",
                               side_effect=lambda f, *a: lines.append(f % a)):
            for _ in range(30):
                motion._log_motion_event(self._block())
            motion._log_motion_event({"type": "event", "event": "zone_clear",
                                      "blocked_dir": "front"})
            motion._log_motion_event(self._block())
        self.assertTrue(any("zone_clear" in ln for ln in lines))
        self.assertEqual(sum("zone_block {" in ln for ln in lines), 2,
                         "both block runs must be visible as separate edges")

    def test_the_suppressed_count_is_reported(self):
        lines = []
        with mock.patch.object(motion._log, "info",
                               side_effect=lambda f, *a: lines.append(f % a)):
            for _ in range(30):
                motion._log_motion_event(self._block())
            motion._log_motion_event({"type": "event", "event": "zone_clear"})
        self.assertTrue(any("+29 identical" in ln for ln in lines),
                        "the flap rate must be recoverable from the log")

    def test_a_different_payload_is_a_different_event(self):
        """front vs left is a distinct fact, not a repeat."""
        with mock.patch.object(motion._log, "info") as info:
            motion._log_motion_event(self._block("front"))
            motion._log_motion_event(self._block("left"))
            self.assertEqual(info.call_count, 2)

    def test_collapsing_can_be_disabled(self):
        with (
            mock.patch.object(config, "MOTION_EVENT_REPEAT_LOG_INTERVAL_SECS", 0.0),
            mock.patch.object(motion._log, "info") as info,
        ):
            for _ in range(5):
                motion._log_motion_event(self._block())
            self.assertEqual(info.call_count, 5)

    def test_a_slow_repeat_still_logs(self):
        """Collapsing is a rate limit, not a mute — a condition that persists for
        minutes should still leave periodic evidence."""
        with mock.patch.object(config, "MOTION_EVENT_REPEAT_LOG_INTERVAL_SECS", 0.05):
            with mock.patch.object(motion._log, "info") as info:
                motion._log_motion_event(self._block())
                time.sleep(0.08)
                motion._log_motion_event(self._block())
                self.assertEqual(info.call_count, 2)


class MouthWithoutAudioTests(unittest.TestCase):
    """leds_head.speak() puts the head firmware into its FREE-RUNNING mouth
    animation (ANIM_SPEAK) — it keeps going with no audio behind it until a stop
    command arrives, or the 1500 ms firmware watchdog gives up. Two helpers in
    sequences/animations did that with nothing attached; excited_burst never
    stopped it at all. Same shape as the bug fixed in efdae3f."""

    def test_excited_burst_closes_the_mouth_it_opened(self):
        from sequences import animations
        with (
            mock.patch.object(animations, "leds_head") as head,
            mock.patch.object(animations, "leds_chest") as chest,
            mock.patch.object(animations, "servos"),
            mock.patch.object(animations.time, "sleep"),
        ):
            animations.excited_burst()
            head.speak.assert_called_once()
            head.speak_stop.assert_called_once()
            chest.active.assert_called_once()

    def test_the_dead_speech_helper_is_documented_as_a_footgun(self):
        """It is unused, but it is exactly the shape of the bug — the next caller
        needs to be told before they reach for it."""
        from sequences import animations
        doc = animations.speech_start.__doc__ or ""
        self.assertIn("UNUSED", doc)
        self.assertIn("free-running", doc.lower())


if __name__ == "__main__":
    unittest.main()
