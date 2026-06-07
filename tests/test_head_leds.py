import unittest
from unittest import mock

import serial


class _FakeSerial:
    def __init__(self):
        self.is_open = True
        self.writes = []
        self.flushes = 0
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True
        self.is_open = False


class _FailingSerial(_FakeSerial):
    def write(self, data):
        raise serial.SerialException("write failed")


class HeadLedTests(unittest.TestCase):
    def setUp(self):
        from hardware import leds_head

        self.leds_head = leds_head
        self.old_ser = leds_head._ser
        self.old_speech_drop_notified = leds_head._speech_drop_notified
        self.old_dropped_counts = dict(leds_head._dropped_counts)
        self.old_eye_color = leds_head._eye_color
        self.old_eyes_should_be_on = leds_head._eyes_should_be_on
        self.old_speaking = leds_head._speaking
        # Default to eyes-off so the bare speak_stop/off tests are deterministic
        # (blink-resume only re-asserts ACTIVE when an eye colour is set).
        leds_head._eye_color = (0, 0, 0)
        leds_head._eyes_should_be_on = False
        leds_head._speaking = False

    def tearDown(self):
        h = self.leds_head
        h._ser = self.old_ser
        h._speech_drop_notified = self.old_speech_drop_notified
        h._dropped_counts.clear()
        h._dropped_counts.update(self.old_dropped_counts)
        h._eye_color = self.old_eye_color
        h._eyes_should_be_on = self.old_eyes_should_be_on
        h._speaking = self.old_speaking

    def test_speak_stop_repeats_stop_command_when_connected(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 3),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.0),
        ):
            h.speak_stop()

        self.assertEqual(
            fake.writes,
            [
                b"SPEAK_LEVEL:0\n",
                b"SPEAK_STOP\n",
                b"SPEAK_STOP\n",
                b"SPEAK_STOP\n",
            ],
        )

    def test_speak_stop_resumes_eye_blink_when_eyes_have_colour(self):
        # The Arduino suspends blinking on SPEAK_STOP; speak_stop must hand the
        # eyes back to ACTIVE (preserving colour) so Rex keeps blinking.
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (255, 200, 0)  # warm gold eyes are set

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 3),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.0),
        ):
            h.speak_stop()

        self.assertEqual(
            fake.writes,
            [
                b"SPEAK_LEVEL:0\n",
                b"SPEAK_STOP\n",
                b"SPEAK_STOP\n",
                b"SPEAK_STOP\n",
                b"ACTIVE\n",  # ← blink re-armed, eye colour preserved
            ],
        )
        self.assertTrue(h._eyes_active)

    def test_speak_stop_leaves_eyes_off_when_no_colour_set(self):
        # If the eyes are intentionally off, don't force them on.
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 1),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.0),
        ):
            h.speak_stop()

        self.assertNotIn(b"ACTIVE\n", fake.writes)

    def test_speak_stop_sends_single_drop_when_disconnected(self):
        h = self.leds_head
        h._ser = None

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 3),
            mock.patch.object(h, "send_command", wraps=h.send_command) as send_command,
        ):
            h.speak_stop()

        send_command.assert_called_once_with("SPEAK_STOP")

    def test_send_command_swallows_serial_write_failure(self):
        h = self.leds_head
        failing = _FailingSerial()
        h._ser = failing

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            self.assertLogs("hardware.leds_head", level="WARNING") as logs,
        ):
            h.send_command("SPEAK_STOP")

        self.assertIsNone(h._ser)
        self.assertTrue(failing.closed)
        self.assertTrue(any("Head Arduino write failed" in line for line in logs.output))

    def test_off_zeros_mouth_before_full_off_when_connected(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 3),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.0),
        ):
            h.off()

        self.assertEqual(
            fake.writes,
            [
                b"SPEAK_LEVEL:0\n",
                b"SPEAK_STOP\n",
                b"OFF\n",
                b"SPEAK_STOP\n",
                b"OFF\n",
                b"SPEAK_STOP\n",
                b"OFF\n",
            ],
        )
        self.assertEqual(fake.flushes, 6)


    # ── Eye keep-alive: ensure_eyes_on + heartbeat ──────────────────────────

    def test_speak_marks_speaking_and_flushes_the_speak_command(self):
        # SPEAK is now a critical/flushed command so the mouth-on trigger isn't
        # left buffered behind the SPEAK_LEVEL flood (mouth sometimes not lighting).
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h.speak("happy")

        self.assertTrue(h._speaking)
        self.assertEqual(fake.writes, [b"SPEAK:happy\n"])
        self.assertEqual(fake.flushes, 1)

    def test_speak_level_is_not_flushed(self):
        # The high-rate mouth flood must stay unflushed so it can coalesce.
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h.speak_level(120)

        self.assertEqual(fake.writes, [b"SPEAK_LEVEL:120\n"])
        self.assertEqual(fake.flushes, 0)

    def test_ensure_eyes_on_lights_eyes_at_emotion_colour(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h.ensure_eyes_on("excited")  # EYE_COLORS["excited"] == (255, 200, 0)

        self.assertEqual(fake.writes, [b"EYE:255,200,0\n"])
        self.assertEqual(h._eye_color, (255, 200, 0))
        self.assertTrue(h._eyes_active)
        self.assertTrue(h._eyes_should_be_on)

    def test_ensure_eyes_on_never_goes_dark_for_a_sleep_styled_line(self):
        # EYE_COLORS["sleep"] is (0,0,0); a sleep-styled speech line must not turn
        # the eyes off mid-run (which would also stand the heartbeat down).
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_RUNNING_EYE_COLOR", (255, 200, 0)),
        ):
            h.ensure_eyes_on("sleep")

        self.assertEqual(fake.writes, [b"EYE:255,200,0\n"])
        self.assertTrue(h._eyes_should_be_on)

    def test_ensure_eyes_on_uses_steady_colour_when_emotion_follow_disabled(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_EYE_FOLLOWS_EMOTION", False),
            mock.patch.object(h.config, "HEAD_LED_RUNNING_EYE_COLOR", (255, 200, 0)),
        ):
            h.ensure_eyes_on("angry")  # would be (255,0,0) if it followed emotion

        self.assertEqual(fake.writes, [b"EYE:255,200,0\n"])

    def test_heartbeat_reasserts_eye_colour_when_running_and_quiet(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 180, 255)
        h._eyes_should_be_on = True
        h._speaking = False

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h._heartbeat_tick()

        self.assertEqual(fake.writes, [b"EYE:0,180,255\n"])

    def test_heartbeat_is_noop_while_speaking(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 180, 255)
        h._eyes_should_be_on = True
        h._speaking = True

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h._heartbeat_tick()

        self.assertEqual(fake.writes, [])

    def test_heartbeat_is_noop_when_eyes_should_be_off(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)
        h._eyes_should_be_on = False
        h._speaking = False

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h._heartbeat_tick()

        self.assertEqual(fake.writes, [])

    def test_heartbeat_uses_default_colour_when_running_without_a_colour(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eye_color = (0, 0, 0)
        h._eyes_should_be_on = True
        h._speaking = False

        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_RUNNING_EYE_COLOR", (255, 200, 0)),
        ):
            h._heartbeat_tick()

        self.assertEqual(fake.writes, [b"EYE:255,200,0\n"])
        self.assertEqual(h._eye_color, (255, 200, 0))

    def test_sleep_and_off_stand_the_heartbeat_down(self):
        h = self.leds_head
        fake = _FakeSerial()
        h._ser = fake
        h._eyes_should_be_on = True

        with mock.patch.object(h, "HEAD_LEDS_ENABLED", True):
            h.sleep()
        self.assertFalse(h._eyes_should_be_on)

        h._eyes_should_be_on = True
        with (
            mock.patch.object(h, "HEAD_LEDS_ENABLED", True),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEATS", 1),
            mock.patch.object(h.config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.0),
        ):
            h.off()
        self.assertFalse(h._eyes_should_be_on)


if __name__ == "__main__":
    unittest.main()
