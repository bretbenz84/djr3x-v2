"""Tests for the mic-stall watchdog in audio/stream.py.

Regression cover for the failure where DJ music playback's OutputStream
open/close silently killed the long-lived mic InputStream callback on macOS,
freezing the rolling buffer so Rex went permanently deaf. The watchdog must
notice that callbacks have stopped and reopen the stream — without ever touching
real audio hardware in the test.
"""

import threading
import time
import unittest
from unittest import mock

import config
from audio import stream


class _DummyStream:
    """Stand-in for a sounddevice.InputStream — no real device touched."""

    def __init__(self):
        self.active = True
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True
        self.active = False

    def close(self):
        self.closed = True


class StreamWatchdogTest(unittest.TestCase):
    def setUp(self):
        # Snapshot the watchdog config so per-test overrides don't leak.
        self._cfg = {
            k: getattr(config, k)
            for k in (
                "AUDIO_STALL_WATCHDOG_ENABLED",
                "AUDIO_STALL_TIMEOUT_SECS",
                "AUDIO_STALL_CHECK_INTERVAL_SECS",
                "AUDIO_STALL_REOPEN_MIN_SPACING_SECS",
                "AUDIO_STALL_REOPEN_TIMEOUT_SECS",
                "AUDIO_STALL_FATAL_SECS",
                "AUDIO_STALL_FATAL_RESTART_ENABLED",
                "AUDIO_STALL_FATAL_EXIT_GRACE_SECS",
            )
        }
        self._real_open = stream._open_stream
        # Reset module state to a known baseline.
        stream._running = False
        stream._last_callback_at = 0.0
        stream._last_reopen_at = 0.0
        stream._reopen_count = 0
        stream._down_since = 0.0
        stream._stream = None
        with stream._buf_lock:
            stream._buf.clear()

    def tearDown(self):
        # Make sure no watchdog thread survives into the next test.
        stream._running = False
        stream._stop_watchdog()
        stream._open_stream = self._real_open
        stream._down_since = 0.0
        stream._stream = None
        with stream._buf_lock:
            stream._buf.clear()
        for k, v in self._cfg.items():
            setattr(config, k, v)

    # ── pure status helpers ──────────────────────────────────────────────────

    def test_age_and_stalled_semantics(self):
        # No callback yet, not running → age is inf, not considered stalled.
        stream._running = False
        stream._last_callback_at = 0.0
        self.assertEqual(stream.last_callback_age(), float("inf"))
        self.assertFalse(stream.is_stalled())

        config.AUDIO_STALL_TIMEOUT_SECS = 1.5
        stream._running = True

        # Fresh callback → healthy.
        stream._last_callback_at = time.monotonic()
        self.assertLess(stream.last_callback_age(), 1.0)
        self.assertFalse(stream.is_stalled())

        # Callback long in the past → stalled.
        stream._last_callback_at = time.monotonic() - 10.0
        self.assertGreater(stream.last_callback_age(), 5.0)
        self.assertTrue(stream.is_stalled())

        # Not running → never reports stalled even with an ancient timestamp.
        stream._running = False
        self.assertFalse(stream.is_stalled())

    # ── reopen mechanics ─────────────────────────────────────────────────────

    def test_reopen_closes_old_stream_clears_buffer_and_reopens(self):
        old = _DummyStream()
        stream._stream = old
        stream._running = True
        with stream._buf_lock:
            stream._buf.append("stale")  # frozen sample left by the dead callback

        opened = {"count": 0}

        def fake_open():
            opened["count"] += 1
            stream._stream = _DummyStream()
            stream._last_callback_at = time.monotonic()
            return True

        stream._open_stream = fake_open

        ok = stream._reopen("unit test")

        self.assertTrue(ok)
        self.assertEqual(opened["count"], 1)
        self.assertTrue(old.stopped and old.closed, "stalled stream must be torn down")
        self.assertEqual(stream._reopen_count, 1)
        self.assertGreater(stream._last_reopen_at, 0.0)
        with stream._buf_lock:
            self.assertEqual(len(stream._buf), 0, "frozen audio must be dropped on reopen")

    def test_reopen_noop_when_not_running(self):
        stream._running = False
        called = {"n": 0}

        def fake_open():
            called["n"] += 1
            return True

        stream._open_stream = fake_open
        self.assertFalse(stream._reopen("should not run"))
        self.assertEqual(called["n"], 0)

    # ── end-to-end loop ──────────────────────────────────────────────────────

    def test_watchdog_thread_reopens_on_stall(self):
        config.AUDIO_STALL_WATCHDOG_ENABLED = True
        config.AUDIO_STALL_CHECK_INTERVAL_SECS = 0.02
        config.AUDIO_STALL_TIMEOUT_SECS = 0.05
        config.AUDIO_STALL_REOPEN_MIN_SPACING_SECS = 0.0

        opened = {"count": 0}

        def fake_open():
            opened["count"] += 1
            stream._stream = _DummyStream()
            # Simulate a healthy reopen: fresh callbacks resume immediately.
            stream._last_callback_at = time.monotonic()
            return True

        stream._open_stream = fake_open
        stream._stream = _DummyStream()
        stream._running = True
        # Simulate a wedged callback: last one was well past the stall timeout.
        stream._last_callback_at = time.monotonic() - 5.0

        stream._start_watchdog()
        try:
            deadline = time.monotonic() + 2.0
            while opened["count"] < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            stream._running = False
            stream._stop_watchdog()

        self.assertGreaterEqual(
            opened["count"], 1, "watchdog should have reopened the stalled stream"
        )


    # ── wedged-device survival ───────────────────────────────────────────────

    def test_wedged_reopen_is_abandoned_and_the_watchdog_survives(self):
        """A reopen that hangs in CoreAudio must not take the retry loop with it.

        Field 2026-08-18 18:20:46: "reopening (attempt 1)" was the last audio log
        of the session. The reopen ran INLINE on the watchdog thread, wedged
        inside the device call, and there was never an attempt 2 or an outcome
        line — Rex was deaf for the rest of the run while still seeing and
        moving. e2dae47 bounded the shutdown paths against this same hang; this
        is the recovery path.
        """
        config.AUDIO_STALL_REOPEN_TIMEOUT_SECS = 0.2
        released = threading.Event()
        entered = threading.Event()

        def wedged_open():
            entered.set()
            released.wait(5.0)      # stands in for a hung CoreAudio call
            return True

        stream._open_stream = wedged_open
        stream._stream = _DummyStream()
        stream._running = True

        t0 = time.monotonic()
        ok = stream._reopen("wedged device")
        elapsed = time.monotonic() - t0
        try:
            self.assertFalse(ok, "a wedged reopen must not report success")
            self.assertLess(elapsed, 2.0, "reopen must be bounded, not inline-forever")
            self.assertTrue(entered.is_set(), "the reopen should have been attempted")

            # The caller is free again, so the watchdog can still retry. The
            # abandoned worker keeps _stream_lock, so the retry fails FAST.
            t1 = time.monotonic()
            self.assertFalse(stream._reopen("still wedged"))
            self.assertLess(time.monotonic() - t1, 2.0, "retry must not block either")
            self.assertEqual(stream._reopen_count, 2, "the watchdog kept retrying")
        finally:
            released.set()
            time.sleep(0.05)

    def test_dead_mic_escalates_to_a_restart_once_reopens_stop_helping(self):
        """A device that ignores every reopen is not recoverable in-process.

        The wedged CoreAudio handles belong to THIS process, so exiting is what
        frees them — the supervisor then reopens from scratch. Staying up is
        worse: he keeps seeing, moving and turning, so he looks alive while
        ignoring the room (the 2026-08-18 field report).
        """
        config.AUDIO_STALL_WATCHDOG_ENABLED = True
        config.AUDIO_STALL_CHECK_INTERVAL_SECS = 0.02
        config.AUDIO_STALL_TIMEOUT_SECS = 0.05
        config.AUDIO_STALL_REOPEN_MIN_SPACING_SECS = 0.0
        config.AUDIO_STALL_REOPEN_TIMEOUT_SECS = 0.05
        config.AUDIO_STALL_FATAL_SECS = 0.3

        stream._open_stream = lambda: False      # the device never comes back
        stream._stream = _DummyStream()
        stream._running = True
        stream._last_callback_at = time.monotonic() - 5.0

        escalated = []
        with mock.patch.object(stream, "_escalate_dead_mic", escalated.append):
            stream._start_watchdog()
            try:
                deadline = time.monotonic() + 3.0
                while not escalated and time.monotonic() < deadline:
                    time.sleep(0.02)
            finally:
                stream._running = False
                stream._stop_watchdog()

        self.assertTrue(escalated, "a permanently dead mic must escalate")
        self.assertGreaterEqual(escalated[0], 0.3, "escalation reports the outage length")

    def test_escalation_can_be_switched_off(self):
        config.AUDIO_STALL_FATAL_RESTART_ENABLED = False
        with mock.patch.object(stream.os, "_exit") as hard_exit:
            stream._escalate_dead_mic(120.0)
        hard_exit.assert_not_called()

    def test_recovery_clears_the_outage_clock(self):
        config.AUDIO_STALL_WATCHDOG_ENABLED = True
        config.AUDIO_STALL_CHECK_INTERVAL_SECS = 0.02
        config.AUDIO_STALL_TIMEOUT_SECS = 0.05
        config.AUDIO_STALL_REOPEN_MIN_SPACING_SECS = 0.0
        config.AUDIO_STALL_FATAL_SECS = 0.0      # escalation off for this test

        def fake_open():
            stream._stream = _DummyStream()
            stream._last_callback_at = time.monotonic()
            return True

        stream._open_stream = fake_open
        stream._stream = _DummyStream()
        stream._running = True
        stream._last_callback_at = time.monotonic() - 5.0

        stream._start_watchdog()
        try:
            deadline = time.monotonic() + 2.0
            while stream._reopen_count < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.1)          # let the loop see the healthy stream
        finally:
            stream._running = False
            stream._stop_watchdog()

        self.assertEqual(stream.mic_down_secs(), 0.0, "outage clock must reset on recovery")

    def test_watchdog_disabled_does_not_start(self):
        config.AUDIO_STALL_WATCHDOG_ENABLED = False
        stream._start_watchdog()
        self.assertIsNone(stream._watchdog_thread)


if __name__ == "__main__":
    unittest.main()
