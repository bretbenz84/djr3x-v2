"""Tests for the mic-stall watchdog in audio/stream.py.

Regression cover for the failure where DJ music playback's OutputStream
open/close silently killed the long-lived mic InputStream callback on macOS,
freezing the rolling buffer so Rex went permanently deaf. The watchdog must
notice that callbacks have stopped and reopen the stream — without ever touching
real audio hardware in the test.
"""

import time
import unittest

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
            )
        }
        self._real_open = stream._open_stream
        # Reset module state to a known baseline.
        stream._running = False
        stream._last_callback_at = 0.0
        stream._last_reopen_at = 0.0
        stream._reopen_count = 0
        stream._stream = None
        with stream._buf_lock:
            stream._buf.clear()

    def tearDown(self):
        # Make sure no watchdog thread survives into the next test.
        stream._running = False
        stream._stop_watchdog()
        stream._open_stream = self._real_open
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

    def test_watchdog_disabled_does_not_start(self):
        config.AUDIO_STALL_WATCHDOG_ENABLED = False
        stream._start_watchdog()
        self.assertIsNone(stream._watchdog_thread)


if __name__ == "__main__":
    unittest.main()
