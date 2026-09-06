"""Every raw stream on the shared ReSpeaker device opens and closes under the
sd_guard device lock, and every playback stream asks for ONE blocksize.

Playback is routed THROUGH the ReSpeaker so its onboard AEC gets a reference,
which makes the mic InputStream and every speaker OutputStream one CoreAudio
device. sd_guard serialized sd.play()/sd.stop() in June; the raw
sd.OutputStream paths (the streamed ElevenLabs reply, the local clone take,
the sfx overlay and loop streams) and the mic's own open/close bypassed it.
Field 2026-09-05 17:14 (and 2026-09-02 23:12 before it): an unguarded open in
the impersonation window landed on top of an sfx sd.play(), the mic callback
stopped, four reopens wedged inside CoreAudio, and the watchdog exited the
process. These pin the guard at every one of those sites.
"""

import re
import sys
import threading
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import config
from audio import local_tts, sd_guard, sound_effects as sfx, tts

_REPO = Path(__file__).resolve().parent.parent


class _ProbeStream:
    """Records whether the device lock was held at each control-plane call."""

    def __init__(self, *a, **kw):
        self.kwargs = kw
        self.locked_at = {"init": sd_guard.is_device_locked_by_me()}
        self.writes = 0

    def start(self):
        self.locked_at["start"] = sd_guard.is_device_locked_by_me()

    def write(self, samples):
        self.writes += 1
        # Steady-state writes must NOT hold the lock (a song-length hold would
        # block every other control call); assert the first one only.
        self.locked_at.setdefault("write", sd_guard.is_device_locked_by_me())

    def stop(self):
        self.locked_at["stop"] = sd_guard.is_device_locked_by_me()

    def abort(self):
        self.locked_at["abort"] = sd_guard.is_device_locked_by_me()

    def close(self):
        self.locked_at["close"] = sd_guard.is_device_locked_by_me()


def _assert_guarded(test, probe: _ProbeStream, *, expect_stop=True):
    test.assertTrue(probe.locked_at.get("init"), "OutputStream() opened unguarded")
    test.assertTrue(probe.locked_at.get("start"), "start() ran unguarded")
    test.assertIn("write", probe.locked_at, "no audio was written through the fake stream")
    test.assertFalse(probe.locked_at["write"], "write() must run lock-free")
    if expect_stop:
        test.assertTrue(probe.locked_at.get("stop"), "stop() ran unguarded")
    test.assertTrue(probe.locked_at.get("close"), "close() ran unguarded")


class _StreamCapture:
    """Patch target for sounddevice.OutputStream that keeps every instance."""

    def __init__(self):
        self.instances = []

    def __call__(self, *a, **kw):
        s = _ProbeStream(*a, **kw)
        self.instances.append(s)
        return s


def _lock_settle_zero():
    return mock.patch.object(config, "AUDIO_PLAYBACK_STOP_SETTLE_SECS", 0.0, create=True)


def _no_hardware(stack: ExitStack) -> None:
    """The playback epilogue drives head/chest LEDs and the speech servos; on the
    robot Mac those are REAL. Mock them so this module never moves anything."""
    for name in ("servos", "leds_head", "leds_chest", "animations"):
        stack.enter_context(mock.patch.object(tts, name))


REX_REF = local_tts.VoiceRef("/nonexistent/rex.wav", "reference text", "rex")


class LocalTakePlaybackGuardTest(unittest.TestCase):
    """audio/tts._speak_local — the stream that opened unguarded the moment the
    Clinton take was ready (2026-09-02 23:12)."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._stack = ExitStack()
        self.capture = _StreamCapture()
        self._stack.enter_context(mock.patch.object(config, "TTS_CACHE_DIR", self._tmp.name))
        self._stack.enter_context(mock.patch("sounddevice.OutputStream", self.capture))
        self._stack.enter_context(mock.patch.object(local_tts, "sample_rate", return_value=24000))
        self._stack.enter_context(_lock_settle_zero())
        _no_hardware(self._stack)
        from audio import echo_cancel
        self._stack.enter_context(mock.patch.object(echo_cancel, "was_canceled", return_value=False))

    def tearDown(self):
        self._stack.close()
        self._tmp.cleanup()

    def test_local_take_stream_opens_and_closes_under_the_device_lock(self):
        def gen(text, voice_ref):
            for _ in range(3):
                yield np.full(1200, 0.1, dtype=np.float32)

        with mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", False), \
             mock.patch.object(local_tts, "generate_stream", gen):
            handled = tts._speak_local("hi rex", REX_REF, "neutral", log_text=False)
        self.assertTrue(handled)
        self.assertEqual(len(self.capture.instances), 1)
        _assert_guarded(self, self.capture.instances[0])
        self.assertFalse(sd_guard.is_device_locked_by_me(), "lock leaked after playback")


class StreamedReplyGuardTest(unittest.TestCase):
    """audio/tts._speak_streaming — the ElevenLabs reply stream that opened with
    the clone deep buffer beside a motion chirp (2026-09-05 17:14)."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._stack = ExitStack()
        self.capture = _StreamCapture()
        self._stack.enter_context(mock.patch.object(config, "TTS_CACHE_DIR", self._tmp.name))
        self._stack.enter_context(mock.patch("sounddevice.OutputStream", self.capture))
        self._stack.enter_context(_lock_settle_zero())
        _no_hardware(self._stack)
        from audio import echo_cancel
        self._stack.enter_context(mock.patch.object(echo_cancel, "was_canceled", return_value=False))
        tts._note_api_success()
        self.addCleanup(tts._note_api_success)

    def tearDown(self):
        self._stack.close()
        self._tmp.cleanup()

    def _client(self):
        client = mock.Mock()
        pcm = (np.zeros(2205, dtype=np.int16)).tobytes()
        client.text_to_speech.stream.return_value = iter([pcm, pcm, pcm])
        return client

    def test_streamed_reply_opens_and_closes_under_the_device_lock(self):
        with mock.patch.object(tts, "_get_el_client", return_value=self._client()):
            handled = tts._speak_streaming(
                "hi", "hi", "v", "m", None, None, "neutral",
                Path(self._tmp.name) / "x.mp3", log_text=False,
            )
        self.assertTrue(handled)
        self.assertEqual(len(self.capture.instances), 1)
        _assert_guarded(self, self.capture.instances[0])
        self.assertFalse(sd_guard.is_device_locked_by_me(), "lock leaked after playback")

    def test_barge_in_abort_is_guarded_too(self):
        from audio import echo_cancel
        calls = iter([False, True, True, True])
        with mock.patch.object(tts, "_get_el_client", return_value=self._client()), \
             mock.patch.object(echo_cancel, "was_canceled", side_effect=lambda: next(calls, True)):
            tts._speak_streaming(
                "hi", "hi", "v", "m", None, None, "neutral",
                Path(self._tmp.name) / "x.mp3", log_text=False,
            )
        probe = self.capture.instances[0]
        self.assertTrue(probe.locked_at.get("abort"), "abort() ran unguarded")
        self.assertTrue(probe.locked_at.get("close"), "close() ran unguarded")


class _StubSD(types.ModuleType):
    def __init__(self):
        super().__init__("sounddevice")
        self.play_blocksizes = []
        self.streams = []

    def play(self, audio, samplerate, blocksize=None):
        self.play_blocksizes.append(blocksize)

    def stop(self):
        pass

    def OutputStream(self, **kwargs):  # noqa: N802 — mirrors the sounddevice API
        s = _ProbeStream(**kwargs)
        self.streams.append(s)
        return s


class SoundEffectStreamsGuardTest(unittest.TestCase):
    def setUp(self):
        # The TTS tests above leave sfx's yield flag set (speech asked for the
        # speaker); a yieldable loop write would bail before its first write.
        sfx.reset()
        self.addCleanup(sfx.reset)
        self.sd = _StubSD()
        self._stack = ExitStack()
        self._stack.enter_context(_lock_settle_zero())
        self.audio = np.zeros(4410, dtype=np.float32)

    def tearDown(self):
        self._stack.close()

    def test_loop_stream_opens_and_closes_under_the_device_lock(self):
        loop = sfx._LoopStream(self.sd)
        self.assertTrue(loop.write(self.audio, 44100, None))
        loop.close()
        self.assertEqual(len(self.sd.streams), 1)
        _assert_guarded(self, self.sd.streams[0])
        self.assertEqual(self.sd.streams[0].kwargs["blocksize"], sfx._playback_blocksize())

    def test_overlay_stream_opens_and_closes_under_the_device_lock(self):
        echo_cancel = mock.Mock()
        output_gate = mock.Mock()
        output_gate.active_source.return_value = None
        started = sfx._play_overlay(
            self.sd, echo_cancel, output_gate, self.audio, 44100,
            Path("motion_whir.wav"), "motion_move",
        )
        self.assertTrue(started)
        self.assertEqual(len(self.sd.streams), 1)
        _assert_guarded(self, self.sd.streams[0])
        self.assertEqual(self.sd.streams[0].kwargs["blocksize"], sfx._playback_blocksize())

    def test_one_playback_blocksize_everywhere(self):
        """sd.play() sites used a private 2048 while TTS used AUDIO_PLAYBACK_BLOCKSIZE
        (4096): PortAudio sets the requested frames-per-buffer on the CoreAudio
        device the mic shares, so the two were reconfiguring the hardware under
        the live input callback several times a turn."""
        self.assertEqual(sfx._playback_blocksize(), int(config.AUDIO_PLAYBACK_BLOCKSIZE))
        for rel in ("audio/sound_effects.py", "audio/speech_queue.py", "audio/tts.py"):
            src = (_REPO / rel).read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"blocksize\s*=\s*\d+", src),
                f"{rel} hardcodes a playback blocksize — use AUDIO_PLAYBACK_BLOCKSIZE",
            )


class _ProbeInputStream:
    def __init__(self, *a, **kw):
        self.locked_at = {"init": sd_guard.is_device_locked_by_me()}
        self.active = True

    def start(self):
        self.locked_at["start"] = sd_guard.is_device_locked_by_me()

    def stop(self):
        self.locked_at["stop"] = sd_guard.is_device_locked_by_me()
        self.active = False

    def close(self):
        self.locked_at["close"] = sd_guard.is_device_locked_by_me()


class MicStreamGuardTest(unittest.TestCase):
    """audio/stream.py — the other side of the same device."""

    def setUp(self):
        from audio import stream
        self.stream = stream
        self.sd = types.ModuleType("sounddevice")
        self.instances = []

        def _input_stream(*a, **kw):
            s = _ProbeInputStream(*a, **kw)
            self.instances.append(s)
            return s

        self.sd.InputStream = _input_stream
        self.sd.query_devices = lambda idx: {"max_input_channels": 2}
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.dict(sys.modules, {"sounddevice": self.sd}))
        self._stack.enter_context(mock.patch.object(stream, "AUDIO_DEVICE_INDEX", 0))
        self._stack.enter_context(_lock_settle_zero())
        self._saved = (stream._stream, stream._running, stream._input_channels)
        stream._stream = None
        stream._running = False

    def tearDown(self):
        self.stream._stream, self.stream._running, self.stream._input_channels = self._saved
        self._stack.close()

    def test_mic_open_and_close_run_under_the_device_lock(self):
        with self.stream._stream_lock:
            self.assertTrue(self.stream._open_stream())
        self.assertEqual(len(self.instances), 1)
        probe = self.instances[0]
        self.assertTrue(probe.locked_at["init"], "InputStream() opened unguarded")
        self.assertTrue(probe.locked_at["start"], "start() ran unguarded")
        self.assertIs(self.stream._stream, probe)

        self.stream._running = True
        with mock.patch.object(self.stream, "_stop_watchdog"):
            self.stream.stop()
        self.assertTrue(probe.locked_at.get("stop"), "stop() ran unguarded")
        self.assertTrue(probe.locked_at.get("close"), "close() ran unguarded")
        self.assertIsNone(self.stream._stream)

    def test_mic_open_does_not_wait_forever_on_a_wedged_device_lock(self):
        """The bounded acquire: a playback control call stuck inside CoreAudio
        holds the lock forever; the mic must still open (unguarded) rather than
        never open at all."""
        holder_started = threading.Event()
        release = threading.Event()

        def _hold():
            with sd_guard.device_control():
                holder_started.set()
                release.wait(5.0)

        t = threading.Thread(target=_hold, daemon=True)
        t.start()
        holder_started.wait(1.0)
        try:
            with mock.patch.object(self.stream, "_DEVICE_LOCK_WAIT_SECS", 0.05), \
                 self.stream._stream_lock, \
                 self.assertLogs(self.stream._log, level="WARNING") as captured:
                self.assertTrue(self.stream._open_stream())
        finally:
            release.set()
            t.join(1.0)
        self.assertEqual(len(self.instances), 1)
        self.assertFalse(self.instances[0].locked_at["init"])
        self.assertTrue(any("unguarded" in line for line in captured.output))


class TryDeviceControlTest(unittest.TestCase):
    def test_acquires_and_releases(self):
        with sd_guard.try_device_control(1.0) as locked:
            self.assertTrue(locked)
            self.assertTrue(sd_guard.is_device_locked_by_me())
        self.assertFalse(sd_guard.is_device_locked_by_me())

    def test_yields_false_without_touching_a_lock_it_did_not_get(self):
        holder_started = threading.Event()
        release = threading.Event()

        def _hold():
            with sd_guard.device_control():
                holder_started.set()
                release.wait(5.0)

        t = threading.Thread(target=_hold, daemon=True)
        t.start()
        holder_started.wait(1.0)
        try:
            with sd_guard.try_device_control(0.05) as locked:
                self.assertFalse(locked)
        finally:
            release.set()
            t.join(1.0)
        # The holder still owns it until it releases; after join it's free.
        with sd_guard.try_device_control(1.0) as locked:
            self.assertTrue(locked)


if __name__ == "__main__":
    unittest.main()
