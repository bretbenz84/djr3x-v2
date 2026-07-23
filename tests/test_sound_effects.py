"""
Tests for audio/sound_effects.py — the droid chirp/whir layer.

No real audio: sounddevice is stubbed (sys.modules patch) and decode is mocked where
playback mechanics are under test. One tolerant test decodes a real committed MP3 to
prove the soundfile path works on this machine (skipped if libsndfile lacks mp3).

The invariants under test are the ones that keep the feature safe on the robot:
effects never play when the gate is busy, they yield to a blocking source within a
wait-slice, cooldowns stop chirp spam, variants randomize, registry stems resolve to
the actual committed files, and the unit-test suite itself never touches speakers.
"""

import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

import config
from audio import output_gate, sound_effects as sfx


class _StubSD(types.ModuleType):
    def __init__(self):
        super().__init__("sounddevice")
        self.play_calls = []
        self.stop_calls = 0

    def play(self, audio, samplerate, blocksize=None):
        self.play_calls.append((len(audio), samplerate))

    def stop(self):
        self.stop_calls += 1


class SoundEffectsTest(unittest.TestCase):
    def setUp(self):
        sfx.reset()
        self.sd = _StubSD()
        self._patches = [
            mock.patch.dict("sys.modules", {"sounddevice": self.sd}),
            mock.patch.object(sfx, "_test_allow_audio", True),
            mock.patch.object(config, "SOUND_EFFECTS_ENABLED", True, create=True),
            mock.patch.object(config, "NO_AUDIO_MODE", False, create=True),
            mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False, create=True),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        self.addCleanup(sfx.reset)

    # ── registry integrity ──
    def test_every_registry_stem_resolves_to_a_committed_file(self):
        missing = {
            key: files
            for key, files in sfx.list_effects().items()
            if any(f.startswith("MISSING:") for f in files)
        }
        self.assertEqual(missing, {}, f"registry stems with no file on disk: {missing}")

    def test_real_mp3_decodes(self):
        path = sfx._resolve_stem("motion_whir")
        self.assertIsNotNone(path)
        audio, sr = sfx._decode(path)
        if audio is None:
            self.skipTest("libsndfile without mp3 support on this machine")
        self.assertGreater(audio.size, 1000)
        self.assertGreater(sr, 8000)

    # ── suppression / gating ──
    def test_suite_never_plays_audio_by_default(self):
        with mock.patch.object(sfx, "_test_allow_audio", False):
            self.assertFalse(sfx.play("happy"))

    def test_no_audio_mode_drops(self):
        with mock.patch.object(config, "NO_AUDIO_MODE", True, create=True):
            self.assertFalse(sfx.play("happy"))

    def test_busy_gate_drops_instead_of_queueing(self):
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)):
            with output_gate.hold("test-tts"):
                sfx._play_path(path, "motion_move")     # gated: gate is busy
        self.assertEqual(self.sd.play_calls, [])        # never played, never waited

    # ── concurrent (speech) discipline ──
    def test_play_for_speech_is_concurrent(self):
        # A concurrent effect must NOT hold the gate, so TTS can acquire it instantly
        # while the chirp is "playing" (no blocking, no preemption of the chirp).
        from audio import echo_cancel
        path = sfx._resolve_stem("Droid_Happy_bouncy")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(48000, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing"):
            done = threading.Event()
            t = threading.Thread(
                target=lambda: (sfx._play_path(path, "happy", concurrent=True), done.set()),
                daemon=True)
            t.start()
            time.sleep(0.1)
            self.assertEqual(len(self.sd.play_calls), 1)          # chirp is playing
            # TTS wants the gate — a concurrent chirp NEVER holds it, so this is instant
            # (a blocking hold that actually waited would be the bug).
            with output_gate.hold("tts", blocking=True, timeout=0.05) as acquired:
                self.assertTrue(acquired)
            self.assertEqual(self.sd.stop_calls, 0)               # chirp not preempted
            self.assertTrue(done.wait(2.0))

    def test_concurrent_drops_when_speaker_busy(self):
        path = sfx._resolve_stem("Droid_Happy_bouncy")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)):
            with output_gate.hold("test-tts"):
                sfx._play_path(path, "happy", concurrent=True)    # speaker busy -> drop
        self.assertEqual(self.sd.play_calls, [])

    def test_concurrent_leaves_suppression_to_tts(self):
        # TTS takes the speaker DURING the chirp -> the chirp must not turn mic
        # suppression off at its end (TTS owns _playing now).
        from audio import echo_cancel
        path = sfx._resolve_stem("Droid_Happy_bouncy")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            t = threading.Thread(
                target=lambda: sfx._play_path(path, "happy", concurrent=True), daemon=True)
            t.start()
            time.sleep(0.02)                                      # chirp started (idle)
            with output_gate.hold("tts"):                         # TTS takes over mid-chirp
                t.join(1.0)
            self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True])

    def test_concurrent_releases_suppression_when_no_tts_follows(self):
        from audio import echo_cancel
        path = sfx._resolve_stem("Droid_Happy_bouncy")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(480, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            sfx._play_path(path, "happy", concurrent=True)        # idle speaker throughout
            self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True, False])

    # ── preemption ──
    def test_yield_stops_playback_within_a_slice(self):
        path = sfx._resolve_stem("motion_whir")
        fake = (np.zeros(48000 * 3, np.float32), 48000)   # a "3 second" clip
        done = threading.Event()

        def run():
            with mock.patch.object(sfx, "_decode", return_value=fake):
                sfx._play_path(path, "motion_move")
            done.set()

        t = threading.Thread(target=run, daemon=True)
        start = time.monotonic()
        t.start()
        time.sleep(0.15)                     # let it start playing
        sfx.yield_output()                   # a blocking source wants the speaker
        self.assertTrue(done.wait(1.0), "playback did not yield")
        self.assertLess(time.monotonic() - start, 1.0)   # nowhere near the 3s clip
        self.assertEqual(self.sd.stop_calls, 1)
        self.assertFalse(output_gate.is_busy())          # gate released for the speaker

    def test_blocking_hold_fires_yield_hooks(self):
        fired = []
        output_gate.register_yield_hook(lambda: fired.append(1))
        with output_gate.hold("test-tts"):
            pass
        self.assertEqual(fired, [1])
        with output_gate.hold("test-sfx", blocking=False):
            pass
        self.assertEqual(fired, [1])         # non-blocking acquirers do NOT fire hooks

    # ── cooldowns / selection ──
    def _spawned_keys(self, calls):
        return [c.args[1] for c in calls]

    def test_family_cooldown_blocks_rapid_chirps(self):
        with mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play("happy"))
            self.assertFalse(sfx.play("curious"))        # same family, inside cooldown
        self.assertEqual(thread.call_count, 1)

    def test_families_have_independent_cooldowns(self):
        with mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play("happy"))           # speech family
            self.assertTrue(sfx.play("motion_turn"))     # motion family
            self.assertTrue(sfx.play("servo"))           # servo family
            self.assertTrue(sfx.play("headlift_up"))     # headlift family
        self.assertEqual(thread.call_count, 4)

    def test_headlift_family_cooldown_covers_both_directions(self):
        with mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play("headlift_up"))
            self.assertFalse(sfx.play("headlift_down"))  # same family, inside cooldown
            self.assertTrue(sfx.play("servo"))           # servo family unaffected
        self.assertEqual(thread.call_count, 2)

    def test_same_key_dedup_outlasts_family_cooldown(self):
        with mock.patch.object(config, "SOUND_EFFECTS_MOTION_COOLDOWN_SECS", 0.05, create=True), \
                mock.patch.object(sfx.threading, "Thread"):
            self.assertTrue(sfx.play("motion_turn"))
            time.sleep(0.07)                             # family cooldown lapsed…
            self.assertFalse(sfx.play("motion_turn"))    # …but same-key dedup (2x) holds
            self.assertTrue(sfx.play("motion_move"))     # different key OK

    def test_variants_randomize(self):
        chosen = set()
        with mock.patch.object(sfx.threading, "Thread") as thread, \
                mock.patch.object(config, "SOUND_EFFECTS_SPEECH_COOLDOWN_SECS", 0.0, create=True):
            for _ in range(30):
                sfx.reset()
                sfx.play("laughing")
        for c in thread.call_args_list:
            chosen.add(c.kwargs["args"][0].name if "args" in c.kwargs else c.kwargs.get("args", c.args)[0])
        names = {getattr(p, "name", str(p)) for p in
                 [c.kwargs["args"][0] for c in thread.call_args_list]}
        self.assertEqual(len(names), 2, f"expected both laughing variants, got {names}")

    # ── speech hook semantics ──
    def test_neutral_emotion_silent(self):
        self.assertFalse(sfx.play_for_speech("neutral"))
        self.assertFalse(sfx.play_for_speech(""))
        self.assertFalse(sfx.play_for_speech(None))

    def test_known_emotions_fire(self):
        with mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play_for_speech("happy"))
        self.assertEqual(thread.call_count, 1)

    def test_unknown_emotion_silent(self):
        self.assertFalse(sfx.play_for_speech("melancholic-jazz"))

    def test_registry_override_wins(self):
        with mock.patch.object(config, "SOUND_EFFECTS_EMOTION_MAP_OVERRIDES",
                               {"happy": ["motion_whir"]}, create=True), \
                mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play("happy"))
        path = thread.call_args.kwargs["args"][0]
        self.assertEqual(path.stem, "motion_whir")


class HeadliftHumTest(unittest.TestCase):
    """The hardware/servos.move_to hook: hums only on sustained large-travel head-lift
    sweeps, in normal operation, past the startup mute."""

    HL = None  # headlift channel id, resolved in setUp

    def setUp(self):
        from hardware import servos
        from state import State
        self.servos = servos
        self.State = State
        self.HL = servos._channel("headlift")
        self._patches = [
            mock.patch.object(config, "SOUND_EFFECTS_HEADLIFT_ENABLED", True, create=True),
            mock.patch.object(config, "SOUND_EFFECTS_HEADLIFT_MIN_TRAVEL_QUS", 1200, create=True),
            mock.patch.object(config, "SOUND_EFFECTS_HEADLIFT_STARTUP_MUTE_SECS", 20.0, create=True),
            # process "started" 100s ago -> outside the mute window by default
            mock.patch.object(servos, "_headlift_hum_boot_at", time.monotonic() - 100.0),
            mock.patch("state.get_state", return_value=State.ACTIVE),
            mock.patch("audio.sound_effects.play"),
        ]
        self.patched = [p.start() for p in self._patches]
        self.play = self.patched[-1]
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _hum(self, frm: int, to: int, channel: int | None = None):
        ch = self.HL if channel is None else channel
        self.servos._maybe_headlift_hum({ch: to}, {ch: frm})

    def test_big_lift_up_hums_up(self):
        self._hum(4000, 6000)
        self.play.assert_called_once_with("headlift_up")

    def test_big_lift_down_hums_down(self):
        self._hum(6000, 4000)
        self.play.assert_called_once_with("headlift_down")

    def test_small_travel_stays_silent(self):
        self._hum(6000, 6900)           # 900 qus < 1200 threshold (tracking-scale)
        self.play.assert_not_called()

    def test_other_channels_stay_silent(self):
        self._hum(2000, 8000, channel=self.servos._channel("neck"))
        self.play.assert_not_called()

    def test_startup_mute_window(self):
        with mock.patch.object(self.servos, "_headlift_hum_boot_at", time.monotonic()):
            self._hum(4000, 6000)
        self.play.assert_not_called()

    def test_silent_outside_normal_operation(self):
        for st in (self.State.SLEEP, self.State.QUIET, self.State.SHUTDOWN):
            with mock.patch("state.get_state", return_value=st):
                self._hum(4000, 6000)
        self.play.assert_not_called()

    def test_disable_flag(self):
        with mock.patch.object(config, "SOUND_EFFECTS_HEADLIFT_ENABLED", False, create=True):
            self._hum(4000, 6000)
        self.play.assert_not_called()

if __name__ == "__main__":
    unittest.main()
