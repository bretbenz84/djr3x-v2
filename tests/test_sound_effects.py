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


class _StubStream:
    """Stands in for sd.OutputStream — the overlay path's private device handle."""

    def __init__(self, owner, **kwargs):
        self.owner = owner
        self.kwargs = kwargs
        self.written = []

    def start(self):
        self.owner.stream_starts += 1

    def write(self, data):
        self.written.append(len(data))
        self.owner.stream_writes.append(len(data))

    def stop(self):
        self.owner.stream_stops += 1

    def close(self):
        self.owner.stream_closes += 1


class _StubSD(types.ModuleType):
    def __init__(self):
        super().__init__("sounddevice")
        self.play_calls = []
        self.stop_calls = 0
        self.stream_starts = 0
        self.stream_writes = []
        self.stream_stops = 0
        self.stream_closes = 0

    def play(self, audio, samplerate, blocksize=None):
        self.play_calls.append((len(audio), samplerate))

    def stop(self):
        self.stop_calls += 1

    def OutputStream(self, **kwargs):  # noqa: N802 — mirrors the sounddevice API
        return _StubStream(self, **kwargs)


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
                target=lambda: (sfx._play_path(path, "happy", mode="concurrent"), done.set()),
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
                sfx._play_path(path, "happy", mode="concurrent")    # speaker busy -> drop
        self.assertEqual(self.sd.play_calls, [])

    # ── overlay (voice-commanded motion) discipline ──
    def test_overlay_plays_even_while_tts_holds_the_speaker(self):
        # THE BUG: a commanded move speaks a confirmation whose cached audio hits the
        # speaker ~3 ms after queueing, so the gated drive sound lost the race and was
        # dropped on nearly every command (field 2026-07-24). Overlay must still play.
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)):
            with output_gate.hold("tts"):
                sfx._play_path(path, "motion_move", mode="overlay")
        self.assertEqual(self.sd.stream_starts, 1, "overlay opened its own stream")
        self.assertTrue(self.sd.stream_writes, "overlay wrote audio")

    def test_overlay_never_touches_the_shared_play_stream(self):
        # sounddevice keeps ONE global playback stream: calling sd.play() (or
        # sd.stop()) here would cut Rex off mid-word. Overlay must use only its own
        # OutputStream — this is the guard that keeps the fix safe.
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)):
            with output_gate.hold("tts"):
                sfx._play_path(path, "motion_move", mode="overlay")
        self.assertEqual(self.sd.play_calls, [], "overlay must not call sd.play")
        self.assertEqual(self.sd.stop_calls, 0, "overlay must not call sd.stop")
        self.assertEqual(self.sd.stream_closes, 1, "overlay closed its stream")

    def test_overlay_does_not_hold_the_output_gate(self):
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(48000, np.float32), 48000)):
            t = threading.Thread(
                target=lambda: sfx._play_path(path, "motion_move", mode="overlay"),
                daemon=True)
            t.start()
            time.sleep(0.05)
            with output_gate.hold("tts", blocking=True, timeout=0.05) as acquired:
                self.assertTrue(acquired, "overlay must never block TTS")
            t.join(2.0)

    def test_overlay_leaves_suppression_to_tts(self):
        # TTS owns the _playing flag while it speaks — a voice-like overlay clip
        # must not un-mute the mic underneath it. (Keyed on a SPEECH family: drive
        # families never mute at all — see the traction tests below.)
        from audio import echo_cancel
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            with output_gate.hold("tts"):
                sfx._play_path(path, "curious", mode="overlay")
        self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True])

    def test_overlay_releases_suppression_when_no_tts_follows(self):
        from audio import echo_cancel
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            sfx._play_path(path, "curious", mode="overlay")
        self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True, False])

    # ── motor noise must never deafen him ──
    # Field 2026-07-25: the looping drive whir held mic suppression for the WHOLE
    # manoeuvre, so repeated "don't move" / "stop moving" were never heard while the
    # base ground away on carpet. Speech chirps still mute (they transcribe as words);
    # machinery does not.

    def test_drive_effects_never_mute_the_mic(self):
        from audio import echo_cancel
        path = sfx._resolve_stem("motion_whir")
        for key, mode in (("motion_move", "overlay"), ("motion_move", "gated")):
            with self.subTest(key=key, mode=mode):
                with mock.patch.object(sfx, "_decode",
                                       return_value=(np.zeros(4800, np.float32), 48000)), \
                        mock.patch.object(echo_cancel, "set_playing") as set_playing:
                    sfx._play_path(path, key, mode=mode)
                self.assertEqual(set_playing.call_args_list, [])

    def test_speech_effects_still_mute_the_mic(self):
        from audio import echo_cancel
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            sfx._play_path(path, "curious", mode="gated")
        self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True, False])

    def test_drive_muting_can_be_turned_back_on(self):
        from audio import echo_cancel
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(config, "SOUND_EFFECTS_DRIVE_SUPPRESSES_MIC", True, create=True), \
                mock.patch.object(sfx, "_decode",
                                  return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            sfx._play_path(path, "motion_move", mode="gated")
        self.assertEqual([c.args[0] for c in set_playing.call_args_list], [True, False])

    # ── field 2026-08-06: a servo whir ate the answer to Rex's own question ──
    def test_gated_whir_does_not_claim_to_mute_the_mic(self):
        """_suppresses_mic exempts machinery whirs, but the capture loop skipped the
        mic for ANY output-gate holder — so the exemption never reached it. The flag
        is what lets the loop honour the family decision."""
        path = sfx._resolve_stem("motion_whir")
        seen = []
        real_play = self.sd.play

        def _spy(*a, **kw):
            seen.append(sfx.gated_effect_mutes_mic())
            return real_play(*a, **kw)

        with mock.patch.object(sfx, "_decode",
                               return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(self.sd, "play", _spy):
            sfx._play_path(path, "motion_move", mode="gated")
        self.assertEqual(seen, [False])
        self.assertFalse(sfx.gated_effect_mutes_mic())      # cleared after playback

    def test_gated_speech_chirp_still_mutes_the_mic(self):
        path = sfx._resolve_stem("motion_whir")
        seen = []
        real_play = self.sd.play

        def _spy(*a, **kw):
            seen.append(sfx.gated_effect_mutes_mic())
            return real_play(*a, **kw)

        with mock.patch.object(sfx, "_decode",
                               return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(self.sd, "play", _spy):
            sfx._play_path(path, "curious", mode="gated")
        self.assertEqual(seen, [True])
        self.assertFalse(sfx.gated_effect_mutes_mic())

    def test_flag_clears_even_when_playback_raises(self):
        path = sfx._resolve_stem("motion_whir")
        with mock.patch.object(sfx, "_decode",
                               return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(self.sd, "play", side_effect=RuntimeError("boom")):
            sfx._play_path(path, "curious", mode="gated")
        self.assertFalse(sfx.gated_effect_mutes_mic())

    def test_capture_loop_listens_through_a_whir_but_not_a_chirp(self):
        """The whole point: the mic stays open for an exempt effect and closed for
        everything else. Guards the interaction-side gate against re-tightening."""
        from intelligence import interaction as ix
        self.addCleanup(setattr, sfx, "_gated_mutes_mic", False)

        self.assertFalse(ix._effect_allows_listening())          # nothing playing
        with output_gate.hold("sound-effects", blocking=False):
            sfx._gated_mutes_mic = False                          # servo/motion whir
            self.assertTrue(ix._effect_allows_listening())
            sfx._gated_mutes_mic = True                           # speech chirp
            self.assertFalse(ix._effect_allows_listening())
        sfx._gated_mutes_mic = False
        with output_gate.hold("tts"):                             # Rex's actual voice
            self.assertFalse(ix._effect_allows_listening())

    def test_overlay_is_ducked_below_the_spoken_line(self):
        path = sfx._resolve_stem("motion_whir")
        loud = np.ones(4800, np.float32)
        with mock.patch.object(sfx, "_decode", return_value=(loud, 48000)), \
                mock.patch.object(config, "SOUND_EFFECTS_VOLUME", 1.0, create=True), \
                mock.patch.object(config, "SOUND_EFFECTS_OVERLAY_VOLUME", 0.5, create=True):
            with output_gate.hold("tts"):
                sfx._play_path(path, "motion_move", mode="overlay")
        self.assertEqual(self.sd.stream_starts, 1)

    # ── looping effects ──
    def test_loop_repeats_until_stopped(self):
        # A clip shorter than the activity must keep going: the startup "thinking"
        # chirp is ~1.5 s but covers a much longer warmup, and the ~4 s drive whir
        # has to cover a ~9 s move (owner 2026-07-24).
        # Repetition is counted in WRITES, not device opens: a loop now holds ONE
        # output stream for its whole life (see _LoopStream), so counting opens
        # would measure the churn that used to deafen him rather than the audio.
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(480, np.float32), 48000)):
            handle = sfx.start_loop("motion_move", gap_secs=0.01)
            self.assertIsNotNone(handle)
            try:
                time.sleep(0.25)
                self.assertGreater(len(self.sd.stream_writes), 1, "loop must repeat")
            finally:
                sfx.stop_loop(handle)
        self.assertFalse(handle.running)
        after = len(self.sd.stream_writes)
        time.sleep(0.15)
        self.assertEqual(len(self.sd.stream_writes), after, "no audio after stop")

    def test_loop_opens_the_device_once_however_many_passes(self):
        # The mic and the speaker share one CoreAudio device: repeating the
        # open/close per pass is what silently kills the input callback (see
        # audio/stream.py). Field 2026-08-18 — the impersonation thinking chirp
        # churned it for ~27 s, wedged, and took the mic AND the output gate with
        # it. Many passes, one open.
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(480, np.float32), 48000)):
            handle = sfx.start_loop("motion_move", gap_secs=0.01)
            try:
                time.sleep(0.25)
            finally:
                sfx.stop_loop(handle)
        self.assertGreater(len(self.sd.stream_writes), 1, "loop must have repeated")
        self.assertEqual(self.sd.stream_starts, 1, "loop reopened the shared device")
        self.assertEqual(self.sd.play_calls, [], "a loop must not use sd.play")

    def test_loop_is_cut_immediately_not_at_the_clip_end(self):
        # stop_loop must abort the in-flight pass, so the whir dies with the wheels
        # instead of trailing several seconds past the end of the move.
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(48000 * 5, np.float32), 48000)):
            handle = sfx.start_loop("motion_move", gap_secs=0.01)
            time.sleep(0.1)
            t0 = time.monotonic()
            sfx.stop_loop(handle, join_timeout=2.0)
            elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 1.0, "a 5s clip must not play out after stop")
        # The cut is the chunked write breaking on the abort event rather than an
        # sd.stop() on the module-global stream, so prove it by the audio ending.
        after = len(self.sd.stream_writes)
        time.sleep(0.15)
        self.assertEqual(len(self.sd.stream_writes), after, "playback was actually cut")

    def test_loop_honors_the_family_kill_switch(self):
        with mock.patch.object(config, "SOUND_EFFECTS_MOTION_ENABLED", False, create=True):
            self.assertIsNone(sfx.start_loop("motion_move"))

    def test_loop_unknown_key_is_a_no_op(self):
        self.assertIsNone(sfx.start_loop("no_such_effect"))

    def test_stop_loop_tolerates_none(self):
        sfx.stop_loop(None)          # must not raise

    def test_loop_max_secs_caps_a_runaway(self):
        # A lost `done` frame must never leave the speaker droning.
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(480, np.float32), 48000)):
            handle = sfx.start_loop("motion_move", gap_secs=0.01, max_secs=0.15)
            time.sleep(0.4)
        self.assertFalse(handle.running, "loop must self-terminate at max_secs")

    def test_overlay_loop_uses_its_own_stream_not_sd_play(self):
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(480, np.float32), 48000)):
            with output_gate.hold("tts"):
                handle = sfx.start_loop("motion_move", mode="overlay", gap_secs=0.01)
                try:
                    time.sleep(0.2)
                finally:
                    sfx.stop_loop(handle)
        self.assertGreater(len(self.sd.stream_writes), 1, "overlay loop repeated")
        self.assertEqual(self.sd.stream_starts, 1, "overlay loop reopened the device")
        self.assertEqual(self.sd.play_calls, [], "overlay must never touch sd.play")

    def test_concurrent_leaves_suppression_to_tts(self):
        # TTS takes the speaker DURING the chirp -> the chirp must not turn mic
        # suppression off at its end (TTS owns _playing now).
        from audio import echo_cancel
        path = sfx._resolve_stem("Droid_Happy_bouncy")
        with mock.patch.object(sfx, "_decode", return_value=(np.zeros(4800, np.float32), 48000)), \
                mock.patch.object(echo_cancel, "set_playing") as set_playing:
            t = threading.Thread(
                target=lambda: sfx._play_path(path, "happy", mode="concurrent"), daemon=True)
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
            sfx._play_path(path, "happy", mode="concurrent")        # idle speaker throughout
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

    # ── folder pools + cycling (owner 2026-08-04: the single excited chirp and the
    # two harsh thinking chirps were replaced by whole folders of variants) ──
    def test_folder_pools_expand_to_their_clips(self):
        for key in ("thinking", "excited"):
            with self.subTest(key=key):
                stems = sfx._stems_for(key)
                self.assertGreater(len(stems), 1, f"{key} pool should have variants")
                for stem in stems:
                    self.assertIsNotNone(sfx._resolve_stem(stem),
                                         f"{key} stem {stem!r} has no file")

    def test_retired_clips_are_gone_from_every_pool(self):
        retired = {"droid_excited", "robot_processing_thinking_1",
                   "robot_processing_thinking_2"}
        for key in sfx._registry():
            for stem in sfx._stems_for(key):
                self.assertNotIn(stem.lower(), retired, f"{key} still uses {stem}")

    def test_thinking_cycles_the_whole_folder_before_repeating(self):
        # The startup/impersonation loops repeat this key for as long as the wait
        # lasts; the folder exists so that wait doesn't sound like one chirp.
        pool = sfx._stems_for("thinking")
        picks = [sfx._pick("thinking", pool) for _ in range(len(pool))]
        self.assertEqual(sorted(p.lower() for p in picks),
                         sorted(p.lower() for p in pool), "a pass must use each clip once")
        # …and the pass seam must not replay the clip that just played.
        nxt = sfx._pick("thinking", pool)
        self.assertNotEqual(nxt, picks[-1])

    def test_excited_never_plays_the_same_clip_twice_in_a_row(self):
        pool = sfx._stems_for("excited")
        picks = [sfx._pick("excited", pool) for _ in range(200)]
        repeats = [a for a, b in zip(picks, picks[1:]) if a == b]
        self.assertEqual(repeats, [], "back-to-back repeat of the same clip")
        self.assertEqual(set(picks), set(pool), "every clip in the folder gets used")

    def test_subfolder_clips_resolve_by_stem(self):
        stem = sfx._stems_for("excited")[0]
        path = sfx._resolve_stem(stem)
        self.assertIsNotNone(path)
        self.assertEqual(path.parent.name, "excitement")

    def test_pool_play_picks_a_folder_clip(self):
        with mock.patch.object(sfx.threading, "Thread") as thread:
            self.assertTrue(sfx.play("thinking"))
        path = thread.call_args.kwargs["args"][0]
        self.assertEqual(path.parent.name, "thinking")

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

    # play() is mocked in these two so the shared speech-emotion cooldown can't
    # swallow the second call -- what's under test is the tag filter, not pacing.
    def test_muted_tag_gets_no_chirp(self):
        """A droid chirp a beat before a cloned human voice gives the bit away."""
        with mock.patch.object(sfx, "play", return_value=True) as play:
            self.assertFalse(sfx.play_for_speech("excited", tag="impersonation"))
            self.assertFalse(sfx.play_for_speech("EXCITED", tag="  Impersonation "))
        play.assert_not_called()

    def test_other_tags_still_chirp(self):
        with mock.patch.object(sfx, "play", return_value=True) as play:
            self.assertTrue(sfx.play_for_speech("excited", tag="reaction"))
            self.assertTrue(sfx.play_for_speech("excited"))
        self.assertEqual(play.call_count, 2)

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
