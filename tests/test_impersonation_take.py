"""
Impersonation take pipeline. Two field problems, one mechanism:

  * 2026-08-01 — a long parody line synthesized slower than real time and
    chunk-level streamed playback starved repeatedly. Rendering the whole take
    first fixed the stutter but made the room wait for every sentence.
  * 2026-08-03 — takes felt cached: the same bit came back on a repeat request.

Pins the fix: the script is hard-capped and previous takes are fed back as a
do-not-repeat list, the take is split into SENTENCES that render one lookahead
ahead of playback, and a take is a live one-shot object that is never reusable.
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

import config
from audio import local_tts
from features import impersonation


class ScriptCapTest(unittest.TestCase):
    def test_short_script_unchanged(self):
        text = "I am Bret. I love my droid."
        self.assertEqual(impersonation._cap_script_words(text), text)

    def test_long_script_truncated_at_sentence_boundary(self):
        long = " ".join(f"Sentence number {i} has exactly six words." for i in range(12))
        with mock.patch.object(config, "IMPERSONATION_SCRIPT_MAX_WORDS", 20, create=True):
            capped = impersonation._cap_script_words(long)
        self.assertLessEqual(len(capped.split()), 20)
        self.assertTrue(capped.endswith("."))

    def test_single_giant_sentence_is_kept_whole(self):
        giant = "word " * 80
        with mock.patch.object(config, "IMPERSONATION_SCRIPT_MAX_WORDS", 20, create=True):
            self.assertEqual(impersonation._cap_script_words(giant.strip()), giant.strip())


def _pipelined(test):
    """Turn sentence pipelining back on for a test that is about pipelining.

    It ships OFF (LOCAL_TTS_TAKE_WHOLE_CLIP): every unit is a separate
    conditioning pass on the reference clip and the passes do not match, so a
    split take changed voice partway through the bit.
    """
    prev = getattr(config, "LOCAL_TTS_TAKE_WHOLE_CLIP", True)
    config.LOCAL_TTS_TAKE_WHOLE_CLIP = False
    test.addCleanup(setattr, config, "LOCAL_TTS_TAKE_WHOLE_CLIP", prev)


class SplitTakeTest(unittest.TestCase):
    def test_whole_clip_is_the_default_so_the_voice_cannot_drift(self):
        text = ("I rewired the neck servo again. It still judges me from the shelf. "
                "Ask me tomorrow if it was worth it.")
        self.assertEqual(local_tts._split_take(text), [text])

    def test_splits_on_sentence_boundaries(self):
        _pipelined(self)
        units = local_tts._split_take(
            "I rewired the neck servo again. It still judges me from the shelf. "
            "Ask me tomorrow if it was worth it."
        )
        self.assertEqual(units, [
            "I rewired the neck servo again.",
            "It still judges me from the shelf.",
            "Ask me tomorrow if it was worth it.",
        ])

    def test_short_fragment_merges_into_the_next_sentence(self):
        _pipelined(self)
        units = local_tts._split_take("Six! And it still looks at me like I'm the problem.")
        self.assertEqual(units, ["Six! And it still looks at me like I'm the problem."])

    def test_empty_text_yields_no_units(self):
        self.assertEqual(local_tts._split_take("   "), [])


class TakePipelineTest(unittest.TestCase):
    """The point of the pipeline: unit 1 is playable before unit 2 is rendered.

    Pipelining is off by default now (see _pipelined) — these tests still cover
    it because the machinery is intact behind LOCAL_TTS_TAKE_WHOLE_CLIP=False.
    """

    def setUp(self):
        _pipelined(self)
        self.ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        self.addCleanup(local_tts.discard_takes)

    def test_first_unit_ready_before_later_units_render(self):
        gate = threading.Event()
        rendered = []

        def fake_unit(text, voice_ref):
            rendered.append(text)
            if len(rendered) > 1:
                gate.wait(3.0)          # unit 2+ blocks until the test releases it
            return np.ones(2400, dtype=np.float32)

        with mock.patch.object(local_tts, "_synthesize_unit", side_effect=fake_unit):
            take = local_tts.Take(
                "The first sentence lives here. The second sentence lives here. "
                "The third sentence lives here.",
                self.ref,
            )
            self.addCleanup(take.close)
            self.assertTrue(take.first_ready.wait(3.0))
            self.assertFalse(take.failed)
            # Unit 2 has not finished, yet the take is already playable.
            self.assertLessEqual(len(rendered), 2)
            gate.set()
            chunks = list(take.stream())
        # Three units of real audio (plus any silence fills written at a seam).
        real = [c for c in chunks if float(np.max(np.abs(c))) > 0.0]
        self.assertEqual(len(real), 3)

    def test_seam_wait_emits_silence_not_a_gap_in_the_stream(self):
        release = threading.Event()

        def fake_unit(text, voice_ref):
            if text.startswith("The second"):
                release.wait(3.0)
            return np.ones(2400, dtype=np.float32)

        with mock.patch.object(config, "LOCAL_TTS_TAKE_FILL_MS", 20.0, create=True), \
             mock.patch.object(local_tts, "_synthesize_unit", side_effect=fake_unit):
            take = local_tts.Take(
                "The first sentence lives here. The second sentence lives here.", self.ref
            )
            self.addCleanup(take.close)
            stream = take.stream()
            first = next(stream)
            self.assertGreater(float(np.max(np.abs(first))), 0.0)
            # Unit 2 is still rendering — the stream keeps the device fed.
            fills = 0
            for _ in range(3):
                chunk = next(stream)
                if float(np.max(np.abs(chunk))) == 0.0:
                    fills += 1
                else:
                    break
            release.set()
            self.assertGreaterEqual(fills, 1)
            stream.close()

    def test_nothing_is_emitted_before_the_first_real_unit(self):
        """Silence fills before unit 1 would satisfy the player's preroll and start
        playback on dead air — so the stream must block instead."""
        release = threading.Event()

        def fake_unit(text, voice_ref):
            release.wait(3.0)
            return np.ones(2400, dtype=np.float32)

        with mock.patch.object(config, "LOCAL_TTS_TAKE_FILL_MS", 20.0, create=True), \
             mock.patch.object(local_tts, "_synthesize_unit", side_effect=fake_unit):
            take = local_tts.Take("Only one sentence here.", self.ref)
            self.addCleanup(take.close)
            stream = take.stream()
            got = []
            reader = threading.Thread(target=lambda: got.append(next(stream)), daemon=True)
            reader.start()
            time.sleep(0.15)             # several fill intervals
            self.assertEqual(got, [])
            release.set()
            reader.join(3.0)
            self.assertEqual(len(got), 1)
            stream.close()

    def test_failed_synthesis_sets_first_ready_and_failed(self):
        with mock.patch.object(local_tts, "_synthesize_unit", side_effect=RuntimeError("boom")):
            take = local_tts.Take("Test line.", self.ref)
            self.addCleanup(take.close)
            self.assertTrue(take.first_ready.wait(3.0))
        self.assertTrue(take.failed)
        self.assertEqual(list(take.stream()), [])

    def test_close_stops_the_renderer(self):
        seen = []
        keep_going = threading.Event()

        def fake_unit(text, voice_ref):
            seen.append(text)
            keep_going.wait(0.05)
            return np.ones(2400, dtype=np.float32)

        with mock.patch.object(local_tts, "_synthesize_unit", side_effect=fake_unit):
            take = local_tts.Take(
                ". ".join(f"Sentence number {i} is here" for i in range(8)) + ".", self.ref
            )
            self.assertTrue(take.first_ready.wait(3.0))
            take.close()
            take._thread.join(3.0)
            self.assertFalse(take._thread.is_alive())
        self.assertLess(len(seen), 8)


class TakeSlotTest(unittest.TestCase):
    """One-shot parking: a take is claimed once and can never be replayed."""

    def setUp(self):
        self.ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        local_tts.discard_takes()
        self.addCleanup(local_tts.discard_takes)

    def test_start_then_pop_roundtrip_is_one_shot(self):
        with mock.patch.object(local_tts, "_synthesize_unit",
                               return_value=np.ones(2400, dtype=np.float32)):
            take = local_tts.start_take("Test line.", self.ref)
            self.addCleanup(take.close)
            self.assertIs(local_tts.pop_take("Test line.", self.ref), take)
            self.assertIsNone(local_tts.pop_take("Test line.", self.ref))

    def test_starting_a_take_closes_whatever_was_left_parked(self):
        with mock.patch.object(local_tts, "_synthesize_unit",
                               return_value=np.ones(2400, dtype=np.float32)):
            stale = local_tts.start_take("Old bit.", self.ref)
            fresh = local_tts.start_take("New bit.", self.ref)
            self.addCleanup(fresh.close)
        self.assertTrue(stale._stop.is_set())
        self.assertIsNone(local_tts.pop_take("Old bit.", self.ref))
        self.assertIs(local_tts.pop_take("New bit.", self.ref), fresh)


class PipelinedPlaybackTest(unittest.TestCase):
    def test_clone_voice_streams_the_parked_take(self):
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        take = mock.Mock()
        take.stream.return_value = iter([np.ones(4800, dtype=np.float32)])
        with mock.patch.object(local_tts, "pop_take", return_value=take), \
             mock.patch.object(local_tts, "generate_stream") as gen, \
             mock.patch.object(local_tts, "synthesize") as synth, \
             mock.patch.object(local_tts, "sample_rate", return_value=24000), \
             mock.patch.object(tts, "output_gate") as gate:
            gate.hold.return_value.__enter__.return_value = False   # skip real playback
            ok = tts._speak_local("Test line.", ref, "excited", log_text=False)
        self.assertTrue(ok)
        take.stream.assert_called_once()
        gen.assert_not_called()
        synth.assert_not_called()

    def test_clone_voice_without_a_parked_take_starts_one(self):
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "famous:carter")
        made = []

        def fake_take(text, voice_ref):
            take = mock.Mock()
            take.stream.return_value = iter([np.ones(4800, dtype=np.float32)])
            made.append(take)
            return take

        with mock.patch.object(local_tts, "pop_take", return_value=None), \
             mock.patch.object(local_tts, "Take", side_effect=fake_take), \
             mock.patch.object(local_tts, "generate_stream") as gen, \
             mock.patch.object(local_tts, "synthesize") as synth, \
             mock.patch.object(local_tts, "sample_rate", return_value=24000), \
             mock.patch.object(tts, "output_gate") as gate:
            gate.hold.return_value.__enter__.return_value = False
            ok = tts._speak_local("Test line.", ref, "excited", log_text=False)
        self.assertTrue(ok)
        self.assertEqual(len(made), 1)
        gen.assert_not_called()
        synth.assert_not_called()

    def test_rex_own_voice_keeps_the_chunk_stream(self):
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "rex")
        with mock.patch.object(local_tts, "pop_take") as pop, \
             mock.patch.object(local_tts, "Take") as take_cls, \
             mock.patch.object(local_tts, "generate_stream",
                               return_value=iter([np.ones(4800, dtype=np.float32)])) as gen, \
             mock.patch.object(local_tts, "sample_rate", return_value=24000), \
             mock.patch.object(tts, "output_gate") as gate:
            gate.hold.return_value.__enter__.return_value = False
            ok = tts._speak_local("Test line.", ref, "excited", log_text=False)
        self.assertTrue(ok)
        gen.assert_called_once()
        pop.assert_not_called()
        take_cls.assert_not_called()


class PerformFlowTest(unittest.TestCase):
    def test_thinking_loop_covers_the_first_sentence_only(self):
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        take = mock.Mock()
        take.first_ready = threading.Event()      # unit 1 still rendering at intro's end
        take.failed = False
        loop_started = threading.Event()

        def start_loop(key, **kw):
            self.assertEqual(key, "thinking")
            loop_started.set()
            take.first_ready.set()                # unit 1 lands while the loop plays
            return object()

        say_done = mock.Mock()
        say_done.wait.return_value = True
        with mock.patch.object(impersonation, "build_parody_script",
                               return_value="I am Bret. I make droids."), \
             mock.patch.object(local_tts, "start_take", return_value=take), \
             mock.patch("audio.speech_queue.enqueue", return_value=say_done) as enq, \
             mock.patch("audio.sound_effects.start_loop", side_effect=start_loop), \
             mock.patch("audio.sound_effects.stop_loop") as stop_loop, \
             mock.patch("memory.episodes.record_episode", create=True), \
             mock.patch.object(config, "LOCAL_TTS_MODE", False, create=True):
            script = impersonation.perform(ref, "Bret", 1, is_self=True)
        self.assertEqual(script, "I am Bret. I make droids.")
        self.assertTrue(loop_started.is_set())
        stop_loop.assert_called_once()
        # The parody line was enqueued with the cloned voice ref.
        voiced = [c for c in enq.call_args_list if c.kwargs.get("voice_ref") is not None]
        self.assertEqual(len(voiced), 1)
        self.assertEqual(voiced[0].kwargs["voice_ref"], ref)
        # Released once the line is done — an unclaimed take must not keep rendering.
        take.close.assert_called_once()

    def test_take_is_started_on_the_exact_text_the_player_will_look_up(self):
        """perform() must key the take on the synthesized form, or _speak_local
        looks up a key that was never parked and silently re-renders."""
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        script = "[excited] I  am Bret.   I fought in WWII."
        take = mock.Mock()
        take.first_ready = threading.Event()
        take.first_ready.set()
        take.failed = False
        say_done = mock.Mock()
        say_done.wait.return_value = True
        with mock.patch.object(impersonation, "build_parody_script", return_value=script), \
             mock.patch.object(local_tts, "start_take", return_value=take) as start, \
             mock.patch("audio.speech_queue.enqueue", return_value=say_done) as enq, \
             mock.patch("memory.episodes.record_episode", create=True), \
             mock.patch.object(config, "LOCAL_TTS_MODE", False, create=True):
            impersonation.perform(ref, "Bret", 1, is_self=True)
        started_text = start.call_args.args[0]
        spoken = [c for c in enq.call_args_list if c.kwargs.get("voice_ref") is not None]
        self.assertEqual(started_text, tts.spoken_form(spoken[0].args[0]))

    def test_failed_take_covers_in_rex_voice(self):
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        take = mock.Mock()
        take.first_ready = threading.Event()
        take.first_ready.set()
        take.failed = True
        say_done = mock.Mock()
        say_done.wait.return_value = True
        with mock.patch.object(impersonation, "build_parody_script",
                               return_value="I am Bret."), \
             mock.patch.object(local_tts, "start_take", return_value=take), \
             mock.patch("audio.speech_queue.enqueue", return_value=say_done) as enq, \
             mock.patch("memory.episodes.record_episode") as rec, \
             mock.patch.object(config, "LOCAL_TTS_MODE", False, create=True):
            result = impersonation.perform(ref, "Bret", 1, is_self=True)
        self.assertIn("fuse", result.lower())
        take.close.assert_called_once()
        rec.assert_not_called()
        for call in enq.call_args_list:
            self.assertIsNone(call.kwargs.get("voice_ref"))


class ScriptFreshnessTest(unittest.TestCase):
    """A repeat request must not come back with the previous bit."""

    def test_recent_scripts_feed_the_do_not_repeat_list(self):
        rows = [
            {"person_id": 1, "detail": '{"subject": "Bret", "script": "I am always late."}'},
            {"person_id": 2, "detail": '{"subject": "JT", "script": "I play volleyball."}'},
            {"person_id": 1, "detail": '{"subject": "Bret", "script": "I solder at midnight."}'},
        ]
        with mock.patch("memory.episodes.recent_episodes", return_value=rows):
            prior = impersonation._recent_scripts("Bret", 1)
        self.assertEqual(prior, ["I am always late.", "I solder at midnight."])

    def test_famous_subject_matches_by_name_not_person_id(self):
        rows = [
            {"person_id": None, "detail": '{"subject": "Jimmy Carter", "script": "Peanuts."}'},
            {"person_id": None, "detail": '{"subject": "Someone Else", "script": "Nope."}'},
        ]
        with mock.patch("memory.episodes.recent_episodes", return_value=rows):
            prior = impersonation._recent_scripts("jimmy carter", None)
        self.assertEqual(prior, ["Peanuts."])

    def test_prompt_carries_the_avoid_block(self):
        prompt = impersonation._script_prompt(
            "Bret", ["likes droids"], [], is_self=True, famous=False,
            avoid=["I am always late."],
        )
        self.assertIn("DIFFERENT bit", prompt)
        self.assertIn("I am always late.", prompt)


if __name__ == "__main__":
    unittest.main()
