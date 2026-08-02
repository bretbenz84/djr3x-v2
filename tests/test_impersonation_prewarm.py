"""
Impersonation anti-stutter (field 2026-08-01): a long parody line synthesized
slower than real time, and streamed playback starved repeatedly. Pins the fix:
the script is hard-capped, the whole take is prewarmed in the background (the
thinking-sfx loop covers any wait after the intro), and cloned-voice playback
is fully buffered — never streamed.
"""

import threading
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


class PrewarmSlotTest(unittest.TestCase):
    def setUp(self):
        self.ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        local_tts._prewarmed.clear()
        self.addCleanup(local_tts._prewarmed.clear)

    def test_prewarm_then_pop_roundtrip(self):
        audio = np.ones(2400, dtype=np.float32)
        with mock.patch.object(local_tts, "synthesize", return_value=(audio, 24000)):
            done = local_tts.prewarm_take("Test line.", self.ref)
            self.assertTrue(done.wait(3.0))
        got = local_tts.pop_prewarmed("Test line.", self.ref)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], 24000)
        # one-shot
        self.assertIsNone(local_tts.pop_prewarmed("Test line.", self.ref))

    def test_failed_synthesis_still_sets_done(self):
        with mock.patch.object(local_tts, "synthesize", side_effect=RuntimeError("boom")):
            done = local_tts.prewarm_take("Test line.", self.ref)
            self.assertTrue(done.wait(3.0))
        self.assertIsNone(local_tts.pop_prewarmed("Test line.", self.ref))


class BufferedPlaybackTest(unittest.TestCase):
    def test_prewarmed_take_plays_buffered_never_streams(self):
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        audio = np.ones(4800, dtype=np.float32)
        with mock.patch.object(local_tts, "pop_prewarmed", return_value=(audio, 24000)), \
             mock.patch.object(local_tts, "generate_stream") as gen, \
             mock.patch.object(local_tts, "sample_rate", return_value=24000), \
             mock.patch.object(tts, "_play") as play:
            ok = tts._speak_local("Test line.", ref, "excited", log_text=False)
        self.assertTrue(ok)
        play.assert_called_once()
        gen.assert_not_called()

    def test_clone_voice_without_prewarm_synthesizes_fully(self):
        from audio import tts
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "famous:carter")
        audio = np.ones(4800, dtype=np.float32)
        with mock.patch.object(local_tts, "pop_prewarmed", return_value=None), \
             mock.patch.object(local_tts, "synthesize", return_value=(audio, 24000)) as synth, \
             mock.patch.object(local_tts, "generate_stream") as gen, \
             mock.patch.object(local_tts, "sample_rate", return_value=24000), \
             mock.patch.object(tts, "_play") as play:
            ok = tts._speak_local("Test line.", ref, "excited", log_text=False)
        self.assertTrue(ok)
        synth.assert_called_once()
        play.assert_called_once()
        gen.assert_not_called()


class PerformFlowTest(unittest.TestCase):
    def test_thinking_loop_covers_unfinished_prewarm(self):
        ref = local_tts.VoiceRef("/tmp/x.wav", "hello", "person:1")
        done = threading.Event()          # prewarm still running when intro ends
        loop_started = threading.Event()

        def start_loop(key, **kw):
            self.assertEqual(key, "thinking")
            loop_started.set()
            done.set()                    # synth "finishes" while the loop plays
            return object()

        say_done = mock.Mock()
        say_done.wait.return_value = True
        with mock.patch.object(impersonation, "build_parody_script",
                               return_value="I am Bret. I make droids."), \
             mock.patch.object(local_tts, "prewarm_take", return_value=done), \
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


if __name__ == "__main__":
    unittest.main()
