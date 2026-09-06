"""Mixed-capture contracts using synthetic evidence, never a real microphone/model."""
import time
import unittest
from unittest import mock
import numpy as np
from audio import speaker_id as S
from intelligence import attribution as A, turn_coordinator as T
from intelligence.clock_query import TIME_QUERY_RE


class ClockQueryTests(unittest.TestCase):
    def test_reported_two_person_capture_is_not_a_clock_command(self):
        text = "I'm going to the store in the morning. What time are you going to the store?"
        from intelligence import action_router, intent_classifier
        for regex in (TIME_QUERY_RE, action_router._TIME_QUERY_RE, intent_classifier._TIME_QUERY_RE):
            self.assertIsNone(regex.search(text))
            self.assertIsNone(regex.search("What time are you going to the store?"))
            for question in ("What time is it?", "Rex, what time is it now?", "Tell me the time", "What's the current time?", "What time?"):
                self.assertIsNotNone(regex.search(question), question)


class WindowTests(unittest.TestCase):
    def evidence(self, embeddings, rankings):
        with mock.patch.object(S.config, 'AUDIO_SAMPLE_RATE', 100), \
                mock.patch.object(S.config, 'SPEAKER_ID_SIMILARITY_THRESHOLD', .5), \
                mock.patch.object(S, 'voiced_secs', return_value=1), \
                mock.patch.object(S, 'get_embedding', side_effect=embeddings), \
                mock.patch.object(S, 'rank_embedding', side_effect=rankings), \
                mock.patch.object(S, 'required_ambiguity_margin', return_value=.07):
            return S.window_evidence(np.ones(400, dtype=np.float32))

    def test_two_enrolled_voices_have_separate_window_evidence(self):
        rows = self.evidence([np.array([1., 0.]), np.array([0., 1.])],
                             [[(1, 'Bret', .85, 6)], [(2, 'JT', .85, 6)]])
        self.assertEqual([r['person_id'] for r in rows], [1, 2])
        self.assertTrue(rows[1]['change_suspected'])
        self.assertLessEqual(rows[0]['end'], rows[1]['start'])

    def test_unenrolled_second_voice_does_not_inherit_first_name(self):
        rows = self.evidence([np.array([1., 0.]), np.array([0., 1.])],
                             [[(1, 'Bret', .85, 6)], [(1, 'Bret', .30, 6)]])
        self.assertIsNone(rows[1]['person_id'])
        self.assertTrue(rows[1]['change_suspected'])
        ev = A.UtteranceEvidence(raw_best_id=1, raw_best_score=.849, margin=.589,
                                 mixed_speakers=any(r['change_suspected'] for r in rows))
        self.assertIsNone(A.resolve_authoritative(ev).person_id)

    def test_same_voice_is_not_a_switch(self):
        rows = self.evidence([np.array([1., 0.]), np.array([1., 0.])],
                             [[(1, 'Bret', .85, 6)], [(1, 'Bret', .80, 6)]])
        self.assertFalse(any(r['change_suspected'] for r in rows))

    def test_short_clip_does_not_load_encoder(self):
        with mock.patch.object(S, 'get_embedding') as encode:
            self.assertEqual(S.window_evidence(np.zeros(100)), [])
        encode.assert_not_called()


class SplitTests(unittest.TestCase):
    windows = [dict(start=0, end=1.5, person_id=1),
               dict(start=2.5, end=4, person_id=2, change_suspected=True)]

    def test_requires_actual_gap_and_does_not_split_overlap(self):
        self.assertEqual(A.voice_boundaries([(0, 1.9), (2.1, 4)], self.windows), [2.])
        self.assertEqual(A.voice_boundaries([(0, 4)], self.windows), [])

    def test_split_samples_and_capture_order_preserved_before_later_input(self):
        from intelligence import interaction as I
        queue = T.PendingTurns()
        at = time.monotonic()
        queue.put(T.CapturedTurn(np.ones(1), at+1, at+2, 7))
        audio = np.arange(400, dtype=np.float32)
        with mock.patch.object(I.config, 'AUDIO_SAMPLE_RATE', 100), \
                mock.patch.object(I, '_last_scan_windows', self.windows), \
                mock.patch.object(I, '_utterance_observations', {'ended_at': at}), \
                mock.patch.object(I.vad, 'get_speech_segments', return_value=[(0, 1.9), (2.1, 4)]), \
                mock.patch.object(I.conv_memory, 'transcript_version', return_value=(7, 0)), \
                mock.patch.object(I.turn_coordinator, 'pending', queue), \
                mock.patch.object(I, '_note_voice_bearing') as note:
            first = I._split_audio_speakers(audio, require_trusted=True)
        np.testing.assert_array_equal(first, audio[:200])
        later = queue.pop(7)
        np.testing.assert_array_equal(later.audio, audio[200:])
        self.assertTrue(later.require_trusted)
        self.assertAlmostEqual(later.started_at, at-2)
        note.assert_called_once_with(at-4, at-2)
        self.assertEqual(len(queue.pop(7).audio), 1)


if __name__ == '__main__':
    unittest.main()
