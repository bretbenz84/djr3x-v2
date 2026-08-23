"""Short-clip continuity (owner request 2026-08-22): a "mmhmm"-length clip is
whoever was just talking, never a new mystery voice and never a who's-that ask.

Field shape pinned: Bret's 3-word "Yeah, that's Max" scored JT 0.468 / Bret 0.444
on the scan, minted unknown_voice_2 and got a mystery-voice reply, minutes after
Bret was matched at 0.89 on his own prints.
"""

import time
import unittest
from unittest import mock

import numpy as np

import config
from intelligence import interaction


def _audio(secs: float) -> np.ndarray:
    rate = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000) or 16000)
    return np.zeros(int(rate * secs), dtype=np.float32)


class ShortClipDetectTest(unittest.TestCase):
    def test_three_words_is_short_even_with_padded_audio(self):
        self.assertTrue(interaction._is_short_clip("Yeah, that's Max.", _audio(3.0)))

    def test_brief_audio_is_short_regardless_of_words(self):
        self.assertTrue(interaction._is_short_clip("okay okay okay okay okay", _audio(0.9)))

    def test_a_real_sentence_is_not_short(self):
        self.assertFalse(interaction._is_short_clip(
            "You better work. I know that's right.", _audio(2.4)))


class RosterPriorTest(unittest.TestCase):
    FIELD_SCOREBOARD = [(4, "JT", 0.468, 1), (1, "Bret Benziger", 0.444, 5)]

    def setUp(self):
        self._saved = dict(interaction._last_confident_voice_at)
        interaction._last_confident_voice_at.clear()

    def tearDown(self):
        interaction._last_confident_voice_at.clear()
        interaction._last_confident_voice_at.update(self._saved)

    def test_field_shape_resolves_to_the_recently_confident_owner(self):
        interaction._last_confident_voice_at[1] = time.monotonic() - 280
        pick = interaction._short_clip_roster_candidate(self.FIELD_SCOREBOARD)
        self.assertIsNotNone(pick)
        self.assertEqual(pick[0], 1)

    def test_nobody_on_the_roster_means_no_pick(self):
        self.assertIsNone(interaction._short_clip_roster_candidate(self.FIELD_SCOREBOARD))

    def test_roster_member_too_far_below_the_top_is_not_picked(self):
        # Joy's "Yeah, yeah.": JT 0.660 top, Bret 0.488 — Bret is on the roster
        # but trails by 0.17; that is not him.
        interaction._last_confident_voice_at[1] = time.monotonic()
        board = [(4, "JT", 0.660, 1), (1, "Bret Benziger", 0.488, 5)]
        self.assertIsNone(interaction._short_clip_roster_candidate(board))

    def test_roster_member_below_the_floor_is_not_picked(self):
        interaction._last_confident_voice_at[1] = time.monotonic()
        board = [(1, "Bret Benziger", 0.31, 5)]
        self.assertIsNone(interaction._short_clip_roster_candidate(board))

    def test_roster_membership_expires(self):
        interaction._last_confident_voice_at[1] = time.monotonic() - 5000
        self.assertIsNone(interaction._short_clip_roster_candidate(self.FIELD_SCOREBOARD))


class NoSlotFromShortClipTest(unittest.TestCase):
    def setUp(self):
        interaction._clear_anonymous_speaker_slots()

    def tearDown(self):
        interaction._clear_anonymous_speaker_slots()

    def test_short_clip_does_not_mint_a_slot(self):
        emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        with mock.patch.object(interaction.speaker_id, "get_embedding", return_value=emb), \
             mock.patch.object(interaction.voice_signatures, "match", return_value=None):
            label, _score, person = interaction._resolve_anonymous_speaker_slot(
                _audio(1.0), person_id=None, raw_best_id=4, raw_best_name="JT",
                raw_best_score=0.468, short_clip=True,
            )
        self.assertIsNone(label)
        self.assertIsNone(person)
        self.assertEqual(interaction._anonymous_speaker_slots, [])

    def test_short_clip_still_matches_an_existing_slot(self):
        emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
        with mock.patch.object(interaction.speaker_id, "get_embedding", return_value=emb), \
             mock.patch.object(interaction.voice_signatures, "match", return_value=None), \
             mock.patch.object(interaction.voice_signatures, "record", return_value=None):
            first, _s, _p = interaction._resolve_anonymous_speaker_slot(
                _audio(3.0), person_id=None, raw_best_id=4, raw_best_name="JT",
                raw_best_score=0.40, short_clip=False,
            )
            again, _s, _p = interaction._resolve_anonymous_speaker_slot(
                _audio(1.0), person_id=None, raw_best_id=4, raw_best_name="JT",
                raw_best_score=0.40, short_clip=True,
            )
        self.assertEqual(first, "unknown_voice_1")
        self.assertEqual(again, "unknown_voice_1")


class LastSpeakerFallbackTest(unittest.TestCase):
    def setUp(self):
        interaction._clear_anonymous_speaker_slots()

    def tearDown(self):
        interaction._clear_anonymous_speaker_slots()

    def test_recent_person_holds_the_floor(self):
        interaction._note_last_speaker_turn(1, "Bret Benziger", None)
        last = interaction._short_clip_last_speaker()
        self.assertEqual(last["person_id"], 1)

    def test_floor_expires(self):
        interaction._note_last_speaker_turn(1, "Bret Benziger", None)
        interaction._last_speaker_turn["at"] = time.monotonic() - 1000
        self.assertIsNone(interaction._short_clip_last_speaker())

    def test_retired_slot_does_not_hold_the_floor(self):
        interaction._note_last_speaker_turn(None, None, "unknown_voice_1")
        self.assertIsNone(interaction._short_clip_last_speaker())  # no such live slot

    def test_session_reset_clears_the_floor(self):
        interaction._note_last_speaker_turn(1, "Bret Benziger", None)
        interaction._clear_anonymous_speaker_slots()
        self.assertIsNone(interaction._short_clip_last_speaker())


if __name__ == "__main__":
    unittest.main()
