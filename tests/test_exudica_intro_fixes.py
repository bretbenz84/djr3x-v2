"""
Regression coverage for the Exudica Royale introduction run (live-logged
2026-06-15, logs/djr3x-2026-06-15-21-24-49.log).

Three independent failures conspired so a pre-populated friend, "Exudica
Royale", was forked into a throwaway "Exutica" record, never got a voice print,
and had her hello credited to the introducer:

  1. Whisper mis-heard the soft 'd' as 't'/'g' (and once as the real word
     "exotica") — WHISPER_CORRECTIONS now normalizes those.
  2. A single mis-heard first name never fuzzy-matched the stored FULL name
     ("exutica" vs "exudica royale" = 0.57, under threshold) because the surname
     dragged the whole-string ratio down — find_potential_person_match now has a
     first-token "fuzzy_first_name" tier.
  3. The newcomer's first reply ("hi, what's your name?") looked like a direct
     turn to Rex, so the pending intro voice-capture window was cleared before
     the enrollment handler ran; and even reached, its bar (0.50) was below the
     newcomer-as-introducer score (~0.64). Both are fixed.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from memory import database as db


class WhisperCorrectionTest(unittest.TestCase):
    def test_exudica_mishearings_corrected(self):
        from audio.transcription import _apply_corrections

        self.assertEqual(
            _apply_corrections("introduce you to my friend Exutica"),
            "introduce you to my friend Exudica",
        )
        self.assertEqual(_apply_corrections("I said exutiga"), "I said Exudica")
        self.assertEqual(_apply_corrections("I'm exotica"), "I'm Exudica")
        self.assertEqual(_apply_corrections("EXUTICA"), "Exudica")

    def test_existing_bret_correction_untouched(self):
        from audio.transcription import _apply_corrections

        self.assertEqual(_apply_corrections("good morning Brett"), "good morning Bret")


class _TempPeopleDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA

        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()


class FuzzyFirstNameMatchTest(_TempPeopleDb):
    def test_misheard_first_name_links_to_full_name_person(self):
        from memory import people

        exudica = people.enroll_person("Exudica Royale")
        self.assertIsNotNone(exudica)

        match = people.find_potential_person_match("Exutica")
        self.assertIsNotNone(match)
        self.assertEqual(match["match_type"], "fuzzy_first_name")
        self.assertEqual(match["person"]["id"], exudica)

        # find_or_create_person REUSES the existing row (created=False) instead of
        # forking a near-duplicate "Exutica".
        pid, created = people.find_or_create_person("Exutica")
        self.assertEqual(pid, exudica)
        self.assertFalse(created)

    def test_corrected_name_links_via_exact_first_token(self):
        from memory import people

        exudica = people.enroll_person("Exudica Royale")
        pid, created = people.find_or_create_person("Exudica")  # post-correction
        self.assertEqual(pid, exudica)
        self.assertFalse(created)

    def test_mishearing_is_not_persisted_as_alias(self):
        from memory import people

        exudica = people.enroll_person("Exudica Royale")
        people.find_or_create_person("Exutica")
        # The mishearing must not become a permanent alias of Exudica.
        self.assertIsNone(people.find_person_by_name("Exutica"))

    def test_ambiguous_first_token_does_not_guess(self):
        from memory import people

        people.enroll_person("Exudica Royale")
        people.enroll_person("Exudica Hart")  # same first token -> two targets
        # Two plausible targets -> refuse to guess which one.
        match = people.find_potential_person_match("Exutica")
        self.assertNotEqual((match or {}).get("match_type"), "fuzzy_first_name")

    def test_distinct_name_still_creates_new_person(self):
        from memory import people

        people.enroll_person("Exudica Royale")
        pid, created = people.find_or_create_person("Marcus Webb")
        self.assertTrue(created)
        self.assertNotEqual(pid, None)


class IntroVoiceCaptureWindowTest(unittest.TestCase):
    """Fix #3a: a fresh intro voice-capture window must survive the direct-turn
    identity-prompt deferral so the enrollment handler downstream can run."""

    def _fresh_ctx(self):
        return {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 5,
            "introduced_name": "Exudica Royale",
            "relationship": "friend",
            "asked_at": time.monotonic(),
        }

    def test_direct_turn_preserves_fresh_window(self):
        from intelligence import interaction as I

        with mock.patch.object(I, "_pending_intro_voice_capture", self._fresh_ctx()), \
             mock.patch.object(I.consciousness, "clear_pending_identity_prompts", return_value=False):
            I._clear_pending_identity_prompts("direct_turn")
            self.assertIsNotNone(I._pending_intro_voice_capture)

    def test_other_reason_still_clears_window(self):
        from intelligence import interaction as I

        with mock.patch.object(I, "_pending_intro_voice_capture", self._fresh_ctx()), \
             mock.patch.object(I.consciousness, "clear_pending_identity_prompts", return_value=False):
            I._clear_pending_identity_prompts("boundary")
            self.assertIsNone(I._pending_intro_voice_capture)

    def test_stale_window_is_cleared_even_on_direct_turn(self):
        from intelligence import interaction as I

        stale = self._fresh_ctx()
        stale["asked_at"] = time.monotonic() - 10_000.0
        with mock.patch.object(I, "_pending_intro_voice_capture", stale), \
             mock.patch.object(I.consciousness, "clear_pending_identity_prompts", return_value=False):
            I._clear_pending_identity_prompts("direct_turn")
            self.assertIsNone(I._pending_intro_voice_capture)


class IntroVoiceCaptureEnrollTest(unittest.TestCase):
    """Fix #3b: a mediocre (sub-confident) introducer score during the window,
    on a newcomer-sounding hello, enrolls the NEWCOMER — not the introducer."""

    def _ctx(self):
        return {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 5,
            "introduced_name": "Exudica Royale",
            "relationship": "friend",
            "asked_at": time.monotonic(),
        }

    def test_mediocre_introducer_score_enrolls_newcomer(self):
        from intelligence import interaction as I

        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(I, "_bind_intro_visible_face_if_present"), \
             mock.patch.object(I.llm, "get_response", return_value="Exudica! Welcome."), \
             mock.patch.object(I.consciousness, "mark_engagement"), \
             mock.patch.object(I.consciousness, "note_person_spoke"):
            resp = I._handle_intro_voice_capture(
                "hi what's your name",
                audio,
                person_id=1,        # mis-resolved to the introducer (Bret)
                raw_best_id=1,
                speaker_score=0.64,  # below the 0.75 confident bar
            )
        self.assertTrue(resp)
        self.assertTrue(enroll.called)
        self.assertEqual(enroll.call_args.args[0], 5)  # enrolled the NEWCOMER

    def test_confident_introducer_score_does_not_enroll(self):
        from intelligence import interaction as I

        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_intro_voice_capture(
                "hi what's your name",
                audio,
                person_id=1,
                raw_best_id=1,
                speaker_score=0.92,  # clearly the introducer re-speaking
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)


if __name__ == "__main__":
    unittest.main()
