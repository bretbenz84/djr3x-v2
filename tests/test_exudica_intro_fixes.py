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

    def test_group_session_mishearings_corrected(self):
        # Field 2026-07-23 (logs/djr3x-2026-07-23-19-50-57): "Zutica", "Brat"
        # (spawned a whole phantom person), and "Impersivate" missed the router.
        from audio.transcription import _apply_corrections

        self.assertEqual(_apply_corrections("we've met, I'm in Zutica"),
                         "we've met, I'm in Exudica")
        self.assertEqual(_apply_corrections("hey Brat come here"),
                         "hey Bret come here")
        self.assertEqual(_apply_corrections("Impersivate me"), "impersonate me")

    def test_corrections_are_word_bounded(self):
        # A bare substring replace corrupted embedded matches ("vibrate" contains
        # "brat", "breadth" contains "bread").
        from audio.transcription import _apply_corrections

        self.assertEqual(_apply_corrections("set it to vibrate mode"),
                         "set it to vibrate mode")
        self.assertEqual(_apply_corrections("the breadth of the room"),
                         "the breadth of the room")


class HairStylistDisplayNameTest(unittest.TestCase):
    """The special-person prompt hook must address her by the name she goes by
    NOW — she was introduced as Exudica and Rex greeted her as 'Joy' (field
    2026-07-23, an awful first impression)."""

    def test_prompt_context_uses_current_alias(self):
        from intelligence import person_specials as PS
        ctx = PS.galactic_hair_stylist_prompt_context("Exudica")
        self.assertIn("ADDRESS HER AS Exudica", ctx)
        ctx_joy = PS.galactic_hair_stylist_prompt_context("Joy")
        self.assertIn("ADDRESS HER AS Joy", ctx_joy)

    def test_intro_ack_uses_current_alias(self):
        from intelligence import person_specials as PS
        ack = PS.galactic_hair_stylist_intro_ack("Exudica")
        self.assertTrue(ack.startswith("Exudica"))


class OwnEchoRejectionTest(unittest.TestCase):
    """Reference-text rejection: Rex's AEC residual crossing the VAD must not
    become a phantom speaker (field 2026-07-23 19:56: the 'Something's in my way'
    announce came back as unknown_voice_2 and got a full LLM reply)."""

    def setUp(self):
        from intelligence import interaction as I
        self.I = I
        with I._recent_rex_lines_lock:
            I._recent_rex_lines.clear()

    def tearDown(self):
        with self.I._recent_rex_lines_lock:
            self.I._recent_rex_lines.clear()

    def test_verbatim_echo_rejected(self):
        self.I._note_rex_spoke("Something's in my way — that's as far as I get.")
        self.assertTrue(self.I._looks_like_own_echo("Something's in my way"))
        self.assertTrue(self.I._looks_like_own_echo("that's as far as I get"))

    def test_emotion_tags_stripped_before_matching(self):
        self.I._note_rex_spoke("[amused] Consider it logged. Onward and upward.")
        self.assertTrue(self.I._looks_like_own_echo("consider it logged onward and upward"))

    def test_short_overlaps_stay_attributable_to_the_human(self):
        self.I._note_rex_spoke("Yeah, okay. Onward.")
        self.assertFalse(self.I._looks_like_own_echo("yeah okay"))

    def test_normal_speech_not_rejected(self):
        self.I._note_rex_spoke("Something's in my way — that's as far as I get.")
        self.assertFalse(self.I._looks_like_own_echo("what do you see in my hand"))
        self.assertFalse(self.I._looks_like_own_echo("turn right and move forward five feet"))

    def test_stale_lines_expire(self):
        norm = self.I._normalize_echo_text("Something's in my way — that's as far as I get.")
        self.I._recent_rex_lines.append((norm, time.monotonic() - 60.0))
        self.assertFalse(self.I._looks_like_own_echo("Something's in my way"))

    def test_kill_switch(self):
        self.I._note_rex_spoke("Something's in my way — that's as far as I get.")
        with mock.patch.object(config, "OWN_ECHO_REJECT_ENABLED", False, create=True):
            self.assertFalse(self.I._looks_like_own_echo("Something's in my way"))


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

    def test_confident_identity_introducer_does_not_enroll_in_band(self):
        # BUG-2: the [SPEAKER_ID_CONFIDENT_THRESHOLD=0.70, 0.75) band — identity
        # said "confidently the introducer" yet the 0.75 intro bar called it
        # "weak" and enrolled Bret's correction onto phantom "Leaf".
        from intelligence import interaction as I

        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_intro_voice_capture(
                "I was answering your question",
                audio,
                person_id=1,
                raw_best_id=1,
                speaker_score=0.707,  # >= 0.70 confident, < 0.75 old intro bar
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)


class IntroAnswerGateTest(unittest.TestCase):
    """BUG-2: a name-shaped answer to Rex's own question must not be promoted
    into an introduction; minting a brand-new person from one utterance by a
    confident known speaker with nobody new present is blocked."""

    def tearDown(self):
        from intelligence import dialogue_act
        dialogue_act.clear()

    def test_intro_is_answer_via_frame_even_after_wait_expired(self):
        from intelligence import interaction as I
        from intelligence import dialogue_act

        dialogue_act.clear()
        dialogue_act.note_rex_turn(
            "What do you need most: sleep, solitude, or the excuse to disappear?",
            source="idle_banter",
            target_person_id=1,
            expected_reply_types=["answer", "statement"],
        )
        with mock.patch.object(
            I.consciousness, "is_waiting_for_response", return_value=False
        ):
            # No visible newcomer; the short 7s wait has expired — the durable
            # frame still marks this turn as an answer.
            self.assertTrue(
                I._intro_is_answer_to_rex_question(False, person_id=1)
            )

    def test_mint_guard_blocks_name_only_from_confident_speaker(self):
        from types import SimpleNamespace
        from intelligence import interaction as I

        parsed = SimpleNamespace(name="Leaf", relationship=None, subject_kind="person")
        with mock.patch.object(I.people_memory, "find_person_by_name", return_value=None):
            self.assertTrue(
                I._intro_would_mint_unknown_name(
                    parsed, person_id=1, off_camera_unknown=False,
                    has_unknown_for_intro=False,
                )
            )

    def test_mint_guard_allows_relationship_intro(self):
        from types import SimpleNamespace
        from intelligence import interaction as I

        parsed = SimpleNamespace(name="Wade", relationship="brother", subject_kind="person")
        with mock.patch.object(I.people_memory, "find_person_by_name", return_value=None):
            self.assertFalse(
                I._intro_would_mint_unknown_name(
                    parsed, person_id=1, off_camera_unknown=False,
                    has_unknown_for_intro=False,
                )
            )

    def test_mint_guard_allows_genuine_newcomer(self):
        from types import SimpleNamespace
        from intelligence import interaction as I

        parsed = SimpleNamespace(name="Leaf", relationship=None, subject_kind="person")
        with mock.patch.object(I.people_memory, "find_person_by_name", return_value=None):
            self.assertFalse(
                I._intro_would_mint_unknown_name(
                    parsed, person_id=1, off_camera_unknown=False,
                    has_unknown_for_intro=True,  # a real newcomer is present
                )
            )

    def test_mint_guard_allows_known_name_link(self):
        from types import SimpleNamespace
        from intelligence import interaction as I

        parsed = SimpleNamespace(name="Sarah", relationship=None, subject_kind="person")
        with mock.patch.object(
            I.people_memory, "find_person_by_name", return_value={"id": 9, "name": "Sarah"}
        ):
            self.assertFalse(
                I._intro_would_mint_unknown_name(
                    parsed, person_id=1, off_camera_unknown=False,
                    has_unknown_for_intro=False,
                )
            )


class IntroCaptureWindowGateTest(unittest.TestCase):
    """Fix #4: the gate that suppresses sticky/visible-face attribution while Rex
    is waiting for a just-introduced newcomer to speak."""

    def test_open_only_for_a_fresh_voice_capture(self):
        from intelligence import interaction as I

        fresh = {"introduced_id": 5, "asked_at": time.monotonic()}
        with mock.patch.object(I, "_pending_intro_voice_capture", fresh):
            self.assertTrue(I._intro_capture_window_open())

    def test_closed_when_no_window(self):
        from intelligence import interaction as I

        with mock.patch.object(I, "_pending_intro_voice_capture", None):
            self.assertFalse(I._intro_capture_window_open())

    def test_closed_when_window_is_stale(self):
        from intelligence import interaction as I

        stale = {"introduced_id": 5, "asked_at": time.monotonic() - 10_000.0}
        with mock.patch.object(I, "_pending_intro_voice_capture", stale):
            self.assertFalse(I._intro_capture_window_open())


if __name__ == "__main__":
    unittest.main()
