"""
The voiceless-face rule + voice-sample request flow, from the 2026-08-23 21:07
session (logs/djr3x-2026-08-23-21-07-31.log).

PJ's face was enrolled (3 references) but his voice never was. His speech,
having no row of its own to match, landed on Bret's centroid at 0.79–0.94 —
CONFIDENT territory — so "voice over visible face" credited off-camera Bret
turn after turn while PJ's recognized face was on camera ("I know, Bret — …"
spoken straight at PJ). And because nothing ever enrolled his voice, the
failure was permanent.

Under test:
  - speaker_id.comparable_print_count: the voiceless-face signature (0 clips
    under the ACTIVE embedder; other-embedder rows don't count).
  - _voice_primary_face_decision "voiceless_face_wins": a cross-match — even a
    confident one — does not override the sole visible known face when that
    face's person has no voice print, unless the matched person was themselves
    on camera moments ago or the visual latch contradicts the face.
  - _maybe_request_voice_sample / _handle_voice_sample_capture: Rex asks the
    person for a line and enrolls the next qualifying utterance onto their row.
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
from audio import speaker_id
from intelligence import interaction as I
from memory import database as db


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


class ComparablePrintCountTest(_TempPeopleDb):
    def setUp(self):
        super().setUp()
        from memory import people

        self.people = people
        self.pj = people.enroll_person("PJ Thomas")
        self.bret = people.enroll_person("Bret Benziger")
        people.add_biometric(self.bret, "voice", np.zeros(192, dtype=np.float32))

    def test_face_only_person_is_voiceless(self):
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.pj), 0)

    def test_enrolled_person_counts_matching_dim(self):
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.bret), 1)

    def test_other_embedder_rows_do_not_count(self):
        # A stale 256-dim Resemblyzer row can never match a live ECAPA query.
        self.people.add_biometric(self.pj, "voice", np.zeros(256, dtype=np.float32))
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.pj), 0)


class VoicelessFaceWinsDecisionTest(unittest.TestCase):
    """ws_pid=7 (PJ, visible, print-less); voice candidate = Bret (id 1)."""

    def _decide(self, **kw):
        base = dict(
            person_id=1,
            raw_best_id=1,
            speaker_score=0.851,          # PJ's field score on Bret's centroid
            ws_pid=7,
            single_visible=True,
            engaged_is_visible=False,
            unknown_visible=False,
            other_known_recently=False,
            ws_voiceless=True,
            raw_best_recently_visible=False,
        )
        base.update(kw)
        return I._voice_primary_face_decision(**base)

    def test_confident_cross_match_loses_to_voiceless_face(self):
        self.assertEqual(self._decide(), "voiceless_face_wins")

    def test_even_the_slam_dunk_score_loses(self):
        # "Come here." hit 0.938 — as high as genuine Bret ever scores.
        self.assertEqual(self._decide(speaker_score=0.938), "voiceless_face_wins")

    def test_sub_confident_cross_match_also_resolves_voiceless(self):
        self.assertEqual(self._decide(speaker_score=0.62), "voiceless_face_wins")

    def test_matched_person_just_left_frame_keeps_voice(self):
        # Bret stepped out of frame seconds ago — a real off-camera speaker.
        self.assertEqual(
            self._decide(raw_best_recently_visible=True), "voice_over_face"
        )

    def test_face_with_a_print_keeps_the_old_rule(self):
        self.assertEqual(self._decide(ws_voiceless=False), "voice_over_face")

    def test_visual_latch_on_someone_else_keeps_voice(self):
        self.assertEqual(
            self._decide(visual_speaker_pid=1), "voice_over_face"
        )

    def test_visual_latch_on_the_face_still_wins(self):
        self.assertEqual(
            self._decide(visual_speaker_pid=7), "voiceless_face_wins"
        )

    def test_multi_face_scene_does_not_use_the_rule(self):
        self.assertNotEqual(
            self._decide(single_visible=False), "voiceless_face_wins"
        )

    def test_disabled_flag_restores_old_behavior(self):
        with mock.patch.object(config, "VOICELESS_FACE_WINS_ENABLED", False, create=True):
            self.assertEqual(self._decide(), "voice_over_face")


class VoiceSampleRequestTest(unittest.TestCase):
    def setUp(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()

    def tearDown(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()

    def test_arms_once_per_person_per_session(self):
        I._maybe_request_voice_sample(7, "PJ Thomas")
        self.assertIsNotNone(I._pending_voice_sample_capture)
        self.assertIsNone(I._pending_voice_sample_capture["asked_at"])
        I._pending_voice_sample_capture = None
        I._maybe_request_voice_sample(7, "PJ Thomas")   # second time: no re-arm
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_does_not_arm_during_intro_capture(self):
        with mock.patch.object(I, "_pending_intro_voice_capture", {"introduced_id": 5}):
            I._maybe_request_voice_sample(7, "PJ Thomas")
        self.assertIsNone(I._pending_voice_sample_capture)


class VoiceSampleCaptureTest(unittest.TestCase):
    def setUp(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()
        # VOICED audio (the min-length guard measures speech frames, not buffer).
        t = np.arange(40000, dtype=np.float32) / 16000.0
        self._audio = (0.1 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    def tearDown(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()

    def _asked_ctx(self):
        return {
            "person_id": 7,
            "name": "PJ Thomas",
            "armed_at": time.monotonic(),
            "asked_at": time.monotonic(),
        }

    def test_not_consumed_before_the_ask_is_spoken(self):
        I._pending_voice_sample_capture = {
            "person_id": 7, "name": "PJ Thomas",
            "armed_at": time.monotonic(), "asked_at": None,
        }
        resp = I._handle_voice_sample_capture("hello there", self._audio, 1, 1, 0.85)
        self.assertIsNone(resp)
        self.assertIsNotNone(I._pending_voice_sample_capture)

    def test_cross_match_to_off_camera_print_still_enrolls_target(self):
        # The exact field shape: PJ replies, scores 0.85 as (not-visible) Bret.
        I._pending_voice_sample_capture = self._asked_ctx()
        visible = {7: True, 1: False}
        with mock.patch.object(I, "_known_person_visible_recently", side_effect=lambda p: visible.get(I._safe_int(p), False)), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(I.consciousness, "mark_engagement"), \
             mock.patch.object(I.consciousness, "note_person_spoke"), \
             mock.patch.object(I.topic_thread, "note_user_turn"), \
             mock.patch.object(I.user_energy, "note_user_turn"):
            resp = I._handle_voice_sample_capture(
                "The Kings are winning it all this year.", self._audio, 1, 1, 0.851
            )
        self.assertTrue(resp)
        self.assertTrue(enroll.called)
        self.assertEqual(enroll.call_args.args[0], 7)
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_confident_match_on_visible_other_person_skips(self):
        # Bret is ALSO on camera and the reply confidently matches him — that
        # is Bret talking, not the target; never enroll it onto PJ.
        I._pending_voice_sample_capture = self._asked_ctx()
        with mock.patch.object(I, "_known_person_visible_recently", return_value=True), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_voice_sample_capture(
                "go ahead, say something", self._audio, 1, 1, 0.90
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)
        self.assertIsNotNone(I._pending_voice_sample_capture)

    def test_target_off_camera_skips(self):
        I._pending_voice_sample_capture = self._asked_ctx()
        with mock.patch.object(I, "_known_person_visible_recently", return_value=False), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_voice_sample_capture("hello", self._audio, 1, 1, 0.85)
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)

    def test_refusal_drops_the_request(self):
        I._pending_voice_sample_capture = self._asked_ctx()
        resp = I._handle_voice_sample_capture("not right now, Rex", self._audio, None, None, 0.0)
        self.assertIsNone(resp)
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_expired_window_clears(self):
        ctx = self._asked_ctx()
        ctx["asked_at"] = time.monotonic() - 10_000.0
        I._pending_voice_sample_capture = ctx
        resp = I._handle_voice_sample_capture("hello", self._audio, None, None, 0.0)
        self.assertIsNone(resp)
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_too_short_a_sample_reasks_instead_of_enrolling(self):
        # Field 2026-08-25: PJ enrolled from a ~1s "Hey Rex." and spent the whole
        # Jeopardy game being read as Bret. A blink of a sample re-asks for a
        # full sentence; the window stays open for the retry.
        I._pending_voice_sample_capture = self._asked_ctx()
        with mock.patch.object(I, "_known_person_visible_recently", return_value=True), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_voice_sample_capture(
                "Hey Rex.", np.zeros(16000, dtype=np.float32), None, None, 0.0
            )
        self.assertIsNotNone(resp)
        self.assertIn("sentence", resp)
        self.assertFalse(enroll.called)
        self.assertIsNotNone(I._pending_voice_sample_capture)

    def test_short_transcript_reasks_even_with_long_audio(self):
        # Duration alone can lie (leading room tone) — a two-word transcript is
        # not enough signal either way.
        I._pending_voice_sample_capture = self._asked_ctx()
        with mock.patch.object(I, "_known_person_visible_recently", return_value=True), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_voice_sample_capture(
                "Hey Rex.", np.zeros(64000, dtype=np.float32), None, None, 0.0
            )
        self.assertIsNotNone(resp)
        self.assertFalse(enroll.called)

    def test_short_sample_pushback_dictates_a_line(self):
        # "Give me a line" froze PJ into "Hey Rex" — the pushback must tell the
        # person exactly what to say (owner call 2026-08-26).
        ctx = self._asked_ctx()
        ctx["expected_text"] = "The quick brown fox jumps over the lazy dog."
        I._pending_voice_sample_capture = ctx
        with mock.patch.object(I, "_known_person_visible_recently", return_value=True), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True):
            resp = I._handle_voice_sample_capture(
                "Hey Rex.", np.zeros(16000, dtype=np.float32), None, None, 0.0
            )
        self.assertIn("Repeat after me", resp)
        self.assertIn("quick brown fox", resp)

    def test_pushback_without_a_stored_line_picks_one(self):
        I._pending_voice_sample_capture = self._asked_ctx()   # no expected_text
        with mock.patch.object(I, "_known_person_visible_recently", return_value=True), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True):
            resp = I._handle_voice_sample_capture(
                "Hey Rex.", np.zeros(16000, dtype=np.float32), None, None, 0.0
            )
        self.assertIn("Repeat after me", resp)
        self.assertEqual(
            I._pending_voice_sample_capture.get("expected_text"),
            resp.split("Repeat after me: ", 1)[1],
            "the dictated line is stored so the next pushback repeats the SAME line",
        )

    def test_dictated_lines_never_trip_the_decline_detector(self):
        # The decline regex drops the request on "no|not now|wait|..." — a
        # dictated line echoed back must never read as a refusal.
        import re as _re
        decline = _re.compile(
            r"\b(no|nope|not now|not right now|later|wait|hold on|can'?t|cannot)\b"
        )
        for line in getattr(config, "VOICE_SAMPLE_LINES", []):
            self.assertIsNone(decline.search(line.lower()), line)


if __name__ == "__main__":
    unittest.main()
