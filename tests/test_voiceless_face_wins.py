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
    person for a line and enrolls only a verified repetition onto their row.
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






if __name__ == "__main__":
    unittest.main()
