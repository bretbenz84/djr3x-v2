"""Dual-unknown introduction flow (owner spec 2026-07-06).

Two unknown faces on camera + an unrecognized voice → Rex asks positionally (left
first, then right); each answer binds name + the face at that position (Rex's left =
smaller camera x) + the answer's voice. One-known-one-unknown keeps the existing
single-unknown ask (not covered here).
"""

import unittest
from unittest import mock

import numpy as np

from intelligence import interaction as I


def _faces():
    # left face at x=100, right face at x=420 (Rex's left = smaller x)
    return [
        {"encoding": np.array([1.0, 0.0], dtype=np.float32), "x": 100},
        {"encoding": np.array([0.0, 1.0], dtype=np.float32), "x": 420},
    ]


def _audio():
    return (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)


class DualIntroReplyTest(unittest.TestCase):
    def setUp(self):
        I._pending_dual_intro = None

    def tearDown(self):
        I._pending_dual_intro = None

    def _arm(self, stage="left"):
        I._pending_dual_intro = {
            "faces": sorted(_faces(), key=lambda f: f["x"]),
            "stage": stage,
            "left_name": "" if stage == "left" else "Sarah",
            "asked_at": __import__("time").monotonic(),
        }

    def _reply(self, text, label=None):
        with (
            mock.patch.object(I.people_memory, "find_or_create_person",
                              side_effect=[(41, True), (42, True)]) as foc,
            mock.patch.object(I.people_memory, "add_biometric") as add_bio,
            mock.patch.object(I.people_memory, "update_familiarity"),
            mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll,
            mock.patch.object(I, "_retire_anonymous_speaker_slot") as retire,
            mock.patch.object(I, "_bind_world_state_identity"),
            mock.patch.object(I, "_episodic_person_enrolled"),
            mock.patch.object(I, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(I.conv_memory, "add_to_transcript"),
            mock.patch.object(I.conv_log, "log_rex"),
            mock.patch.object(I, "_register_rex_utterance"),
        ):
            consumed, spoken = I._handle_pending_dual_intro_reply(text, _audio(), label)
            return consumed, spoken, foc, add_bio, enroll, speak, retire

    def test_left_answer_binds_left_face_and_asks_right(self):
        self._arm("left")
        consumed, spoken, foc, add_bio, enroll, speak, _r = self._reply("I'm Sarah")
        self.assertTrue(consumed)
        foc.assert_called_once_with("Sarah")
        # the LEFT face (x=100) was bound
        bound = add_bio.call_args[0]
        self.assertEqual(bound[0], 41)
        self.assertEqual(bound[1], "face")
        self.assertTrue((bound[2] == np.array([1.0, 0.0], dtype=np.float32)).all())
        enroll.assert_called_once()
        # flow advanced to the right-side ask
        self.assertIsNotNone(I._pending_dual_intro)
        self.assertEqual(I._pending_dual_intro["stage"], "right")
        self.assertEqual(I._pending_dual_intro["left_name"], "Sarah")
        self.assertIn("right", spoken.lower())

    def test_right_answer_binds_right_face_and_finishes(self):
        self._arm("right")
        consumed, spoken, foc, add_bio, enroll, _s, retire = self._reply("Mike", label="unknown_voice_3")
        self.assertTrue(consumed)
        foc.assert_called_once_with("Mike")
        bound = add_bio.call_args[0]
        self.assertTrue((bound[2] == np.array([0.0, 1.0], dtype=np.float32)).all())
        retire.assert_called_once()          # answerer's anon slot retired to Mike
        self.assertIsNone(I._pending_dual_intro)   # flow complete
        self.assertIn("Mike", spoken)

    def test_nameless_reply_clears_without_consuming(self):
        self._arm("left")
        consumed, _sp, foc, _b, _e, _s, _r = self._reply("what do you want from me")
        self.assertFalse(consumed)
        foc.assert_not_called()
        self.assertIsNone(I._pending_dual_intro)   # dropped, no badgering

    def test_expired_window_clears(self):
        self._arm("left")
        I._pending_dual_intro["asked_at"] -= 9999
        consumed, _sp, foc, _b, _e, _s, _r = self._reply("I'm Sarah")
        self.assertFalse(consumed)
        foc.assert_not_called()
        self.assertIsNone(I._pending_dual_intro)

    def test_name_extraction_forms(self):
        self.assertEqual(I._dual_intro_name_from_reply("I'm Sarah"), "Sarah")
        self.assertEqual(I._dual_intro_name_from_reply("This is Mike"), "Mike")
        self.assertEqual(I._dual_intro_name_from_reply("Sarah"), "Sarah")
        self.assertIsNone(I._dual_intro_name_from_reply("what do you want from me"))


if __name__ == "__main__":
    unittest.main()
