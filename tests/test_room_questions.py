"""
tests/test_room_questions.py — the learn-by-asking loop (curiosity Phase 1):
room-model question queue + rarity gating, answer extraction, corroboration
counting, latch expiry. Runs against a temp rex.db — no hardware, no LLM.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from memory import rex_db


def _fresh_db(tmpdir):
    # db_path() reads config.REX_DB_PATH at call time — patch the attr directly.
    config.REX_DB_PATH = str(Path(tmpdir) / "rex.db")
    rex_db.ensure_schema()


class RoomModelQueueTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = getattr(config, "REX_DB_PATH", None)
        _fresh_db(self._tmp.name)
        from memory import room_model
        self.rm = room_model

    def tearDown(self):
        config.REX_DB_PATH = self._orig
        config.ROOM_QUESTION_MIN_ROOM_AGE_DAYS = 1.0
        self._tmp.cleanup()

    def _establish_baseline(self, n=5, sightings=25):
        # First meet every fixture with the age gate CLOSED (day-one behavior),
        # then open it — mirrors a room whose furniture predates novelty tracking.
        config.ROOM_QUESTION_MIN_ROOM_AGE_DAYS = 9999
        for i in range(n):
            for _ in range(sightings):
                self.rm.record_objects([{"label": f"fixture{i}", "position": "background"}])
        config.ROOM_QUESTION_MIN_ROOM_AGE_DAYS = 0

    def test_no_pending_before_baseline(self):
        # Fresh install: first-ever objects must NOT queue questions.
        self.rm.record_objects([{"label": "couch", "position": "left"}])
        self.assertIsNone(self.rm.pending_question(min_sightings=1))

    def test_new_object_queues_after_baseline(self):
        self._establish_baseline()
        self.rm.record_objects([{"label": "guitar", "position": "foreground left"}])
        self.rm.record_objects([{"label": "guitar", "position": "foreground left"}])
        row = self.rm.pending_question(min_sightings=2)
        self.assertIsNotNone(row)
        self.assertEqual(row["label"], "guitar")

    def test_unconfirmed_object_not_offered(self):
        self._establish_baseline()
        self.rm.record_objects([{"label": "kite", "position": "center"}])   # one frame
        self.assertIsNone(self.rm.pending_question(min_sightings=2))

    def test_answer_and_corroboration(self):
        self._establish_baseline()
        for _ in range(2):
            self.rm.record_objects([{"label": "vase", "position": "counter"}])
        self.assertTrue(self.rm.record_answer("vase", "the sourdough starter"))
        got = self.rm.human_label("vase")
        self.assertEqual(got["name"], "the sourdough starter")
        self.assertEqual(got["confidence"], 1)
        # Matching repeat corroborates...
        self.assertTrue(self.rm.record_answer("vase", "The Sourdough Starter"))
        self.assertEqual(self.rm.human_label("vase")["confidence"], 2)
        # ...and a twice-confirmed name resists a contradicting joker.
        self.assertFalse(self.rm.record_answer("vase", "a cursed artifact"))
        self.assertEqual(self.rm.human_label("vase")["name"], "the sourdough starter")

    def test_single_source_name_can_be_replaced(self):
        self._establish_baseline()
        self.rm.record_objects([{"label": "bowl", "position": "table"}])
        self.rm.record_answer("bowl", "dog water")
        self.assertTrue(self.rm.record_answer("bowl", "the popcorn bowl"))
        self.assertEqual(self.rm.human_label("bowl")["name"], "the popcorn bowl")


class AnswerExtractionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import room_questions
        self.rq = room_questions

    def test_identity_patterns(self):
        f = self.rq._extract_identity
        self.assertEqual(f("Oh, that's my sourdough starter"), "sourdough starter")
        self.assertEqual(f("it's called Bessie"), "Bessie")
        self.assertEqual(f("That is the telescope case, buddy"), "telescope case")
        self.assertEqual(f("we call it the doom shelf"), "the doom shelf")

    def test_short_direct_reply(self):
        self.assertEqual(self.rq._extract_identity("Sourdough starter."), "Sourdough starter")

    def test_non_answers_rejected(self):
        f = self.rq._extract_identity
        self.assertIsNone(f("I don't know honestly"))
        self.assertIsNone(f("What do you think it is?"))
        self.assertIsNone(f("Why do you want to know about that thing over there anyway my friend"))


class LatchTest(unittest.TestCase):
    def setUp(self):
        from intelligence import room_questions
        self.rq = room_questions
        self.rq.reset()

    def tearDown(self):
        self.rq.reset()

    def test_capture_writes_back(self):
        with mock.patch("memory.room_model.note_question_asked"), \
             mock.patch("memory.room_model.record_answer", return_value=True) as rec:
            self.rq.note_asked("guitar")
            self.assertTrue(self.rq.maybe_capture_answer("that's my dad's old guitar"))
            rec.assert_called_once()
            self.assertEqual(rec.call_args[0][0], "guitar")

    def test_latch_expires_after_turns(self):
        with mock.patch("memory.room_model.note_question_asked"), \
             mock.patch("memory.room_model.dismiss_question") as dis:
            self.rq.note_asked("guitar")
            self.assertFalse(self.rq.maybe_capture_answer("anyway how was your day going"))
            self.assertFalse(self.rq.maybe_capture_answer("did you sleep well?"))
            dis.assert_called_once_with("guitar")
        # Latch gone: further turns are ignored.
        self.assertFalse(self.rq.maybe_capture_answer("it's my guitar"))

    def test_cooldown_blocks_next_question(self):
        with mock.patch("memory.room_model.note_question_asked"):
            self.rq.note_asked("guitar")
        self.assertIsNone(self.rq.next_room_question())   # cooldown just armed


if __name__ == "__main__":
    unittest.main()


class CorrectionLatchTest(unittest.TestCase):
    """The remark-correction path (field 2026-07-18: 'Actually, that's a pillow')."""

    def setUp(self):
        from intelligence import room_questions
        self.rq = room_questions
        self.rq.reset()

    def tearDown(self):
        self.rq.reset()

    def test_remark_correction_captured(self):
        with mock.patch("memory.room_model.record_answer", return_value=True):
            self.rq.note_room_remark("handbag")
            self.assertTrue(self.rq.maybe_capture_answer("Actually, that's a pillow"))
        self.assertTrue(self.rq.recently_captured())
        cap = self.rq.last_capture()
        self.assertEqual(cap["label"], "handbag")
        self.assertEqual(cap["name"], "pillow")
        self.assertEqual(cap["kind"], "remark")

    def test_no_latch_no_capture(self):
        self.assertFalse(self.rq.maybe_capture_answer("Actually, that's a pillow"))
        self.assertFalse(self.rq.recently_captured())
