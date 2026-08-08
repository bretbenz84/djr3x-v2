"""Cross-generator object Q&A ledger.

Field 2026-08-08 09:43/09:45: visual curiosity asked "what's in the bowl?",
got "frosted mini wheats", and asked the identical question again two minutes
later — the object-question generators shared no per-object memory and the
answer was never stored. intelligence/object_qa.py is the shared ledger; the
visual-curiosity objects line consumes it so answered objects become riff
material instead of repeat questions.
"""

import unittest
from unittest import mock

from intelligence import object_qa


class ObjectQaLedgerTest(unittest.TestCase):
    def setUp(self):
        object_qa.reset()
        self.addCleanup(object_qa.reset)
        # Keep DB write-through/fallback out of unit scope.
        self._db = mock.patch("memory.rex_db.execute", return_value=None)
        self._db2 = mock.patch("memory.rex_db.fetchone", return_value=None)
        self._rm = mock.patch("memory.room_model.note_question_asked")
        self._db.start(); self._db2.start(); self._rm.start()
        self.addCleanup(self._db.stop)
        self.addCleanup(self._db2.stop)
        self.addCleanup(self._rm.stop)

    def test_asked_and_answered_roundtrip(self):
        object_qa.note_asked("bowl", source="visual_curiosity",
                             question="Bret, what's in the bowl?")
        self.assertTrue(object_qa.was_asked("bowl"))
        self.assertIsNone(object_qa.known_answer("bowl"))
        captured = object_qa.maybe_capture_answer("Frosted mini wheats.")
        self.assertEqual(captured, "bowl")
        self.assertEqual(object_qa.known_answer("bowl"), "Frosted mini wheats")

    def test_freeform_question_label_detection(self):
        matched = object_qa.mark_asked_labels(
            "Bret, what's in the bowl?", ["bowl", "spoon", "chair"],
            source="visual_curiosity",
        )
        self.assertEqual(matched, ["bowl"])
        self.assertTrue(object_qa.was_asked("bowl"))
        self.assertFalse(object_qa.was_asked("spoon"))

    def test_non_answer_does_not_capture(self):
        object_qa.note_asked("spoon", source="room_question")
        self.assertIsNone(object_qa.maybe_capture_answer("I don't know."))
        self.assertIsNone(object_qa.known_answer("spoon"))
        # Latch survives one non-answer turn, expires after the second.
        self.assertIsNone(object_qa.maybe_capture_answer("What are you talking about?"))
        self.assertIsNone(object_qa.maybe_capture_answer("Frosted mini wheats."))
        self.assertIsNone(object_qa.known_answer("spoon"))

    def test_no_latch_no_capture(self):
        self.assertIsNone(object_qa.maybe_capture_answer("Frosted mini wheats."))


class VisualCuriosityLedgerIntegrationTest(unittest.TestCase):
    """The objects line must stop presenting answered objects as ask-fodder."""

    def setUp(self):
        object_qa.reset()
        self.addCleanup(object_qa.reset)
        self._db = mock.patch("memory.rex_db.execute", return_value=None)
        self._db2 = mock.patch("memory.rex_db.fetchone", return_value=None)
        self._rm = mock.patch("memory.room_model.note_question_asked")
        self._db.start(); self._db2.start(); self._rm.start()
        self.addCleanup(self._db.stop)
        self.addCleanup(self._db2.stop)
        self.addCleanup(self._rm.stop)

    def _line(self, objs):
        from intelligence import consciousness as c
        with mock.patch.object(c.world_state, "get", return_value=objs):
            return c._visual_curiosity_objects_line()

    def test_answered_object_becomes_do_not_reask_note(self):
        object_qa.note_asked("bowl", source="visual_curiosity")
        object_qa.maybe_capture_answer("Frosted mini wheats.")
        line = self._line([
            {"label": "bowl", "position": "foreground center", "confidence": 0.9},
            {"label": "guitar", "position": "center", "confidence": 0.8},
        ])
        self.assertIn("do NOT ask about", line)
        self.assertIn("Frosted mini wheats", line)
        self.assertIn("guitar", line)
        # The bowl must not be listed as a fresh ask target.
        self.assertNotIn("bowl (foreground center)", line)

    def test_asked_unanswered_object_is_dropped(self):
        object_qa.note_asked("bowl", source="visual_curiosity")
        line = self._line([
            {"label": "bowl", "position": "foreground center", "confidence": 0.9},
            {"label": "guitar", "position": "center", "confidence": 0.8},
        ])
        self.assertNotIn("bowl", line)
        self.assertIn("guitar", line)

    def test_all_objects_answered_still_returns_riff_context(self):
        object_qa.note_asked("bowl", source="visual_curiosity")
        object_qa.maybe_capture_answer("Frosted mini wheats.")
        line = self._line([
            {"label": "bowl", "position": "foreground center", "confidence": 0.9},
        ])
        self.assertIn("Frosted mini wheats", line)
        self.assertIn("do NOT ask about", line)


if __name__ == "__main__":
    unittest.main()
