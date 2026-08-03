"""Cross-day bit ledger (field 2026-07-31 → 08-02: the haircut observation ran
on BOTH Jul 31 and Aug 2, "I made you" was re-roasted twice the next afternoon,
and the hydration bit played on both ends of the weekend — session anti-repeat
can't see yesterday).

Uses the real field lines as fixtures. rex.db writes are suppressed under the
test runner on the default path, so REX_DB_PATH is pointed at a temp file.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from intelligence import bit_ledger
from memory import rex_db


class _TempRexDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = str(Path(self._tmp.name) / "rex.db")
        self._p = mock.patch.object(config, "REX_DB_PATH", path, create=True)
        self._p.start()
        rex_db.ensure_schema()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()


class BitLedgerRepeatTest(_TempRexDb):
    def test_haircut_bit_blocked_across_days(self):
        bit_ledger.record(
            1, "That brown short hair is doing a lot of work for a Friday night.",
            source="visual_riff",
        )
        self.assertTrue(bit_ledger.is_repeat(
            1, "Brown short hair—efficient, low-drama, and deeply suspicious "
               "of any attempt at style."
        ))

    def test_quoted_phrase_bit_blocked_despite_full_rewording(self):
        bit_ledger.record(
            1, "Mm-hmm. “I made you” is either a compliment or a cry for help.",
        )
        self.assertTrue(bit_ledger.is_repeat(
            1, "I'm still amused by “I made you” — that's either a "
               "masterpiece or a lawsuit waiting to happen."
        ))

    def test_distinctive_word_blocks_reworded_bit(self):
        bit_ledger.record(
            1, "Nice, social hydration and bad decisions in the same evening "
               "— a classic Friday victory.",
        )
        self.assertTrue(bit_ledger.is_repeat(
            1, "Nice — surviving hydration and company at the same time is a "
               "respectable evening."
        ))

    def test_unrelated_line_not_blocked(self):
        bit_ledger.record(
            1, "That brown short hair is doing a lot of work for a Friday night.",
        )
        self.assertFalse(bit_ledger.is_repeat(
            1, "Is that a model train, or does it just have a ridiculous backstory?"
        ))

    def test_other_person_not_blocked(self):
        bit_ledger.record(1, "That brown short hair is doing a lot of work.")
        self.assertFalse(bit_ledger.is_repeat(
            2, "Brown short hair — efficient and low-drama."
        ))

    def test_expired_bit_not_blocked(self):
        bit_ledger.record(1, "That brown short hair is doing a lot of work.")
        # Age the row past the cooldown.
        rex_db.execute(
            "UPDATE bit_ledger SET spoken_at = ?",
            (bit_ledger._cutoff_iso(bit_ledger._cooldown_days() + 1.0),),
        )
        self.assertFalse(bit_ledger.is_repeat(
            1, "Brown short hair — efficient and low-drama."
        ))

    def test_kill_switch(self):
        bit_ledger.record(1, "That brown short hair is doing a lot of work.")
        with mock.patch.object(config, "BIT_LEDGER_ENABLED", False, create=True):
            self.assertFalse(bit_ledger.is_repeat(
                1, "Brown short hair — efficient and low-drama."
            ))


class RecentTopicsTest(_TempRexDb):
    def test_topics_surface_for_prompt_exclusion(self):
        bit_ledger.record(1, "That brown short hair is doing a lot of work tonight.")
        bit_ledger.record(1, "“I made you” is either a compliment or a cry for help.")
        topics = bit_ledger.recent_topics(1)
        self.assertEqual(len(topics), 2)
        joined = " | ".join(topics)
        self.assertIn("'i made you'", joined)
        self.assertIn("hair", joined)

    def test_no_person_no_topics(self):
        self.assertEqual(bit_ledger.recent_topics(None), [])


class FailSafeTest(unittest.TestCase):
    def test_db_errors_read_as_not_a_repeat(self):
        with mock.patch.object(bit_ledger.rex_db, "fetchall",
                               side_effect=RuntimeError("boom")):
            self.assertFalse(bit_ledger.is_repeat(1, "any line at all here"))
        with mock.patch.object(bit_ledger.rex_db, "execute",
                               side_effect=RuntimeError("boom")):
            bit_ledger.record(1, "any line at all here")   # must not raise


if __name__ == "__main__":
    unittest.main()
