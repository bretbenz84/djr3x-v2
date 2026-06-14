import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock


class GreetingsTodayCounterTests(unittest.TestCase):
    """memory.people tracks per-local-day greetings so Rex can do same-day
    'oh, it's you again' repeat-visit banter."""

    def _create_db(self, path: Path) -> None:
        from setup_assets import DB_SCHEMA

        with sqlite3.connect(path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")

    def test_count_starts_at_zero_and_increments_same_day(self):
        from memory import people

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "people.db"
            self._create_db(db_path)
            with mock.patch.object(people.db, "_DB_FILE", db_path):
                # First time today: no prior greetings.
                self.assertEqual(people.greetings_today_count(1), 0)
                people.record_greeting(1)
                # Now one greeting today → a second activation is a repeat.
                self.assertEqual(people.greetings_today_count(1), 1)
                people.record_greeting(1)
                self.assertEqual(people.greetings_today_count(1), 2)

    def test_count_resets_on_a_new_day(self):
        from memory import people

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "people.db"
            self._create_db(db_path)
            with mock.patch.object(people.db, "_DB_FILE", db_path):
                people.record_greeting(1)
                people.record_greeting(1)
                self.assertEqual(people.greetings_today_count(1), 2)

                # Simulate yesterday's tally lingering in the row.
                yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
                people.db.execute(
                    "UPDATE people SET greetings_today_date = ? WHERE id = 1",
                    (yesterday,),
                )
                # Stale day → reads as 0 (fresh start today).
                self.assertEqual(people.greetings_today_count(1), 0)
                # And the next greeting restarts the tally at 1, not 3.
                people.record_greeting(1)
                self.assertEqual(people.greetings_today_count(1), 1)

    def test_lifetime_count_still_accumulates_across_days(self):
        from memory import people

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "people.db"
            self._create_db(db_path)
            with mock.patch.object(people.db, "_DB_FILE", db_path):
                people.record_greeting(1)
                yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
                people.db.execute(
                    "UPDATE people SET greetings_today_date = ? WHERE id = 1",
                    (yesterday,),
                )
                people.record_greeting(1)  # new day → greetings_today resets to 1
                row = people.db.fetchone(
                    "SELECT lifetime_greeting_count, greetings_today FROM people WHERE id = 1"
                )
                self.assertEqual(row["lifetime_greeting_count"], 2)
                self.assertEqual(row["greetings_today"], 1)

    def test_bad_person_id_is_safe(self):
        from memory import people

        self.assertEqual(people.greetings_today_count(None), 0)
        people.record_greeting("not-an-int")  # must not raise


class SameDayReturnPromptTests(unittest.TestCase):
    def test_count_zero_means_not_a_repeat(self):
        from intelligence import consciousness

        with mock.patch("memory.people.greetings_today_count", return_value=0):
            self.assertEqual(consciousness._same_day_return_count(1), 0)

    def test_disabled_flag_suppresses_repeat(self):
        from intelligence import consciousness

        with (
            mock.patch("memory.people.greetings_today_count", return_value=3),
            mock.patch.object(consciousness.config, "PRESENCE_SAME_DAY_RETURN_ENABLED", False),
        ):
            self.assertEqual(consciousness._same_day_return_count(1), 0)

    def test_prompt_is_warm_and_uses_name(self):
        """P1: a same-day return is a warm hello, no longer a roast."""
        from intelligence import consciousness

        prompt = consciousness._build_same_day_return_prompt("Bret", 1)
        self.assertIn("Bret", prompt)
        self.assertIn("how are you", prompt.lower())
        # Drops into conversation, ends as a question.
        self.assertIn("question", prompt.lower())
        # The old roast escalation ("punch up the bit", ordinal tally) is gone.
        self.assertNotIn("punch up", prompt.lower())
        self.assertNotIn("3rd time today", prompt)

    def test_prompt_no_longer_escalates_with_an_ordinal_tally(self):
        """P1: it softens ('glad they keep coming back'), never tallies '3rd time today'."""
        from intelligence import consciousness

        third = consciousness._build_same_day_return_prompt("Bret", 2)
        self.assertNotIn("3rd time today", third)
        self.assertIn("how are you", third.lower())

    def test_ordinal_helper(self):
        from intelligence import consciousness

        self.assertEqual(consciousness._ordinal(1), "1st")
        self.assertEqual(consciousness._ordinal(2), "2nd")
        self.assertEqual(consciousness._ordinal(3), "3rd")
        self.assertEqual(consciousness._ordinal(4), "4th")
        self.assertEqual(consciousness._ordinal(11), "11th")
        self.assertEqual(consciousness._ordinal(21), "21st")


if __name__ == "__main__":
    unittest.main()
