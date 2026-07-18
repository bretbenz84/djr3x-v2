"""
tests/test_current_events.py — awareness/current_events.py storage, date gating,
parsing, and the pick/mark cycle. No network: the fetch is stubbed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from awareness import current_events


class CurrentEventsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = config.CURRENT_EVENTS_PATH
        config.CURRENT_EVENTS_PATH = str(Path(self._tmp.name) / "current_events.json")

    def tearDown(self):
        config.CURRENT_EVENTS_PATH = self._orig_path
        self._tmp.cleanup()

    @staticmethod
    def _stories(n=3):
        return [{"headline": f"Story {i}", "summary": f"Summary {i}.",
                 "topic": "test", "mentioned": False} for i in range(n)]

    def test_parse_tolerates_fences_and_prose(self):
        text = ('Here you go:\n```json\n'
                '[{"headline": "H1", "summary": "S1.", "topic": "space"},'
                ' {"headline": "H2", "summary": "S2.", "topic": ""}]\n```')
        parsed = current_events._parse_stories(text)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["headline"], "H1")
        self.assertFalse(parsed[0]["mentioned"])

    def test_parse_rejects_garbage(self):
        self.assertEqual(current_events._parse_stories("no json here"), [])
        self.assertEqual(current_events._parse_stories('[{"headline": ""}]'), [])

    def test_refresh_fetches_when_stale_and_skips_when_fresh(self):
        with mock.patch.object(current_events, "_fetch_via_web_search",
                               return_value=self._stories()) as fetch:
            self.assertTrue(current_events.refresh_if_stale())
            self.assertTrue(current_events.is_fresh())
            self.assertEqual(len(current_events.stories()), 3)
            # Second call same day: date gate — NO second fetch.
            self.assertFalse(current_events.refresh_if_stale())
            self.assertEqual(fetch.call_count, 1)

    def test_failed_fetch_keeps_previous_cache(self):
        current_events._save({"date": "2020-01-01", "fetched_at": "then",
                              "stories": self._stories(2)})
        with mock.patch.object(current_events, "_fetch_via_web_search",
                               side_effect=RuntimeError("offline")):
            self.assertFalse(current_events.refresh_if_stale())
        self.assertEqual(len(current_events.stories()), 2)   # stale beats none

    def test_pick_and_mark_cycle(self):
        from datetime import datetime
        current_events._save({"date": current_events._today(),
                              "fetched_at": datetime.now().isoformat(timespec="seconds"),
                              "stories": self._stories(2)})
        first = current_events.pick_story()
        self.assertEqual(first["headline"], "Story 0")
        current_events.mark_mentioned(first)
        second = current_events.pick_story()
        self.assertEqual(second["headline"], "Story 1")
        current_events.mark_mentioned(second)
        self.assertIsNone(current_events.pick_story())
        # Spent flags persisted to disk.
        on_disk = json.loads(Path(config.CURRENT_EVENTS_PATH).read_text())
        self.assertTrue(all(s["mentioned"] for s in on_disk["stories"]))

    def test_stale_cache_yields_no_story(self):
        # Field 2026-07-18: "did you hear" about a day-old cache — pick_story
        # now refuses anything older than CURRENT_EVENTS_MAX_AGE_HOURS.
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
        current_events._save({"date": "2026-01-01", "fetched_at": old,
                              "stories": self._stories(2)})
        self.assertIsNone(current_events.pick_story())

    def test_disabled_never_fetches(self):
        with mock.patch.object(config, "CURRENT_EVENTS_ENABLED", False), \
             mock.patch.object(current_events, "_fetch_via_web_search") as fetch:
            self.assertFalse(current_events.refresh_if_stale())
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
