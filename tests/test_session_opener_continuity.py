"""
Session-opener continuity (owner spec 2026-07-06): "last night you never told me how
the soup turned out."

Undated planned/promised events used to wait FOLLOWUP_UNDATED_DAYS (7) before any
follow-up, so the next-morning greeting never referenced them. get_recent_open_threads
surfaces them the very NEXT session — mentioned before this process booted, within a
short lookback, still open. mentioned_when_label turns the timestamp into the phrase
the greeting speaks ("last night", "yesterday", "earlier today").
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import config
from memory import database as db
from memory import events


def _build_people_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")


def _insert_event(person_id=1, name="making soup", *, event_date=None,
                  mentioned_hours_ago=14.0, followed_up=False, status="planned"):
    mentioned = (datetime.now(timezone.utc)
                 - timedelta(hours=mentioned_hours_ago)).isoformat()
    return db.execute(
        """INSERT INTO person_events
             (person_id, event_name, event_date, event_notes, mentioned_at,
              followed_up, status, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (person_id, name, event_date, "", mentioned, followed_up, status, mentioned),
    )


class RecentOpenThreadsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "people.db"
        _build_people_db(path)
        self._db_patch = mock.patch.object(db, "_DB_FILE", path)
        self._db_patch.start()
        # Boot "now": events mentioned in the past belong to a previous session.
        self._boot_patch = mock.patch.object(
            events, "_BOOT_AT_ISO", datetime.now(timezone.utc).isoformat())
        self._boot_patch.start()

    def tearDown(self):
        self._boot_patch.stop()
        self._db_patch.stop()
        self._tmp.cleanup()

    def test_last_nights_undated_thread_is_returned(self):
        _insert_event(name="making soup", mentioned_hours_ago=14)
        threads = events.get_recent_open_threads(1)
        self.assertEqual([t["event_name"] for t in threads], ["making soup"])

    def test_promised_status_also_counts(self):
        _insert_event(name="adding wheels to your body", status="promised",
                      mentioned_hours_ago=30)
        self.assertEqual(len(events.get_recent_open_threads(1)), 1)

    def test_dated_events_are_excluded(self):
        # Dated-past is Priority 2.5's job; dated-future is anticipation.
        _insert_event(name="concert", event_date="2026-07-01", mentioned_hours_ago=20)
        _insert_event(name="camping", event_date="2099-01-01", mentioned_hours_ago=20)
        self.assertEqual(events.get_recent_open_threads(1), [])

    def test_followed_up_and_closed_are_excluded(self):
        _insert_event(name="soup", followed_up=True, mentioned_hours_ago=14)
        _insert_event(name="movie", status="completed", mentioned_hours_ago=14)
        _insert_event(name="party", status="canceled", mentioned_hours_ago=14)
        self.assertEqual(events.get_recent_open_threads(1), [])

    def test_current_session_mentions_are_excluded(self):
        # Mentioned AFTER boot = live conversation, not a thread to greet with.
        with mock.patch.object(
            events, "_BOOT_AT_ISO",
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        ):
            _insert_event(name="soup", mentioned_hours_ago=1)  # 1h ago, boot 2h ago
            self.assertEqual(events.get_recent_open_threads(1), [])

    def test_older_than_lookback_falls_through(self):
        # 5 days old with a 3-day lookback: left to the normal 7-day pending path.
        _insert_event(name="soup", mentioned_hours_ago=5 * 24)
        self.assertEqual(events.get_recent_open_threads(1), [])
        self.assertEqual(len(events.get_recent_open_threads(1, lookback_days=6)), 1)

    def test_newest_thread_first(self):
        _insert_event(name="older thread", mentioned_hours_ago=40)
        _insert_event(name="soup", mentioned_hours_ago=14)
        threads = events.get_recent_open_threads(1)
        self.assertEqual(threads[0]["event_name"], "soup")


class WhenLabelTest(unittest.TestCase):
    def _local(self, days_ago: int, hour: int) -> str:
        now_local = datetime.now(timezone.utc).astimezone()
        target = (now_local - timedelta(days=days_ago)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        return target.isoformat()

    def test_yesterday_evening_is_last_night(self):
        self.assertEqual(events.mentioned_when_label(self._local(1, 21)), "last night")

    def test_yesterday_morning_is_yesterday(self):
        self.assertEqual(events.mentioned_when_label(self._local(1, 9)), "yesterday")

    def test_same_day_is_earlier_today(self):
        now_local = datetime.now(timezone.utc).astimezone()
        # Any hour today that is not in the future relative to "now".
        self.assertEqual(
            events.mentioned_when_label(self._local(0, max(0, now_local.hour - 1))),
            "earlier today")

    def test_two_days_ago(self):
        self.assertEqual(
            events.mentioned_when_label(self._local(2, 12)), "a couple of days ago")

    def test_three_days_ago_is_the_other_day(self):
        self.assertEqual(
            events.mentioned_when_label(self._local(3, 12)), "the other day")

    def test_garbage_timestamp_is_safe(self):
        self.assertEqual(events.mentioned_when_label(None), "the other day")
        self.assertEqual(events.mentioned_when_label("not-a-date"), "the other day")


if __name__ == "__main__":
    unittest.main()
