"""
tests/test_open_threads.py — cross-session open-thread follow-ups: freshness
window, once-ever spending (persisted), age phrasing. Temp rex.db, no LLM.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from memory import rex_db


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class OpenThreadsTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = getattr(config, "REX_DB_PATH", None)
        config.REX_DB_PATH = str(Path(self._tmp.name) / "rex.db")
        rex_db.ensure_schema()
        from intelligence import open_threads
        self.ot = open_threads

    def tearDown(self):
        config.REX_DB_PATH = self._orig
        self._tmp.cleanup()

    def _episode(self, *, person_id=1, age_hours=24.0, threads=None, asked=None):
        detail = {}
        if threads is not None:
            detail["open_threads"] = threads
        if asked is not None:
            detail["threads_asked"] = asked
        return rex_db.execute(
            "INSERT INTO rex_episodes "
            "(created_at, kind, summary, person_id, person_name, detail, salience, session_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_iso(datetime.now() - timedelta(hours=age_hours)), "conversation_summary",
             "Bret told me things.", person_id, "Bret",
             json.dumps(detail), 0.6, "run-test"),
        )

    def test_pending_returns_fresh_threads(self):
        self._episode(age_hours=24, threads=["whether the motor swap happened"])
        got = self.ot.pending_for_person(1)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["thread"], "whether the motor swap happened")

    def test_too_fresh_and_too_old_excluded(self):
        self._episode(age_hours=1, threads=["too fresh"])              # < 6h
        self._episode(age_hours=30 * 24, threads=["too old"])          # > 21d
        self.assertEqual(self.ot.pending_for_person(1), [])

    def test_other_person_excluded(self):
        self._episode(person_id=2, age_hours=24, threads=["their thing"])
        self.assertEqual(self.ot.pending_for_person(1), [])

    def test_mark_asked_is_permanent(self):
        ep = self._episode(age_hours=24, threads=["thread A", "thread B"])
        got = self.ot.pending_for_person(1)
        self.assertEqual(len(got), 2)
        self.ot.mark_asked(ep, "thread A")
        remaining = self.ot.pending_for_person(1)
        self.assertEqual([t["thread"] for t in remaining], ["thread B"])
        # Persisted in the row itself (survives restart).
        row = rex_db.fetchone("SELECT detail FROM rex_episodes WHERE id = ?", (ep,))
        self.assertIn("thread A", json.loads(row["detail"])["threads_asked"])

    def test_pre_asked_threads_skipped(self):
        self._episode(age_hours=24, threads=["done", "open"], asked=["done"])
        self.assertEqual([t["thread"] for t in self.ot.pending_for_person(1)], ["open"])

    def test_describe_age(self):
        self.assertEqual(self.ot.describe_age(0.4), "earlier today")
        self.assertEqual(self.ot.describe_age(1.4), "yesterday")
        self.assertEqual(self.ot.describe_age(4.0), "4 days ago")
        self.assertEqual(self.ot.describe_age(15.0), "a while back")


if __name__ == "__main__":
    unittest.main()


class ConsolidationTest(unittest.TestCase):
    """memory/consolidation.py retention rules on a temp rex.db."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = getattr(config, "REX_DB_PATH", None)
        config.REX_DB_PATH = str(Path(self._tmp.name) / "rex.db")
        rex_db.ensure_schema()
        from memory import consolidation
        self.con = consolidation

    def tearDown(self):
        config.REX_DB_PATH = self._orig
        self._tmp.cleanup()

    def _seen(self, person_id, age_days, hour):
        ts = (datetime.now() - timedelta(days=age_days)).replace(hour=hour, minute=0, second=0)
        rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, person_id, salience, session_id) "
            "VALUES (?,?,?,?,?,?)",
            (_iso(ts), "person_seen", "I saw Bret.", person_id, 0.45, "t"),
        )

    def test_person_seen_dedup_and_ageout(self):
        self._seen(1, 0, 9)
        self._seen(1, 0, 15)        # same day: dedup to newest
        self._seen(1, 2, 12)        # different day: kept
        self._seen(1, 60, 12)       # past retention: deleted
        self.con.run()
        rows = rex_db.fetchall("SELECT created_at FROM rex_episodes WHERE kind='person_seen'")
        self.assertEqual(len(rows), 2)

    def test_stale_pending_room_question_dismissed(self):
        old = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        rex_db.execute(
            "INSERT INTO room_objects (label, first_seen, last_seen, sighting_count, ask_status) "
            "VALUES ('guitar', ?, ?, 5, 'pending')", (old, old))
        fresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rex_db.execute(
            "INSERT INTO room_objects (label, first_seen, last_seen, sighting_count, ask_status) "
            "VALUES ('ladder', ?, ?, 5, 'pending')", (fresh, fresh))
        self.con.run()
        rows = {r["label"]: r["ask_status"] for r in rex_db.fetchall(
            "SELECT label, ask_status FROM room_objects")}
        self.assertEqual(rows["guitar"], "dismissed")
        self.assertEqual(rows["ladder"], "pending")


class DatedFollowupExpiryTest(unittest.TestCase):
    """memory/events.get_pending_followups: dated events past FOLLOWUP_DATED_MAX_AGE_DAYS
    are lazily expired (field 2026-07-18: the week-old dentist opener)."""

    def test_stale_dated_event_expired(self):
        import sqlite3
        from unittest import mock
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE person_events (id INTEGER PRIMARY KEY, person_id INT, "
            "event_name TEXT, event_date TEXT, mentioned_at TEXT, "
            "followed_up BOOL DEFAULT FALSE, status TEXT DEFAULT 'planned')"
        )
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        recent_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO person_events (person_id, event_name, event_date, mentioned_at) "
                     "VALUES (1, 'dentist appointment', ?, ?)", (old_date, now))
        conn.execute("INSERT INTO person_events (person_id, event_name, event_date, mentioned_at) "
                     "VALUES (1, 'job interview', ?, ?)", (recent_date, now))

        class FakeDb:
            @staticmethod
            def execute(q, p=()):
                conn.execute(q, p); conn.commit()
            @staticmethod
            def fetchall(q, p=()):
                return conn.execute(q, p).fetchall()

        from memory import events as events_mod
        with mock.patch.object(events_mod, "db", FakeDb):
            pending = events_mod.get_pending_followups(1)
        names = [e["event_name"] for e in pending]
        self.assertIn("job interview", names)
        self.assertNotIn("dentist appointment", names)
        # ...and the stale one was permanently marked, not just filtered.
        row = conn.execute(
            "SELECT followed_up FROM person_events WHERE event_name='dentist appointment'"
        ).fetchone()
        self.assertTrue(row["followed_up"])


class ResolvedPlanGuardTest(unittest.TestCase):
    """An episode thread about a plan a follow-up already RESOLVED must not
    re-ask it (field 2026-08-19 20:01: 'did that actually happen?' 48 s after
    'No, I didn't go' settled the library plan in person_events)."""

    # Reuse the rex.db fixture by delegation — subclassing would re-run all of
    # OpenThreadsTest's own tests under this class too.
    _episode = OpenThreadsTest._episode

    def setUp(self):
        OpenThreadsTest.setUp(self)
        import sqlite3
        import tempfile
        from unittest import mock
        from memory import database as people_db
        from setup_assets import DB_SCHEMA
        self._ptmp = tempfile.TemporaryDirectory()
        path = Path(self._ptmp.name) / "people.db"
        with sqlite3.connect(path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")
            conn.execute(
                "INSERT INTO person_events (person_id, event_name, event_date, "
                "status, followed_up, outcome, mentioned_at, updated_at) VALUES "
                "(1, 'visit presidential library', NULL, 'completed', 1, "
                "'No, I didn''t go.', ?, ?)",
                (datetime.now().astimezone().isoformat(),
                 datetime.now().astimezone().isoformat()),
            )
        self._ppatch = mock.patch.object(people_db, "_DB_FILE", path)
        self._ppatch.start()

    def tearDown(self):
        self._ppatch.stop()
        self._ptmp.cleanup()
        OpenThreadsTest.tearDown(self)

    def test_thread_covering_a_resolved_plan_is_dropped(self):
        self._episode(threads=[
            "whether Bret's presidential library visit ended up happening",
            "how the garden project is going",
        ])
        pending = [p["thread"] for p in self.ot.pending_for_person(1)]
        self.assertNotIn(
            "whether Bret's presidential library visit ended up happening", pending)
        self.assertIn("how the garden project is going", pending)

    def test_single_shared_token_never_nukes_a_thread(self):
        # 'library' alone must not kill an unrelated thread; the guard needs the
        # plan's full content-token set inside the thread.
        self._episode(threads=["the little free library Bret is building"])
        pending = [p["thread"] for p in self.ot.pending_for_person(1)]
        self.assertIn("the little free library Bret is building", pending)


class SessionAwarenessTest(OpenThreadsTest):
    """Field 2026-09-01 23:05:13: the reply model asked about the circus at 23:01:49
    (previous session's recap in its prompt), Bret answered at length, and the lull
    lane — which only knows about threads IT spent — asked "The circus came up the
    other day — did you end up enjoying it?" four minutes later. A thread whose
    subject already came up this session is HELD (not spent)."""

    def _pending_with_session(self, turns):
        from unittest import mock
        from memory import dedup
        toks = set()
        for t in turns:
            toks.update(dedup._token_set(t))
        with mock.patch.object(self.ot, "_session_transcript_tokens",
                               return_value=frozenset(toks)):
            return self.ot.pending_for_person(1)

    def test_field_case_circus_thread_is_held(self):
        self._episode(age_hours=48, threads=["whether the circus happened"])
        got = self._pending_with_session([
            "So, did the circus actually happen, or did reality cancel the act?",
            "Oh, the circus was fantastic. We had a lot of fun at the circus.",
        ])
        self.assertEqual(got, [])

    def test_held_threads_are_not_spent(self):
        self._episode(age_hours=48, threads=["whether the circus happened"])
        self._pending_with_session(["the circus was fantastic"])
        # Next session, nothing about the circus said yet → it is still pending.
        got = self._pending_with_session(["how are you doing today"])
        self.assertEqual([g["thread"] for g in got], ["whether the circus happened"])

    def test_unrelated_session_leaves_the_thread_pending(self):
        self._episode(age_hours=48, threads=["whether the dentist appointment happened"])
        got = self._pending_with_session([
            "I am going to Atlanta, Georgia with my father.",
            "We're not leaving until the eighth.",
        ])
        self.assertEqual(len(got), 1)

    def test_frame_words_alone_never_match(self):
        # "happened"/"whether"/"how"/"went" are how a thread is PHRASED, not its subject.
        self._episode(age_hours=48, threads=["whether the motor swap happened"])
        got = self._pending_with_session(["what happened yesterday, how did it go"])
        self.assertEqual(len(got), 1)

    def test_half_of_a_multi_word_core_is_enough(self):
        self._episode(age_hours=48, threads=["how the Huntsville space center visit went"])
        got = self._pending_with_session(["then we're going to the Huntsville Space Center"])
        self.assertEqual(got, [])

    def test_empty_session_transcript_holds_nothing(self):
        self._episode(age_hours=48, threads=["whether the circus happened"])
        self.assertEqual(len(self._pending_with_session([])), 1)


class GameMechanicsReadFilterTest(OpenThreadsTest):
    """Threads stored BEFORE the game-mechanics guard shipped must die at read
    time too (field 2026-08-26: episode 902 held "whether T'Joy's points were
    actually taken away" and "how the game will proceed next" from a Jeopardy
    session, and the lull lane spoke both)."""

    def test_stored_game_mechanics_threads_are_dropped(self):
        self._episode(threads=[
            "whether T'Joy's points were actually taken away",
            "how the game will proceed next",
            "whether the motor swap happened",
        ])
        pending = [p["thread"] for p in self.ot.pending_for_person(1)]
        self.assertEqual(pending, ["whether the motor swap happened"])


class CuriosityReadFilterTest(OpenThreadsTest):
    """Preference / favourite / performance-bit shapes are questions Rex WANTS
    to ask, not something the person left unresolved — and stored ones must
    die at read time (field 2026-09-03 11:45: "Impersonate Bill Clinton" became
    "whether Bret has any favorite quotes from Clinton", spoken the next morning
    as "The Clinton quotes came up the other day — did you ever pick one?")."""

    def test_stored_curiosity_threads_are_dropped(self):
        self._episode(threads=[
            "whether Bret has any favorite quotes from Clinton",
            "what other impersonations Bret wants to hear",
            "Bret's opinion on the new Star Trek captain",
            "whether Bret went to the presidential library this week",
            "how Toby is adjusting to being blind",
        ])
        pending = [p["thread"] for p in self.ot.pending_for_person(1)]
        self.assertEqual(pending, [
            "whether Bret went to the presidential library this week",
            "how Toby is adjusting to being blind",
        ])

    def test_shapes(self):
        rx = self.ot.CURIOSITY_RE
        for t in ("whether Bret has any favorite quotes from Clinton",
                  "which quotes Bret likes", "if PJ prefers the old mic",
                  "Bret's opinion on the captain", "whether Rex should do another impression",
                  "whether Bret wants Rex to impersonate Trump again",
                  "how Bret and JT met"):
            self.assertTrue(rx.search(t), t)
        for t in ("whether the dentist appointment happened",
                  "how the trip to Atlanta went",
                  "what mounting scheme Bret decided on for the sensors",
                  "whether the motor swap happened",
                  "how long they will stay in Huntsville"):
            self.assertFalse(rx.search(t), t)
