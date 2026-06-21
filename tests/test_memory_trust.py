"""
Tier C — "trust the store" tests:

  C1: extracted facts are provisional (inferred), not faked explicit/0.95.
  C2: evidence_count only grows on SPACED corroboration, not within-session repeats.
  C3: time-bound facts get a short horizon; stale uncorroborated fast-decay facts are
      dropped from prompt injection (the decay queue).
  C4: an active "don't bring up X" boundary suppresses the matching fact from injection.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import config
from memory import database as db


def _build_people_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")


class _PeopleDbTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        _build_people_db(self._path)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _exec(self, sql, params=()):
        with sqlite3.connect(self._path) as conn:
            conn.execute(sql, params)

    def _rows(self, sql, params=()):
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ── C1 ──────────────────────────────────────────────────────────────────────

class ExtractedFactProvenanceTest(unittest.TestCase):
    def test_provisional_flag_routes_to_inferred(self):
        from intelligence import interaction
        with mock.patch.object(config, "MEMORY_EXTRACTED_FACTS_PROVISIONAL", True):
            self.assertEqual(interaction._extracted_fact_provenance(), ("inferred", None))
        with mock.patch.object(config, "MEMORY_EXTRACTED_FACTS_PROVISIONAL", False):
            self.assertEqual(interaction._extracted_fact_provenance(), ("explicit", 0.95))


class InferredFactStorageTest(_PeopleDbTestCase):
    def test_inferred_fact_is_low_confidence_fast_decay_and_hedged(self):
        from memory import facts
        facts.add_fact(1, "other", "hobby", "maybe likes kayaking", source="inferred")
        row = self._rows("SELECT * FROM person_facts WHERE person_id=1")[0]
        self.assertLess(float(row["confidence"]), 0.7)       # not a fake 0.95
        self.assertEqual(row["decay_rate"], "fast")
        rendered = facts.format_fact_for_prompt(facts.get_facts(1)[0])
        self.assertIn("inferred", rendered.lower())


# ── C2 ──────────────────────────────────────────────────────────────────────

class EvidenceWindowTest(_PeopleDbTestCase):
    def test_same_session_repeats_do_not_inflate_evidence(self):
        from memory import facts
        facts.add_fact(1, "family", "kids", "two kids", source="explicit")
        facts.add_fact(1, "family", "kids", "two kids", source="explicit")
        facts.add_fact(1, "family", "kids", "two kids", source="explicit")
        row = self._rows("SELECT evidence_count FROM person_facts WHERE person_id=1")[0]
        self.assertEqual(row["evidence_count"], 1)  # not 3

    def test_spaced_reconfirmation_counts(self):
        from memory import facts
        facts.add_fact(1, "family", "kids", "two kids", source="explicit")
        # Age the last confirmation beyond the reconfirm window.
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        self._exec("UPDATE person_facts SET last_confirmed_at=? WHERE person_id=1", (old,))
        facts.add_fact(1, "family", "kids", "two kids", source="explicit")
        row = self._rows("SELECT evidence_count FROM person_facts WHERE person_id=1")[0]
        self.assertEqual(row["evidence_count"], 2)


# ── C3 ──────────────────────────────────────────────────────────────────────

class TimeBoundFactTest(_PeopleDbTestCase):
    def test_future_phrase_sets_fast_decay_and_short_horizon(self):
        from memory import facts
        facts.add_fact(1, "plan", "trip", "going camping next month", source="explicit")
        row = self._rows("SELECT * FROM person_facts WHERE person_id=1")[0]
        self.assertEqual(row["decay_rate"], "fast")
        self.assertLessEqual(int(row["stale_after_days"]), 45)

    def test_stale_uncorroborated_fast_fact_is_dropped_from_injection(self):
        from memory import facts
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        # Stale + fast + single-evidence → decay queue (dropped).
        self._exec(
            "INSERT INTO person_facts (person_id, category, key, value, confidence, source, "
            "created_at, updated_at, last_confirmed_at, evidence_count, importance, decay_rate, stale_after_days) "
            "VALUES (1,'plan','trip','camping next month',0.55,'inferred',?,?,?,1,0.4,'fast',12)",
            (old, old, old),
        )
        self.assertEqual(facts.get_prompt_worthy_facts(1), [])
        # get_facts (direct recall) still sees it.
        self.assertEqual(len(facts.get_facts(1)), 1)

    def test_corroborated_stale_fact_survives(self):
        from memory import facts
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self._exec(
            "INSERT INTO person_facts (person_id, category, key, value, confidence, source, "
            "created_at, updated_at, last_confirmed_at, evidence_count, importance, decay_rate, stale_after_days) "
            "VALUES (1,'plan','trip','camping spot',0.7,'explicit',?,?,?,2,0.5,'fast',12)",
            (old, old, old),
        )
        self.assertEqual(len(facts.get_prompt_worthy_facts(1)), 1)


# ── C4 ──────────────────────────────────────────────────────────────────────

class BoundaryFactMuteTest(_PeopleDbTestCase):
    def test_conversation_boundary_terms(self):
        from memory import boundaries
        boundaries.add_boundary(1, "bring up", "my mother")
        self.assertIn("mother", boundaries.muted_topic_terms(1))

    def test_preference_boundary_terms(self):
        from memory import preferences, boundaries
        preferences.upsert_preference(
            1, "interaction", "boundary", "mother_topic", "do not bring up his mother"
        )
        self.assertIn("mother", boundaries.muted_topic_terms(1))

    def test_muted_fact_is_suppressed_but_others_kept(self):
        from memory import facts, boundaries
        facts.add_fact(1, "family", "mother", "lost his mother years ago", source="explicit")
        facts.add_fact(1, "identity", "city", "Sacramento", source="explicit")
        boundaries.add_boundary(1, "bring up", "mother")
        mute = boundaries.muted_topic_terms(1)
        kept = facts.get_prompt_worthy_facts(1, mute_terms=mute)
        keys = {f["key"] for f in kept}
        self.assertNotIn("mother", keys)   # boundary-covered fact suppressed
        self.assertIn("city", keys)         # unrelated fact preserved
        # Direct recall is unaffected.
        self.assertEqual(len(facts.get_facts(1)), 2)

    def test_ask_boundary_does_not_mute_fact(self):
        from memory import boundaries
        # "don't ASK about X" should NOT suppress the fact (Rex may still know it).
        boundaries.add_boundary(1, "ask", "mother")
        self.assertNotIn("mother", boundaries.muted_topic_terms(1))


if __name__ == "__main__":
    unittest.main()
