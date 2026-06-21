"""
Tier D — unified cross-silo retrieval (memory/retrieval.py):

  * one global budget shared by facts + interests (no wasted per-silo slots),
  * topic relevance lifts the connected item into the selection,
  * boundary mute terms still suppress facts,
  * a registered relevance backend (the semantic seam) is used.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import database as db, retrieval


def _build_people_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")


class _PeopleDbTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        _build_people_db(self._path)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        retrieval.set_relevance_backend(None)
        self._patch.stop()
        self._tmp.cleanup()


class UnifiedRetrievalTest(_PeopleDbTestCase):
    # Distinct, non-fuzzy-similar names (numbered names like "interest0"/"interest1"
    # would be folded together by the Phase-B write-time dedup).
    _INTEREST_WORDS = [
        "kayaking", "painting", "chess", "gardening", "astronomy", "baking",
        "cycling", "photography", "hiking", "pottery", "sailing", "origami",
        "fishing", "climbing", "knitting", "surfing", "archery", "fencing",
        "birding", "woodworking", "calligraphy", "juggling",
    ]

    def _seed(self, n_facts, n_interests):
        from memory import facts, interests
        for i in range(n_facts):
            facts.add_fact(1, "other", f"factkey{i}", f"value {i}", source="explicit")
        for i in range(n_interests):
            interests.upsert_interest(1, self._INTEREST_WORDS[i], interest_strength="medium")

    def test_global_budget_caps_total(self):
        self._seed(20, 20)
        with mock.patch.object(config, "MEMORY_PROMPT_BUDGET_ITEMS", 16):
            bundle = retrieval.retrieve_person_memory(1)
        total = len(bundle["facts"]) + len(bundle["interests"])
        self.assertEqual(total, 16)

    def test_budget_reallocates_to_the_richer_silo(self):
        # Many facts, few interests → facts get most of the budget (no wasted interest
        # slots, unlike the old fixed 12+8).
        self._seed(20, 2)
        with mock.patch.object(config, "MEMORY_PROMPT_BUDGET_ITEMS", 16):
            bundle = retrieval.retrieve_person_memory(1)
        self.assertEqual(len(bundle["interests"]), 2)   # both weak interests kept
        self.assertEqual(len(bundle["facts"]), 14)       # remainder of budget → facts

    def test_topic_relevant_item_is_selected(self):
        from memory import facts, interests
        # A big pool of generic facts plus one on-topic interest.
        self._seed(20, 0)
        interests.upsert_interest(1, "kayaking", interest_strength="low")
        with mock.patch.object(config, "MEMORY_PROMPT_BUDGET_ITEMS", 6):
            bundle = retrieval.retrieve_person_memory(1, topic_tokens={"kayaking"})
        names = {it["name"] for it in bundle["interests"]}
        self.assertIn("kayaking", names)  # relevance lifted a low interest into a tight budget

    def test_mute_terms_suppress_facts(self):
        from memory import facts
        facts.add_fact(1, "family", "mother", "lost his mother", source="explicit")
        facts.add_fact(1, "identity", "city", "Sacramento", source="explicit")
        bundle = retrieval.retrieve_person_memory(1, mute_terms={"mother"})
        keys = {f["key"] for f in bundle["facts"]}
        self.assertNotIn("mother", keys)
        self.assertIn("city", keys)

    def test_registered_relevance_backend_is_used(self):
        from memory import facts
        self._seed(20, 0)
        facts.add_fact(1, "other", "special", "the magic one", source="explicit")
        # Semantic-seam stand-in: rate anything containing 'magic' maximally relevant.
        retrieval.set_relevance_backend(
            lambda tt, text, cap: float(cap) if "magic" in text.lower() else 0.0
        )
        with mock.patch.object(config, "MEMORY_PROMPT_BUDGET_ITEMS", 3):
            bundle = retrieval.retrieve_person_memory(1, topic_tokens={"anything"})
        keys = {f["key"] for f in bundle["facts"]}
        self.assertIn("special", keys)  # backend forced the magic fact into a tiny budget

    def test_disabled_flag_uses_legacy_per_silo(self):
        # With the flag off, _build_person_context takes the legacy path — sanity that
        # retrieve() still works standalone regardless (the flag is read by the caller).
        self._seed(3, 3)
        bundle = retrieval.retrieve_person_memory(1)
        self.assertEqual(len(bundle["facts"]) + len(bundle["interests"]), 6)


if __name__ == "__main__":
    unittest.main()
