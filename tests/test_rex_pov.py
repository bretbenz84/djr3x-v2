"""
Tests for intelligence/rex_pov.py — Rex's persistent "current preoccupation" POV.

Deterministic and suite-safe: selection is driven entirely by injected `context`
and `exchange` (the transcript-length hold clock), so these never touch world_state,
the conversation arc, or the network. A separate class validates the SHIPPED seed
pool in config (incl. the user's hard "no cantina" constraint).
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import rex_pov

# Captured BEFORE any test patches config, so the shipped-pool guard sees the real pool.
_SHIPPED_SEEDS = list(getattr(config, "REX_POV_SEEDS", []) or [])

# Mixed pool for selection/bias tests: people-tagged, quiet-tagged, and neutral.
_TEST_SEEDS = [
    {"id": "p1", "pov": "POV people one.", "fits": ["people", "any"]},
    {"id": "p2", "pov": "POV people two.", "fits": ["people", "any"]},
    {"id": "q1", "pov": "POV quiet one.", "fits": ["quiet", "any"]},
    {"id": "q2", "pov": "POV quiet two.", "fits": ["quiet", "any"]},
    {"id": "a1", "pov": "POV any one.", "fits": ["any"]},
]

# All-neutral pool so every seed ties → exercises pure anti-repeat rotation.
_ROT_SEEDS = [{"id": f"s{i}", "pov": f"POV {i}.", "fits": ["any"]} for i in range(5)]

_PEOPLE = {"people": True, "flat": False}
_QUIET = {"people": False, "flat": False}


class RexPovSelectionTest(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(config, "REX_POV_ENABLED", True),
            mock.patch.object(config, "REX_POV_SEEDS", _TEST_SEEDS),
            mock.patch.object(config, "REX_POV_MIN_HOLD_EXCHANGES", 2),
            mock.patch.object(config, "REX_POV_MAX_HOLD_EXCHANGES", 6),
        ]
        for p in self._patches:
            p.start()
        rex_pov.clear()

    def tearDown(self):
        rex_pov.clear()
        for p in self._patches:
            p.stop()

    # ── holding policy ────────────────────────────────────────────────────────
    def test_carries_within_min_hold(self):
        d0 = rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        first = rex_pov.active_seed_id()
        d1 = rex_pov.current_pov_directive(context=_PEOPLE, exchange=1)  # held=1 < MIN
        self.assertEqual(rex_pov.active_seed_id(), first)
        self.assertEqual(d1, d0)

    def test_reselects_after_max_hold(self):
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        first = rex_pov.active_seed_id()
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=6)  # held=6 >= MAX
        self.assertNotEqual(rex_pov.active_seed_id(), first)

    def test_reselects_on_context_change_after_min(self):
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        first = rex_pov.active_seed_id()
        rex_pov.current_pov_directive(context=_QUIET, exchange=2)  # held=2 >= MIN, sig changed
        self.assertNotEqual(rex_pov.active_seed_id(), first)
        # quiet context should now prefer a quiet-tagged seed
        self.assertIn(rex_pov.active_seed_id(), {"q1", "q2"})

    def test_min_hold_wins_over_context_change(self):
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        first = rex_pov.active_seed_id()
        # context flips but held=1 < MIN → must keep carrying the same POV
        rex_pov.current_pov_directive(context=_QUIET, exchange=1)
        self.assertEqual(rex_pov.active_seed_id(), first)

    def test_flat_flip_changes_signature_and_reselects(self):
        rex_pov.current_pov_directive(context={"people": True, "flat": False}, exchange=0)
        first = rex_pov.active_seed_id()
        rex_pov.current_pov_directive(context={"people": True, "flat": True}, exchange=2)
        self.assertNotEqual(rex_pov.active_seed_id(), first)

    # ── hybrid context bias ───────────────────────────────────────────────────
    def test_people_context_prefers_people_seed(self):
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        self.assertIn(rex_pov.active_seed_id(), {"p1", "p2"})

    def test_quiet_context_prefers_quiet_seed(self):
        rex_pov.current_pov_directive(context=_QUIET, exchange=0)
        self.assertIn(rex_pov.active_seed_id(), {"q1", "q2"})

    # ── anti-repeat / rotation ────────────────────────────────────────────────
    def test_anti_repeat_no_consecutive_and_full_cycle(self):
        ids = []
        ex = 0
        with mock.patch.object(config, "REX_POV_SEEDS", _ROT_SEEDS):
            rex_pov.clear()
            for _ in range(10):
                rex_pov.current_pov_directive(context=_PEOPLE, exchange=ex)
                ids.append(rex_pov.active_seed_id())
                ex += 6  # >= MAX each step → force a re-selection every call
        # never the same POV twice in a row
        for a, b in zip(ids, ids[1:]):
            self.assertNotEqual(a, b)
        # the first full cycle (pool size) visits every seed exactly once
        self.assertEqual(set(ids[:5]), {s["id"] for s in _ROT_SEEDS})

    # ── lifecycle / gating ────────────────────────────────────────────────────
    def test_clear_resets_state(self):
        rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        self.assertIsNotNone(rex_pov.active_seed_id())
        rex_pov.clear()
        self.assertIsNone(rex_pov.active_seed_id())
        self.assertEqual(rex_pov._used_ids, set())

    def test_kill_switch_disables_everything(self):
        with mock.patch.object(config, "REX_POV_ENABLED", False):
            rex_pov.clear()
            self.assertEqual(rex_pov.current_pov_directive(context=_PEOPLE, exchange=0), "")
            self.assertEqual(rex_pov.active_pov_text(context=_PEOPLE, exchange=0), "")
            self.assertIsNone(rex_pov.active_seed_id())

    def test_empty_pool_is_safe(self):
        with mock.patch.object(config, "REX_POV_SEEDS", []):
            rex_pov.clear()
            self.assertEqual(rex_pov.current_pov_directive(context=_PEOPLE, exchange=0), "")
            self.assertIsNone(rex_pov.active_seed_id())

    # ── rendered output ───────────────────────────────────────────────────────
    def test_directive_contains_pov_and_volunteer_instruction(self):
        d = rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        self.assertIn("POV people one.", d)
        self.assertIn("preoccupation", d.lower())
        self.assertIn("volunteer", d.lower())

    def test_active_pov_text_matches_active_seed(self):
        text = rex_pov.active_pov_text(context=_PEOPLE, exchange=0)
        self.assertEqual(text, "POV people one.")
        self.assertEqual(rex_pov.active_seed_id(), "p1")
        # and the directive embeds that same text
        d = rex_pov.current_pov_directive(context=_PEOPLE, exchange=0)
        self.assertIn(text, d)


class RexPovShippedPoolTest(unittest.TestCase):
    """Validates the real authored pool/knobs in config (not the test fixtures)."""

    def test_kill_switch_flag_exists_and_default_on(self):
        self.assertTrue(getattr(config, "REX_POV_ENABLED", False))

    def test_hold_knobs_exist_and_ordered(self):
        lo = getattr(config, "REX_POV_MIN_HOLD_EXCHANGES")
        hi = getattr(config, "REX_POV_MAX_HOLD_EXCHANGES")
        self.assertGreaterEqual(int(hi), int(lo))

    def test_pool_nonempty_and_wellformed(self):
        self.assertTrue(_SHIPPED_SEEDS, "REX_POV_SEEDS must not be empty")
        ids = []
        for seed in _SHIPPED_SEEDS:
            self.assertIsInstance(seed, dict)
            self.assertTrue(str(seed.get("id") or "").strip(), f"seed missing id: {seed}")
            self.assertTrue(str(seed.get("pov") or "").strip(), f"seed missing pov: {seed}")
            fits = seed.get("fits", ["any"])
            self.assertIsInstance(fits, (list, tuple))
            ids.append(seed["id"])
        self.assertEqual(len(ids), len(set(ids)), "seed ids must be unique")

    def test_no_cantina_in_shipped_pool(self):
        # Hard user constraint: Rex usually isn't in a cantina — keep seeds venue-neutral.
        for seed in _SHIPPED_SEEDS:
            blob = f"{seed.get('id', '')} {seed.get('pov', '')}".lower()
            self.assertNotIn("cantina", blob, f"seed leans on 'cantina': {seed.get('id')}")


if __name__ == "__main__":
    unittest.main()
