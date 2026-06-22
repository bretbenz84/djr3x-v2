"""Tests for the callback-humor feature (docs/callback_humor_design.md).

Covers the storage lifecycle (memory/callbacks.py against a temp people.db),
the deterministic sensitivity wall, the banker's parsing/grounding guards,
every reactive trigger gate, spend-at-speak settle semantics, the lull pick,
boundary/forget retirement, and the schema wiring (migrations, person-table
delete coverage).
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _make_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA

    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)


def _frame(allow_roast="normal", purpose="answer"):
    return SimpleNamespace(allow_roast=allow_roast, purpose=purpose)


def _comedy(allow_callback=True):
    return SimpleNamespace(allow_callback=allow_callback, key="dry_ack")


class _TempDbCase(unittest.TestCase):
    """Shared temp-people.db scaffolding + engine state reset."""

    def setUp(self):
        from memory import database as db
        from intelligence import callback_engine

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patches = [mock.patch.object(db, "_DB_FILE", db_path)]
        for p in self._patches:
            p.start()
        callback_engine.reset_state_for_tests()
        from memory import people as people_memory
        self.person_id = people_memory.enroll_person("Bret")

    def tearDown(self):
        from intelligence import callback_engine

        callback_engine.reset_state_for_tests()
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _bank(self, premise="does astrophotography and prints telescopes",
              topic="astrophotography", sensitivity="safe", **kw):
        from memory import callbacks
        return callbacks.bank(
            self.person_id, premise,
            category=kw.pop("category", "passion"),
            topic=topic, sensitivity=sensitivity, **kw,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity wall
# ─────────────────────────────────────────────────────────────────────────────

class SensitivityWallTests(unittest.TestCase):
    def test_every_protected_category_trips(self):
        from intelligence import callback_engine as ce

        cases = [
            ("I was diagnosed with something last year", "health"),
            ("therapy has been helping me a lot", "health"),
            ("my mother passed away in March", "grief"),
            ("I've been trying to lose weight", "body"),
            ("I came out to my parents last year", "orientation_romance"),
            ("my friend is gay but keeps it quiet", "orientation_romance"),
            ("I'm drowning in debt right now", "finances"),
            ("I got laid off last month", "finances"),
            ("I go to church every Sunday", "religion_politics"),
            ("going through a divorce right now", "family_conflict"),
            ("I'm two years sober", "addiction_legal"),
            ("my visa situation is a mess", "addiction_legal"),
        ]
        for text, expected in cases:
            self.assertEqual(
                ce.protected_category_hit(text), expected, f"for {text!r}"
            )

    def test_safe_material_does_not_trip(self):
        from intelligence import callback_engine as ce

        for text in [
            "I do astrophotography and 3D-print my own telescopes",
            "I build mechanical keyboards from scratch",
            "pineapple on pizza is a crime against nature",
            "I'm a night owl, mornings are for droids",
            "I collect vintage synthesizers",
            "I've been painting tiny miniatures all week",  # 'pain' inside 'painting'
        ]:
            self.assertIsNone(ce.protected_category_hit(text), f"for {text!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Storage lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class StorageLifecycleTests(_TempDbCase):
    def test_bank_and_active_pool_hard_filter(self):
        from memory import callbacks

        safe_id = self._bank()
        self._bank(premise="misses his hometown sometimes",
                   topic="hometown nostalgia", sensitivity="guarded")
        self._bank(premise="had surgery last year",
                   topic="knee surgery", sensitivity="excluded")
        pool = callbacks.active_pool(self.person_id)
        self.assertEqual([row["id"] for row in pool], [safe_id])
        # ... but all three rows exist for audit/idempotence.
        self.assertEqual(len(callbacks.get_all(self.person_id)), 3)

    def test_upsert_keeps_usage_and_only_demotes_sensitivity(self):
        from memory import callbacks

        row_id = self._bank()
        callbacks.mark_used(row_id)
        # Re-bank same topic: premise refreshes, usage history survives.
        again = self._bank(premise="prints telescope parts at 2am")
        self.assertEqual(again, row_id)
        row = callbacks.get_all(self.person_id)[0]
        self.assertEqual(row["premise"], "prints telescope parts at 2am")
        self.assertEqual(row["use_count"], 1)
        self.assertIsNotNone(row["last_used_at"])
        # Demotion sticks; promotion is refused.
        self._bank(sensitivity="guarded")
        self.assertEqual(callbacks.get_all(self.person_id)[0]["sensitivity"], "guarded")
        self._bank(sensitivity="safe")
        self.assertEqual(callbacks.get_all(self.person_id)[0]["sensitivity"], "guarded")

    def test_retired_rows_stay_retired_through_rebank(self):
        from memory import callbacks

        row_id = self._bank()
        callbacks.retire(row_id, "boundary: roast astrophotography")
        self._bank(premise="still loves astrophotography")
        self.assertEqual(callbacks.active_pool(self.person_id), [])
        row = callbacks.get_all(self.person_id)[0]
        self.assertIsNotNone(row["retired_at"])

    def test_cooldown_and_freshness_decay(self):
        import config
        from memory import callbacks

        row_id = self._bank()
        row = callbacks.get_all(self.person_id)[0]
        self.assertTrue(callbacks.off_cooldown(row))
        callbacks.mark_used(row_id)
        row = callbacks.get_all(self.person_id)[0]
        with mock.patch.object(config, "CALLBACK_REUSE_COOLDOWN_DAYS", 7, create=True):
            self.assertFalse(callbacks.off_cooldown(row))
            future = datetime.now(timezone.utc) + timedelta(days=8)
            self.assertTrue(callbacks.off_cooldown(row, now=future))
        with mock.patch.object(config, "CALLBACK_USE_DECAY_HALFLIFE_USES", 3, create=True):
            self.assertAlmostEqual(
                callbacks.freshness_factor({"use_count": 3}), 0.5, places=3
            )
            self.assertAlmostEqual(
                callbacks.freshness_factor({"use_count": 0}), 1.0, places=3
            )

    def test_pool_cap_retires_overflow(self):
        import config
        from memory import callbacks

        with mock.patch.object(config, "CALLBACK_BANK_MAX_PER_PERSON", 3, create=True):
            for i in range(5):
                self._bank(premise=f"premise number {i}", topic=f"topic {i}")
            self.assertEqual(len(callbacks.active_pool(self.person_id)), 3)
        retired = [
            r for r in callbacks.get_all(self.person_id) if r["retired_at"]
        ]
        self.assertEqual(len(retired), 2)
        self.assertTrue(all(r["retired_reason"] == "pool_overflow" for r in retired))

    def test_bank_gated_by_kill_switch(self):
        import config
        from memory import callbacks

        with mock.patch.object(config, "CALLBACK_BANK_ENABLED", False, create=True):
            self.assertIsNone(self._bank())
        self.assertEqual(callbacks.get_all(self.person_id), [])

    def test_boundary_retires_matching_premises(self):
        from memory import boundaries, callbacks

        self._bank()
        self._bank(premise="collects vintage synthesizers", topic="synthesizers")
        applied = boundaries.apply_detected_boundary(
            self.person_id,
            {"action": "add", "behavior": "roast", "topic": "astrophotography",
             "source_text": "stop joking about my astrophotography"},
        )
        self.assertEqual(applied["action"], "add")
        pool = callbacks.active_pool(self.person_id)
        self.assertEqual([r["topic_slug"] for r in pool], ["synthesizers"])

    def test_forget_flow_deletes_matching_rows(self):
        from memory import callbacks, forgetting

        self._bank()
        self._bank(premise="collects vintage synthesizers", topic="synthesizers")
        result = forgetting.forget_specific_memory(self.person_id, "astrophotography")
        self.assertEqual(result.deleted.get("callbacks"), 1)
        remaining = callbacks.get_all(self.person_id)
        self.assertEqual([r["topic_slug"] for r in remaining], ["synthesizers"])

    def test_delete_person_clears_callback_rows(self):
        from memory import callbacks, people as people_memory

        self._bank()
        people_memory.delete_person(self.person_id)
        self.assertEqual(callbacks.get_all(self.person_id), [])

    def test_migrations_add_table_to_old_db(self):
        from memory import database as db

        old = Path(self._tmp.name) / "old.db"
        with sqlite3.connect(old) as conn:
            conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
        with mock.patch.object(db, "_DB_FILE", old):
            db._run_migrations()
            row = db.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='person_callback_material'"
            )
        self.assertIsNotNone(row)


# ─────────────────────────────────────────────────────────────────────────────
# Banker
# ─────────────────────────────────────────────────────────────────────────────

class BankerTests(_TempDbCase):
    """bank_from_turn is inert under the test runner, so these tests opt back
    in by patching _under_test_runner (the temp DB makes the writes safe)."""

    def setUp(self):
        super().setUp()
        from intelligence import callback_engine as ce

        patcher = mock.patch.object(ce, "_under_test_runner", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_suppressed_under_test_runner_by_default(self):
        from intelligence import callback_engine as ce
        from memory import callbacks

        with mock.patch.object(ce, "_under_test_runner", return_value=True):
            self.assertFalse(ce._llm_allowed())
            self.assertIsNone(ce.bank_from_turn(
                self.person_id, "I collect vintage synthesizers and pedals"
            ))
            # refresh_relevance must not even read the DB under the runner.
            self._bank()
            ce.refresh_relevance(self.person_id)
            with ce._lock:
                self.assertIsNone(ce._relevance_stash)
        self.assertEqual(len(callbacks.get_all(self.person_id)), 1)

    def test_llm_candidate_happy_path_banks_safe(self):
        from intelligence import callback_engine as ce
        from memory import callbacks

        raw = (
            "Found: yes\n"
            "Premise: does astrophotography and prints his own telescopes\n"
            "Topic: astrophotography\n"
            "Category: passion"
        )
        with (
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value=raw),
        ):
            row_id = ce.bank_from_turn(
                self.person_id,
                "I do astrophotography and I 3D-print my own telescopes",
            )
        self.assertIsNotNone(row_id)
        row = callbacks.get_all(self.person_id)[0]
        self.assertEqual(row["sensitivity"], "safe")
        self.assertEqual(row["category"], "passion")
        self.assertEqual(row["topic_slug"], "astrophotography")

    def test_protected_content_lands_excluded_even_if_model_says_yes(self):
        from intelligence import callback_engine as ce
        from memory import callbacks

        raw = (
            "Found: yes\n"
            "Premise: spends weekends at the hospital volunteering\n"
            "Topic: hospital volunteering\n"
            "Category: hobby"
        )
        with (
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value=raw),
        ):
            row_id = ce.bank_from_turn(
                self.person_id, "I spend my weekends at the hospital volunteering",
            )
        self.assertIsNotNone(row_id)
        row = callbacks.get_all(self.person_id)[0]
        self.assertEqual(row["sensitivity"], "excluded")
        self.assertEqual(callbacks.active_pool(self.person_id), [])

    def test_hallucinated_premise_is_dropped(self):
        from intelligence import callback_engine as ce
        from memory import callbacks

        raw = (
            "Found: yes\n"
            "Premise: races motorcycles on weekends\n"
            "Topic: motorcycle racing\n"
            "Category: hobby"
        )
        with (
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value=raw),
        ):
            row_id = ce.bank_from_turn(
                self.person_id, "the weather has been pretty nice lately honestly",
            )
        self.assertIsNone(row_id)
        self.assertEqual(callbacks.get_all(self.person_id), [])

    def test_heuristic_fallback_when_llm_unavailable(self):
        from intelligence import callback_engine as ce
        from memory import callbacks

        with mock.patch.object(ce, "_llm_allowed", return_value=False):
            row_id = ce.bank_from_turn(
                self.person_id, "honestly I'm really into astrophotography these days",
            )
        self.assertIsNotNone(row_id)
        row = callbacks.get_all(self.person_id)[0]
        self.assertEqual(row["category"], "passion")
        self.assertIn("astrophotography", row["premise"])

    def test_short_or_personless_turns_skipped(self):
        from intelligence import callback_engine as ce

        with mock.patch.object(ce, "_llm_allowed", return_value=True):
            self.assertIsNone(ce.bank_from_turn(self.person_id, "yeah totally"))
            self.assertIsNone(ce.bank_from_turn(None, "I collect vintage synths and pedals"))


# ─────────────────────────────────────────────────────────────────────────────
# Reactive trigger
# ─────────────────────────────────────────────────────────────────────────────

class ReactiveTriggerTests(_TempDbCase):
    """Drive maybe_claim_reactive gate by gate. Helpers that read live robot
    state are patched green, then un-greened one at a time."""

    def setUp(self):
        super().setUp()
        from intelligence import callback_engine as ce

        self.ce = ce
        self.premise_id = self._bank()
        self._stash(self.premise_id)
        # Green-light every environment-reading helper; individual tests
        # flip one back to red.
        self._green = [
            mock.patch.object(ce, "_empathy_clear", return_value=True),
            mock.patch.object(ce, "unacked_emotional_event_pending", return_value=False),
            mock.patch.object(ce, "_crowd_ok", return_value=True),
            mock.patch.object(ce, "_tier_eligible", return_value=True),
            mock.patch.object(ce, "_restraint_preferred", return_value=False),
            mock.patch.object(ce, "_boundary_blocked", return_value=False),
            mock.patch.object(ce, "_transcript_len", return_value=10),
            mock.patch("intelligence.repair_moves.recent_tone_repair", return_value=False),
            mock.patch("intelligence.topic_thread.snapshot", return_value={
                "emotional_weight": "light", "user_stance": "engaged",
            }),
            mock.patch("intelligence.topic_thread.arc_reads_flat", return_value=False),
            mock.patch("random.random", return_value=0.0),
        ]
        for p in self._green:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._green])

    def _stash(self, premise_id, *, score=1.0, transcript_len=10):
        with self.ce._lock:
            self.ce._relevance_stash = {
                "person_id": self.person_id,
                "premise_id": premise_id,
                "score": score,
                "transcript_len": transcript_len,
                "ts": 0.0,
            }

    def _claim(self, text="the sky is so clear tonight, perfect evening",
               frame=None, comedy=None, **kw):
        return self.ce.maybe_claim_reactive(
            self.person_id, text,
            frame=frame or _frame(),
            comedy_mode=comedy if comedy is not None else _comedy(),
            **kw,
        )

    def test_happy_path_claims_and_chain_stands_down(self):
        claim = self._claim()
        self.assertIsNotNone(claim)
        self.assertEqual(claim.premise_id, self.premise_id)
        self.assertTrue(self.ce.turn_claim_active(self.person_id))
        self.assertFalse(self.ce.turn_claim_active(self.person_id + 99))
        directive = self.ce.build_callback_directive(claim)
        self.assertIn("astrophotography", directive)
        self.assertIn("affectionate", directive)

    def test_humor_kill_switch(self):
        import config
        with mock.patch.object(config, "CALLBACK_HUMOR_ENABLED", False, create=True):
            self.assertIsNone(self._claim())

    def test_straight_comedy_mode_blocks(self):
        self.assertIsNone(self._claim(comedy=_comedy(allow_callback=False)))

    def test_none_roast_frame_blocks_light_allowed_by_default(self):
        # 'none' is never callback territory; 'light' IS allowed by default now
        # (CALLBACK_ALLOW_LIGHT_ROAST_FRAME defaults True) — 'brief'/'micro', the
        # common conversational target, downgrade 'normal'->'light', so requiring
        # exactly 'normal' confined reactive callbacks to a near-silent surface.
        self.assertIsNone(self._claim(frame=_frame(allow_roast="none")))
        self.assertIsNotNone(self._claim(frame=_frame(allow_roast="light")))

    def test_light_frame_blocked_when_flag_disabled(self):
        import config
        with mock.patch.object(
            config, "CALLBACK_ALLOW_LIGHT_ROAST_FRAME", False, create=True
        ):
            self.assertIsNone(self._claim(frame=_frame(allow_roast="light")))

    def test_safety_purposes_block(self):
        for purpose in ("closure", "repair", "identity", "answer_ack", "boundary"):
            self.assertIsNone(self._claim(frame=_frame(purpose=purpose)), purpose)

    def test_sensitive_live_text_blocks_and_arms_sober_room(self):
        self.assertIsNone(self._claim(text="my friend died yesterday actually"))
        # The heavy moment it noted now also blocks a follow-up clean turn.
        self.assertIsNone(self._claim())

    def test_boundary_live_text_blocks(self):
        self.assertIsNone(self._claim(text="can we change the subject please"))

    def test_caring_empathy_state_blocks(self):
        with mock.patch.object(self.ce, "_empathy_clear", return_value=False):
            self.assertIsNone(self._claim())

    def test_unacked_emotional_event_blocks(self):
        with mock.patch.object(
            self.ce, "unacked_emotional_event_pending", return_value=True
        ):
            self.assertIsNone(self._claim())

    def test_tone_repair_blocks(self):
        with mock.patch(
            "intelligence.repair_moves.recent_tone_repair", return_value=True
        ):
            self.assertIsNone(self._claim())

    def test_heavy_thread_avoidant_stance_and_flat_arc_block(self):
        with mock.patch(
            "intelligence.topic_thread.snapshot",
            return_value={"emotional_weight": "heavy", "user_stance": "engaged"},
        ):
            self.assertIsNone(self._claim())
        with mock.patch(
            "intelligence.topic_thread.snapshot",
            return_value={"emotional_weight": "light", "user_stance": "avoidant"},
        ):
            self.assertIsNone(self._claim())
        with mock.patch("intelligence.topic_thread.arc_reads_flat", return_value=True):
            self.assertIsNone(self._claim())

    def test_crowd_tier_restraint_and_consent_block(self):
        with mock.patch.object(self.ce, "_crowd_ok", return_value=False):
            self.assertIsNone(self._claim())
        with mock.patch.object(self.ce, "_tier_eligible", return_value=False):
            self.assertIsNone(self._claim())
        with mock.patch.object(self.ce, "_restraint_preferred", return_value=True):
            self.assertIsNone(self._claim())
        with mock.patch.object(self.ce, "_boundary_blocked", return_value=True):
            self.assertIsNone(self._claim())

    def test_stale_weak_or_foreign_stash_blocks(self):
        self._stash(self.premise_id, transcript_len=2)  # 8 lines stale > max 4
        self.assertIsNone(self._claim())
        self._stash(self.premise_id, score=0.4)  # weak < 0.5 threshold
        self.assertIsNone(self._claim())
        with self.ce._lock:
            self.ce._relevance_stash = None
        self.assertIsNone(self._claim())

    def test_settle_spends_only_when_voiced(self):
        from memory import callbacks

        claim = self._claim()
        self.assertIsNotNone(claim)
        # Not voiced → released, no spend, claim gone.
        self.ce.settle_turn("Quiet night in the cantina, huh.")
        self.assertEqual(callbacks.get_all(self.person_id)[0]["use_count"], 0)
        self.assertFalse(self.ce.turn_claim_active(self.person_id))

        # Soft backoff: an immediately following turn can't re-claim...
        self.assertIsNone(self._claim())
        # ...but two exchanges later it can. A reply that merely echoes the
        # TOPIC word (normal on-topic conversation, bit skipped) must not
        # spend either — that's the false-positive case.
        with mock.patch.object(self.ce, "_transcript_len", return_value=13):
            claim = self._claim()
            self.assertIsNotNone(claim)
        self.ce.settle_turn("Astrophotography talk again. Bold choice.")
        self.assertEqual(callbacks.get_all(self.person_id)[0]["use_count"], 0)

        # A line carrying premise content beyond the topic word DOES spend.
        self._stash(self.premise_id, transcript_len=16)  # fresh relevance read
        with mock.patch.object(self.ce, "_transcript_len", return_value=16):
            claim = self._claim()
            self.assertIsNotNone(claim)
        self.ce.settle_turn(
            "Counting ceiling panels — fewer than the stars you chase with "
            "those printed telescopes of yours."
        )
        row = callbacks.get_all(self.person_id)[0]
        self.assertEqual(row["use_count"], 1)
        self.assertIsNotNone(row["last_used_at"])

    def test_session_cap_and_no_repeat(self):
        import config
        from memory import callbacks

        with mock.patch.object(config, "CALLBACK_MAX_PER_SESSION", 1, create=True):
            claim = self._claim()
            self.assertIsNotNone(claim)
            self.ce.settle_turn(
                "Still printing telescopes for the astrophotography habit, I see."
            )
            self.assertEqual(callbacks.get_all(self.person_id)[0]["use_count"], 1)
            # Cap hit → no more claims this session even with a fresh stash.
            other = self._bank(premise="collects synthesizers", topic="synths")
            self._stash(other, transcript_len=30)
            with mock.patch.object(self.ce, "_transcript_len", return_value=30):
                self.assertIsNone(self._claim())
        # New session resets the ledger.
        self.ce.clear_session()
        self._stash(other, transcript_len=30)
        with mock.patch.object(self.ce, "_transcript_len", return_value=30):
            self.assertIsNotNone(self._claim())


# ─────────────────────────────────────────────────────────────────────────────
# Lull path
# ─────────────────────────────────────────────────────────────────────────────

class LullPathTests(_TempDbCase):
    def setUp(self):
        super().setUp()
        from intelligence import callback_engine as ce

        self.ce = ce
        self._green = [
            mock.patch.object(ce, "_empathy_clear", return_value=True),
            mock.patch.object(ce, "unacked_emotional_event_pending", return_value=False),
            mock.patch.object(ce, "_crowd_ok", return_value=True),
            mock.patch.object(ce, "_tier_eligible", return_value=True),
            mock.patch.object(ce, "_restraint_preferred", return_value=False),
            mock.patch.object(ce, "_boundary_blocked", return_value=False),
            mock.patch.object(ce, "_transcript_len", return_value=10),
            mock.patch("intelligence.repair_moves.recent_tone_repair", return_value=False),
        ]
        for p in self._green:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._green])

    def test_gates_clear_then_each_blocker(self):
        import config

        self.assertTrue(self.ce.lull_gates_clear(self.person_id))
        with mock.patch.object(config, "CALLBACK_HUMOR_ENABLED", False, create=True):
            self.assertFalse(self.ce.lull_gates_clear(self.person_id))
        with mock.patch.object(config, "CALLBACK_LULL_ENABLED", False, create=True):
            self.assertFalse(self.ce.lull_gates_clear(self.person_id))
        self.ce.note_heavy_moment()
        self.assertFalse(self.ce.lull_gates_clear(self.person_id))
        self.ce.reset_state_for_tests()
        with mock.patch.object(self.ce, "_empathy_clear", return_value=False):
            self.assertFalse(self.ce.lull_gates_clear(self.person_id))
        self.assertFalse(self.ce.lull_gates_clear(None))

    def test_pick_prefers_same_session_and_skips_spent(self):
        from memory import callbacks

        old = self._bank(premise="collects synthesizers", topic="synths",
                         session_id="elsewhere")
        fresh = self._bank(premise="prints telescopes at 2am",
                           topic="telescopes", session_id=self.ce.session_token())
        picked = self.ce.pick_lull_premise(self.person_id)
        self.assertEqual(picked["id"], fresh)

        self.ce.spend_lull_premise(picked)
        self.assertEqual(
            callbacks.get_all(self.person_id)[1]["use_count"], 1
        )
        # Spent-this-session premise is skipped; the other one wins.
        picked2 = self.ce.pick_lull_premise(self.person_id)
        self.assertEqual(picked2["id"], old)

    def test_lull_prompt_carries_premise_and_safety_rules(self):
        prompt = self.ce.build_lull_prompt(
            "Bret", {"premise": "does astrophotography and prints telescopes"}
        )
        self.assertIn("astrophotography", prompt)
        self.assertIn("gone quiet", prompt)
        for banned in ("body", "health", "money", "grief"):
            self.assertIn(banned, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Relevance judge
# ─────────────────────────────────────────────────────────────────────────────

class RelevanceTests(_TempDbCase):
    def test_deterministic_overlap_wins_without_model(self):
        from intelligence import callback_engine as ce

        pool = [
            {"id": 7, "premise": "does astrophotography", "topic_slug": "astrophotography"},
            {"id": 8, "premise": "collects synthesizers", "topic_slug": "synths"},
        ]
        with mock.patch.object(ce, "_llm_allowed", return_value=False):
            best, score = ce._judge_relevance(
                "topic: stargazing | the astrophotography rig is out tonight", pool
            )
        self.assertEqual((best, score), (7, 1.0))

    def test_model_verdict_parsed_and_validated(self):
        from intelligence import callback_engine as ce

        pool = [
            {"id": 7, "premise": "does astrophotography", "topic_slug": "astrophotography"},
            {"id": 8, "premise": "collects synthesizers", "topic_slug": "synths"},
        ]
        with (
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value="Match: 2\nStrength: strong"),
        ):
            self.assertEqual(ce._judge_relevance("music gear talk", pool), (8, 1.0))
        with (
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value="Match: none\nStrength: none"),
        ):
            self.assertEqual(ce._judge_relevance("music gear talk", pool), (None, 0.0))
        with (  # out-of-range index → fail closed
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", return_value="Match: 9\nStrength: strong"),
        ):
            self.assertEqual(ce._judge_relevance("music gear talk", pool), (None, 0.0))
        with (  # backend down → fail closed
            mock.patch.object(ce, "_llm_allowed", return_value=True),
            mock.patch.object(ce, "_generate", side_effect=RuntimeError("ollama down")),
        ):
            self.assertEqual(ce._judge_relevance("music gear talk", pool), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
