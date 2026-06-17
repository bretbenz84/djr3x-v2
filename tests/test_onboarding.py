"""Tests for the new-person onboarding burst (intelligence/onboarding.py).

Covers the pure logic (answer sentiment -> retort bank, exit/disengagement
detection, value tidying, the templated depth follow-up) and the DB-backed
pieces against a temp people.db (eligibility gating, tier-ordered question
selection with asked/answered/known-fact/boundary skips, and the answer ->
memory writes with the familiarity bump).
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config


def _make_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA

    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# Pure logic — no DB
# ─────────────────────────────────────────────────────────────────────────────

class SentimentRetortTests(unittest.TestCase):
    def test_classify_answer_sentiments(self):
        from intelligence import onboarding

        self.assertEqual(onboarding.classify_answer("I don't know"), "flat")
        self.assertEqual(onboarding.classify_answer("nope"), "flat")
        self.assertEqual(
            onboarding.classify_answer("Because it honestly means a lot to me"), "warm"
        )
        self.assertEqual(
            onboarding.classify_answer("I absolutely love rock climbing!"), "positive"
        )
        self.assertEqual(
            onboarding.classify_answer("Actually I've been a pilot for twenty years"),
            "surprise",
        )
        self.assertEqual(onboarding.classify_answer("I'm an accountant"), "neutral")

    def test_retort_is_short_and_not_a_question(self):
        from intelligence import onboarding

        for answer in ("I'm an accountant", "I don't know", "I love climbing!",
                       "It means a lot", "Actually, never"):
            retort = onboarding.retort_for(answer)
            self.assertTrue(retort, "retort should be non-empty")
            self.assertNotIn("?", retort, f"retort must not be a question: {retort!r}")
            self.assertLessEqual(len(retort.split()), 5, f"retort too long: {retort!r}")

    def test_warm_answer_draws_from_warm_bank(self):
        from intelligence import onboarding

        with mock.patch.object(
            config, "COMEDY_LINE_BANKS",
            {"onboarding_retort_warm": ["WARMONLY"],
             "onboarding_retort_neutral": ["NEUTRALONLY"]},
        ):
            self.assertEqual(
                onboarding.retort_for("Honestly, because my family means everything"),
                "WARMONLY",
            )
            self.assertEqual(onboarding.retort_for("an accountant"), "NEUTRALONLY")


class ExitDetectionTests(unittest.TestCase):
    def test_hard_decline(self):
        from intelligence import onboarding

        for text in ("I'd rather not say", "stop asking me questions",
                     "none of your business", "change the subject"):
            self.assertTrue(onboarding.is_hard_decline(text), text)
        self.assertFalse(onboarding.is_hard_decline("I'm a teacher from Ohio"))

    def test_pivot(self):
        from intelligence import onboarding

        for text in ("can you play some music?", "what's the weather?",
                     "what about you?", "let's play a game"):
            self.assertTrue(onboarding.is_pivot(text), text)
        self.assertFalse(onboarding.is_pivot("I work in finance"))
        self.assertFalse(onboarding.is_pivot("I grew up in Texas"))

    def test_soft_disengage(self):
        from intelligence import onboarding

        for text in ("", "meh", "dunno", "I don't know", "stuff"):
            self.assertTrue(onboarding.is_soft_disengage(text), repr(text))
        self.assertFalse(onboarding.is_soft_disengage("I'm a software engineer"))


class TidyValueTests(unittest.TestCase):
    def test_strips_filler_and_framing(self):
        from intelligence import onboarding

        self.assertEqual(
            onboarding.tidy_value("um, I'm a paramedic actually", "fact").lower(),
            "paramedic actually",
        )
        self.assertEqual(
            onboarding.tidy_value("I love jazz mostly", "interest").lower(), "jazz mostly"
        )
        self.assertEqual(onboarding.tidy_value("I don't know", "fact"), "")
        self.assertEqual(onboarding.tidy_value("", "interest"), "")

    def test_caps_length(self):
        from intelligence import onboarding

        long_answer = "I do a whole bunch of different unrelated random things every single day honestly"
        self.assertLessEqual(len(onboarding.tidy_value(long_answer, "fact").split()), 10)

    def test_clause_trim_and_bad_lead_guard(self):
        from intelligence import onboarding

        # Em-dash aside is trimmed; a comma value ("Austin, Texas") survives.
        self.assertEqual(
            onboarding.tidy_value("rock climbing — I'm obsessed", "interest").lower(),
            "rock climbing",
        )
        self.assertEqual(onboarding.tidy_value("Austin, Texas", "fact"), "Austin, Texas")
        # Non-noun answers are dropped rather than filed as junk.
        self.assertEqual(onboarding.tidy_value("going great, better than I hoped", "interest"), "")
        self.assertEqual(onboarding.tidy_value("nothing in particular", "interest"), "")


class FollowupTemplateTests(unittest.TestCase):
    def test_template_fallback_when_llm_disabled(self):
        from intelligence import onboarding

        with mock.patch.object(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", False, create=True):
            q = onboarding.generate_followup("rock climbing every weekend")
            self.assertTrue(q.endswith("?"))
            self.assertIn("get into", q.lower())
            # A vague (non-topic) answer yields no template question, so selection
            # falls through to an authored Tier-C question instead of garbling.
            self.assertIsNone(onboarding.generate_followup("it's going great, better than I hoped"))
        self.assertIsNone(onboarding.generate_followup(""))

    def test_openai_path_used_when_enabled(self):
        from intelligence import onboarding

        with mock.patch.object(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", True, create=True), \
                mock.patch("intelligence.llm.generate_curiosity_question",
                           return_value="Sure! What got you hooked on climbing?") as gen:
            q = onboarding.generate_followup(
                "rock climbing", person_id=7, prev_question="What are you into?"
            )
        gen.assert_called_once()
        self.assertEqual(q, "What got you hooked on climbing?")

    def test_openai_empty_skips_without_template(self):
        from intelligence import onboarding

        # An empty LLM return (e.g. a heavy/sensitive answer, where the curiosity
        # generator deliberately stays quiet) must NOT fall back to the template.
        with mock.patch.object(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", True, create=True), \
                mock.patch("intelligence.llm.generate_curiosity_question", return_value=""):
            self.assertIsNone(onboarding.generate_followup("my mom just passed away", person_id=7))

    def test_first_question_extraction(self):
        from intelligence import onboarding

        self.assertEqual(
            onboarding._first_question('Sure! How long have you climbed? And more.'),
            "How long have you climbed?",
        )
        self.assertEqual(onboarding._first_question("What got you started"), "What got you started?")
        # A statement gets normalized to a question without a stray ".?".
        self.assertEqual(
            onboarding._first_question("That bus breaks down every other mile."),
            "That bus breaks down every other mile?",
        )
        # Runaway output is rejected so the caller falls back to the template.
        self.assertEqual(onboarding._first_question(" ".join(["word"] * 25) + "?"), "")


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed — temp people.db
# ─────────────────────────────────────────────────────────────────────────────

class OnboardingDBTests(unittest.TestCase):
    def setUp(self):
        from memory import database as db

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patches = [
            mock.patch.object(db, "_DB_FILE", db_path),
            mock.patch.object(config, "ONBOARDING_ENABLED", True, create=True),
        ]
        for p in self._patches:
            p.start()

        from memory import people as people_memory

        self.people = people_memory
        self.person_id, _ = people_memory.find_or_create_person("Sarah")

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    # ── eligibility ──────────────────────────────────────────────────────────
    def test_eligible_fresh_person(self):
        from intelligence import onboarding

        self.assertTrue(onboarding.eligible(self.person_id))

    def test_disabled_flag_blocks(self):
        from intelligence import onboarding

        with mock.patch.object(config, "ONBOARDING_ENABLED", False):
            self.assertFalse(onboarding.eligible(self.person_id))

    def test_too_many_facts_blocks(self):
        from intelligence import onboarding
        from memory import facts as facts_memory

        # profile_fact_count excludes identity/appearance/boundary/relationship,
        # so eligibility is tripped by substantive (preference/interest) facts.
        for i in range(4):
            facts_memory.add_fact(self.person_id, "preference", f"pref_{i}", "x", "explicit")
        with mock.patch.object(config, "ONBOARDING_FACT_FLOOR", 3):
            self.assertFalse(onboarding.eligible(self.person_id))

    def test_minor_blocked(self):
        from intelligence import onboarding
        from memory import facts as facts_memory

        facts_memory.add_fact(self.person_id, "identity", "age_category", "child", "explicit")
        self.assertFalse(onboarding.eligible(self.person_id))

    # ── selection ────────────────────────────────────────────────────────────
    def test_selection_tier_order(self):
        from intelligence import onboarding

        first = onboarding.next_question(self.person_id, asked_keys=set())
        self.assertEqual(first["key"], "job")
        self.assertEqual(first["tier"], "A")

        nxt = onboarding.next_question(
            self.person_id, asked_keys={"job", "how_found_rex", "hometown"}
        )
        self.assertEqual(nxt["tier"], "B")

    def test_known_fact_key_skipped(self):
        from intelligence import onboarding
        from memory import facts as facts_memory

        facts_memory.add_fact(self.person_id, "identity", "job", "teacher", "explicit")
        q = onboarding.next_question(self.person_id, asked_keys=set())
        self.assertNotEqual(q["key"], "job")

    def test_answered_question_skipped(self):
        from intelligence import onboarding
        from memory import relationships as rel_memory

        rel_memory.save_qa(self.person_id, "job", "what do you do?", "engineer", 1)
        q = onboarding.next_question(self.person_id, asked_keys=set())
        self.assertNotEqual(q["key"], "job")

    def test_depth_gated(self):
        from intelligence import onboarding

        all_ab = {e["key"] for e in config.ONBOARDING_QUESTION_POOL if e["tier"] in ("A", "B")}
        # With depth disallowed and only Tier-C left, nothing should come back.
        self.assertIsNone(
            onboarding.next_question(self.person_id, asked_keys=all_ab, allow_depth=False)
        )
        # With depth allowed and a prior answer, a Tier-C question appears.
        with mock.patch.object(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", False):
            q = onboarding.next_question(
                self.person_id, asked_keys=all_ab, allow_depth=True,
                last_answer="competitive chess",
            )
        self.assertIsNotNone(q)
        self.assertEqual(q["tier"], "C")

    def test_followup_needs_prior_answer(self):
        from intelligence import onboarding

        # origin_followup (text=None) must be skipped when there is no last answer;
        # selection should fall through to an authored Tier-C question.
        all_ab = {e["key"] for e in config.ONBOARDING_QUESTION_POOL if e["tier"] in ("A", "B")}
        q = onboarding.next_question(
            self.person_id, asked_keys=all_ab, allow_depth=True, last_answer=None
        )
        self.assertIsNotNone(q)
        self.assertNotEqual(q["key"], "origin_followup")
        self.assertIsNotNone(q["text"])

    # ── answer -> memory ─────────────────────────────────────────────────────
    def test_record_answer_writes_fact_and_bumps_familiarity(self):
        from intelligence import onboarding
        from memory import facts as facts_memory
        from memory import relationships as rel_memory

        before = self.people.get_person(self.person_id)["familiarity_score"]
        jobq = next(e for e in config.ONBOARDING_QUESTION_POOL if e["key"] == "job")
        onboarding.note_question_asked(self.person_id, {**jobq, "text": jobq["text"]})
        onboarding.record_answer(self.person_id, jobq, "um, I'm a paramedic actually")

        facts = {f["key"]: f["value"] for f in facts_memory.get_facts(self.person_id)}
        self.assertIn("job", facts)
        self.assertIn("paramedic", facts["job"].lower())

        answered = rel_memory.get_answered_question_keys(self.person_id)
        self.assertIn("job", answered)

        after = self.people.get_person(self.person_id)["familiarity_score"]
        self.assertGreater(after, before)

    def test_record_answer_writes_interest(self):
        from intelligence import onboarding
        from memory import interests as interests_memory

        musicq = next(e for e in config.ONBOARDING_QUESTION_POOL if e["key"] == "favorite_music")
        onboarding.note_question_asked(self.person_id, {**musicq, "text": musicq["text"]})
        onboarding.record_answer(self.person_id, musicq, "mostly jazz and funk")

        names = [i["name"].lower() for i in interests_memory.get_interests_for_prompt(self.person_id)]
        self.assertTrue(any("jazz" in n for n in names), names)


# ─────────────────────────────────────────────────────────────────────────────
# Flow wiring — interaction.py state machine
# ─────────────────────────────────────────────────────────────────────────────

class OnboardingFlowTests(unittest.TestCase):
    def setUp(self):
        from memory import database as db

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patches = [
            mock.patch.object(db, "_DB_FILE", db_path),
            mock.patch.object(config, "ONBOARDING_ENABLED", True, create=True),
            mock.patch.object(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", False, create=True),
        ]
        for p in self._patches:
            p.start()

        import intelligence.interaction as interaction
        from intelligence import onboarding
        from memory import people as people_memory

        self.interaction = interaction
        self.onboarding = onboarding
        self.people = people_memory
        interaction._pending_onboarding = None
        self.person_id, _ = people_memory.find_or_create_person("Sarah")

    def tearDown(self):
        self.interaction._pending_onboarding = None
        self.interaction._low_memory_idle_questions_spoken.discard(self.person_id)
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _arm_awaiting(self, question_key="job"):
        import time

        q = dict(next(e for e in config.ONBOARDING_QUESTION_POOL if e["key"] == question_key))
        self.onboarding.note_question_asked(self.person_id, q)
        self.interaction._pending_onboarding = {
            "person_id": self.person_id, "name": "Sarah", "step": "awaiting_answer",
            "pending_question": q, "asked_keys": {question_key}, "asked_count": 1,
            "answered_count": 0, "soft_streak": 0, "since_reveal": 1,
            "last_answer": None, "created_at": time.monotonic(), "asked_at": time.monotonic(),
        }

    # ── begin / active ───────────────────────────────────────────────────────
    def test_begin_arms_flow(self):
        self.interaction._maybe_begin_onboarding(self.person_id, "Sarah")
        self.assertTrue(self.interaction.onboarding_flow_active())
        self.assertEqual(self.interaction._pending_onboarding["step"], "kickoff")

    def test_begin_noop_when_disabled(self):
        with mock.patch.object(config, "ONBOARDING_ENABLED", False):
            self.interaction._maybe_begin_onboarding(self.person_id, "Sarah")
        self.assertFalse(self.interaction.onboarding_flow_active())

    # ── answer loop ──────────────────────────────────────────────────────────
    def test_answer_writes_fact_and_advances(self):
        from memory import facts as facts_memory

        self._arm_awaiting("job")
        resp = self.interaction._handle_onboarding_turn("I'm a paramedic", self.person_id)
        self.assertIsNotNone(resp)
        self.assertIn("?", resp)  # carries the next question
        facts = {f["key"]: f["value"] for f in facts_memory.get_facts(self.person_id)}
        self.assertIn("job", facts)
        state = self.interaction._pending_onboarding
        self.assertEqual(state["answered_count"], 1)
        self.assertEqual(state["asked_count"], 2)
        self.assertEqual(state["step"], "awaiting_answer")

    def test_retort_leads_the_reply(self):
        # Bank-fallback path (answer-aware reaction disabled): the authored retort
        # still leads the reply, ahead of the next question.
        with mock.patch.object(config, "ONBOARDING_LLM_REACT_ENABLED", False), \
            mock.patch.object(
                config, "COMEDY_LINE_BANKS",
                {"onboarding_retort_neutral": ["Noted."], "onboarding_retort_positive": ["Noted."],
                 "onboarding_retort_warm": ["Noted."], "onboarding_retort_surprise": ["Noted."]},
            ):
            self._arm_awaiting("job")
            resp = self.interaction._handle_onboarding_turn("an accountant", self.person_id)
        self.assertTrue(resp.startswith("Noted."), resp)

    def test_answer_aware_reaction_leads_the_reply(self):
        # Answer-aware path: the genuine, content-reflecting reaction (not a flat bank
        # pick) leads the reply, ahead of the next question — the "I created you" fix.
        from intelligence import llm as llm_module
        with mock.patch.object(config, "ONBOARDING_LLM_REACT_ENABLED", True), \
            mock.patch.object(
                llm_module, "generate_onboarding_reaction",
                return_value="Wait, you BUILT me?",
            ):
            self._arm_awaiting("job")
            resp = self.interaction._handle_onboarding_turn("I created you", self.person_id)
        self.assertTrue(resp.startswith("Wait, you BUILT me?"), resp)
        self.assertIn("?", resp)  # still carries the next question

    def test_hard_decline_backs_off_and_closes(self):
        self._arm_awaiting("job")
        resp = self.interaction._handle_onboarding_turn("I'd rather not say", self.person_id)
        self.assertIsNotNone(resp)
        self.assertNotIn("?", resp)
        self.assertIsNone(self.interaction._pending_onboarding)

    def test_pivot_releases_turn_and_closes(self):
        self._arm_awaiting("job")
        resp = self.interaction._handle_onboarding_turn("can you play some music?", self.person_id)
        self.assertIsNone(resp)  # released to normal routing
        self.assertIsNone(self.interaction._pending_onboarding)

    def test_speaker_mismatch_keeps_flow_open(self):
        self._arm_awaiting("job")
        resp = self.interaction._handle_onboarding_turn("hello", 99999)
        self.assertIsNone(resp)
        self.assertIsNotNone(self.interaction._pending_onboarding)

    def test_reaches_max_closes_without_question(self):
        self._arm_awaiting("job")
        state = self.interaction._pending_onboarding
        state["asked_count"] = self.onboarding.max_questions()
        state["answered_count"] = self.onboarding.max_questions() - 1
        resp = self.interaction._handle_onboarding_turn("I'm an engineer", self.person_id)
        self.assertIsNotNone(resp)
        self.assertNotIn("?", resp)
        self.assertIsNone(self.interaction._pending_onboarding)

    def test_soft_answers_do_not_abort_before_min(self):
        self._arm_awaiting("job")
        resp = self.interaction._handle_onboarding_turn("dunno", self.person_id)
        # below MIN: keep going, do not close
        self.assertIsNotNone(resp)
        self.assertIsNotNone(self.interaction._pending_onboarding)

    def test_wind_down_after_min_on_soft_streak(self):
        self._arm_awaiting("job")
        state = self.interaction._pending_onboarding
        state["answered_count"] = self.onboarding.min_questions()
        state["soft_streak"] = int(getattr(config, "ONBOARDING_SOFT_DISENGAGE_LIMIT", 2)) - 1
        resp = self.interaction._handle_onboarding_turn("meh", self.person_id)
        self.assertIsNotNone(resp)
        self.assertNotIn("?", resp)  # winds down with a closer, no further question
        self.assertIsNone(self.interaction._pending_onboarding)

    def test_close_suppresses_low_memory_question(self):
        I = self.interaction
        I._low_memory_idle_questions_spoken.discard(self.person_id)
        self._arm_awaiting("job")
        state = I._pending_onboarding
        state["asked_count"] = self.onboarding.max_questions()
        state["answered_count"] = self.onboarding.max_questions() - 1
        I._handle_onboarding_turn("an engineer", self.person_id)
        self.assertIsNone(I._pending_onboarding)
        # The separate low-memory idle profile question must not pile on this session.
        self.assertIn(self.person_id, I._low_memory_idle_questions_spoken)

    def test_intro_answer_gate(self):
        I = self.interaction
        with mock.patch.object(I, "_response_wait_active", return_value=True):
            # No newcomer + awaiting an answer => "this is X" is the ANSWER.
            self.assertTrue(I._intro_is_answer_to_rex_question(False))
            # A genuinely present newcomer still introduces fine.
            self.assertFalse(I._intro_is_answer_to_rex_question(True))
        with mock.patch.object(I, "_response_wait_active", return_value=False):
            # Not awaiting a reply => normal introduction path (off-camera intro).
            self.assertFalse(I._intro_is_answer_to_rex_question(False))

    # ── kickoff ──────────────────────────────────────────────────────────────
    def test_kickoff_fires_opener(self):
        import time

        self.interaction._maybe_begin_onboarding(self.person_id, "Sarah")
        self.interaction._pending_onboarding["asked_at"] = time.monotonic() - 5.0
        with mock.patch.object(self.interaction, "_speak_proactive", return_value=True) as sp, \
                mock.patch.object(self.interaction.speech_queue, "is_speaking", return_value=False), \
                mock.patch.object(self.interaction.output_gate, "is_busy", return_value=False), \
                mock.patch.object(self.interaction.echo_cancel, "is_suppressed", return_value=False), \
                mock.patch.object(self.interaction.end_thread, "is_grace_active", return_value=False), \
                mock.patch.object(self.interaction.conv_memory, "add_to_transcript"), \
                mock.patch.object(self.interaction.conv_log, "log_rex"), \
                mock.patch.object(self.interaction, "_register_rex_utterance"):
            fired = self.interaction._maybe_onboarding_question()
        self.assertTrue(fired)
        sp.assert_called_once()
        state = self.interaction._pending_onboarding
        self.assertEqual(state["step"], "awaiting_answer")
        self.assertEqual(state["asked_count"], 1)
        self.assertIsNotNone(state["pending_question"])


if __name__ == "__main__":
    unittest.main()
