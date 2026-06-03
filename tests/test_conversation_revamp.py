"""
Tests for the conversation revamp: make Rex a curious conversationalist instead
of a snark generator that deletes its own curiosity.

Covers (see the proposal in this change):
  C  - short shares (one-word interest answers) are topic seeds, not throwaways
  B  - earned on-thread follow-ups bypass the question budget
  A  - a question-only reply is never replaced with a dead "Fair enough."
  D  - sincere shares lead with engagement, roast optional
  E  - "I didn't say that" is not a wrong-person identity repair
  H  - topic threading uses the real topic, not keyword garbage
"""

from __future__ import annotations

import unittest
from unittest import mock


class InterestSeedClassificationTest(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def tearDown(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def test_profile_interest_keys_are_seeds_but_emotional_keys_are_not(self):
        from intelligence import conversation_steering as cs
        self.assertTrue(cs.is_interest_seed_question("obsession"))
        self.assertTrue(cs.is_interest_seed_question("hobbies"))
        self.assertTrue(cs.is_interest_seed_question("favorite_movie"))
        self.assertTrue(cs.is_interest_seed_question("travel"))
        # Emotional / biographical / music-offer keys are not interest seeds.
        self.assertFalse(cs.is_interest_seed_question("fears"))
        self.assertFalse(cs.is_interest_seed_question("proudest_moment"))
        self.assertFalse(cs.is_interest_seed_question("favorite_music"))
        self.assertFalse(cs.is_interest_seed_question(None))

    def test_one_word_passion_is_a_seed_but_a_refusal_is_not(self):
        from intelligence import conversation_steering as cs
        self.assertTrue(cs.looks_like_interest_seed_answer("astrophotography", "obsession"))
        self.assertTrue(
            cs.looks_like_interest_seed_answer("mostly nebulae and galaxies", "obsession")
        )
        self.assertFalse(cs.looks_like_interest_seed_answer("I don't know", "obsession"))
        self.assertFalse(cs.looks_like_interest_seed_answer("nope", "obsession"))
        self.assertFalse(cs.looks_like_interest_seed_answer("astrophotography", "fears"))

    def test_seed_from_answer_sets_the_active_topic(self):
        from intelligence import conversation_steering as cs
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
        ):
            ctx = cs.seed_from_answer(1, "astrophotography", "obsession")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.topic, "astrophotography")
        # The agenda reads the active topic on the next turn via build_context.
        again = cs.build_context(1)
        self.assertIsNotNone(again)
        self.assertEqual(again.topic, "astrophotography")

    def test_seed_from_answer_noop_for_non_seed_key(self):
        from intelligence import conversation_steering as cs
        ctx = cs.seed_from_answer(1, "my late father", "proudest_moment")
        self.assertIsNone(ctx)
        self.assertIsNone(cs.build_context(1))


class InterestSeedLengthTest(unittest.TestCase):
    def test_one_word_interest_answer_gets_room_not_micro(self):
        from intelligence import response_length
        plan = response_length.classify(
            "astrophotography",
            answered_question={"question_key": "obsession"},
        )
        # The regression: this used to be micro/12/1 with no question allowed.
        self.assertNotEqual(plan.target, "micro")
        self.assertGreaterEqual(plan.max_words, 40)
        self.assertGreaterEqual(plan.max_sentences, 2)
        self.assertIn("interest", plan.reason)
        self.assertIn("follow-up", plan.instruction.lower())

    def test_short_emotional_answer_stays_micro(self):
        from intelligence import response_length
        plan = response_length.classify(
            "my dog",
            answered_question={"question_key": "relationships"},
        )
        # Non-interest-seed short answers keep their tight acknowledgement budget.
        self.assertEqual(plan.target, "micro")


class InterestSeedEnergyTest(unittest.TestCase):
    def setUp(self):
        from intelligence import user_energy
        user_energy.clear()

    def tearDown(self):
        from intelligence import user_energy
        user_energy.clear()

    def test_one_word_interest_answer_is_engaged_not_quiet(self):
        from intelligence import user_energy
        profile = user_energy.note_user_turn(
            "astrophotography",
            1,
            answered_question={"question_key": "obsession"},
        )
        self.assertNotEqual(profile["mode"], "quiet")
        self.assertEqual(profile["engagement"], "engaged")
        self.assertNotEqual(profile["question_appetite"], "low")

    def test_one_word_non_answer_is_still_quiet(self):
        from intelligence import user_energy
        profile = user_energy.note_user_turn("sure", 1)
        self.assertEqual(profile["mode"], "quiet")


class EarnedFollowupBypassesBudgetTest(unittest.TestCase):
    """B: an on-thread follow-up to what the human just shared is not rationed by
    the question budget (which exists to stop NEW-topic interview pivots)."""

    def _build(self, directive, answered_question=None, user_text="astrophotography"):
        from intelligence import social_frame
        with (
            mock.patch("intelligence.question_budget.can_ask", return_value=False),
            mock.patch.object(
                social_frame.world_state, "snapshot", return_value={"people": []}
            ),
        ):
            return social_frame.build_frame(
                user_text,
                person_id=1,
                answered_question=answered_question,
                agenda_directive=directive,
            )

    def test_interest_thread_followup_allowed_when_budget_spent(self):
        directive = (
            "Conversation steering: The current thread matches a known/active "
            "interest: 'astrophotography'. Sound curious about their skill.\n"
            "Primary purpose: deepen the interest thread the human opened. Give "
            "one specific subject-aware reaction or tidbit, then ask one natural "
            "follow-up about their experience with that topic."
        )
        frame = self._build(directive, answered_question={"question_key": "obsession"})
        self.assertTrue(frame.allow_question)

    def test_answered_question_followup_allowed_when_budget_spent(self):
        directive = (
            "Primary purpose: the human just answered a question Rex asked. "
            "React to the actual content with genuine, specific interest. After "
            "answering, ask at most one short follow-up that stays on this exact "
            "topic. Do not pivot into a new interview topic."
        )
        frame = self._build(
            directive,
            answered_question={"question_key": "hobbies", "answer_text": "woodworking"},
            user_text="woodworking",
        )
        self.assertTrue(frame.allow_question)

    def test_generic_budget_text_still_does_not_invite_a_question(self):
        # Regression guard: a generic "ask at most one" budget line must NOT
        # bypass the budget — only earned interest/answer/identity follow-ups do.
        directive = (
            "Primary purpose: react to the human's latest thought. At most one "
            "tightly related follow-up question. Ask at most one, and only if it "
            "naturally serves this turn."
        )
        frame = self._build(directive, user_text="I'm from Waterford")
        self.assertFalse(frame.allow_question)


class NoDeadAckFallbackTest(unittest.TestCase):
    """A: a question-only reply is kept (curiosity > dead ack); the fallback is
    never the dismissive "Fair enough." that killed the astrophotography thread."""

    def _frame(self, purpose, allow_roast="normal"):
        from intelligence import social_frame
        return social_frame.SocialFrame(
            addressee="Bret",
            purpose=purpose,
            max_words=40,
            max_sentences=2,
            allow_question=False,
            allow_roast=allow_roast,
            allow_visual_comment=True,
            reason="test",
        )

    def test_pure_question_is_kept_not_dead_acked(self):
        from intelligence import social_frame
        governed = social_frame.govern_response(
            "What have you actually pointed that telescope at lately?",
            self._frame("quiet"),
        )
        self.assertEqual(
            governed.text,
            "What have you actually pointed that telescope at lately?",
        )
        self.assertNotEqual(governed.text, "Fair enough.")
        self.assertIn("kept_question_over_dead_ack", governed.notes)

    def test_closure_question_lands_and_stops(self):
        from intelligence import social_frame
        governed = social_frame.govern_response(
            "Heading out already?",
            self._frame("closure"),
        )
        self.assertNotIn("?", governed.text)
        self.assertNotEqual(governed.text, "Fair enough.")

    def test_statement_plus_question_still_drops_the_question(self):
        # Salvage must NOT fire when there is a real statement to keep — the
        # project deliberately drops unearned questions in that case.
        from intelligence import social_frame
        governed = social_frame.govern_response(
            "Classic deep-sky obsession. So where are you from?",
            self._frame("quiet"),
        )
        self.assertIn("Classic deep-sky obsession.", governed.text)
        self.assertNotIn("where are you from", governed.text.lower())
        self.assertIn("removed_question", governed.notes)
        self.assertNotIn("kept_question_over_dead_ack", governed.notes)

    def test_fallback_is_never_fair_enough(self):
        from intelligence import social_frame
        for purpose in ("quiet", "neutral", "banter", "interest"):
            self.assertNotEqual(
                social_frame._fallback(self._frame(purpose)),
                "Fair enough.",
            )


class EngageFirstOnSincereSharesTest(unittest.TestCase):
    """D: when the human shares an interest or answers a real question, Rex
    engages first; the roast rides on top. Rex answering the user, and plain
    banter/general turns, keep the roast-first default."""

    def _directive(self, purpose, allow_roast="normal"):
        from intelligence import social_frame
        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose=purpose,
            max_words=50,
            max_sentences=3,
            allow_question=True,
            allow_roast=allow_roast,
            allow_visual_comment=True,
            reason="test",
        )
        return social_frame.build_directive(frame)

    def test_interest_turn_is_engage_first(self):
        directive = self._directive("interest")
        self.assertIn("ENGAGE-FIRST", directive)
        self.assertNotIn("ROAST-FIRST", directive)

    def test_answer_ack_turn_is_engage_first(self):
        self.assertIn("ENGAGE-FIRST", self._directive("answer_ack"))

    def test_general_banter_turn_stays_roast_first(self):
        self.assertIn("ROAST-FIRST", self._directive("banter"))

    def test_rex_answering_user_stays_roast_first(self):
        # purpose="answer" = Rex answering the user's question, not a user share.
        self.assertIn("ROAST-FIRST", self._directive("answer"))


class WrongPersonRepairNarrowingTest(unittest.TestCase):
    """E: "I didn't say that" is a content disagreement, not an identity repair."""

    def setUp(self):
        from intelligence import repair_moves
        repair_moves.clear()

    def tearDown(self):
        from intelligence import repair_moves
        repair_moves.clear()

    def test_bare_disagreement_is_not_a_repair(self):
        from intelligence import repair_moves
        self.assertIsNone(repair_moves.detect("No, I didn't say that"))
        self.assertIsNone(repair_moves.detect("I didn't say that."))
        self.assertIsNone(repair_moves.detect("I did not say that"))

    def test_explicit_identity_cue_still_repairs_as_wrong_person(self):
        from intelligence import repair_moves
        self.assertEqual(repair_moves.detect("wrong person")["kind"], "wrong_person")
        self.assertEqual(
            repair_moves.detect("That was Jane, not me")["kind"], "wrong_person"
        )

    def test_real_correction_still_handled(self):
        from intelligence import repair_moves
        # A denial that carries an actual correction is still a repair (misheard),
        # just not a bare brush-off.
        move = repair_moves.detect("I didn't say jazz, I said blues")
        self.assertIsNotNone(move)
        self.assertNotEqual(move["kind"], "wrong_person")


class CelebrationColdOpenGateTest(unittest.TestCase):
    """F: Rex doesn't open a greeting with a vague/inferred/stale 'good news'
    memory; only a concrete, recent, or self-reported milestone leads."""

    @staticmethod
    def _event(description, *, age_days=0.0, invited=False):
        from datetime import datetime, timedelta
        when = datetime.utcnow() - timedelta(days=age_days)
        return {
            "description": description,
            "mentioned_at": when.strftime("%Y-%m-%d %H:%M:%S"),
            "person_invited_topic": 1 if invited else 0,
        }

    def test_vague_affect_does_not_lead_a_greeting(self):
        from intelligence import consciousness
        ev = self._event("the speaker feels proud of their problem-solving skills")
        self.assertFalse(consciousness._celebration_worth_leading_with(ev))

    def test_concrete_recent_milestone_leads(self):
        from intelligence import consciousness
        ev = self._event("won the regional volleyball championship", age_days=1.0)
        self.assertTrue(consciousness._celebration_worth_leading_with(ev))

    def test_concrete_but_stale_and_uninvited_does_not_lead(self):
        from intelligence import consciousness
        ev = self._event("got the new job at the observatory", age_days=120.0)
        self.assertFalse(consciousness._celebration_worth_leading_with(ev))

    def test_person_reported_milestone_leads_even_if_older(self):
        from intelligence import consciousness
        ev = self._event(
            "got the new job at the observatory", age_days=120.0, invited=True
        )
        self.assertTrue(consciousness._celebration_worth_leading_with(ev))

    def test_kill_switch_restores_old_behavior(self):
        from unittest import mock
        from intelligence import consciousness
        ev = self._event("feels good about things")
        with mock.patch.object(
            consciousness.config, "PRESENCE_CELEBRATION_REQUIRE_CONCRETE", False
        ):
            self.assertTrue(consciousness._celebration_worth_leading_with(ev))


class TopicThreadLabelTest(unittest.TestCase):
    """H: the topic label is the real topic, not keyword garbage, and an answer
    to Rex's question retitles the thread instead of staying stuck."""

    def setUp(self):
        from intelligence import topic_thread
        topic_thread.clear()

    def tearDown(self):
        from intelligence import topic_thread
        topic_thread.clear()

    def test_plain_filler_does_not_become_a_topic_label(self):
        from intelligence import topic_thread
        topic_thread.note_user_turn("things are going well", 1)
        self.assertEqual(topic_thread.snapshot()["label"], "current exchange")

    def test_answer_retitles_the_thread(self):
        from intelligence import topic_thread
        topic_thread.note_user_turn("things are going well", 1)
        topic_thread.note_assistant_turn("What are you completely obsessed with?")
        topic_thread.note_user_turn(
            "astrophotography", 1, answered_question={"question_key": "obsession"}
        )
        self.assertEqual(topic_thread.snapshot()["label"], "astrophotography")


class CuriosityFollowupRotationTest(unittest.TestCase):
    """G: Rex walks down a stack of follow-up angles as a topic continues, so he
    keeps getting more curious instead of re-asking the same opener."""

    def setUp(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def tearDown(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def _angle(self, ctx):
        import re
        m = re.search(r"aim it at (.*?), and do not", ctx.directive)
        return m.group(1) if m else None

    def test_followup_angle_advances_across_turns(self):
        from intelligence import conversation_steering as cs
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
        ):
            first = cs.note_user_turn(1, "I'm really into astrophotography")
            second = cs.note_user_turn(1, "it is genuinely so relaxing for me")
            third = cs.note_user_turn(1, "i shoot from the backyard most nights")
        angles = [self._angle(first), self._angle(second), self._angle(third)]
        self.assertEqual(angles[0], "what first got them into it")
        # Each continuation turn digs at a different angle.
        self.assertEqual(len(set(angles)), 3)


class ComedyAlignsWithInterestTest(unittest.TestCase):
    """I: on an interest turn, comedy complements curiosity (tease the hobby)
    instead of derailing into a self-own / fake-system-error bit."""

    def _frame(self, purpose):
        from intelligence import social_frame
        return social_frame.SocialFrame(
            addressee="Bret",
            purpose=purpose,
            max_words=50,
            max_sentences=3,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=True,
            reason="test",
        )

    def test_interest_turn_uses_curiosity_friendly_comedy(self):
        from intelligence import comedy_modes
        comedy_modes.reset_recent_state()
        seen = set()
        for _ in range(40):
            mode = comedy_modes.select_mode(
                "astrophotography",
                1,
                frame=self._frame("interest"),
                agenda_directive="Conversation steering: known/active interest.",
            )
            seen.add(mode.key)
        # Never the self-absorbed / glitch bits that ignore the human's topic.
        self.assertNotIn("self_own", seen)
        self.assertNotIn("fake_system_error", seen)


class AstrophotographyTurnEndToEndTest(unittest.TestCase):
    """The whole point: the exact turn that produced "Bet you wish you could beam
    me up for those sweet cosmic selfies" (a 12-word brush-off with no curiosity,
    on a SPENT question budget) now produces an engaged, curious, question-bearing
    reply. Guards that A+B+C+D+G+H actually cooperate, not just pass in isolation."""

    def test_one_word_passion_answer_drives_engaged_curiosity(self):
        from intelligence import (
            conversation_steering as cs,
            conversation_agenda,
            social_frame,
            response_length,
            user_energy,
        )

        cs.clear()
        user_energy.clear()
        answered = {
            "question_key": "obsession",
            "question_text": "What are you completely obsessed with right now?",
            "answer_text": "astrophotography",
        }
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
            # The budget is SPENT — the failure condition from the live log.
            mock.patch("intelligence.question_budget.can_ask", return_value=False),
            mock.patch.object(
                conversation_agenda.world_state,
                "snapshot",
                return_value={"people": [], "environment": {}},
            ),
            mock.patch.object(
                social_frame.world_state, "snapshot", return_value={"people": []}
            ),
            mock.patch.object(
                conversation_agenda.rel_memory,
                "get_latest_pending_question",
                return_value=None,
            ),
            mock.patch.object(
                conversation_agenda.empathy,
                "classify_local_sensitivity",
                return_value=None,
            ),
            mock.patch.object(conversation_agenda.empathy, "peek", return_value={}),
        ):
            cs.seed_from_answer(1, "astrophotography", "obsession")
            profile = user_energy.note_user_turn(
                "astrophotography", 1, answered_question=answered
            )
            directive = conversation_agenda.build_turn_directive(
                "astrophotography", 1, answered_question=answered
            )
            frame = social_frame.build_frame(
                "astrophotography",
                person_id=1,
                answered_question=answered,
                agenda_directive=directive,
            )
            contract = social_frame.build_directive(frame)
            plan = response_length.classify("astrophotography", answered_question=answered)

        self.assertEqual(profile["engagement"], "engaged")        # C: not "quiet"
        self.assertNotEqual(plan.target, "micro")                 # C: room to breathe
        self.assertIn("conversation steering", directive.lower())  # G: deepen the interest
        self.assertEqual(frame.purpose, "interest")
        self.assertTrue(frame.allow_question)                     # A+B: curiosity survives a spent budget
        self.assertIn("ENGAGE-FIRST", contract)                   # D: curiosity leads, roast rides on top
        cs.clear()
        user_energy.clear()


if __name__ == "__main__":
    unittest.main()
