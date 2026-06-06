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
        self.assertNotIn("ROAST-LEAN", directive)

    def test_answer_ack_turn_is_engage_first(self):
        self.assertIn("ENGAGE-FIRST", self._directive("answer_ack"))

    def test_general_banter_turn_stays_roast_lean(self):
        directive = self._directive("banter")
        self.assertIn("ROAST-LEAN", directive)
        self.assertNotIn("ENGAGE-FIRST", directive)

    def test_rex_answering_user_stays_roast_lean(self):
        # purpose="answer" = Rex answering the user's question, not a user share.
        self.assertIn("ROAST-LEAN", self._directive("answer"))


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


class CuriosityFollowupVarietyTest(unittest.TestCase):
    """G (post arc-cleanup): the deterministic angle ROTATION (_FOLLOWUP_ANGLES) was
    removed — the conversation arc now lets Rex see what he already asked. The steering
    directive still STEERS toward a fresh follow-up angle and tells him not to re-ask a
    covered one; the model + arc choose the angle instead of a hardcoded rotation."""

    def setUp(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def tearDown(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def test_steering_directive_asks_for_a_fresh_unrepeated_angle(self):
        from intelligence import conversation_steering as cs
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
        ):
            ctx = cs.note_user_turn(1, "I'm really into astrophotography")
        self.assertIsNotNone(ctx)
        directive = ctx.directive.lower()
        self.assertIn("fresh angle", directive)
        self.assertIn("do not re-ask an angle you've already covered", directive)
        # The deterministic rotation phrasing is gone.
        self.assertNotIn("aim it at", directive)


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


class TurnTakingTest(unittest.TestCase):
    """A1/A2: hold unfinished thoughts; don't treat rhetorical reformulations as
    questions that arm the no-response quip."""

    def test_dangling_preposition_is_held(self):
        from intelligence import turn_completion
        # The live failure: Rex interrupted "well we're currently in" (a pause).
        self.assertIsNotNone(turn_completion.classify("well we're currently in"))
        self.assertIsNotNone(turn_completion.classify("it depends on what kind of"))

    def test_complete_preposition_questions_not_held(self):
        from intelligence import turn_completion
        for complete in (
            "what are you into",
            "what's it based on",
            "where are you from",
            "the photos are the documentation",
        ):
            self.assertIsNone(
                turn_completion.classify(complete), f"falsely held: {complete!r}"
            )

    def test_rhetorical_reformulation_does_not_expect_a_response(self):
        from intelligence import interaction
        # "So what you're saying is …?" restates the human's point — it must not
        # arm the no-response quip ("No answer. Bold strategy.").
        self.assertFalse(
            interaction._question_expects_response(
                "So what you're saying is, you capture blurry disappointments?"
            )
        )
        self.assertFalse(
            interaction._question_expects_response("So you're telling me that worked?")
        )
        # A genuine question still expects an answer.
        self.assertTrue(
            interaction._question_expects_response("What galaxy are you shooting next?")
        )

    def test_no_response_quips_are_gentle_not_accusatory(self):
        import config
        text = " ".join(config.CONVERSATION_NO_RESPONSE_QUIPS).lower()
        self.assertNotIn("bold strategy", text)
        self.assertNotIn("rude", text)


class TimeQueryAndRepairRoutingTest(unittest.TestCase):
    """B1/B2: 'give me time to answer' is not a clock query, and a router-judged
    repair never falls through to a keyword data intent ('It's 8:33 PM.')."""

    def test_pacing_complaint_is_not_a_time_query(self):
        from intelligence import intent_classifier as ic
        self.assertNotEqual(
            ic.classify_deterministic("You didn't give me any time to answer"),
            "query_time",
        )
        self.assertNotEqual(
            ic.classify_deterministic("give me time to think"), "query_time"
        )
        # Real clock queries still classify.
        self.assertEqual(ic.classify_deterministic("what time is it"), "query_time")

    def test_router_repair_blocks_deterministic_data_intent(self):
        from intelligence import interaction
        reason = interaction._intent_execution_block_reason(
            "query_time",
            text="You didn't give me any time to answer",
            router_action="conversation.repair",
        )
        self.assertEqual(reason, "router_classified_repair")

    def test_interruption_complaint_is_a_repair(self):
        from intelligence import repair_moves
        repair_moves.clear()
        repair_moves.note_assistant_turn("So, which galaxy next?")
        for t in (
            "You didn't give me any time to answer",
            "you cut me off",
            "you didn't let me finish",
        ):
            self.assertEqual(repair_moves.detect(t)["kind"], "interruption")
        repair_moves.clear()


class OpenerVarietyTest(unittest.TestCase):
    """C2: strip stock filler openers and remember openers to vary the next one."""

    def setUp(self):
        from intelligence import comedy_modes
        comedy_modes.reset_recent_state()

    def tearDown(self):
        from intelligence import comedy_modes
        comedy_modes.reset_recent_state()

    def test_banned_openers_are_stripped(self):
        from intelligence import comedy_modes as cm
        self.assertEqual(cm.strip_banned_opener("Ah, chasing galaxies"), "Chasing galaxies")
        self.assertEqual(cm.strip_banned_opener("Oh, so now you do"), "So now you do")
        self.assertEqual(
            cm.strip_banned_opener("Well, well, well, look who's here"),
            "Look who's here",
        )
        # A single "Well," is left alone.
        self.assertEqual(cm.strip_banned_opener("Well, that's fair"), "Well, that's fair")

    def test_polish_strips_banned_opener_on_roast_turns(self):
        from intelligence import comedy_modes as cm
        out = cm.polish_response("Ah, the Whirlpool Galaxy again", cm._MODES["friendly_roast"])
        self.assertFalse(out.lower().startswith("ah"))

    def test_recent_openers_are_tracked_for_variety(self):
        from intelligence import comedy_modes as cm
        cm.note_spoken_line("Glad to hear you're better")
        cm.note_spoken_line("Glad you found that funny")
        self.assertIn("glad", cm.recent_openers_to_avoid())
        directive = cm.build_directive(cm._MODES["friendly_roast"])
        self.assertIn("Opening variety", directive)


class NoRepeatQuestionTest(unittest.TestCase):
    """P1a: Rex doesn't ask the same question twice over the user's answer."""

    def setUp(self):
        from intelligence import comedy_modes
        comedy_modes.reset_recent_state()

    def tearDown(self):
        from intelligence import comedy_modes
        comedy_modes.reset_recent_state()

    def _frame(self):
        from intelligence import social_frame
        return social_frame.SocialFrame(
            addressee="Bret", purpose="interest", max_words=36, max_sentences=2,
            allow_question=False, allow_roast="normal", allow_visual_comment=True,
            reason="test",
        )

    def test_salvage_skips_repeat_and_keeps_a_new_question(self):
        from intelligence import social_frame, comedy_modes
        comedy_modes.note_spoken_line("A solo project, huh?")
        governed = social_frame.govern_response(
            "A solo project, huh? Do you expect glory or disaster?", self._frame()
        )
        self.assertNotEqual(governed.text, "A solo project, huh?")
        self.assertIn("glory or disaster", governed.text)

    def test_pure_repeat_is_not_spoken(self):
        from intelligence import social_frame, comedy_modes
        comedy_modes.note_spoken_line("A solo project, huh?")
        governed = social_frame.govern_response("A solo project, huh?", self._frame())
        self.assertNotIn("solo project", governed.text.lower())


class SubtitleHallucinationTest(unittest.TestCase):
    """P1b: Whisper subtitle/credit hallucinations are filtered."""

    def test_subtitle_credits_are_hallucinations(self):
        from audio import transcription
        for junk in (
            "Subs by www.zeoranger.co.uk",
            "Subtitles by the Amara.org community",
            "Thanks for watching!",
            "like and subscribe",
        ):
            self.assertTrue(transcription._is_hallucination(junk), junk)

    def test_real_speech_survives(self):
        from audio import transcription
        for real in (
            "I like watching Apple TV Plus shows",
            "I'm building the R3X droid",
            "I need to subscribe to Netflix",
        ):
            self.assertFalse(transcription._is_hallucination(real), real)


class StreamTailTruncationTest(unittest.TestCase):
    """P2a: an unpunctuated stream remainder (mid-sentence cut) is dropped."""

    def test_incomplete_tail_dropped_finished_tail_spoken(self):
        from intelligence import interaction
        self.assertFalse(interaction._tail_is_speakable("What's the deal"))
        self.assertFalse(interaction._tail_is_speakable("Glad to"))
        self.assertTrue(interaction._tail_is_speakable("What galaxy next?"))
        self.assertTrue(interaction._tail_is_speakable("You got this."))


class StaleSteeringAndReassuranceTest(unittest.TestCase):
    """P3a/P3b: active interest updates to a new build/make topic; a reassurance
    is taken at face value, not roasted."""

    def setUp(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def tearDown(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def test_building_something_becomes_the_active_interest(self):
        from intelligence import conversation_steering as cs
        self.assertEqual(cs.detect_interest("I'm building the R3X droid"), "R3X droid")
        self.assertEqual(cs.detect_interest("I am building the R3X droid"), "R3X droid")
        # "trying to make" is not a clean interest declaration.
        self.assertIsNone(cs.detect_interest("I'm trying to make him funny"))

    def test_reassurance_directive_forbids_the_needle(self):
        from unittest import mock
        from intelligence import conversation_agenda as ca
        with (
            mock.patch.object(
                ca.world_state, "snapshot",
                return_value={"people": [], "environment": {}},
            ),
            mock.patch.object(
                ca.empathy, "classify_local_sensitivity", return_value=None
            ),
        ):
            directive = ca.build_turn_directive("I'm not sad, it's okay", None)
        self.assertIn("do not roast or needle", directive.lower())
        self.assertIn("repressing", directive.lower())


class SubjectPivotTest(unittest.TestCase):
    """When a subject stops engaging the user, Rex pivots to a related/new one
    instead of probing a dead topic."""

    def setUp(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def tearDown(self):
        from intelligence import conversation_steering
        conversation_steering.clear()

    def _steer(self, *turns):
        from intelligence import conversation_steering as cs
        ctx = None
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
        ):
            for t in turns:
                ctx = cs.note_user_turn(1, t)
        return ctx

    def test_disengagement_triggers_a_pivot(self):
        from intelligence import conversation_steering as cs
        self.assertEqual(self._steer("I love astrophotography").mode, "deepen")
        # one flat reply is tolerated...
        ctx1 = self._steer("I love astrophotography", "yeah")
        self.assertEqual(ctx1.mode, "deepen")
        # ...two in a row and Rex pivots, dropping the dead topic.
        ctx2 = self._steer("I love astrophotography", "yeah", "I guess")
        self.assertEqual(ctx2.mode, "pivot")
        self.assertIn("stopped landing", ctx2.directive.lower())
        self.assertIsNone(cs.build_context(1))  # topic dropped

    def test_substantive_short_answer_is_not_disengagement(self):
        # A real short answer keeps deepening; only bare acks count as flat.
        ctx = self._steer(
            "I love astrophotography", "yeah", "mostly nebulae and the Whirlpool"
        )
        self.assertEqual(ctx.mode, "deepen")

    def test_pivot_directive_steers_to_related_or_new(self):
        from intelligence import conversation_steering as cs
        ctx = self._steer("I love astrophotography", "sure", "not really")
        self.assertEqual(ctx.mode, "pivot")
        low = ctx.directive.lower()
        self.assertTrue("related subject" in low or "new topic" in low)
        self.assertIn("do not keep", low)

    def test_pivot_turn_offers_a_fresh_question_and_allows_it(self):
        from intelligence import (
            conversation_steering as cs,
            conversation_agenda as ca,
            social_frame as sf,
        )
        cs.clear()
        with (
            mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(cs.facts_memory, "add_fact"),
            mock.patch.object(cs.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(cs.interests_memory, "upsert_interest"),
            mock.patch.object(
                ca.world_state, "snapshot",
                return_value={"people": [], "environment": {}},
            ),
            mock.patch.object(sf.world_state, "snapshot", return_value={"people": []}),
            mock.patch.object(ca.empathy, "classify_local_sensitivity", return_value=None),
            mock.patch.object(ca.empathy, "peek", return_value={}),
            mock.patch.object(ca.rel_memory, "get_latest_pending_question", return_value=None),
            mock.patch.object(
                ca, "_next_useful_question",
                return_value={"text": "What kind of music are you into?"},
            ),
            # Even with the budget spent, a pivot question is allowed (it's the
            # re-engagement move, not interview spam).
            mock.patch("intelligence.question_budget.can_ask", return_value=False),
        ):
            cs.note_user_turn(1, "I love astrophotography")
            cs.note_user_turn(1, "yeah")
            directive = ca.build_turn_directive("I guess", 1)
            frame = sf.build_frame("I guess", person_id=1, agenda_directive=directive)
        self.assertIn("pivot", directive.lower())
        self.assertIn("what kind of music", directive.lower())
        self.assertEqual(frame.purpose, "interest")
        self.assertTrue(frame.allow_question)
        cs.clear()


if __name__ == "__main__":
    unittest.main()
