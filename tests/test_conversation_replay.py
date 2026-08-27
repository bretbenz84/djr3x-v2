"""
Offline conversational-quality replay harness.

The only conversational eval used to be "run the robot and read the logs" — which is
how the "A solo project" repeat shipped. This replays scenarios through the
DETERMINISTIC conversation stack (conversation_agenda.build_turn_plan →
social_frame.build_frame → social_frame.govern_response) and asserts STRUCTURAL
properties on the result. No LLM is called (the creative reply is the only
non-deterministic part; we test the routing/governance decisions around it), so it
runs in the suite with zero network/cost — the whole point being to validate
conversational behavior WITHOUT the robot.

Add a regression by dropping a scenario into tests/fixtures/conversation_replays.json
(no Python needed). Each scenario:
  - context (optional): rex_last_line, answered_question, steering_seed {answer,key},
    prior_user_turns, question_budget_allows
  - utterance: the user's turn
  - candidate (optional): a Rex reply to run through govern_response
  - expect: any of purpose / allow_question / allow_roast / directive_includes[] /
    governed_includes[] / governed_excludes[] / governed_max_questions /
    governed_notes_include[]
"""

from __future__ import annotations

import contextlib
import json
import os
import unittest
from unittest import mock

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "conversation_replays.json")


@contextlib.contextmanager
def _pipeline_mocks(can_ask=True):
    """Neutralize external state (world/memory/budget/empathy) so the stack is
    deterministic and makes no DB/network calls."""
    from intelligence import conversation_steering as cs, conversation_agenda as ca, social_frame as sf
    with contextlib.ExitStack() as stack:
        e = stack.enter_context
        e(mock.patch.object(cs.boundary_memory, "is_blocked", return_value=False))
        e(mock.patch.object(cs.facts_memory, "add_fact"))
        e(mock.patch.object(cs.facts_memory, "get_facts", return_value=[]))
        e(mock.patch.object(cs.interests_memory, "upsert_interest"))
        e(mock.patch("intelligence.question_budget.can_ask", return_value=can_ask))
        e(mock.patch.object(ca.world_state, "snapshot", return_value={"people": [], "environment": {}}))
        e(mock.patch.object(ca.rel_memory, "get_latest_pending_question", return_value=None))
        e(mock.patch.object(ca.empathy, "classify_local_sensitivity", return_value=None))
        e(mock.patch.object(ca.empathy, "peek", return_value={}))
        e(mock.patch.object(sf.boundary_memory, "is_blocked", return_value=False))
        e(mock.patch.object(sf.facts_memory, "get_facts_by_category", return_value=[]))
        yield


def _reset_state():
    from intelligence import (
        conversation_steering, user_energy, topic_thread, comedy_modes, repair_moves, end_thread,
    )
    for module in (conversation_steering, user_energy, topic_thread, repair_moves, end_thread):
        try:
            module.clear()
        except Exception:
            pass
    try:
        comedy_modes.reset_recent_state()
    except Exception:
        pass


def run_scenario(scenario: dict) -> dict:
    """Run one scenario through the deterministic stack; return the observables."""
    from intelligence import (
        conversation_agenda as ca, social_frame as sf, conversation_steering as cs,
        comedy_modes, repair_moves, topic_thread, end_thread,
    )
    _reset_state()
    person_id = scenario.get("person_id")
    answered = scenario.get("answered_question")
    can_ask = bool(scenario.get("question_budget_allows", True))

    rex_last = scenario.get("rex_last_line")
    if rex_last:
        for fn in (comedy_modes.note_spoken_line, repair_moves.note_assistant_turn,
                   topic_thread.note_assistant_turn, end_thread.note_assistant_turn):
            try:
                fn(rex_last)
            except Exception:
                pass

    utterance = scenario["utterance"]
    with _pipeline_mocks(can_ask=can_ask):
        seed = scenario.get("steering_seed")
        if seed:
            try:
                cs.seed_from_answer(person_id, seed["answer"], seed["key"])
            except Exception:
                pass
        for prior in scenario.get("prior_user_turns", []):
            try:
                cs.note_user_turn(person_id, prior)
            except Exception:
                pass
        try:
            # The closure / invitation-acceptance decision is made on the user turn,
            # BEFORE the agenda builds — replay it in the same order the live path does.
            end_thread.note_user_turn(utterance, person_id, answered_question=answered)
        except Exception:
            pass
        plan = ca.build_turn_plan(utterance, person_id, answered_question=answered)
        frame = sf.build_frame(
            utterance, person_id, answered_question=answered,
            agenda_directive=plan.directive, turn_plan=plan,
        )
        out = {
            "purpose": frame.purpose,
            "allow_question": frame.allow_question,
            "allow_roast": frame.allow_roast,
            "max_words": frame.max_words,
            "max_sentences": frame.max_sentences,
            "directive": plan.directive,
            "governed_text": None,
            "governed_notes": [],
        }
        candidate = scenario.get("candidate")
        if candidate is not None:
            governed = sf.govern_response(candidate, frame)
            out["governed_text"] = governed.text
            out["governed_notes"] = list(governed.notes)
    return out


def assert_scenario(tc: unittest.TestCase, scenario: dict) -> dict:
    out = run_scenario(scenario)
    expect = scenario.get("expect", {})
    name = scenario.get("name", scenario.get("utterance", "?"))
    if "purpose" in expect:
        tc.assertEqual(out["purpose"], expect["purpose"], f"[{name}] purpose")
    if "allow_question" in expect:
        tc.assertEqual(out["allow_question"], expect["allow_question"], f"[{name}] allow_question")
    if "allow_roast" in expect:
        tc.assertEqual(out["allow_roast"], expect["allow_roast"], f"[{name}] allow_roast")
    directive = (out["directive"] or "").lower()
    for sub in expect.get("directive_includes", []):
        tc.assertIn(sub.lower(), directive, f"[{name}] directive_includes {sub!r}")
    governed = (out["governed_text"] or "").lower()
    for sub in expect.get("governed_includes", []):
        tc.assertIn(sub.lower(), governed, f"[{name}] governed_includes {sub!r}")
    for sub in expect.get("governed_excludes", []):
        tc.assertNotIn(sub.lower(), governed, f"[{name}] governed_excludes {sub!r}")
    if "governed_max_questions" in expect:
        tc.assertLessEqual(
            (out["governed_text"] or "").count("?"), expect["governed_max_questions"],
            f"[{name}] governed_max_questions",
        )
    for note in expect.get("governed_notes_include", []):
        tc.assertIn(note, out["governed_notes"], f"[{name}] governed_notes_include {note!r}")
    return out


class ConversationReplayCorpusTest(unittest.TestCase):
    def test_corpus_scenarios(self):
        with open(_FIXTURE, encoding="utf-8") as f:
            corpus = json.load(f)
        self.assertTrue(corpus, "replay corpus is empty")
        for scenario in corpus:
            with self.subTest(scenario=scenario.get("name")):
                assert_scenario(self, scenario)


class HarnessSelfTest(unittest.TestCase):
    """The engine itself runs and yields the observables (guards the harness)."""

    def test_engine_returns_observables_for_a_plain_turn(self):
        out = run_scenario({"name": "smoke", "person_id": 1, "utterance": "I had a long day"})
        for key in ("purpose", "allow_question", "allow_roast", "directive"):
            self.assertIn(key, out)
        self.assertIsInstance(out["allow_question"], bool)

    def test_candidate_is_governed_when_present(self):
        out = run_scenario({
            "person_id": 1, "utterance": "I'm from Waterford", "question_budget_allows": False,
            "candidate": "Neat. So what's your favorite color?",
        })
        self.assertIsNotNone(out["governed_text"])


class InvitationAcceptanceLifetimeTest(unittest.TestCase):
    """An acceptance belongs to the turn that armed it.

    Review 2026-08-27: many turns return before build_turn_plan ever consumes the
    flag (the face-reveal ask, the off-camera identify ask, repair/game/router
    acks), so an unread acceptance survived into the NEXT turn and outranked a
    genuine closure — Rex would settle in exactly when told to wrap up.
    """

    INVITATION = "Hey Bret, I'm thinking about you. Want to sit with me a minute?"

    def setUp(self):
        from intelligence import end_thread
        self.et = end_thread
        self.et.clear()
        self.addCleanup(self.et.clear)

    def _accept(self):
        self.et.note_assistant_turn(self.INVITATION)
        self.et.note_user_turn(
            "Yeah.",
            answered_question={"question_text": self.INVITATION, "answer_text": "Yeah."},
        )

    def test_the_accepting_turn_can_still_consume_it(self):
        self._accept()
        self.assertTrue(self.et.consume_invitation_acceptance())

    def test_an_unconsumed_acceptance_does_not_survive_the_next_turn(self):
        self._accept()
        self.et.note_assistant_turn("Cool.")
        self.et.note_user_turn("Alright, I gotta go.", answered_question=None)
        self.assertFalse(self.et.consume_invitation_acceptance())

    def test_a_real_goodbye_still_closes_after_an_unconsumed_acceptance(self):
        self._accept()
        self.et.note_assistant_turn("Cool.")
        state = self.et.note_user_turn("Alright, I gotta go.", answered_question=None)
        self.assertTrue((state or {}).get("closing_pending"))


if __name__ == "__main__":
    unittest.main()
