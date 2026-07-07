"""Tests for the role-reversed 20 Questions game (features/games.py + features/twentyq_kb.py).

Roles are reversed from classic 20Q: the PLAYER thinks of a person/place/thing and REX
guesses it by asking yes/no questions, grounded by the allenai/twentyquestions knowledge
base (a spine of proven discriminator questions + a real-subject vocabulary for the guess).

The LLM-facing helpers (_rex_respond, _quick_call) are mocked; everything tested here is
the deterministic turn machinery: answer parsing, spine narrowing, guess/win/lose flow,
forced guess at the cap, and the episodic outcome string.
"""

import json
import unittest
from contextlib import contextmanager
from unittest import mock

from features import games, twentyq_kb


@contextmanager
def _mocked_llm(llm_reply):
    """Patch every LLM boundary + body motion. `_rex_respond` echoes its directive so tests
    can inspect what Rex was told to say; `_quick_call` (cheap classifier) and `_smart_call`
    (question/guess reasoning) both return `llm_reply` so a single per-test value drives them."""
    with mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx), \
         mock.patch.object(games, "_body_beat", return_value=None), \
         mock.patch.object(games, "_quick_call", side_effect=llm_reply), \
         mock.patch.object(games, "_smart_call", side_effect=llm_reply):
        yield


def _reset_game():
    games._active_game = "20_questions"
    games._game_state = {}


# ── Answer parsing ──────────────────────────────────────────────────────────────

class ClassifyAnswerTest(unittest.TestCase):
    def _classify(self, text):
        # Keyword path should not need the LLM; make a call an explicit failure.
        with mock.patch.object(games, "_quick_call",
                               side_effect=AssertionError("should not hit LLM")):
            return games._20q_classify_answer(text)

    def test_yes_variants(self):
        for t in ("yes", "Yep!", "yeah", "correct", "you got it", "Definitely.", "totally yes"):
            self.assertEqual(self._classify(t), "yes", t)

    def test_no_variants(self):
        for t in ("no", "Nope.", "nah", "not really", "definitely not", "incorrect"):
            self.assertEqual(self._classify(t), "no", t)

    def test_sometimes_variants(self):
        for t in ("sometimes", "kind of", "sort of", "maybe", "occasionally"):
            self.assertEqual(self._classify(t), "sometimes", t)

    def test_unknown_variants(self):
        # "no idea" must resolve to unknown, NOT "no".
        for t in ("no idea", "I don't know", "not sure", "dunno", ""):
            self.assertEqual(self._classify(t), "unknown", t)

    def test_ambiguous_falls_back_to_llm(self):
        with mock.patch.object(games, "_quick_call", return_value="yes") as qc:
            self.assertEqual(games._20q_classify_answer("the third one from the left"), "yes")
            qc.assert_called_once()

    def test_now_mishear_corrects_to_no_without_llm(self):
        # Whisper hears the clipped "no" as "now"/"know" far-field — correct it deterministically
        # inside the game, with no LLM round-trip.
        with mock.patch.object(games, "_quick_call",
                               side_effect=AssertionError("should not hit LLM")):
            for t in ("now", "now.", "know", "gnaw"):
                self.assertEqual(games._20q_classify_answer(t), "no", t)

    def test_llm_unknown_verdict_is_not_swallowed_by_no_substring(self):
        # The LLM fallback returns the literal word "unknown"; "no" is a substring of it, so a
        # naive `if label in raw` would wrongly return "no". It must return "unknown".
        with mock.patch.object(games, "_quick_call", return_value="unknown"):
            self.assertEqual(games._20q_classify_answer("the one on the left, perhaps"), "unknown")
        with mock.patch.object(games, "_quick_call", return_value="no"):
            self.assertEqual(games._20q_classify_answer("the one on the left, perhaps"), "no")


# ── Knowledge base ──────────────────────────────────────────────────────────────

class KnowledgeBaseTest(unittest.TestCase):
    def test_kb_loaded(self):
        self.assertTrue(twentyq_kb.is_loaded())
        self.assertGreater(len(twentyq_kb.subjects()), 1000)

    def test_first_question_is_alive(self):
        self.assertEqual(twentyq_kb.next_spine_question({}, set())["question"], "is it alive?")

    def test_not_alive_defers_person_until_manmade_known(self):
        # alive=no goes straight to "is it man made?" — the old parentless "is it a person?"
        # at Q2 was a wasted question on every object game (live-logged 2026-07-07).
        nxt = twentyq_kb.next_spine_question({"alive": False}, {"is it alive?"})
        self.assertEqual(nxt["concept"], "manmade")

    def test_not_alive_not_manmade_still_checks_person(self):
        # Famous DEAD people / fictional characters (alive=no, man-made=no) are still
        # discovered instead of being chased as inanimate objects.
        nxt = twentyq_kb.next_spine_question(
            {"alive": False, "manmade": False}, {"is it alive?", "is it man made?"})
        self.assertEqual(nxt["concept"], "person")

    def test_manmade_object_never_asks_person(self):
        # alive=no + man-made=yes: "is it a person?" is permanently pruned.
        seen = self._walk_spine(
            {"alive": False, "manmade": True}, {"is it alive?", "is it man made?"})
        self.assertNotIn("person", seen)

    def test_not_alive_not_person_skips_to_object_branch(self):
        # Once it's confirmed not alive and not a person, stay in the object branch
        # and never ask the animal/plant questions.
        nxt = twentyq_kb.next_spine_question(
            {"alive": False, "person": False}, {"is it alive?", "is it a person?"})
        self.assertEqual(nxt["concept"], "manmade")

    def test_alive_opens_person_branch(self):
        nxt = twentyq_kb.next_spine_question({"alive": True}, {"is it alive?"})
        self.assertEqual(nxt["concept"], "person")

    def test_person_prunes_animal_and_plant(self):
        nxt = twentyq_kb.next_spine_question(
            {"alive": True, "person": True}, {"is it alive?", "is it a person?"})
        self.assertNotIn(nxt["concept"], ("animal", "plant"))

    def _walk_spine(self, answers, asked):
        """Walk the remaining spine, answering 'no' to each, and return the concepts asked."""
        answers, asked, seen = dict(answers), set(asked), []
        for _ in range(12):
            entry = twentyq_kb.next_spine_question(answers, asked)
            if entry is None:
                break
            seen.append(entry["concept"])
            asked.add(entry["question"])
            answers[entry["concept"]] = False
        return seen

    def test_animal_prunes_plant_and_place(self):
        # Once it's an animal, don't waste a question on "is it a plant?" or "is it a place?".
        seen = self._walk_spine(
            {"alive": True, "person": False, "animal": True},
            {"is it alive?", "is it a person?", "is it an animal?"})
        self.assertNotIn("plant", seen)
        self.assertNotIn("place", seen)

    def test_manmade_keeps_place(self):
        # "is it a place?" is NOT pruned by man-made=yes: man-made PLACES (Coney Island,
        # the Eiffel Tower, a stadium) need that signal, so it must still be asked.
        seen = self._walk_spine(
            {"alive": False, "person": False, "manmade": True},
            {"is it alive?", "is it a person?", "is it man made?"})
        self.assertIn("place", seen)

    def test_alive_prunes_place(self):
        # A living thing isn't a "place" either.
        seen = self._walk_spine(
            {"alive": True, "person": False, "animal": True},
            {"is it alive?", "is it a person?", "is it an animal?"})
        self.assertNotIn("place", seen)

    def test_handheld_prunes_place(self):
        # Something you can hold in your hands is never a place.
        seen = self._walk_spine(
            {"alive": False, "manmade": True, "handheld": True},
            {"is it alive?", "is it man made?", "can you hold it?"})
        self.assertNotIn("place", seen)

    def test_edible_prunes_material_and_category_probes(self):
        # Nothing edible is metal/electronic/a tool/wearable/a vehicle — the food
        # tier-2 branch asks the useful questions instead.
        answers = {"alive": False, "manmade": True, "edible": True}
        asked = {"is it alive?", "is it man made?", "is it edible?"}
        seen = self._walk_spine(answers, asked)
        for wasted in ("metal", "electronic", "tool", "wearable", "vehicle"):
            self.assertNotIn(wasted, seen)
        self.assertIn("drink", seen)
        self.assertIn("sweet", seen)

    def test_object_branch_reaches_toy_question(self):
        # The missing splitter that lost the 2026-07-07 rubber-ducky game: a man-made
        # non-edible object game must reach "is it a toy or a game?" within the spine.
        seen = self._walk_spine(
            {"alive": False, "manmade": True}, {"is it alive?", "is it man made?"})
        self.assertIn("toy", seen)

    def test_bathroom_question_requires_indoors(self):
        answers = {"alive": False, "manmade": True, "indoors": True, "toy": True}
        entry = twentyq_kb.next_spine_question(
            dict(answers), {"is it alive?", "is it man made?", "is it found in a house?",
                            "is it a toy or a game?"})
        walked = self._walk_spine(answers, {"is it alive?", "is it man made?",
                                            "is it found in a house?", "is it a toy or a game?"})
        self.assertIsNotNone(entry)
        self.assertIn("bathroom", walked)

    def test_person_branch_prunes_object_probes_and_asks_person_questions(self):
        # "can you hold it?" aimed at someone's grandmother is a wasted (and weird)
        # question — the person tier-2 branch takes over instead.
        seen = self._walk_spine(
            {"alive": True, "person": True}, {"is it alive?", "is it a person?"})
        for wasted in ("bigger", "handheld", "indoors", "edible", "place"):
            self.assertNotIn(wasted, seen)
        self.assertIn("real_person", seen)
        self.assertIn("famous", seen)

    def test_animal_branch_asks_animal_questions(self):
        seen = self._walk_spine(
            {"alive": True, "animal": True}, {"is it alive?", "is it an animal?"})
        self.assertIn("pet", seen)
        self.assertIn("four_legs", seen)

    def test_spine_menu_lists_applicable_unasked_questions(self):
        menu = twentyq_kb.spine_menu(
            {"alive": False, "manmade": True}, {"is it alive?", "is it man made?"}, limit=3)
        self.assertEqual(len(menu), 3)
        self.assertNotIn("is it alive?", menu)
        self.assertNotIn("is it a person?", menu)   # pruned for man-made objects

    def test_snap_guess_grounds_to_vocab(self):
        self.assertEqual(twentyq_kb.snap_guess("a guitar"), "guitar")
        self.assertEqual(twentyq_kb.snap_guess("GUITAR"), "guitar")

    def test_snap_guess_returns_none_for_unknown(self):
        self.assertIsNone(twentyq_kb.snap_guess("xyzzy qwerty nonsense"))
        self.assertIsNone(twentyq_kb.snap_guess(""))


# ── Turn flow ───────────────────────────────────────────────────────────────────

class GuesserFlowTest(unittest.TestCase):
    def setUp(self):
        _reset_game()

    def test_start_reverses_roles_no_secret(self):
        with _mocked_llm(lambda *a, **k: ""):
            opener = games._20q_start(None)   # the mock echoes the directive text
        # Rex no longer holds a secret — the player does.
        self.assertNotIn("secret", games._game_state)
        self.assertEqual(games._game_state["phase"], "ready")
        self.assertEqual(games._game_state["question_count"], 0)
        # The opener must make all three player instructions explicit: pick a thing,
        # keep it secret, and signal when ready.
        low = opener.lower()
        self.assertIn("person, place, or thing", low)
        self.assertIn("secret", low)
        self.assertIn("ready", low)

    def test_spine_opening_and_branch_narrowing(self):
        with _mocked_llm(lambda *a, **k: ""):
            games._20q_start(None)
            # Player signals ready → Rex asks Q1 from the spine.
            resp, done = games._20q_handle("ok ready", None)
            self.assertFalse(done)
            self.assertEqual(games._game_state["question_count"], 1)
            self.assertEqual(games._game_state["last_question"], "is it alive?")
            self.assertIn("is it alive?", resp)

            # Answer "no" → concept recorded; the object branch (man-made) comes next.
            # "is it a person?" is deferred until man-made is known, so an object game
            # never wastes Q2 on it (dead/fictional people surface via man-made=no).
            resp, done = games._20q_handle("no", None)
            self.assertFalse(done)
            self.assertEqual(games._game_state["concept_answers"], {"alive": False})
            self.assertEqual(games._game_state["question_count"], 2)
            self.assertIn("is it man made?", resp)

            # Man-made → size next (broad splitters before category probes).
            resp, done = games._20q_handle("yes", None)
            self.assertEqual(games._game_state["question_count"], 3)
            self.assertIn("is it big?", resp)

    def test_forced_guess_at_question_cap(self):
        _reset_game()
        games._game_state.update({
            "phase": "asking", "qa_log": [{"q": "is it alive?", "a": "no"}],
            "asked": ["is it alive?"], "concept_answers": {"alive": False},
            "question_count": games._20Q_MAX_QUESTIONS, "guesses": [],
            "last_question": "is it man made?", "last_concept": "manmade",
        })
        with _mocked_llm(lambda *a, **k: "guitar"):
            resp, done = games._20q_handle("yes", None)
        # Out of questions → Rex must commit to a guess (not ask another).
        self.assertFalse(done)
        self.assertEqual(games._game_state["phase"], "guessing")
        self.assertEqual(games._game_state["pending_guess"], "guitar")
        self.assertIn("final answer", resp.lower())

    def test_llm_guess_is_grounded_to_vocabulary(self):
        _reset_game()
        games._game_state.update({
            "phase": "asking", "qa_log": [], "asked": [], "concept_answers": {},
            "question_count": games._20Q_MAX_QUESTIONS, "guesses": [],
        })
        # Model says "a guitar" → grounded to the canonical subject "guitar".
        with _mocked_llm(lambda *a, **k: "a guitar"):
            games._20q_ask_next(None)
        self.assertEqual(games._game_state["pending_guess"], "guitar")

    def test_correct_guess_wins(self):
        _reset_game()
        games._game_state.update({
            "phase": "guessing", "pending_guess": "guitar", "question_count": 7,
            "guesses": ["guitar"], "qa_log": [],
        })
        with _mocked_llm(lambda *a, **k: ""):
            resp, done = games._20q_handle("yes that's it!", None)
        self.assertTrue(done)
        self.assertEqual(games._game_state["result"], "win")
        self.assertEqual(games._game_state["final_guess"], "guitar")

    def test_wrong_guess_with_road_left_keeps_narrowing(self):
        _reset_game()
        games._game_state.update({
            "phase": "guessing", "pending_guess": "banjo", "question_count": 9,
            "guesses": ["banjo"], "asked": [], "qa_log": [],
        })
        ask_move = json.dumps({"action": "ask", "question": "Does it have strings?"})
        with _mocked_llm(lambda *a, **k: ask_move):
            resp, done = games._20q_handle("nope", None)
        self.assertFalse(done)
        self.assertEqual(games._game_state["phase"], "asking")
        # The rejected guess is logged so the model won't circle back to it.
        self.assertIn({"q": "is it banjo?", "a": "no"}, games._game_state["qa_log"])

    def test_wrong_guess_out_of_guesses_loses(self):
        _reset_game()
        games._game_state.update({
            "phase": "guessing", "pending_guess": "tuba", "question_count": 12,
            "guesses": ["banjo", "guitar", "tuba"], "qa_log": [],
        })
        with _mocked_llm(lambda *a, **k: ""):
            resp, done = games._20q_handle("no", None)
        self.assertTrue(done)
        self.assertEqual(games._game_state["result"], "lose")


# ── Decide-engine discipline ─────────────────────────────────────────────────────

class DecideEngineTest(unittest.TestCase):
    """The deterministic rails around the LLM turn engine: a guess must pass the evidence
    gate, a near-duplicate question is replaced by a proven splitter, and a confident
    early exit can pre-empt the spine. These are the fixes for the live 2026-07-07 loss
    (wallet guess at Q14 off a broad shortlist; back-to-back 'container' questions)."""

    def setUp(self):
        _reset_game()
        games._game_state = {"concept_answers": {}, "qa_log": [], "asked": [],
                             "candidates": [], "guesses": []}

    def test_premature_guess_is_converted_to_ask(self):
        # 5 candidates at Q8 with 12 questions left — the gate holds fire and asks instead.
        move = json.dumps({"candidates": ["wallet", "purse", "belt", "hat", "mug"],
                           "action": "guess", "subject": "wallet"})
        with _mocked_llm(lambda *a, **k: move):
            decision = games._20q_decide([], [], 8, [], [])
        self.assertEqual(decision["action"], "ask")
        self.assertTrue(decision["question"])

    def test_confident_guess_passes_the_gate(self):
        # A 2-candidate shortlist is the model's strike signal — the guess goes through.
        move = json.dumps({"candidates": ["rubber duck", "bath toy"],
                           "action": "guess", "subject": "rubber duck"})
        with _mocked_llm(lambda *a, **k: move):
            decision = games._20q_decide([], [], 8, [], [])
        self.assertEqual(decision["action"], "guess")
        self.assertEqual(decision["subject"], "rubber duck")

    def test_endgame_guess_always_allowed(self):
        # 2 or fewer questions remaining: even a broad shortlist must commit.
        move = json.dumps({"candidates": ["a", "b", "c", "d"],
                           "action": "guess", "subject": "a"})
        with _mocked_llm(lambda *a, **k: move):
            decision = games._20q_decide([], [], games._20Q_MAX_QUESTIONS - 2, [], [])
        self.assertEqual(decision["action"], "guess")

    def test_near_duplicate_question_replaced_by_fallback(self):
        asked = [games._norm_q("is it mainly a container for storing things?")]
        move = json.dumps({"candidates": ["box", "jar", "bag", "bin"], "action": "ask",
                           "question": "Is it mainly a container for storing things"})
        with _mocked_llm(lambda *a, **k: move):
            decision = games._20q_decide([], list(asked), 8, [], [])
        self.assertEqual(decision["action"], "ask")
        self.assertNotEqual(games._norm_q(decision["question"]), asked[0])

    def test_narrowing_question_is_not_blocked(self):
        # A legit narrowing of an earlier category must survive the duplicate check.
        self.assertFalse(games._20q_question_is_redundant(
            "is it a stringed instrument?", [games._norm_q("is it a musical instrument?")]))
        self.assertTrue(games._20q_question_is_redundant(
            "Is it a musical instrument", [games._norm_q("is it a musical instrument?")]))

    def test_early_exit_guesses_mid_spine_after_yes(self):
        # Q8, spine still available, last answer YES, model confident (2 candidates):
        # Rex strikes instead of grinding through the rest of the spine.
        games._game_state.update({
            "phase": "asking",
            "qa_log": [{"q": "is it a toy or a game?", "a": "yes"}],
            "asked": [games._norm_q("is it a toy or a game?")],
            "concept_answers": {"alive": False, "manmade": True, "toy": True},
            "question_count": 8, "guesses": [], "candidates": [],
        })
        move = json.dumps({"candidates": ["rubber duck", "bath toy"],
                           "action": "guess", "subject": "rubber duck"})
        with _mocked_llm(lambda *a, **k: move):
            games._20q_ask_next(None)
        self.assertEqual(games._game_state["phase"], "guessing")
        self.assertEqual(games._game_state["pending_guess"], "rubber duck")

    def test_no_early_exit_check_after_a_no(self):
        # Same setup but the last answer was NO — the spine continues without an
        # early-exit model call (the mock would raise if consulted).
        games._game_state.update({
            "phase": "asking",
            "qa_log": [{"q": "is it a tool?", "a": "no"}],
            "asked": [games._norm_q("is it a tool?")],
            "concept_answers": {"alive": False, "manmade": True, "tool": False},
            "question_count": 8, "guesses": [], "candidates": [],
        })
        def _boom(*a, **k):
            raise AssertionError("early-exit decide should not run after a NO")
        with mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx), \
             mock.patch.object(games, "_body_beat", return_value=None), \
             mock.patch.object(games, "_smart_call", side_effect=_boom):
            games._20q_ask_next(None)
        self.assertEqual(games._game_state["phase"], "asking")
        self.assertEqual(games._game_state["question_count"], 9)

    def test_wrong_guess_recorded_in_asked_list(self):
        games._game_state.update({
            "phase": "guessing", "pending_guess": "wallet", "question_count": 9,
            "guesses": ["wallet"], "asked": [], "qa_log": [], "concept_answers": {},
        })
        ask_move = json.dumps({"candidates": ["mug", "vase", "frame", "candle"],
                               "action": "ask", "question": "Is it decorative?"})
        with _mocked_llm(lambda *a, **k: ask_move):
            games._20q_handle("nope", None)
        self.assertIn(games._norm_q("is it wallet?"), games._game_state["asked"])


# ── Episodic outcome ────────────────────────────────────────────────────────────

class OutcomeStringTest(unittest.TestCase):
    def test_win_outcome(self):
        state = {"result": "win", "final_guess": "guitar", "question_count": 7}
        self.assertEqual(games._extract_game_outcome(state),
                         "guessed it — guitar — in 7 questions")

    def test_lose_outcome(self):
        self.assertEqual(games._extract_game_outcome({"result": "lose"}), "couldn't guess it")

    def test_trivia_outcome_still_works(self):
        state = {"score": 4, "total_questions": 5, "history": [1, 2, 3, 4]}
        self.assertEqual(games._extract_game_outcome(state), "scored 4 out of 5")


if __name__ == "__main__":
    unittest.main()
