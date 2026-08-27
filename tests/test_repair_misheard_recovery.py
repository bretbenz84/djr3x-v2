"""
Misheard/misunderstood recovery: when the human flags a mishearing WITHOUT re-saying it,
Rex makes a short save-face circuit-glitch joke and invites them to repeat — varied, not
the same line every time. When they DO supply the corrected words, that's a real
correction (accepted via the LLM path), not a "say it again" prompt.
"""

import time
import unittest

from intelligence import repair_moves as rm


class CorrectionHasContentTest(unittest.TestCase):
    def test_real_corrections_have_content(self):
        for c in ["blues not jazz", "Tom Foster", "blues", "the Crab Nebula"]:
            self.assertTrue(rm.correction_has_content(c), c)

    def test_signal_only_is_not_content(self):
        # Echoes of the correction trigger — no new words supplied.
        for c in ["", "I said", "you misunderstood me", "what I said", "me",
                  "you misheard me", "that"]:
            self.assertFalse(rm.correction_has_content(c), c)


class MisheardRecoveryResponseTest(unittest.TestCase):
    def test_two_beat_structure(self):
        # save-face joke first, invitation to repeat second.
        r = rm.misheard_recovery_response()
        self.assertTrue(any(r.startswith(sf) for sf in rm._SAVE_FACE_LINES), r)
        self.assertTrue(any(r.endswith(rp) for rp in rm._REPROMPT_LINES), r)

    def test_varies_across_calls(self):
        seen = {rm.misheard_recovery_response() for _ in range(16)}
        self.assertGreater(len(seen), 3)

    def test_consecutive_lines_differ(self):
        prev = None
        for _ in range(8):
            r = rm.misheard_recovery_response()
            self.assertNotEqual(r, prev)
            prev = r

    def test_star_tours_line_preserved(self):
        # The sign-off now carries an authored [excited] delivery tag for eleven_v3;
        # the spoken words themselves are unchanged.
        self.assertIn(
            "I'm sure we'll have better luck next time!",
            [rm.strip_audio_tags(line) for line in rm._SAVE_FACE_LINES],
        )


class DetectBroadenedTest(unittest.TestCase):
    def setUp(self):
        rm.note_assistant_turn("What targets have you been shooting lately?")

    def test_broadened_misheard_phrases_detected(self):
        for t in ["you misheard me", "you didn't catch that", "you got my words wrong",
                  "No, you misunderstood me", "That's not what I said"]:
            with self.subTest(t=t):
                m = rm.detect(t)
                self.assertIsNotNone(m, t)
                self.assertIn(m["kind"], {"misheard", "misunderstood"}, t)
                # None of these supplied real corrected content → joke+reprompt path.
                self.assertFalse(rm.correction_has_content(m.get("correction")), t)

    def test_emphasis_is_not_a_repair(self):
        self.assertIsNone(rm.detect("Like I said, I love astrophotography"))
        self.assertIsNone(rm.detect("As I said, the telescope is new"))

    def test_real_correction_routes_with_content(self):
        m = rm.detect("No, I said blues not jazz")
        self.assertEqual(m["kind"], "misheard")
        self.assertTrue(rm.correction_has_content(m["correction"]))


class NarrationIsNotACorrectionTest(unittest.TestCase):
    """A story that happens to mention a name must not be read as "you misheard me".

    Field 2026-07-24: "We got it from a dear friend who lived on a boat. Her name
    was... Goldnatt." matched the bare `(his|her|their|the) name was` alternative, so
    Rex answered a warm story with "Static in the receptors; that came through
    garbled. Give it to me again?" — and the next two turns were spent untangling it.
    """

    def test_storytelling_is_not_a_repair(self):
        for text in (
            "We got it from a dear friend who lived on a boat. Her name was... Goldnatt.",
            "Her name was Ada",
            "his name was Jeff and he lived next door",
            "the name was on the box",
            "my name was on the list",
        ):
            self.assertIsNone(rm.detect(text), text)

    def test_real_name_corrections_still_fire(self):
        for text in ("no, her name was Sarah",
                     "actually his name was Jeff",
                     "not my name was wrong"):
            self.assertIsNotNone(rm.detect(text), text)

    def test_other_misheard_forms_unaffected(self):
        for text in ("you misheard me",
                     "that's not what I said",
                     "no, I said blues",
                     "you didn't catch that"):
            m = rm.detect(text)
            self.assertIsNotNone(m, text)


class PhantomAudioStandDownTest(unittest.TestCase):
    """Field 2026-08-27 13:34: Rex's boot line came back through the mic, he answered it,
    and Bret's "I didn't say anything." was classified kind=misheard — so Rex asked him to
    repeat words that never existed ("Run that by me one more time?"). The denial of THAT
    was heard as a new turn and got "Hit me with it again and I'll get it right." A denial
    of speech must stand down, never ask."""

    def setUp(self):
        rm.clear()
        rm.note_assistant_turn("Okay what, exactly?")

    def tearDown(self):
        rm.clear()

    def test_field_utterances_are_phantom_audio_not_misheard(self):
        for text in (
            "I didn't say anything.",
            "There's nothing to run by you. I didn't say anything.",
            "I said nothing.",
            "Nobody said anything.",
            "No one said anything",
            "I didn't speak",
            "I wasn't talking",
            "I didn't say a word",
            "That wasn't me talking",
            "Nothing was said.",
        ):
            with self.subTest(text=text):
                move = rm.detect(text)
                self.assertIsNotNone(move, text)
                self.assertEqual(move["kind"], "phantom_audio", text)

    def test_stand_down_never_asks_a_question(self):
        # A question is itself an ask to repeat, and the denial of it re-arms the loop.
        for line in rm._PHANTOM_AUDIO_LINES:
            self.assertNotIn("?", line, line)
        for _ in range(12):
            self.assertNotIn("?", rm.phantom_audio_response())

    def test_stand_down_varies_and_arms_the_phantom_window(self):
        self.assertFalse(rm.phantom_recent())
        prev = None
        for _ in range(6):
            line = rm.phantom_audio_response()
            self.assertNotEqual(line, prev)
            prev = line
        self.assertTrue(rm.phantom_recent())
        self.assertFalse(rm.phantom_recent(max_age_secs=0.0))

    def test_fallback_response_also_stands_down(self):
        resp = rm.fallback_response({"kind": "phantom_audio"})
        self.assertIn(resp, rm._PHANTOM_AUDIO_LINES)

    def test_prompt_forbids_asking_for_a_repeat(self):
        prompt = rm.build_prompt({"kind": "phantom_audio", "user_text": "I didn't say anything."})
        self.assertIn("phantom audio", prompt.lower())
        self.assertIn("do NOT ask any question", prompt)

    def test_nothing_to_repeat_needs_rexs_own_ask_to_repeat(self):
        # Standalone it is ordinary speech; only after Rex asked for a repeat does it
        # mean "you heard a ghost".
        rm.clear()
        rm.note_assistant_turn("So what did you think of the festival?")
        self.assertIsNone(rm.detect("There's nothing to hit you again with."))
        rm.note_assistant_turn("Hit me with it again and I'll get it right.")
        move = rm.detect("There's nothing to hit you again with.")
        self.assertIsNotNone(move)
        self.assertEqual(move["kind"], "phantom_audio")

    def test_ordinary_nothing_to_phrases_stay_conversation(self):
        rm.note_assistant_turn("Say that again?")
        for text in ("there's nothing to worry about", "there's nothing to it"):
            self.assertIsNone(rm.detect(text), text)

    def test_genuine_mishear_repairs_still_ask_for_a_repeat(self):
        # MUST KEEP WORKING: the human really did speak and Rex really did mishear.
        for text in ("you misheard me",
                     "that's not what I said",
                     "you didn't catch that",
                     "No, you misunderstood me"):
            with self.subTest(text=text):
                move = rm.detect(text)
                self.assertIsNotNone(move, text)
                self.assertIn(move["kind"], {"misheard", "misunderstood"}, text)

    def test_content_disputes_and_corrections_keep_their_routing(self):
        self.assertEqual(rm.detect("No, I said blues not jazz")["kind"], "misheard")
        self.assertIsNone(rm.detect("I didn't say that"))
        # "about" makes it a content dispute, not a denial of speaking.
        move = rm.detect("I didn't say anything about the cat")
        self.assertIsNotNone(move)
        self.assertNotEqual(move["kind"], "phantom_audio")
        # Identity corrections stay in the wrong_person lane.
        self.assertEqual(rm.detect("That was Jane, not me")["kind"], "wrong_person")
        self.assertEqual(rm.detect("wrong person")["kind"], "wrong_person")

    def test_phantom_needs_rex_to_have_spoken_recently(self):
        rm.clear()
        self.assertIsNone(rm.detect("Nobody said anything."))


class AskToRepeatStrikeCapTest(unittest.TestCase):
    """Rex asked three times in 75 seconds on 2026-08-27 — once from the low-trust lane
    (13:34:17) and twice from the misheard repair lane (13:34:42, 13:34:57). The cap is
    shared across both lanes on purpose; a per-lane cap just lets them alternate."""

    def setUp(self):
        rm.clear()

    def tearDown(self):
        rm.clear()

    def test_cap_trips_on_the_third_consecutive_ask(self):
        self.assertFalse(rm.ask_to_repeat_exhausted())
        rm.note_ask_to_repeat()
        self.assertFalse(rm.ask_to_repeat_exhausted())
        rm.note_ask_to_repeat()
        self.assertTrue(rm.ask_to_repeat_exhausted())

    def test_cap_clears_so_a_later_genuine_mishear_still_gets_one_ask(self):
        rm.note_ask_to_repeat()
        rm.note_ask_to_repeat()
        self.assertTrue(rm.ask_to_repeat_exhausted())
        rm.clear_ask_to_repeat_strikes()
        self.assertFalse(rm.ask_to_repeat_exhausted())

    def test_strikes_lapse_with_the_window(self):
        rm.note_ask_to_repeat()
        rm.note_ask_to_repeat()
        self.assertTrue(rm.ask_to_repeat_exhausted())
        rm._last_ask_to_repeat_at = (
            time.monotonic() - rm.ASK_TO_REPEAT_STRIKE_WINDOW_SECS - 1.0
        )
        self.assertFalse(rm.ask_to_repeat_exhausted())

    def test_session_clear_resets_everything(self):
        rm.note_ask_to_repeat()
        rm.note_ask_to_repeat()
        rm.phantom_audio_response()
        rm.clear()
        self.assertFalse(rm.ask_to_repeat_exhausted())
        self.assertFalse(rm.phantom_recent())

    def test_rex_own_reprompt_pools_are_recognised_as_asks(self):
        for line in rm._REPROMPT_LINES:
            self.assertTrue(rm.looks_like_ask_to_repeat(line), line)
        for line in ("Sorry — what was that?", "I didn't catch that. Say it again?",
                     "What was that?", "Sorry, one more time?",
                     "Hm? Run that by me again.",
                     "I only caught about half of that — say it again?"):
            self.assertTrue(rm.looks_like_ask_to_repeat(line), line)
        for line in ("Okay what, exactly?", "With who?",
                     "What room is this, so I can recognize it next time?"):
            self.assertFalse(rm.looks_like_ask_to_repeat(line), line)


class PhantomAudioOverReachTest(unittest.TestCase):
    """The phantom stand-down must claim ONLY a denial of this exchange.

    Review 2026-08-27: the pattern used a blocklist of following words, which
    only ever saw the word after the VERB — so "Nobody said anything for like a
    full minute, it was so awkward" (ordinary storytelling) was answered with
    "that was my own echo coming back at me", and that false hit then armed the
    90s window that muzzles the reprompt lanes.
    """

    def setUp(self):
        rm.clear()
        self.addCleanup(rm.clear)
        rm.note_assistant_turn("Okay what, exactly?")

    def _kind(self, text):
        d = rm.detect(text)
        return d.get("kind") if isinstance(d, dict) else None

    def test_narrative_about_a_silence_is_not_a_stand_down(self):
        for text in (
            "Nobody said anything for like a full minute, it was so awkward.",
            "No one spoke up at the meeting.",
            "Nothing was said at dinner.",
            "I wasn't talking, I was singing.",
        ):
            with self.subTest(text=text):
                self.assertNotEqual(self._kind(text), "phantom_audio")

    def test_a_content_defence_keeps_its_own_lane(self):
        # "I didn't say anything WRONG" disputes what was said, not that it was.
        for text in ("I didn't say anything wrong.", "I didn't say anything bad.",
                     "I didn't say anything yet.", "I didn't say anything about the cat."):
            with self.subTest(text=text):
                self.assertNotEqual(self._kind(text), "phantom_audio")

    def test_habitual_present_is_not_a_denial(self):
        self.assertNotEqual(self._kind("I never say anything interesting."), "phantom_audio")

    def test_the_real_denials_still_stand_down(self):
        for text in ("I didn't say anything.", "I said nothing.", "I didn't speak.",
                     "Nobody said anything.", "I never said anything.",
                     "That wasn't me talking.", "Nothing was said.",
                     "There's nothing to run by you. I didn't say anything."):
            with self.subTest(text=text):
                self.assertEqual(self._kind(text), "phantom_audio")

    def test_the_warm_window_does_not_claim_ordinary_speech(self):
        rm.phantom_audio_response()
        for text in ("There's nothing to say.", "I have nothing to add.",
                     "I've got nothing to tell you."):
            with self.subTest(text=text):
                self.assertNotEqual(self._kind(text), "phantom_audio")


class AskCapStandDownWordingTest(unittest.TestCase):
    """The cap is reached from the MISHEARD lane, where the human really did
    speak — so it must not reuse the phantom pool, which asserts silence.
    Review 2026-08-27: telling someone who has now said it three times that they
    said nothing is worse than the third ask."""

    def setUp(self):
        rm.clear()
        self.addCleanup(rm.clear)

    def test_the_pool_never_asks_a_question(self):
        for _ in range(30):
            self.assertNotIn("?", rm.ask_cap_stand_down_response())

    def test_the_pool_never_claims_the_human_was_silent(self):
        seen = {rm.ask_cap_stand_down_response() for _ in range(30)}
        self.assertTrue(seen)
        for line in seen:
            self.assertNotIn(line, rm._PHANTOM_AUDIO_LINES)

    def test_it_arms_the_same_muzzle_window(self):
        rm.ask_cap_stand_down_response()
        self.assertTrue(rm.phantom_recent())


if __name__ == "__main__":
    unittest.main()
