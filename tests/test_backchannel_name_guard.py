"""Backchannel utterances must never become person names.

Live incident 2026-07-26: Whisper transcribed a backchannel "Mm-hmm" as the
answer to Rex's identity prompt, enrolling the speaker's own face and voice
under a phantom person whose fresher voiceprint then outscored the real person
on their own speech (0.93-0.99 vs 0.43-0.79) in every later session.
"""

import unittest

from memory.name_validation import normalize_person_name


class BackchannelNameGuardTest(unittest.TestCase):
    def test_backchannels_rejected(self):
        for utterance in [
            "Mm-hmm", "Mm-hmm.", "mm hmm", "Uh-huh", "uh huh", "Mhm",
            "Yeah", "Yep", "Nah", "Ha ha", "Haha", "Ha ha ha", "Whoa",
            "Hmm", "Uh", "Um",
        ]:
            self.assertIsNone(
                normalize_person_name(utterance),
                f"{utterance!r} must not normalize to a person name",
            )

    def test_real_names_still_accepted(self):
        for name in ["Bret", "Bret Benziger", "Hannah", "Mia", "Hema", "Hamm", "Uma"]:
            self.assertEqual(normalize_person_name(name), name)

    def test_prompted_identity_reply_backchannel_triggers_reask_path(self):
        """A backchannel reply to Rex's "what's your name?" must parse to None,
        which routes interaction into the bounded gentle re-ask instead of
        enrolling a phantom person."""
        from intelligence.interaction import _extract_introduced_name

        for reply in ["Mm-hmm.", "Uh-huh", "Yeah"]:
            self.assertIsNone(_extract_introduced_name(reply, allow_bare_name=True))
        self.assertEqual(
            _extract_introduced_name("I'm Bret", allow_bare_name=True), "Bret"
        )


class ProfaneNameGuardTest(unittest.TestCase):
    """Profanity and slurs must never become person names.

    Live incident 2026-08-26 20:10:44: the Jeopardy roster prompt heard
    "Jeremy, Bret, J T. Ah, fuck. Never mind. We don't know about that." and the
    trailing fragment was minted as person id 10 named "Fuck". The next night it
    surfaced unprompted in a lull: "I met someone named Fuck once, which is
    honestly the most honest introduction this room has ever offered."
    """

    def test_the_exact_field_fragment_is_rejected(self):
        self.assertIsNone(
            normalize_person_name("Fuck. Never Mind. We Don't Know About That")
        )

    def test_profanity_and_slurs_rejected(self):
        for utterance in [
            "Fuck", "fuck", "Fucking", "Shit", "Bullshit", "Bitch", "Asshole",
            "Cunt", "Bastard", "Damn", "Goddamn", "Hell", "Crap", "Retard",
            "Nigger", "Faggot", "Chink", "Spic", "Tranny",
        ]:
            with self.subTest(utterance=utterance):
                self.assertIsNone(normalize_person_name(utterance))

    def test_one_profane_token_poisons_the_whole_name(self):
        self.assertIsNone(normalize_person_name("Fucking Bret"))
        self.assertIsNone(normalize_person_name("Bret Asshole"))

    def test_real_names_that_merely_contain_the_letters_survive(self):
        # The check is TOKEN-exact, never a substring: an over-broad list would
        # cost a real guest their identity permanently.
        for name in [
            "Cassidy", "Bassett", "Shitake", "Damon", "Hellman", "Cassandra",
            "Pissarro", "Bret Benziger", "Homer",
        ]:
            with self.subTest(name=name):
                self.assertEqual(normalize_person_name(name), name)

    def test_names_deliberately_left_off_the_blocklist_still_work(self):
        # dick / cock / coon / dyke / randy are real given names or surnames and
        # are intentionally absent from _PROFANE_NAME_TOKENS.
        for name in ["Dick", "Randy", "Peter", "Dick Van Dyke"]:
            with self.subTest(name=name):
                self.assertIsNotNone(normalize_person_name(name))

    def test_bare_question_and_command_words_are_not_names(self):
        # Found next to the "Fuck" mint: "what" only lived in _BAD_PHRASE_STARTS,
        # which the single-token branch never reads. Field 2026-08-27 13:35:17
        # — "HEARD | Bret Benziger: What?"
        for utterance in [
            "What", "What?", "How", "When", "Where", "Why", "Stop", "Wait",
            "Nothing", "Yes", "Huh", "Never mind",
        ]:
            with self.subTest(utterance=utterance):
                self.assertIsNone(normalize_person_name(utterance))

    def test_contains_profane_token_is_the_same_table(self):
        from memory.name_validation import contains_profane_token

        self.assertTrue(contains_profane_token("fuck. Never mind. We got."))
        self.assertFalse(contains_profane_token("Jeremy"))
        self.assertFalse(contains_profane_token("Cassidy Bassett"))


if __name__ == "__main__":
    unittest.main()


class DictatedInitialsTest(unittest.TestCase):
    """Dictated initials must survive normalization as one token.

    Live 2026-08-23: someone answered "It was JT" and a phantom person named "J"
    was enrolled with a stranger's face+voice, one fuzzy match from the real JT.
    """

    def test_spaced_and_dotted_initials_collapse(self):
        for spelling in ("J T", "J.T.", "J. T.", "J.T", "j t"):
            with self.subTest(spelling=spelling):
                self.assertEqual(normalize_person_name(spelling), "JT")

    def test_initials_keep_a_following_surname(self):
        self.assertEqual(normalize_person_name("J T Thomas"), "JT Thomas")
        self.assertEqual(normalize_person_name("P. J. Thomas"), "PJ Thomas")
        self.assertEqual(normalize_person_name("A. J. Foyt"), "AJ Foyt")

    def test_lone_middle_initial_is_left_alone(self):
        self.assertEqual(normalize_person_name("Bret M Benziger"), "Bret M Benziger")
        self.assertEqual(normalize_person_name("Mary J Blige"), "Mary J Blige")

    def test_non_name_phrase_still_rejected(self):
        self.assertIsNone(normalize_person_name("It was J T."))
