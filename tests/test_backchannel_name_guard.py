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
