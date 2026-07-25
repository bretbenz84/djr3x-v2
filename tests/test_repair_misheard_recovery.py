"""
Misheard/misunderstood recovery: when the human flags a mishearing WITHOUT re-saying it,
Rex makes a short save-face circuit-glitch joke and invites them to repeat — varied, not
the same line every time. When they DO supply the corrected words, that's a real
correction (accepted via the LLM path), not a "say it again" prompt.
"""

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


if __name__ == "__main__":
    unittest.main()
