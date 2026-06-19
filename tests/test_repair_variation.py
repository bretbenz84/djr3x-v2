"""
Repair-line variation (R2): the canned recovery tag appended to misheard / bare-
negation repairs must rotate with anti-repeat, so two consecutive repairs don't end
with the identical line (a bystander noticed: "Is it just gonna keep asking the same
question?").
"""

from __future__ import annotations

import unittest

from intelligence import repair_moves as rm


class GroundingCorrectionDetectTest(unittest.TestCase):
    """BUG-8: invented-detail corrections route to the 'factual' drop-it frame,
    and an object correction is no longer mislabeled 'wrong_person'."""

    def setUp(self):
        # Rex must have spoken recently for the gated 'makes no sense' to count.
        rm.note_assistant_turn("Are you building the optics side or the mount first?")

    def test_makes_no_sense_detected_as_factual(self):
        for text in (
            "What? That makes no sense",
            "That doesn't make any sense",
            "none of that makes sense",
        ):
            with self.subTest(text=text):
                move = rm.detect(text)
                self.assertIsNotNone(move)
                self.assertEqual(move["kind"], "factual")
                self.assertEqual(move["severity"], "high")

    def test_makes_no_sense_ignored_when_rex_silent(self):
        rm._last_assistant_at = 0.0  # Rex hasn't spoken
        self.assertIsNone(rm.detect("That makes no sense"))

    def test_you_mean_object_not_wrong_person(self):
        move = rm.detect("You mean my telescope?")
        self.assertTrue(move is None or move.get("kind") != "wrong_person")

    def test_you_mean_name_still_wrong_person(self):
        self.assertEqual(rm.detect("You mean Sarah?")["kind"], "wrong_person")
        self.assertEqual(rm.detect("You mean him?")["kind"], "wrong_person")

    def test_what_do_you_mean_still_clarify(self):
        # The new pattern must not swallow genuine clarification requests.
        self.assertEqual(rm.detect("What do you mean?")["kind"], "clarify")

    def test_factual_prompt_forbids_re_explaining(self):
        prompt = rm.build_prompt({"kind": "factual", "user_text": "that makes no sense"})
        low = prompt.lower()
        self.assertIn("do not re-explain", low)
        self.assertIn("do not re-ask", low)
        for kind in ("wrong_person", "misunderstood"):
            p = rm.build_prompt({"kind": kind, "user_text": "x"}).lower()
            self.assertIn("re-explain", p)


class RecoveryLineVariationTest(unittest.TestCase):
    def setUp(self):
        rm._last_recovery_line = ""

    def test_consecutive_picks_differ(self):
        picks = [rm.pick_recovery_line() for _ in range(8)]
        for a, b in zip(picks, picks[1:]):
            self.assertNotEqual(a, b, "consecutive recovery lines must differ")
        for p in picks:
            self.assertIn(p, rm._RECOVERY_LINES)

    def test_add_better_luck_line_varies_across_calls(self):
        r1 = rm.add_better_luck_line("Guess I misfired there!")
        r2 = rm.add_better_luck_line("Got it, no cool uncle award then?")
        self.assertTrue(rm._contains_recovery_line(r1))
        self.assertTrue(rm._contains_recovery_line(r2))
        # the appended tails should not be identical on consecutive repairs
        tail1 = r1.replace("Guess I misfired there!", "").strip()
        tail2 = r2.replace("Got it, no cool uncle award then?", "").strip()
        self.assertNotEqual(tail1, tail2)

    def test_no_double_append_when_already_present(self):
        line = rm._RECOVERY_LINES[0]
        out = rm.add_better_luck_line(f"Something happened. {line}", line)
        self.assertEqual(out.lower().count(line.lower()), 1)

    def test_build_prompt_and_append_use_the_same_line(self):
        repair = {"kind": "bare_negation", "user_text": "no", "last_assistant_text": "Q?"}
        prompt = rm.build_prompt(repair)
        self.assertIn("recovery_line", repair)
        self.assertIn(repair["recovery_line"], prompt)
        out = rm.add_better_luck_line("Guess I misfired there!", repair["recovery_line"])
        self.assertIn(repair["recovery_line"], out)


if __name__ == "__main__":
    unittest.main()
