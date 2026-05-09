import unittest
from unittest import mock


class ComedyModesTests(unittest.TestCase):
    def setUp(self):
        from intelligence import comedy_modes

        comedy_modes.reset_recent_state()

    def _frame(self, *, purpose="answer", allow_roast="normal"):
        from intelligence import social_frame

        return social_frame.SocialFrame(
            addressee="Bret",
            purpose=purpose,
            max_words=32,
            max_sentences=2,
            allow_question=False,
            allow_roast=allow_roast,
            allow_visual_comment=False,
            reason="test",
        )

    def test_sensitive_turn_uses_straight_mode(self):
        from intelligence import comedy_modes

        mode = comedy_modes.select_mode(
            "my friend died yesterday",
            person_id=1,
            frame=self._frame(),
        )

        self.assertEqual(mode.key, "straight")
        self.assertIn("Do not add a joke", comedy_modes.build_directive(mode))

    def test_explicit_humor_can_use_programming_self_own_mode(self):
        from intelligence import comedy_modes

        with mock.patch.object(comedy_modes.random, "choice", return_value="self_own"):
            mode = comedy_modes.select_mode(
                "say something funny",
                person_id=1,
                frame=self._frame(),
            )

        self.assertEqual(mode.key, "self_own")
        self.assertIn("I'm still getting used to my programming", comedy_modes.build_directive(mode))

    def test_bland_ack_gets_replaced_by_curated_dry_ack(self):
        from intelligence import comedy_modes

        mode = comedy_modes.ComedyMode("dry_ack", "dry", "Comedy mode: dry_ack.")
        polished = comedy_modes.polish_response("Okay.", mode)

        self.assertNotEqual(polished.lower(), "okay.")
        self.assertTrue(polished)

    def test_directive_mentions_recent_premise_for_anti_repeat(self):
        from intelligence import comedy_modes

        mode = comedy_modes.ComedyMode("self_own", "self", "Comedy mode: self_own.")
        comedy_modes.polish_response("My flight record says supervised confidence.", mode)

        directive = comedy_modes.build_directive(mode)

        self.assertIn("Anti-repeat", directive)
        self.assertIn("rex_self_own_programming", directive)


if __name__ == "__main__":
    unittest.main()
