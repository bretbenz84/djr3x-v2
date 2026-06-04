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


class RoastForwardDirectiveTests(unittest.TestCase):
    def _frame(self, *, allow_roast="normal", allow_visual_comment=False):
        from intelligence import social_frame

        return social_frame.SocialFrame(
            addressee="Bret",
            purpose="answer",
            max_words=32,
            max_sentences=2,
            allow_question=False,
            allow_roast=allow_roast,
            allow_visual_comment=allow_visual_comment,
            reason="test",
        )

    def test_normal_roast_directive_is_roast_lean_and_specific(self):
        from intelligence import social_frame

        directive = social_frame.build_directive(self._frame(allow_roast="normal"))
        # Roast-lean: still sharp and SPECIFIC when it roasts, but no longer a
        # mandatory jab every single turn (a plain genuine reaction is allowed).
        self.assertIn("ROAST-LEAN", directive)
        self.assertIn("SPECIFIC", directive)
        self.assertIn("have to roast every single turn", directive)
        # The old socially-on-target softener must still be gone.
        self.assertNotIn("keep it socially on-target", directive)

    def test_tender_roast_levels_stay_gentle(self):
        from intelligence import social_frame

        none_directive = social_frame.build_directive(self._frame(allow_roast="none"))
        self.assertIn("No roasts", none_directive)
        light_directive = social_frame.build_directive(self._frame(allow_roast="light"))
        self.assertIn("tiny surface-level tap", light_directive)

    def test_visual_directive_invites_roasting_what_he_sees(self):
        from intelligence import social_frame

        directive = social_frame.build_directive(
            self._frame(allow_visual_comment=True)
        )
        self.assertIn("What you actually SEE", directive)
        # Must reinforce the no-invented-props guardrail right where the
        # temptation is introduced (Rex kept inventing a "drink in their hand").
        self.assertIn("never invent", directive)

    def test_visual_allowed_on_normal_upbeat_turn_without_invitation(self):
        from intelligence import social_frame

        # No "look at this" cue, neutral affect, no sensitivity → still allowed so
        # appearance/props become roast material.
        self.assertTrue(
            social_frame._visual_allowed(
                "I made you from scratch", "", "normal", "banter", "neutral", "none"
            )
        )

    def test_visual_suppressed_on_sensitive_or_micro_turns(self):
        from intelligence import social_frame

        # Sad affect, support mode, sensitivity, and micro all bail before the
        # normal-turn allowance.
        self.assertFalse(
            social_frame._visual_allowed("the funeral", "", "normal", "support", "sad", "none")
        )
        self.assertFalse(
            social_frame._visual_allowed("hi", "", "micro", "banter", "neutral", "none")
        )
        self.assertFalse(
            social_frame._visual_allowed("my dog died", "", "normal", "banter", "neutral", "grief")
        )

    def test_visual_normal_turn_allowance_is_configurable(self):
        from intelligence import social_frame

        with mock.patch.object(social_frame.config, "VISUAL_ROAST_ON_NORMAL_TURNS", False):
            self.assertFalse(
                social_frame._visual_allowed(
                    "I made you from scratch", "", "normal", "banter", "neutral", "none"
                )
            )


if __name__ == "__main__":
    unittest.main()
