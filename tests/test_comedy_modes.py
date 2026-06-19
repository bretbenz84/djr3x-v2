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


class BoundaryDetectionTest(unittest.TestCase):
    """A boundary / withdrawal must be detected (and never roasted). The quality
    eval found 'I'll be quiet' was Rex's biggest roasted_sincere offender."""

    def test_boundaries_are_detected(self):
        from intelligence import social_frame
        for text in [
            "I'll be quiet", "I'd rather not", "I'm gonna be quiet",
            "let's drop it", "can we change the subject", "give me a minute",
            "I'll just listen", "not in the mood", "maybe later", "I'll pass",
            "I don't want to talk about it",
            # Broadened phrasings (the _BOUNDARY_RE the eval missed):
            "we don't need to talk about that anymore", "no need to talk about it",
            "I'd prefer not to", "don't wanna talk about that", "don't want to discuss this",
            "can we move on", "let's move on", "enough about that", "stop asking",
            "that's private", "we're done talking about that",
        ]:
            with self.subTest(text=text):
                self.assertTrue(social_frame._looks_like_boundary(text))

    def test_normal_turns_are_not_boundaries(self):
        from intelligence import social_frame
        for text in [
            "stretches are helping", "I'm making a robot DJ", "I had a long day",
            "I just got back from an incredible stargazing trip", "hey rex",
            "I'd rather go to the quiet bar", "let's talk about your music", "",
            # Tricky near-misses that must NOT trip the broadened patterns:
            "move on to the next track please", "that's a private jet I'm building",
            "I need a new keyboard",
        ]:
            with self.subTest(text=text):
                self.assertFalse(social_frame._looks_like_boundary(text))

    def test_boundary_eases_roast_to_none(self):
        from intelligence import social_frame
        # person_id=None skips the DB-backed pref/boundary lookups → deterministic.
        # A boundary forces roast off even on an otherwise default ("normal") turn.
        self.assertEqual(
            social_frame._roast_level(None, "short", "default", "neutral", "none", "I'll be quiet"),
            "none",
        )
        self.assertEqual(
            social_frame._roast_level(None, "short", "default", "neutral", "none", "tell me about space"),
            "normal",
        )


class ComedyModesNoCantinaBleedTests(unittest.TestCase):
    """Rex's unprompted comedy must stay venue-neutral (the user's standing
    no-cantina-overuse intent). The old `cantina_color` mode (in ~6 rotation pools)
    literally told Rex to add Batuu/cantina flavor — the #1 unprompted bleed source.
    It was renamed to venue-neutral `dj_flair`; guard against any cantina/Batuu
    creeping back into the shipped comedy directives or line banks. (Rex's BACKSTORY
    origin and user-REQUESTED DJ/cantina patter live elsewhere and are intentional.)"""

    _BLEED = ("cantina", "batuu", "oga")

    def test_no_comedy_mode_directive_mentions_cantina(self):
        from intelligence import comedy_modes
        for key, mode in comedy_modes._MODES.items():
            blob = f"{key} {mode.label} {mode.directive}".lower()
            for word in self._BLEED:
                self.assertNotIn(
                    word, blob,
                    f"comedy mode {key!r} reintroduced '{word}' bleed: {mode.directive!r}",
                )

    def test_no_comedy_line_bank_mentions_cantina(self):
        import config
        for key, lines in getattr(config, "COMEDY_LINE_BANKS", {}).items():
            for line in lines:
                low = str(line).lower()
                for word in self._BLEED:
                    self.assertNotIn(
                        word, low,
                        f"COMEDY_LINE_BANKS[{key!r}] reintroduced '{word}' bleed: {line!r}",
                    )

    def test_dj_flair_mode_exists_and_is_in_rotation(self):
        # The renamed mode must still be a real, selectable stance (not dangling refs).
        from intelligence import comedy_modes
        self.assertIn("dj_flair", comedy_modes._MODES)
        self.assertNotIn("cantina_color", comedy_modes._MODES)

    # ── Slim-contract comedy directive: the per-turn stance must reach the LLM on the
    # default (slim) path, or every humor mechanism downstream is dead text. ──

    def test_build_slim_directive_carries_stance_for_humor_modes(self):
        from intelligence import comedy_modes
        for key in ("dry_ack", "friendly_roast", "self_own", "dj_flair",
                    "fake_system_error", "callback"):
            out = comedy_modes.build_slim_directive(comedy_modes._MODES[key])
            self.assertTrue(out, f"{key} produced no slim directive")
            self.assertIn("comedy", out.lower())
            self.assertLessEqual(len(out.split()), 40)  # stays compact

    def test_build_slim_directive_empty_for_straight_care_turn(self):
        from intelligence import comedy_modes
        self.assertEqual(
            comedy_modes.build_slim_directive(comedy_modes._MODES["straight"]), ""
        )

    def test_slim_directive_includes_recent_premise_avoid_list(self):
        from intelligence import comedy_modes
        comedy_modes._remember_line(
            "Blame my programming again.", comedy_modes._MODES["self_own"]
        )
        out = comedy_modes.build_slim_directive(comedy_modes._MODES["dry_ack"])
        self.assertIn("avoid reusing recent", out.lower())


if __name__ == "__main__":
    unittest.main()
