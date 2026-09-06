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

    def test_normal_conversation_allows_optional_personal_teasing(self):
        from intelligence import social_frame
        directive = social_frame.build_directive(self._frame(allow_roast="normal"))
        self.assertIn("respond to their meaning first", directive)
        self.assertIn("ordinary answer needs no punchline", directive)
        self.assertNotIn("ROAST-LEAN", directive)

    def test_tender_roast_levels_stay_gentle(self):
        from intelligence import social_frame

        none_directive = social_frame.build_directive(self._frame(allow_roast="none"))
        self.assertIn("no roasts", none_directive.lower())
        light_directive = social_frame.build_directive(self._frame(allow_roast="light"))
        self.assertIn("light, optional tease", light_directive)

    def test_visual_directive_invites_grounded_engagement(self):
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


class GentleProbeTenderModeTest(unittest.TestCase):
    """Masked distress -> gentle_probe keeps affect 'neutral'/sensitivity 'none',
    so it must be caught as a TENDER mode: no roasts, no visual jabs — even when
    the turn contains a visual word ('look at me')."""

    def test_gentle_probe_suppresses_roast(self):
        from intelligence import social_frame
        self.assertEqual(
            social_frame._roast_level(
                None, "normal", "gentle_probe", "neutral", "none", "I'm fine"
            ),
            "none",
        )

    def test_gentle_probe_suppresses_visual_even_with_visual_word(self):
        from intelligence import social_frame
        self.assertFalse(
            social_frame._visual_allowed(
                "look at me, I'm fine", "", "normal", "gentle_probe", "neutral", "none"
            )
        )


class ComedyDeliveryProfileTests(unittest.TestCase):
    """voice_settings_for_mode: comedy STANCE -> ElevenLabs timbre, layered
    under empathy (precedence is enforced by the caller, not here)."""

    def _mode(self, key):
        from intelligence import comedy_modes

        return comedy_modes._MODES[key]

    def test_dry_ack_maps_to_deadpan_profile(self):
        import config
        from intelligence import comedy_modes

        settings = comedy_modes.voice_settings_for_mode(self._mode("dry_ack"))
        self.assertEqual(settings, config.COMEDY_DELIVERY_PROFILES["deadpan"])

    def test_friendly_roast_maps_to_smug_profile(self):
        import config
        from intelligence import comedy_modes

        settings = comedy_modes.voice_settings_for_mode(self._mode("friendly_roast"))
        self.assertEqual(settings, config.COMEDY_DELIVERY_PROFILES["smug"])

    def test_self_own_and_callback_are_deadpan(self):
        import config
        from intelligence import comedy_modes

        deadpan = config.COMEDY_DELIVERY_PROFILES["deadpan"]
        self.assertEqual(comedy_modes.voice_settings_for_mode(self._mode("self_own")), deadpan)
        self.assertEqual(comedy_modes.voice_settings_for_mode(self._mode("callback")), deadpan)

    def test_banked_callback_inherits_deadpan(self):
        import config
        from intelligence import comedy_modes

        banked = comedy_modes.with_banked_premise(
            self._mode("dry_ack"), "echo the toaster bit"
        )
        self.assertEqual(banked.key, "callback_banked")
        self.assertEqual(
            comedy_modes.voice_settings_for_mode(banked),
            config.COMEDY_DELIVERY_PROFILES["deadpan"],
        )

    def test_straight_care_mode_gets_no_profile(self):
        from intelligence import comedy_modes

        self.assertIsNone(comedy_modes.voice_settings_for_mode(self._mode("straight")))

    def test_unmapped_modes_get_no_profile(self):
        from intelligence import comedy_modes

        # dj_flair / fake_system_error have no profile yet (mischief/dj_hype later).
        self.assertIsNone(comedy_modes.voice_settings_for_mode(self._mode("dj_flair")))
        self.assertIsNone(
            comedy_modes.voice_settings_for_mode(self._mode("fake_system_error"))
        )

    def test_none_mode_is_safe(self):
        from intelligence import comedy_modes

        self.assertIsNone(comedy_modes.voice_settings_for_mode(None))

    def test_disabled_flag_suppresses_profile(self):
        from intelligence import comedy_modes

        with mock.patch("config.COMEDY_DELIVERY_PROFILES_ENABLED", False):
            self.assertIsNone(
                comedy_modes.voice_settings_for_mode(self._mode("dry_ack"))
            )

    def test_expressive_voice_off_suppresses_profile(self):
        from intelligence import comedy_modes

        # Comedy rides under the global expressive-voice switch.
        with mock.patch("config.TTS_EXPRESSIVE_VOICE_ENABLED", False):
            self.assertIsNone(
                comedy_modes.voice_settings_for_mode(self._mode("friendly_roast"))
            )

    def test_returns_fresh_dict_not_the_shared_config(self):
        import config
        from intelligence import comedy_modes

        settings = comedy_modes.voice_settings_for_mode(self._mode("dry_ack"))
        settings["stability"] = 0.999
        self.assertNotEqual(
            config.COMEDY_DELIVERY_PROFILES["deadpan"]["stability"], 0.999
        )


class ComedicPersonaTests(unittest.TestCase):
    """smug_superiority / appliance_conspiracy / dramatic_narrator recurring stances,
    each paired with a delivery profile, kept off interest/engaged-1:1 turns."""

    _PERSONAS = ("smug_superiority", "appliance_conspiracy", "dramatic_narrator")
    _SELF_ABSORBED = ("appliance_conspiracy", "dramatic_narrator")

    def setUp(self):
        from intelligence import comedy_modes

        comedy_modes.reset_recent_state()

    def _frame(self, *, purpose="answer", allow_roast="normal"):
        from intelligence import social_frame

        return social_frame.SocialFrame(
            addressee="Bret", purpose=purpose, max_words=32, max_sentences=2,
            allow_question=False, allow_roast=allow_roast, allow_visual_comment=False,
            reason="test",
        )

    def _captured_pool(self, text, frame):
        from intelligence import comedy_modes

        captured = {}

        def fake_choose(pool):
            captured["pool"] = list(pool)
            return pool[0]

        with mock.patch.object(comedy_modes, "_choose_without_stutter", side_effect=fake_choose):
            comedy_modes.select_mode(text, person_id=1, frame=frame)
        return captured["pool"]

    def test_personas_are_registered_with_directives_and_stances(self):
        from intelligence import comedy_modes

        for key in self._PERSONAS:
            self.assertIn(key, comedy_modes._MODES)
            mode = comedy_modes._MODES[key]
            self.assertTrue(comedy_modes.build_directive(mode).strip())
            self.assertTrue(comedy_modes.build_slim_directive(mode).strip())

    def test_each_persona_has_a_delivery_profile(self):
        import config
        from intelligence import comedy_modes

        profiles = config.COMEDY_DELIVERY_PROFILES
        self.assertEqual(
            comedy_modes.voice_settings_for_mode(comedy_modes._MODES["smug_superiority"]),
            profiles["smug"],
        )
        self.assertEqual(
            comedy_modes.voice_settings_for_mode(comedy_modes._MODES["appliance_conspiracy"]),
            profiles["deadpan"],
        )
        self.assertEqual(
            comedy_modes.voice_settings_for_mode(comedy_modes._MODES["dramatic_narrator"]),
            profiles["theatrical"],
        )

    def test_personas_are_in_the_explicit_humor_pool(self):
        pool = self._captured_pool("say something funny", self._frame())
        for key in self._PERSONAS:
            self.assertIn(key, pool)

    def test_self_absorbed_personas_excluded_from_interest_turn(self):
        pool = self._captured_pool("I really love astronomy", self._frame(purpose="interest"))
        for key in self._SELF_ABSORBED:
            self.assertNotIn(key, pool)

    def test_engaged_pool_allows_smug_but_not_self_absorbed_personas(self):
        # A plain engaged 1:1 turn (no explicit-humor / system / music words).
        pool = self._captured_pool("the weather is fine today", self._frame())
        self.assertIn("smug_superiority", pool)
        for key in self._SELF_ABSORBED:
            self.assertNotIn(key, pool)

    def test_appliance_conspiracy_in_system_words_pool(self):
        pool = self._captured_pool("is your processor acting up?", self._frame())
        self.assertIn("appliance_conspiracy", pool)

    def test_persona_premise_tags_enable_anti_repeat(self):
        from intelligence import comedy_modes

        for key in self._PERSONAS:
            self.assertEqual(
                comedy_modes._premise_for("whatever", comedy_modes._MODES[key]), key
            )


if __name__ == "__main__":
    unittest.main()
