import json
from types import SimpleNamespace
import unittest
from unittest import mock


class ActionRouterCatalogTests(unittest.TestCase):
    def test_catalog_keys_are_stable_and_namespaced(self):
        from intelligence import action_router

        keys = [spec.key for spec in action_router.ACTION_SPECS]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("conversation.reply", keys)
        self.assertIn("conversation.repair", keys)
        self.assertIn("identity.name_correction", keys)
        self.assertIn("memory.recent_discard", keys)
        self.assertIn("performance.mood_pose", keys)
        self.assertIn("vision.snapshot", keys)
        for key in keys:
            self.assertRegex(key, r"^[a-z]+(?:_[a-z]+)*\.[a-z]+(?:_[a-z]+)*$")

    def test_catalog_derivatives_stay_in_sync(self):
        from intelligence import action_router

        spec_keys = {spec.key for spec in action_router.ACTION_SPECS}

        self.assertEqual(set(action_router.ACTION_CATALOG), spec_keys)
        self.assertEqual(set(action_router.ACTION_CATEGORIES), spec_keys)
        self.assertTrue(action_router.EXECUTABLE_ACTIONS.issubset(spec_keys))
        self.assertTrue(action_router.PERFORMANCE_ACTIONS.issubset(spec_keys))

    def test_humor_and_performance_actions_are_executable(self):
        from intelligence import action_router

        planned = {
            "humor.tell_joke",
            "humor.roast",
            "humor.free_bit",
            "performance.dj_bit",
            "performance.body_beat",
            "performance.mood_pose",
        }

        self.assertTrue(planned.issubset(action_router.ACTION_CATALOG))
        self.assertTrue(planned.issubset(action_router.PERFORMANCE_ACTIONS))
        self.assertTrue(planned.issubset(action_router.EXECUTABLE_ACTIONS))

    def test_router_accepts_new_catalog_actions_from_llm(self):
        from intelligence import action_router

        decision = action_router._coerce_decision({
            "action": "humor.tell_joke",
            "confidence": 0.96,
            "args": {"style": "rex"},
            "requires_confirmation": False,
            "reason": "explicit joke request",
        })

        self.assertEqual(decision.action, "humor.tell_joke")
        self.assertEqual(decision.confidence, 0.96)
        self.assertEqual(decision.args["style"], "rex")

    def test_router_prompt_teaches_humor_and_performance_boundaries(self):
        from intelligence import action_router

        prompt = action_router._SYSTEM_PROMPT

        self.assertIn("tell me a joke", prompt)
        self.assertIn("Use humor.roast only for explicit roast/tease requests", prompt)
        self.assertIn("Use performance.dj_bit", prompt)
        self.assertIn("Use performance.body_beat", prompt)
        self.assertIn("Use performance.mood_pose", prompt)
        self.assertIn("Use vision.snapshot", prompt)
        self.assertIn("Use identity.name_correction", prompt)
        self.assertIn("Use memory.recent_discard", prompt)
        self.assertIn("args.body_beat", prompt)
        self.assertIn("tiny_victory_dance", prompt)

    def test_explicit_humor_classifier_routes_obvious_requests(self):
        from intelligence import action_router

        joke = action_router.classify_explicit_humor("tell me a joke")
        roast = action_router.classify_explicit_humor("roast me")
        bit = action_router.classify_explicit_humor("say something funny")

        self.assertEqual(joke.action, "humor.tell_joke")
        self.assertEqual(roast.action, "humor.roast")
        self.assertEqual(roast.args["target"], "speaker")
        self.assertEqual(bit.action, "humor.free_bit")

    def test_explicit_humor_classifier_ignores_plain_joke_mentions(self):
        from intelligence import action_router

        self.assertIsNone(action_router.classify_explicit_humor("that joke was funny"))
        self.assertIsNone(action_router.classify_explicit_humor("I ate roast beef"))

    def test_explicit_performance_classifier_routes_dj_bit_requests(self):
        from intelligence import action_router

        for text in (
            "do your DJ thing",
            "give me some cantina patter",
            "hype the room",
            "make an announcement",
        ):
            with self.subTest(text=text):
                decision = action_router.classify_explicit_performance(text)
                self.assertEqual(decision.action, "performance.dj_bit")

    def test_explicit_performance_classifier_ignores_music_playback_requests(self):
        from intelligence import action_router

        self.assertIsNone(action_router.classify_explicit_performance("play some jazz"))
        self.assertIsNone(action_router.classify_explicit_performance("put on music"))
        self.assertIsNone(action_router.classify_explicit_performance("look at the camera"))

    def test_explicit_performance_classifier_routes_body_beat_requests(self):
        from intelligence import action_router

        examples = {
            "do a victory dance": "tiny_victory_dance",
            "look suspicious": "suspicious_glance",
            "do the offended recoil": "offended_recoil",
            "do a thinking tilt": "thinking_tilt",
            "do a dramatic visor peek": "dramatic_visor_peek",
            "strike a proud DJ pose": "proud_dj_pose",
            "look surprised": "surprise_pop",
            "look disgusted": "disgust_recoil",
            "shake your head": "disagreement_shake",
            "nod yes": "agreement_nod",
        }

        for text, beat in examples.items():
            with self.subTest(text=text):
                decision = action_router.classify_explicit_performance(text)
                self.assertEqual(decision.action, "performance.body_beat")
                self.assertEqual(decision.args["body_beat"], beat)

    def test_explicit_performance_classifier_routes_mood_pose_requests(self):
        from intelligence import action_router

        examples = {
            "act embarrassed": "embarrassed",
            "look annoyed": "annoyed",
            "look proud": "proud",
            "be sad": "sad",
            "be angry": "angry",
        }

        for text, mood in examples.items():
            with self.subTest(text=text):
                decision = action_router.classify_explicit_performance(text)
                self.assertEqual(decision.action, "performance.mood_pose")
                self.assertEqual(decision.args["mood"], mood)

    def test_explicit_control_classifier_routes_safe_controls(self):
        from intelligence import action_router

        discard = action_router.classify_explicit_control("forget I said that")
        rename = action_router.classify_explicit_control("that's not Bret, I'm Daniel")
        snapshot = action_router.classify_explicit_control("remember what you see")

        self.assertEqual(discard.action, "memory.recent_discard")
        self.assertEqual(rename.action, "identity.name_correction")
        self.assertEqual(rename.args["name"], "Daniel")
        self.assertEqual(snapshot.action, "vision.snapshot")
        self.assertTrue(snapshot.requires_confirmation)

    def test_explicit_control_classifier_ignores_non_name_thats_not_phrase(self):
        from intelligence import action_router

        cases = [
            "That's not no good Oh, because you said kisses",
            "Yeah that's not happening anymore.",
        ]

        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(action_router.classify_explicit_control(text))

    def test_router_reroutes_status_retraction_from_identity_correction(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={},
            reason="misread status retraction as speaker name correction",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Yeah that's not happening anymore.",
            {},
        )

        self.assertEqual(routed.action, "event.cancel")
        self.assertEqual(
            routed.reason,
            "plan/status retraction is not an identity name correction",
        )

        decision_with_bad_name = action_router.ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={"name": "Happening Anymore"},
            reason="misread status retraction as speaker name correction",
        )

        routed = action_router._apply_context_overrides(
            decision_with_bad_name,
            "Yeah that's not happening anymore.",
            {},
        )

        self.assertEqual(routed.action, "event.cancel")

    def test_router_keeps_real_identity_correction_with_thats_not_phrase(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={"name": "Daniel"},
            reason="speaker name correction",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "That's not Bret, I'm Daniel.",
            {},
        )

        self.assertEqual(routed.action, "identity.name_correction")
        self.assertEqual(routed.args["name"], "Daniel")

    def test_dialogue_act_context_blocks_reply_misroute(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={},
            reason="misread contextual reply",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Yeah that's not happening anymore.",
            {
                "dialogue_act": {
                    "label": "answer_to_rex",
                    "blocked_actions": ["identity.name_correction"],
                },
            },
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertEqual(routed.reason, "dialogue act says utterance is a reply to Rex")

    def test_rex_opinions_are_context_not_a_routed_action(self):
        """Retired 2026-08-13: an opinion question is conversation.

        The classifier used to claim these at 0.95 and a canned answerer replied,
        with a SHA1 hash bucket inventing a stance for anything off-table. The
        authored tastes now ride the reply call as one context bullet.
        """
        from intelligence import action_router, rex_preferences

        self.assertFalse(hasattr(action_router, "classify_explicit_character_preference"))
        self.assertNotIn("character.preference_query", action_router.ACTION_CATALOG)

        for text in ("do you like music", "what's your favorite color?",
                     "do you prefer jazz or silence?"):
            lines = rex_preferences.prompt_lines(text)
            self.assertTrue(lines, text)
            self.assertTrue(lines[0].startswith("YOUR OWN TASTE"), text)

    def test_unknown_topic_gets_no_invented_stance(self):
        """The hash bucket is gone: no opinion on file means no hint at all."""
        from intelligence import rex_preferences

        for text in ("How do you feel about Daniel?",
                     "What do you think about my new haircut?",
                     "What's your favorite memory of us?"):
            self.assertEqual(rex_preferences.prompt_lines(text), [], text)

    def test_human_preferences_produce_no_taste_hint(self):
        from intelligence import rex_preferences

        self.assertEqual(rex_preferences.prompt_lines("I like music"), [])
        self.assertEqual(rex_preferences.prompt_lines("Bret likes music"), [])

    def test_opinion_question_is_not_claimed_by_any_deterministic_lane(self):
        """No regex claims it any more, so the model gets the turn.

        "music" is an _ACTION_CUE_RE token, so this still consults the LLM router
        rather than short-circuiting — the point is only that nothing deterministic
        answers it with a canned line first.
        """
        from intelligence import action_router

        payload = json.dumps({
            "action": "conversation.reply",
            "confidence": 0.4,
            "args": {},
            "reason": "opinion question",
        })
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )
        with mock.patch("intelligence.llm_compat.create", return_value=response):
            decision = action_router.decide("do you like music?", {})

        self.assertEqual(decision.action, "conversation.reply")


if __name__ == "__main__":
    unittest.main()
