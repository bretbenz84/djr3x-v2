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
        self.assertIn("character.preference_query", keys)
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
        self.assertIn("Use character.preference_query", prompt)
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

    def test_explicit_character_preference_classifier_routes_rex_opinions(self):
        from intelligence import action_router

        like = action_router.classify_explicit_character_preference("do you like music")
        favorite = action_router.classify_explicit_character_preference("what's your favorite color?")
        compare = action_router.classify_explicit_character_preference("do you prefer jazz or silence?")

        self.assertEqual(like.action, "character.preference_query")
        self.assertEqual(like.args["topic"], "music")
        self.assertEqual(like.args["verb"], "like")
        self.assertEqual(favorite.args["mode"], "favorite")
        self.assertEqual(compare.args["mode"], "compare")
        self.assertEqual(compare.args["options"], ["jazz", "silence"])

    def test_explicit_character_preference_classifier_ignores_human_preferences(self):
        from intelligence import action_router

        self.assertIsNone(action_router.classify_explicit_character_preference("I like music"))
        self.assertIsNone(action_router.classify_explicit_character_preference("Bret likes music"))

    def test_body_beat_llm_decision_requires_known_beat(self):
        from intelligence import action_router

        valid = action_router._coerce_decision({
            "action": "performance.body_beat",
            "confidence": 0.99,
            "args": {"gesture": "victory dance"},
            "reason": "explicit physical request",
        })
        invalid = action_router._coerce_decision({
            "action": "performance.body_beat",
            "confidence": 0.99,
            "args": {"gesture": "spin the dangerous servo"},
            "reason": "unknown physical request",
        })

        self.assertEqual(valid.args["body_beat"], "tiny_victory_dance")
        self.assertEqual(valid.confidence, 0.99)
        self.assertLess(invalid.confidence, 0.85)

    def test_mood_pose_llm_decision_requires_known_mood(self):
        from intelligence import action_router

        valid = action_router._coerce_decision({
            "action": "performance.mood_pose",
            "confidence": 0.99,
            "args": {"emotion": "bashful"},
            "reason": "explicit mood pose",
        })
        invalid = action_router._coerce_decision({
            "action": "performance.mood_pose",
            "confidence": 0.99,
            "args": {"emotion": "danger servo"},
            "reason": "unknown mood pose",
        })

        self.assertEqual(valid.args["mood"], "embarrassed")
        self.assertEqual(valid.confidence, 0.99)
        self.assertLess(invalid.confidence, 0.85)

    def test_character_preference_llm_decision_requires_topic_or_options(self):
        from intelligence import action_router

        valid = action_router._coerce_decision({
            "action": "character.preference_query",
            "confidence": 0.99,
            "args": {"topic": "music", "verb": "like"},
            "reason": "asks Rex preference",
        })
        invalid = action_router._coerce_decision({
            "action": "character.preference_query",
            "confidence": 0.99,
            "args": {},
            "reason": "missing topic",
        })

        self.assertEqual(valid.action, "character.preference_query")
        self.assertEqual(valid.confidence, 0.99)
        self.assertLess(invalid.confidence, 0.85)

    def test_vision_snapshot_requires_confirmation(self):
        from intelligence import action_router

        decision = action_router._coerce_decision({
            "action": "vision.snapshot",
            "confidence": 0.99,
            "args": {"scope": "current_view"},
            "requires_confirmation": False,
            "reason": "remember current scene",
        })

        self.assertTrue(decision.requires_confirmation)

    def test_vision_snapshot_downgrades_human_photo_plan(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="vision.snapshot",
            confidence=0.90,
            args={"scope": "current_view"},
            requires_confirmation=True,
            reason="privacy-sensitive current view request",
        )

        downgraded = action_router._apply_context_overrides(
            decision,
            "I wanna take a picture of the Leo triplet",
            {},
        )

        self.assertEqual(downgraded.action, "conversation.reply")
        self.assertFalse(downgraded.requires_confirmation)
        self.assertLess(downgraded.confidence, 0.85)

    def test_vision_snapshot_keeps_direct_rex_memory_request(self):
        from intelligence import action_router

        decision = action_router.classify_explicit_control(
            "I want you to remember what you see"
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "vision.snapshot")
        self.assertTrue(decision.requires_confirmation)

        self.assertIsNone(
            action_router.classify_explicit_control("I'm going to save this view")
        )

    def test_decide_downgrades_llm_snapshot_for_human_photo_plan(self):
        from intelligence import action_router

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({
                            "action": "vision.snapshot",
                            "confidence": 0.9,
                            "args": {"scope": "current_view"},
                            "requires_confirmation": True,
                            "reason": "mentions taking a picture",
                        })
                    )
                )
            ]
        )

        with mock.patch.object(
            action_router._client.chat.completions,
            "create",
            return_value=response,
        ):
            decision = action_router.decide(
                "I wanna take a picture of the Leo triplet",
                {},
            )

        self.assertEqual(decision.action, "conversation.reply")
        self.assertFalse(decision.requires_confirmation)

    def test_decide_short_circuits_explicit_humor_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("tell me a joke", {})

        self.assertEqual(decision.action, "humor.tell_joke")
        create.assert_not_called()

    def test_decide_short_circuits_explicit_dj_bit_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("do your DJ thing", {})

        self.assertEqual(decision.action, "performance.dj_bit")
        create.assert_not_called()

    def test_decide_short_circuits_explicit_body_beat_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("do a victory dance", {})

        self.assertEqual(decision.action, "performance.body_beat")
        self.assertEqual(decision.args["body_beat"], "tiny_victory_dance")
        create.assert_not_called()

    def test_decide_short_circuits_explicit_control_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("forget I said that", {})

        self.assertEqual(decision.action, "memory.recent_discard")
        create.assert_not_called()

    def test_decide_short_circuits_rex_preference_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("do you like music?", {})

        self.assertEqual(decision.action, "character.preference_query")
        self.assertEqual(decision.args["topic"], "music")
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
