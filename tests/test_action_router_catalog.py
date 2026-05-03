import unittest
from unittest import mock


class ActionRouterCatalogTests(unittest.TestCase):
    def test_catalog_keys_are_stable_and_namespaced(self):
        from intelligence import action_router

        keys = [spec.key for spec in action_router.ACTION_SPECS]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("conversation.reply", keys)
        self.assertIn("conversation.repair", keys)
        for key in keys:
            self.assertRegex(key, r"^[a-z]+(?:_[a-z]+)*\.[a-z]+(?:_[a-z]+)*$")

    def test_catalog_derivatives_stay_in_sync(self):
        from intelligence import action_router

        spec_keys = {spec.key for spec in action_router.ACTION_SPECS}

        self.assertEqual(set(action_router.ACTION_CATALOG), spec_keys)
        self.assertEqual(set(action_router.ACTION_CATEGORIES), spec_keys)
        self.assertTrue(action_router.EXECUTABLE_ACTIONS.issubset(spec_keys))
        self.assertTrue(action_router.PERFORMANCE_ACTIONS.issubset(spec_keys))

    def test_humor_actions_are_executable_before_broader_performance_actions(self):
        from intelligence import action_router

        humor = {
            "humor.tell_joke",
            "humor.roast",
            "humor.free_bit",
        }
        broader_performance = {"performance.dj_bit", "performance.body_beat"}

        self.assertTrue(humor.issubset(action_router.ACTION_CATALOG))
        self.assertTrue(humor.issubset(action_router.PERFORMANCE_ACTIONS))
        self.assertTrue(humor.issubset(action_router.EXECUTABLE_ACTIONS))
        self.assertTrue(broader_performance.issubset(action_router.PERFORMANCE_ACTIONS))
        self.assertTrue(broader_performance.isdisjoint(action_router.EXECUTABLE_ACTIONS))

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

    def test_decide_short_circuits_explicit_humor_without_llm_router_call(self):
        from intelligence import action_router

        with mock.patch.object(action_router._client.chat.completions, "create") as create:
            decision = action_router.decide("tell me a joke", {})

        self.assertEqual(decision.action, "humor.tell_joke")
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
