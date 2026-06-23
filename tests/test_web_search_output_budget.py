"""
#26 — the web-search output-token budget is shared between the model's hidden reasoning
and the visible answer, so a longer reply could truncate. Verify the budget was raised and
reasoning is disabled for the search call so the whole budget reaches the answer.
"""

import unittest
from unittest import mock

import config
from intelligence import web_search


class WebSearchOutputBudgetTest(unittest.TestCase):
    def test_config_defaults_raised(self):
        self.assertEqual(config.WEB_SEARCH_MAX_OUTPUT_TOKENS, 1200)
        self.assertEqual(config.WEB_SEARCH_REASONING_EFFORT, "none")

    def _capture_kwargs(self, model):
        with mock.patch.object(web_search._client.responses, "create") as create:
            web_search._create_search_response(
                model, instructions="SYS", user_input="q", forced=False)
        return create.call_args.kwargs

    def test_reasoning_model_disables_reasoning_and_gets_full_budget(self):
        kw = self._capture_kwargs("gpt-5.4-mini")
        self.assertEqual(kw["max_output_tokens"], 1200)
        self.assertEqual(kw.get("reasoning"), {"effort": "none"})

    def test_non_reasoning_model_omits_reasoning(self):
        kw = self._capture_kwargs("gpt-4o-mini")
        self.assertEqual(kw["max_output_tokens"], 1200)
        self.assertNotIn("reasoning", kw)


if __name__ == "__main__":
    unittest.main()
