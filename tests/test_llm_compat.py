"""Unit tests for intelligence/llm_compat — the OpenAI model-compatibility shim.

Pure param-translation tests, no network. These lock the contract that wiring a call
site through the shim is BEHAVIOR-NEUTRAL for gpt-4o-mini and produces the correct
GPT-5-family parameter set for a reasoning model. See docs/gpt-5_4_mini.md.
"""

import unittest
from unittest import mock

import config
from intelligence import llm_compat as LC


class IsReasoningModelTest(unittest.TestCase):
    def test_gpt5_and_o_series_are_reasoning(self):
        for m in ("gpt-5.4-mini", "gpt-5-mini", "gpt-5", "GPT-5.4-MINI", "o1", "o3-mini", "o4-mini"):
            self.assertTrue(LC.is_reasoning_model(m), m)

    def test_classic_chat_models_are_not_reasoning(self):
        for m in ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "", None):
            self.assertFalse(LC.is_reasoning_model(m), m)


class ConversationModelTest(unittest.TestCase):
    def test_defaults_to_llm_model(self):
        with mock.patch.object(config, "LLM_CONVERSATION_MODEL", config.LLM_MODEL):
            self.assertEqual(LC.conversation_model(), config.LLM_MODEL)

    def test_override_is_respected(self):
        with mock.patch.object(config, "LLM_CONVERSATION_MODEL", "gpt-5.4-mini"):
            self.assertEqual(LC.conversation_model(), "gpt-5.4-mini")

    def test_empty_override_falls_back_to_llm_model(self):
        with mock.patch.object(config, "LLM_CONVERSATION_MODEL", None):
            self.assertEqual(LC.conversation_model(), config.LLM_MODEL)


class PrepareParamsClassicTest(unittest.TestCase):
    """gpt-4o-mini path must be a pure pass-through (behavior-neutral wiring)."""

    def test_classic_passes_max_tokens_and_temperature_unchanged(self):
        p = LC.prepare_chat_params(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
            max_tokens=60, temperature=0.0, stream=True, timeout=18.0,
        )
        self.assertEqual(p["model"], "gpt-4o-mini")
        self.assertEqual(p["max_tokens"], 60)
        self.assertEqual(p["temperature"], 0.0)
        self.assertTrue(p["stream"])
        self.assertEqual(p["timeout"], 18.0)
        # No GPT-5-only keys leak into a classic call.
        self.assertNotIn("max_completion_tokens", p)
        self.assertNotIn("reasoning_effort", p)
        self.assertNotIn("verbosity", p)

    def test_classic_ignores_reasoning_config(self):
        with mock.patch.object(config, "LLM_REASONING_EFFORT", "low"), \
             mock.patch.object(config, "LLM_VERBOSITY", "high"):
            p = LC.prepare_chat_params(model="gpt-4o-mini", messages=[], max_tokens=10)
        self.assertNotIn("reasoning_effort", p)
        self.assertNotIn("verbosity", p)

    def test_response_format_passes_through(self):
        p = LC.prepare_chat_params(
            model="gpt-4o-mini", messages=[], max_tokens=80, temperature=0,
            response_format={"type": "json_object"},
        )
        self.assertEqual(p["response_format"], {"type": "json_object"})


class PrepareParamsReasoningTest(unittest.TestCase):
    """gpt-5.4-mini path: renames the token cap and handles the restricted temperature."""

    def setUp(self):
        # Default the scaffolding to its shipped (off) values for each test.
        self._patches = [
            mock.patch.object(config, "LLM_REASONING_EFFORT", None),
            mock.patch.object(config, "LLM_VERBOSITY", None),
            mock.patch.object(config, "LLM_GPT5_PASS_TEMPERATURE", False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_max_tokens_renamed_to_max_completion_tokens(self):
        p = LC.prepare_chat_params(model="gpt-5.4-mini", messages=[], max_tokens=60)
        self.assertEqual(p["max_completion_tokens"], 60)
        self.assertNotIn("max_tokens", p)

    def test_temperature_dropped_by_default(self):
        p = LC.prepare_chat_params(model="gpt-5.4-mini", messages=[], max_tokens=60, temperature=0.0)
        self.assertNotIn("temperature", p)

    def test_temperature_kept_when_flag_enabled(self):
        with mock.patch.object(config, "LLM_GPT5_PASS_TEMPERATURE", True):
            p = LC.prepare_chat_params(model="gpt-5.4-mini", messages=[], max_tokens=60, temperature=0.2)
        self.assertEqual(p["temperature"], 0.2)

    def test_reasoning_effort_from_config(self):
        with mock.patch.object(config, "LLM_REASONING_EFFORT", "none"):
            p = LC.prepare_chat_params(model="gpt-5.4-mini", messages=[], max_tokens=60)
        self.assertEqual(p["reasoning_effort"], "none")

    def test_explicit_reasoning_effort_overrides_config(self):
        with mock.patch.object(config, "LLM_REASONING_EFFORT", "medium"):
            p = LC.prepare_chat_params(
                model="gpt-5.4-mini", messages=[], max_tokens=60, reasoning_effort="minimal",
            )
        self.assertEqual(p["reasoning_effort"], "minimal")

    def test_verbosity_from_config(self):
        with mock.patch.object(config, "LLM_VERBOSITY", "low"):
            p = LC.prepare_chat_params(model="gpt-5.4-mini", messages=[], max_tokens=60)
        self.assertEqual(p["verbosity"], "low")

    def test_stream_and_response_format_still_pass_through(self):
        p = LC.prepare_chat_params(
            model="gpt-5.4-mini", messages=[], max_tokens=60, stream=True,
            response_format={"type": "json_object"}, timeout=18.0,
        )
        self.assertTrue(p["stream"])
        self.assertEqual(p["response_format"], {"type": "json_object"})
        self.assertEqual(p["timeout"], 18.0)


class CreateTest(unittest.TestCase):
    def test_create_calls_client_with_translated_params(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = "RESPONSE"
        with mock.patch.object(config, "LLM_GPT5_PASS_TEMPERATURE", False):
            out = LC.create(
                client, model="gpt-5.4-mini", messages=[{"role": "user", "content": "x"}],
                max_tokens=40, temperature=0.8,
            )
        self.assertEqual(out, "RESPONSE")
        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["max_completion_tokens"], 40)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_create_is_passthrough_for_classic_model(self):
        client = mock.Mock()
        LC.create(client, model="gpt-4o-mini", messages=[], max_tokens=40, temperature=0.8)
        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["max_tokens"], 40)
        self.assertEqual(kwargs["temperature"], 0.8)


if __name__ == "__main__":
    unittest.main()
