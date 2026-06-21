"""Unit tests for intelligence/web_search and the interaction search branch.

No network: the OpenAI Responses call and the gate classifier are mocked. These lock
the trigger policy (explicit phrases always search; autonomous needs a question +
currentness keyword, then the optional gate), the Responses API call shape (hosted
web_search tool, persona instructions, reasoning knob only on reasoning models), and
the interaction hook (stall line + spoken answer, graceful fall-through, kill switch).
"""

import unittest
from unittest import mock

import config
from intelligence import web_search


def _resp(output_text="", output=None):
    r = mock.Mock()
    r.output_text = output_text
    r.output = output if output is not None else []
    return r


class ShouldSearchTest(unittest.TestCase):
    def setUp(self):
        # Deterministic config for the policy under test.
        self._patches = [
            mock.patch.object(config, "WEB_SEARCH_TRIGGER_PHRASES",
                              ["look that up", "search the web", "what's the latest on"]),
            mock.patch.object(config, "WEB_SEARCH_AUTONOMOUS_KEYWORDS",
                              ["latest", "today", "score", "who won"]),
            mock.patch.object(config, "WEB_SEARCH_AUTONOMOUS_ENABLED", True),
            mock.patch.object(config, "WEB_SEARCH_AUTONOMOUS_GATE_ENABLED", True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_explicit_phrase_forces_search(self):
        d = web_search.should_search("Hey, can you look that up for me?")
        self.assertTrue(d.triggered)
        self.assertTrue(d.forced)
        self.assertTrue(d.reason.startswith("explicit:"))

    def test_explicit_works_even_when_autonomous_disabled(self):
        with mock.patch.object(config, "WEB_SEARCH_AUTONOMOUS_ENABLED", False):
            d = web_search.should_search("search the web for ticket prices")
        self.assertTrue(d.triggered)
        self.assertTrue(d.forced)

    def test_plain_chitchat_does_not_trigger(self):
        for text in ("How are you doing today buddy?", "Tell me a joke.", "I like jazz."):
            # "today" is a keyword but these are not info-questions / are handled by gate.
            with mock.patch.object(web_search, "_gate_says_needs_search", return_value=False):
                d = web_search.should_search(text)
            self.assertFalse(d.triggered, text)

    def test_autonomous_keyword_only_when_gate_disabled(self):
        with mock.patch.object(config, "WEB_SEARCH_AUTONOMOUS_GATE_ENABLED", False):
            d = web_search.should_search("What's the latest news on the Mars mission?")
        self.assertTrue(d.triggered)
        self.assertFalse(d.forced)
        self.assertEqual(d.reason, "autonomous:keyword")

    def test_autonomous_gate_confirms(self):
        with mock.patch.object(web_search, "_gate_says_needs_search", return_value=True) as gate:
            d = web_search.should_search("Who won the game today?")
        self.assertTrue(d.triggered)
        self.assertEqual(d.reason, "autonomous:gate")
        gate.assert_called_once()

    def test_autonomous_gate_rejects(self):
        with mock.patch.object(web_search, "_gate_says_needs_search", return_value=False):
            d = web_search.should_search("Who won the game today?")
        self.assertFalse(d.triggered)

    def test_keyword_without_question_is_filtered_before_gate(self):
        # "today" present but not question-shaped → prefilter drops it; gate not called.
        with mock.patch.object(web_search, "_gate_says_needs_search") as gate:
            d = web_search.should_search("I really loved today.")
        self.assertFalse(d.triggered)
        gate.assert_not_called()

    def test_question_without_keyword_is_filtered_before_gate(self):
        with mock.patch.object(web_search, "_gate_says_needs_search") as gate:
            d = web_search.should_search("What is your favorite color?")
        self.assertFalse(d.triggered)
        gate.assert_not_called()

    def test_empty_text(self):
        self.assertFalse(web_search.should_search("").triggered)
        self.assertFalse(web_search.should_search("   ").triggered)


class GateClassifierTest(unittest.TestCase):
    def test_yes_parses_true(self):
        fake = mock.Mock()
        fake.choices = [mock.Mock(message=mock.Mock(content="Yes"))]
        with mock.patch.object(web_search._client.chat.completions, "create", return_value=fake) as create:
            self.assertTrue(web_search._gate_says_needs_search("what's the score"))
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], config.WEB_SEARCH_GATE_MODEL)
        self.assertEqual(kwargs["max_tokens"], 3)

    def test_no_parses_false(self):
        fake = mock.Mock()
        fake.choices = [mock.Mock(message=mock.Mock(content="no"))]
        with mock.patch.object(web_search._client.chat.completions, "create", return_value=fake):
            self.assertFalse(web_search._gate_says_needs_search("who are you"))

    def test_error_is_false(self):
        with mock.patch.object(web_search._client.chat.completions, "create", side_effect=RuntimeError("boom")):
            self.assertFalse(web_search._gate_says_needs_search("anything"))


class PickStallLineTest(unittest.TestCase):
    def test_returns_configured_line(self):
        with mock.patch.object(config, "WEB_SEARCH_STALL_LINES", ["A", "B", "C"]):
            self.assertIn(web_search.pick_stall_line(), {"A", "B", "C"})

    def test_no_immediate_repeat(self):
        with mock.patch.object(config, "WEB_SEARCH_STALL_LINES", ["A", "B"]):
            web_search._last_stall_line = None
            first = web_search.pick_stall_line()
            second = web_search.pick_stall_line()
            self.assertNotEqual(first, second)

    def test_empty_pool_returns_blank(self):
        with mock.patch.object(config, "WEB_SEARCH_STALL_LINES", []):
            self.assertEqual(web_search.pick_stall_line(), "")


class AnswerTest(unittest.TestCase):
    def setUp(self):
        self._instr = mock.patch.object(web_search, "_build_instructions", return_value="SYS")
        self._instr.start()

    def tearDown(self):
        self._instr.stop()

    def test_happy_path_uses_hosted_tool_and_persona(self):
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create",
                               return_value=_resp(output_text="  The latest model is X.  ")) as create:
            result = web_search.answer("what's the latest", person_id=1, forced=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "The latest model is X.")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertEqual(kwargs["instructions"], "SYS")
        self.assertEqual(kwargs["tools"], [{"type": "web_search"}])
        self.assertEqual(kwargs["tool_choice"], "auto")
        # gpt-4o-mini is not a reasoning model → no reasoning knob.
        self.assertNotIn("reasoning", kwargs)

    def test_forced_uses_required_tool_choice_and_marks_input(self):
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create",
                               return_value=_resp(output_text="Answer.")) as create:
            result = web_search.answer("look that up", person_id=None, forced=True)
        self.assertTrue(result.ok)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "required")
        self.assertIn("web search", kwargs["input"].lower())

    def test_reasoning_model_gets_effort(self):
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-5.4-mini"), \
             mock.patch.object(config, "WEB_SEARCH_REASONING_EFFORT", "low"), \
             mock.patch.object(web_search._client.responses, "create",
                               return_value=_resp(output_text="Answer.")) as create:
            web_search.answer("what's the latest", person_id=1)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})

    def test_api_error_returns_not_ok(self):
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create", side_effect=RuntimeError("boom")):
            result = web_search.answer("x", person_id=1)
        self.assertFalse(result.ok)

    def test_empty_output_returns_not_ok(self):
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create", return_value=_resp(output_text="")):
            result = web_search.answer("x", person_id=1)
        self.assertFalse(result.ok)


class StripLinksTest(unittest.TestCase):
    def test_removes_bare_url(self):
        out = web_search.strip_links("The launch is Friday. https://example.com/very/long/path here.")
        self.assertNotIn("http", out)
        self.assertNotIn("example.com", out)
        self.assertIn("The launch is Friday.", out)

    def test_removes_www_url(self):
        self.assertNotIn("www.", web_search.strip_links("See www.nasa.gov/news for more."))

    def test_markdown_link_keeps_label(self):
        out = web_search.strip_links("The [Lakers](https://nba.com/lakers) won.")
        self.assertEqual(out, "The Lakers won.")

    def test_removes_bare_domain(self):
        out = web_search.strip_links("According to reuters.com the vote passed.")
        self.assertNotIn("reuters.com", out)
        self.assertIn("the vote passed", out)

    def test_removes_source_parenthetical(self):
        out = web_search.strip_links("It rained today (source: weather.com).")
        self.assertNotIn("weather.com", out)
        self.assertNotIn("source", out.lower())
        self.assertTrue(out.startswith("It rained today"))

    def test_strips_footnote_markers(self):
        self.assertEqual(web_search.strip_links("True enough[1]."), "True enough.")

    def test_plain_text_untouched(self):
        for s in ("I can't see you.", "That was a great landing.", "Who knows."):
            self.assertEqual(web_search.strip_links(s), s)

    def test_empty(self):
        self.assertEqual(web_search.strip_links(""), "")


class AnswerStripsLinksTest(unittest.TestCase):
    def setUp(self):
        self._instr = mock.patch.object(web_search, "_build_instructions", return_value="SYS")
        self._instr.start()

    def tearDown(self):
        self._instr.stop()

    def test_answer_text_has_no_links(self):
        out_text = "The rover landed Monday. Details at https://mars.nasa.gov/news/9999."
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create",
                               return_value=_resp(output_text=out_text)):
            result = web_search.answer("what's the latest on mars", person_id=1)
        self.assertTrue(result.ok)
        self.assertNotIn("http", result.text)
        self.assertNotIn("nasa.gov", result.text)
        self.assertIn("The rover landed Monday.", result.text)

    def test_bare_link_answer_falls_through(self):
        # If the whole "answer" was just a link, stripping empties it → not ok.
        with mock.patch.object(config, "WEB_SEARCH_MODEL", "gpt-4o-mini"), \
             mock.patch.object(web_search._client.responses, "create",
                               return_value=_resp(output_text="https://example.com/x")):
            result = web_search.answer("x", person_id=1)
        self.assertFalse(result.ok)


class RecentSearchMarkerTest(unittest.TestCase):
    def setUp(self):
        web_search._recent_search = None

    def tearDown(self):
        web_search._recent_search = None

    def test_topic_extraction(self):
        cases = {
            "I'd like you to search the web about Star Trek Voyager": "Star Trek Voyager",
            "can you look up the James Webb telescope": "the James Webb telescope",
            "search the web for tonight's Lakers score": "tonight's Lakers score",
            "what's the latest on the Mars mission?": "the Mars mission",
        }
        for query, expected in cases.items():
            self.assertEqual(web_search._search_topic(query), expected, query)

    def test_note_then_recent(self):
        web_search.note_search("search the web about Star Trek Voyager")
        self.assertEqual(web_search.recent_search(), "Star Trek Voyager")

    def test_expired_window_returns_none(self):
        web_search.note_search("look up the weather on Mars")
        # Age the marker past the window.
        web_search._recent_search["at"] -= 10_000
        with mock.patch.object(config, "WEB_SEARCH_FOLLOWUP_WINDOW_SECS", 120.0):
            self.assertIsNone(web_search.recent_search())

    def test_disabled_flag(self):
        web_search.note_search("look up X")
        with mock.patch.object(config, "WEB_SEARCH_FOLLOWUP_INQUISITIVE_ENABLED", False):
            self.assertIsNone(web_search.recent_search())

    def test_clear(self):
        web_search.note_search("look up X")
        web_search.clear_recent_search()
        self.assertIsNone(web_search.recent_search())

    def test_clear_min_age_guard_keeps_fresh_marker(self):
        web_search.note_search("look up X")          # just set, at = now
        web_search.clear_recent_search(min_age_secs=3.0)
        self.assertEqual(web_search.recent_search(), "X")  # not wiped same-turn

    def test_no_marker_recent_is_none(self):
        self.assertIsNone(web_search.recent_search())


class ProactiveSteerTest(unittest.TestCase):
    """conversation_agenda.with_proactive_directive flips proactive lull lines to be
    inquisitive while a recent search is armed."""

    def setUp(self):
        from intelligence import conversation_agenda
        self.ca = conversation_agenda

    def test_steer_present_after_search(self):
        with mock.patch.object(web_search, "recent_search", return_value="Star Trek Voyager"):
            out = self.ca.with_proactive_directive("BASE PROMPT", "small_talk")
        self.assertIn("POST-SEARCH FOLLOW-UP", out)
        self.assertIn("Star Trek Voyager", out)
        self.assertIn("INQUISITIVE", out)
        self.assertIn("BASE PROMPT", out)

    def test_no_steer_without_search(self):
        with mock.patch.object(web_search, "recent_search", return_value=None):
            out = self.ca.with_proactive_directive("BASE PROMPT", "small_talk")
        self.assertNotIn("POST-SEARCH FOLLOW-UP", out)
        self.assertIn("BASE PROMPT", out)


class InteractionHookTest(unittest.TestCase):
    """The _maybe_web_search_reply branch in interaction.py."""

    @classmethod
    def setUpClass(cls):
        from intelligence import interaction
        cls.interaction = interaction

    def _patch_common(self):
        # Always speakable, no interruption; trace is a ContextVar that defaults to None.
        return [
            mock.patch.object(self.interaction, "_can_speak", return_value=True),
            mock.patch.object(self.interaction, "_speak_blocking", return_value=True),
            mock.patch.object(self.interaction, "_apply_post_tts_handoff"),
            mock.patch.object(self.interaction.speech_queue, "enqueue"),
            mock.patch.object(config, "WEB_SEARCH_ENABLED", True),
        ]

    def test_kill_switch_returns_none(self):
        with mock.patch.object(config, "WEB_SEARCH_ENABLED", False), \
             mock.patch.object(web_search, "should_search") as ss:
            out = self.interaction._maybe_web_search_reply("look that up", person_id=1)
        self.assertIsNone(out)
        ss.assert_not_called()

    def test_not_triggered_returns_none(self):
        patches = self._patch_common()
        for p in patches:
            p.start()
        try:
            with mock.patch.object(web_search, "should_search",
                                   return_value=web_search.SearchDecision(False, False, "")):
                out = self.interaction._maybe_web_search_reply("hello", person_id=1)
            self.assertIsNone(out)
            self.interaction.speech_queue.enqueue.assert_not_called()
        finally:
            for p in patches:
                p.stop()

    def test_triggered_speaks_stall_and_answer(self):
        patches = self._patch_common()
        for p in patches:
            p.start()
        try:
            with mock.patch.object(web_search, "should_search",
                                   return_value=web_search.SearchDecision(True, True, "explicit:look that up")), \
                 mock.patch.object(web_search, "pick_stall_line", return_value="Let me check the archives."), \
                 mock.patch.object(web_search, "answer",
                                   return_value=web_search.SearchResult(True, "It is 42.", [])):
                out = self.interaction._maybe_web_search_reply("look that up", person_id=1)
            self.assertEqual(out, "It is 42.")
            # Stall line was enqueued (non-blocking), answer was spoken via _speak_blocking.
            self.interaction.speech_queue.enqueue.assert_called_once()
            self.assertEqual(self.interaction.speech_queue.enqueue.call_args.args[0],
                             "Let me check the archives.")
            self.interaction._speak_blocking.assert_called_once()
            self.assertEqual(self.interaction._speak_blocking.call_args.args[0], "It is 42.")
        finally:
            for p in patches:
                p.stop()

    def test_triggered_but_no_result_falls_through(self):
        patches = self._patch_common()
        for p in patches:
            p.start()
        try:
            with mock.patch.object(web_search, "should_search",
                                   return_value=web_search.SearchDecision(True, False, "autonomous:gate")), \
                 mock.patch.object(web_search, "pick_stall_line", return_value="One sec."), \
                 mock.patch.object(web_search, "answer",
                                   return_value=web_search.SearchResult(False, "", [])):
                out = self.interaction._maybe_web_search_reply("who won today", person_id=1)
            # No usable result → None so the normal reply path takes over.
            self.assertIsNone(out)
            self.interaction._speak_blocking.assert_not_called()
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
