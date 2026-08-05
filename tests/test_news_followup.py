"""News follow-up elaboration (field 2026-08-04 23:09).

Rex offered an interest-news story ("did you catch the 2026 Vegas Trek
thing...?"), the person asked "No. Can you tell me more about that?", and the
turn died: the follow-up bound as answer_to_rex (router skipped), the reply
model hallucinated a TEXT-shaped tool call ('[web_search]\n{"query":...}') that
was dropped as an unspeakable stream tail, nothing was spoken, and the lull
impulse changed the subject.

These tests lock the fix's three layers:
  * _news_followup_story — deterministic detection that a turn is asking to
    hear more about the story Rex just offered.
  * _maybe_web_search_reply(news_story=...) — a FORCED search grounded on the
    cached headline/summary, not the bare "tell me more" words.
  * _parse_text_tool_call — a text-shaped live-tool call is rescued and
    dispatched instead of being dropped to silence.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import dialogue_act, interaction, web_search
from tests._lean_impulse_state import reset_impulse_state

_STORY = {
    "headline": "Star Trek Las Vegas 60th anniversary convention runs August 6-9",
    "summary": "The Las Vegas convention bills itself as the franchise's "
               "60th-anniversary showpiece with cast panels and screenings.",
    "topic": "star trek",
    "at": 0.0,  # stamped fresh in _stash()
}


class NewsFollowupDetectionTest(unittest.TestCase):
    def setUp(self):
        reset_impulse_state(self)
        dialogue_act.clear()
        self.addCleanup(dialogue_act.clear)

    def _stash(self, age_secs=5.0):
        interaction._last_news_story_offered = dict(
            _STORY, at=time.monotonic() - age_secs
        )

    def test_tell_me_more_matches_without_frame(self):
        self._stash()
        out = interaction._news_followup_story("No. Can you tell me more about that?")
        self.assertIsNotNone(out)
        self.assertEqual(out["headline"], _STORY["headline"])

    def test_what_is_that_about_matches(self):
        self._stash()
        self.assertIsNotNone(interaction._news_followup_story("What's that about?"))

    def test_no_stash_returns_none(self):
        interaction._last_news_story_offered = None
        self.assertIsNone(interaction._news_followup_story("Tell me more about that."))

    def test_stale_stash_returns_none(self):
        window = float(getattr(config, "NEWS_FOLLOWUP_WINDOW_SECS", 240.0))
        self._stash(age_secs=window + 30.0)
        self.assertIsNone(interaction._news_followup_story("Tell me more about that."))

    def test_kill_switch_returns_none(self):
        self._stash()
        with mock.patch.object(config, "NEWS_FOLLOWUP_ELABORATION_ENABLED", False):
            self.assertIsNone(
                interaction._news_followup_story("Tell me more about that.")
            )

    def test_referential_question_needs_active_news_frame(self):
        self._stash()
        # No frame registered → the loose "when is that?" shape must NOT bind.
        self.assertIsNone(interaction._news_followup_story("When is that happening?"))
        # With the news offer as the active frame (topic carries the headline),
        # the same question binds.
        dialogue_act.note_rex_turn(
            "Bret, did you catch the Vegas Trek thing?",
            source="lean_impulse",
            topic=_STORY["headline"],
        )
        self.assertIsNotNone(interaction._news_followup_story("When is that happening?"))

    def test_headline_word_question_matches_with_frame(self):
        self._stash()
        dialogue_act.note_rex_turn(
            "Bret, did you catch the Vegas Trek thing?",
            source="lean_impulse",
            topic=_STORY["headline"],
        )
        self.assertIsNotNone(
            interaction._news_followup_story("Is the convention in Vegas?")
        )

    def test_unrelated_statement_does_not_match(self):
        self._stash()
        dialogue_act.note_rex_turn(
            "Bret, did you catch the Vegas Trek thing?",
            source="lean_impulse",
            topic=_STORY["headline"],
        )
        self.assertIsNone(interaction._news_followup_story("I had pasta for dinner."))

    def test_newer_frame_disarms_referential_branch(self):
        self._stash()
        dialogue_act.note_rex_turn(
            "Bret, did you catch the Vegas Trek thing?",
            source="lean_impulse",
            topic=_STORY["headline"],
        )
        # Rex spoke again about something else — the loose referential shape
        # must stop hijacking, while explicit "tell me more" still works.
        dialogue_act.note_rex_turn("What are you up to tonight?", source="question")
        self.assertIsNone(interaction._news_followup_story("Can you believe that?"))
        self.assertIsNotNone(
            interaction._news_followup_story("Tell me more about the story.")
        )


class NewsFollowupSearchTest(unittest.TestCase):
    """_maybe_web_search_reply with news_story: forced, grounded, topic-noted."""

    def setUp(self):
        interaction._interrupted.clear()

    def test_forced_search_grounded_on_story(self):
        story = dict(_STORY, at=time.monotonic())
        result = web_search.SearchResult(True, "Here's the scoop.", ["u"])
        with mock.patch.object(config, "WEB_SEARCH_ENABLED", True), \
             mock.patch.object(interaction, "_can_speak", return_value=True), \
             mock.patch.object(web_search, "should_search") as should, \
             mock.patch.object(web_search, "pick_stall_line", return_value=""), \
             mock.patch.object(web_search, "answer", return_value=result) as answer, \
             mock.patch.object(web_search, "note_search") as note, \
             mock.patch.object(interaction, "_apply_post_tts_handoff"), \
             mock.patch.object(interaction, "_speak_blocking"):
            out = interaction._maybe_web_search_reply(
                "No. Can you tell me more about that?", person_id=1, news_story=story
            )
        self.assertEqual(out, "Here's the scoop.")
        # The trigger policy is bypassed — a news follow-up ALWAYS searches.
        should.assert_not_called()
        answer.assert_called_once()
        composed = answer.call_args.args[0]
        self.assertIn("tell me more about that", composed)
        self.assertIn(_STORY["headline"], composed)
        self.assertIn("60th-anniversary showpiece", composed)
        self.assertTrue(answer.call_args.kwargs.get("forced"))
        # The follow-up topic marker carries the story, not the composed blob.
        note.assert_called_once_with(_STORY["headline"])

    def test_search_failure_falls_through_to_normal_reply(self):
        story = dict(_STORY, at=time.monotonic())
        with mock.patch.object(config, "WEB_SEARCH_ENABLED", True), \
             mock.patch.object(interaction, "_can_speak", return_value=True), \
             mock.patch.object(web_search, "pick_stall_line", return_value=""), \
             mock.patch.object(web_search, "answer",
                               return_value=web_search.SearchResult(False, "", [])), \
             mock.patch.object(interaction, "_speak_blocking") as speak:
            out = interaction._maybe_web_search_reply(
                "Tell me more about that.", person_id=1, news_story=story
            )
        self.assertIsNone(out)   # caller proceeds to the summary-grounded reply
        speak.assert_not_called()


class TextToolCallRescueTest(unittest.TestCase):
    """A whole-reply text-shaped tool call is parsed and dispatched, not dropped."""

    def test_web_search_text_call_resolves(self):
        raw = '[web_search]\n{"query":"2026 Vegas Trek 60th anniversary"}'
        parsed = interaction._parse_text_tool_call(raw)
        self.assertIsNotNone(parsed)
        action, args = parsed
        self.assertEqual(action, "web.search")
        self.assertEqual(args, {"query": "2026 Vegas Trek 60th anniversary"})

    def test_dotted_name_also_resolves(self):
        parsed = interaction._parse_text_tool_call('[web.search] {"query":"x"}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0], "web.search")

    def test_audio_tag_with_prose_does_not_match(self):
        self.assertIsNone(
            interaction._parse_text_tool_call("[curious] Hey, did you hear the news?")
        )

    def test_bare_audio_tag_resolves_to_no_tool(self):
        self.assertIsNone(interaction._parse_text_tool_call("[sarcastic]"))

    def test_plain_prose_does_not_match(self):
        self.assertIsNone(
            interaction._parse_text_tool_call("The convention runs August 6-9.")
        )

    def test_non_live_tool_is_not_dispatched(self):
        with mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", False):
            self.assertIsNone(
                interaction._parse_text_tool_call('[web_search] {"query":"x"}')
            )

    def test_malformed_json_still_dispatches_with_empty_args(self):
        parsed = interaction._parse_text_tool_call("[web_search] {broken json}")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed, ("web.search", {}))


if __name__ == "__main__":
    unittest.main()
