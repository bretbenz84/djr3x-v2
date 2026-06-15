"""
Birthday extraction must not store TODAY just because 'birthday' was mentioned. Field
bug (2026-06-14): on 06-13 the user said their birthday "was a week ago" and it got
stored as 06-13 (today) instead of ~06-06. birthday is a permanent, never-reconfirmed
fact, so a wrong value sticks forever — the prompt now handles relative dates and omits
when unsure. This pins the guidance in the prompt (no LLM/network call).
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import llm


class BirthdayExtractionPromptTest(unittest.TestCase):
    def _prompt(self) -> str:
        captured = {}

        def _fake_create(*args, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            raise RuntimeError("stop after capturing prompt")  # no network

        with mock.patch.object(llm._client.chat.completions, "create", side_effect=_fake_create):
            llm.extract_facts(1, [{"speaker": "Bret", "text": "my birthday was a week ago"}])
        return captured.get("prompt", "")

    def test_prompt_handles_relative_dates_and_forbids_defaulting_to_today(self):
        p = self._prompt()
        self.assertIn("TODAY minus 7 days", p)        # 'a week ago' is computed, not today
        self.assertIn("TODAY minus 1 day", p)         # 'yesterday'
        self.assertIn("never store today's date", p)  # the guard-rail
        self.assertIn("PAST or FUTURE reference is NOT today", p)
        self.assertIn("OMIT it entirely", p)          # omit when no specific date

    def test_today_is_still_handled(self):
        self.assertIn("TODAY's MM-DD", self._prompt())


if __name__ == "__main__":
    unittest.main()
