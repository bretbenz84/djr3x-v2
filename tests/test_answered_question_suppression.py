"""
Answered-question suppression: the system prompt must carry a hard no-repeat rule anchored
to the LIVE transcript, so neither the reply path nor the proactive small-talk path re-asks
something the human already answered this session (live-run failure: Rex re-asked "best
photon lately?" right after Bret had named his astrophotography targets).
"""

import unittest

from intelligence import llm
from memory import conversations as conv_db

_RULE = "Before you ask ANY question, scan the exchanges above"


class AnsweredQuestionSuppressionTest(unittest.TestCase):
    def setUp(self):
        conv_db.clear_transcript()

    def tearDown(self):
        conv_db.clear_transcript()

    def test_rule_present_when_transcript_exists(self):
        conv_db.add_to_transcript("Rex", "What targets have you been shooting lately?")
        conv_db.add_to_transcript("Bret", "The North American nebula, mostly.")
        prompt = llm.assemble_system_prompt(person_id=None)
        self.assertIn(_RULE, prompt)

    def test_rule_absent_when_no_transcript(self):
        prompt = llm.assemble_system_prompt(person_id=None)
        self.assertNotIn(_RULE, prompt)


if __name__ == "__main__":
    unittest.main()
