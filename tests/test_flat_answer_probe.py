"""
Reply-side flat-answer follow-up (owner spec 2026-07-06): "It's okay" answering
"how's your day?" gets the probe IN the reply — quip plus "what's the missing
30%?" in one breath — instead of waiting ~15s for the lull impulse.

Anti-interview guards under test: fires only when Rex's LAST line was a question
(an "okay" acknowledging a statement is agreement, not flatness), once per
cooldown window, never in a heavy/give-space window, directive path exempt.
"""

import unittest
from unittest import mock

import config
from intelligence import lean_brain as LB


def _turns(last_rex="How's your day going?"):
    return [
        ("user", "Bret Benziger", "hey"),
        ("assistant", "Rex", last_rex),
    ]


class FlatAnswerDetectorTest(unittest.TestCase):
    def test_flat_answers_detected(self):
        for t in ("It's okay", "fine", "meh", "not much", "Uh, it's okay I guess",
                  "I'm alright", "same old same old", "nothing really", "idk",
                  "pretty good", "could be worse", "it was fine"):
            self.assertTrue(LB._is_flat_answer(t), t)

    def test_real_content_not_flat(self):
        for t in ("It's okay but my boss yelled at me today",
                  "I finally fixed the robot's wheels",
                  "okay so here's the thing",   # 6+ words of real content
                  "not much besides the volleyball tournament"):
            self.assertFalse(LB._is_flat_answer(t), t)

    def test_empty_and_long_are_not_flat(self):
        self.assertFalse(LB._is_flat_answer(""))
        self.assertFalse(LB._is_flat_answer("okay okay okay okay okay okay okay"))


class ProbeLineTest(unittest.TestCase):
    def setUp(self):
        LB._last_flat_probe_at = 0.0

    def tearDown(self):
        LB._last_flat_probe_at = 0.0

    def _probe(self, text="It's okay", last_rex="How's your day going?"):
        with mock.patch("intelligence.callback_engine.recently_heavy",
                        return_value=False, create=True):
            return LB._flat_answer_probe_line(text, _turns(last_rex))

    def test_flat_answer_to_rex_question_probes(self):
        line = self._probe()
        self.assertIsNotNone(line)
        self.assertIn("FLAT-ANSWER FOLLOW-UP", line)
        self.assertIn("EXCEPTION", line)               # overrides question restraint
        self.assertIn("let small things be small", line)  # deflection = drop it

    def test_okay_after_rex_statement_is_acknowledgment(self):
        # Rex's last line wasn't a question — "okay" is agreement, never probed.
        self.assertIsNone(self._probe(last_rex="Queuing that up now."))

    def test_cooldown_blocks_consecutive_probes(self):
        self.assertIsNotNone(self._probe())
        self.assertIsNone(self._probe())   # second flat answer inside the window

    def test_heavy_window_blocks(self):
        with mock.patch("intelligence.callback_engine.recently_heavy",
                        return_value=True, create=True):
            self.assertIsNone(
                LB._flat_answer_probe_line("It's okay", _turns()))

    def test_kill_switch(self):
        with mock.patch.object(config, "FLAT_ANSWER_PROBE_ENABLED", False, create=True):
            self.assertIsNone(self._probe())


class MessagesIntegrationTest(unittest.TestCase):
    def setUp(self):
        LB._last_flat_probe_at = 0.0

    def tearDown(self):
        LB._last_flat_probe_at = 0.0

    def _msgs(self, user_text, *, label=True):
        transcript = [
            {"speaker": "Bret Benziger", "text": "hey"},
            {"speaker": "Rex", "text": "How's your day going?"},
        ]
        with (
            mock.patch.object(LB, "_persona", return_value="PERSONA"),
            mock.patch.object(LB, "_person_lines", return_value=[]),
            mock.patch.object(LB, "_scene_lines", return_value=[]),
            mock.patch("intelligence.callback_engine.recently_heavy",
                       return_value=False, create=True),
        ):
            return LB._messages(user_text, 1, transcript, None,
                                label_current_speaker=label)

    def test_probe_rides_in_system_prompt(self):
        msgs = self._msgs("It's okay")
        self.assertIn("FLAT-ANSWER FOLLOW-UP", msgs[0]["content"])

    def test_normal_answer_gets_no_probe(self):
        msgs = self._msgs("Pretty great actually, we went hiking")
        self.assertNotIn("FLAT-ANSWER FOLLOW-UP", msgs[0]["content"])

    def test_directive_path_never_probes(self):
        msgs = self._msgs("It's okay", label=False)
        self.assertNotIn("FLAT-ANSWER FOLLOW-UP", msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
