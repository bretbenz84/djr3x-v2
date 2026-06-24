"""Game turns own the conversation: proactive prompt layers stand down (intelligence/llm.py).

During a game the player's terse yes/no answers are gameplay, not emotional or social
signals. Two proactive prompt layers were hijacking game turns and are now gated on
games.suppresses_conversation_interruptions() via llm._game_active():
  - the empathy directive — a "worsening mood" misread put a "That landed wrong, let me try
    again—" preamble on plain game questions, and
  - the "UNKNOWN PERSON IN FRAME" curiosity.

These are end-to-end checks: they build the real system prompt and assert the layer is
present without a game and absent during one.
"""

import unittest
from unittest import mock

from features import games
from intelligence import llm, empathy

_EMPATHY_SENTINEL = "ZZ_EMPATHY_DIRECTIVE_SENTINEL_ZZ"


def _ws_with_unknown_face():
    return {
        "people": [
            {"person_db_id": 1, "face_id": "Bret Benziger",
             "face_box": [1, 2, 3, 4], "face_visible": True},
            {"person_db_id": None, "face_box": [5, 6, 7, 8], "face_visible": True},
        ],
        "self_state": {"emotion": "neutral"},
        "crowd": {"count": 2},
    }


class GameActiveHelperTest(unittest.TestCase):
    def setUp(self):
        games._active_game = None
        self.addCleanup(setattr, games, "_active_game", None)

    def test_helper_reflects_game_state(self):
        self.assertFalse(llm._game_active())
        games._active_game = "20_questions"
        self.assertTrue(llm._game_active())


class EmpathySuppressionTest(unittest.TestCase):
    def setUp(self):
        games._active_game = None
        self.addCleanup(setattr, games, "_active_game", None)

    def test_empathy_present_when_no_game(self):
        with mock.patch.object(empathy, "get_directive", return_value=_EMPATHY_SENTINEL):
            prompt = llm.assemble_system_prompt(person_id=1)
        self.assertIn(_EMPATHY_SENTINEL, prompt)

    def test_empathy_suppressed_during_game(self):
        games._active_game = "20_questions"
        with mock.patch.object(empathy, "get_directive", return_value=_EMPATHY_SENTINEL):
            prompt = llm.assemble_system_prompt(person_id=1)
        self.assertNotIn(_EMPATHY_SENTINEL, prompt)


class UnknownPersonSuppressionTest(unittest.TestCase):
    def setUp(self):
        games._active_game = None
        self.addCleanup(setattr, games, "_active_game", None)

    def test_unknown_curiosity_present_when_no_game(self):
        with mock.patch.object(llm.world_state, "snapshot", return_value=_ws_with_unknown_face()):
            prompt = llm.assemble_system_prompt(person_id=1)
        self.assertIn("UNKNOWN PERSON IN FRAME", prompt)

    def test_unknown_curiosity_suppressed_during_game(self):
        games._active_game = "20_questions"
        with mock.patch.object(llm.world_state, "snapshot", return_value=_ws_with_unknown_face()):
            prompt = llm.assemble_system_prompt(person_id=1)
        self.assertNotIn("UNKNOWN PERSON IN FRAME", prompt)


if __name__ == "__main__":
    unittest.main()
