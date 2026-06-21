"""
Wave back: when a visible person waves (pose pipeline classifies gesture="waving" onto
world_state.people), Rex returns the wave + one short warm line — debounced per person and
globally so a single wave fires one reaction, and gated like other proactive beats.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import consciousness as c


class _Profile:
    def __init__(self, suppress_proactive=False):
        self.suppress_proactive = suppress_proactive


def _waving(person_db_id=1, face_id="Bret", visible=True):
    return {
        "person_db_id": person_db_id, "face_id": face_id,
        "face_visible": visible, "gesture": "waving",
    }


class WaveBackHelpersTest(unittest.TestCase):
    def test_line_uses_name_when_known(self):
        with mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]):
            self.assertEqual(c._wave_back_line("Bret"), "Hey, Bret!")

    def test_line_falls_back_without_name(self):
        with mock.patch.object(c.config, "WAVE_BACK_LINES_NO_NAME", ["Hey there!"]):
            self.assertEqual(c._wave_back_line(""), "Hey there!")

    def test_person_key_prefers_db_id(self):
        self.assertEqual(c._wave_person_key({"person_db_id": 7}), "db:7")
        self.assertEqual(c._wave_person_key({"id": "person_2"}), "slot:person_2")
        self.assertEqual(c._wave_person_key({}), "unknown")


class StepWaveReactionTest(unittest.TestCase):
    def setUp(self):
        c._wave_reacted_keys.clear()
        c._last_wave_reaction_at = 0.0

    def tearDown(self):
        c._wave_reacted_keys.clear()
        c._last_wave_reaction_at = 0.0

    def _run(self, snapshot, *, profile=None, can_speak=True):
        captured = {}
        with mock.patch.object(c, "_can_proactive_speak", return_value=can_speak), \
             mock.patch.object(c, "_first_name", return_value="Bret"), \
             mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: captured.update(line=line, kw=k)), \
             mock.patch("sequences.animations.wake_word_ack_wave") as wave:
            c._step_wave_reaction(snapshot, profile or _Profile())
            captured["waved"] = wave.called
        return captured

    def test_waves_back_at_a_waver(self):
        out = self._run({"people": [_waving()]})
        self.assertTrue(out["waved"])
        self.assertIn("Bret", out["line"])
        self.assertEqual(out["kw"].get("purpose"), "wave_back")
        # reactive=True so the hello breaks through awaiting-reply / active-conversation
        # gates (a wave during a conversation should still be acknowledged).
        self.assertTrue(out["kw"].get("reactive"))

    def test_neutral_gesture_does_nothing(self):
        p = _waving(); p["gesture"] = "neutral"
        out = self._run({"people": [p]})
        self.assertFalse(out["waved"])
        self.assertNotIn("line", out)

    def test_not_visible_does_nothing(self):
        out = self._run({"people": [_waving(visible=False)]})
        self.assertFalse(out["waved"])

    def test_disabled_flag(self):
        with mock.patch.object(c.config, "WAVE_BACK_ENABLED", False):
            out = self._run({"people": [_waving()]})
        self.assertFalse(out["waved"])

    def test_suppress_proactive_blocks(self):
        out = self._run({"people": [_waving()]}, profile=_Profile(suppress_proactive=True))
        self.assertFalse(out["waved"])

    def test_give_space_gate_blocks(self):
        # _can_proactive_speak False (e.g. sober window / DJ / active game) → no wave-back.
        out = self._run({"people": [_waving()]}, can_speak=False)
        self.assertFalse(out["waved"])

    def test_per_person_cooldown_debounces(self):
        self.assertTrue(self._run({"people": [_waving()]})["waved"])  # fires
        # Same person still waving next tick, but within the per-person cooldown → no re-fire
        # (also reset the global gap so we isolate the per-person debounce).
        c._last_wave_reaction_at = 0.0
        self.assertFalse(self._run({"people": [_waving()]})["waved"])

    def test_unknown_waver_gets_no_name_line(self):
        captured = {}
        with mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: captured.update(line=line)), \
             mock.patch.object(c.config, "WAVE_BACK_LINES_NO_NAME", ["Hey there!"]), \
             mock.patch("sequences.animations.wake_word_ack_wave"):
            c._step_wave_reaction({"people": [_waving(person_db_id=None, face_id=None)]}, _Profile())
        self.assertEqual(captured.get("line"), "Hey there!")


if __name__ == "__main__":
    unittest.main()
