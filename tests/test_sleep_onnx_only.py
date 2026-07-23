"""SLEEP has one interactive exit: the dedicated wakeuprex ONNX model."""

import unittest
from unittest import mock


class SleepOnnxOnlyTest(unittest.TestCase):
    def test_only_sleep_model_is_active_while_asleep(self):
        from audio import wake_word
        from state import State

        loaded = wake_word._loaded_models
        self.addCleanup(setattr, wake_word, "_loaded_models", loaded)
        wake_word._loaded_models = frozenset({
            "wakeuprex", "Hey_rex", "Dee-Jay_Rex", "shut_down",
        })
        active = wake_word._active_for_state(State.SLEEP)
        self.assertIn("wakeuprex", active)
        self.assertIn("shut_down", active)  # kill-switch; it does not wake Rex
        self.assertNotIn("Hey_rex", active)
        self.assertNotIn("Dee-Jay_Rex", active)

    def test_interaction_ignores_general_wake_model_while_asleep(self):
        from intelligence import interaction
        from state import State

        interaction._wake_word_fired.clear()
        with mock.patch.object(interaction.state_module, "get_state", return_value=State.SLEEP):
            interaction._on_wake_word("Hey_rex")
        self.assertFalse(interaction._wake_word_fired.is_set())

    def test_boredom_sleep_can_supply_resignation_line(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_speak_blocking") as speak,
            mock.patch.object(interaction, "_clear_listening_state_for_sleep"),
            mock.patch.object(interaction.state_module, "set_state"),
            mock.patch.object(interaction.motion_controller, "stop") as stop_motion,
            mock.patch.object(interaction, "_run_sleep_animation"),
        ):
            line = interaction._enter_sleep_mode(transition_line="The room wins. Going to sleep.")
        self.assertEqual(line, "The room wins. Going to sleep.")
        speak.assert_called_once_with("The room wins. Going to sleep.", emotion="sleepy")
        stop_motion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
