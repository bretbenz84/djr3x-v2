"""During-DJ-playback restricted command listener.

Field 2026-07-30: during radio playback the conversation loop was fully
suppressed and the only override was a wake word at a raised threshold —
"stop the music" did nothing and the owner had to kill the process. The
listener keeps a narrow ear open (hardware AEC required) and executes ONLY
music-control/shutdown commands; all other transcripts are dropped.
"""

import unittest
from unittest import mock

import numpy as np

from intelligence import interaction
from state import State


def _step(text, *, states=(State.ACTIVE,)):
    """Run one listener step with the audio path stubbed to yield *text*."""
    from features import dj as dj_mod

    calls = {"spoken": [], "dj": []}
    chunk = np.ones(1600, dtype=np.float32)
    with mock.patch.object(interaction.stream, "get_audio_chunk", return_value=chunk), \
         mock.patch.object(interaction.vad, "is_speech", return_value=True), \
         mock.patch.object(interaction, "_accumulate_speech", return_value=chunk), \
         mock.patch.object(interaction.transcription, "transcribe", return_value=text), \
         mock.patch.object(interaction, "_duck_dj_for_speech", return_value=None), \
         mock.patch.object(interaction, "_restore_dj_volume"), \
         mock.patch.object(
             interaction, "_speak_blocking",
             side_effect=lambda line, **kw: calls["spoken"].append(line) or True), \
         mock.patch.object(interaction.state_module, "set_state") as set_state, \
         mock.patch.object(dj_mod, "stop",
                           side_effect=lambda **kw: calls["dj"].append("stop")), \
         mock.patch.object(dj_mod, "skip",
                           side_effect=lambda: calls["dj"].append("skip")), \
         mock.patch.object(dj_mod, "volume_up",
                           side_effect=lambda step=None: calls["dj"].append("volume_up")), \
         mock.patch.object(dj_mod, "volume_down",
                           side_effect=lambda step=None: calls["dj"].append("volume_down")), \
         mock.patch.object(dj_mod, "is_playing", return_value=True):
        interaction._dj_command_listen_step(allowed_states=states)
    calls["set_state"] = set_state
    return calls


class DJCommandListenTest(unittest.TestCase):
    def test_stop_the_music_stops_playback(self):
        calls = _step("Stop the music.")
        self.assertIn("stop", calls["dj"])
        self.assertTrue(calls["spoken"])

    def test_bare_stop_stops_playback(self):
        calls = _step("Stop.")
        self.assertIn("stop", calls["dj"])

    def test_volume_and_skip_commands_execute(self):
        self.assertIn("volume_down", _step("Turn it down.")["dj"])
        self.assertIn("skip", _step("Skip this song.")["dj"])

    def test_radio_chatter_is_dropped(self):
        # Whatever the station says must never execute or speak.
        for text in (
            "Tonight on Classical KDFC, Beethoven's Fifth.",
            "What's the weather tomorrow?",
            "Tell me a joke.",
            "",
        ):
            calls = _step(text)
            self.assertEqual(calls["dj"], [], f"{text!r} must not execute")
            self.assertEqual(calls["spoken"], [])
            calls["set_state"].assert_not_called()

    def test_shutdown_powers_off(self):
        calls = _step("Shut down.")
        self.assertIn("stop", calls["dj"])
        calls["set_state"].assert_called_once_with(State.SHUTDOWN)

    def test_listener_requires_hardware_aec(self):
        with mock.patch.object(interaction.config, "DJ_COMMAND_LISTEN_ENABLED", True), \
             mock.patch("audio.hardware_aec.is_active", return_value=False):
            self.assertFalse(interaction._dj_command_listen_available())
        with mock.patch.object(interaction.config, "DJ_COMMAND_LISTEN_ENABLED", True), \
             mock.patch("audio.hardware_aec.is_active", return_value=True):
            self.assertTrue(interaction._dj_command_listen_available())
        with mock.patch.object(interaction.config, "DJ_COMMAND_LISTEN_ENABLED", False):
            self.assertFalse(interaction._dj_command_listen_available())


if __name__ == "__main__":
    unittest.main()
