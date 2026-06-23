"""
Gamepad soundboard + animation buttons.

The 8BitDo Pro 2 pairs to the ESP32; its non-drive buttons arrive on the Mac as
firmware `event:"button"` messages, which motion_controller dispatches to a sound
clip (audio/soundboard.py) and/or a servo animation. These lock the resolution,
no-audio safety, the default button map, and the dispatch wiring — all without
audio hardware.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config


class SoundboardResolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        (d / "Air Horn.mp3").write_bytes(b"x")
        (d / "Scratch.mp3").write_bytes(b"x")
        self._p = mock.patch.object(config, "SOUNDBOARD_CLIPS_DIR", str(d))
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def test_resolves_by_stem_case_insensitive(self):
        from audio import soundboard
        self.assertEqual(soundboard.resolve_clip("Air Horn").name, "Air Horn.mp3")
        self.assertEqual(soundboard.resolve_clip("air horn").name, "Air Horn.mp3")
        self.assertEqual(soundboard.resolve_clip("AIR HORN.mp3").name, "Air Horn.mp3")

    def test_missing_and_empty_return_none(self):
        from audio import soundboard
        self.assertIsNone(soundboard.resolve_clip("not a clip"))
        self.assertIsNone(soundboard.resolve_clip(""))

    def test_list_clips(self):
        from audio import soundboard
        self.assertEqual(soundboard.list_clips(), ["Air Horn", "Scratch"])

    def test_no_audio_mode_skips_without_playing(self):
        from audio import soundboard
        with mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", True):
            self.assertFalse(soundboard.play("Air Horn"))

    def test_play_missing_clip_returns_false(self):
        from audio import soundboard
        with mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False):
            self.assertFalse(soundboard.play("nope nope"))


class GamepadButtonDispatchTest(unittest.TestCase):
    def test_button_event_triggers_clip_and_animation(self):
        from intelligence import motion_controller as mc
        actions = {"a": {"clip": "Air Horn", "animation": "tiny_victory_dance"}}
        with mock.patch.object(config, "MOTION_GAMEPAD_BUTTON_ACTIONS", actions), \
             mock.patch("sequences.animations.play_body_beat") as beat, \
             mock.patch("audio.soundboard.play") as snd:
            mc._on_motion_event({"type": "event", "event": "button", "btn": "a"})
        beat.assert_called_once_with("tiny_victory_dance")
        snd.assert_called_once_with("Air Horn")

    def test_button_name_is_case_insensitive(self):
        from intelligence import motion_controller as mc
        actions = {"x": {"clip": "Scratch"}}
        with mock.patch.object(config, "MOTION_GAMEPAD_BUTTON_ACTIONS", actions), \
             mock.patch("sequences.animations.play_body_beat") as beat, \
             mock.patch("audio.soundboard.play") as snd:
            mc._on_motion_event({"event": "button", "btn": "X"})
        snd.assert_called_once_with("Scratch")
        beat.assert_not_called()

    def test_unmapped_button_is_noop(self):
        from intelligence import motion_controller as mc
        with mock.patch.object(config, "MOTION_GAMEPAD_BUTTON_ACTIONS", {}), \
             mock.patch("sequences.animations.play_body_beat") as beat, \
             mock.patch("audio.soundboard.play") as snd:
            mc._on_motion_event({"event": "button", "btn": "a"})
        beat.assert_not_called()
        snd.assert_not_called()

    def test_non_button_event_ignored(self):
        from intelligence import motion_controller as mc
        with mock.patch("audio.soundboard.play") as snd, \
             mock.patch("sequences.animations.play_body_beat") as beat:
            mc._on_motion_event({"event": "estop"})
            mc._on_motion_event({"type": "telemetry", "owner": "manual"})
        snd.assert_not_called()
        beat.assert_not_called()


class DefaultButtonMapTest(unittest.TestCase):
    """The shipped default map must reference REAL animation beats and (when the user's
    clips are installed) real clip files."""

    def test_default_map_animations_are_real_beats(self):
        from sequences import animations
        for btn, act in config.MOTION_GAMEPAD_BUTTON_ACTIONS.items():
            anim = act.get("animation")
            if anim:
                self.assertIsNotNone(
                    animations._canonical_body_beat(anim), f"{btn} -> {anim!r}"
                )

    def test_default_map_clips_resolve_when_installed(self):
        from audio import soundboard
        if not soundboard.list_clips():
            self.skipTest("no clips installed in SOUNDBOARD_CLIPS_DIR")
        for btn, act in config.MOTION_GAMEPAD_BUTTON_ACTIONS.items():
            clip = act.get("clip")
            if clip:
                self.assertIsNotNone(
                    soundboard.resolve_clip(clip), f"{btn} -> {clip!r}"
                )


if __name__ == "__main__":
    unittest.main()
