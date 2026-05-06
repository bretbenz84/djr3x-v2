import unittest
from unittest import mock

import numpy as np


class DirectedLookTests(unittest.TestCase):
    def test_parser_accepts_each_cardinal_look_direction(self):
        from intelligence import command_parser

        cases = {
            "look to your left": "left",
            "look to your right": "right",
            "look up": "up",
            "look down": "down",
        }
        for utterance, direction in cases.items():
            with self.subTest(utterance=utterance):
                match = command_parser.parse(utterance)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "directed_look")
                self.assertEqual(match.args["direction"], direction)

    def test_parser_accepts_bare_direction(self):
        from intelligence import command_parser

        match = command_parser.parse("down")

        self.assertIsNotNone(match)
        self.assertEqual(match.command_key, "directed_look")
        self.assertEqual(match.args["direction"], "down")
        self.assertFalse(match.args["search_target"])

    def test_parser_extracts_embedded_imperative_look_direction(self):
        from intelligence import command_parser

        match = command_parser.parse(
            "Guess what? Stop playing with me. Look down. Like he said, look down."
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.command_key, "directed_look")
        self.assertEqual(match.args["direction"], "down")

    def test_parser_accepts_look_for_target_search(self):
        from intelligence import command_parser

        match = command_parser.parse("look for the filament spool")

        self.assertIsNotNone(match)
        self.assertEqual(match.command_key, "directed_look")
        self.assertEqual(match.args["direction"], "current")
        self.assertEqual(match.args["target_hint"], "the filament spool")
        self.assertTrue(match.args["search_target"])

    def test_unadorned_multiword_search_target_defaults_to_object(self):
        from intelligence import interaction

        self.assertEqual(interaction._directed_target_kind("filament spool"), "object")

    def test_directed_look_left_and_right_use_configured_neck_limits(self):
        import config
        from sequences import animations

        neck_cfg = config.SERVO_CHANNELS["neck"]
        neck_ch = int(neck_cfg["ch"])

        with (
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "set_face_tracking_baseline"),
            mock.patch.object(animations.time, "sleep"),
            mock.patch.object(animations, "_record_directed_look"),
        ):
            self.assertEqual(animations.directed_look_pose("left"), "left")
            self.assertEqual(move_to.call_args.args[0][neck_ch], neck_cfg["min"])

            self.assertEqual(animations.directed_look_pose("right"), "right")
            self.assertEqual(move_to.call_args.args[0][neck_ch], neck_cfg["max"])

    def test_directed_look_center_uses_configured_neutral_pose(self):
        import config
        from sequences import animations

        channels = {
            name: int(cfg["ch"])
            for name, cfg in config.SERVO_CHANNELS.items()
            if name in {"neck", "headlift", "headtilt"}
        }

        with (
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "set_face_tracking_baseline"),
            mock.patch.object(animations.time, "sleep"),
            mock.patch.object(animations, "_record_directed_look"),
        ):
            self.assertEqual(animations.directed_look_pose("center"), "center")

        targets = move_to.call_args.args[0]
        self.assertEqual(
            targets[channels["neck"]],
            config.SERVO_CHANNELS["neck"]["neutral"],
        )
        self.assertEqual(
            targets[channels["headlift"]],
            config.SERVO_CHANNELS["headlift"]["neutral"],
        )
        self.assertEqual(
            targets[channels["headtilt"]],
            config.SERVO_CHANNELS["headtilt"]["neutral"],
        )

    def test_directed_look_up_and_down_use_configured_vertical_limits(self):
        import config
        from sequences import animations

        lift_cfg = config.SERVO_CHANNELS["headlift"]
        tilt_cfg = config.SERVO_CHANNELS["headtilt"]
        lift_ch = int(lift_cfg["ch"])
        tilt_ch = int(tilt_cfg["ch"])

        with (
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "set_face_tracking_baseline") as baseline,
            mock.patch.object(animations.time, "sleep"),
            mock.patch.object(animations, "_record_directed_look"),
        ):
            self.assertEqual(animations.directed_look_pose("up"), "up")
            up_targets = move_to.call_args.args[0]
            self.assertEqual(up_targets[lift_ch], lift_cfg["max"])
            self.assertEqual(up_targets[tilt_ch], tilt_cfg["min"])
            baseline.assert_called_with(
                neck=None,
                lift=lift_cfg["max"],
                tilt=tilt_cfg["min"],
            )

            self.assertEqual(animations.directed_look_pose("down"), "down")
            down_targets = move_to.call_args.args[0]
            self.assertEqual(down_targets[lift_ch], lift_cfg["min"])
            self.assertEqual(down_targets[tilt_ch], tilt_cfg["max"])
            baseline.assert_called_with(
                neck=None,
                lift=lift_cfg["min"],
                tilt=tilt_cfg["max"],
            )

    def test_bare_directional_look_moves_without_scene_analysis_or_speech(self):
        from intelligence import interaction

        old_context = dict(interaction._directed_look_context)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            interaction._reset_directed_look_context()
            with (
                mock.patch.object(interaction, "_move_and_capture_gaze", return_value=("left", frame)) as move,
                mock.patch.object(interaction, "_detect_faces_in_gaze", return_value=[]) as detect_faces,
                mock.patch.object(interaction, "_visible_known_face_candidate", return_value=None),
                mock.patch("vision.scene.analyze_directed_attention") as analyze,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._execute_directed_look_command(
                    {"direction": "left", "target_hint": "", "search_target": False},
                    person_id=1,
                    person_name="Bret Penziger",
                    raw_text="look left",
                )

            move.assert_called_once()
            detect_faces.assert_not_called()
            analyze.assert_not_called()
            speak.assert_not_called()
            self.assertTrue(interaction._is_silent_command_response(response))
            self.assertEqual(interaction._directed_look_context["bare_count"], 1)
        finally:
            interaction._directed_look_context.update(old_context)

    def test_third_bare_directional_look_asks_for_target(self):
        from intelligence import interaction

        old_context = dict(interaction._directed_look_context)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            interaction._reset_directed_look_context()
            with (
                mock.patch.object(interaction, "_move_and_capture_gaze", return_value=("left", frame)),
                mock.patch.object(interaction, "_detect_faces_in_gaze") as detect_faces,
                mock.patch.object(interaction, "_visible_known_face_candidate", return_value=None),
                mock.patch.object(interaction, "_speak_blocking") as speak,
                mock.patch.object(interaction.config, "DIRECTED_LOOK_CLARIFY_AFTER_COMMANDS", 3),
            ):
                for _ in range(2):
                    self.assertTrue(interaction._is_silent_command_response(
                        interaction._execute_directed_look_command(
                            {"direction": "left", "target_hint": "", "search_target": False},
                            person_id=1,
                            person_name="Bret Penziger",
                            raw_text="look left",
                        )
                    ))
                response = interaction._execute_directed_look_command(
                    {"direction": "down", "target_hint": "", "search_target": False},
                    person_id=1,
                    person_name="Bret Penziger",
                    raw_text="look down",
                )

            self.assertEqual(response, "What am I looking for?")
            detect_faces.assert_not_called()
            speak.assert_called_once_with("What am I looking for?", emotion="curious")
        finally:
            interaction._directed_look_context.update(old_context)

    def test_object_search_uses_prior_down_clue_and_stops_when_found(self):
        from intelligence import interaction

        old_context = dict(interaction._directed_look_context)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            interaction._reset_directed_look_context()
            interaction._note_directed_look_context(direction="down")

            def fake_analyze(_frame, *, direction, utterance, target_hint):
                return {
                    "target_summary": "A wrench is visible on the floor.",
                    "target_visible": direction == "down",
                    "subject_type": "object",
                    "visible_people_count": 0,
                    "animals": [],
                    "notable_details": ["wrench on floor"],
                    "roast_angle": "A floor tool, naturally choosing the least ergonomic shelf.",
                    "confidence": "high",
                }

            with (
                mock.patch.object(interaction, "_move_and_capture_gaze", return_value=("down", frame)) as move,
                mock.patch("vision.scene.analyze_directed_attention", side_effect=fake_analyze) as analyze,
                mock.patch.object(interaction.llm, "get_response", return_value="Found it. Your floor is cosplaying as a toolbox."),
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._execute_directed_look_command(
                    {"direction": "current", "target_hint": "the wrench", "search_target": True},
                    person_id=1,
                    person_name="Bret Penziger",
                    raw_text="look for the wrench",
                )

            move.assert_called_once()
            self.assertEqual(move.call_args.kwargs["target_hint"], "the wrench")
            analyze.assert_called_once()
            self.assertEqual(response, "Found it. Your floor is cosplaying as a toolbox.")
            speak.assert_called_once()
        finally:
            interaction._directed_look_context.update(old_context)


if __name__ == "__main__":
    unittest.main()
