import unittest
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
