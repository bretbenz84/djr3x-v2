import unittest
from unittest import mock


class DirectedLookTests(unittest.TestCase):
    def test_directed_look_left_and_right_use_configured_neck_limits(self):
        import config
        from sequences import animations

        neck_cfg = config.SERVO_CHANNELS["neck"]
        neck_ch = int(neck_cfg["ch"])

        with (
            mock.patch.object(animations.servos, "move_to") as move_to,
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


if __name__ == "__main__":
    unittest.main()
