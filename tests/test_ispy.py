"""I Spy look-around behavior (features/games.py).

On the physical droid, Rex LOOKS AROUND the room (left/center/right sweep under a
directed-gaze hold) before picking the secret object — the showmanship the game
was always supposed to have (owner call 2026-07-07) — then glances back toward
the object's view at the reveal. Servo-less machines degrade to the old
single-frame behavior.
"""

import unittest
from unittest import mock

import config
from features import games


def _fake_frame():
    return object()


class ScanRoomTest(unittest.TestCase):
    def test_sweeps_three_views_with_servos(self):
        poses = []
        holds = []
        with mock.patch("hardware.servos.connected", return_value=True), \
             mock.patch("intelligence.consciousness.hold_directed_gaze",
                        side_effect=lambda v, secs=None: holds.append(v)), \
             mock.patch("intelligence.consciousness.clear_directed_gaze_hold") as clear_hold, \
             mock.patch("sequences.animations.directed_look_pose",
                        side_effect=lambda v, **k: poses.append(v)), \
             mock.patch("vision.camera.capture_current_gaze", side_effect=lambda **k: _fake_frame()):
            views = games._ispy_scan_room()
        self.assertEqual([v for v, _ in views], ["left", "center", "right"])
        self.assertEqual(holds, ["left", "center", "right"])
        # The sweep ends by recentering and releasing the gaze hold.
        self.assertEqual(poses[-1], "center")
        clear_hold.assert_called_once()

    def test_degrades_to_single_frame_without_servos(self):
        with mock.patch("hardware.servos.connected", return_value=False), \
             mock.patch("vision.camera.get_frame", return_value=_fake_frame()), \
             mock.patch("sequences.animations.directed_look_pose") as pose:
            views = games._ispy_scan_room()
        self.assertEqual([v for v, _ in views], ["center"])
        pose.assert_not_called()

    def test_scan_disabled_uses_single_frame(self):
        with mock.patch.object(config, "ISPY_SCAN_ENABLED", False, create=True), \
             mock.patch("vision.camera.get_frame", return_value=_fake_frame()), \
             mock.patch("sequences.animations.directed_look_pose") as pose:
            views = games._ispy_scan_room()
        self.assertEqual([v for v, _ in views], ["center"])
        pose.assert_not_called()

    def test_no_camera_frames_returns_empty(self):
        with mock.patch("hardware.servos.connected", return_value=False), \
             mock.patch("vision.camera.get_frame", return_value=None):
            self.assertEqual(games._ispy_scan_room(), [])

    def test_all_sweep_captures_failing_falls_back_to_live_frame(self):
        with mock.patch("hardware.servos.connected", return_value=True), \
             mock.patch("intelligence.consciousness.hold_directed_gaze"), \
             mock.patch("intelligence.consciousness.clear_directed_gaze_hold"), \
             mock.patch("sequences.animations.directed_look_pose"), \
             mock.patch("vision.camera.capture_current_gaze", return_value=None), \
             mock.patch("vision.camera.get_frame", return_value=_fake_frame()):
            views = games._ispy_scan_room()
        self.assertEqual([v for v, _ in views], ["center"])


class PickTargetTest(unittest.TestCase):
    def _resp(self, payload):
        resp = mock.Mock()
        resp.choices = [mock.Mock()]
        resp.choices[0].message.content = payload
        return resp

    def test_picks_object_with_view(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = self._resp(
            '{"object": "red chair", "clue": "red", "view": "left"}'
        )
        with mock.patch.object(games, "_get_client", return_value=client), \
             mock.patch.object(games, "encode_jpeg_base64", return_value="abc"):
            target = games._ispy_pick_target([("left", _fake_frame()), ("right", _fake_frame())])
        self.assertEqual(target["object"], "red chair")
        self.assertEqual(target["view"], "left")
        # All views were sent to the model as labeled images.
        content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        images = [part for part in content if part.get("type") == "image_url"]
        self.assertEqual(len(images), 2)

    def test_bogus_view_coerced_to_a_real_one(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = self._resp(
            '{"object": "mug", "clue": "shiny", "view": "behind you"}'
        )
        with mock.patch.object(games, "_get_client", return_value=client), \
             mock.patch.object(games, "encode_jpeg_base64", return_value="abc"):
            target = games._ispy_pick_target([("center", _fake_frame())])
        self.assertEqual(target["view"], "center")

    def test_no_views_returns_none(self):
        self.assertIsNone(games._ispy_pick_target([]))


class StartAndRevealTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "i_spy"
        games._game_state = {}

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_start_announces_scan_and_stores_view(self):
        target = {"object": "red chair", "clue": "red", "view": "right"}
        with mock.patch.object(games, "_ispy_announce_scan") as announce, \
             mock.patch.object(games, "_ispy_get_target", return_value=target), \
             mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx):
            opener = games._ispy_start(None)
        announce.assert_called_once()
        self.assertEqual(games._game_state["target_view"], "right")
        self.assertIn("something that is red", opener)
        self.assertIn("Do not reveal the object or where it is", opener)

    def test_start_camera_failure_apologizes(self):
        with mock.patch.object(games, "_ispy_announce_scan"), \
             mock.patch.object(games, "_ispy_get_target", return_value=None), \
             mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx):
            opener = games._ispy_start(None)
        self.assertIn("camera isn't cooperating", opener)

    def test_correct_guess_glances_at_target_view(self):
        games._game_state.update({
            "target_object": "red chair", "clue": "red",
            "target_view": "left", "guess_count": 0,
        })
        with mock.patch("sequences.animations.directed_look_pose") as pose, \
             mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx):
            resp, done = games._ispy_handle("the red chair", None)
        self.assertTrue(done)
        pose.assert_called_once_with("left")

    def test_stop_glances_at_target_view(self):
        games._game_state.update({
            "target_object": "red chair", "clue": "red", "target_view": "right",
        })
        with mock.patch("sequences.animations.directed_look_pose") as pose, \
             mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx):
            games._ispy_stop(None)
        pose.assert_called_once_with("right")

    def test_wrong_guess_does_not_glance(self):
        games._game_state.update({
            "target_object": "red chair", "clue": "red",
            "target_view": "left", "guess_count": 0,
        })
        with mock.patch("sequences.animations.directed_look_pose") as pose, \
             mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_rex_respond", side_effect=lambda ctx, pid=None: ctx):
            resp, done = games._ispy_handle("the lamp", None)
        self.assertFalse(done)
        pose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
