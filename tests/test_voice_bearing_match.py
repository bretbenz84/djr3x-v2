"""Voice bearing ↔ visible face matching (perception/voice_bearing_match.py).

The one sign in this feature: a face's bearing is built in the camera/neck
convention (+ = Rex's RIGHT) and converted ONCE into the base frame the Flex
DoA reports in (+ = LEFT). A face on the right of the frame must land on the
negative side, where a voice from the right lands.

    venv/bin/python -m unittest tests.test_voice_bearing_match
"""

import unittest

from perception import voice_bearing_match as vbm

W = 1920.0       # frame width
FOV = 25.0       # half-FOV in degrees
KW = dict(frame_width=W, half_fov_deg=FOV, tolerance_deg=20.0, margin_deg=10.0,
          contradiction_deg=45.0)


def _face(pid, centre_x, w=200):
    return {"person_db_id": pid, "face_box": (centre_x - w / 2, 300, w, w), "face_id": f"p{pid}"}


class FaceBearingTest(unittest.TestCase):
    def test_centred_face_neck_centred_is_dead_ahead(self):
        self.assertAlmostEqual(vbm.face_bearing_deg(_face(1, W / 2), 0.0, frame_width=W, half_fov_deg=FOV), 0.0)

    def test_face_at_the_right_edge_is_on_the_right(self):
        # Box centred ON the right edge = +half-FOV to the right = -25 in the base frame.
        b = vbm.face_bearing_deg(_face(1, W, 200), 0.0, frame_width=W, half_fov_deg=FOV)
        self.assertAlmostEqual(b, -25.0)

    def test_face_at_the_left_edge_is_on_the_left(self):
        b = vbm.face_bearing_deg(_face(1, 0, 200), 0.0, frame_width=W, half_fov_deg=FOV)
        self.assertAlmostEqual(b, 25.0)

    def test_neck_yaw_adds_in_the_right_sense(self):
        # Head panned 20° right, face centred in the frame: the person is 20° right.
        b = vbm.face_bearing_deg(_face(1, W / 2), 20.0, frame_width=W, half_fov_deg=FOV)
        self.assertAlmostEqual(b, -20.0)
        # Head panned 20° right, face at the LEFT edge: 25 left of the head = 5 left of the body.
        b = vbm.face_bearing_deg(_face(1, 0, 200), 20.0, frame_width=W, half_fov_deg=FOV)
        self.assertAlmostEqual(b, 5.0)

    def test_no_box_no_bearing(self):
        self.assertIsNone(vbm.face_bearing_deg({"person_db_id": 1}, 0.0, frame_width=W, half_fov_deg=FOV))


class MatchTest(unittest.TestCase):
    def setUp(self):
        # Bret 20° right (frame x = centre + 0.8 * half-width), PJ 20° left.
        self.bret = _face(1, W / 2 + 0.8 * (W / 2))
        self.pj = _face(7, W / 2 - 0.8 * (W / 2))

    def test_voice_from_the_right_selects_the_face_on_the_right(self):
        res = vbm.match_faces_to_voice([self.pj, self.bret], -18.0, 0.0, **KW)
        self.assertEqual(res["confirm_pid"], 1)
        self.assertIs(res["selected"], self.bret)
        self.assertFalse(res["contradicts"])
        self.assertEqual([r["pid"] for r in res["faces"]], [1, 7])

    def test_voice_from_the_left_selects_the_face_on_the_left(self):
        res = vbm.match_faces_to_voice([self.pj, self.bret], 22.0, 0.0, **KW)
        self.assertEqual(res["confirm_pid"], 7)
        self.assertIs(res["selected"], self.pj)

    def test_between_two_faces_is_not_a_selection(self):
        # Voice at +1: PJ (+20) and Bret (-20) are 19 and 21 away — within
        # tolerance for PJ but only 2° better than Bret: confirm, don't select.
        res = vbm.match_faces_to_voice([self.pj, self.bret], 1.0, 0.0, **KW)
        self.assertEqual(res["confirm_pid"], 7)
        self.assertIsNone(res["selected"])
        self.assertFalse(res["contradicts"])

    def test_voice_far_from_every_face_contradicts(self):
        res = vbm.match_faces_to_voice([self.pj, self.bret], 150.0, 0.0, **KW)
        self.assertIsNone(res["confirm_pid"])
        self.assertIsNone(res["selected"])
        self.assertTrue(res["contradicts"])

    def test_the_grey_zone_is_neither(self):
        # 35° from the nearest face: outside tolerance, inside the contradiction band.
        res = vbm.match_faces_to_voice([self.bret], 15.0, 0.0, **KW)
        self.assertIsNone(res["confirm_pid"])
        self.assertFalse(res["contradicts"])

    def test_single_face_needs_no_margin(self):
        res = vbm.match_faces_to_voice([self.bret], -25.0, 0.0, **KW)
        self.assertIs(res["selected"], self.bret)

    def test_unknown_face_can_confirm_but_never_select(self):
        anon = _face(None, W / 2 + 0.8 * (W / 2))
        res = vbm.match_faces_to_voice([anon], -20.0, 0.0, **KW)
        self.assertIsNone(res["confirm_pid"])
        self.assertIsNone(res["selected"])

    def test_faces_without_boxes_are_skipped(self):
        res = vbm.match_faces_to_voice([{"person_db_id": 3}, self.bret], -20.0, 0.0, **KW)
        self.assertEqual(len(res["faces"]), 1)

    def test_describe_reads_sanely(self):
        res = vbm.match_faces_to_voice([self.pj, self.bret], -18.0, 0.0, **KW)
        text = vbm.describe(res, {1: "Bret", 7: "PJ"})
        self.assertIn("voice -18°", text)
        self.assertIn("Bret -20°", text)
        self.assertTrue(text.endswith("→ Bret"))


if __name__ == "__main__":
    unittest.main()


class LensModelTest(unittest.TestCase):
    """The fisheye model: pixels off centre / px_per_deg, plus a constant yaw offset."""

    def test_px_per_deg_replaces_the_fraction_model(self):
        # 320 px right of centre at 16 px/deg = 20° right = -20 in the base frame.
        b = vbm.face_bearing_deg(_face(1, W / 2 + 320), 0.0, frame_width=W, half_fov_deg=FOV,
                                 px_per_deg=16.0)
        self.assertAlmostEqual(b, -20.0)

    def test_yaw_offset_adds_like_neck_yaw(self):
        b = vbm.face_bearing_deg(_face(1, W / 2), 0.0, frame_width=W, half_fov_deg=FOV,
                                 px_per_deg=16.0, yaw_offset_deg=14.0)
        self.assertAlmostEqual(b, -14.0)

    def test_zero_px_per_deg_falls_back(self):
        b = vbm.face_bearing_deg(_face(1, W), 0.0, frame_width=W, half_fov_deg=FOV, px_per_deg=0.0)
        self.assertAlmostEqual(b, -25.0)

    def test_fit_recovers_a_known_model(self):
        k, c = 16.0, 14.0
        samples = []
        for px in (-700.0, -300.0, 0.0, 250.0, 600.0):
            voice = -(c + px / k)
            samples.append((px, voice, 0.0))
        fit = vbm.fit_camera_model(samples)
        self.assertAlmostEqual(fit["px_per_deg"], k, places=6)
        self.assertAlmostEqual(fit["yaw_offset_deg"], c, places=6)
        self.assertAlmostEqual(fit["rms_deg"], 0.0, places=6)

    def test_fit_takes_the_neck_into_account(self):
        # Same geometry, but the head was panned 10° right during every take:
        # the fitted constant must exclude that (the live app reads the neck).
        samples = [(px, -(14.0 + 10.0 + px / 16.0), 10.0) for px in (-600.0, 0.0, 500.0)]
        fit = vbm.fit_camera_model(samples)
        self.assertAlmostEqual(fit["yaw_offset_deg"], 14.0, places=6)

    def test_fit_needs_spread(self):
        self.assertIsNone(vbm.fit_camera_model([(100.0, -20.0, 0.0), (100.0, -25.0, 0.0)]))
        self.assertIsNone(vbm.fit_camera_model([(100.0, -20.0, 0.0)]))

    def test_tonights_three_takes_fit_within_a_few_degrees(self):
        # 2026-09-02 21:41-21:44, neck reported centred: (px, voice)
        samples = [(253.5, -33.1, 0.0), (378.5, -39.7, 0.0), (-698.0, 33.0, 0.0)]
        fit = vbm.fit_camera_model(samples)
        self.assertGreater(fit["px_per_deg"], 12.0)
        self.assertLess(fit["px_per_deg"], 22.0)
        self.assertLess(fit["rms_deg"], 5.0)
