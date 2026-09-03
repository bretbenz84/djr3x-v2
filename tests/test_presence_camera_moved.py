"""A face gone because REX moved the camera is not a departure.

Field 2026-09-03 12:57 (logs/djr3x-2026-09-03-12-53-39.log): Rex panned from Bret
to PJ, and 12 s later announced "Bret slipped off-camera like he's being hunted by
responsibility". Bret: "I didn't go anywhere, and you just turned your head." A
realign base turn then made an unknown slot vanish: "And off goes you there".

The missing clock must not run while the camera points somewhere else than where
the person was last seen; it starts when the camera is back.

    venv/bin/python -m unittest tests.test_presence_camera_moved
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness as C


def _snapshot(*people):
    return {"people": [dict(p) for p in people]}


BRET = {"id": "person_1", "person_db_id": 1, "face_id": "Bret Benziger", "face_box": (800, 400, 200, 200)}


class _Profile:
    likely_still_present = False
    apparent_departure = False
    user_mid_sentence = False
    interaction_busy = False


class CameraMovedTest(unittest.TestCase):
    def setUp(self):
        for d in (C._first_missing_at, C._pending_departure_keys, C._confirmed_absent_at,
                  C._last_seen_pose, C._camera_moved_since, C._visit_started_at):
            d.clear()
        C._camera_moved_logged.clear()
        C._visible_people.clear()
        C._last_snapshot = {"people": []}
        self.now = 1000.0
        self.pose = (0.0, 0.0)
        self._t = mock.patch.object(C.time, "monotonic", side_effect=lambda: self.now)
        self._t.start()
        self._p = mock.patch.object(C, "_camera_pose", side_effect=lambda: self.pose)
        self._p.start()
        self._c = mock.patch.object(config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", 12.0, create=True)
        self._c.start()
        self._c2 = mock.patch.object(config, "PRESENCE_DEPARTURE_CONFIRM_SECS", 40.0, create=True)
        self._c2.start()
        # Staging is what these tests assert; the spoken quip has its own gates.
        self._f = mock.patch.object(C, "_should_fire_presence", return_value=False)
        self._f.start()

    def tearDown(self):
        for m in (self._t, self._p, self._c, self._c2, self._f):
            m.stop()
        for d in (C._first_missing_at, C._pending_departure_keys, C._confirmed_absent_at,
                  C._last_seen_pose, C._camera_moved_since):
            d.clear()
        C._visible_people.clear()

    def _tick(self, snapshot, dt=1.0):
        self.now += dt
        C._step_presence_tracking(snapshot, _Profile())
        C._visible_people = set(C._presence_tracking_map(snapshot, self.now).keys())
        C._last_snapshot = snapshot

    def _see_bret(self):
        self._tick(_snapshot(BRET))
        self.assertIn(1, C._last_seen_pose)

    def test_head_turned_away_holds_the_missing_clock(self):
        self._see_bret()
        self.pose = (40.0, 0.0)             # neck panned 40° to PJ
        for _ in range(60):
            self._tick(_snapshot())
        self.assertNotIn(1, C._pending_departure_keys)
        self.assertIn(1, C._first_missing_at)
        self.assertAlmostEqual(C._first_missing_at[1], self.now)   # restarted every tick

    def test_base_turned_away_holds_the_missing_clock(self):
        self._see_bret()
        self.pose = (0.0, -45.0)            # realign base turn
        for _ in range(60):
            self._tick(_snapshot())
        self.assertNotIn(1, C._pending_departure_keys)

    def test_clock_runs_once_the_camera_is_back(self):
        self._see_bret()
        self.pose = (40.0, 0.0)
        for _ in range(30):
            self._tick(_snapshot())
        self.pose = (2.0, 0.0)              # looked back — still no Bret
        for _ in range(45):
            self._tick(_snapshot())
        self.assertIn(1, C._pending_departure_keys)
        departed_at = C._pending_departure_keys[1][0]
        # Counted from the camera's return, not from the head turn.
        self.assertGreaterEqual(departed_at, 1000.0 + 31.0)

    def test_camera_still_means_a_real_absence(self):
        self._see_bret()
        for _ in range(60):
            self._tick(_snapshot())
        self.assertIn(1, C._pending_departure_keys)

    def test_hold_is_bounded(self):
        self._see_bret()
        self.pose = (40.0, 0.0)
        with mock.patch.object(config, "PRESENCE_CAMERA_MOVED_MAX_HOLD_SECS", 20.0, create=True):
            for _ in range(80):
                self._tick(_snapshot())
        self.assertIn(1, C._pending_departure_keys)

    def test_no_pose_readback_changes_nothing(self):
        self.pose = (None, None)
        self._see_bret()
        for _ in range(60):
            self._tick(_snapshot())
        self.assertIn(1, C._pending_departure_keys)


if __name__ == "__main__":
    unittest.main()
