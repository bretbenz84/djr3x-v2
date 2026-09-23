import unittest
from unittest import mock
from tools import throttle_pose_tour as tour


class TourTest(unittest.TestCase):
    def test_low_to_low_uses_raised_transition_and_parallel_downstream(self):
        start = {8: 9088, 9: 9984, 10: 2048}
        end = {8: 9088, 9: 5589, 10: 9511}
        self.assertEqual(tour.transition(start, end), [{8: 2176}, {9: 5589, 10: 9511}, {8: 9088}])

    def test_high_pose_changes_only_downstream(self):
        self.assertEqual(tour.transition({8:2176,9:9984,10:2048}, {8:2176,9:2700,10:6200}), [{9:2700,10:6200}])

    def test_parked_requires_no_writes(self):
        self.assertEqual(tour.transition(tour.PARK, tour.PARK), [])

    def test_all_saved_poses_included(self):
        poses = tour.load_poses()
        self.assertEqual(len(poses), 11)
        self.assertEqual({p['id'] for p in poses}, set(range(1,12)))

    def test_conflicting_saved_poses_are_excluded(self):
        poses = tour.load_poses()
        self.assertEqual([p['id'] for p in poses if not tour.respects_elbow_limit(p['target'])], [3, 4])
        valid = [p for p in poses if tour.respects_elbow_limit(p['target'])]
        current = tour.PARK.copy()
        for pose in valid + [{'target':tour.PARK}]:
            for stage in tour.transition(current, pose['target']):
                current.update(stage)
                self.assertTrue(tour.respects_elbow_limit(current))

    def test_recorded_mode_accepts_exact_saved_exception_only(self):
        saved = {8:9088, 9:5589, 10:9511}
        default = tour.Controller(mock.Mock(), {})
        recorded = tour.Controller(mock.Mock(), {}, [saved])
        self.assertFalse(default.allowed(saved))
        self.assertTrue(recorded.allowed(saved))
        self.assertFalse(recorded.allowed({**saved, 9:5500}))
        self.assertFalse(recorded.allowed({**saved, 10:6000}))

    def test_non_throttle_write_rejected(self):
        port = mock.Mock()
        controller = tour.Controller(port, {})
        with self.assertRaises(ValueError):
            controller.move({0:6000})
        port.write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
