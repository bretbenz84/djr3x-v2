import copy
import unittest
from unittest import mock
from tools import throttle_fluid_tour as fluid
from tools import throttle_pose_tour as tour


class FluidTourTest(unittest.TestCase):
    def setUp(self):
        self.limits = {cfg['ch']: cfg for cfg in fluid.helper._SERVO_DEFAULTS.values()}

    def test_all_poses_no_dwell_park_and_verified_corridor(self):
        plan = fluid.build_plan()
        fluid.verify_plan(plan, self.limits)
        self.assertEqual([s['pose_id'] for s in plan if s['pose_id'] is not None],
                         [1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5, 'park'])
        self.assertEqual(plan[0]['start'], tour.PARK)
        self.assertEqual(plan[-1]['end'], tour.PARK)
        self.assertTrue(any(s['start'][9] != s['end'][9] and s['start'][10] != s['end'][10] for s in plan))

    def test_interpolation_is_synchronized_and_no_overshoot(self):
        start = {8:2176,9:2048,10:9984}
        end = {8:2176,9:9984,10:2048}
        self.assertEqual(fluid.interpolate(start, end, 0), start)
        self.assertEqual(fluid.interpolate(start, end, 1), end)
        for i in range(101):
            p = fluid.interpolate(start, end, i/100)
            self.assertEqual(p[8],2176)
            self.assertEqual(p[9]+p[10],12032)
            self.assertTrue(2048 <= p[9] <= 9984)

    def test_segment_peak_velocity_and_acceleration_within_budgets(self):
        for segment in fluid.build_plan():
            duration = segment['seconds']
            for ch in tour.CHANNELS:
                distance = abs(segment['end'][ch]-segment['start'][ch])/4
                self.assertLessEqual(1.875*distance/duration, fluid.VELOCITY_US[ch]+1e-8)
                self.assertLessEqual(10/(3**.5)*distance/duration**2, fluid.ACCELERATION_US[ch]+1e-8)

    def test_rejects_new_shoulder_downstream_blend(self):
        plan = copy.deepcopy(fluid.build_plan())
        plan[0]['end'][9] -= 100
        with self.assertRaises(ValueError):
            fluid.verify_plan(plan, self.limits)

    def test_batch_packet_only_addresses_channels_eight_through_ten(self):
        port=mock.Mock()
        port.write.side_effect=len
        fluid.send_frame(port, tour.PARK)
        packet=port.write.call_args.args[0]
        self.assertEqual(packet[:3],bytes([0x9F,3,8]))
        self.assertEqual(len(packet),9)
        self.assertEqual([packet[i]+128*packet[i+1] for i in (3,5,7)],list(tour.PARK.values()))

    def test_refuses_nonparked_start_without_motion(self):
        port=mock.Mock()
        with mock.patch.object(fluid,'current_pose',return_value={8:2176,9:9984,10:2048}), self.assertRaises(ValueError):
            fluid.run(port,fluid.build_plan(),self.limits)
        port.write.assert_not_called()

    def test_stop_before_arrival_check(self):
        port=mock.Mock()
        with mock.patch.object(tour,'check_stop',side_effect=InterruptedError), self.assertRaises(InterruptedError):
            fluid.wait_arrival(port,tour.PARK,self.limits)
        port.write.assert_not_called()

    def test_incomplete_batch_write_fails(self):
        port=mock.Mock()
        port.write.return_value=8
        with self.assertRaises(OSError):
            fluid.send_frame(port,tour.PARK)


if __name__ == '__main__':
    unittest.main()
