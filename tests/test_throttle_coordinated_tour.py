import unittest
from unittest import mock
from tools import throttle_coordinated_tour as c
from tools import throttle_pose_tour as t


class CoordinatedTest(unittest.TestCase):
    def setUp(self):
        self.limits={cfg['ch']:cfg for cfg in c.helper._SERVO_DEFAULTS.values()}

    def test_forward_then_three_upward_reaches_with_only_final_home(self):
        plan = c.make_reach_plan(c.REACH_AND_UP_US)
        original = c.make_reach_plan()
        self.assertEqual(plan[:8], original[:8])
        self.assertEqual(sum(m['end'] == t.PARK for m in plan), 1)
        self.assertEqual(plan[-1]['end'], t.PARK)
        upward = [m for m in plan if m['pose_id'].startswith('reach-up-')]
        self.assertEqual(len(upward), 3)
        self.assertGreaterEqual(max(m['end'][10] for m in upward) -
                                min(m['end'][10] for m in upward), 600 * 4)
        for move in upward:
            self.assertLess(move['end'][8], move['start'][8])
            self.assertGreater(move['end'][9], move['start'][9])
            self.assertGreater(move['end'][10], move['start'][10])
        sweeps = [m for m in plan if m['pose_id'].startswith('sweep-forward-')]
        self.assertEqual(len(sweeps), 3)
        for reach, sweep in zip(upward, sweeps):
            self.assertEqual(sweep['start'], reach['end'])
            self.assertLess(sweep['end'][9], 800 * 4)
            self.assertTrue(1450 * 4 <= sweep['end'][10] <= 1550 * 4)
            for ch in t.CHANNELS:
                self.assertGreaterEqual(abs(sweep['end'][ch] - sweep['start'][ch]), 150 * 4)
        for move in plan:
            self.assertTrue(c.clearance_box(move['start'], move['end']))
            self.assertNotEqual(move['start'][8], move['end'][8])
            self.assertNotEqual(move['start'][9], move['end'][9])
            for ch,value in move['end'].items():
                self.assertTrue(self.limits[ch]['min'] <= value <= self.limits[ch]['max'])

    def test_upward_choreography_moves_all_three_with_visible_wrist_travel(self):
        plan = c.make_reach_plan(c.REACH_AND_UP_US)
        for move in plan[8:-3]:
            for ch in t.CHANNELS:
                self.assertNotEqual(move['start'][ch], move['end'][ch])
            self.assertGreaterEqual(abs(move['start'][10] - move['end'][10]), 300 * 4)

    def test_reach_study_always_moves_shoulder_and_elbow_together(self):
        plan = c.make_reach_plan()
        self.assertEqual(plan[0]['start'], t.PARK)
        self.assertEqual(plan[-1]['end'], t.PARK)
        self.assertEqual(len(plan), 11)
        for move in plan:
            self.assertNotEqual(move['start'][8], move['end'][8])
            self.assertNotEqual(move['start'][9], move['end'][9])
            self.assertTrue(c.clearance_box(move['start'], move['end']))
            self.assertTrue(t.respects_elbow_limit(move['end']))
            for ch, value in move['end'].items():
                self.assertTrue(self.limits[ch]['min'] <= value <= self.limits[ch]['max'])

    def test_reach_beats_lift_shoulder_open_elbow_and_uncurl_wrist(self):
        names = {'curious-reach','reach-forward','offer','extend-offer','reach-higher'}
        for move in c.make_reach_plan():
            if move['pose_id'] in names:
                self.assertLess(move['end'][8], move['start'][8])
                self.assertLess(move['end'][9], move['start'][9])
                self.assertGreater(move['end'][10], move['start'][10])
                self.assertLessEqual(move['end'][10], 1550 * 4)

    def test_reach_study_boxes_tolerate_independent_progress(self):
        for move in c.make_reach_plan(c.REACH_AND_UP_US):
            for f8 in (0,.25,.5,.75,1):
                for f9 in (0,.25,.5,.75,1):
                    for f10 in (0,.25,.5,.75,1):
                        pose={ch:round(move['start'][ch]+(move['end'][ch]-move['start'][ch])*f)
                              for ch,f in zip(t.CHANNELS,(f8,f9,f10))}
                        self.assertTrue(c.clearance_box(pose,pose))

    def test_shoulder_and_elbow_really_change_together(self):
        plan=c.make_plan()
        self.assertEqual(plan[0]['end'],{8:6544,9:6184,10:2048})
        self.assertTrue(plan[0]['combined'])
        self.assertTrue(plan[1]['combined'])
        self.assertEqual(sum(m['combined'] for m in plan),6)

    def test_all_poses_and_final_park_preserved(self):
        plan=c.make_plan()
        self.assertEqual([m['pose_id'] for m in plan if m['pose_id'] is not None],
                         [1,2,6,7,8,9,10,11,3,4,5,'park'])
        self.assertEqual(plan[-1]['end'],t.PARK)

    def test_direct_low_to_extended_shortcut_is_rejected(self):
        self.assertFalse(c.clearance_box(t.PARK,{8:2176,9:2601,10:5938}))
        # Even perfect timing must not authorize the broader unsafe rectangle.
        self.assertFalse(c.clearance_box({8:9088,9:5589,10:2048},{8:2176,9:9984,10:2048}))

    def test_wrist_boundary_at_shoulder_1636(self):
        start={8:6544,9:2000,10:6000}
        self.assertTrue(c.clearance_box(start,{8:2176,9:2600,10:2048}))
        self.assertFalse(c.clearance_box({**start,10:6001},{8:2176,9:2600,10:2048}))

    def test_every_combined_move_tolerates_independent_progress(self):
        for move in c.make_plan():
            if not move['combined']:
                continue
            for f8 in (0,.25,.5,.75,1):
                for f9 in (0,.25,.5,.75,1):
                    for f10 in (0,.25,.5,.75,1):
                        pose={ch:round(move['start'][ch]+(move['end'][ch]-move['start'][ch])*f)
                              for ch,f in zip(t.CHANNELS,(f8,f9,f10))}
                        self.assertTrue(t.respects_elbow_limit(pose))
                        self.assertTrue(c.clearance_box(pose,pose))

    def test_controller_sends_one_batch_with_bounded_profiles(self):
        port=mock.Mock()
        port.write.side_effect=len
        controller=c.CoordinatedController(port,self.limits)
        start=t.PARK.copy()
        end={8:6544,9:6184,10:2048}
        with mock.patch.object(controller,'read',side_effect=[start,end]):
            controller.move_pose(end)
        packets=[call.args[0] for call in port.write.call_args_list]
        self.assertEqual(packets[-1][:3],bytes([0x9F,3,8]))
        self.assertEqual(len([p for p in packets if p[0]==0x9F]),1)
        for packet in packets[:-1]:
            value=packet[2]+128*packet[3]
            key='speed' if packet[0]==0x87 else 'acceleration'
            self.assertTrue(1<=value<=self.limits[packet[1]][key])

    def test_controller_rejects_unsafe_combination_without_writing(self):
        port=mock.Mock()
        controller=c.CoordinatedController(port,self.limits)
        with mock.patch.object(controller,'read',return_value=t.PARK), self.assertRaises(ValueError):
            controller.move_pose({8:2176,9:2601,10:5938})
        port.write.assert_not_called()


if __name__=='__main__':
    unittest.main()
