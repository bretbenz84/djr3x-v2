"""Mac owns caller arrival; ESP32 only receives motion primitives. No live I/O."""
import time
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from intelligence.approach import Approach, target_in_frame, face_range_m


def telemetry(now, front=4000, **changes):
    row = dict(rx_monotonic=now, state='moving', blocked_dir='none',
        odom={'x': 0., 'y': 0., 'lin': .0},
        tof_mm={'fl': front, 'fr': front, 'lf': 2000, 'lb': 2000, 'rf': 2000, 'rb': 2000})
    row.update(changes)
    return row


class ApproachTests(unittest.TestCase):
    def test_passing_three_feet_is_not_arrival_with_caller_still_far(self):
        plan = Approach(0, 1.3, .4)
        for tick in range(1, 35):
            now = tick*.1
            result = plan.step(now, telemetry(now), 3.0-now*.3, 0.)
            self.assertIsNone(result.result)
            self.assertGreater(result.lin, .1)

    def test_camera_arrival_sheds_speed_and_requires_stable_near_target(self):
        plan = Approach(0, 1.3, .4)
        distance, speed = 3., 0.
        for tick in range(1, 190):
            now = tick*.1
            row = telemetry(now, odom={'x': 3-distance, 'y': 0, 'lin': speed})
            result = plan.step(now, row, distance, 0)
            speed += max(-.035, min(.035, result.lin-speed))
            distance -= speed*.1
            if result.result:
                break
        self.assertEqual(result.result, 'completed')
        self.assertGreater(distance, 1.1)
        self.assertLess(distance, 1.4)
        self.assertLess(speed, .1)

    def test_obstacle_is_a_pause_not_person_arrival(self):
        plan = Approach(0, 1.3, .4)
        plan.step(.1, telemetry(.1), 3., 0)
        hit = plan.step(.2, telemetry(.2, front=70, state='blocked', blocked_dir='front'), 3., 0)
        self.assertIsNone(hit.result)
        self.assertEqual(hit.lin, 0)
        resumed = plan.step(.6, telemetry(.6), 3., 0)
        self.assertGreater(resumed.lin, 0)
        self.assertIsNone(resumed.result)
        plan.step(1, telemetry(1, front=70), 3., 0)
        self.assertEqual(plan.step(7.1, telemetry(7.1, front=70), 3., 0).result, 'blocked')

    def test_camera_loss_is_bounded_and_does_not_forge_arrival(self):
        plan = Approach(0, 1.3, .4)
        plan.step(.1, telemetry(.1), 3., 0)
        self.assertGreater(plan.step(.5, telemetry(.5), None, None).lin, 0)
        self.assertEqual(plan.step(2, telemetry(2), None, None).lin, 0)
        self.assertEqual(plan.step(9, telemetry(9), None, None).result, 'aborted')

    def test_front_right_hold_keeps_fresh_caller_observations(self):
        plan = Approach(0, 1.3, .4)
        plan.step(.1, telemetry(.1), 3., 0)
        for tick in range(2, 30):
            now = tick*.1
            held = telemetry(now)
            held['tof_mm'].update(fr=65, fr_radial=65)
            result = plan.step(now, held, 2.8, 0, target_stamp=now)
            self.assertEqual(result.lin, 0.)
            self.assertIsNone(result.result)
        # Clearance returns between camera frames: no artificial second pause.
        resumed = plan.step(3., telemetry(3.), None, None)
        self.assertGreater(resumed.lin, 0.)
        self.assertAlmostEqual(plan.last_range, 2.8)

    def test_one_cached_near_face_cannot_finish_arrival(self):
        plan = Approach(0, 1.3, .4)
        for tick in range(1, 12):
            now=tick*.1
            result=plan.step(now, telemetry(now), 1.2, 0, target_stamp=123.)
            self.assertIsNone(result.result)
            self.assertEqual(result.lin, 0)
        self.assertEqual(plan.step(1.2, telemetry(1.2), None, None).lin, 0)

    def test_stale_telemetry_and_blind_front_abort(self):
        self.assertEqual(Approach(0, 1.3, .4).step(1, telemetry(0), 3., 0).result, 'aborted')
        self.assertEqual(Approach(0, 1.3, .4).step(.1, telemetry(.1, front=-1), 3., 0).result, 'aborted')

    def test_steering_yields_to_obstacles_then_faces_caller(self):
        plan = Approach(0, 1.3, .4)
        self.assertEqual(plan.step(.1, telemetry(.1, front=850), 3., 20).ang, 0)
        self.assertEqual(plan.step(.2, telemetry(.2), 3., 20).ang, 0)
        self.assertGreater(plan.step(.7, telemetry(.7), 3., 20).ang, 0)
        self.assertEqual(plan.step(.8, telemetry(.8, front=850), 3., 20).ang, 0)

    def test_target_recognition_change_does_not_change_destination(self):
        face = {'id': 'track', 'face_visible': True, 'face_box': (800, 200, 160, 160)}
        for pid in (None, 1):
            row = dict(face, person_db_id=pid)
            self.assertIs(target_in_frame({'people': [row]}, 1, 'track'), row)
        self.assertIsNone(target_in_frame({'people': [dict(face, person_db_id=4)]}, 1, 'track'))
        self.assertAlmostEqual(face_range_m(face, 1920), 2.059, delta=.01)


class MacTransportTests(unittest.TestCase):
    def setUp(self):
        from intelligence import motion_controller as MC
        self.mc = MC
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = [100.]
        self.sent = []
        self.face = {'id': 'track', 'person_db_id': 1, 'face_box': (800,200,120,120),
                     'face_visible': True, 'face_last_seen_at': 100.}
        self.scene = {'people': [self.face], 'self_state': {'frame_size': {'width': 1920}}}
        def send(payload):
            self.sent.append(payload)
            return len(self.sent)
        for name, value in {'_host_approach': None, '_last_come_seq': None, '_last_come_result': None,
                            '_last_come_detail': {}, '_arc_active': False}.items():
            self.stack.enter_context(patch.object(MC, name, value))
        self.stack.enter_context(patch.object(MC.time, 'monotonic', side_effect=lambda: self.clock[0]))
        self.stack.enter_context(patch.object(MC.time, 'time', side_effect=lambda: self.clock[0]))
        for name in ('_note_issued', '_fx_drive_loop_start', '_fx_drive_loop_stop', '_fx_drive_loop_stop_all', '_fx', '_handle_turn_verification_done'):
            self.stack.enter_context(patch.object(MC, name))
        self.stack.enter_context(patch.object(MC, '_autonomous_allowed', return_value=None))
        self.stack.enter_context(patch.object(MC, '_tof_should_cut_inflight', return_value=False))
        self.stack.enter_context(patch.object(MC.motion, 'connected', return_value=True))
        self.stack.enter_context(patch.object(MC.motion, 'caps', return_value=['drive','turn','move','stop']))
        self.stack.enter_context(patch.object(MC.motion, 'send', side_effect=send))
        self.stack.enter_context(patch.object(MC.motion, 'ping'))
        self.stack.enter_context(patch.object(MC.motion, 'telemetry', side_effect=lambda: telemetry(self.clock[0])))
        self.stack.enter_context(patch('world_state.world_state.snapshot', side_effect=lambda: self.scene))
        self.stack.enter_context(patch('intelligence.motion_agency._come_bearing_deg', return_value=0.))

    def test_host_refreshes_primitives_and_stop_cannot_be_overwritten(self):
        self.assertIsNotNone(self.mc.come(stop_at=1.3, target=self.face))
        for _ in range(8):
            self.clock[0]+=.15
            self.face['face_last_seen_at']=self.clock[0]
            self.mc._heartbeat_tick()
        self.assertTrue(any(row.get('lin',0)>0 for row in self.sent))
        self.assertTrue(all(row['cmd']=='drive' for row in self.sent))
        self.mc.stop()
        count = len(self.sent)
        self.mc._heartbeat_tick()
        self.assertEqual(len(self.sent), count)
        self.assertEqual(self.mc.last_come_result()[1], 'aborted')

    def test_mac_arrival_is_not_a_firmware_done_event(self):
        seq=self.mc.come(stop_at=1.3, target=self.face)
        self.mc._on_motion_done({'seq': seq, 'result': 'completed'})
        self.assertIsNone(self.mc.last_come_result()[1])
        self.face['face_box']=(800,200,260,260)
        for _ in range(8):
            self.clock[0]+=.15
            self.face['face_last_seen_at']=self.clock[0]
            self.mc._heartbeat_tick()
        self.assertEqual(self.mc.last_come_result()[1], 'completed')
        self.assertEqual(self.mc.last_come_detail()['owner'], 'mac')

    def test_stale_camera_never_starts_driving(self):
        self.face['face_last_seen_at']=0.
        self.mc.come(stop_at=1.3, target=self.face)
        for _ in range(8):
            self.clock[0]+=.15
            self.mc._heartbeat_tick()
        self.assertTrue(all(row.get('lin', 0)==0 for row in self.sent))
