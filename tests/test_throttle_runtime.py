"""Throttle animation contracts with simulated Maestro I/O; never move hardware."""
import itertools
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import config
from hardware import servos, throttle_motion as motion
from sequences import throttle_arm as arm


class Maestro:
    def __init__(self, pose=None):
        self.pose = dict(motion.PARK if pose is None else pose)
        self.is_open = True
        self.packets = []
        self.channel = None
        self.on_target = lambda pose: None

    def write(self, packet):
        self.packets.append(bytes(packet))
        if packet[0] == 0x90:
            self.channel = packet[1]
        elif packet[0] == 0x9F:
            self.pose = {ch: packet[3 + i * 2] + 128 * packet[4 + i * 2]
                         for i, ch in enumerate(motion.CHANNELS)}
            self.on_target(dict(self.pose))
        return len(packet)

    def read(self, count):
        return self.pose[self.channel].to_bytes(count, 'little')

    def close(self):
        self.is_open = False

    @property
    def targets(self):
        return [p for p in self.packets if p[0] == 0x9F]


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.port = Maestro()
        for patch in (
            mock.patch('serial.Serial', side_effect=AssertionError('No live serial in tests')),
            mock.patch.object(motion, 'STATE_FILE', Path(self.temp.name) / 'park.json'),
            mock.patch.object(config, 'THROTTLE_ARM_ENABLED', True),
            mock.patch.object(servos, 'SERVOS_ENABLED', True),
            mock.patch.object(servos, '_ser', self.port),
            mock.patch.object(servos, '_program_servo_updates_blocked', return_value=False),
            mock.patch.object(servos, '_automatic_motion_allowed', return_value=True),
            mock.patch.object(arm, '_controller', None),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(arm.stop)
        self.limits = {cfg['ch']: cfg for cfg in config.THROTTLE_SERVO_CHANNELS.values()}

    def send(self, target, **kwargs):
        servos.move_throttle_pose(self.port, target, speed_caps=config.THROTTLE_SPEECH_SPEED,
                                 accel_caps=config.THROTTLE_SPEECH_ACCEL, duration=3.5, **kwargs)

    def test_every_repertoire_path_stays_clear_for_independent_joint_progress(self):
        arm.validate_repertoire()
        normal = arm.IDLE + tuple(p for gesture in arm.SPEECH for p in gesture)
        edges = list(itertools.product(normal, normal + (arm.TUCK,)))
        edges += list(zip((motion.PARK,) + arm.STARTUP, arm.STARTUP))
        edges.append((arm.TUCK, motion.PARK))
        for start, end in edges:
            for fractions in itertools.product((0, .5, 1), repeat=3):
                pose = {ch: round(start[ch] + (end[ch] - start[ch]) * f)
                        for ch, f in zip(motion.CHANNELS, fractions)}
                motion.validate_pose(pose, self.limits)
                self.assertTrue(motion.clearance_box(pose, pose))

    def test_background_never_extends_forward_and_speech_moves_all_joints(self):
        normal = arm.IDLE + tuple(p for gesture in arm.SPEECH for p in gesture)
        for target in normal:
            self.assertGreaterEqual(target[9], 1800 * 4)
            self.assertLessEqual(target[8], 1200 * 4)
        for start, end in arm.SPEECH:
            for ch in motion.CHANNELS:
                self.assertNotEqual(start[ch], end[ch])
            self.assertGreaterEqual(abs(end[10] - start[10]), 900 * 4)
        self.assertEqual(arm.STARTUP[0][10], motion.PARK[10])

    def test_emotions_and_pride_use_clear_paths_and_decay(self):
        normal = arm.IDLE + tuple(p for gesture in arm.SPEECH for p in gesture)
        for base, mood, intensity, pride in itertools.product(
                normal, ('sad', 'bored', 'resigned', 'excited', 'giddy', 'neutral'),
                (0, .2, .5, 1), (False, True)):
            target = arm.expressive_pose(base, mood, intensity, pride)
            motion.validate_pose(target, self.limits)
            self.assertTrue(motion.clearance_box(target, arm.REST))
            self.assertTrue(motion.clearance_box(arm.REST, target))
        self.assertEqual(arm.expressive_pose(arm.REST, 'sad', 0), arm.REST)
        self.assertGreater(arm.expressive_pose(arm.REST, 'sad', 1)[8], arm.REST[8])
        self.assertLess(arm.expressive_pose(arm.REST, 'excited', 1)[8], arm.REST[8])
        self.assertEqual(arm.expressive_pose(arm.REST, pride=True)[10], arm.PRIDE_WRIST)
        self.assertEqual(arm.expressive_pose(arm.REST, pride=True, introducing=True),
                         arm.INTRODUCTION)

    def test_moderate_sadness_stays_low_through_every_speech_pose(self):
        # Reproduce the ambient fallback (0.4) and sad speech profile (0.48).
        for intensity in (.4, .48):
            for base in (arm.REST,) + tuple(p for gesture in arm.SPEECH for p in gesture):
                target = arm.expressive_pose(base, 'sad', intensity)
                self.assertGreaterEqual(target[8], 1500 * 4)
                self.assertLessEqual(target[9], 1650 * 4)
                self.assertTrue(motion.clearance_box(target, arm.REST))
                self.assertTrue(motion.clearance_box(target, arm.TUCK))
        self.assertEqual(arm.expressive_pose(arm.REST, 'sad', 0), arm.REST)

    def test_sad_reply_overrides_offended_mood_until_speech_settles(self):
        controller = arm.Controller()
        with (mock.patch.object(arm, '_controller', controller),
              mock.patch('intelligence.body_mood.current_mood', return_value=('offended', .9)),
              mock.patch('intelligence.pride.is_active', return_value=False),
              mock.patch.object(arm.time, 'monotonic', return_value=10) as clock):
            arm.speech_start({'affect': 'sad', 'intensity': .48})
            self.assertEqual(arm.expression_state(), ('sad', .48, False))
            clock.return_value = 30  # Long speech retains the explicit frame.
            self.assertEqual(arm.expression_state(), ('sad', .48, False))
            arm.speech_stop()
            self.assertEqual(arm.expression_state(), ('sad', .48, False))
            clock.return_value = 30 + config.THROTTLE_SPEECH_SETTLE_SECS + .1
            self.assertEqual(arm.expression_state(), ('offended', .9, False))
            arm.speech_start({'affect': 'excited', 'intensity': .8})
            self.assertEqual(arm.expression_state(), ('excited', .8, False))
            controller.done.set()
            self.assertEqual(arm.expression_state(), ('offended', .9, False))

    def test_lowered_pride_to_introduction_and_park_bridge(self):
        controller = arm.Controller()
        controller.connection = self.port
        self.port.pose = arm.expressive_pose(arm.LOW, pride=True)
        self.assertTrue(controller.move(arm.INTRODUCTION, 'IDLE', 2.5))
        self.assertEqual(len(self.port.targets), 2)
        self.assertEqual(self.port.pose, arm.INTRODUCTION)
        controller._park()
        self.assertTrue(controller.parked)

    def test_down_command_reaches_recorded_pose_and_retracts_in_stages(self):
        import csv
        with open('data/throttle_measurements.csv') as source:
            recorded = next(r for r in csv.DictReader(source)
                            if r['note'] == 'Arm fully down including forearm and wrist')
        expected = arm.pose(*(float(recorded[k]) for k in
                              ('shoulder_us', 'elbow_us', 'wrist_us')))
        self.assertEqual(arm.COMMAND_POSES['down'], expected)
        controller = arm.Controller()
        controller.connection = self.port
        self.port.pose = dict(arm.REST)
        seen = []
        self.port.on_target = lambda p: seen.append(p)
        self.assertTrue(controller.move(expected, 'IDLE', 2))
        self.assertEqual(seen, [motion.RETRACT_RAISED, motion.FULL_DOWN_RAISED, expected])
        seen.clear()
        with mock.patch.object(config, 'THROTTLE_RETRACT_SETTLE_SECS', 0):
            controller._park(for_base=True)
        self.assertTrue(controller.parked)
        self.assertEqual(seen, [motion.FULL_DOWN_RAISED, motion.RETRACT_RAISED,
                                motion.TUCK, motion.PARK])

    def test_full_down_exception_does_not_relax_other_low_shoulder_poses(self):
        self.assertFalse(motion.clearance_box(arm.REST, motion.FULL_DOWN))
        for ch in (9, 10):
            other = dict(motion.FULL_DOWN)
            other[ch] += 40
            self.assertFalse(motion.clearance_box(other, other))
        partial = dict(motion.FULL_DOWN)
        partial[8] = 1900 * 4
        self.port.pose = partial
        controller = arm.Controller()
        controller.connection = self.port
        self.assertTrue(controller.move(arm.INTRODUCTION, 'IDLE', 2))
        self.assertEqual(self.port.pose, arm.INTRODUCTION)

    def test_worker_applies_live_mood_then_parks(self):
        controller = arm.Controller()
        seen = []
        target = arm.expressive_pose(arm.REST, 'sad', 1, True)
        def arrived(pose):
            seen.append(pose)
            if pose == target:
                controller.park_event.set()
        self.port.on_target = arrived
        with mock.patch.object(arm, 'expression_state', return_value=('sad', 1, True)):
            controller.run()
        self.assertIn(target, seen)
        self.assertIsNone(controller.fault)
        self.assertTrue(controller.parked)

    def test_introduction_callback_cannot_start_worker(self):
        arm.introduction()
        self.assertIsNone(arm._controller)
        controller = arm.Controller()
        with mock.patch.object(arm, '_controller', controller), mock.patch.object(arm.time, 'monotonic', return_value=10):
            arm.introduction()
            self.assertEqual(controller.introduction_until, 18)
            controller.done.set()
            controller.introduction_until = 0
            arm.introduction()
            self.assertEqual(controller.introduction_until, 0)

    def test_wire_profiles_are_capped_and_all_targets_share_one_packet(self):
        self.send(arm.TUCK)
        self.assertEqual(len(self.port.targets), 1)
        self.assertEqual(self.port.targets[0][:3], bytes([0x9F, 3, 8]))
        self.assertEqual(self.port.pose, arm.TUCK)
        for packet in self.port.packets:
            if packet[0] in (0x87, 0x89):
                cap = config.THROTTLE_SPEECH_SPEED if packet[0] == 0x87 else config.THROTTLE_SPEECH_ACCEL
                self.assertTrue(1 <= packet[2] + 128 * packet[3] <= cap[packet[1]])

    def test_unsafe_shortcut_and_board_clipping_rejected_without_targets(self):
        for target in (arm.pose(1100, 600, 2200), arm.pose(535, 2000, 1500)):
            with self.assertRaises(ValueError):
                self.send(target)
        self.assertFalse(self.port.targets)

    def test_short_write_closes_connection_without_reconnect_or_targets(self):
        original = self.port.write
        def write(packet):
            return 0 if packet[0] == 0x87 else original(packet)
        with mock.patch.object(self.port, 'write', side_effect=write), \
                mock.patch.object(servos, '_open_serial_with_retries') as reconnect:
            with self.assertRaises(OSError):
                self.send(arm.TUCK)
            reconnect.assert_not_called()
        self.assertFalse(self.port.targets)
        self.assertIsNone(servos._ser)

    def test_reconnected_port_cannot_replay_worker_targets(self):
        replacement = Maestro()
        with mock.patch.object(servos, '_ser', replacement), self.assertRaises(RuntimeError):
            self.send(arm.TUCK)
        self.assertFalse(replacement.packets)

    def test_cancel_and_manual_override_block_targets(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            self.send(arm.TUCK, cancel=stop)
        with mock.patch.object(servos, '_program_servo_updates_blocked', return_value=True), \
                self.assertRaises(InterruptedError):
            self.send(arm.TUCK)
        self.assertFalse(self.port.targets)

    def test_latched_head_allows_only_tuck_and_park_and_still_honors_manual_control(self):
        self.port.pose = dict(arm.REST)
        with (mock.patch.object(servos, '_program_servo_updates_blocked', return_value=True),
              mock.patch.object(servos._manual_override, 'is_set', return_value=False)):
            with self.assertRaises(InterruptedError):
                self.send(arm.TUCK)
            with self.assertRaises(ValueError):
                self.send(arm.REST, parking=True)
            self.send(arm.TUCK, parking=True)
            self.send(motion.PARK, parking=True)
            count = len(self.port.packets)
            servos.set_servo(0, 6000)
            self.assertEqual(len(self.port.packets), count)
            with mock.patch.object(servos._manual_override, 'is_set', return_value=True), \
                    self.assertRaises(InterruptedError):
                self.send(arm.TUCK, parking=True)
        self.assertEqual(len(self.port.targets), 2)

    def test_head_latch_racing_a_target_write_transfers_worker_to_parking(self):
        original = servos.move_throttle_pose
        requests = []
        def send(*args, **kwargs):
            requests.append(kwargs['parking'])
            if len(requests) == 1:
                arm.request_park()
                servos._program_servo_updates_blocked.return_value = True
            return original(*args, **kwargs)
        with (mock.patch.object(servos, 'move_throttle_pose', side_effect=send),
              mock.patch.object(servos._manual_override, 'is_set', return_value=False)):
            self.assertTrue(arm.start())
            self.assertTrue(arm._controller.done.wait(2.0))
        self.assertEqual(requests, [False, True, True])
        self.assertIsNone(arm._controller.fault)
        self.assertTrue(arm._controller.parked)
        self.assertEqual(self.port.pose, motion.PARK)

    def test_runtime_pace_reaches_wire_even_when_duration_limits_speed(self):
        self.port.pose = dict(arm.REST)
        target = {ch: value + 2000 for ch, value in arm.REST.items()}
        servos.move_throttle_pose(
            self.port, target, speed_caps=config.THROTTLE_SPEECH_SPEED,
            accel_caps=config.THROTTLE_SPEECH_ACCEL, duration=5,
        )
        speeds = {p[1]: p[2] + 128 * p[3] for p in self.port.packets if p[0] == 0x87}
        # Equal travel formerly sent 4 to every joint. Pace must affect actual
        # packets, not merely raise caps that a slow requested duration defeats.
        self.assertEqual(speeds, {8: 5, 9: 6, 10: 8})

    def test_startup_speed_is_brisk_but_within_configured_joint_caps(self):
        servos.move_throttle_pose(
            self.port, arm.TUCK, speed_caps=config.THROTTLE_STARTUP_SPEED,
            accel_caps=config.THROTTLE_STARTUP_ACCEL, duration=config.THROTTLE_STARTUP_MOVE_SECS,
        )
        shoulder_speed = next(p[2] + 128 * p[3] for p in self.port.packets if p[:2] == bytes([0x87, 8]))
        self.assertGreaterEqual(shoulder_speed, 18)
        for kind in ('STARTUP', 'IDLE', 'SPEECH', 'PARK', 'RETRACT'):
            for ch in motion.CHANNELS:
                self.assertLessEqual(getattr(config, f'THROTTLE_{kind}_SPEED')[ch], self.limits[ch]['speed'])
                self.assertLessEqual(getattr(config, f'THROTTLE_{kind}_ACCEL')[ch], self.limits[ch]['acceleration'])

    def test_cold_start_requires_clean_park_and_invalidates_it_before_moving(self):
        self.port.pose = {ch: 0 for ch in motion.CHANNELS}
        with self.assertRaises(ValueError):
            self.send(motion.PARK, cold_start=True)
        motion.remember_park()
        self.port.on_target = lambda pose: self.assertFalse(motion.cold_start_park_known())
        self.send(motion.PARK, cold_start=True)
        self.assertEqual(self.port.pose, motion.PARK)
        self.assertFalse(motion.cold_start_park_known())

    def test_partially_off_arm_is_not_treated_as_parked(self):
        motion.remember_park()
        self.port.pose[8] = 0
        with self.assertRaises(ValueError):
            self.send(motion.PARK, cold_start=True)
        self.assertFalse(self.port.targets)

    def test_full_startup_speech_and_park_through_actual_transport_code(self):
        ready, spoke = threading.Event(), threading.Event()
        seen = []
        def arrived(pose):
            seen.append(pose)
            if pose == arm.REST:
                ready.set()
            if pose in [p for gesture in arm.SPEECH for p in gesture]:
                spoke.set()
        self.port.on_target = arrived
        self.assertTrue(arm.start())
        self.assertTrue(ready.wait(2.0))
        self.assertEqual(seen[:len(arm.STARTUP)], list(arm.STARTUP))
        arm.speech_start()
        with arm._controller.lock:
            now = time.monotonic()
            arm._controller.cadence.level(.5, now - 2)
            arm._controller.cadence.level(.5, now)
        self.assertTrue(spoke.wait(2.0))
        self.assertTrue(arm.park())
        self.assertEqual(self.port.pose, motion.PARK)
        self.assertEqual(seen[-2:], [arm.TUCK, motion.PARK])
        self.assertTrue(motion.cold_start_park_known())
        count = len(self.port.targets)
        arm.speech_start()
        arm.speech_level(1.0)
        arm.speech_stop()
        self.assertEqual(len(self.port.targets), count)

    def test_startup_fault_does_not_enable_autonomous_reposition_or_speech_restart(self):
        self.port.pose = dict(arm.REST)
        self.assertTrue(arm.start())
        self.assertTrue(arm._controller.done.wait(2.0))
        self.assertIsNotNone(arm._controller.fault)
        self.assertFalse(arm.start())
        # Only an at-current-pulse hold may be sent after rejecting startup.
        self.assertEqual(self.port.pose, arm.REST)

    def test_sleep_state_requests_parking_and_holds_off_speech(self):
        ready = threading.Event()
        self.port.on_target = lambda pose: ready.set() if pose == arm.REST else None
        self.assertTrue(arm.start())
        self.assertTrue(ready.wait(2.0))
        with mock.patch.object(servos, '_automatic_motion_allowed', return_value=False):
            self.assertTrue(arm._controller.done.wait(2.0))
        self.assertTrue(arm._controller.parked)

    def test_manual_control_invalidates_cold_start_marker(self):
        from tools import rex_servo_menubar as helper
        motion.remember_park()
        helper._write_target(self.port, config.THROTTLE_SERVO_CHANNELS['throttle_wrist'], 6000)
        self.assertFalse(motion.cold_start_park_known())

    def test_disabled_runtime_does_not_create_worker(self):
        with mock.patch.object(config, 'THROTTLE_ARM_ENABLED', False):
            self.assertFalse(arm.start())
        self.assertIsNone(arm._controller)


class CadenceTest(unittest.TestCase):
    def test_short_acknowledgments_do_not_get_a_gesture(self):
        c = arm.SpeechCadence()
        c.begin(0)
        for i in range(10):
            c.level(.5, i / 10)
            self.assertFalse(c.due(i / 10))
        c.end(1)
        self.assertFalse(c.due(2))

    def test_gaps_not_every_word_and_cooldown_survives_sentence_boundaries(self):
        c = arm.SpeechCadence()
        c.begin(0)
        c.level(.5, 0)
        c.level(.5, 1.3)
        self.assertTrue(c.due(1.3))
        c.consume(1.3, 8)
        c.end(2)
        c.begin(2.1)
        c.level(.5, 2.1)
        for i in range(30, 92):
            c.level(.5 if i % 2 else 0, i / 10)
            self.assertFalse(c.due(i / 10))
        c.level(0, 9.5)
        c.level(0, 10.0)
        self.assertTrue(c.due(10.0))
        c.end(10.1)
        self.assertFalse(c.due(10.2))

    def test_pauses_queue_only_one_expiring_cue(self):
        c = arm.SpeechCadence()
        c.begin(0)
        c.level(.5, 0)
        c.level(.5, 1.3)
        c.level(0, 2)
        c.level(0, 2.5)
        self.assertTrue(c.due(2.5))
        self.assertFalse(c.due(5))

    def test_long_unbroken_speech_has_sparse_fallback_but_silence_does_not(self):
        c = arm.SpeechCadence()
        c.begin(0)
        c.level(.5, 0)
        c.level(.5, 1.3)
        c.consume(1.3, 4)
        for i in range(2, 8):
            c.level(.5, i)
            self.assertFalse(c.due(i))
        # Fallback is measured from the previous gesture, not added after cooldown.
        c.level(.5, 8)
        self.assertTrue(c.due(8))
        c.level(0, 9)
        self.assertFalse(c.due(9))


class AnimationWiringTest(unittest.TestCase):
    def test_head_parks_and_latches_before_waiting_for_throttle_completion(self):
        from sequences import animations
        events = []
        fake_servos = mock.Mock()
        fake_arm = mock.Mock()
        fake_arm.request_park.side_effect = lambda: events.append('request-park')
        fake_arm.park.side_effect = lambda: events.append('wait-for-park')
        fake_servos.move_to.side_effect = lambda *args, **kwargs: events.append('head-move')
        fake_servos.latch_sleep_pose.side_effect = lambda: events.append('sleep-latch')
        fake_servos.latch_shutdown_pose.side_effect = lambda: events.append('shutdown-latch')
        with (mock.patch.object(animations, 'servos', fake_servos),
              mock.patch.object(animations, 'throttle_arm', fake_arm),
              mock.patch.object(animations, 'leds_head'), mock.patch.object(animations, 'leds_chest'),
              mock.patch.object(animations.time, 'sleep'),
              mock.patch('intelligence.voice_learning.mic_moved')):
            animations.startup()
            animations.wake()
            self.assertEqual(fake_arm.start.call_count, 2)
            events.clear()
            animations.sleep()
            animations.shutdown()
        self.assertEqual(events, [
            'request-park', 'head-move', 'head-move', 'sleep-latch', 'wait-for-park',
            'request-park', 'head-move', 'shutdown-latch', 'wait-for-park',
        ])


if __name__ == '__main__':
    unittest.main()
