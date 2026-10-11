"""hardware/servos.py on the two-servo head and the poker elbow/wrist (October 10 2026).

Fake serial only: the driver must refuse a lone head or poker channel, refuse a head
pair the linkage cannot take, keep the two head servos arriving together through a
glide, stage every poker move under the clearance ceiling, read live channels on a
cold connect instead of yanking them to rest, and name the channel that never
arrived.
"""

import struct
import unittest
from unittest import mock

import config
from hardware import head_pose as hp
from hardware import servos


class FakeSerial:
    """Records every Set Target and answers Get Position from a position table."""
    is_open = True

    def __init__(self, positions=None):
        self.writes = []
        self.positions = dict(positions or {})
        self._pending = None

    def write(self, raw):
        raw = bytes(raw)
        self.writes.append(raw)
        if raw[0] == servos._CMD_GET_POSITION:
            self._pending = raw[1]
        elif raw[0] == servos._CMD_SET_TARGET:
            self.positions[raw[1]] = raw[2] | (raw[3] << 7)
        return len(raw)

    def read(self, n):
        ch, self._pending = self._pending, None
        return struct.pack("<H", int(self.positions.get(ch, 0)))

    def close(self):
        self.is_open = False

    def targets(self):
        return [(w[1], w[2] | (w[3] << 7)) for w in self.writes if w[0] == servos._CMD_SET_TARGET]


class _Live(unittest.TestCase):
    """A driver talking to a fake board."""

    def setUp(self):
        self.fake = FakeSerial()
        self._patches = [
            mock.patch.object(servos, "SERVOS_ENABLED", True),
            mock.patch.object(servos, "_ser", self.fake),
            mock.patch.object(servos, "_record_servo_positions"),
            mock.patch.object(servos.time, "sleep"),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        servos._manual_override.clear()
        servos._shutdown_latch.clear()
        servos._sleep_latch.clear()
        self.saved = dict(servos._commanded_positions)
        self.addCleanup(lambda: servos._commanded_positions.update(self.saved))
        servos._commanded_positions.update({**hp.head_pulses(0, 0), 0: 5472, 1: 3600, 3: 5120,
                                            12: 6000, 13: 3768})


class RefusalTest(_Live):
    def test_a_lone_head_channel_is_refused_before_the_wire(self):
        for call in (lambda: servos.set_servo(2, 5832), lambda: servos.set_servos({11: 6480}),
                     lambda: servos.move_to({2: 6000, 0: 5472})):
            with self.assertRaisesRegex(ValueError, "IncompleteHeadPose"):
                call()
        self.assertEqual(self.fake.targets(), [])

    def test_a_pair_outside_the_envelope_is_refused(self):
        with self.assertRaisesRegex(ValueError, "HeadPairOutsideEnvelope"):
            servos.set_servos({2: 8128, 11: 7616})
        self.assertEqual(self.fake.targets(), [])

    def test_a_valid_pair_goes_out_together(self):
        servos.set_servos(hp.head_pulses(0.5, -0.3))
        sent = dict(self.fake.targets())
        self.assertEqual(sent, hp.head_pulses(0.5, -0.3))

    def test_set_head_pose_keeps_the_other_axis(self):
        servos.set_servos(hp.head_pulses(-0.4, 0.5))
        servos.set_head_pose(pitch=0.2)
        pitch, tilt = hp.head_pose(dict(self.fake.targets()[-2:]))
        self.assertAlmostEqual(pitch, 0.2, places=1)
        self.assertAlmostEqual(tilt, 0.5, places=1)
        self.assertAlmostEqual(servos.commanded_head_pose()[1], 0.5, places=1)

    def test_poker_collision_is_refused_and_a_lone_joint_is_completed(self):
        with self.assertRaisesRegex(ValueError, "PokerElbowWristCollision"):
            servos.set_servos({12: 7680, 13: 9000})
        self.assertEqual(self.fake.targets(), [])
        # A lone wrist takes the elbow from where it is (6000) and must stay clear.
        servos.set_servos({13: 5000})
        self.assertIn((13, 5000), self.fake.targets())

    def test_head_gesture_hold_covers_both_servos(self):
        servos.begin_head_gesture([2, 3])
        try:
            self.assertEqual(servos.head_gesture_channels(), {2, 11, 3})
        finally:
            servos.end_head_gesture()


class GlideTest(_Live):
    def test_move_to_keeps_the_head_servos_arriving_together(self):
        start = hp.head_pulses(-1, 0)          # fully down: left 4052, right 7616
        target = hp.head_pulses(1, 0)          # fully up: left 8128, right 5248 (shorter travel)
        servos.move_to(target, step_us=60, step_delay=0.0, start=start)
        pairs = []
        left = right = None
        for ch, pos in self.fake.targets():
            if ch == 2:
                left = pos
            elif ch == 11:
                right = pos
            if left is not None and right is not None:
                pairs.append({2: left, 11: right})
        # Every intermediate pair is one the linkage can take, and both arrive on the
        # same step: the right servo never parks at its end while the left is mid-way.
        for pair in pairs:
            self.assertLessEqual(hp.head_fit_error(pair), hp.HEAD_ENVELOPE, pair)
        self.assertEqual(pairs[-1], target)
        arrivals = [i for i, p in enumerate(pairs) if p[11] == target[11]]
        self.assertEqual(arrivals[0], len(pairs) - 1)

    def test_move_to_routes_the_poker_pair_through_stages(self):
        with mock.patch.object(servos, "move_poker") as move_poker:
            servos.move_to({12: 7680, 13: 6256, 0: 5472}, step_us=400, step_delay=0.0, start={0: 5472})
        move_poker.assert_called_once()
        self.assertEqual(move_poker.call_args.args, (7680, 6256))
        self.assertEqual(dict(self.fake.targets()).get(12), None)


class PokerStagingTest(_Live):
    def test_awake_to_park_moves_the_wrist_first_then_the_elbow(self):
        servos.move_poker(7680, 6256, speed=12, acceleration=3, block=True)
        sent = self.fake.targets()
        stages = [sent[i:i + 2] for i in range(0, len(sent), 2)]
        self.assertEqual([dict(s) for s in stages], [{12: 6000, 13: 6256}, {12: 7680, 13: 6256}])
        self.assertEqual(servos._commanded_positions[12], 7680)
        self.assertEqual(servos._commanded_positions[13], 6256)

    def test_profile_is_applied_before_each_stage(self):
        servos.move_poker(7680, 6256, speed=12, acceleration=3, block=True)
        kinds = [w[0] for w in self.fake.writes]
        first_target = kinds.index(servos._CMD_SET_TARGET)
        self.assertIn(servos._CMD_SET_SPEED, kinds[:first_target])
        self.assertIn(servos._CMD_SET_ACCEL, kinds[:first_target])

    def test_parking_passes_the_shutdown_latch(self):
        servos._shutdown_latch.set()
        try:
            self.assertFalse(servos.move_poker(7680, 6256, block=True))
            self.assertTrue(servos.move_poker(7680, 6256, block=True, parking=True))
        finally:
            servos._shutdown_latch.clear()
        self.assertEqual(dict(self.fake.targets()[-2:]), {12: 7680, 13: 6256})


class ColdStartTest(unittest.TestCase):
    def test_off_channels_get_rest_and_live_channels_keep_their_reading(self):
        # The head pair and the poker arm read 0 (off since power-up); the hero elbow
        # reads a live pulse (powered through a restart): rest for the former only.
        fake = FakeSerial({4: 6900})
        saved = dict(servos._commanded_positions)
        try:
            with (
                mock.patch.object(servos, "SERVOS_ENABLED", True),
                mock.patch.object(servos, "_open_serial_with_retries", return_value=fake),
                mock.patch.object(servos, "_record_servo_positions"),
            ):
                self.assertTrue(servos.connect())
            targets = dict(fake.targets())
            self.assertEqual(targets, {2: 4052, 11: 7616, 12: 7680, 13: 6256})
            self.assertEqual(servos._commanded_positions[4], 6900)
            self.assertEqual(servos._commanded_positions[2], 4052)
            # The rest pose is a pair the linkage knows and a clear poker pose.
            hp.validate_head_pair({2: targets[2], 11: targets[11]})
            hp.validate_poker_pair({12: targets[12], 13: targets[13]})
        finally:
            servos._commanded_positions.update(saved)
            servos._ser = None

    def test_rest_pose_matches_the_owner_readings(self):
        self.assertEqual(config.servo_rest_position("head_left"), 4052)     # 1013 us
        self.assertEqual(config.servo_rest_position("head_right"), 7616)    # 1904 us
        self.assertEqual(config.servo_rest_position("poker_elbow"), 7680)   # 1920 us: the park
        self.assertEqual(config.servo_rest_position("poker_wrist"), 6256)   # 1564 us


class ArrivalTest(_Live):
    def test_timeout_names_the_short_channel_and_its_pulse(self):
        self.fake.positions.update({3: 6400, 2: 4052, 11: 7616})
        with mock.patch.object(servos.time, "monotonic", side_effect=[0.0, 0.0, 10.0, 10.0, 10.0, 10.0]):
            short = servos.wait_for_arrival({3: 6976, 2: 4052, 11: 7616}, timeout=1.0)
        self.assertEqual(short, "ch3@6400->6976")

    def test_arrival_returns_empty(self):
        self.fake.positions.update({3: 6976})
        self.assertEqual(servos.wait_for_arrival({3: 6976}, timeout=1.0), "")


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(servos._face_tracking_baseline)
        self.addCleanup(lambda: (servos._face_tracking_baseline.clear(),
                                 servos._face_tracking_baseline.update(self.saved)))
        servos._manual_override.clear()
        servos._shutdown_latch.clear()
        servos._sleep_latch.clear()

    def test_fractions_in_and_qus_refused(self):
        servos.set_face_tracking_baseline(neck=5472, lift=3600, pitch=-0.4, tilt=0.2)
        pair = {ch: servos._face_tracking_baseline[ch] for ch in hp.HEAD_PAIR}
        pitch, tilt = hp.head_pose(pair)
        self.assertAlmostEqual(pitch, -0.4, places=1)
        self.assertAlmostEqual(tilt, 0.2, places=1)
        servos.set_face_tracking_baseline(pitch=0.1)       # tilt kept
        pair = {ch: servos._face_tracking_baseline[ch] for ch in hp.HEAD_PAIR}
        self.assertAlmostEqual(hp.head_pose(pair)[1], 0.2, places=1)
        with self.assertRaises(ValueError):
            servos.set_face_tracking_baseline(tilt=4320)

    def test_body_state_reads_pitch(self):
        down = {"neck": 5472, "headlift": 3600, **{servos._CHANNEL_TO_NAME[ch]: v for ch, v in hp.head_pulses(-0.8, 0).items()}}
        up = {"neck": 5472, "headlift": 3600, **{servos._CHANNEL_TO_NAME[ch]: v for ch, v in hp.head_pulses(0.8, 0.5).items()}}
        self.assertEqual(servos._derive_body_state(down), "looking_down")
        self.assertEqual(servos._derive_body_state(up), "looking_up")


class TalkingAndListeningTest(unittest.TestCase):
    def test_talking_sways_the_pair_around_its_anchor(self):
        captured = []
        anchor = hp.head_pulses(-0.3, 0.2)
        try:
            servos._manual_override.clear()
            servos._speech_active.set()
            servos._speech_emotion_frame = {"speech_motion": {}}
            servos._speech_baseline = {0: 5472, 1: 3600, **anchor}
            seen = []
            for t in (0.0, 0.6, 1.2, 1.8):
                servos._last_speech_move_at = 0.0
                with (
                    mock.patch.object(servos, "SERVOS_ENABLED", True),
                    mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
                    mock.patch.object(servos.time, "monotonic", return_value=100.0 + t),
                    mock.patch.object(servos, "set_servos", side_effect=lambda x: captured.append(dict(x))),
                ):
                    servos.speech_reactive_move(0.8)
                pair = {ch: captured[-1][ch] for ch in hp.HEAD_PAIR}
                hp.validate_head_pair(pair)
                seen.append(hp.head_pose(pair))
            pitches = [p for p, _ in seen]
            tilts = [t for _, t in seen]
            self.assertLess(max(pitches) - min(pitches), 0.2)       # a small sway...
            self.assertGreater(max(tilts) - min(tilts), 0.02)        # ...with a slight cock each way
            for pitch, tilt in seen:
                self.assertAlmostEqual(pitch, -0.3, delta=0.12)
                self.assertAlmostEqual(tilt, 0.2, delta=0.15)
            # Visor stays inside the speaking band (inverted servo).
            low, high = config.VISOR_SPEECH_RANGE_QUS
            for frame in captured:
                self.assertTrue(low <= frame[3] <= high, frame[3])
        finally:
            servos._speech_active.clear()
            servos._speech_emotion_frame = {}
            servos._speech_baseline = {}

    def test_listening_nod_pitches_the_pair_down_and_keeps_the_cock(self):
        baseline = {0: 5472, 1: 3600, **hp.head_pulses(0.0, 0.3)}
        with mock.patch.object(servos, "_face_tracking_baseline", dict(baseline)):
            servos._listening_session.clear()
            servos._begin_listening_session(300.0)
            servos._listening_session["nod_started_at"] = 300.0
            mid = 300.0 + float(config.SERVO_LISTENING_NOD_SECS) / 2.0
            t = servos._listening_targets(mid)
        servos._listening_session.clear()
        pitch, tilt = hp.head_pose({ch: t[ch] for ch in hp.HEAD_PAIR})
        self.assertLess(pitch, -0.03)
        self.assertAlmostEqual(tilt, 0.3, delta=0.05)
        self.assertTrue(config.VISOR_SPEECH_RANGE_QUS[0] <= t[3] <= config.VISOR_SPEECH_RANGE_QUS[1])


if __name__ == "__main__":
    unittest.main()
