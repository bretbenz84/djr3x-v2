"""Unit tests for the Mac-side motion stack against a fake ESP32 serial port.

Covers the hardware.motion transport (handshake, seq/ack, telemetry snapshot),
the motion_controller policy (clamping, autonomous gate, voice verbs), and the
deterministic action_router.classify_explicit_motion classifier. No hardware
needed — a FakeESP32Serial stands in for the firmware.
"""

import json
import threading
import time
import unittest
from unittest import mock

import config
from hardware import motion
from intelligence import motion_controller as mc
from intelligence import action_router as ar


class FakeESP32Serial:
    """Minimal stand-in for the firmware: replies to hello, acks commands, and
    streams telemetry reflecting a simple moving/idle + owner state."""

    def __init__(
        self, *args, reply_hello=True, owner="auto", charging=False,
        batt_mv=12000, batt_ma=1000, batt_soc=50, **kwargs
    ):
        self.is_open = True
        self._out = bytearray()
        self._lock = threading.Lock()
        self.received: list[dict] = []     # non-ping commands the host sent
        self.reply_hello = reply_hello
        self.owner = owner
        self.charging = charging
        self.batt_mv = batt_mv
        self.batt_ma = batt_ma
        self.batt_soc = batt_soc
        self._state = "idle"

    def _emit(self, obj):
        with self._lock:
            self._out += (json.dumps(obj) + "\n").encode()

    def write(self, data):
        for raw in bytes(data).split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw.decode())
            except Exception:
                continue
            cmd = msg.get("cmd")
            seq = msg.get("seq")
            if cmd == "hello":
                if self.reply_hello:
                    self._emit({"v": 1, "type": "hello", "proto": 1, "fw": "fake",
                                "caps": ["drive", "turn", "move", "come", "stop"], "boot_id": 1234})
                continue
            if cmd == "ping":
                continue
            if cmd:
                with self._lock:
                    self.received.append(msg)
                self._emit({"v": 1, "type": "ack", "seq": seq, "accepted": True, "reason": None})
                if cmd in ("drive", "turn", "move", "come"):
                    self._state = "moving"
                elif cmd in ("stop", "estop"):
                    self._state = "idle"
        return len(data)

    def _telemetry(self):
        return {"v": 1, "type": "telemetry", "t": 1, "state": self._state, "owner": self.owner,
                "gamepad": "none", "fault": None, "zone": "clear", "blocked_dir": "none",
                "cmd_seq": 0, "odom": {"x": 0, "y": 0, "theta": 0, "lin": 0, "ang": 0},
                "tof_mm": {"fl": 4000, "fr": 4000, "rl": 4000, "rr": 4000,
                           "lf": 1500, "lb": 1500, "rf": 1500, "rb": 1500},
                "batt_mv": self.batt_mv, "batt_ma": self.batt_ma,
                "batt_soc": self.batt_soc, "charging": self.charging, "errs": 0}

    def read(self, n=1):
        with self._lock:
            if not self._out:
                self._out += (json.dumps(self._telemetry()) + "\n").encode()
            data = bytes(self._out[:n])
            del self._out[:n]
        time.sleep(0.003)
        return data

    def close(self):
        self.is_open = False


class _MotionTestBase(unittest.TestCase):
    def setUp(self):
        self._orig_serial = motion.serial.Serial
        self._orig_paused = getattr(config, "INTERACTION_PAUSED", False)
        self._orig_enabled = config.MOTION_ENABLED
        self._orig_timeout = config.MOTION_HANDSHAKE_TIMEOUT_MS
        config.INTERACTION_PAUSED = False
        config.MOTION_ENABLED = True
        config.MOTION_HANDSHAKE_TIMEOUT_MS = 400
        self.fake = None

    def tearDown(self):
        try:
            mc.disconnect()
        except Exception:
            pass
        motion.serial.Serial = self._orig_serial
        config.INTERACTION_PAUSED = self._orig_paused
        config.MOTION_ENABLED = self._orig_enabled
        config.MOTION_HANDSHAKE_TIMEOUT_MS = self._orig_timeout

    def _install(self, **kw):
        self.fake = FakeESP32Serial(**kw)
        motion.serial.Serial = lambda *a, **k: self.fake
        return self.fake

    def _connect(self, **kw):
        self._install(**kw)
        ok = mc.connect(port="FAKE")
        # let a telemetry frame land
        time.sleep(0.1)
        return ok

    def _last(self, cmd):
        for m in reversed(self.fake.received):
            if m.get("cmd") == cmd:
                return m
        return None


class TransportTest(_MotionTestBase):
    def test_handshake_ok(self):
        self.assertTrue(self._connect())
        self.assertTrue(motion.connected())
        self.assertIn("drive", motion.caps())
        self.assertEqual(motion.boot_id(), 1234)

    def test_handshake_timeout_disables(self):
        self.assertFalse(self._connect(reply_hello=False))
        self.assertFalse(motion.connected())

    def test_send_seq_and_ack(self):
        self._connect()
        seq = motion.send({"cmd": "turn", "deg": 90})
        self.assertIsInstance(seq, int)
        ack = motion.wait_ack(seq, 1.0)
        self.assertIsNotNone(ack)
        self.assertTrue(ack["accepted"])

    def test_telemetry_snapshot(self):
        self._connect()
        tel = motion.telemetry()
        self.assertIsNotNone(tel)
        self.assertEqual(tel["owner"], "auto")
        self.assertIn("odom", tel)

    def test_reconnect_after_drop(self):
        # Fresh fake on each open so a reconnect gets a working port.
        fakes = []

        def factory(*a, **k):
            f = FakeESP32Serial()
            fakes.append(f)
            return f

        motion.serial.Serial = factory
        self.assertTrue(motion.connect(port="FAKE"))   # no manager thread (direct)
        self.assertTrue(motion.connected())
        # Simulate an unplug detected by a failed write: close the link.
        with motion._write_lock:
            motion._close_serial_locked()
        self.assertFalse(motion.connected())
        # Auto-reconnect heals it (reopens + re-handshakes on a fresh port).
        self.assertTrue(motion.reconnect())
        self.assertTrue(motion.connected())
        self.assertGreaterEqual(len(fakes), 2)
        motion.disconnect()

    def test_reconnect_bogus_port_fails_fast(self):
        # No device at the remembered port -> reconnect returns False, stays down.
        def factory(*a, **k):
            raise __import__("serial").SerialException("no such device")

        motion.serial.Serial = factory
        motion._last_port = "/dev/cu.does-not-exist"
        self.assertFalse(motion.reconnect())
        self.assertFalse(motion.connected())


class ControllerTest(_MotionTestBase):
    def test_turn_left_right_defaults(self):
        self._connect()
        mc.turn_left()
        m = self._last("turn")
        self.assertEqual(m["deg"], config.MOTION_DEFAULT_TURN_DEG)
        self.assertEqual(m["rate"], config.MOTION_DEFAULT_TURN_RATE)
        mc.turn_right()
        self.assertEqual(self._last("turn")["deg"], -config.MOTION_DEFAULT_TURN_DEG)

    def test_move_clamps_speed(self):
        self._connect()
        mc.move(0.3, speed=5.0)   # speed way over cap
        m = self._last("move")
        self.assertAlmostEqual(m["dist"], 0.3)
        self.assertAlmostEqual(m["speed"], config.MOTION_MAX_LINEAR_MS)

    def test_move_back_sign(self):
        self._connect()
        mc.move_back(0.5)
        self.assertAlmostEqual(self._last("move")["dist"], -0.5)

    def test_stop_always_allowed_even_manual(self):
        self._connect(owner="manual")
        self.assertEqual(motion.owner(), "manual")
        self.assertIsNone(mc.turn_left())          # autonomous suppressed
        self.assertIsNone(self._last("turn"))
        self.assertIsNotNone(mc.stop())            # stop still goes through
        self.assertIsNotNone(self._last("stop"))

    def test_gate_interaction_paused(self):
        self._connect()
        config.INTERACTION_PAUSED = True
        self.assertIsNone(mc.move_forward())
        self.assertIsNone(self._last("move"))

    def test_sleep_blocks_every_autonomous_motion_command(self):
        from state import State
        self._connect()
        with mock.patch.object(mc.state_module, "get_state", return_value=State.SLEEP):
            self.assertIsNone(mc.move_forward())
            self.assertIsNone(mc.turn_left())
            self.assertIsNone(mc.come_here())
            self.assertIsNone(mc.arc_move())
        self.assertIsNone(self._last("move"))
        self.assertIsNone(self._last("turn"))
        self.assertIsNone(self._last("come"))
        self.assertIsNone(self._last("drive"))
        self.assertIsNotNone(mc.stop())  # safety stop always remains available

    def test_charging_blocks_autonomous_and_manual_drive(self):
        self._connect(charging=True)
        self.assertIsNone(mc.move_forward())
        self.assertIsNone(mc.turn_left())
        self.assertIsNone(mc.drive_manual(0.1, 0.0))
        self.assertIsNone(self._last("move"))
        self.assertIsNone(self._last("turn"))
        self.assertIsNone(self._last("drive"))
        self.assertIsNotNone(mc.stop())

    def test_charger_voltage_blocks_even_when_old_firmware_flag_is_false(self):
        self._connect(charging=False, batt_mv=14200, batt_ma=0, batt_soc=100)
        self.assertTrue(mc.charging())
        self.assertIsNone(mc.turn_left())
        self.assertIsNone(self._last("turn"))

    def test_full_unplugged_voltage_does_not_block(self):
        self._connect(charging=False, batt_mv=13400, batt_ma=1200, batt_soc=100)
        self.assertFalse(mc.charging())
        self.assertIsNotNone(mc.turn_left())

    def test_charging_release_is_sticky_across_a_flap(self):
        # A servo sag briefly flaps charging False; within the grace, charging() stays
        # True (wheels stay locked) so a flinch can't wake the base mid-charge.
        with mock.patch.object(config, "MOTION_CHARGING_RELEASE_GRACE_SECS", 20.0, create=True):
            self._connect(charging=True, batt_mv=14200)
            self.assertTrue(mc.charging())                 # arms the sticky latch
            self.fake.charging = False                     # firmware flag flaps off...
            self.fake.batt_mv = 13750                       # ...and voltage sags under load
            time.sleep(0.15)                               # let a fresh telemetry frame land
            self.assertTrue(mc.charging())                 # still locked (within grace)
            self.assertIsNone(mc.turn_left())              # drive stays blocked

    def test_charging_releases_after_grace_on_real_unplug(self):
        with mock.patch.object(config, "MOTION_CHARGING_RELEASE_GRACE_SECS", 0.0, create=True):
            self._connect(charging=True, batt_mv=14200)
            self.assertTrue(mc.charging())
            self.fake.charging = False
            self.fake.batt_mv = 13400                       # settled unplugged rest
            time.sleep(0.15)
            self.assertFalse(mc.charging())                # grace 0 -> releases immediately
            self.assertIsNotNone(mc.turn_left())

    def test_arc_sets_state_then_cancels(self):
        self._connect()
        self.assertIsNotNone(mc.arc_move(forward=True, left=False, small=True))
        self.assertTrue(mc._arc_active)        # heartbeat will refresh the curve...
        mc.stop()                              # ...until a command/stop supersedes it
        self.assertFalse(mc._arc_active)

    def test_arc_suppressed_when_manual(self):
        self._connect(owner="manual")
        self.assertIsNone(mc.arc_move())       # gamepad owns the base -> no autonomous arc
        self.assertFalse(mc._arc_active)

    def test_disabled_is_noop(self):
        config.MOTION_ENABLED = False
        self.assertFalse(mc.connect(port="FAKE"))
        self.assertFalse(mc.available())
        self.assertIsNone(mc.turn_left())


class CompassTurnVerificationTest(_MotionTestBase):
    """Absolute-heading correction is calibrated-only and cannot supersede a
    newer command."""

    def setUp(self):
        super().setUp()
        self._orig_verify = config.MOTION_COMPASS_TURN_VERIFY_ENABLED
        config.MOTION_COMPASS_TURN_VERIFY_ENABLED = True

    def tearDown(self):
        config.MOTION_COMPASS_TURN_VERIFY_ENABLED = self._orig_verify
        super().tearDown()

    def _record(self, *, desired=90.0, start=10.0, attempt=0):
        epoch = mc._invalidate_turn_verification()
        return {
            "desired_deg": desired,
            "rate": 40.0,
            "start_yaw": start,
            "epoch": epoch,
            "attempt": attempt,
        }

    def test_completed_turn_is_corrected_once_when_compass_disagrees(self):
        self._connect()
        record = self._record()
        with mock.patch.object(mc._stop, "wait", return_value=False), \
             mock.patch.object(mc, "_calibrated_compass_yaw", return_value=90.0), \
             mock.patch.object(mc, "turn") as correction:
            mc._verify_completed_turn(record)
        correction.assert_called_once_with(10.0, rate=25.0, _verify_attempt=1)

    def test_turn_within_compass_tolerance_needs_no_correction(self):
        self._connect()
        record = self._record()
        with mock.patch.object(mc._stop, "wait", return_value=False), \
             mock.patch.object(mc, "_calibrated_compass_yaw", return_value=99.0), \
             mock.patch.object(mc, "turn") as correction:
            mc._verify_completed_turn(record)
        correction.assert_not_called()

    def test_newer_command_invalidates_delayed_compass_correction(self):
        self._connect()
        record = self._record()
        mc._invalidate_turn_verification()
        with mock.patch.object(mc._stop, "wait", return_value=False), \
             mock.patch.object(mc, "_calibrated_compass_yaw", return_value=80.0), \
             mock.patch.object(mc, "turn") as correction:
            mc._verify_completed_turn(record)
        correction.assert_not_called()

    def test_uncalibrated_compass_does_not_arm_verification(self):
        mc._invalidate_turn_verification()
        with mc._turn_verify_lock:
            epoch = mc._turn_verify_epoch
        mc._remember_turn_verification(
            17, desired_deg=90.0, rate=40.0, start_yaw=None, epoch=epoch, attempt=0,
        )
        with mc._turn_verify_lock:
            self.assertNotIn(17, mc._pending_turn_verify)


class ConfigPushTest(_MotionTestBase):
    """_push_config sends caps/zones/ramps on connect; optional bench-tuning keys
    remain opt-in so a connect never clobbers their firmware values."""

    def setUp(self):
        super().setUp()
        self._orig_tuning = {k: getattr(config, k, None) for k in (
            "MOTION_WHEEL_KP", "MOTION_WHEEL_KI", "MOTION_WHEEL_KD",
            "MOTION_COUNTS_PER_METER", "MOTION_TRACK_WIDTH_M")}

    def tearDown(self):
        for k, v in self._orig_tuning.items():
            setattr(config, k, v)
        super().tearDown()

    def test_push_includes_caps(self):
        self._connect()
        cfg = self._last("config")
        self.assertIsNotNone(cfg)
        self.assertAlmostEqual(cfg["max_lin"], config.MOTION_MAX_LINEAR_MS)
        self.assertIn("stop_zone_m", cfg)
        self.assertAlmostEqual(cfg["accel_lin"], config.MOTION_ACCEL_LINEAR_MS2)
        self.assertAlmostEqual(cfg["accel_ang"], config.MOTION_ACCEL_ANGULAR_RAD_S2)

    def test_tuning_keys_omitted_when_unset(self):
        config.MOTION_WHEEL_KP = None
        config.MOTION_COUNTS_PER_METER = None
        self._connect()
        cfg = self._last("config")
        self.assertNotIn("kp", cfg)
        self.assertNotIn("counts_per_meter", cfg)

    def test_tuning_keys_pushed_when_set(self):
        config.MOTION_WHEEL_KP = 2200.0
        config.MOTION_COUNTS_PER_METER = 31000.0
        self._connect()
        cfg = self._last("config")
        self.assertAlmostEqual(cfg["kp"], 2200.0)
        self.assertAlmostEqual(cfg["counts_per_meter"], 31000.0)
        self.assertNotIn("ki", cfg)   # still opt-in per key


class ClassifierTest(unittest.TestCase):
    def _act(self, text):
        d = ar.classify_explicit_motion(text)
        return None if d is None else d.action

    def test_directions(self):
        self.assertEqual(self._act("turn left"), "motion.turn")
        self.assertEqual(self._act("spin around"), "motion.turn")
        self.assertEqual(self._act("move forward"), "motion.move")
        self.assertEqual(self._act("back up"), "motion.move")
        self.assertEqual(self._act("come here"), "motion.come")
        self.assertEqual(self._act("halt"), "motion.stop")

    def test_requested_come_phrases(self):
        for text in ("come here", "come over here", "come to me"):
            self.assertEqual(self._act(text), "motion.come")

    def test_contextual_motion_continuations(self):
        left = ar.classify_explicit_motion("turn left")
        forward = ar.classify_explicit_motion("move forward")
        back = ar.classify_explicit_motion("back up 30 cm")

        d = ar.classify_motion_continuation("more", left)
        self.assertEqual((d.action, d.args["direction"]), ("motion.turn", "left"))
        d = ar.classify_motion_continuation("a little more", left)
        self.assertEqual(d.args["deg"], 15.0)
        d = ar.classify_motion_continuation("keep turning", left)
        self.assertEqual(d.action, "motion.turn")

        d = ar.classify_motion_continuation("keep moving", forward)
        self.assertEqual((d.action, d.args["direction"]), ("motion.move", "forward"))
        self.assertEqual(ar.classify_motion_continuation("keep going", forward).action,
                         "motion.move")
        self.assertEqual(ar.classify_motion_continuation("keep going", left).action,
                         "motion.turn")
        d = ar.classify_motion_continuation("a bit more", back)
        self.assertEqual((d.args["direction"], d.args["dist_m"]), ("back", 0.15))

        self.assertIsNone(ar.classify_motion_continuation("keep moving", left))
        self.assertIsNone(ar.classify_motion_continuation("keep turning", forward))
        self.assertIsNone(ar.classify_motion_continuation("more", None))
        self.assertIsNone(ar.classify_motion_continuation("more details", forward))

    def test_ordered_motion_sequences(self):
        seq = ar.classify_explicit_motion_sequence(
            "Turn to your left (90 degrees) then move forward 5 feet."
        )
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(seq[0].args, {"direction": "left", "deg": 90.0})
        self.assertAlmostEqual(seq[1].args["dist_m"], 1.524, places=3)

        seq = ar.classify_explicit_motion_sequence("Turn around and move back 4 feet")
        self.assertEqual(seq[0].args, {"direction": "around", "deg": 180.0})
        self.assertEqual(seq[1].args["direction"], "back")
        self.assertAlmostEqual(seq[1].args["dist_m"], 1.2192, places=4)

        seq = ar.classify_explicit_motion_sequence(
            "go forward 2 meters, turn right 45 degrees, then move forward 10 feet"
        )
        self.assertEqual(len(seq), 3)
        self.assertAlmostEqual(seq[0].args["dist_m"], 2.0)
        self.assertEqual(seq[1].args, {"direction": "right", "deg": 45.0})
        self.assertAlmostEqual(seq[2].args["dist_m"], 3.048, places=3)

    def test_sequence_parser_preserves_arc_and_rejects_partial_execution(self):
        self.assertEqual(
            ar.classify_explicit_motion_sequence("move forward and to your right"), []
        )
        self.assertIsNone(ar.classify_explicit_motion_sequence("turn left then sing a song"))
        self.assertIsNone(ar.classify_explicit_motion_sequence("turn left then stop"))

    def test_come_forward_splits_as_its_own_clause(self):
        # Field 2026-07-24: "come" was missing from the `and`-split verb lookahead,
        # so "turn around and come forward 5 feet" never split — it fell through to
        # the single-command path and only the FORWARD half ran; the turn-around was
        # silently dropped.
        seq = ar.classify_explicit_motion_sequence("turn around and come forward 5 feet")
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(seq[0].args, {"direction": "around", "deg": 180.0})
        self.assertAlmostEqual(seq[1].args["dist_m"], 1.524, places=3)

    def test_trailing_vocative_does_not_kill_the_route(self):
        # Field 2026-07-24: Whisper heard "...5 feet" as "..., Ozzie"; the junk
        # trailing clause tripped the no-partial-execution refusal and NOTHING ran.
        # Plain politeness had the same shape — addressing Rex by name at the end
        # refused the whole command.
        for text in (
            "Turn around and come forward, Ozzie",
            "turn left then move forward five feet, Rex",
            "turn left then move forward, buddy",
        ):
            seq = ar.classify_explicit_motion_sequence(text)
            self.assertIsNotNone(seq, text)
            self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"], text)

    def test_comma_before_a_magnitude_is_not_a_route_boundary(self):
        # Field 2026-07-24: "Turn left, 15 degrees." drew "I couldn't safely parse
        # that whole route" and Rex never moved — the comma split it into
        # ["Turn left", "15 degrees"], the second clause classified as nothing, and
        # the mixed motion/non-motion guard refused the WHOLE utterance.
        for text, deg in (("Turn left, 15 degrees.", 15.0),
                          ("turn right, 30 degrees", 30.0),
                          ("turn left, about 45 degrees", 45.0)):
            self.assertEqual(ar.classify_explicit_motion_sequence(text), [], text)
            single = ar.classify_explicit_motion(text)
            self.assertIsNotNone(single, text)
            self.assertEqual(single.action, "motion.turn", text)
            self.assertEqual(single.args.get("deg"), deg, text)

    def test_comma_before_a_distance_is_not_a_route_boundary(self):
        for text, metres in (("move forward, 3 feet", 0.9144),
                             ("move forward, two feet", 0.6096),
                             ("move forward, 2 meters", 2.0)):
            self.assertEqual(ar.classify_explicit_motion_sequence(text), [], text)
            single = ar.classify_explicit_motion(text)
            self.assertIsNotNone(single, text)
            self.assertAlmostEqual(single.args["dist_m"], metres, places=3, msg=text)

    def test_magnitude_survives_inside_a_real_route(self):
        # The rejoin must not eat a genuine following clause, and the magnitude must
        # still land on the RIGHT step.
        seq = ar.classify_explicit_motion_sequence(
            "turn left, 15 degrees, then move forward 3 feet")
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(seq[0].args["deg"], 15.0)
        self.assertAlmostEqual(seq[1].args["dist_m"], 0.9144, places=4)

        seq = ar.classify_explicit_motion_sequence(
            "turn right, 30 degrees and move forward 2 meters")
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(seq[0].args["deg"], 30.0)
        self.assertAlmostEqual(seq[1].args["dist_m"], 2.0, places=4)

    def test_trailing_real_action_still_refuses(self):
        # The vocative strip must NOT weaken the partial-execution guard: a genuine
        # non-motion action after a comma is still a refusal.
        self.assertIsNone(ar.classify_explicit_motion_sequence("turn left, sing"))
        self.assertIsNone(ar.classify_explicit_motion_sequence("turn left then move forward, dance"))
        # A terse but REAL motion clause after a comma is kept, not stripped —
        # the strip only ever removes fragments the sequence parser would itself
        # have rejected, converting "refuse everything" into "do the understood part".
        seq = ar.classify_explicit_motion_sequence("turn left, go forward")
        self.assertIsNotNone(seq)
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])

    def test_leading_connective_single_clause_is_not_a_sequence(self):
        # Field 2026-07-21: "and move backwards" was split into ["", "move backwards"],
        # rejected as an invalid sequence, and nothing moved. A lone clause behind a
        # connective must fall through ([]) to the plain single-command path — which
        # must then classify it.
        self.assertEqual(ar.classify_explicit_motion_sequence("and move backwards"), [])
        self.assertEqual(ar.classify_explicit_motion_sequence("then turn right"), [])
        # Pure connectives / disfluencies with ZERO clauses are conversation, not a
        # rejected route (field 2026-07-23: "and then," drew "I couldn't safely parse
        # that whole route" — the 0-clause case fell through to None).
        for filler in ("and then,", "and then", "then,", "then"):
            self.assertEqual(ar.classify_explicit_motion_sequence(filler), [], filler)
        # Chatty comma with ZERO motion clauses is conversation, not a rejected route
        # (pre-existing: this drew "I couldn't safely parse that whole route").
        self.assertEqual(
            ar.classify_explicit_motion_sequence("yeah that sounds great, thanks"), []
        )
        d = ar.classify_explicit_motion("and move backwards")
        self.assertEqual(d.action, "motion.move")
        self.assertEqual(d.args["direction"], "back")

    def test_spoken_word_distances_parse(self):
        # Whisper writes small counts as words; digits-only parsing silently dropped
        # every spoken distance (field 2026-07-21: "go backwards four feet" moved the
        # default nudge instead).
        d = ar.classify_explicit_motion("go backwards four feet")
        self.assertEqual(d.args["direction"], "back")
        self.assertAlmostEqual(d.args["dist_m"], 4 * 0.3048, places=4)
        d = ar.classify_explicit_motion("move forward two meters")
        self.assertAlmostEqual(d.args["dist_m"], 2.0)
        d = ar.classify_explicit_motion("go forward half a meter")
        self.assertAlmostEqual(d.args["dist_m"], 0.5)
        d = ar.classify_explicit_motion("move back a foot")
        self.assertAlmostEqual(d.args["dist_m"], 0.3048, places=4)
        # And inside a sequence:
        seq = ar.classify_explicit_motion_sequence(
            "go backwards four feet, turn right, then go forward ten feet"
        )
        self.assertEqual(len(seq), 3)
        self.assertAlmostEqual(seq[0].args["dist_m"], 4 * 0.3048, places=4)
        self.assertAlmostEqual(seq[2].args["dist_m"], 10 * 0.3048, places=4)

    def test_turn_args(self):
        d = ar.classify_explicit_motion("turn right 45 degrees")
        self.assertEqual(d.args, {"direction": "right", "deg": 45.0})
        d = ar.classify_explicit_motion("spin around")
        self.assertEqual(d.args.get("deg"), 180.0)
        d = ar.classify_explicit_motion("turn around")
        self.assertEqual(d.args, {"direction": "around", "deg": 180.0})
        d = ar.classify_explicit_motion("turn 180")
        self.assertEqual(d.args, {"direction": "around", "deg": 180.0})

    def test_direct_small_turn_is_45_degrees(self):
        for text, direction in (
            ("turn right a little", "right"),
            ("turn left a bit", "left"),
            ("turn slightly right", "right"),
            ("rotate a little to your left", "left"),
        ):
            d = ar.classify_explicit_motion(text)
            self.assertIsNotNone(d, text)
            self.assertEqual(d.action, "motion.turn", text)
            self.assertEqual(d.args, {"direction": direction, "deg": 45.0}, text)

    def test_distance_parse(self):
        d = ar.classify_explicit_motion("move forward 2 feet")
        self.assertAlmostEqual(d.args["dist_m"], 0.6096, places=4)
        d = ar.classify_explicit_motion("back up 30 cm")
        self.assertAlmostEqual(d.args["dist_m"], 0.30, places=4)

    def test_turn_to_your_side(self):
        self.assertEqual(ar.classify_explicit_motion("turn to your left").args["direction"], "left")
        self.assertEqual(ar.classify_explicit_motion("turn to your right").args["direction"], "right")

    def test_small_amount_move(self):
        # "a little / a bit" between the verb and direction classifies, as a SMALL move.
        for t in ("move a little forward", "ease slightly forward"):
            d = ar.classify_explicit_motion(t)
            self.assertEqual((d.action, d.args["direction"]), ("motion.move", "forward"))
            self.assertAlmostEqual(d.args["dist_m"], 0.15)
        for t in ("move a little back", "move a little backwards"):
            d = ar.classify_explicit_motion(t)
            self.assertEqual(d.args["direction"], "back")
            self.assertAlmostEqual(d.args["dist_m"], 0.15)

    def test_arc_compound(self):
        # A forward/back + left/right joined by "and" in ONE utterance -> a curve (arc).
        d = ar.classify_explicit_motion("move a little forward and to your right")
        self.assertEqual(d.action, "motion.arc")
        self.assertEqual((d.args["lin_dir"], d.args["ang_dir"], d.args["small"]),
                         ("forward", "right", True))
        d = ar.classify_explicit_motion("back up and to the left")
        self.assertEqual((d.action, d.args["lin_dir"], d.args["ang_dir"]),
                         ("motion.arc", "back", "left"))
        # Single utterances stay single finite commands (NOT arcs) — sequential when said
        # one after another.
        self.assertEqual(self._act("move forward"), "motion.move")
        self.assertEqual(self._act("turn left"), "motion.turn")

    def test_lateral_move_is_an_arc(self):
        # Field-logged 2026-07-11: "Move to your left" fell through to conversation
        # and got a quip. A lateral request (move verb + side, no forward/back word)
        # executes as a small forward arc toward that side — the base can't strafe.
        d = ar.classify_explicit_motion("Move to your left")
        self.assertEqual(d.action, "motion.arc")
        self.assertEqual((d.args["lin_dir"], d.args["ang_dir"], d.args["small"]),
                         ("forward", "left", True))
        for text, side in (
            ("move to the right", "right"),
            ("go left", "left"),
            ("scoot over to the left", "left"),
            ("slide to your right", "right"),
            ("move a little to the left", "left"),
            ("shift right", "right"),
        ):
            d = ar.classify_explicit_motion(text)
            self.assertEqual(
                (d.action, d.args["ang_dir"]), ("motion.arc", side), text)
        # Turn verbs stay pure turns — never hijacked by the lateral family.
        self.assertEqual(self._act("turn left"), "motion.turn")
        self.assertEqual(self._act("spin right"), "motion.turn")

    def test_no_false_positives(self):
        for t in ["stop", "play some music", "turn it up", "turn off the lights",
                  "how do I get back to the menu", "let's move on",
                  "How come you didn't move forward?",
                  "So how come he didn't move forward?",
                  "Why did you turn left?",
                  "Don't move forward",
                  "Never back up into that trash can",
                  # figurative "move forward" / "go ahead" must NOT drive the base
                  "let's move forward with the plan", "I want to move forward in life",
                  "go ahead and tell me", "move the box forward",
                  "move forward towards the goal", "I went left and right all day"]:
            self.assertIsNone(self._act(t), f"{t!r} should not classify as motion")


class RampTowardTest(unittest.TestCase):
    """The GUI joystick's asymmetric slew (gentle accel, faster non-abrupt decel)."""

    def test_accel_step_speeding_up_from_zero(self):
        self.assertAlmostEqual(mc.ramp_toward(0.0, 1.0, 0.1, 0.3), 0.1)

    def test_decel_step_slowing_toward_zero(self):
        # Releasing the stick (target 0) uses the faster decel step, not accel.
        self.assertAlmostEqual(mc.ramp_toward(1.0, 0.0, 0.1, 0.3), 0.7)

    def test_no_overshoot_and_equal(self):
        self.assertEqual(mc.ramp_toward(0.95, 1.0, 0.1, 0.3), 1.0)   # accel within a step
        self.assertEqual(mc.ramp_toward(0.2, 0.0, 0.1, 0.3), 0.0)    # decel within a step
        self.assertEqual(mc.ramp_toward(0.5, 0.5, 0.1, 0.3), 0.5)    # already there

    def test_decel_reaches_zero_in_fewer_ticks_than_accel(self):
        def ticks(cur, tgt):
            n = 0
            while abs(cur - tgt) > 1e-9 and n < 1000:
                cur = mc.ramp_toward(cur, tgt, 0.1, 0.25)
                n += 1
            return n
        self.assertLess(ticks(1.0, 0.0), ticks(0.0, 1.0))           # down faster than up

    def test_reversal_decelerates_through_zero_first(self):
        # +0.2 heading to -1.0: shrink the current magnitude (decel) before reversing.
        self.assertAlmostEqual(mc.ramp_toward(0.2, -1.0, 0.1, 0.3), -0.1)

    def test_nonpositive_step_jumps_to_target(self):
        self.assertEqual(mc.ramp_toward(0.0, 1.0, 0.0, 0.0), 1.0)


class CommandedMotionFxTest(unittest.TestCase):
    """A voice-COMMANDED move must still make its motor sound. Its spoken
    confirmation ("Spinning around.") reaches the speaker ~3 ms after queueing, so
    the gated clip lost the race and was silently dropped on nearly every command —
    while autonomous moves, which say nothing, kept theirs (field 2026-07-24)."""

    def setUp(self):
        mc._user_commanded_motion_at = 0.0
        self.addCleanup(lambda: setattr(mc, "_user_commanded_motion_at", 0.0))

    def test_autonomous_motion_uses_the_gated_path(self):
        with mock.patch("audio.sound_effects.play") as play:
            mc._fx("motion_turn")
        play.assert_called_once()
        self.assertFalse(play.call_args.kwargs.get("overlay"))

    def test_commanded_motion_overlays(self):
        mc.note_user_commanded_motion()
        with mock.patch("audio.sound_effects.play") as play:
            mc._fx("motion_turn")
        play.assert_called_once()
        self.assertTrue(play.call_args.kwargs.get("overlay"))

    def test_overlay_marking_expires(self):
        # A later AUTONOMOUS move must not inherit overlay mode from an old command.
        mc.note_user_commanded_motion()
        mc._user_commanded_motion_at = time.monotonic() - (
            float(config.MOTION_COMMANDED_FX_WINDOW_SECS) + 1.0
        )
        with mock.patch("audio.sound_effects.play") as play:
            mc._fx("motion_turn")
        self.assertFalse(play.call_args.kwargs.get("overlay"))

    def test_fx_never_raises_when_the_effects_layer_fails(self):
        mc.note_user_commanded_motion()
        with mock.patch("audio.sound_effects.play", side_effect=RuntimeError("no audio")):
            mc._fx("motion_turn")   # must not propagate


class BlockedAnnounceTest(unittest.TestCase):
    """A VOICE-commanded move the firmware cuts on an obstacle must SAY so —
    silence read as 'he ignores my commands' (field 2026-07-23, 'move forward 5
    feet' stopped at ~2 ft with no acknowledgement). Autonomous legs stay quiet."""

    def setUp(self):
        with mc._announce_blocked_lock:
            mc._announce_blocked_seqs.clear()
        mc._announce_blocked_last_spoken = 0.0

    def _done(self, seq, result="blocked"):
        mc._on_motion_done({"seq": seq, "result": result})

    def test_registered_blocked_move_speaks(self):
        from unittest import mock
        mc.announce_if_blocked(41)
        with mock.patch("audio.speech_queue.enqueue") as enq, \
                mock.patch.object(mc, "_fx"):
            self._done(41)
        enq.assert_called_once()
        self.assertIn("way", enq.call_args[0][0].lower())

    def test_unregistered_blocked_move_is_silent(self):
        from unittest import mock
        with mock.patch("audio.speech_queue.enqueue") as enq, \
                mock.patch.object(mc, "_fx"):
            self._done(99)                      # autonomous/exploration leg
        enq.assert_not_called()

    def test_completed_registered_move_is_silent(self):
        from unittest import mock
        mc.announce_if_blocked(42)
        with mock.patch("audio.speech_queue.enqueue") as enq, \
                mock.patch.object(mc, "_fx"):
            self._done(42, result="completed")
        enq.assert_not_called()

    def test_cooldown_suppresses_back_to_back_announcements(self):
        from unittest import mock
        mc.announce_if_blocked(43)
        mc.announce_if_blocked(44)
        with mock.patch("audio.speech_queue.enqueue") as enq, \
                mock.patch.object(mc, "_fx"):
            self._done(43)
            self._done(44)                      # within the 10 s cooldown
        enq.assert_called_once()


class MotionTakeoverTest(_MotionTestBase):
    """interaction._explicit_motion_takeover runs BEFORE the dialogue-act gate, so an
    explicit command isn't swallowed as an answer_to_rex reply when Rex has just spoken
    (the live 2026-06-23 bug: "move forward." / "Move backwards" -> conversation.reply).
    It executes motion regardless of dialogue state, and is a clean no-op otherwise."""

    def test_no_base_drive_command_is_verbally_denied(self):
        from unittest import mock
        from intelligence import interaction as I
        # No base connected: an explicit DRIVE command is refused OUT LOUD (in character)
        # instead of silently falling through to conversation.
        with mock.patch.object(I, "_speak_blocking", return_value=True) as spoke:
            resp = I._explicit_motion_takeover("turn left")
        self.assertIn(resp, config.MOTION_NO_BASE_DENIAL_LINES)
        spoke.assert_called_once()
        self.assertEqual(spoke.call_args.args[0], resp)   # the returned line is the spoken line

    def test_no_base_bare_stop_is_noop(self):
        from unittest import mock
        from intelligence import interaction as I
        # "halt"/bare-stop and non-motion text must NOT be denied (no wheels to stop, and
        # "stop" must stay free for stop-music/game).
        with mock.patch.object(I, "_speak_blocking", return_value=True) as spoke:
            self.assertIsNone(I._explicit_motion_takeover("halt"))
            self.assertIsNone(I._explicit_motion_takeover("yeah that sounds great, thanks"))
        spoke.assert_not_called()

    def test_no_base_denial_can_be_disabled(self):
        from unittest import mock
        from intelligence import interaction as I
        orig = config.MOTION_NO_BASE_DENIAL_ENABLED
        config.MOTION_NO_BASE_DENIAL_ENABLED = False
        try:
            with mock.patch.object(I, "_speak_blocking", return_value=True) as spoke:
                self.assertIsNone(I._explicit_motion_takeover("turn left"))
            spoke.assert_not_called()
        finally:
            config.MOTION_NO_BASE_DENIAL_ENABLED = orig

    def test_explicit_motion_executes(self):
        from unittest import mock
        from intelligence import interaction as I
        self._connect()
        with mock.patch.object(I, "_speak_blocking", return_value=True) as spoke:
            self.assertEqual(I._explicit_motion_takeover("turn left"), "Turning left.")
            self.assertIsNotNone(self._last("turn"))
            self.assertEqual(I._explicit_motion_takeover("move forward"), "Rolling forward.")
            self.assertIsNotNone(self._last("move"))
        self.assertEqual(
            [call.args[0] for call in spoke.call_args_list],
            ["Turning left.", "Rolling forward."],
        )

    def test_more_repeats_last_successful_motion(self):
        from intelligence import interaction as I
        self._connect()
        I._clear_motion_continuation()
        self.assertEqual(I._explicit_motion_takeover("turn left"), "Turning left.")
        self.assertEqual(I._explicit_motion_takeover("more"), "Turning left.")
        turns = [m for m in self.fake.received if m.get("cmd") == "turn"]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[-1]["deg"], config.MOTION_DEFAULT_TURN_DEG)

    def test_little_more_uses_small_increment_and_stop_clears(self):
        from intelligence import interaction as I
        self._connect()
        I._clear_motion_continuation()
        self.assertEqual(I._explicit_motion_takeover("move forward"), "Rolling forward.")
        self.assertEqual(I._explicit_motion_takeover("a little more"), "Rolling forward.")
        self.assertAlmostEqual(self._last("move")["dist"],
                               config.MOTION_CONTINUATION_SMALL_MOVE_M)
        self.assertEqual(I._explicit_motion_takeover("halt"), "Stopping.")
        self.assertIsNone(I._explicit_motion_takeover("more"))

    def test_keep_phrase_must_match_previous_motion_kind(self):
        from intelligence import interaction as I
        self._connect()
        I._clear_motion_continuation()
        self.assertEqual(I._explicit_motion_takeover("turn right"), "Turning right.")
        self.assertIsNone(I._explicit_motion_takeover("keep moving"))
        # The mismatched intervening phrase also retires the continuation.
        self.assertIsNone(I._explicit_motion_takeover("keep turning"))

    def test_explicit_come_arms_person_search_instead_of_driving_blind(self):
        from intelligence import interaction as I
        from intelligence import motion_agency
        self._connect()
        motion_agency.cancel_requested_come("test reset")
        try:
            self.assertEqual(I._explicit_motion_takeover("come to me"), "On my way.")
            self.assertTrue(motion_agency.requested_come_active())
            self.assertIsNone(self._last("come"))
        finally:
            motion_agency.cancel_requested_come("test cleanup")

    def test_non_motion_is_noop(self):
        from intelligence import interaction as I
        self._connect()
        self.assertIsNone(I._explicit_motion_takeover("yeah that sounds great, thanks"))
        self.assertIsNone(self._last("turn"))
        self.assertIsNone(self._last("move"))


if __name__ == "__main__":
    unittest.main()
