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

import config
from hardware import motion
from intelligence import motion_controller as mc
from intelligence import action_router as ar


class FakeESP32Serial:
    """Minimal stand-in for the firmware: replies to hello, acks commands, and
    streams telemetry reflecting a simple moving/idle + owner state."""

    def __init__(self, *args, reply_hello=True, owner="auto", **kwargs):
        self.is_open = True
        self._out = bytearray()
        self._lock = threading.Lock()
        self.received: list[dict] = []     # non-ping commands the host sent
        self.reply_hello = reply_hello
        self.owner = owner
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
                "tof_mm": {"fl": 1500, "fc": 1500, "fr": 1500, "rear": 1500, "down": 60},
                "batt_mv": 12000, "errs": 0}

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

    def test_disabled_is_noop(self):
        config.MOTION_ENABLED = False
        self.assertFalse(mc.connect(port="FAKE"))
        self.assertFalse(mc.available())
        self.assertIsNone(mc.turn_left())


class ConfigPushTest(_MotionTestBase):
    """_push_config sends caps/zones on connect, and the drive-tuning keys only when
    the matching config.py value is set (opt-in, so a connect never clobbers a
    bench-tuned value with a placeholder)."""

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

    def test_turn_args(self):
        d = ar.classify_explicit_motion("turn right 45 degrees")
        self.assertEqual(d.args, {"direction": "right", "deg": 45.0})
        d = ar.classify_explicit_motion("spin around")
        self.assertEqual(d.args.get("deg"), 180.0)

    def test_distance_parse(self):
        d = ar.classify_explicit_motion("move forward 2 feet")
        self.assertAlmostEqual(d.args["dist_m"], 0.6096, places=4)
        d = ar.classify_explicit_motion("back up 30 cm")
        self.assertAlmostEqual(d.args["dist_m"], 0.30, places=4)

    def test_no_false_positives(self):
        for t in ["stop", "play some music", "turn it up", "turn off the lights",
                  "how do I get back to the menu", "let's move on"]:
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


if __name__ == "__main__":
    unittest.main()
