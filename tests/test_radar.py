"""Unit tests for the Mac-side radar-ring transport (hardware/radar.py) against
a fake ESP32-S3 serial port.

Covers the handshake (including the wrong-board refusal — the drive base also
answers hello), telemetry/target normalization, the dropout latch, and the
serial-number port resolution. No hardware needed — FakeRadarSerial stands in
for the firmware, mirroring tests/test_motion.py's FakeESP32Serial style.
"""

import json
import threading
import time
import unittest
from unittest import mock

import config
from hardware import radar


class FakeRadarSerial:
    """Minimal stand-in for the radar firmware: replies to hello and streams
    telemetry with a configurable fused target list. Targets use the wire
    schema ({"b","r","c","s","m"}); set .targets_wire live to change what the
    next frames carry."""

    def __init__(self, *args, reply_hello=True, caps=("radar",), proto=1,
                 targets=None, radar_ok=True, **kwargs):
        self.is_open = True
        self._out = bytearray()
        self._lock = threading.Lock()
        self.received: list[dict] = []
        self.reply_hello = reply_hello
        self.caps = list(caps)
        self.proto = proto
        self.targets_wire = list(targets or [])
        self.radar_ok = radar_ok

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
            if cmd:
                with self._lock:
                    self.received.append(msg)
            if cmd == "hello" and self.reply_hello:
                self._emit({"v": 1, "type": "hello", "proto": self.proto,
                            "fw": "fake-radar", "caps": self.caps, "boot_id": 777,
                            "sensors": [{"mount": 180.0, "cfg": True},   # pins.h ring:
                                        {"mount": -60.0, "cfg": True},   # S0 rear, S1/S2
                                        {"mount": 60.0, "cfg": True}]})   # forward pair
        return len(data)

    def _telemetry(self):
        return {"v": 1, "type": "telemetry", "t": 1,
                "radar": {"ok": self.radar_ok, "up": 3 if self.radar_ok else 0,
                          "targets": list(self.targets_wire)},
                "sens": [{"ok": self.radar_ok, "frames": 100, "bad": 0, "drop": 0}] * 3,
                "errs": 0}

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


_TARGET_WIRE = {"b": 137.2, "r": 4.10, "c": 0.82, "s": -0.30, "m": 6}


class _RadarTestBase(unittest.TestCase):
    def setUp(self):
        self._orig_serial = radar.serial.Serial
        self._orig_enabled = getattr(config, "RADAR_ENABLED", True)
        self._orig_timeout = getattr(config, "RADAR_HANDSHAKE_TIMEOUT_MS", 1500)
        config.RADAR_ENABLED = True
        config.RADAR_HANDSHAKE_TIMEOUT_MS = 400
        # .env on a dev Mac has no radar keys; give the module an identity so
        # _enabled() passes, and park the self-heal monitor so it can't race a
        # test's fake (its loop exits while _shutting_down is set).
        self._patches = [
            mock.patch.object(radar, "RADAR_ESP32_PORT", "FAKE"),
            mock.patch.object(radar, "RADAR_ESP32_SERIAL", None),
        ]
        for p in self._patches:
            p.start()
        radar._shutting_down = True
        # Latch state leaks between tests by design (module globals) — reset.
        radar._latched_targets = []
        radar._latched_at = 0.0
        radar._had_targets = False
        self.fake = None

    def tearDown(self):
        try:
            radar.disconnect()
        except Exception:
            pass
        for p in self._patches:
            p.stop()
        radar.serial.Serial = self._orig_serial
        radar._shutting_down = False
        config.RADAR_ENABLED = self._orig_enabled
        config.RADAR_HANDSHAKE_TIMEOUT_MS = self._orig_timeout

    def _connect(self, **kw):
        self.fake = FakeRadarSerial(**kw)
        radar.serial.Serial = lambda *a, **k: self.fake
        ok = radar.connect(port="FAKE")
        time.sleep(0.15)   # let a telemetry frame land
        return ok


class TransportTest(_RadarTestBase):
    def test_handshake_ok(self):
        self.assertTrue(self._connect(targets=[_TARGET_WIRE]))
        self.assertTrue(radar.connected())
        self.assertEqual(radar.boot_id(), 777)
        self.assertEqual(len(radar.hello_info()["sensors"]), 3)

    def test_handshake_timeout_disables(self):
        self.assertFalse(self._connect(reply_hello=False))
        self.assertFalse(radar.connected())

    def test_wrong_board_is_refused(self):
        # The drive base answers hello on the same proto — caps without
        # "radar" must be refused, not silently consumed as a radar.
        self.assertFalse(self._connect(caps=["drive", "turn", "move", "stop"]))
        self.assertFalse(radar.connected())

    def test_incompatible_proto_is_refused(self):
        self.assertFalse(self._connect(proto=99))
        self.assertFalse(radar.connected())

    def test_disabled_master_switch_skips(self):
        config.RADAR_ENABLED = False
        self.assertFalse(self._connect())
        self.assertEqual(self.fake.received, [])   # never even spoke to the port

    def test_targets_normalized_from_wire(self):
        self._connect(targets=[_TARGET_WIRE])
        (t,) = radar.targets()
        self.assertAlmostEqual(t["bearing_deg"], 137.2)
        self.assertAlmostEqual(t["range_m"], 4.10)
        self.assertAlmostEqual(t["confidence"], 0.82)
        self.assertAlmostEqual(t["speed_mps"], -0.30)
        self.assertEqual(t["sensors"], 6)

    def test_radar_ok_reflects_firmware_health(self):
        self._connect(radar_ok=False)
        self.assertFalse(radar.radar_ok())
        self.fake.radar_ok = True
        time.sleep(0.1)
        self.assertTrue(radar.radar_ok())


class LatchTest(_RadarTestBase):
    def test_dropout_latches_then_expires(self):
        # The LD2450 drops a person who freezes — targets() must keep the last
        # non-empty list for RADAR_TARGET_LATCH_SECS, then admit it's empty.
        self._connect(targets=[_TARGET_WIRE])
        self.assertEqual(len(radar.targets()), 1)
        self.fake.targets_wire = []               # person freezes mid-frame
        time.sleep(0.15)                          # empty frames arrive
        self.assertEqual(len(radar.targets()), 1)  # latched, not gone
        with radar._state_lock:                   # age the latch past expiry
            radar._latched_at -= float(config.RADAR_TARGET_LATCH_SECS) + 1.0
        self.assertEqual(radar.targets(), [])

    def test_no_targets_ever_is_empty_not_latched(self):
        self._connect(targets=[])
        self.assertEqual(radar.targets(), [])

    def test_new_targets_refresh_the_latch(self):
        self._connect(targets=[_TARGET_WIRE])
        time.sleep(0.05)
        self.fake.targets_wire = [{"b": -10.0, "r": 1.5, "c": 0.9, "s": 0.0, "m": 1}]
        time.sleep(0.15)
        (t,) = radar.targets()
        self.assertAlmostEqual(t["bearing_deg"], -10.0)


class RecentFramesTest(_RadarTestBase):
    """recent_targets(): the un-latched per-frame history a body-turn decision
    reads, so it can ignore everything received before a turn settled."""

    def test_recent_frames_are_stamped_and_include_empties(self):
        self._connect(targets=[_TARGET_WIRE])
        time.sleep(0.05)
        self.fake.targets_wire = []                # person freezes: empty frames
        time.sleep(0.15)
        frames = radar.recent_targets(window_secs=5.0)
        self.assertGreater(len(frames), 2)
        stamps = [s for s, _ in frames]
        self.assertEqual(stamps, sorted(stamps))   # oldest first
        self.assertTrue(any(ts for _, ts in frames))       # the occupied frames...
        self.assertTrue(any(not ts for _, ts in frames))   # ...AND the empty ones
        # Not latched: the LAST frames are empty even though targets() still
        # remembers the person.
        self.assertEqual(frames[-1][1], [])
        self.assertEqual(len(radar.targets()), 1)

    def test_since_excludes_frames_received_before_the_stamp(self):
        self._connect(targets=[_TARGET_WIRE])
        time.sleep(0.1)
        self.fake.targets_wire = [{"b": -10.0, "r": 1.5, "c": 0.9, "s": 0.0, "m": 1}]
        time.sleep(0.15)                          # old-bearing frames drain through
        cut = time.monotonic()
        time.sleep(0.15)
        after = radar.recent_targets(window_secs=5.0, since=cut)
        self.assertTrue(after)
        self.assertTrue(all(stamp >= cut for stamp, _ in after))
        # Everything after the cut is the new bearing; the earlier 137.2° frames
        # (which the base may since have rotated away from) are not offered —
        # yet they ARE still in the un-cut window, so it is `since` doing the
        # excluding, not the buffer forgetting.
        seen_after = {t["bearing_deg"] for _, ts in after for t in ts}
        self.assertEqual(seen_after, {-10.0})
        seen_all = {t["bearing_deg"] for _, ts in radar.recent_targets(window_secs=5.0)
                    for t in ts}
        self.assertIn(137.2, seen_all)

    def test_disconnected_ring_offers_nothing(self):
        self.assertEqual(radar.recent_targets(window_secs=5.0), [])


class _FakePortInfo:
    def __init__(self, device, serial_number):
        self.device = device
        self.serial_number = serial_number


class ResolvePortTest(unittest.TestCase):
    def _resolve(self, ports, serial_no, path=None):
        with mock.patch.object(radar, "RADAR_ESP32_SERIAL", serial_no), \
             mock.patch.object(radar, "RADAR_ESP32_PORT", path), \
             mock.patch.object(radar.list_ports, "comports", return_value=ports):
            return radar.resolve_port()

    def test_serial_number_match_prefers_cu(self):
        ports = [
            _FakePortInfo("/dev/tty.usbmodem1101", "S3RADAR01"),
            _FakePortInfo("/dev/cu.usbmodem1101", "S3RADAR01"),
            _FakePortInfo("/dev/cu.usbserial-110", "BASE01"),
        ]
        self.assertEqual(self._resolve(ports, "S3RADAR01"), "/dev/cu.usbmodem1101")

    def test_serial_match_is_case_insensitive(self):
        ports = [_FakePortInfo("/dev/cu.usbmodem42", "s3radar01")]
        self.assertEqual(self._resolve(ports, "S3RADAR01"), "/dev/cu.usbmodem42")

    def test_missing_serial_falls_back_to_path(self):
        ports = [_FakePortInfo("/dev/cu.usbserial-110", "BASE01")]
        self.assertEqual(
            self._resolve(ports, "S3RADAR01", path="/dev/cu.usbmodem99"),
            "/dev/cu.usbmodem99",
        )

    def test_nothing_configured_resolves_none(self):
        self.assertIsNone(self._resolve([], None, path=None))

    def test_path_only_config(self):
        self.assertEqual(
            self._resolve([], None, path="/dev/cu.usbmodem7"), "/dev/cu.usbmodem7"
        )


if __name__ == "__main__":
    unittest.main()
