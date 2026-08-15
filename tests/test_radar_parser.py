"""Regression tests for the radar firmware's LD2450 parser + fusion.

These compile the ACTUAL firmware sources (firmware/djr3x_radar/ld2450.cpp and
fusion.cpp — Arduino-free by contract) with clang++ into a small host binary
(firmware/tools/radar_parse_host.cpp) and drive it with synthetic byte streams
from firmware/tools/ld2450_synth.py. No hardware, no Arduino toolchain — the
same C++ the ESP32-S3 runs is what's under test, so parser drift can't hide
behind a Python mirror.

Run:  venv/bin/python -m unittest tests.test_radar_parser
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "firmware" / "tools"))
import ld2450_synth as synth  # noqa: E402

_FIRMWARE = _ROOT / "firmware" / "djr3x_radar"
_SHIM = _ROOT / "firmware" / "tools" / "radar_parse_host.cpp"

_build_dir: tempfile.TemporaryDirectory | None = None
_binary: Path | None = None


def _build() -> Path:
    global _build_dir, _binary
    if _binary is not None:
        return _binary
    _build_dir = tempfile.TemporaryDirectory(prefix="radar_host_")
    out = Path(_build_dir.name) / "radar_parse_host"
    cmd = [
        "clang++", "-std=c++17", "-O1", "-Wall",
        str(_SHIM),
        str(_FIRMWARE / "ld2450.cpp"),
        str(_FIRMWARE / "fusion.cpp"),
        "-o", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    _binary = out
    return out


def _run(mode: list[str], data: bytes) -> list[dict]:
    proc = subprocess.run(
        [str(_build())] + mode, input=data, capture_output=True, timeout=30
    )
    if proc.returncode != 0:
        raise AssertionError(f"shim failed: {proc.stderr.decode()}")
    return [json.loads(line) for line in proc.stdout.decode().splitlines() if line]


def parse(data: bytes) -> tuple[list[dict], dict]:
    out = _run(["parse"], data)
    return [m["targets"] for m in out if "targets" in m], out[-1]["summary"]


def build_cmd(word: str, value: str = "") -> str:
    return _run(["build", word] + ([value] if value else []), b"")[0]["cmd"]


def find_ack(word: str, data: bytes) -> dict:
    return _run(["ack", word], data)[0]


def fuse(ticks: list[list[tuple]], flip: bool = False) -> list[list[dict]]:
    """ticks: each a list of (sensor, mount_deg, x_mm, y_mm, speed_cms)."""
    lines = []
    for tick in ticks:
        for t in tick:
            lines.append(" ".join(str(v) for v in t))
        lines.append("")
    out = _run(["fuse"] + (["--flip"] if flip else []), "\n".join(lines).encode())
    return [m["fused"] for m in out]


class ParserTest(unittest.TestCase):
    def test_official_datasheet_example(self):
        # The worked example from Hi-Link's protocol doc §2.3 — the sign trap:
        # high bit 1 = POSITIVE. A driver with the (common) inverted decode
        # returns x=+782 / y=-1713 here and must fail this test.
        frames, summary = parse(synth.official_example_frame())
        self.assertEqual(summary["frames_ok"], 1)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), 1)
        t = frames[0][0]
        self.assertEqual(t["x"], -782)
        self.assertEqual(t["y"], 1713)
        self.assertEqual(t["speed"], -16)
        self.assertEqual(t["res"], 320)

    def test_two_targets_one_empty_slot(self):
        frames, summary = parse(synth.scenario_two_targets())
        self.assertEqual(summary["frames_ok"], 1)
        self.assertEqual([(t["x"], t["y"], t["speed"]) for t in frames[0]],
                         [(-782, 1713, -16), (400, 2500, 8)])

    def test_zero_targets_frame_parses_empty(self):
        frames, summary = parse(synth.scenario_zero_targets())
        self.assertEqual(summary["frames_ok"], 1)
        self.assertEqual(frames[0], [])

    def test_encode_decode_round_trip(self):
        cases = [(-3000, 1, -128), (0, 6000, 0), (32767, 32767, 100),
                 (-32767, 500, -100), (1, 2, 3)]
        frames, summary = parse(b"".join(synth.build_frame([c]) for c in cases))
        self.assertEqual(summary["frames_ok"], len(cases))
        got = [(f[0]["x"], f[0]["y"], f[0]["speed"]) for f in frames]
        self.assertEqual(got, cases)

    def test_malformed_tail_counted_and_recovered(self):
        good = synth.build_frame([(250, 1500, 10)])
        frames, summary = parse(synth.scenario_malformed_frame() + good + good)
        self.assertGreaterEqual(summary["frames_bad"], 1)
        self.assertEqual(summary["frames_ok"], 2)
        self.assertEqual(frames[-1][0]["x"], 250)

    def test_mid_frame_truncation_recovers(self):
        good = synth.build_frame([(250, 1500, 10)])
        frames, summary = parse(synth.scenario_truncated_frame() + good + good)
        # The truncated fragment is absorbed into a bad frame; both good frames
        # must still come out.
        self.assertEqual(summary["frames_ok"], 2)
        self.assertEqual(len(frames), 2)

    def test_garbage_prefix_resyncs(self):
        # Boot chatter / config-ACK bytes before the first clean frame.
        garbage = bytes.fromhex("fdfcfbfa0400ff0001000403020100aaff")
        good = synth.build_frame([(100, 1000, 5)])
        frames, summary = parse(garbage + good)
        self.assertEqual(summary["frames_ok"], 1)
        self.assertGreater(summary["bytes_dropped"], 0)
        self.assertEqual(frames[0][0]["y"], 1000)

    def test_seam_crossing_stream(self):
        # The spec's canonical scenario: a person crossing the 0°/120° seam.
        # Every emitted A-frame and B-frame must parse; nothing bad, nothing
        # dropped in a clean stream.
        stream = b"".join(fa + fb for fa, fb in synth.scenario_seam_crossing())
        frames, summary = parse(stream)
        self.assertEqual(summary["frames_ok"], 42)
        self.assertEqual(summary["frames_bad"], 0)
        self.assertEqual(summary["bytes_dropped"], 0)


class ConfigCommandTest(unittest.TestCase):
    """The Bluetooth write + MAC readback, pinned to the protocol doc's own
    worked byte examples (V1.03 §2.2.10 / §2.2.11). These command words are the
    two things boot config can't get wrong quietly: a bad word ACKs nothing and
    looks exactly like an absent module."""

    # ld2450.h's sentinel: the MAC a module reports once its radio is down.
    MAC_BT_OFF = "080504030201"

    def test_bluetooth_off_frame_matches_doc(self):
        # Doc §2.2.10 shows the ON frame; off is the same with value 0x0000.
        self.assertEqual(build_cmd("00a4", "0100"), "fdfcfbfa0400a400010004030201")
        self.assertEqual(build_cmd("00a4", "0000"), "fdfcfbfa0400a400000004030201")

    def test_mac_query_frame_matches_doc(self):
        # Doc §2.2.11 send data, byte for byte.
        self.assertEqual(build_cmd("00a5", "0100"), "fdfcfbfa0400a500010004030201")

    def test_mac_ack_yields_six_address_bytes(self):
        # Doc §2.2.11 ACK: status 0000 then the MAC 8F272EB80F65. The value the
        # firmware gets back is status-stripped, so a module with its radio ON
        # can never be mistaken for the off sentinel.
        ack = bytes.fromhex("fdfcfbfa0a00a50100008f272eb80f6504030201")
        got = find_ack("00a5", ack)
        self.assertTrue(got["found"])
        self.assertEqual(got["value"], "00008f272eb80f65")
        self.assertNotEqual(got["value"][4:], self.MAC_BT_OFF)

    def test_mac_ack_reports_bluetooth_off_sentinel(self):
        ack = bytes.fromhex("fdfcfbfa0a00a5010000" + self.MAC_BT_OFF + "04030201")
        got = find_ack("00a5", ack)
        self.assertTrue(got["found"])
        self.assertEqual(got["value"][4:], self.MAC_BT_OFF)

    def test_mac_ack_found_among_interleaved_data_frames(self):
        # Config mode pauses data reporting, but frames already in flight land
        # in the same read buffer — the readback must survive that.
        ack = bytes.fromhex("fdfcfbfa0a00a5010000" + self.MAC_BT_OFF + "04030201")
        stream = synth.build_frame([(250, 1500, 10)]) + ack + \
            synth.build_frame([(250, 1500, 10)])
        got = find_ack("00a5", stream)
        self.assertTrue(got["found"])
        self.assertEqual(got["value"][4:], self.MAC_BT_OFF)

    def test_ack_for_a_different_word_is_not_matched(self):
        # A firmware-version ACK must never be read as a MAC — that would
        # decode fw bytes as an address and report a phantom "radio on".
        ack = bytes.fromhex("fdfcfbfa0c00a00100000100020416022206" + "04030201")
        self.assertFalse(find_ack("00a5", ack)["found"])


class FusionTest(unittest.TestCase):
    def _local(self, bearing: float, range_m: float, mount: float) -> tuple[int, int]:
        xy = synth.local_from_robot(bearing, range_m, mount)
        assert xy is not None
        return xy

    def test_seam_dedup_merges_two_sensors(self):
        # One person at robot bearing 60°, 3 m — dead on the seam between the
        # 0° and 120° mounts, so both report them at their ±60° FOV edge.
        x0, y0 = self._local(60.0, 3.0, 0.0)
        x1, y1 = self._local(60.0, 3.0, 120.0)
        (fused,) = fuse([[(0, 0.0, x0, y0, 20), (1, 120.0, x1, y1, 20)]])
        self.assertEqual(len(fused), 1)
        t = fused[0]
        self.assertAlmostEqual(t["b"], 60.0, delta=1.0)
        self.assertAlmostEqual(t["r"], 3.0, delta=0.1)
        self.assertEqual(t["m"], 0b011)          # both sensors contributed
        # Agreement raises confidence above what either edge return carries.
        (solo,) = fuse([[(0, 0.0, x0, y0, 20)]])
        self.assertGreater(t["c"], solo[0]["c"])

    def test_distinct_targets_stay_separate(self):
        xa, ya = self._local(10.0, 2.0, 0.0)
        xb, yb = self._local(-45.0, 4.0, 0.0)
        (fused,) = fuse([[(0, 0.0, xa, ya, 0), (0, 0.0, xb, yb, 0)]])
        self.assertEqual(len(fused), 2)
        bearings = sorted(t["b"] for t in fused)
        self.assertAlmostEqual(bearings[0], -45.0, delta=1.0)
        self.assertAlmostEqual(bearings[1], 10.0, delta=1.0)

    def test_confidence_falls_off_toward_fov_edge(self):
        xc, yc = self._local(0.0, 3.0, 0.0)      # boresight
        xe, ye = self._local(55.0, 3.0, 0.0)     # near the ±60° edge
        (fused,) = fuse([[(0, 0.0, xc, yc, 0), (0, 0.0, xe, ye, 0)]])
        by_bearing = {round(t["b"]): t for t in fused}
        self.assertEqual(by_bearing[0]["c"], 1.0)
        self.assertLess(by_bearing[55]["c"], 0.7)

    def test_rear_seam_wraps_correctly(self):
        # A person dead astern (±180°) seen by both rear sensors (+120/-120
        # mounts, each at local ±60°). A naive bearing average would say 0°
        # (dead ahead) — the circular mean must keep them astern.
        x1, y1 = self._local(180.0, 2.5, 120.0)
        x2, y2 = self._local(-180.0, 2.5, -120.0)
        (fused,) = fuse([[(1, 120.0, x1, y1, 0), (2, -120.0, x2, y2, 0)]])
        self.assertEqual(len(fused), 1)
        self.assertGreater(abs(fused[0]["b"]), 179.0)

    def test_bearing_sign_convention_is_ccw_left(self):
        # +x = right of sensor (official) must land at a NEGATIVE (right/CW)
        # bearing in the robot frame — the project-wide +left/CCW convention.
        (fused,) = fuse([[(0, 0.0, 1000, 1732, 0)]])   # 30° right of boresight
        self.assertAlmostEqual(fused[0]["b"], -30.0, delta=0.5)
        # RADAR_FLIP_X mirrors it — the one-flag fix if bring-up shows mirrored
        # bearings (x polarity is unverified in the official docs).
        (flipped,) = fuse([[(0, 0.0, 1000, 1732, 0)]], flip=True)
        self.assertAlmostEqual(flipped[0]["b"], 30.0, delta=0.5)

    def test_range_band_discards_implausible(self):
        # 5 cm (inside the shell) and 12 m (beyond spec) both vanish.
        (fused,) = fuse([[(0, 0.0, 0, 50, 0), (0, 0.0, 0, 12000, 0),
                          (0, 0.0, 0, 2000, 0)]])
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["r"], 2.0, delta=0.01)

    def test_speed_passthrough_and_units(self):
        (fused,) = fuse([[(0, 0.0, 0, 3000, -40)]])   # -40 cm/s -> -0.4 m/s
        self.assertAlmostEqual(fused[0]["s"], -0.4, delta=0.01)


if __name__ == "__main__":
    unittest.main()
