#!/usr/bin/env python3
"""ld2450_synth.py — synthetic HLK-LD2450 frame generator.

WHY THIS EXISTS: the radar modules were ordered before the firmware was
written (docs/radar-bearing-prior-spec.md — "everything up to and including
the Mac-side consumer must be buildable and testable today against synthetic
data"), and it stays in the tree afterward as the regression fixture source
for the frame parser (tests/test_radar_parser.py).

Encodes the OFFICIAL wire format, cross-checked against Hi-Link's protocol doc
V1.03 + the ESPHome core driver (see firmware/djr3x_radar/ld2450.h for the
source notes): 30-byte frames AA FF 03 00 | 3x8-byte slots | 55 CC,
little-endian, sign-and-magnitude with an INVERTED flag (high bit 1 = positive,
value = raw - 0x8000; high bit 0 = negative, value = -raw). x/y in mm, speed
in cm/s. Empty slots are all zeros. Do NOT "fix" the sign encoding against the
csRon or TillFleisch drivers — both decode x/speed with the opposite polarity
to the official doc's own worked example.

Usage (demo — hex-dump a scenario's frames):
    venv/bin/python firmware/tools/ld2450_synth.py --scenario seam_crossing

Read-only tool: it never opens a serial port; streaming synthetic bytes at the
Mac reader is done in-process by the unit tests (tests/test_radar.py).
"""
from __future__ import annotations

import argparse
import math
import struct
import sys

HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
TAIL = bytes([0x55, 0xCC])
FRAME_BYTES = 30
SLOTS = 3


def encode_signed(v: int) -> int:
    """Official LD2450 data-frame sign encoding (NOT two's complement)."""
    if v >= 0:
        if v > 0x7FFF:
            raise ValueError(f"magnitude too large: {v}")
        return v | 0x8000
    if -v > 0x7FFF:
        raise ValueError(f"magnitude too large: {v}")
    return -v


def decode_signed(raw: int) -> int:
    """Inverse of encode_signed — the reference decode for tests."""
    return raw - 0x8000 if raw & 0x8000 else -raw


def build_frame(targets: list[tuple[int, int, int]], res_mm: int = 360) -> bytes:
    """One 30-byte data frame. targets: up to 3 (x_mm, y_mm, speed_cms) tuples;
    remaining slots are zero-filled (absent)."""
    if len(targets) > SLOTS:
        raise ValueError("LD2450 reports at most 3 targets")
    out = bytearray(HEADER)
    for i in range(SLOTS):
        if i < len(targets):
            x, y, speed = targets[i]
            out += struct.pack(
                "<HHHH", encode_signed(x), encode_signed(y), encode_signed(speed), res_mm
            )
        else:
            out += bytes(8)
    out += TAIL
    assert len(out) == FRAME_BYTES
    return bytes(out)


def local_from_robot(bearing_deg: float, range_m: float, mount_deg: float
                     ) -> tuple[int, int] | None:
    """Robot-frame (bearing +=left/CCW, 0=forward) -> one sensor's local
    (x_mm, y_mm), official convention (+x = right of sensor). None when the
    target is outside the sensor's ±60° FOV."""
    local = (bearing_deg - mount_deg + 180.0) % 360.0 - 180.0
    if abs(local) > 60.0:
        return None
    rad = math.radians(local)
    return (round(-range_m * 1000.0 * math.sin(rad)),
            round(range_m * 1000.0 * math.cos(rad)))


# ---- Scripted scenarios (the spec's list) ---------------------------------

def scenario_seam_crossing(mount_deg: float = 0.0, other_mount_deg: float = 120.0,
                           steps: int = 21) -> list[tuple[bytes, bytes]]:
    """A single person walking across the seam between two sensors at 3 m:
    robot bearing sweeps 30°..90°, so they leave sensor A's FOV as they enter
    sensor B's, overlapping in the middle. Returns (frame_a, frame_b) pairs."""
    frames = []
    for i in range(steps):
        bearing = 30.0 + 60.0 * i / (steps - 1)
        pair = []
        for mount in (mount_deg, other_mount_deg):
            xy = local_from_robot(bearing, 3.0, mount)
            pair.append(build_frame([(*xy, 25)] if xy else []))
        frames.append(tuple(pair))
    return frames


def scenario_two_targets() -> bytes:
    """Two targets + one empty slot in a single frame."""
    return build_frame([(-782, 1713, -16), (400, 2500, 8)])


def scenario_zero_targets() -> bytes:
    return build_frame([])


def scenario_malformed_frame() -> bytes:
    """A frame whose tail is corrupted — the parser must count it bad and
    recover on the next good frame."""
    good = bytearray(build_frame([(100, 1000, 5)]))
    good[-1] = 0x00
    return bytes(good)


def scenario_truncated_frame() -> bytes:
    """The first 17 bytes of a valid frame (cut mid-slot, as a hot-unplug
    would) — feeding this then a good frame must yield only the good one."""
    return build_frame([(100, 1000, 5)])[:17]


def official_example_frame() -> bytes:
    """The worked example from Hi-Link's protocol doc §2.3: slot 0 raw bytes
    0E 03 B1 86 10 00 40 01 -> x=-782 mm, y=+1713 mm, speed=-16 cm/s,
    resolution=320 mm; slots 1/2 absent."""
    return HEADER + bytes.fromhex("0e03b186100040 01".replace(" ", "")) + bytes(16) + TAIL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="seam_crossing",
                    choices=["seam_crossing", "two_targets", "zero_targets",
                             "malformed_frame", "truncated_frame", "official_example"])
    args = ap.parse_args()
    if args.scenario == "seam_crossing":
        for fa, fb in scenario_seam_crossing():
            print(f"A:{fa.hex()}  B:{fb.hex()}")
    else:
        data = globals()[f"scenario_{args.scenario}"]() \
            if args.scenario != "official_example" else official_example_frame()
        print(data.hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
