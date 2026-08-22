#!/usr/bin/env python3
"""
tools/compass_calibrate.py — in-situ hard/soft-iron calibration for the QMC5883L.

The compass must be calibrated MOUNTED ON R3X: the robot's own motors, drivers,
pack, and wiring are the hard/soft iron being corrected, so a bench calibration
of the bare board is worthless. Procedure:

  1. Robot ON, Rex OFF (this tool needs the motion serial port; it politely
     refuses while main.py holds the single-instance lock).
  2. Run:  venv/bin/python tools/compass_calibrate.py [--secs 60]
  3. While it samples, sweep the WHOLE ROBOT through slow figure-8s: rotate it
     through a full 360° of heading, tipping it forward/back and side to side
     as far as is safe (the cloud needs all three axes to see field extremes).
     A lazy flat-only spin degenerates the Z axis and the tool will say so.
  4. On success the calibration (offsets, scales, ambient |B|) is written to
     config.COMPASS_CALIBRATION_PATH and hardware/compass.py loads it on init.

Motors stay off throughout — drive current would contaminate the very field
being measured (the idle-electronics field is part of the ambient signature,
which is exactly what we want to calibrate around).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | compass_cal | %(message)s")
log = logging.getLogger("compass_cal")


def main() -> int:
    ap = argparse.ArgumentParser(description="In-situ QMC5883L figure-8 calibration")
    ap.add_argument("--secs", type=float, default=60.0,
                    help="sampling window (default 60s — two slow figure-8s)")
    ap.add_argument("--out", default=None,
                    help="output path (default: config.COMPASS_CALIBRATION_PATH)")
    args = ap.parse_args()

    from utils import single_instance
    if single_instance.is_held_by_other():
        print("main.py is running and owns the serial port — stop Rex first.")
        return 1

    # TAKE the lock, don't just check it. The battery menubar yields the port when
    # the single-instance lock is HELD (its documented dormant handoff) — checking
    # that nobody else holds it gives the meter no reason to let go, so this tool
    # used to fail to open the port and tell the human to go quit a menubar app by
    # hand. main.py takes the lock at startup for exactly this reason; so does this.
    if not single_instance.acquire():
        print("Could not take the single-instance lock — is something else "
              "starting up? Try again in a moment.")
        return 1

    from hardware import motion
    from hardware.compass import compute_calibration, save_calibration

    try:
        time.sleep(2.0)          # the meter polls the lock ~1/s; let it stand down
        if not motion.connect():
            print("Could not open the motion serial link. If the battery menubar "
                  "is still holding it, quit the menubar app and re-run.")
            return 1
        return _sample_and_write(args, motion, compute_calibration, save_calibration)
    finally:
        single_instance.release()


def _sample_and_write(args, motion, compute_calibration, save_calibration) -> int:
    try:
        # Wait for the mag block to prove the sensor is alive before asking the
        # human to wave a 20 kg robot around.
        t0 = time.monotonic()
        while True:
            snap = motion.telemetry() or {}
            if (snap.get("mag") or {}).get("ok"):
                break
            if time.monotonic() - t0 > 10.0:
                print("No magnetometer in telemetry (mag.ok false) — is the "
                      "QMC5883L wired and the firmware current?")
                return 1
            time.sleep(0.3)

        print(f"Sampling for {args.secs:.0f}s — sweep the robot through slow "
              f"figure-8s NOW (full 360° of heading, tip it as far as safe).")
        samples: list[tuple] = []
        t0 = time.monotonic()
        last_note = 0.0
        while time.monotonic() - t0 < args.secs:
            snap = motion.telemetry() or {}
            mag = snap.get("mag") or {}
            if mag.get("ok") and not mag.get("ovl"):
                samples.append((float(mag["x"]), float(mag["y"]), float(mag["z"])))
            elapsed = time.monotonic() - t0
            if elapsed - last_note >= 10.0:
                last_note = elapsed
                print(f"  {elapsed:4.0f}s — {len(samples)} samples")
            time.sleep(0.05)                     # ~2x the firmware's 10 Hz publish

        print(f"Collected {len(samples)} samples.")
        try:
            cal = compute_calibration(samples)
        except ValueError as exc:
            print(f"✗ Calibration failed: {exc}")
            return 1

        # Coverage sanity. REFUSES on a sweep that never happened, warns on a
        # lopsided one — a stationary run used to sail through and install a
        # calibration built from wherever the sensor was parked.
        from hardware.compass import calibration_coverage_ok
        xs, ys, zs = zip(*samples)
        spans = [max(v) - min(v) for v in (xs, ys, zs)]
        ok, note = calibration_coverage_ok(cal, spans)
        if not ok:
            print(f"✗ Calibration REFUSED: {note}")
            print("   Sweep the whole robot through slow figure-8s while it samples "
                  "— a full 360° of heading, tipping forward/back and side to side.")
            return 1
        if note:
            print(f"⚠  {note}")

        path = save_calibration(cal, args.out)
        print(f"✓ Calibration written to {path}")
        print(f"  offsets: {['%.1f' % v for v in cal.offset]}")
        print(f"  scales:  {['%.3f' % v for v in cal.scale]}")
        print(f"  ambient |B|: {cal.field_norm:.1f} counts")
        print("Verify with:  venv/bin/python -m hardware.compass")
        return 0
    finally:
        motion.disconnect()


if __name__ == "__main__":
    sys.exit(main())
