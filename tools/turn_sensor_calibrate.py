#!/usr/bin/env python3
"""Level, motor-driven compass/gyro turn calibration for the assembled robot.

Turns in stationary increments and samples the magnetometer only after each
turn has completed and motor current has settled. This intentionally calibrates
horizontal heading only; it is for robots that cannot safely be figure-8 tilted.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def wait_telemetry(motion, timeout=4.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        motion.send({"cmd": "ping"})
        tel = motion.telemetry() or {}
        if (tel.get("imu") or {}).get("ok") and (tel.get("mag") or {}).get("ok"):
            return tel
        time.sleep(0.12)
    return None


def wait_done_with_heartbeat(motion, seq: int, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        motion.send({"cmd": "ping"})
        done = motion.wait_done(seq, timeout=0.12)
        if done:
            return done
    return None


def settled_samples(motion, seconds=2.0):
    samples = []
    yaws = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        motion.send({"cmd": "ping"})
        tel = motion.telemetry() or {}
        mag = tel.get("mag") or {}
        imu = tel.get("imu") or {}
        if mag.get("ok") and not mag.get("ovl"):
            samples.append((float(mag["x"]), float(mag["y"]), float(mag["z"])))
        if imu.get("ok"):
            yaws.append(float(imu["yaw"]))
        time.sleep(0.1)
    return samples, (yaws[-1] if yaws else None), motion.telemetry() or {}


def main() -> int:
    from hardware import motion
    from hardware.compass import Calibration, save_calibration
    from utils import single_instance

    if single_instance.is_held_by_other() or not single_instance.acquire():
        print("Another Rex process owns the motion base; aborting.")
        return 1
    try:
        if not motion.connect():
            print("Could not connect to the motion controller.")
            return 1
        motion.send({"cmd": "stop"})
        tel = wait_telemetry(motion)
        if not tel:
            print("IMU or compass telemetry unavailable; aborting.")
            return 1
        odom = tel.get("odom") or {}
        if tel.get("owner") == "manual":
            print("Gamepad still owns the base (manual mode); aborting.")
            return 1
        if abs(float(odom.get("lin") or 0.0)) > 0.01 or abs(float(odom.get("ang") or 0.0)) > 0.03:
            print("Base is not stationary after stop; aborting.")
            return 1
        if bool(tel.get("charging")) or float(tel.get("batt_mv") or 0) >= 14000:
            print("Charger detected; motor calibration is blocked.")
            return 1

        all_samples = []
        observations = []
        for direction, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
            print(f"\n{direction} 360° sweep")
            for step in range(8):
                before = motion.telemetry() or {}
                yaw0 = float((before.get("imu") or {}).get("yaw", 0.0))
                seq = motion.send({"cmd": "turn", "deg": sign * 45.0, "rate": 45.0})
                done = wait_done_with_heartbeat(motion, seq)
                if not done or done.get("result") not in ("ok", "completed"):
                    print(f"Turn {step + 1} failed: {done}")
                    return 1
                samples, yaw1, after = settled_samples(motion)
                if yaw1 is None or not samples:
                    print("Sensor telemetry disappeared; aborting.")
                    return 1
                delta = diff(yaw1, yaw0)
                observations.append((sign * 45.0, delta))
                all_samples.extend(samples)
                print(f"  {step + 1}/8  gyro {delta:+5.1f}°  "
                      f"mag=({samples[-1][0]:.0f},{samples[-1][1]:.0f},{samples[-1][2]:.0f})")

        # Horizontal hard/soft-iron correction. Z cannot be independently
        # characterized without tilting, so center it on the level sweep and
        # retain its native scale.
        xs, ys, zs = zip(*all_samples)
        ox, oy, oz = ((max(v) + min(v)) / 2.0 for v in (xs, ys, zs))
        hx, hy = (max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0
        if min(hx, hy) < 50:
            print("Compass horizontal coverage was too small; calibration not saved.")
            return 1
        radius = (hx + hy) / 2.0
        cal = Calibration(offset=(ox, oy, oz), scale=(radius / hx, radius / hy, 1.0), loaded=True)
        mags = [math.sqrt(sum(c * c for c in cal.apply(*s))) for s in all_samples]
        cal.field_norm = sum(mags) / len(mags)
        path = save_calibration(cal)

        left = [actual for wanted, actual in observations if wanted > 0]
        right = [actual for wanted, actual in observations if wanted < 0]
        print(f"\nSaved level calibration: {path}")
        print(f"Offsets: {[round(v, 1) for v in cal.offset]}")
        print(f"Scales:  {[round(v, 3) for v in cal.scale]}")
        print(f"Mean gyro increment: left {sum(left)/len(left):+.1f}°, "
              f"right {sum(right)/len(right):+.1f}° (commanded ±45°)")
        return 0
    finally:
        try:
            motion.send({"cmd": "stop"})
            time.sleep(0.2)
            motion.disconnect()
        except Exception:
            pass
        single_instance.release()


if __name__ == "__main__":
    raise SystemExit(main())
