#!/usr/bin/env python3
"""
firmware/tools/sensor_monitor.py — live bench readout of the ESP32 drive base's sensors.

The point: validate a sensor (accelerometer, compass, ToF, …) on the bench and watch it
respond BEFORE mounting it — without booting the whole Rex stack just to see a tilt
indicator. Read-only: it never sends a motion command.

Port sharing: the battery menu-bar meter holds the serial port while Rex is off, and both
it and main.py coordinate through one advisory flock (utils/single_instance). This tool
acquires that same lock, so the menu bar yields the port within ~1 s, and releases it on
exit so the meter reclaims it. If Rex (main.py) is actually running, its own Motivator
Control already shows this — so we bow out rather than fight for the port.

    venv/bin/python firmware/tools/sensor_monitor.py            # live until Ctrl-C
    venv/bin/python firmware/tools/sensor_monitor.py --seconds 8
    venv/bin/python firmware/tools/sensor_monitor.py --port /dev/cu.usbserial-110
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_env_port() -> "str | None":
    """Minimal .env scrape for MOTION_ESP32_PORT (no heavy config import)."""
    env = _ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("MOTION_ESP32_PORT"):
            _, _, val = line.partition("=")
            return val.strip().strip('"').strip("'") or None
    return None


def _bar(value: float, lo: float, hi: float, width: int = 21) -> str:
    """A centered ASCII gauge: marker position maps value in [lo,hi] across width."""
    frac = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    pos = int(round(frac * (width - 1)))
    cells = ["·"] * width
    cells[(width - 1) // 2] = "|"       # center reference
    cells[pos] = "●"
    return "".join(cells)


def _ok(flag) -> str:
    return "\033[32m ok \033[0m" if flag else "\033[31mMISS\033[0m"


def _render(tel: dict, tofmx: "dict | None", age: float) -> str:
    imu = tel.get("imu") or {}
    mag = tel.get("mag") or {}
    env = tel.get("env") or {}
    tof = tel.get("tof_mm") or {}
    L = []
    L.append("\033[2J\033[H")  # clear + home
    L.append("  DJ-R3X drive-base sensor monitor   (read-only · Ctrl-C to exit)")
    L.append("  " + "─" * 62)
    stale = "  \033[33m(telemetry stale %0.1fs)\033[0m" % age if age > 1.5 else ""
    L.append(f"  base state: {tel.get('state','—'):<10}  fault: {tel.get('fault') or 'none'}{stale}")
    L.append("")

    # IMU — the accelerometer/gyro attitude, with live tilt bars.
    ok = bool(imu.get("ok"))
    L.append(f"  IMU  (LSM6DS3 accel/gyro)   [{_ok(ok)}]")
    if ok:
        pitch = float(imu.get("pitch") or 0.0)
        roll = float(imu.get("roll") or 0.0)
        yaw = float(imu.get("yaw") or 0.0)
        L.append(f"     pitch {pitch:+6.1f}°  {_bar(pitch, -90, 90)}")
        L.append(f"     roll  {roll:+6.1f}°  {_bar(roll, -90, 90)}")
        L.append(f"     yaw   {yaw:+6.1f}°  (relative to boot heading)")
        L.append("     → tilt the board; pitch/roll should track it live.")
    else:
        L.append("     no IMU detected — firmware reprobes every ~5 s; reseat and wait,")
        L.append("     or power-cycle/reset the ESP32 after seating it.")
    L.append("")

    # Compass, climate, battery, ToF — the rest of the I2C bus at a glance.
    mok = bool(mag.get("ok"))
    heading = ""
    if mok:
        try:
            heading = "  heading≈%3.0f°" % (math.degrees(math.atan2(
                float(mag.get("y") or 0.0), float(mag.get("x") or 0.0))) % 360.0)
        except Exception:
            heading = ""
    L.append(f"  MAG  (QMC5883 compass)      [{_ok(mok)}]"
             + (f"  x={mag.get('x')} y={mag.get('y')} z={mag.get('z')}{heading}" if mok else ""))
    eok = bool(env.get("ok"))
    L.append(f"  ENV  (BMP/BME280)           [{_ok(eok)}]"
             + (f"  {env.get('t')}°C  {env.get('hpa')} hPa  {env.get('rh')}%RH" if eok else ""))
    mv = tel.get("batt_mv")
    bok = isinstance(mv, (int, float)) and mv and mv > 0
    L.append(f"  BATT (INA226 gauge)         [{_ok(bok)}]"
             + (f"  {mv/1000.0:.2f} V  {tel.get('batt_ma')} mA"
                + (f"  {tel.get('batt_soc')}%" if tel.get('batt_soc') is not None else "") if bok else ""))
    L.append("")
    L.append("  ToF radial (mm):  "
             + "  ".join(f"{k}={tof.get(k,'—')}" for k in ("fl", "fr", "rl", "rr")))
    L.append("                    "
             + "  ".join(f"{k}={tof.get(k,'—')}" for k in ("lf", "lb", "rf", "rb")))
    if tofmx:
        grid = tofmx.get("grid") or tofmx.get("mm")
        if isinstance(grid, list) and grid:
            valid = [v for v in grid if isinstance(v, (int, float)) and v > 0]
            near = min(valid) if valid else None
            L.append(f"  8×8 matrix ToF (front):  {len(grid)} cells"
                     + (f", nearest {near} mm" if near is not None else ""))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port (default: MOTION_ESP32_PORT in .env)")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds (0 = until Ctrl-C)")
    args = ap.parse_args(argv)

    port = args.port or _read_env_port()
    if not port:
        print("No serial port — set MOTION_ESP32_PORT in .env or pass --port.", file=sys.stderr)
        return 2

    try:
        import serial
    except ImportError:
        print("pyserial not installed (pip install pyserial).", file=sys.stderr)
        return 2
    from utils import single_instance

    if single_instance.is_held_by_other():
        print("Rex (main.py) is running and owns the base — open its Motivator Control "
              "for the live attitude view, or shut Rex down first.", file=sys.stderr)
        return 1
    if not single_instance.acquire():
        print("Could not acquire the base lock (another owner?). Aborting.", file=sys.stderr)
        return 1

    ser = None
    try:
        time.sleep(1.3)                         # let the menu-bar meter release the port
        ser = serial.Serial(port, 115200, timeout=0.25)
        tel = None
        tofmx = None
        last_frame_at = 0.0
        started = time.monotonic()
        last_draw = 0.0
        # If nothing streams for ~2 s, nudge the board with a DTR reset pulse.
        nudged = False
        while True:
            if args.seconds and (time.monotonic() - started) >= args.seconds:
                break
            line = ser.readline().decode("utf-8", "replace").strip()
            now = time.monotonic()
            if line.startswith("{"):
                try:
                    msg = json.loads(line)
                except Exception:
                    msg = None
                if isinstance(msg, dict):
                    if msg.get("type") == "telemetry":
                        tel, last_frame_at = msg, now
                    elif msg.get("type") == "tofmx":
                        tofmx = msg
            if not nudged and tel is None and (now - started) > 2.0:
                # Proven ESP32 auto-reset (matches the battery meter): a default open
                # leaves DTR+RTS asserted (EN high, running); drop DTR ONLY — RTS stays
                # asserted, pulling EN low — then re-assert to boot. NEVER drop RTS here:
                # leaving DTR≠RTS holds the chip in reset and the stream goes silent.
                nudged = True
                try:
                    ser.dtr = False
                    time.sleep(0.15)
                    ser.dtr = True
                    ser.reset_input_buffer()
                except Exception:
                    pass
            if tel is not None and (now - last_draw) >= 0.2:
                last_draw = now
                sys.stdout.write(_render(tel, tofmx, now - last_frame_at))
                sys.stdout.write("\n")
                sys.stdout.flush()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        single_instance.release()
        sys.stdout.write("\033[0m\n")


if __name__ == "__main__":
    raise SystemExit(main())
