#!/usr/bin/env python3
"""
firmware/tools/gamepad_watch.py — watch the ESP32's Bluetooth gamepad pairing live.

WHY THIS EXISTS. Pad connection attempts are invisible everywhere else. The battery
menu-bar meter logs firmware LOG lines but not EVENTS, so `gamepad state=connected` /
`disconnected` never reaches any log file, and main.py only shows a pad once it is
already working. Field 2026-07-24: "gamepad support seems to not work after the
main.py program is shut down... even if I click restart ESP32 it just never
connects" — and every artifact available said the board was healthy (Bluepad32 came
up, BT stack ready, no reboot loop, nothing paired to the Mac). The 2026-07-20
no-pair bug needed raw-serial archaeology for the same reason; this is that
technique, packaged.

Shows every log line and every event the board emits, with gamepad traffic called
out, so you can watch what happens WHILE putting the pad into pairing mode.

Port sharing: takes the same advisory flock main.py and the menu bar meter use, so
the meter yields within ~1 s and reclaims the port on exit.

    venv/bin/python firmware/tools/gamepad_watch.py             # watch until Ctrl-C
    venv/bin/python firmware/tools/gamepad_watch.py --reset     # reboot first, watch a clean boot
    venv/bin/python firmware/tools/gamepad_watch.py --seconds 60

Read-only: it never sends a motion command. --reset only pulses the board's reset
line (the same DTR-only pulse the meter uses).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _read_env_port() -> "str | None":
    env = _ROOT / ".env"
    if not env.exists():
        return None
    for raw in env.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("MOTION_ESP32_PORT="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _describe(msg: dict) -> "tuple[str, str] | None":
    """(tag, text) for a line worth printing, or None to ignore telemetry noise."""
    kind = str(msg.get("type") or "")
    if kind == "log":
        text = str(msg.get("msg") or "")
        tag = "GAMEPAD" if "gamepad" in text.lower() or "bluepad" in text.lower() else "log"
        return tag, text
    if kind == "event":
        name = str(msg.get("event") or msg.get("name") or "")
        rest = {k: v for k, v in msg.items() if k not in ("v", "type", "event", "name")}
        tag = "GAMEPAD" if "gamepad" in name.lower() else "event"
        return tag, f"{name} {rest}" if rest else name
    if kind == "ack":
        return "ack", json.dumps({k: v for k, v in msg.items() if k not in ("v", "type")})
    return None      # telemetry / tofmx — far too chatty to be useful here


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port (default: MOTION_ESP32_PORT in .env)")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--reset", action="store_true",
                    help="pulse the board's reset line first, so you see a clean boot")
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
        print("Rex (main.py) is running and owns the base — shut Rex down first, since "
              "this tool is for the pad NOT connecting while Rex is off.", file=sys.stderr)
        return 1
    if not single_instance.acquire():
        print("Could not acquire the base lock (another owner?). Aborting.", file=sys.stderr)
        return 1

    ser = None
    try:
        time.sleep(1.3)                       # let the menu-bar meter release the port
        ser = serial.Serial(port, 115200, timeout=0.25)
        if args.reset:
            # Proven ESP32 auto-reset (matches the meter and sensor_monitor): a default
            # open leaves DTR+RTS asserted (EN high, running); drop DTR ONLY — RTS stays
            # asserted, pulling EN low — then re-assert to boot. NEVER drop RTS: leaving
            # DTR != RTS holds the chip in reset and the board goes silent.
            ser.dtr = False
            time.sleep(0.15)
            ser.dtr = True
            ser.reset_input_buffer()
            print(f"[{_stamp()}] reset pulse sent — watching a clean boot")

        print(f"[{_stamp()}] watching {port}. Put the pad in PAIRING mode now.")
        print("            (8BitDo Pro 2: hold the pair button until the LEDs sweep.)")
        print("            Looking for: 'gamepad: Bluepad32 ready' then 'gamepad state=connected'.")
        print("-" * 78)
        started = time.monotonic()
        seen_ready = False
        seen_connect = False
        while True:
            if args.seconds and (time.monotonic() - started) >= args.seconds:
                break
            line = ser.readline().decode("utf-8", "replace").strip()
            if not line.startswith("{"):
                if line:
                    print(f"[{_stamp()}] raw   {line}")
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            described = _describe(msg)
            if described is None:
                continue
            tag, text = described
            print(f"[{_stamp()}] {tag:7} {text}")
            low = text.lower()
            if "bluepad32 ready" in low:
                seen_ready = True
            if "connected" in low and "dis" not in low and tag == "GAMEPAD":
                seen_connect = True

        print("-" * 78)
        print(f"BT stack came up : {seen_ready}")
        print(f"pad connected    : {seen_connect}")
        if seen_ready and not seen_connect:
            print("\nThe board is listening but the pad never completed a connection.")
            print("That points at the PAD side or the RF link, not the firmware:")
            print("  * make sure the pad is in the right mode (8BitDo Pro 2: the mode")
            print("    switch matters — it remembers a DIFFERENT host per mode)")
            print("  * hold the pair button until the LEDs sweep, not just power-on")
            print("  * if it still refuses, the pad may be bonded to another host that")
            print("    is in range and grabbing it first")
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


if __name__ == "__main__":
    raise SystemExit(main())
