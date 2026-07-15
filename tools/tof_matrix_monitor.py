#!/usr/bin/env python3
"""Live 8x8 depth-grid viewer for the DFRobot Matrix ToF bench firmware.

Pairs with firmware/tof_matrix_test/tof_matrix_test.ino running on a spare
ESP32. That firmware streams one frame per line over USB serial:

    D,v0,v1,...,v63          64 distances in mm, row-major (index = y*8 + x)
    # ...                    human-readable status from the firmware

This tool reads those frames and paints an 8x8 heat-map in the terminal in
real time — near = warm/red, far = cool/blue, no-return = dim — with live
min/max/center stats and an FPS counter. It is read-only: it never writes to
the serial port, so it is safe to run against the sensor firmware as-is.

Usage:
    ./venv/bin/python tools/tof_matrix_monitor.py               # auto-detect port
    ./venv/bin/python tools/tof_matrix_monitor.py --port /dev/cu.usbserial-0001
    ./venv/bin/python tools/tof_matrix_monitor.py --list        # list serial ports
    ./venv/bin/python tools/tof_matrix_monitor.py --raw         # dump lines, no UI

Ctrl-C to quit.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from collections import deque

try:
    import serial  # pyserial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed — run: ./venv/bin/pip install pyserial")

BAUD = 115200
GRID = 8                      # 8x8
NCELLS = GRID * GRID
# Color ramp bounds (mm): distances are clamped to this window before mapping to
# hue. The VL53L7CX ranges ~20 mm..4 m; 50..3000 gives good spread on a desk.
RAMP_NEAR_MM = 50
RAMP_FAR_MM = 3000
CENTER_IDX = (27, 28, 35, 36)  # the 4 middle zones of the 8x8 grid

# ANSI helpers -------------------------------------------------------------
CLEAR = "\x1b[2J"
HOME = "\x1b[H"
EOL = "\x1b[K"          # erase to end of line
HIDE_CUR = "\x1b[?25l"
SHOW_CUR = "\x1b[?25h"
RESET = "\x1b[0m"


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """h in [0,360), s/v in [0,1] -> (r,g,b) in 0..255."""
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def dist_to_rgb(mm: int) -> tuple[int, int, int]:
    """Near -> red, far -> blue. Invalid (0) -> dark gray."""
    if mm <= 0:
        return (30, 30, 30)
    d = max(RAMP_NEAR_MM, min(RAMP_FAR_MM, mm))
    t = (d - RAMP_NEAR_MM) / (RAMP_FAR_MM - RAMP_NEAR_MM)   # 0 near .. 1 far
    hue = t * 240.0                                          # 0=red .. 240=blue
    return hsv_to_rgb(hue, 0.85, 0.95)


def cell(mm: int) -> str:
    """One 6-wide colored grid cell."""
    r, g, b = dist_to_rgb(mm)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "0;0;0" if lum > 140 else "255;255;255"
    text = "  --  " if mm <= 0 else f" {mm:4d} "
    return f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fg}m{text}{RESET}"


def find_port() -> str | None:
    """Best-guess USB-serial port for the ESP32 (CP2102/CH340/etc.)."""
    for p in list_ports.comports():
        blob = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in blob for k in ("usbserial", "slab", "wch", "cp210", "ch340", "silicon labs")):
            return p.device
    hits = (glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*")
            + glob.glob("/dev/cu.wchusbserial*") + glob.glob("/dev/ttyUSB*"))
    return hits[0] if hits else None


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Serial ports:")
    for p in ports:
        print(f"  {p.device:24s} {p.description}")


def render(frame: list[int], meta: dict, status: deque[str]) -> str:
    valid = [d for d in frame if d > 0]
    dmin = min(valid) if valid else 0
    dmax = max(valid) if valid else 0
    cvals = [frame[i] for i in CENTER_IDX if frame[i] > 0]
    center = round(sum(cvals) / len(cvals)) if cvals else 0

    out = [HOME]
    out.append(f"DFRobot 8x8 Matrix ToF  (VL53L7CX)  addr 0x33   {meta['port']}{EOL}")
    out.append(
        f"frame {meta['count']:>6}   {meta['fps']:4.1f} fps   "
        f"valid {len(valid):2d}/64   near {dmin:>4} mm   far {dmax:>4} mm   "
        f"center {center:>4} mm{EOL}"
    )
    out.append(EOL)

    header = "      " + "".join(f"  X{x}  " for x in range(GRID))
    out.append(header + EOL)
    for y in range(GRID):
        row = f"  Y{y}  " + "".join(cell(frame[y * GRID + x]) for x in range(GRID))
        out.append(row + EOL)

    out.append(EOL)
    # Distance legend.
    legend = "  near "
    for mm in (100, 500, 1000, 1500, 2000, 2500, 3000):
        r, g, b = dist_to_rgb(mm)
        legend += f"\x1b[48;2;{r};{g};{b}m   {RESET}"
    legend += " far   (mm; dim = no return)"
    out.append(legend + EOL)

    out.append(EOL)
    out.append(f"status ({len(status)} recent):{EOL}")
    for line in status:
        out.append(f"  {line}{EOL}")
    out.append("\x1b[J")   # clear anything below (shrinking status block)
    out.append("\nCtrl-C to quit.")
    return "".join(out)


def run_ui(ser: serial.Serial, port: str) -> None:
    frame = [0] * NCELLS
    status: deque[str] = deque(maxlen=6)
    count = 0
    fps = 0.0
    last_frame_t = time.time()
    last_draw = 0.0

    sys.stdout.write(HIDE_CUR + CLEAR)
    sys.stdout.flush()
    status.append("waiting for first frame (firmware settles ~5s after boot)...")

    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue

        if line.startswith("D,"):
            parts = line[2:].split(",")
            if len(parts) != NCELLS:
                status.append(f"bad frame: {len(parts)} values (want {NCELLS})")
                continue
            try:
                frame = [int(p) for p in parts]
            except ValueError:
                status.append(f"unparseable frame: {line[:40]}...")
                continue
            count += 1
            now = time.time()
            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt
        elif line.startswith("#"):
            status.append(line)
        else:
            status.append(f"?: {line[:60]}")

        # Throttle redraws to ~20 Hz max so a fast/noisy stream can't thrash the TTY.
        now = time.time()
        if now - last_draw >= 0.05:
            last_draw = now
            meta = {"port": port, "count": count, "fps": fps}
            sys.stdout.write(render(frame, meta, status))
            sys.stdout.flush()


def run_raw(ser: serial.Serial) -> None:
    while True:
        raw = ser.readline()
        if raw:
            sys.stdout.write(raw.decode("utf-8", "replace"))
            sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live 8x8 ToF depth-grid viewer.")
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=BAUD, help=f"baud (default {BAUD})")
    ap.add_argument("--raw", action="store_true", help="print raw serial lines, no UI")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list:
        list_serial_ports()
        return

    port = args.port or find_port()
    if not port:
        sys.exit("No serial port found. Plug in the ESP32 or pass --port "
                 "(see --list).")

    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open {port}: {e}")

    print(f"Reading {port} @ {args.baud}...  (Ctrl-C to quit)")
    try:
        if args.raw:
            run_raw(ser)
        else:
            run_ui(ser, port)
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        sys.stdout.write(SHOW_CUR + RESET)
        sys.exit(f"\nSerial error: {e}")
    finally:
        sys.stdout.write(SHOW_CUR + RESET + "\n")
        sys.stdout.flush()
        ser.close()


if __name__ == "__main__":
    main()
