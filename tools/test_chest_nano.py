#!/usr/bin/env python3
"""
Chest Arduino serial diagnostic tool.

Usage:
    python tools/test_chest_nano.py [port] [baud]

Defaults to values in .env / config.py.  Override on the command line for
troubleshooting, e.g.:
    python tools/test_chest_nano.py /dev/cu.usbserial-11420 115200

The tool lists all detected serial ports, opens the chest port, and lets you
send commands interactively so you can verify the LEDs respond.

Available commands (from chest_nano.ino):
    STARTUP          ShortCircuit animation → auto IDLE
    IDLE             RandomBlocks2 normal brightness
    ACTIVE           RandomBlocks2 high brightness (200)
    SPEAK:excited    AllRed full brightness
    SPEAK:sad        AllBlue dim
    SPEAK:angry      rapid red strobe
    SPEAK:happy      confetti
    SPEAK:neutral    RandomBlocks2
    SLEEP            dim red slow breathe
    OFF              all LEDs off
    NEXT             cycle to next pattern
    scan             re-scan and list serial ports
    quit / q         exit
"""

import sys
import time
import glob

# ── Port discovery ─────────────────────────────────────────────────────────────

def list_serial_ports() -> list[str]:
    """Return all /dev/cu.* USB serial ports sorted by type."""
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*"))
    return ports


def detect_default_port() -> tuple[str | None, int]:
    """Read port and baud from project config, falling back to detection."""
    import os, pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"

    port, baud = None, 115200

    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ARDUINO_CHEST_PORT="):
                port = line.split("=", 1)[1].strip()
            # config.py baud
    try:
        sys.path.insert(0, str(project_root))
        import config
        baud = getattr(config, "CHEST_ARDUINO_BAUD", 115200)
    except Exception:
        pass

    return port, baud


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed.  Run: pip install pyserial")
        sys.exit(1)

    # Determine port / baud from args or config
    default_port, default_baud = detect_default_port()

    port = sys.argv[1] if len(sys.argv) > 1 else default_port
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else default_baud

    print("\n=== Chest Arduino Diagnostic ===\n")
    print("Detected USB serial ports:")
    ports = list_serial_ports()
    if ports:
        for p in ports:
            marker = " ← configured" if p == port else ""
            print(f"  {p}{marker}")
    else:
        print("  (none found)")

    if port is None:
        print("\nNo port configured — pass a port as the first argument.")
        sys.exit(1)

    print(f"\nOpening {port} at {baud} baud…")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"ERROR: Could not open port: {exc}")
        sys.exit(1)

    print("Waiting 2 s for Arduino boot…")
    time.sleep(2.0)
    ser.reset_input_buffer()
    print("Ready.\n")

    print("Type a command and press Enter.  The LED response confirms communication.")
    print("Commands: STARTUP IDLE ACTIVE SPEAK:excited SPEAK:sad SPEAK:angry")
    print("          SPEAK:happy SPEAK:neutral SLEEP OFF NEXT  |  'scan' 'quit'\n")

    try:
        while True:
            try:
                cmd = input("chest> ").strip()
            except EOFError:
                break

            if not cmd:
                continue

            if cmd.lower() in ("quit", "q", "exit"):
                break

            if cmd.lower() == "scan":
                print("Serial ports:")
                for p in list_serial_ports():
                    print(f"  {p}")
                continue

            payload = (cmd + "\n").encode()
            ser.write(payload)
            print(f"  → sent {payload!r}")
            time.sleep(0.05)

    finally:
        print("\nSending OFF and closing port.")
        try:
            ser.write(b"OFF\n")
            time.sleep(0.1)
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
