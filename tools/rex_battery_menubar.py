#!/usr/bin/env python3
"""
rex_battery_menubar.py — macOS menu bar battery meter for the ESP32 motion base.

A tiny always-on menu bar app (rumps/Cocoa) that shows the drive base's pack
state of charge, voltage, and current even when the robot (main.py) is OFF.
Installed as a LaunchAgent alongside the wake-word supervisor by
scripts/install_supervisor.sh.

How it gets the data with ZERO firmware or protocol changes:
  The motion firmware streams a telemetry frame at 10 Hz from the moment it
  boots — unconditionally, before any handshake (telemetryTask in
  firmware/djr3x_motion/djr3x_motion.ino). Every frame carries batt_mv /
  batt_ma / batt_soc. So this app just opens MOTION_ESP32_PORT read-only and
  parses NDJSON. It NEVER WRITES A BYTE: the firmware's comms watchdog only
  arms after the first line received from the Mac (seen_mac), so a purely
  passive listener can never cause a comms_lost fault or claim ownership.

How it shares the serial port with main.py (ports are exclusive-open):
  Same dormant pattern the supervisor uses for the microphone. main.py holds
  the single-instance flock (utils/single_instance.py) for its whole lifetime,
  awake or asleep. This app polls that lock ~1×/s:
    - lock held  → close the port (main.py owns the base), show the last
                   reading greyed as "Rex is running"
    - lock free  → reopen the port and resume the live meter
  The flock auto-frees if main.py crashes, so the meter recovers on its own.
  main.py takes the lock at startup, well before motion connects, and
  hardware/motion.py opens with retries — so the ~1 s release lag is absorbed.

Run directly for debugging:
    venv/bin/python tools/rex_battery_menubar.py            # the menu bar app
    venv/bin/python tools/rex_battery_menubar.py --probe    # 5 s of raw battery
                                                            # frames to stdout
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make utils.single_instance importable without importing the heavy project
# config (mirrors rex_supervisor.py — this process must start even when
# apikeys.py / full config would fail).
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | rex_battery | %(levelname)s | %(message)s",
)
log = logging.getLogger("rex_battery")

_SOC_LOW_PCT = 20          # 🪫 at or below this
_CHARGING_MA = -50         # batt_ma below this (signed, + = discharging) → ⚡
_STALE_SECS = 5.0          # no telemetry for this long while open → reopen port
_LOCK_POLL_SECS = 1.0      # how often the worker re-checks port/lock state


# ── Minimal .env reading (no project config import) ────────────────────────────

def _read_env_file() -> dict[str, str]:
    """Parse KEY=VALUE lines from .env (same tolerant parser as the supervisor)."""
    env: dict[str, str] = {}
    path = _PROJECT_ROOT / ".env"
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            env[key.strip()] = value
    except OSError:
        pass
    return env


def _motion_port() -> str:
    env = _read_env_file()
    return (os.environ.get("MOTION_ESP32_PORT") or env.get("MOTION_ESP32_PORT") or "").strip()


def _motion_baud() -> int:
    env = _read_env_file()
    raw = (os.environ.get("MOTION_BAUD") or env.get("MOTION_BAUD") or "").strip()
    try:
        return int(raw) if raw else 115200
    except ValueError:
        return 115200


def _rex_running() -> bool:
    """True while any main.py controller is alive (awake OR asleep)."""
    try:
        from utils import single_instance
        return single_instance.is_held_by_other()
    except Exception as exc:
        log.debug("single_instance check failed: %s", exc)
        return False


# ── Shared snapshot (worker thread → UI timer) ─────────────────────────────────

_snap_lock = threading.Lock()
_snap: dict = {
    "mode": "connecting",   # connecting | live | dormant | no_port
    "port": "",
    "batt_mv": None,        # int mV, None = never seen, -1 = sensor unwired
    "batt_ma": None,        # int mA signed (+ = discharging)
    "batt_soc": None,       # int %, -1 = unknown
    "state": "",            # base state string (idle/moving/…)
    "fault": None,
    "frame_at": 0.0,        # time.time() of the last telemetry frame
    "detail": "starting…",
}
_stop = threading.Event()


def _update(**kw) -> None:
    with _snap_lock:
        _snap.update(kw)


def _snapshot() -> dict:
    with _snap_lock:
        return dict(_snap)


# ── Serial worker ──────────────────────────────────────────────────────────────

def _handle_line(raw: bytes) -> None:
    """Parse one NDJSON line; keep only telemetry battery/state fields.

    Protocol robustness rule (docs/motion_protocol.md §1): anything that isn't
    valid JSON with the expected fields is dropped silently.
    """
    try:
        msg = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return
    if not isinstance(msg, dict) or msg.get("type") != "telemetry":
        return
    _update(
        batt_mv=msg.get("batt_mv"),
        batt_ma=msg.get("batt_ma"),
        batt_soc=msg.get("batt_soc"),
        state=str(msg.get("state") or ""),
        fault=msg.get("fault"),
        frame_at=time.time(),
        mode="live",
    )


def _worker() -> None:
    """Own the serial port; poll the flock; feed the snapshot.

    Read-only by design — never writes to the port (see module docstring).
    """
    import serial

    ser = None
    was_dormant = False

    def _close():
        nonlocal ser
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            ser = None

    while not _stop.is_set():
        port = _motion_port()
        if not port:
            _close()
            _update(mode="no_port", port="",
                    detail="MOTION_ESP32_PORT not set in .env")
            _stop.wait(5.0)
            continue

        if _rex_running():
            # Dormant: release the port so main.py's motion link owns it.
            if ser is not None or not was_dormant:
                _close()
                log.info("Rex is running — port released (dormant).")
            was_dormant = True
            _update(mode="dormant", port=port,
                    detail="Rex is running — port handed to the robot")
            _stop.wait(_LOCK_POLL_SECS)
            continue
        was_dormant = False

        if ser is None:
            try:
                # exclusive (TIOCEXCL): if the dormant handoff ever races
                # main.py's motion connect, main.py sees a clean "resource
                # busy" (absorbed by its open retries) instead of two readers
                # silently splitting the byte stream.
                ser = serial.Serial(port, _motion_baud(), timeout=1.0, exclusive=True)
            except Exception as exc:
                _update(mode="connecting", port=port,
                        detail=f"waiting for board on {port}")
                log.debug("open %s failed: %s", port, exc)
                _stop.wait(2.0)
                continue
            log.info("Listening on %s (read-only).", port)
            # Opening the port usually auto-resets the ESP32 (DTR toggle) —
            # give it a moment to boot, then drop any partial line.
            _stop.wait(0.5)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            _update(mode="connecting", port=port, detail=f"listening on {port}")

        try:
            line = ser.readline()   # 1 s timeout → empty bytes
        except Exception as exc:
            log.info("Serial read failed (%s) — board unplugged? Reopening.", exc)
            _close()
            _update(mode="connecting", detail=f"waiting for board on {port}")
            _stop.wait(2.0)
            continue

        if line:
            _handle_line(line)
        elif time.time() - _snapshot()["frame_at"] > _STALE_SECS:
            # Port open but silent (board held in reset / wrong device):
            # cycle it rather than sit on a dead fd forever.
            log.info("No telemetry for %.0fs — cycling the port.", _STALE_SECS)
            _close()
            _update(mode="connecting", detail=f"no telemetry from {port}")
            _stop.wait(2.0)

        # Re-check the flock between reads. readline()'s 1 s timeout bounds
        # how long a quiet line can delay the dormant handoff.

    _close()


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fmt_title(s: dict) -> str:
    """Compact menu bar title: glyph + best available number."""
    if s["mode"] == "no_port":
        return "🔋 —"
    soc = s["batt_soc"]
    mv = s["batt_mv"]
    ma = s["batt_ma"]

    if soc is not None and soc >= 0:
        reading = f"{soc}%"
    elif mv is not None and mv >= 0:
        reading = f"{mv / 1000:.1f}V"
    else:
        reading = "—"

    if s["mode"] == "dormant":
        return f"🤖 {reading}"          # robot owns the port; reading may be old
    if s["mode"] != "live":
        return "🔋 …"
    if ma is not None and ma < _CHARGING_MA:
        return f"⚡ {reading}"
    if soc is not None and 0 <= soc <= _SOC_LOW_PCT:
        return f"🪫 {reading}"
    return f"🔋 {reading}"


def _fmt_lines(s: dict) -> list[str]:
    """Dropdown detail lines."""
    lines: list[str] = []

    if s["mode"] == "live":
        lines.append("Live — reading telemetry")
    elif s["mode"] == "dormant":
        lines.append("Rex is running — port handed over")
    elif s["mode"] == "no_port":
        lines.append("MOTION_ESP32_PORT not set in .env")
    else:
        lines.append(s["detail"] or "Connecting…")

    soc, mv, ma = s["batt_soc"], s["batt_mv"], s["batt_ma"]
    if soc is not None:
        lines.append(f"Charge: {soc}%" if soc >= 0 else "Charge: unknown (gauge not synced)")
    if mv is not None:
        lines.append(f"Voltage: {mv / 1000:.2f} V" if mv >= 0 else "Voltage: no INA226 wired")
    if ma is not None:
        amps = ma / 1000.0
        if ma < _CHARGING_MA:
            lines.append(f"Current: {abs(amps):.2f} A charging")
        else:
            lines.append(f"Current: {amps:.2f} A draw")
        if mv is not None and mv > 0:
            lines.append(f"Power: {abs(mv * ma) / 1_000_000:.1f} W")

    if s["state"]:
        fault = f" — FAULT: {s['fault']}" if s["fault"] else ""
        lines.append(f"Base: {s['state']}{fault}")

    if s["frame_at"]:
        age = time.time() - s["frame_at"]
        lines.append(f"Updated: {_fmt_age(age)}")

    return lines


def _fmt_age(age: float) -> str:
    if age < 2.0:
        return "just now"
    if age < 90.0:
        return f"{age:.0f}s ago"
    if age < 5400.0:
        return f"{age / 60:.0f}m ago"
    return f"{age / 3600:.1f}h ago"


# ── Menu bar app ───────────────────────────────────────────────────────────────

def run_app() -> int:
    try:
        import rumps
    except ImportError:
        log.error("rumps not installed in venv — run: venv/bin/pip install rumps")
        return 1

    # Fixed pool of menu rows updated in place each tick (rumps menus are
    # simplest when the item set is stable; unused rows are hidden).
    _MAX_ROWS = 8

    class RexBatteryApp(rumps.App):
        def __init__(self):
            super().__init__("R3X", title="🔋 …", quit_button="Quit Rex Battery Meter")
            self._rows = [rumps.MenuItem(f"row{i}") for i in range(_MAX_ROWS)]
            self.menu = list(self._rows)
            self._timer = rumps.Timer(self._refresh, 1.0)
            self._timer.start()

        def _refresh(self, _timer):
            s = _snapshot()
            self.title = _fmt_title(s)
            lines = _fmt_lines(s)
            for i, row in enumerate(self._rows):
                if i < len(lines):
                    row.title = lines[i]
                    row.hidden = False
                else:
                    row.hidden = True

    threading.Thread(target=_worker, daemon=True, name="rex-battery-serial").start()
    log.info("Battery menu bar app online (port=%s).", _motion_port() or "<unset>")
    try:
        RexBatteryApp().run()
    finally:
        _stop.set()
    return 0


# ── Probe mode (no GUI — bring-up check) ───────────────────────────────────────

def probe(seconds: float = 5.0) -> int:
    """Print live battery telemetry to stdout to confirm the wiring end-to-end."""
    port = _motion_port()
    if not port:
        print("MOTION_ESP32_PORT is not set in .env — nothing to probe.")
        return 1
    if _rex_running():
        print("main.py is running and owns the port — stop Rex first (or just")
        print("watch the menu bar app go live after Rex shuts down).")
        return 1
    try:
        import serial
        ser = serial.Serial(port, _motion_baud(), timeout=1.0, exclusive=True)
    except Exception as exc:
        print(f"Could not open {port}: {exc}")
        return 1
    print(f"Reading {port} for {seconds:.0f}s (read-only)…")
    time.sleep(0.5)
    ser.reset_input_buffer()
    end = time.monotonic() + seconds
    frames = 0
    try:
        while time.monotonic() < end:
            line = ser.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if msg.get("type") != "telemetry":
                continue
            frames += 1
            print(f"  batt_soc={msg.get('batt_soc')}%  batt_mv={msg.get('batt_mv')}mV  "
                  f"batt_ma={msg.get('batt_ma')}mA  state={msg.get('state')}  "
                  f"fault={msg.get('fault')}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    if frames == 0:
        print("⚠  No telemetry frames — is the motion firmware flashed and the")
        print("   board on this port? Try: firmware/tools/motion_serial_smoketest.py")
        return 1
    print(f"✓  {frames} telemetry frames received.")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("--probe", "-p", "probe"):
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        sys.exit(probe(secs))
    elif arg in ("--help", "-h", "help"):
        print("Usage: rex_battery_menubar.py [--probe [secs]]\n"
              "  (no args)      run the menu bar battery meter\n"
              "  --probe [secs] print raw battery telemetry to stdout and exit")
        sys.exit(0)
    else:
        sys.exit(run_app())
