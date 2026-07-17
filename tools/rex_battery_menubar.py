#!/usr/bin/env python3
"""
rex_battery_menubar.py — macOS menu bar battery meter for the ESP32 motion base.

A tiny always-on menu bar app (rumps/Cocoa) that shows the drive base's pack
state of charge, voltage, and current even when the robot (main.py) is OFF.
Installed as a LaunchAgent alongside the wake-word supervisor by
scripts/install_supervisor.sh.

How it gets the data with no protocol handshake:
  The motion firmware streams a telemetry frame at 10 Hz from the moment it
  boots — unconditionally, before any handshake (telemetryTask in
  firmware/djr3x_motion/djr3x_motion.ino). Every frame carries batt_mv /
  batt_ma / batt_soc. So this app just opens MOTION_ESP32_PORT and parses
  NDJSON. It is PASSIVE with exactly one exception: the "Set Battery to 100%"
  menu item sends a single `batt_full` command (docs/motion_protocol.md §5.11)
  to sync the coulomb gauge when the operator watches the charger's taper
  current hit cutoff — evidence of "full" the firmware can't see on its own
  (mid-absorption the pack is never at rest, so the boot rest-voltage anchor
  can't fire until a power-cycle). It never sends motion commands. Side effect
  of any write: the firmware's comms watchdog arms (seen_mac) and re-latches
  comms_lost once we go quiet — harmless while the base is idle, and the
  dropdown already presents that state as benign standby.

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
import math
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
# Runtime estimate: remaining_mah / smoothed_current. Capacity mirrors the
# firmware ledger (calib.h BATT_CAPACITY_MAH); keep the two in sync. The current
# is EMA-smoothed because a droid's draw is spiky (idle ~1 A, drive bursts far
# higher) — an instantaneous divide would make the estimate jump wildly.
_BATT_CAPACITY_MAH = 40000  # 2x 12.8 V 20 Ah in parallel (== calib.h)
_RUNTIME_TAU_SECS = 45.0    # EMA time constant for the estimate's current
_RUNTIME_MIN_MA = 80        # |smoothed current| under this → no estimate (idle on
                            # a charger / disconnected pack → the number is nonsense)
# Supply-set-too-high watch: pack terminals at/above 14.55 V while STILL being
# pushed hard means the bench supply is set above 14.6 V (at a 14.6 setpoint the
# current is near zero by the time the pack gets here — the IR drop has died).
# 14.6 V = 3.65 V/cell is the 4S LiFePO4 ceiling; time to dial the supply back.
_SUPPLY_HIGH_MV = 14550
_SUPPLY_HIGH_MA = -800
_SUPPLY_HIGH_RENOTIFY_SECS = 300.0
_STALE_SECS = 10.0         # no telemetry for this long after open → reopen port
                           # (> the board's ~6 s boot: ToF init + IMU bias cal)
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
    "batt_ma_avg": None,    # float mA, EMA of batt_ma for the runtime estimate
    "batt_ma_avg_at": 0.0,  # time.time() the EMA last advanced (gap detection)
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


# ── Outbound commands (UI thread → serial worker) ─────────────────────────────
# The meter is passive except for `batt_full`; the worker drains this queue
# between telemetry reads, so a click reaches the wire within ~1 s (bounded by
# readline's timeout).

_tx_lock = threading.Lock()
_tx_queue: list[bytes] = []
_tx_seq = 0


def _queue_batt_full() -> None:
    global _tx_seq
    with _tx_lock:
        _tx_seq += 1
        payload = json.dumps({"v": 1, "cmd": "batt_full", "seq": _tx_seq}) + "\n"
        _tx_queue.append(payload.encode("utf-8"))


def _drain_tx(ser) -> None:
    """Send any queued commands. Serial write errors are left to the read path
    to detect (it owns reopen); a failed command is dropped, not retried —
    the user just clicks again."""
    with _tx_lock:
        pending = _tx_queue[:]
        _tx_queue.clear()
    for payload in pending:
        try:
            ser.write(payload)
            log.info("Sent %s", payload.decode().strip())
        except Exception as exc:
            log.warning("Command write failed (%s) — dropped.", exc)


def _clear_tx() -> None:
    with _tx_lock:
        _tx_queue.clear()


# "Restart ESP32" menu click → the worker (which owns the open handle) pulses the
# DTR line. With the default open both DTR and RTS sit asserted (EN high — the
# transistor pair cancels); dropping DTR alone leaves RTS-only asserted, which is
# the EN-low reset state. Re-asserting DTR releases EN into a normal boot (IO0
# stays high at the release edge). Not a serial WRITE, so it rides its own flag.
_reset_flag = threading.Event()


def _queue_esp32_reset() -> None:
    _reset_flag.set()


def _service_reset(ser) -> None:
    if not _reset_flag.is_set():
        return
    _reset_flag.clear()
    try:
        ser.dtr = False       # RTS stays asserted → EN low, chip held in reset
        time.sleep(0.15)
        ser.dtr = True        # both asserted again → EN high, normal boot
        try:
            ser.reset_input_buffer()   # drop any partial pre-reset line
        except Exception:
            pass
        log.info("ESP32 reset pulse sent (DTR toggle) — board rebooting.")
        _notify("Rex Battery", "ESP32 restarted.")
    except Exception as exc:
        log.warning("ESP32 reset failed: %s", exc)


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
    if not isinstance(msg, dict):
        return
    if msg.get("type") == "log":
        # Surface firmware decisions (SOC anchors/clamps, sensor bring-up) with
        # host timestamps — evidence for validating the gauge against a bench
        # supply: `tail -f logs/battery_menubar.err.log` while charging.
        log.info("fw %s: %s", msg.get("level", "log"), msg.get("msg"))
        return
    if msg.get("type") != "telemetry":
        return
    _advance_current_ema(msg.get("batt_ma"))
    _update(
        batt_mv=msg.get("batt_mv"),
        batt_ma=msg.get("batt_ma"),
        batt_soc=msg.get("batt_soc"),
        state=str(msg.get("state") or ""),
        fault=msg.get("fault"),
        frame_at=time.time(),
        mode="live",
    )


def _advance_current_ema(ma) -> None:
    """Fold one batt_ma sample into the smoothed current used for the runtime
    estimate. Time-aware so telemetry gaps (dormant handoff, reconnect) don't
    blend across the hole — a gap longer than a few time constants resets the
    average to the fresh sample instead of averaging stale current into it."""
    if ma is None:
        return
    now = time.time()
    with _snap_lock:
        prev = _snap.get("batt_ma_avg")
        prev_t = _snap.get("batt_ma_avg_at") or 0.0
        dt = now - prev_t
        if prev is None or dt <= 0.0 or dt > 5.0 * _RUNTIME_TAU_SECS:
            avg = float(ma)
        else:
            alpha = 1.0 - math.exp(-dt / _RUNTIME_TAU_SECS)
            avg = prev + alpha * (float(ma) - prev)
        _snap["batt_ma_avg"] = avg
        _snap["batt_ma_avg_at"] = now


def _worker() -> None:
    """Own the serial port; poll the flock; feed the snapshot.

    Read-only by design — never writes to the port (see module docstring).
    """
    import serial

    ser = None
    was_dormant = False
    opened_at = 0.0      # when THIS serial connection was opened (staleness base)
    last_summary = 0.0   # once-a-minute battery line in the log (taper record)

    def _close():
        nonlocal ser
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            ser = None
        # A queued click must not fire surprisingly late on a future reconnect.
        _clear_tx()
        _reset_flag.clear()

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
                # DEFAULT open on purpose — do NOT pre-drop DTR/RTS. On macOS
                # the kernel asserts BOTH lines during open (both-asserted is
                # benign: the ESP32's auto-reset transistor pair cancels), and
                # pyserial then applies pre-set values DTR-FIRST — passing
                # through DTR-low+RTS-high, the EN-low RESET state. The Linux
                # "no-reset" pre-drop trick therefore CAUSES a reboot here
                # (measured 2026-07-13: pre-drop open → boot at +1.6 s; default
                # open and close → no reset). Leave the lines asserted for the
                # whole session; _service_reset() pulses DTR to command a reset.
                ser = serial.Serial(port, _motion_baud(), timeout=1.0, exclusive=True)
            except Exception as exc:
                _update(mode="connecting", port=port,
                        detail=f"waiting for board on {port}")
                log.debug("open %s failed: %s", port, exc)
                _stop.wait(2.0)
                continue
            opened_at = time.time()
            log.info("Listening on %s (passive; writes only on Set-Battery-100%%).", port)
            # Brief settle, then drop any partial line mid-stream.
            _stop.wait(0.5)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            _update(mode="connecting", port=port, detail=f"listening on {port}")

        _drain_tx(ser)
        _service_reset(ser)

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
            now = time.time()
            if now - last_summary >= 60.0:
                s = _snapshot()
                if s["mode"] == "live" and s["batt_ma"] is not None:
                    last_summary = now
                    log.info("[batt] soc=%s%% mv=%s ma=%s state=%s",
                             s["batt_soc"], s["batt_mv"], s["batt_ma"], s["state"])
        elif time.time() - max(_snapshot()["frame_at"], opened_at) > _STALE_SECS:
            # Port open but silent (board held in reset / wrong device):
            # cycle it rather than sit on a dead fd forever. Staleness is
            # measured from THIS connection's open, not just the last frame —
            # a global-only clock carried pre-reconnect staleness across every
            # reopen and cycled a booting board (~6 s of ToF init + IMU cal)
            # forever (field bug 2026-07-13: 141-reboot storm in one evening).
            log.info("No telemetry for %.0fs — cycling the port.", _STALE_SECS)
            _close()
            _update(mode="connecting", detail=f"no telemetry from {port}")
            _stop.wait(2.0)

        # Re-check the flock between reads. readline()'s 1 s timeout bounds
        # how long a quiet line can delay the dormant handoff.

    _close()


def _supply_too_high(s: dict) -> bool:
    """True while charging hard with pack terminals at/above the 4S ceiling."""
    mv, ma = s.get("batt_mv"), s.get("batt_ma")
    return (s.get("mode") == "live" and mv is not None and ma is not None
            and mv >= _SUPPLY_HIGH_MV and ma <= _SUPPLY_HIGH_MA)


def _notify(title: str, message: str) -> None:
    """macOS notification via osascript (works from a plain LaunchAgent;
    rumps' own notification API needs an app bundle)."""
    import subprocess
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Basso"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        log.debug("notification failed: %s", exc)


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fmt_title(s: dict) -> str:
    """Full live readout in the menu bar (owner request 2026-07-16): percentage,
    voltage, current, watts, and the runtime estimate as a clock (39:30 left) —
    no click needed. The 1 Hz refresh timer keeps every field live."""
    if s["mode"] == "no_port":
        return "🔋 —"
    soc = s["batt_soc"]
    mv = s["batt_mv"]
    ma = s["batt_ma"]

    if s["mode"] == "dormant":
        # Robot owns the port; the reading may be stale — keep it short.
        reading = f"{soc}%" if (soc is not None and soc >= 0) else "—"
        return f"🤖 {reading}"
    if s["mode"] != "live":
        return "🔋 …"

    # Glyph: charging / low / over-voltage warning / normal.
    if _supply_too_high(s):
        glyph = "⚠️"
    elif ma is not None and ma < _CHARGING_MA:
        glyph = "⚡"
    elif soc is not None and 0 <= soc <= _SOC_LOW_PCT:
        glyph = "🪫"
    else:
        glyph = "🔋"

    parts: list[str] = [glyph]
    if soc is not None and soc >= 0:
        parts.append(f"{soc}%")
    if mv is not None and mv >= 0:
        parts.append(f"{mv / 1000:.2f}V")
    if ma is not None:
        if abs(ma) < abs(_CHARGING_MA):
            parts.append("~0A")
        else:
            parts.append(f"{abs(ma) / 1000:.2f}A")
            if mv is not None and mv > 0:
                parts.append(f"{abs(mv * ma) / 1_000_000:.1f}W")

    clock = _fmt_title_clock(s)
    if clock:
        parts.append(clock)
    return " ".join(parts)


def _fmt_title_clock(s: dict) -> "str | None":
    """H:MM runtime estimate for the title bar: "39:30 left" discharging,
    "2:15 to full" charging. Same coulomb math + guards as _fmt_time_left."""
    soc = s.get("batt_soc")
    avg = s.get("batt_ma_avg")
    if soc is None or soc < 0 or avg is None or abs(avg) < _RUNTIME_MIN_MA:
        return None
    if avg > 0:                                   # discharging → time to empty
        hours = (soc / 100.0) * _BATT_CAPACITY_MAH / avg
        return f"{_fmt_clock(hours)} left"
    if soc >= 99:                                 # charging but essentially full
        return None
    hours = ((100 - soc) / 100.0) * _BATT_CAPACITY_MAH / abs(avg)
    return f"{_fmt_clock(hours)} to full"


def _fmt_clock(hours: float) -> str:
    if hours >= 100.0:
        return ">99:59"
    h = int(hours)
    m = int(round((hours - h) * 60.0))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}:{m:02d}"


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
        if abs(ma) < abs(_CHARGING_MA):
            # Near-zero battery current: full pack on a charger (bus carries the
            # load), or a genuinely disconnected pack — don't show "-0.03 A draw".
            lines.append(f"Current: ~0 A ({amps * 1000:+.0f} mA)")
        elif ma < 0:
            lines.append(f"Current: {abs(amps):.2f} A charging")
        else:
            lines.append(f"Current: {amps:.2f} A draw")
        if mv is not None and mv > 0 and abs(ma) >= abs(_CHARGING_MA):
            lines.append(f"Power: {abs(mv * ma) / 1_000_000:.1f} W")

    # Estimated runtime — only while live (a dormant/stale reading would give a
    # confidently wrong number). Guards itself when it can't be estimated.
    if s["mode"] == "live":
        tleft = _fmt_time_left(s)
        if tleft:
            lines.append(tleft)

    if _supply_too_high(s):
        lines.append("⚠ Pack over 14.55 V under charge — set supply back to 14.6 V")

    if s["state"]:
        # comms_lost is the NORMAL resting state while Rex is off: the firmware
        # watchdog latches it when main.py's heartbeats stop, and only a new
        # Mac→ESP32 line clears it — which this passive listener never sends.
        # Present it as standby, not as a fault.
        if s["state"] == "comms_lost" or s["fault"] == "comms_lost":
            lines.append("Base: standby (normal while Rex is off)")
        else:
            fault = f" — FAULT: {s['fault']}" if s["fault"] else ""
            lines.append(f"Base: {s['state']}{fault}")

    if s["frame_at"]:
        age = time.time() - s["frame_at"]
        lines.append(f"Updated: {_fmt_age(age)}")

    return lines


def _fmt_time_left(s: dict) -> "str | None":
    """Estimated time-to-empty (discharging) or time-to-full (charging), from
    the SOC ledger and the smoothed current. Coulomb-based, so LiFePO4's flat
    voltage curve doesn't distort it. None when it can't be estimated: no synced
    gauge, near-zero current (idle on a charger), or already full."""
    soc = s.get("batt_soc")
    avg = s.get("batt_ma_avg")
    if soc is None or soc < 0 or avg is None or abs(avg) < _RUNTIME_MIN_MA:
        return None
    if avg > 0:                                   # discharging → time to empty
        hours = (soc / 100.0) * _BATT_CAPACITY_MAH / avg
        return f"Est. runtime: ~{_fmt_duration(hours)} left"
    if soc >= 99:                                 # charging but essentially full
        return None
    hours = ((100 - soc) / 100.0) * _BATT_CAPACITY_MAH / abs(avg)
    return f"Est. charge: ~{_fmt_duration(hours)} to full"


def _fmt_duration(hours: float) -> str:
    if hours >= 99.0:
        return ">99h"
    h = int(hours)
    m = int(round((hours - h) * 60.0))
    if m == 60:
        h, m = h + 1, 0
    return f"{m}m" if h == 0 else f"{h}h {m}m"


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
            # rumps renders items WITHOUT a callback as disabled (grey text);
            # a no-op callback keeps these info rows in normal colour.
            self._rows = [rumps.MenuItem(f"row{i}", callback=lambda _: None)
                          for i in range(_MAX_ROWS)]
            # "Charger taper hit cutoff" → sync the firmware's coulomb gauge.
            # Only shown while we own the port (live); hidden when dormant/off.
            self._mark_full = rumps.MenuItem("Set Battery to 100%",
                                             callback=self._on_mark_full)
            # Hardware reboot of the motion base (RTS pulse via the open port) —
            # for un-wedging the board after a bad flash / wedged state without
            # crawling to the USB plug. Live-mode only, like Set-Battery-100%.
            self._restart_esp = rumps.MenuItem("Restart ESP32",
                                               callback=self._on_restart_esp)
            self._hv_notified_at = 0.0   # last supply-too-high notification
            self.menu = list(self._rows) + [None, self._mark_full, self._restart_esp]
            self._timer = rumps.Timer(self._refresh, 1.0)
            self._timer.start()
            # rumps schedules its NSTimer in the DEFAULT run-loop mode only,
            # but while the dropdown is open macOS switches to EVENT-TRACKING
            # mode — so the readouts froze exactly while being looked at.
            # Registering the same timer in the tracking mode keeps the rows
            # live-updating with the menu open. Guarded: if a future rumps
            # renames its private _nstimer, we just fall back to frozen-while-
            # open instead of crashing the meter.
            try:
                from AppKit import NSEventTrackingRunLoopMode
                from Foundation import NSRunLoop
                NSRunLoop.currentRunLoop().addTimer_forMode_(
                    self._timer._nstimer, NSEventTrackingRunLoopMode)
            except Exception as exc:
                log.warning("Could not enable open-menu live updates: %s", exc)

        def _on_mark_full(self, _item):
            if _snapshot()["mode"] != "live":
                return
            log.info("User clicked Set Battery to 100% — queueing batt_full.")
            _queue_batt_full()

        def _on_restart_esp(self, _item):
            if _snapshot()["mode"] != "live":
                return
            log.info("User clicked Restart ESP32 — queueing reset pulse.")
            _queue_esp32_reset()

        def _refresh(self, _timer):
            s = _snapshot()
            self.title = _fmt_title(s)
            self._mark_full.hidden = (s["mode"] != "live")
            self._restart_esp.hidden = (s["mode"] != "live")
            if _supply_too_high(s):
                now = time.time()
                if now - self._hv_notified_at >= _SUPPLY_HIGH_RENOTIFY_SECS:
                    self._hv_notified_at = now
                    log.warning("Pack %.2f V at %.1f A charge — supply set above 14.6 V.",
                                s["batt_mv"] / 1000, abs(s["batt_ma"]) / 1000)
                    _notify("Rex Battery",
                            f"Pack at {s['batt_mv'] / 1000:.2f} V and still charging — "
                            "dial the bench supply back to 14.6 V.")
            else:
                self._hv_notified_at = 0.0
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
    print(f"Reading {port} for {seconds:.0f}s (passive)…")
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


def mark_full_cli() -> int:
    """Send batt_full from the terminal (what the menu item does) and confirm
    the ack + the SOC actually reaching 100% in telemetry."""
    port = _motion_port()
    if not port:
        print("MOTION_ESP32_PORT is not set in .env.")
        return 1
    if _rex_running():
        print("main.py is running and owns the port — stop Rex first.")
        return 1
    try:
        import serial
        ser = serial.Serial(port, _motion_baud(), timeout=1.0, exclusive=True)
    except Exception as exc:
        print(f"Could not open {port}: {exc}")
        print("(If the menu bar meter is running, it holds the port — quit it "
              "or use its 'Set Battery to 100%' item instead.)")
        return 1
    cmd = json.dumps({"v": 1, "cmd": "batt_full", "seq": 1}) + "\n"
    print(f"Sending on {port}: {cmd.strip()}")
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.write(cmd.encode())
    acked = soc100 = False
    end = time.monotonic() + 6.0
    try:
        while time.monotonic() < end and not (acked and soc100):
            line = ser.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if msg.get("type") == "ack" and msg.get("seq") == 1:
                acked = True
                print(f"  ack: accepted={msg.get('accepted')} reason={msg.get('reason')}")
                if not msg.get("accepted"):
                    break
            elif msg.get("type") == "log":
                print(f"  fw log: {msg.get('msg')}")
            elif msg.get("type") == "telemetry" and msg.get("batt_soc") == 100:
                soc100 = True
                print(f"  telemetry: batt_soc=100%  batt_mv={msg.get('batt_mv')}mV")
    finally:
        ser.close()
    if acked and soc100:
        print("✓  SOC gauge synced to 100%.")
        return 0
    print(f"⚠  Incomplete: ack={acked}, soc@100%={soc100}. Old firmware without "
          "batt_full support? Reflash firmware/djr3x_motion.")
    return 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("--probe", "-p", "probe"):
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        sys.exit(probe(secs))
    elif arg in ("--mark-full", "mark-full"):
        sys.exit(mark_full_cli())
    elif arg in ("--help", "-h", "help"):
        print("Usage: rex_battery_menubar.py [--probe [secs] | --mark-full]\n"
              "  (no args)      run the menu bar battery meter\n"
              "  --probe [secs] print raw battery telemetry to stdout and exit\n"
              "  --mark-full    send batt_full (sync SOC gauge to 100%) and exit")
        sys.exit(0)
    else:
        sys.exit(run_app())
