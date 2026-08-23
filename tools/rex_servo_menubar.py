#!/usr/bin/env python3
"""
rex_servo_menubar.py — macOS menu bar servo console for the Pololu Maestro.

A small always-on menu bar app (rumps/Cocoa, sibling of rex_battery_menubar.py)
titled "Servo Control". The dropdown shows all 8 of Rex's servos as live sliders
(labelled with the current position in Maestro microseconds) plus a "Restart
Pololu" action. Slide a slider and the servo moves — the same Pololu compact
protocol `set target` commands hardware/servos.py sends for the main GUI's
manual sliders, just spoken directly on the wire so this app has no dependency
on the project config (which refuses to import without API keys).

How it shares the serial port with main.py (ports are exclusive-open):
  Same dormant pattern as the battery meter. main.py holds the single-instance
  flock for its whole lifetime; this app polls it ~1×/s:
    - lock held  → close the port (main.py owns the servos), sliders inert,
                   status row shows "Rex is running"
    - lock free  → reopen the port and the sliders go live
  On (re)connect it reads each channel's actual position (0x90 GET POSITION)
  and snaps the sliders to reality; channels reporting 0 (servo off) show their
  startup/neutral position instead.

"Restart Pololu" sends the Maestro GO HOME command (0xA2): every channel
returns to its configured home/startup position — the recover-a-weird-pose
button. (The Maestro has no soft-reboot over serial; go-home + a fresh
position read is the meaningful restart.)

Servo definitions mirror config.SERVO_CHANNELS (channels, min/max/neutral in
quarter-microseconds) and honor the same .env overrides
(SERVO_<NAME>_MIN_US / _MAX_US / _NEUTRAL_US, Maestro Control Center
microseconds) — keep _SERVO_DEFAULTS in sync if the robot gains a servo.

Run directly for debugging:
    venv/bin/python tools/rex_servo_menubar.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# utils.single_instance must be importable WITHOUT the heavy project config
# (mirrors the battery meter — this process must start even when apikeys.py
# would fail).
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | rex_servo | %(levelname)s | %(message)s",
)
log = logging.getLogger("rex_servo")

_LOCK_POLL_SECS = 1.0
_MAESTRO_BAUD = 9600          # config.SERVO_BAUD
_CMD_SET_TARGET = 0x84        # Pololu compact protocol (hardware/servos.py)
_CMD_GET_POSITION = 0x90
_CMD_GO_HOME = 0xA2

# Mirror of config.SERVO_CHANNELS (q-µs). Keep in sync when the robot changes.
_SERVO_DEFAULTS: dict[str, dict[str, int]] = {
    "neck":     {"ch": 0, "min": 1984, "max": 8960, "neutral": 5472},
    "headlift": {"ch": 1, "min": 2600, "max": 7744, "neutral": 6000},  # min 650 us
    "headtilt": {"ch": 2, "min": 3904, "max": 5504, "neutral": 4320},
    "visor":    {"ch": 3, "min": 4544, "max": 6976, "neutral": 6560},  # 1640 µs — 6000 hid part of the camera
    "elbow":    {"ch": 4, "min": 6300, "max": 7424, "neutral": 6720},  # 7424 = the Maestro's own stored channel limit (1856 us)
    "hand":     {"ch": 5, "min": 1984, "max": 9984, "neutral": 6000},
    "pokerarm": {"ch": 6, "min": 3968, "max": 8000, "neutral": 6000},
    "heroarm":  {"ch": 7, "min": 3968, "max": 8000, "neutral": 6000},
}


# ── Minimal .env reading (no project config import) ────────────────────────────

def _read_env_file() -> dict[str, str]:
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


def _maestro_port() -> str:
    env = _read_env_file()
    return (os.environ.get("MAESTRO_PORT") or env.get("MAESTRO_PORT") or "").strip()


def _servos() -> dict[str, dict[str, int]]:
    """Servo table with the same .env µs overrides config.py applies (µs × 4 = q-µs)."""
    env = _read_env_file()
    table: dict[str, dict[str, int]] = {}
    for name, cfg in _SERVO_DEFAULTS.items():
        entry = dict(cfg)
        prefix = f"SERVO_{name.upper()}"
        for env_suffix, key in (("_MIN_US", "min"), ("_MAX_US", "max"), ("_NEUTRAL_US", "neutral")):
            raw = (env.get(prefix + env_suffix) or "").strip()
            if raw:
                try:
                    entry[key] = int(round(float(raw) * 4))
                except ValueError:
                    log.warning("Ignoring non-numeric %s%s=%r", prefix, env_suffix, raw)
        if entry["min"] > entry["max"]:
            entry["min"], entry["max"] = entry["max"], entry["min"]
        entry["neutral"] = max(entry["min"], min(entry["max"], entry["neutral"]))
        table[name] = entry
    return table


def _rex_running() -> bool:
    try:
        from utils import single_instance
        return single_instance.is_held_by_other()
    except Exception as exc:
        log.debug("single_instance check failed: %s", exc)
        return False


# ── Shared state (worker thread ↔ UI timer) ────────────────────────────────────

_snap_lock = threading.Lock()
_snap: dict = {
    "mode": "connecting",       # connecting | live | dormant | no_port
    "detail": "starting…",
    # channel → q-µs read from the board at (re)connect / after go-home; the UI
    # timer applies these to the sliders once, then clears the entry.
    "pending_positions": {},
}
_stop = threading.Event()


def _update(**kw) -> None:
    with _snap_lock:
        _snap.update(kw)


def _snapshot() -> dict:
    with _snap_lock:
        snap = dict(_snap)
        snap["pending_positions"] = dict(_snap["pending_positions"])
        return snap


def _take_pending_positions() -> dict[int, int]:
    with _snap_lock:
        pending = dict(_snap["pending_positions"])
        _snap["pending_positions"].clear()
    return pending


# ── Outbound commands (UI thread → serial worker) ─────────────────────────────
# Slider drags fire rapidly; per-channel targets COALESCE (only the latest value
# per channel is sent each worker pass), so dragging never floods the wire.

_tx_lock = threading.Lock()
_targets: dict[int, int] = {}
_go_home = threading.Event()


def _queue_target(channel: int, qus: int) -> None:
    with _tx_lock:
        _targets[channel] = int(qus)


def _queue_go_home() -> None:
    with _tx_lock:
        _targets.clear()      # a queued drag must not immediately undo the home
    _go_home.set()


def _encode_set_target(channel: int, qus: int) -> bytes:
    return bytes([_CMD_SET_TARGET, channel, qus & 0x7F, (qus >> 7) & 0x7F])


# ── Serial worker ──────────────────────────────────────────────────────────────

def _read_positions(ser, channels: list[int]) -> dict[int, int]:
    """GET POSITION per channel (2-byte little-endian q-µs reply); {} on failure."""
    positions: dict[int, int] = {}
    for ch in channels:
        try:
            ser.reset_input_buffer()
            ser.write(bytes([_CMD_GET_POSITION, ch]))
            raw = ser.read(2)
            if len(raw) == 2:
                positions[ch] = raw[0] | (raw[1] << 8)
        except Exception as exc:
            log.debug("get_position ch%d failed: %s", ch, exc)
            return {}
    return positions


def _worker() -> None:
    import serial

    ser = None
    channels = [cfg["ch"] for cfg in _servos().values()]

    def _close():
        nonlocal ser
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            ser = None
        with _tx_lock:
            _targets.clear()
        _go_home.clear()

    while not _stop.is_set():
        port = _maestro_port()
        if not port:
            _close()
            _update(mode="no_port", detail="MAESTRO_PORT not set in .env")
            _stop.wait(5.0)
            continue

        if _rex_running():
            if ser is not None:
                _close()
                log.info("Rex is running — Maestro port released (dormant).")
            _update(mode="dormant", detail="Rex is running — servos owned by the robot")
            _stop.wait(_LOCK_POLL_SECS)
            continue

        if ser is None:
            try:
                ser = serial.Serial(port, _MAESTRO_BAUD, timeout=0.2, exclusive=True)
            except Exception as exc:
                _update(mode="connecting", detail=f"waiting for Maestro on {port}")
                log.debug("open %s failed: %s", port, exc)
                _stop.wait(2.0)
                continue
            log.info("Maestro connected on %s.", port)
            positions = _read_positions(ser, channels)
            _update(mode="live", detail=f"live on {port}",
                    pending_positions=positions)

        # Go-home outranks queued targets (it also cleared them at queue time).
        if _go_home.is_set():
            _go_home.clear()
            try:
                ser.write(bytes([_CMD_GO_HOME]))
                log.info("Sent GO HOME — all channels to their home positions.")
                time.sleep(0.6)                     # let the servos travel
                positions = _read_positions(ser, channels)
                with _snap_lock:
                    _snap["pending_positions"].update(positions)
            except Exception as exc:
                log.warning("GO HOME failed (%s) — reopening.", exc)
                _close()
                continue

        with _tx_lock:
            pending = dict(_targets)
            _targets.clear()
        for ch, qus in pending.items():
            try:
                ser.write(_encode_set_target(ch, qus))
            except Exception as exc:
                log.info("set_target write failed (%s) — reopening.", exc)
                _close()
                break

        _stop.wait(0.05 if pending else 0.25)

    _close()


# ── Menu bar app ───────────────────────────────────────────────────────────────

def run_app() -> int:
    try:
        import rumps
    except ImportError:
        log.error("rumps not installed in venv — run: venv/bin/pip install rumps")
        return 1

    servos = _servos()

    class RexServoApp(rumps.App):
        def __init__(self):
            super().__init__("R3XServo", title="🦾 Servo Control",
                             quit_button="Quit Servo Control")
            self._status = rumps.MenuItem("status", callback=lambda _: None)
            self._labels: dict[str, rumps.MenuItem] = {}
            self._sliders: dict[str, object] = {}
            self._by_channel: dict[int, str] = {}
            menu: list = [self._status, None]
            for name, cfg in servos.items():
                self._by_channel[cfg["ch"]] = name
                label = rumps.MenuItem(f"{name}", callback=lambda _: None)
                slider = rumps.SliderMenuItem(
                    value=cfg["neutral"], min_value=cfg["min"], max_value=cfg["max"],
                    callback=self._make_slider_cb(name), dimensions=(200, 20),
                )
                self._labels[name] = label
                self._sliders[name] = slider
                menu += [label, slider]
                self._set_label(name, cfg["neutral"])
            self._restart = rumps.MenuItem("Restart Pololu (all home)",
                                           callback=self._on_restart)
            menu += [None, self._restart]
            self.menu = menu
            self._timer = rumps.Timer(self._refresh, 1.0)
            self._timer.start()
            # Keep refreshing while the dropdown is open (see the battery meter
            # for the run-loop-mode story); fall back gracefully if rumps changes.
            try:
                from AppKit import NSEventTrackingRunLoopMode
                from Foundation import NSRunLoop
                NSRunLoop.currentRunLoop().addTimer_forMode_(
                    self._timer._nstimer, NSEventTrackingRunLoopMode)
            except Exception as exc:
                log.warning("Could not enable open-menu live updates: %s", exc)

        def _set_label(self, name: str, qus: float) -> None:
            self._labels[name].title = f"{name}:  {qus / 4.0:.0f} µs"

        def _make_slider_cb(self, name: str):
            cfg = servos[name]

            def _cb(sender):
                qus = int(round(sender.value))
                self._set_label(name, qus)
                if _snapshot()["mode"] != "live":
                    return                      # inert while Rex owns the servos
                _queue_target(cfg["ch"], qus)
            return _cb

        def _on_restart(self, _item):
            if _snapshot()["mode"] != "live":
                return
            log.info("User clicked Restart Pololu — queueing GO HOME.")
            _queue_go_home()

        def _refresh(self, _timer):
            s = _snapshot()
            mode_line = {
                "live": s["detail"],
                "dormant": "⏸  Rex is running — sliders inert",
                "connecting": s["detail"],
                "no_port": s["detail"],
            }.get(s["mode"], s["detail"])
            self._status.title = mode_line
            self._restart.hidden = (s["mode"] != "live")
            for ch, qus in _take_pending_positions().items():
                name = self._by_channel.get(ch)
                if name is None:
                    continue
                cfg = servos[name]
                # Position 0 = servo off (no target yet) → show startup/neutral.
                shown = qus if qus > 0 else cfg["neutral"]
                shown = max(cfg["min"], min(cfg["max"], shown))
                try:
                    self._sliders[name].value = shown
                except Exception:
                    pass
                self._set_label(name, shown)

    threading.Thread(target=_worker, daemon=True, name="rex-servo-serial").start()
    log.info("Servo Control menu bar app online (port=%s).", _maestro_port() or "<unset>")
    try:
        RexServoApp().run()
    finally:
        _stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(run_app())
