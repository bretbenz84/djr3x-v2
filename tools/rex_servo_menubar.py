#!/usr/bin/env python3
"""
rex_servo_menubar.py — macOS menu bar servo console for the Pololu Maestro.

A small always-on menu bar app (rumps/Cocoa, sibling of rex_battery_menubar.py)
titled "Servo Control". The dropdown shows all 11 of Rex's servos as live sliders
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
  On (re)connect it reads each channel's commanded pulse position (0x90 GET POSITION)
  and updates the sliders; channels reporting 0 (servo off) show their
  neutral slider placeholder with an explicit off label instead.

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

import csv
from datetime import datetime, timezone
import logging
import queue
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

from tools.throttle_sequences import Recorder, load_sequence, require_start_pose, validate_pose
from hardware.throttle_motion import invalidate_park

_LOCK_POLL_SECS = 1.0
_MAESTRO_BAUD = 9600          # config.SERVO_BAUD
_CMD_SET_TARGET = 0x84        # Pololu compact protocol (hardware/servos.py)
_CMD_SET_SPEED = 0x87
_CMD_SET_ACCEL = 0x89
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
    # Neutral is only an unsent slider placeholder, never a startup command.
    "throttle_shoulder": {"ch": 8, "min": 2140, "max": 9120, "neutral": 6000,
                          "speed": 30, "acceleration": 6},
    "throttle_elbow": {"ch": 9, "min": 2000, "max": 10000, "neutral": 6000,
                       "speed": 70, "acceleration": 12},
    "throttle_wrist": {"ch": 10, "min": 2000, "max": 10000, "neutral": 6000,
                       "speed": 70, "acceleration": 12},
}
_THROTTLE_DIRECTIONS = {
    "throttle_shoulder": "low ↑ / high ↓",
    "throttle_elbow": "low ↓ / high ↑",
    "throttle_wrist": "low ↑ / high ↓",
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
        if name in _THROTTLE_DIRECTIONS:
            lo_raw = (env.get(prefix + "_MIN_US") or "").strip()
            hi_raw = (env.get(prefix + "_MAX_US") or "").strip()
            if bool(lo_raw) != bool(hi_raw):
                raise ValueError(f"{prefix} limits require both MIN_US and MAX_US")
            if lo_raw:
                lo, hi = (int(round(float(raw) * 4)) for raw in (lo_raw, hi_raw))
                if not cfg["min"] <= lo < hi <= cfg["max"]:
                    raise ValueError(f"{prefix} limits must stay within measured travel")
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
_measurement_mode = threading.Event()
_measurement_requests = queue.Queue()
_MEASUREMENT_FILE = _PROJECT_ROOT / "data" / "throttle_measurements.csv"
_SEQUENCE_DIR = _PROJECT_ROOT / "data" / "throttle_sequences"
_sequence_requests = queue.Queue()
_recorder = None  # Serial worker owns these objects.
_playback = None
_playback_stop = threading.Event()
_MEASUREMENT_PROFILES = {8: (10, 2), 9: (20, 3), 10: (20, 3)}



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
        if _snapshot().get("sequence_mode") == "playing":
            _update(sequence_status="Stop playback before using manual controls")
            return
        _targets[channel] = int(qus)


def _queue_go_home() -> None:
    with _tx_lock:
        _targets.clear()      # a queued drag must not immediately undo the home
    _go_home.set()


def _encode_set_target(channel: int, qus: int) -> bytes:
    return bytes([_CMD_SET_TARGET, channel, qus & 0x7F, (qus >> 7) & 0x7F])


def _write_target(ser, cfg: dict[str, int], qus: int, *, playback=False) -> None:
    """Apply the commissioning profile before a user-requested throttle target."""
    ch = cfg["ch"]
    if not playback and _measurement_mode.is_set() and ch in _MEASUREMENT_PROFILES:
        speed, acceleration = _MEASUREMENT_PROFILES[ch]
        cfg = dict(cfg, speed=speed, acceleration=acceleration)
    qus = max(cfg["min"], min(cfg["max"], int(qus)))
    if ch in (8, 9, 10):
        invalidate_park()
    commands = []
    for key, cmd in (("acceleration", _CMD_SET_ACCEL), ("speed", _CMD_SET_SPEED)):
        if key in cfg:
            value = cfg[key]
            commands.append(bytes([cmd, ch, value & 0x7F, (value >> 7) & 0x7F]))
    commands.append(_encode_set_target(ch, qus))
    for command in commands:
        if ser.write(command) != len(command):
            raise OSError("Incomplete Maestro command write")
    if _recorder is not None and not playback and ch in (8, 9, 10):
        _recorder.record(ch, qus, cfg["speed"], cfg["acceleration"])
        _update(sequence_status=f"Recording: {_recorder.data['name']} · {len(_recorder.data['events'])} commands")


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



def _require_stationary(ser) -> None:
    ser.reset_input_buffer()
    ser.write(bytes([0x93]))  # Maestro GET MOVING STATE; no movement command.
    if ser.read(1) != bytes([0]):
        raise ValueError("Wait for movement to finish, then try again")


def _capture_measurement(ser, by_channel, note: str, path=None) -> dict[int, int]:
    """Record fresh pulse readbacks only; no shaft-feedback claim or movement."""
    _require_stationary(ser)
    positions = _read_positions(ser, [8, 9, 10])
    for ch in (8, 9, 10):
        cfg = by_channel[ch]
        value = positions.get(ch, 0)
        if not cfg["min"] <= value <= cfg["max"]:
            raise ValueError(f"Channel {ch} is off, unreadable, or outside configured limits")
    _require_stationary(ser)
    path = Path(path) if path is not None else _MEASUREMENT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if header_needed:
            writer.writerow(["timestamp_utc", "shoulder_us", "elbow_us", "wrist_us",
                             "note", "source"])
        writer.writerow([datetime.now(timezone.utc).isoformat(),
                         *(positions[ch] / 4 for ch in (8, 9, 10)), note,
                         "maestro_pulse_readback_not_shaft_feedback"])
    return positions


def _nudge(ser, cfg, delta_us: int) -> int:
    """Nudge from fresh board pulse, never from an off-channel placeholder."""
    if not _measurement_mode.is_set():
        raise ValueError("Enable measurement mode first")
    if delta_us not in (-10, -5, -1, 1, 5, 10):
        raise ValueError("Unsupported nudge size")
    _require_stationary(ser)
    ch = cfg["ch"]
    current = _read_positions(ser, [ch]).get(ch, 0)
    if not cfg["min"] <= current <= cfg["max"]:
        raise ValueError(f"Channel {ch} is off or unreadable; select its initial pulse manually")
    target = max(cfg["min"], min(cfg["max"], current + delta_us * 4))
    _write_target(ser, cfg, target)
    return target


def _discard_measurement_requests() -> None:
    discarded = False
    while True:
        try:
            _measurement_requests.get_nowait()
            discarded = True
        except queue.Empty:
            break
    if discarded:
        _update(measurement_status="Request canceled — connection or control mode changed")



def _sequence_pose(ser, by_channel):
    _require_stationary(ser)
    pose = _read_positions(ser, [8, 9, 10])
    validate_pose(pose, by_channel)
    _require_stationary(ser)
    return pose


def _cancel_sequence_state():
    global _recorder, _playback
    active = _recorder is not None or _playback is not None
    _recorder = _playback = None
    _playback_stop.clear()
    while True:
        try:
            _sequence_requests.get_nowait()
        except queue.Empty:
            break
    if active:
        _update(sequence_mode="idle", sequence_status="Sequence canceled — connection/control changed")


def _hold_throttle(ser, by_channel):
    pose = _read_positions(ser, [8, 9, 10])
    validate_pose(pose, by_channel)
    invalidate_park()
    for ch, value in pose.items():
        packet = _encode_set_target(ch, value)
        if ser.write(packet) != len(packet):
            raise OSError('Could not hold current throttle pulse positions')
    _update(pending_positions=pose)


def _sequence_request(ser, by_channel, request):
    global _recorder, _playback
    action, value = request
    if action == "discard":
        if _playback is not None:
            raise ValueError("Use Stop playback to hold the current pose")
        _recorder = None
        _update(sequence_mode="idle", sequence_status="Recording discarded; pose unchanged")
    elif action == "start":
        if _recorder is not None or _playback is not None:
            raise ValueError('Finish the current sequence first')
        pose = _sequence_pose(ser, by_channel)
        _recorder = Recorder(value, pose, by_channel)
        _update(sequence_mode="recording", sequence_status=f"Recording: {value} · move throttle sliders or nudges")
    elif action == "save":
        if _recorder is None:
            raise ValueError('No sequence is recording')
        pose = _sequence_pose(ser, by_channel)
        path = _recorder.save(pose, by_channel, _SEQUENCE_DIR)
        _recorder = None
        _update(sequence_mode="idle", sequence_status=f"Saved sequence: {path.stem}")
    elif action == "play":
        if _recorder is not None or _playback is not None:
            raise ValueError('Finish the current sequence first')
        data = load_sequence(value, by_channel)
        require_start_pose(data, _sequence_pose(ser, by_channel), by_channel)
        _playback_stop.clear()
        _playback = dict(data=data, started=time.monotonic(), index=0)
        _update(sequence_mode="playing", sequence_status=f"Playing: {data['name']}")


def _playback_tick(ser, by_channel):
    global _playback
    if _playback is None:
        return
    if _playback_stop.is_set():
        _playback = None  # Cancel future targets even if holding fails.
        _playback_stop.clear()
        _hold_throttle(ser, by_channel)
        _update(sequence_mode="idle", sequence_status="Playback stopped — holding current pulses")
        return
    elapsed = time.monotonic() - _playback['started']
    data = _playback['data']
    events = data['events']
    # One event per pass prevents overdue commands from being burst onto the wire.
    if _playback['index'] < len(events):
        event = events[_playback['index']]
        if elapsed >= event['at']:
            if elapsed - event["at"] > 0.25:
                raise ValueError("Playback timing slipped; stopping rather than skipping the demonstrated path")
            cfg = dict(by_channel[event['channel']], speed=event['speed'], acceleration=event['acceleration'])
            _write_target(ser, cfg, event['target'], playback=True)
            _playback['index'] += 1
            _update(pending_positions={event['channel']: event['target']})
    elif elapsed >= data['duration']:
        # Never report completion while pulses are still ramping or unreadable.
        try:
            pose = _sequence_pose(ser, by_channel)
            if any(abs(pose[ch] - data['end_pose'][ch]) > 4 for ch in (8, 9, 10)):
                raise ValueError('End pose does not match recorded pulse values')
        except ValueError:
            if elapsed < data['duration'] + 10:
                return
            raise ValueError('Playback did not reach its recorded end pose')
        _playback = None
        _update(sequence_mode="idle", sequence_status="Playback complete", pending_positions=pose)


def _worker() -> None:
    import serial

    ser = None
    by_channel = {cfg["ch"]: cfg for cfg in _servos().values()}
    channels = list(by_channel)

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
        _discard_measurement_requests()
        _cancel_sequence_state()

    while not _stop.is_set():
        port = _maestro_port()
        if not port:
            _close()
            _update(mode="no_port", detail="MAESTRO_PORT not set in .env")
            _stop.wait(5.0)
            continue

        if _rex_running():
            _close()
            if ser is None:
                log.debug("Rex is running — Maestro port released (dormant).")
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
        if _go_home.is_set() and (_recorder is not None or _playback is not None):
            _go_home.clear()
            _update(sequence_status="Finish or stop the sequence before go-home")
        if _go_home.is_set():
            _go_home.clear()
            try:
                invalidate_park()
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
        if _playback is not None:
            pending.clear()
        for ch, qus in pending.items():
            try:
                _write_target(ser, by_channel[ch], qus)
            except Exception as exc:
                log.info("set_target write failed (%s) — reopening.", exc)
                _close()
                break

        if ser is not None:
            try:
                request = _measurement_requests.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                try:
                    # Keep slider targets from racing the readback snapshot.
                    with _tx_lock:
                        if pending or _targets or _go_home.is_set():
                            raise ValueError("Wait for pending movement, then try again")
                        if request[0] == "record":
                            positions = _capture_measurement(ser, by_channel, request[1])
                            summary = " / ".join(f"{positions[ch] / 4:g}" for ch in (8, 9, 10))
                            _update(measurement_status=f"Saved S / E / W: {summary} µs",
                                    pending_positions=positions)
                        else:
                            if _playback is not None:
                                raise ValueError("Stop playback before nudging")
                            _, ch, delta = request
                            target = _nudge(ser, by_channel[ch], delta)
                            _update(measurement_status=f"Nudged ch{ch} to {target / 4:g} µs",
                                    pending_positions={ch: target})
                except Exception as exc:
                    _update(measurement_status=str(exc))
                    log.warning("Measurement request: %s", exc)

        if ser is not None:
            try:
                request = _sequence_requests.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                try:
                    with _tx_lock:
                        if pending or _targets or _go_home.is_set():
                            raise ValueError("Wait for pending movement, then try again")
                        _sequence_request(ser, by_channel, request)
                except Exception as exc:
                    _update(sequence_status=str(exc))
                    log.warning("Sequence request: %s", exc)
            try:
                _playback_tick(ser, by_channel)
            except Exception as exc:
                log.exception("Playback failed")
                try:
                    _hold_throttle(ser, by_channel)
                except Exception:
                    log.exception("Could not hold throttle after playback failure")
                _close()
                _update(sequence_mode="idle", sequence_status=f"Playback aborted: {exc}")

        _stop.wait(0.005 if _playback is not None else (0.05 if pending else 0.25))

    _close()


# ── Menu bar app ───────────────────────────────────────────────────────────────



def _after_menu_closes(rumps, callback):
    """Run once in the default AppKit run-loop mode, outside menu tracking."""
    def fire(timer):
        timer.stop()
        callback()
    timer = rumps.Timer(fire, 0.1)
    # Unlike the status refresh timer, never add this timer to event-tracking mode.
    timer.start()
    return timer


def _pose_note_dialog(rumps, *, title="Record throttle pose", message=None, ok="Record"):
    """Give the menu-bar accessory app a visible, keyboard-focused note field."""
    from AppKit import NSApplication, NSApplicationActivationPolicyRegular

    app = NSApplication.sharedApplication()
    # A launchd-started Python process can remain activation-prohibited: ordering
    # its alert forward alone cannot make it receive keyboard input. Promote it
    # to a foreground app for the modal, then restore accessory mode afterward.
    if not app.setActivationPolicy_(NSApplicationActivationPolicyRegular):
        raise RuntimeError("macOS refused to activate the recording window")
    app.activateIgnoringOtherApps_(True)
    dialog = rumps.Window(
        message or "Clearance note (optional): describe the obstacle or safe pose. "
        "Recording reads all three pulse values without moving the arm.",
        title=title, default_text="", ok=ok, cancel="Cancel",
        dimensions=(380, 28))
    field = dialog._textfield
    field.setEditable_(True)
    field.setSelectable_(True)
    field.setBezeled_(True)
    field.setDrawsBackground_(True)
    field.setPlaceholderString_("e.g. park to reach" if message else "e.g. wrist straight; clear of floor")
    dialog._alert.layout()
    dialog._alert.window().setInitialFirstResponder_(field)
    dialog._alert.window().makeFirstResponder_(field)
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    dialog._alert.window().center()
    dialog._alert.window().makeKeyAndOrderFront_(None)
    dialog._alert.window().orderFrontRegardless()
    dialog._alert.window().makeFirstResponder_(field)
    field.selectText_(None)
    log.info("Recording dialog activation policy=%s, active=%s, key=%s",
             app.activationPolicy(), app.isActive(), dialog._alert.window().isKeyWindow())
    return dialog


def run_app() -> int:
    try:
        import rumps
    except ImportError:
        log.error("rumps not installed in venv — run: venv/bin/pip install rumps")
        return 1

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    servos = _servos()

    class RexServoApp(rumps.App):
        def __init__(self):
            super().__init__("R3XServo", title="🦾 Servo Control",
                             quit_button="Quit Servo Control")
            self._status = rumps.MenuItem("status", callback=lambda _: None)
            self._labels: dict[str, rumps.MenuItem] = {}
            self._sliders: dict[str, object] = {}
            self._by_channel: dict[int, str] = {}
            self._record_dialog_pending = False
            self._record_timer = None
            self._measurement = rumps.MenuItem("Measurement mode (slow throttle)", callback=self._toggle_measurement)
            self._measurement_status = rumps.MenuItem("Measurement: off")
            self._sequence_status = rumps.MenuItem("Sequence: idle")
            self._sequence_menu = rumps.MenuItem("Throttle sequences")
            self._sequence_menu.add(rumps.MenuItem("Start recording…", callback=self._start_sequence))
            self._sequence_menu.add(rumps.MenuItem("Stop and save recording", callback=lambda _: self._queue_sequence("save")))
            self._sequence_menu.add(rumps.MenuItem("Discard recording", callback=lambda _: self._queue_sequence("discard")))
            self._sequence_menu.add(rumps.MenuItem("Stop playback — hold position", callback=lambda _: _playback_stop.set()))
            self._play_menu = rumps.MenuItem("Play saved sequence")
            self._play_menu.add(rumps.MenuItem("No saved sequences"))
            self._sequence_menu.add(self._play_menu)
            self._sequence_files = None
            self._record = rumps.MenuItem("Record this pose…", callback=self._record_pose)
            menu: list = [self._status, self._sequence_menu, self._sequence_status, self._measurement, self._record, self._measurement_status, rumps.MenuItem("Throttle: manual clearance required"), None]
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
                if name in _THROTTLE_DIRECTIONS:
                    nudges = rumps.MenuItem(f"Nudge {name.removeprefix('throttle_')} (µs)")
                    for delta in (-10, -5, -1, 1, 5, 10):
                        nudges.add(rumps.MenuItem(f"{delta:+d} µs", callback=self._make_nudge_cb(cfg["ch"], delta)))
                    menu.append(nudges)
                self._set_label(name, cfg["neutral"], "unread")
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

        def _queue_sequence(self, action, value=None):
            if _snapshot()["mode"] != "live":
                _update(sequence_status="Connect the Maestro and stop Rex first")
                return
            _sequence_requests.put((action, value))

        def _start_sequence(self, _sender):
            if self._record_dialog_pending:
                return
            self._record_dialog_pending = True
            self.menu._menu.cancelTracking()
            self._record_timer = _after_menu_closes(rumps, self._name_sequence)

        def _name_sequence(self):
            try:
                response = _pose_note_dialog(rumps, title="Name throttle sequence", ok="Start recording",
                    message="Start at a known safe pose. Name this demonstration, then move the throttle joints manually. Nothing moves when recording starts.").run()
                if response.clicked:
                    self._queue_sequence("start", response.text.strip())
            finally:
                NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
                self._record_dialog_pending = False
                self._record_timer = None

        def _refresh_sequences(self):
            paths = tuple(sorted(_SEQUENCE_DIR.glob("*.json"))) if _SEQUENCE_DIR.exists() else ()
            if paths == self._sequence_files:
                return
            self._sequence_files = paths
            self._play_menu.clear()
            if not paths:
                self._play_menu.add(rumps.MenuItem("No saved sequences"))
            for path in paths:
                self._play_menu.add(rumps.MenuItem(path.stem,
                    callback=lambda _, p=path: self._queue_sequence("play", p)))

        def _toggle_measurement(self, sender):
            if _measurement_mode.is_set():
                _measurement_mode.clear()
                sender.state = False
                _update(measurement_status="Measurement: off — normal speed on next move")
            else:
                _measurement_mode.set()
                sender.state = True
                _update(measurement_status="Slow on next move · nudge one joint at a time")
            _discard_measurement_requests()

        def _make_nudge_cb(self, channel, delta):
            def callback(_sender):
                if _snapshot()["mode"] != "live" or not _measurement_mode.is_set():
                    _update(measurement_status="Enable measurement mode while connected")
                    return
                _measurement_requests.put(("nudge", channel, delta))
            return callback

        def _record_pose(self, _sender):
            if self._record_dialog_pending:
                return
            self._record_dialog_pending = True
            self.menu._menu.cancelTracking()
            self._record_timer = _after_menu_closes(rumps, self._show_record_dialog)

        def _show_record_dialog(self):
            try:
                if _snapshot()["mode"] != "live":
                    rumps.alert("Cannot record pose", "Connect the Maestro and stop Rex so Servo Control can read the joints.")
                    return
                log.info("Opening throttle pose note dialog after menu closed")
                response = _pose_note_dialog(rumps).run()
                if response.clicked and _snapshot()["mode"] == "live":
                    _measurement_requests.put(("record", response.text.strip()))
                    _update(measurement_status="Reading pose…")
            except Exception:
                log.exception("Could not open throttle recording dialog")
                _update(measurement_status="Could not open recording dialog — see helper log")
            finally:
                NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
                self._record_dialog_pending = False
                self._record_timer = None

        def _set_label(self, name: str, qus: float, state: str = "target") -> None:
            direction = _THROTTLE_DIRECTIONS.get(name, "")
            value = f"{qus / 4.0:.2f}".rstrip("0").rstrip(".")
            reading = f"{value} µs ({state})" if state == "target" else state
            self._labels[name].title = f"ch{servos[name]['ch']} {name}: {reading} {direction}".rstrip()

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
            if _measurement_mode.is_set() or _snapshot().get("sequence_mode") in ("recording", "playing"):
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
            self._sequence_status.title = s.get("sequence_status", "Sequence: idle")
            self._refresh_sequences()
            self._measurement_status.title = s.get("measurement_status", "Measurement: off")
            self._restart.hidden = (s["mode"] != "live" or _measurement_mode.is_set() or s.get("sequence_mode") in ("recording", "playing"))
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
                self._set_label(name, qus, "target" if qus > 0 else "off — no pulse")

    threading.Thread(target=_worker, daemon=True, name="rex-servo-serial").start()
    log.info("Servo Control menu bar app online (port=%s).", _maestro_port() or "<unset>")
    try:
        RexServoApp().run()
    finally:
        _stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(run_app())
