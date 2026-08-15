"""
ESP32-S3 radar-ring serial transport (LD2450 bearing prior).

Low-level link to the mmWave radar board: resolve the port, run the `hello`
handshake, and keep a thread-safe snapshot of the fused target list the
firmware streams at 10 Hz. The wire contract mirrors docs/motion_protocol.md
(NDJSON, v1); the firmware is firmware/djr3x_radar; the feature spec is
docs/radar-bearing-prior-spec.md.

This module is just the pipe. Consumers: the Motivator Control radar scope
(gui/dashboard.py RadarRingWidget) polls telemetry()/targets()/hello_info() for
display, and the come-here search (intelligence/motion_agency.py, radar-first
since 2026-08-15) reads `recent_targets()` to decide where to turn first. Every
consumer must treat the radar as a hint source, never a detector: it says
"a body is at 137°", the camera says whether it is the requester.

Two accessors, for two different questions:

- `targets()` — "where was somebody, most recently?" The LATCHED list: keeps
  returning the last non-empty frame for RADAR_TARGET_LATCH_SECS. Right for
  display and for "is anyone around". WRONG for deciding a body turn right
  after the base rotated: a latched bearing is in the PRE-turn frame.
- `recent_targets(window_secs, since=stamp)` — "what has the ring seen since
  this moment?" The raw per-frame history (empties included), so a caller can
  read only frames received after a turn's `done` plus a settle, and can
  demand a body show up in several frames before believing it (the LD2450
  tracker flickers; a one-frame return is not a person).

Differences from hardware/motion.py, each deliberate:

- **Port is resolved by USB serial number** (`RADAR_ESP32_SERIAL` in .env) via
  serial.tools.list_ports, with `RADAR_ESP32_PORT` as a literal-path escape
  hatch. There are two ESP32s on this Mac now and /dev/cu.usbmodem* numbering
  shuffles across reboots/replug order; the serial number doesn't. (The drive
  base still pins by path — flagged as a follow-up.)
- **No heartbeat.** The board is a sensor; it has nothing to stop if the Mac
  dies, and it streams from boot without a handshake.
- **A self-healing monitor thread.** The S3's native USB CDC device VANISHES
  from /dev when the firmware crashes (it doesn't error like a bridge chip),
  so the link can only be healed by re-resolving the serial number and
  reconnecting — which the monitor does every RADAR_RECONNECT_INTERVAL_SECS
  while the link is down. No controller owns this link, so the pipe heals
  itself.
- **The dropout latch lives in the getter.** The LD2450 tracks MOVING targets
  (no stationary channel), so a person who freezes falls off the list within a
  frame or two. `targets()` keeps returning the last non-empty list for
  RADAR_TARGET_LATCH_SECS so "they stopped walking" doesn't read as "nobody
  there". A dropout is not an empty room.

All operations are no-ops (with a debug log) when the radar is disabled
(config.RADAR_ENABLED off, or neither .env key set) — exactly like servos.py.
"""

import json
import logging
import threading
import time
from collections import deque

import serial
from serial.tools import list_ports

import config
from utils.config_loader import RADAR_ESP32_PORT, RADAR_ESP32_SERIAL

_log = logging.getLogger(__name__)
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException, OSError)

_PROTO_VERSION = int(getattr(config, "RADAR_PROTO_VERSION", 1))

# ── Module state ────────────────────────────────────────────────────────────────
_ser: "serial.Serial | None" = None
_write_lock = threading.Lock()     # serializes writes + close
_state_lock = threading.Lock()     # guards the snapshots below
_reader_thread: "threading.Thread | None" = None
_monitor_thread: "threading.Thread | None" = None
_stop = threading.Event()

_connected = False
_hello: "dict | None" = None
_latest_telemetry: "dict | None" = None   # + rx_monotonic stamp
_latched_targets: "list[dict]" = []       # last NON-EMPTY normalized target list
_latched_at = 0.0                         # monotonic stamp of that list
# Per-frame history for recent_targets(): (rx_monotonic, [normalized targets]),
# EMPTY frames included — "the ring saw nobody at that instant" is data. Sized
# for a few seconds at the firmware's 10 Hz; consumers ask for a window.
_RECENT_FRAMES_MAX = 60
_recent_frames: "deque[tuple[float, list[dict]]]" = deque(maxlen=_RECENT_FRAMES_MAX)
_parse_errors = 0

_last_log_at = 0.0                        # throttles the [radar] target INFO line
_had_targets = False                      # transition edge detection for logs
_shutting_down = False                    # set by shutdown(); parks the monitor


def _get_int(name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _get_float(name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _enabled() -> bool:
    return bool(getattr(config, "RADAR_ENABLED", True)) and (
        RADAR_ESP32_SERIAL is not None or RADAR_ESP32_PORT is not None
    )


# ── Port resolution ─────────────────────────────────────────────────────────────

def resolve_port() -> "str | None":
    """The radar board's current device path.

    Prefers matching RADAR_ESP32_SERIAL against the USB serial numbers of the
    attached ports (stable across reboots and replug order — the reason this
    board is NOT pinned by path); falls back to the literal RADAR_ESP32_PORT.
    Resolved fresh on every connect so a replug that lands on a new
    /dev/cu.usbmodem* number still finds the board.
    """
    if RADAR_ESP32_SERIAL:
        matches = []
        try:
            for p in list_ports.comports():
                sn = (p.serial_number or "").strip()
                if sn and sn.lower() == RADAR_ESP32_SERIAL.lower():
                    matches.append(p.device)
        except Exception as exc:
            _log.warning("Radar port enumeration failed: %s", exc)
            matches = []
        # macOS lists each USB CDC device as both /dev/cu.* and (sometimes)
        # /dev/tty.*; prefer cu (non-blocking open, project convention).
        if matches:
            matches.sort(key=lambda d: (0 if "/cu." in d else 1, d))
            if len(matches) > 2:   # >2 = genuinely ambiguous, not the cu/tty pair
                _log.warning(
                    "Radar serial %r matched multiple devices %s — using %s",
                    RADAR_ESP32_SERIAL, matches, matches[0],
                )
            return matches[0]
        if RADAR_ESP32_PORT:
            _log.debug(
                "Radar serial %r not found among attached ports — falling back to path %s",
                RADAR_ESP32_SERIAL, RADAR_ESP32_PORT,
            )
    return RADAR_ESP32_PORT


# ── Serial connection ───────────────────────────────────────────────────────────

def _open_serial_with_retries(
    port: str, *, log_errors: bool = True,
    attempts: "int | None" = None, delay: "float | None" = None,
) -> "serial.Serial | None":
    attempts = max(1, attempts if attempts is not None else _get_int("RADAR_CONNECT_RETRY_ATTEMPTS", 3))
    delay = max(0.0, delay if delay is not None else _get_float("RADAR_CONNECT_RETRY_DELAY_SECS", 1.0))
    timeout = max(0.01, _get_float("RADAR_SERIAL_TIMEOUT_SECS", 0.1))
    baud = _get_int("RADAR_BAUD", 115200)   # nominal — the link is native USB CDC

    for attempt in range(1, attempts + 1):
        try:
            # DEFAULT open on purpose — no DTR/RTS pre-drop, same rule as the
            # drive base (hardware/motion.py records the 2026-07-13 incident).
            # Native CDC has no auto-reset transistor pair at all, so a plain
            # open is doubly safe here.
            ser = serial.Serial(port, baud, timeout=timeout)
            _log.info(
                "Radar ring connected on %s at %d baud (attempt %d/%d)",
                port, baud, attempt, attempts,
            )
            return ser
        except _SERIAL_ERRORS as exc:
            level = logging.ERROR if attempt == attempts else logging.WARNING
            if log_errors:
                _log.log(
                    level, "Failed to open radar port %s (attempt %d/%d): %s",
                    port, attempt, attempts, exc,
                )
            if attempt < attempts and delay:
                time.sleep(delay)
    return None


def _close_serial_locked() -> None:
    global _ser, _connected
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
    _ser = None
    _connected = False


def connect(
    port: "str | None" = None, *,
    attempts: "int | None" = None, delay: "float | None" = None, log_errors: bool = True,
) -> bool:
    """Open the link and run the handshake. Returns True only on a clean
    handshake with a protocol-compatible firmware that advertises the "radar"
    cap (so the drive base — or any other hello-speaking ESP32 — is refused,
    not silently consumed). Starts the self-healing monitor either way, so a
    board that's absent at boot is picked up when plugged in."""
    global _ser, _connected, _hello, _latest_telemetry

    if not bool(getattr(config, "RADAR_ENABLED", True)):
        _log.debug("RADAR_ENABLED=False — skipping radar connect")
        return False
    if not _enabled():
        _log.debug("RADAR_ESP32_SERIAL/RADAR_ESP32_PORT not set — skipping radar connect")
        return False
    _start_monitor()

    port = port or resolve_port()
    if not port:
        if log_errors:
            _log.warning("Radar ring: no matching USB device found — will keep watching")
        return False

    ser = _open_serial_with_retries(port, log_errors=log_errors, attempts=attempts, delay=delay)
    if ser is None:
        return False

    with _write_lock:
        _ser = ser
    _stop.clear()
    with _state_lock:
        _hello = None
        _latest_telemetry = None
        _recent_frames.clear()
    _start_reader()

    # Handshake: send hello, await the hello reply. (Telemetry streams from
    # boot regardless — the handshake is identification, not activation.)
    send({"cmd": "hello", "host": "djr3x", "proto": _PROTO_VERSION})
    deadline = time.monotonic() + _get_int("RADAR_HANDSHAKE_TIMEOUT_MS", 1500) / 1000.0
    hello = None
    while time.monotonic() < deadline:
        with _state_lock:
            hello = _hello
        if hello is not None:
            break
        time.sleep(0.02)

    if hello is None:
        if log_errors:
            _log.warning("Radar ring: no hello reply within handshake timeout — disabling")
        disconnect()
        return False
    if hello.get("proto") != _PROTO_VERSION:
        if log_errors:
            _log.warning(
                "Radar ring: incompatible firmware proto=%s (need %d) — disabling",
                hello.get("proto"), _PROTO_VERSION,
            )
        disconnect()
        return False
    if "radar" not in (hello.get("caps") or []):
        if log_errors:
            _log.warning(
                "Radar ring: device on %s answers hello but caps=%s has no 'radar' — "
                "wrong board (drive base?); disabling", port, hello.get("caps"),
            )
        disconnect()
        return False

    _connected = True
    _log.info(
        "Radar ring: handshake OK (fw=%s boot_id=%s sensors=%s)",
        hello.get("fw"), hello.get("boot_id"),
        [s.get("mount") for s in (hello.get("sensors") or [])],
    )
    return True


def disconnect() -> None:
    global _connected
    _connected = False
    _stop.set()
    thread = _reader_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)
    with _write_lock:
        _close_serial_locked()


def shutdown() -> None:
    """Process shutdown: stop the monitor from healing the link back open
    (disconnect() alone is also what a failed handshake does, and THAT one
    wants the monitor to keep watching)."""
    global _shutting_down
    _shutting_down = True
    disconnect()


# ── Self-healing monitor ────────────────────────────────────────────────────────

def _start_monitor() -> None:
    """One daemon thread that re-resolves + reconnects while the link is down.

    Needed because (a) no controller owns this link the way motion_controller
    owns the base's, and (b) a firmware crash makes the CDC device VANISH from
    /dev — the reader gets a hard error, and only a fresh serial-number
    resolution can find the board again after it re-enumerates.
    """
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return
    _monitor_thread = threading.Thread(target=_monitor_loop, name="radar-monitor", daemon=True)
    _monitor_thread.start()


def _monitor_loop() -> None:
    was_connected = True   # suppress the "down" log until we've been up once
    while not _shutting_down:
        interval = max(1.0, _get_float("RADAR_RECONNECT_INTERVAL_SECS", 5.0))
        time.sleep(interval)
        if _shutting_down or not _enabled():
            continue
        if connected():
            was_connected = True
            continue
        if was_connected:
            _log.info("Radar ring: link down — watching for the board (serial-number match)")
            was_connected = False
        if connect(attempts=1, delay=0.0, log_errors=False):
            was_connected = True


# ── Background reader ───────────────────────────────────────────────────────────

def _start_reader() -> None:
    global _reader_thread
    if _reader_thread is not None and _reader_thread.is_alive():
        return
    _reader_thread = threading.Thread(target=_reader_loop, name="radar-reader", daemon=True)
    _reader_thread.start()


def _normalize_target(t: dict) -> "dict | None":
    try:
        return {
            "bearing_deg": float(t["b"]),
            "range_m": float(t["r"]),
            "confidence": float(t["c"]),
            "speed_mps": float(t.get("s", 0.0)),
            "sensors": int(t.get("m", 0)),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _log_targets(targets: "list[dict]") -> None:
    """The deliverable of this pass: bearings visible in the logs. Throttled
    while targets persist; appearance/disappearance edges always log."""
    global _last_log_at, _had_targets
    now = time.monotonic()
    if targets:
        interval = _get_float("RADAR_LOG_INTERVAL_SECS", 2.0)
        if not _had_targets or now - _last_log_at >= interval:
            _last_log_at = now
            _log.info(
                "[radar] %d target%s: %s", len(targets), "" if len(targets) == 1 else "s",
                " | ".join(
                    f"{t['bearing_deg']:+.0f}° {t['range_m']:.1f}m c={t['confidence']:.2f}"
                    for t in targets
                ),
            )
        _had_targets = True
    elif _had_targets:
        _had_targets = False
        _log.info(
            "[radar] targets lost (latched for %.0fs)",
            _get_float("RADAR_TARGET_LATCH_SECS", 3.0),
        )


def _dispatch(msg: dict) -> None:
    global _hello, _latest_telemetry, _latched_targets, _latched_at
    mtype = msg.get("type")
    if mtype == "telemetry":
        msg["rx_monotonic"] = time.monotonic()
        radar = msg.get("radar") or {}
        targets = [
            t for t in (_normalize_target(x) for x in radar.get("targets") or [])
            if t is not None
        ]
        with _state_lock:
            _latest_telemetry = msg
            _recent_frames.append((msg["rx_monotonic"], targets))
            if targets:
                _latched_targets = targets
                _latched_at = msg["rx_monotonic"]
        _log_targets(targets)
    elif mtype == "hello":
        with _state_lock:
            _hello = msg
    elif mtype == "event":
        _log.info("[radar] event %s %s", msg.get("event"),
                  {k: v for k, v in msg.items() if k not in ("type", "v", "t")})
    elif mtype == "log":
        lvl = {"debug": logging.DEBUG, "info": logging.INFO,
               "warn": logging.WARNING, "error": logging.ERROR}.get(msg.get("lvl"), logging.INFO)
        _log.log(lvl, "[radar_fw] %s", msg.get("msg"))


def _reader_loop() -> None:
    global _parse_errors
    buf = b""
    while not _stop.is_set():
        ser = _ser
        if ser is None:
            break
        try:
            chunk = ser.read(256)
        except _SERIAL_ERRORS as exc:
            # A crashed S3 takes the whole CDC device with it — this is the
            # "port vanished" path. The monitor thread re-resolves and heals.
            _log.debug("radar read failed: %s", exc)
            with _write_lock:
                _close_serial_locked()
            break
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8", "replace"))
            except Exception:
                _parse_errors += 1
                continue
            if isinstance(msg, dict):
                _dispatch(msg)


# ── Sending ─────────────────────────────────────────────────────────────────────

def send(obj: dict) -> bool:
    """Write one NDJSON line (adds v). The radar command set is tiny (hello);
    there is no seq/ack machinery to correlate."""
    ser = _ser
    if ser is None:
        return False
    msg = {"v": _PROTO_VERSION, **obj}
    line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
    with _write_lock:
        if _ser is None:
            return False
        try:
            _ser.write(line)
        except _SERIAL_ERRORS as exc:
            _log.warning("Radar write failed — closing link: %s", exc)
            _close_serial_locked()
            return False
    return True


# ── Accessors (thread-safe snapshots) ───────────────────────────────────────────

def connected() -> bool:
    return _connected and _ser is not None


def telemetry() -> "dict | None":
    """Latest raw telemetry frame (with rx_monotonic), or None."""
    with _state_lock:
        return dict(_latest_telemetry) if _latest_telemetry is not None else None


def targets() -> "list[dict]":
    """The fused target list, LATCHED across short dropouts.

    Each entry: {bearing_deg (+ = left/CCW, 0 = robot forward), range_m,
    confidence 0..1, speed_mps, sensors (contributing-sensor bitmask)},
    best-first. Returns the live list when the current frame has targets;
    otherwise keeps returning the last non-empty list for
    RADAR_TARGET_LATCH_SECS (the LD2450 drops people who freeze — a dropout is
    not "nobody there"). [] once the latch expires or nothing was ever seen.
    """
    now = time.monotonic()
    with _state_lock:
        latched = list(_latched_targets)
        latched_at = _latched_at
    if not latched:
        return []
    if now - latched_at > _get_float("RADAR_TARGET_LATCH_SECS", 3.0):
        return []
    return latched


def recent_targets(window_secs: float = 1.5,
                   since: "float | None" = None) -> "list[tuple[float, list[dict]]]":
    """The per-frame target history: ``[(rx_monotonic, [targets...]), ...]``,
    oldest first, for frames received within the last ``window_secs`` AND (when
    given) at or after the monotonic stamp ``since``. Empty frames are included
    — they are how a caller can tell "the ring watched for a second and saw
    nobody" from "the ring hasn't reported yet". Never latched: this is the
    accessor for anyone about to act on a bearing (a body turn), where a
    pre-rotation frame would point the wrong way. Each target dict is the same
    normalized shape as targets(). [] when disconnected or nothing qualifies."""
    if not connected():
        return []
    cutoff = time.monotonic() - max(0.0, float(window_secs))
    if since is not None:
        cutoff = max(cutoff, float(since))
    with _state_lock:
        return [(stamp, list(ts)) for stamp, ts in _recent_frames if stamp >= cutoff]


def radar_ok() -> bool:
    """True when the firmware reports at least one sensor delivering frames
    (radar.ok) AND the telemetry stream itself is fresh. False = the ring
    cannot see — distinct from "sees nobody" (ok with empty targets)."""
    t = telemetry()
    if not t:
        return False
    if time.monotonic() - t.get("rx_monotonic", 0.0) > 1.0:
        return False
    return bool((t.get("radar") or {}).get("ok"))


def hello_info() -> "dict | None":
    with _state_lock:
        return dict(_hello) if _hello is not None else None


def boot_id() -> "int | None":
    with _state_lock:
        return (_hello or {}).get("boot_id")
