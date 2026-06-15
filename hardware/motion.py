"""
ESP32 motion-base serial transport.

Low-level link to the differential-drive controller: open the port, run the
`hello` handshake, stream NDJSON commands, and keep a thread-safe snapshot of the
latest telemetry/acks/dones/events from a background reader. The wire contract is
docs/motion_protocol.md (v1); the firmware is firmware/djr3x_motion.

All operations are no-ops (with a debug log) when motion is disabled
(config.MOTION_ENABLED off or MOTION_ESP32_PORT unset) — exactly like servos.py.
High-level behavior (turn/move/heartbeat/safety gates) lives in
intelligence/motion_controller.py; this module is just the pipe.
"""

import json
import logging
import threading
import time

import serial

import config
from utils.config_loader import MOTION_ESP32_PORT

_log = logging.getLogger(__name__)
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException, OSError)

_PROTO_VERSION = int(getattr(config, "MOTION_PROTO_VERSION", 1))
_MAX_REMEMBERED = 64          # cap on retained acks/dones/events

# ── Module state ────────────────────────────────────────────────────────────────
_ser: "serial.Serial | None" = None
_write_lock = threading.Lock()     # serializes writes + close
_state_lock = threading.Lock()     # guards the snapshots below
_reader_thread: "threading.Thread | None" = None
_stop = threading.Event()
_seq_lock = threading.Lock()
_seq = 0

_connected = False
_last_port: "str | None" = None     # remembered for auto-reconnect after a drop
_hello: "dict | None" = None
_latest_telemetry: "dict | None" = None
_acks: "dict[int, dict]" = {}
_dones: "dict[int, dict]" = {}
_events: "list[dict]" = []
_parse_errors = 0

# Optional callbacks (set by the controller) — called from the reader thread.
_on_done = None       # fn(done_dict)
_on_event = None      # fn(event_dict)


def _get_int(name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _get_float(name: str, default: float) -> float:
    return float(getattr(config, name, default))


# ── Serial connection ───────────────────────────────────────────────────────────

def _open_serial_with_retries(
    port: str, *, log_errors: bool = True,
    attempts: "int | None" = None, delay: "float | None" = None,
) -> "serial.Serial | None":
    attempts = max(1, attempts if attempts is not None else _get_int("MOTION_CONNECT_RETRY_ATTEMPTS", 3))
    delay = max(0.0, delay if delay is not None else _get_float("MOTION_CONNECT_RETRY_DELAY_SECS", 1.0))
    timeout = max(0.01, _get_float("MOTION_SERIAL_TIMEOUT_SECS", 0.1))
    baud = _get_int("MOTION_BAUD", 115200)

    for attempt in range(1, attempts + 1):
        try:
            ser = serial.Serial(port, baud, timeout=timeout)
            _log.info(
                "Motion base connected on %s at %d baud (attempt %d/%d)",
                port, baud, attempt, attempts,
            )
            return ser
        except _SERIAL_ERRORS as exc:
            level = logging.ERROR if attempt == attempts else logging.WARNING
            if log_errors:
                _log.log(
                    level, "Failed to open motion port %s (attempt %d/%d): %s",
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
    _connected = False   # a closed link is never "connected" (incl. during a reconnect handshake)


def connect(
    port: "str | None" = None, *,
    attempts: "int | None" = None, delay: "float | None" = None, log_errors: bool = True,
) -> bool:
    """Open the link and run the handshake. Returns True only on a clean
    handshake with a protocol-compatible firmware. `attempts`/`delay`/`log_errors`
    let reconnect() retry fast and quietly."""
    global _ser, _connected, _hello, _last_port

    if not bool(getattr(config, "MOTION_ENABLED", True)):
        _log.debug("MOTION_ENABLED=False — skipping motion connect")
        return False
    port = port or MOTION_ESP32_PORT
    if not port:
        _log.debug("MOTION_ESP32_PORT not set — skipping motion connect")
        return False
    _last_port = port

    ser = _open_serial_with_retries(port, log_errors=log_errors, attempts=attempts, delay=delay)
    if ser is None:
        return False

    with _write_lock:
        _ser = ser
    _stop.clear()
    with _state_lock:
        _hello = None
        _latest_telemetry = None
        _acks.clear()
        _dones.clear()
        _events.clear()

    _start_reader()

    # ESP32 commonly auto-resets when the port opens; give it a moment to boot.
    time.sleep(0.3)

    # Handshake: send hello, await the hello reply.
    send({"cmd": "hello", "host": "djr3x", "proto": _PROTO_VERSION})
    deadline = time.monotonic() + _get_int("MOTION_HANDSHAKE_TIMEOUT_MS", 1500) / 1000.0
    while time.monotonic() < deadline:
        with _state_lock:
            hello = _hello
        if hello is not None:
            break
        time.sleep(0.02)

    if hello is None:
        if log_errors:
            _log.warning("Motion base: no hello reply within handshake timeout — disabling")
        disconnect()
        return False

    proto = hello.get("proto")
    if proto != _PROTO_VERSION:
        if log_errors:
            _log.warning(
                "Motion base: incompatible firmware proto=%s (need %d) — disabling",
                proto, _PROTO_VERSION,
            )
        disconnect()
        return False

    _connected = True
    _log.info(
        "Motion base: handshake OK (fw=%s caps=%s boot_id=%s)",
        hello.get("fw"), hello.get("caps"), hello.get("boot_id"),
    )
    return True


def reconnect() -> bool:
    """Re-open the last-used port with a single fast, quiet attempt + handshake.
    Used by the controller's link manager to heal after an unplug/replug (the
    device usually reappears on the same /dev path). Returns True on success."""
    port = _last_port or MOTION_ESP32_PORT
    if not port:
        return False
    return connect(port, attempts=1, delay=0.0, log_errors=False)


def disconnect() -> None:
    global _connected
    _connected = False
    _stop.set()
    thread = _reader_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)
    with _write_lock:
        _close_serial_locked()


# ── Background reader ─────────────────────────────────────────────────────────────

def _start_reader() -> None:
    global _reader_thread
    if _reader_thread is not None and _reader_thread.is_alive():
        return
    _reader_thread = threading.Thread(target=_reader_loop, name="motion-reader", daemon=True)
    _reader_thread.start()


def _remember(store: dict, key: int, value: dict) -> None:
    store[key] = value
    if len(store) > _MAX_REMEMBERED:
        # drop the oldest by insertion order
        for old in list(store.keys())[: len(store) - _MAX_REMEMBERED]:
            store.pop(old, None)


def _dispatch(msg: dict) -> None:
    global _hello, _latest_telemetry, _on_done, _on_event
    mtype = msg.get("type")
    if mtype == "telemetry":
        with _state_lock:
            _latest_telemetry = msg
    elif mtype == "hello":
        with _state_lock:
            _hello = msg
    elif mtype == "ack":
        seq = msg.get("seq")
        if isinstance(seq, int):
            with _state_lock:
                _remember(_acks, seq, msg)
    elif mtype == "done":
        seq = msg.get("seq")
        if isinstance(seq, int):
            with _state_lock:
                _remember(_dones, seq, msg)
        if _on_done is not None:
            try:
                _on_done(msg)
            except Exception:
                _log.debug("motion on_done callback failed", exc_info=True)
    elif mtype == "event":
        with _state_lock:
            _events.append(msg)
            if len(_events) > _MAX_REMEMBERED:
                del _events[0 : len(_events) - _MAX_REMEMBERED]
        if _on_event is not None:
            try:
                _on_event(msg)
            except Exception:
                _log.debug("motion on_event callback failed", exc_info=True)
    elif mtype == "log":
        lvl = {"debug": logging.DEBUG, "info": logging.INFO,
               "warn": logging.WARNING, "error": logging.ERROR}.get(msg.get("lvl"), logging.INFO)
        _log.log(lvl, "[motion_fw] %s", msg.get("msg"))


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
            _log.debug("motion read failed: %s", exc)
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


# ── Sending ───────────────────────────────────────────────────────────────────────

def _next_seq() -> int:
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


def send(obj: dict) -> "int | None":
    """Write one NDJSON command line. Adds v + seq. Returns the seq, or None if
    the link is down / the write failed (caller treats None as not-sent)."""
    ser = _ser
    if ser is None:
        return None
    msg = {"v": _PROTO_VERSION, **obj}
    seq = None
    if msg.get("cmd") is not None:
        seq = _next_seq()
        msg["seq"] = seq
    line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
    with _write_lock:
        if _ser is None:
            return None
        try:
            _ser.write(line)
        except _SERIAL_ERRORS as exc:
            _log.warning("Motion write failed — closing link: %s", exc)
            _close_serial_locked()
            return None
    return seq


def ping() -> "int | None":
    return send({"cmd": "ping"})


def set_callbacks(*, on_done=None, on_event=None) -> None:
    global _on_done, _on_event
    _on_done = on_done
    _on_event = on_event


# ── Accessors (thread-safe snapshots) ──────────────────────────────────────────────

def connected() -> bool:
    return _connected and _ser is not None


def telemetry() -> "dict | None":
    with _state_lock:
        return dict(_latest_telemetry) if _latest_telemetry is not None else None


def hello_info() -> "dict | None":
    with _state_lock:
        return dict(_hello) if _hello is not None else None


def caps() -> "list[str]":
    with _state_lock:
        return list((_hello or {}).get("caps", []))


def boot_id() -> "int | None":
    with _state_lock:
        return (_hello or {}).get("boot_id")


def owner() -> str:
    """'auto' | 'manual' | 'unknown' (no telemetry yet)."""
    t = telemetry()
    return t.get("owner", "unknown") if t else "unknown"


def state() -> str:
    t = telemetry()
    return t.get("state", "unknown") if t else "unknown"


def wait_ack(seq: int, timeout: float = 0.5) -> "dict | None":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _state_lock:
            ack = _acks.get(seq)
        if ack is not None:
            return ack
        time.sleep(0.01)
    return None


def wait_done(seq: int, timeout: float = 8.0) -> "dict | None":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _state_lock:
            done = _dones.get(seq)
        if done is not None:
            return done
        time.sleep(0.02)
    return None
