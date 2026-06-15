"""
Head Arduino (Uno) LED controller — 82 WS2812B NeoPixels.

Pixels 0–1: eyes (RGB).
Pixels 2–81: mouth trapezoid PCB (physically GRB, but the Arduino handles the swap
in its EMOTION_COLORS table — Python sends plain RGB unchanged).

While Rex is awake and not speaking (ACTIVE / IDLE / after SPEAK_STOP), the
firmware autonomously pulses the mouth at a dim 15–25 % of the current emotion
colour (the emotion of the last SPEAK:, neutral amber before the first turn) —
no host traffic needed to maintain it. Speaking brightens the mouth through the
wave animation, then it settles back to the glow. OFF / SLEEP / FADEOFF keep
the mouth dark / red-breathing / fading as before.

All operations are no-ops (with a debug log) when HEAD_LEDS_ENABLED is False.
"""

import logging
import threading
import time

import serial

import config
from utils.config_loader import ARDUINO_HEAD_PORT, HEAD_LEDS_ENABLED

_log = logging.getLogger(__name__)
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException, OSError)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_DROP_REPORT_INTERVAL_SECS = 5.0
_dropped_counts: dict[str, int] = {}
_drop_window_started_at = 0.0
_next_drop_report_at = 0.0
_drop_window_open = False
_speech_drop_notified = False
# Consecutive write timeouts (not disconnects). Reset on any successful write.
_consecutive_write_timeouts = 0
# Throttle for offline reconnect attempts from the heartbeat loop.
_last_reconnect_attempt_at = 0.0
_eye_color: tuple[int, int, int] = (0, 0, 0)
_eyes_active = False
_led_mode = "off"

# Heartbeat / running-eye state.
#   _eyes_should_be_on — True while Rex is awake and the eyes are meant to show
#     (set by active/idle/set_eye_color; cleared by off/sleep). The "intentionally
#     dark" signal, distinct from the transient _eyes_active blink-suspend flag.
#   _speaking — True between speak() and speak_stop(); the heartbeat yields then so
#     it never fights the mouth animation or adds serial traffic during the flood.
_eyes_should_be_on = False
_speaking = False
_heartbeat_thread: "threading.Thread | None" = None
_heartbeat_stop = threading.Event()


def _mirror_gui_head_led_state(
    *,
    mode: str | None = None,
    eye_color: tuple[int, int, int] | None = None,
    eyes_active: bool | None = None,
) -> None:
    """Best-effort mirror of head LED state for the optional GUI avatar."""
    try:
        from gui.state_bridge import gui_bridge

        gui_bridge.update_head_led_state(
            mode=mode,
            eye_color=eye_color,
            eyes_active=eyes_active,
        )
    except Exception:
        pass


def _cmd_family(cmd: str) -> str:
    return (cmd.split(":", 1)[0].strip().upper() or "UNKNOWN")


def _is_speech_led_command(family: str) -> bool:
    return family in {"SPEAK", "SPEAK_LEVEL", "SPEAK_STOP"}


def _is_critical_led_command(family: str) -> bool:
    # SPEAK and EYE are flushed too: SPEAK is the mouth-on trigger (must not sit
    # buffered behind the SPEAK_LEVEL flood, or the mouth fails to light) and EYE
    # is the eye-on assertion (speech-start + heartbeat). SPEAK_LEVEL stays
    # unflushed — it is the high-rate flood we deliberately let coalesce.
    return family in {"SPEAK", "SPEAK_STOP", "OFF", "IDLE", "ACTIVE", "SLEEP", "EYE", "FADEOFF"}


def _report_drops_if_due(now: float) -> None:
    global _dropped_counts, _drop_window_started_at, _next_drop_report_at
    if not _dropped_counts or now < _next_drop_report_at:
        return
    total = sum(_dropped_counts.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(_dropped_counts.items()))
    elapsed = now - _drop_window_started_at
    _log.warning(
        "Head Arduino not connected — dropped %d command(s) in %.1fs (%s). "
        "Suppressing per-command logs; summary repeats every %.0fs while disconnected.",
        total,
        elapsed,
        breakdown,
        _DROP_REPORT_INTERVAL_SECS,
    )
    _dropped_counts = {}
    _drop_window_started_at = now
    _next_drop_report_at = now + _DROP_REPORT_INTERVAL_SECS


def _record_drop(cmd: str) -> None:
    global _drop_window_started_at, _next_drop_report_at, _drop_window_open
    now = time.monotonic()
    # Open the window (and report immediately) only on the FIRST drop of a new
    # disconnect. Once open, subsequent drops accumulate silently and surface as
    # a single rolled-up summary every _DROP_REPORT_INTERVAL_SECS — _report_drops_if_due
    # clears the per-window counts but leaves the window open, so we must not treat
    # the now-empty dict as a brand-new disconnect (that bug logged every drop).
    if not _drop_window_open:
        _drop_window_started_at = now
        _next_drop_report_at = now  # report first drop immediately
        _drop_window_open = True
    family = _cmd_family(cmd)
    _dropped_counts[family] = _dropped_counts.get(family, 0) + 1
    _report_drops_if_due(now)


def _flush_drop_summary(reason: str) -> None:
    """Emit one final drop summary (if pending) and clear counters."""
    global _dropped_counts, _drop_window_started_at, _next_drop_report_at, _drop_window_open
    # Going back online closes any open drop window so the next disconnect logs
    # its first drop immediately again.
    _drop_window_open = False
    if not _dropped_counts:
        return
    now = time.monotonic()
    total = sum(_dropped_counts.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(_dropped_counts.items()))
    elapsed = now - _drop_window_started_at
    _log.info(
        "Head Arduino %s — %d command(s) were dropped over %.1fs (%s).",
        reason,
        total,
        elapsed,
        breakdown,
    )
    _dropped_counts = {}
    _drop_window_started_at = 0.0
    _next_drop_report_at = 0.0


# ── Connection ─────────────────────────────────────────────────────────────────

def connect() -> bool:
    global _ser, _speech_drop_notified, _consecutive_write_timeouts
    if not HEAD_LEDS_ENABLED:
        _log.debug("HEAD_LEDS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(
            ARDUINO_HEAD_PORT,
            config.HEAD_ARDUINO_BAUD,
            timeout=1,
            write_timeout=float(getattr(config, "HEAD_ARDUINO_WRITE_TIMEOUT_SECS", 0.75)),
        )
        _log.info("Head Arduino connected on %s at %d baud", ARDUINO_HEAD_PORT, config.HEAD_ARDUINO_BAUD)
        _flush_drop_summary("reconnected")
        _speech_drop_notified = False
        _consecutive_write_timeouts = 0
        _start_heartbeat()
        return True
    except _SERIAL_ERRORS as exc:
        _log.error("Failed to open head Arduino port %s: %s", ARDUINO_HEAD_PORT, exc)
        _ser = None
        return False


def _write_timeout_disconnect_limit() -> int:
    """Consecutive write timeouts tolerated before we treat the link as down."""
    try:
        return max(1, int(getattr(config, "HEAD_ARDUINO_WRITE_TIMEOUT_MAX_CONSECUTIVE", 5)))
    except Exception:
        return 5


def _latch_offline_locked(cmd: str, family: str) -> None:
    """Close the port and mark the head Arduino offline. Caller must hold _lock."""
    global _ser, _speech_drop_notified
    try:
        if _ser and _ser.is_open:
            _ser.close()
    except Exception:
        pass
    _ser = None
    if _is_speech_led_command(family):
        # Re-arm the one-shot speech-drop notice so the next offline speech
        # routine logs once.
        _speech_drop_notified = False
    else:
        _record_drop(cmd)


def disconnect() -> None:
    global _ser, _speech_drop_notified
    _stop_heartbeat()
    with _lock:
        if _ser and _ser.is_open:
            _ser.close()
        _ser = None
        _speech_drop_notified = False


def _serial_online_locked() -> bool:
    return bool(_ser is not None and _ser.is_open)


def _serial_online() -> bool:
    with _lock:
        return _serial_online_locked()


def connected() -> bool:
    """True when the head-LED Arduino serial link is open (live status)."""
    return _serial_online()


# ── Transport ──────────────────────────────────────────────────────────────────

def send_command(cmd: str) -> None:
    """Send a newline-terminated command string to the head Arduino."""
    global _ser, _speech_drop_notified, _consecutive_write_timeouts
    if not HEAD_LEDS_ENABLED:
        _log.debug("send_command no-op: HEAD_LEDS_ENABLED=False (cmd=%r)", cmd)
        return
    family = _cmd_family(cmd)
    with _lock:
        if not _serial_online_locked():
            if _is_speech_led_command(family):
                if not _speech_drop_notified:
                    _log.warning(
                        "Head Arduino not connected — ignoring mouth LED updates for this speech routine."
                    )
                    _speech_drop_notified = True
                if family == "SPEAK_STOP":
                    _speech_drop_notified = False
                return
            _record_drop(cmd)
            return
        if family == "SPEAK_STOP":
            _speech_drop_notified = False
        _flush_drop_summary("is online")
        try:
            _ser.write((cmd + "\n").encode())
            if _is_critical_led_command(family):
                _ser.flush()
            _consecutive_write_timeouts = 0
        except serial.SerialTimeoutException as exc:
            # A write *timeout* is not a disconnect: the USB-CDC buffer was
            # momentarily full (common on macOS under load), but the board is
            # still there. Skip this one write and keep the port open — closing
            # here is what used to latch the head LEDs offline for the rest of
            # the session. Only give up after a run of consecutive timeouts,
            # which does suggest a genuinely wedged link.
            _consecutive_write_timeouts += 1
            limit = _write_timeout_disconnect_limit()
            if _consecutive_write_timeouts >= limit:
                _log.warning(
                    "Head Arduino: %d consecutive write timeouts on %s — treating "
                    "as disconnect; heartbeat will attempt to reconnect.",
                    _consecutive_write_timeouts, family,
                )
                _latch_offline_locked(cmd, family)
            else:
                _log.debug(
                    "Head Arduino write timeout on %s (%d/%d) — skipping one write, "
                    "port stays open: %s",
                    family, _consecutive_write_timeouts, limit, exc,
                )
        except (serial.SerialException, OSError) as exc:
            _log.warning("Head Arduino write failed for %s command: %s", family, exc)
            _latch_offline_locked(cmd, family)


# ── Command API ────────────────────────────────────────────────────────────────

def speak(emotion: str) -> None:
    """Start mouth speak animation for the given emotion. Eyes stay unchanged."""
    global _speaking
    _speaking = True
    send_command(f"SPEAK:{emotion}")


def _default_running_color() -> tuple[int, int, int]:
    """The 'awake' eye colour the heartbeat/ensure assert when none is set."""
    c = getattr(config, "HEAD_LED_RUNNING_EYE_COLOR", (255, 200, 0))
    try:
        return (max(0, min(255, int(c[0]))), max(0, min(255, int(c[1]))), max(0, min(255, int(c[2]))))
    except Exception:
        return (255, 200, 0)


def _resolve_running_eye_color(emotion: str | None) -> tuple[int, int, int]:
    """Pick a NON-black eye colour for the running/speaking state.

    Prefers the emotion's EYE_COLORS entry (when not the dark 'sleep' style),
    then the last set colour, then the configured default. Never returns black —
    the eyes must stay visible while Rex is awake, even on a 'sleep'-styled line.
    """
    if emotion:
        color = config.EYE_COLORS.get(emotion)
        if color and any(color):
            return (int(color[0]), int(color[1]), int(color[2]))
    if any(_eye_color):
        return _eye_color
    return _default_running_color()


def ensure_eyes_on(emotion: str | None = None) -> None:
    """Assert that the eyes are ON (lit + blinking) right now, at a live colour.

    Called at speech start so the eyes are reliably lit for the turn regardless of
    whether an earlier ACTIVE/EYE command was dropped, and reusable anywhere the
    eyes must be guaranteed on. Sends EYE:r,g,b — which lights the eyes and resumes
    blinking on the Arduino in any non-sleep mode — and marks the running state so
    the heartbeat keeps re-asserting it. Never turns the eyes off.
    """
    global _eye_color, _eyes_active, _eyes_should_be_on, _led_mode
    if not bool(getattr(config, "HEAD_LED_EYE_FOLLOWS_EMOTION", True)):
        emotion = None  # keep a steady running colour instead of per-turn emotion
    r, g, b = _resolve_running_eye_color(emotion)
    _eye_color = (r, g, b)
    _eyes_active = True
    _eyes_should_be_on = True
    _led_mode = "eye"
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=True)
    send_command(f"EYE:{r},{g},{b}")


def speak_level(brightness: int) -> None:
    """Set mouth brightness directly (0–255). Used to drive LEDs from audio level."""
    brightness = max(0, min(255, brightness))
    send_command(f"SPEAK_LEVEL:{brightness}")


def _resume_eye_blink() -> None:
    """Re-arm the head Arduino's autonomous eye blinking after a mouth stop.

    The Arduino SUSPENDS blinking on SPEAK_STOP (eyesActive→false) and only
    resumes it on an ACTIVE/EYE command. Without this, Rex blinks until his first
    spoken line and then never again. Re-assert ACTIVE (which preserves the
    current eye colour and resumes blinking). If the eyes are intentionally off
    (no colour set), leave them off rather than forcing them on.
    """
    if any(_eye_color):
        active()


def speak_stop() -> None:
    """Stop the mouth speak animation; the mouth settles into the firmware's dim
    emotional idle glow (15–25 % of the last SPEAK emotion's colour).

    Also re-arms eye blinking (see _resume_eye_blink): SPEAK_STOP suspends the
    Arduino's blink loop, so we must hand the eyes back to ACTIVE or Rex freezes
    his eyes open after the first thing he says.
    """
    global _eyes_active, _led_mode, _speaking
    _led_mode = "speak_stop"
    _eyes_active = False
    # Keep _speaking True through the ENTIRE stop sequence. The eye keep-alive
    # heartbeat is gated only on _speaking (see _heartbeat_tick); if we cleared it
    # here, the heartbeat could grab the lock during the SPEAK_STOP repeat loop's
    # GIL-releasing sleeps and inject flushed EYE: writes that contend with the
    # critical SPEAK_STOP on the lossy serial link — and a dropped SPEAK_STOP
    # leaves the firmware's autonomous mouth animation (ANIM_SPEAK) running
    # forever. Clear _speaking only AFTER the stop + eye re-arm are fully sent.
    _mirror_gui_head_led_state(mode=_led_mode, eyes_active=False)
    if not _serial_online():
        send_command("SPEAK_STOP")
        _resume_eye_blink()
        _speaking = False
        return
    send_command("SPEAK_LEVEL:0")
    repeats = int(getattr(config, "HEAD_LED_SPEAK_STOP_REPEATS", 3) or 1)
    repeats = max(1, min(10, repeats))
    delay = float(getattr(config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.025) or 0.0)
    delay = max(0.0, min(1.0, delay))
    for idx in range(repeats):
        send_command("SPEAK_STOP")
        if not _serial_online():
            break
        if idx < repeats - 1 and delay > 0.0:
            time.sleep(delay)
    _resume_eye_blink()
    _speaking = False


def idle() -> None:
    """Enter idle LED pattern (slow eye breathing + dim mouth glow)."""
    global _eyes_active, _led_mode, _eyes_should_be_on
    _led_mode = "idle"
    if any(_eye_color):
        _eyes_active = True
    _eyes_should_be_on = True
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=_eyes_active)
    send_command("IDLE")


def active() -> None:
    """Enter active LED pattern (steady eyes + dim mouth glow)."""
    global _eye_color, _eyes_active, _led_mode, _eyes_should_be_on
    _led_mode = "active"
    if not any(_eye_color):
        _eye_color = (255, 255, 255)
    _eyes_active = True
    _eyes_should_be_on = True
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=True)
    send_command("ACTIVE")


def set_eye_color(r: int, g: int, b: int) -> None:
    """
    Set eye pixels 0–1 to an RGB color.
    Eyes are standard RGB LEDs — values are passed through unchanged.
    """
    global _eye_color, _eyes_active, _led_mode, _eyes_should_be_on
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    _eye_color = (r, g, b)
    _eyes_active = any(_eye_color)
    _eyes_should_be_on = _eyes_active
    _led_mode = "eye"
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=_eyes_active)
    send_command(f"EYE:{r},{g},{b}")


def set_eye_emotion(emotion: str) -> None:
    """Convenience wrapper: looks up emotion in config.EYE_COLORS and sets eye color."""
    color = config.EYE_COLORS.get(emotion, config.EYE_COLORS["neutral"])
    set_eye_color(*color)


def off() -> None:
    """Turn all head LEDs off immediately."""
    global _eye_color, _eyes_active, _led_mode, _eyes_should_be_on, _speaking
    _eye_color = (0, 0, 0)
    _eyes_active = False
    _eyes_should_be_on = False
    _speaking = False
    _led_mode = "off"
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=False)
    if not _serial_online():
        send_command("OFF")
        return
    send_command("SPEAK_LEVEL:0")
    repeats = int(getattr(config, "HEAD_LED_SPEAK_STOP_REPEATS", 3) or 1)
    repeats = max(1, min(10, repeats))
    delay = float(getattr(config, "HEAD_LED_SPEAK_STOP_REPEAT_DELAY_SECS", 0.025) or 0.0)
    delay = max(0.0, min(1.0, delay))
    for idx in range(repeats):
        send_command("SPEAK_STOP")
        if not _serial_online():
            break
        send_command("OFF")
        if not _serial_online():
            break
        if idx < repeats - 1 and delay > 0.0:
            time.sleep(delay)


def fade_off() -> None:
    """Smoothly fade the head LEDs (eyes) to black instead of an instant off — a
    lifelike power-down for shutdown. Marks the eyes off and clears the heartbeat's
    keep-alive so nothing re-lights them; the firmware runs the ~4s brightness
    ramp autonomously, so this returns immediately (no blocking)."""
    global _eye_color, _eyes_active, _led_mode, _eyes_should_be_on, _speaking
    # Stop the eye heartbeat from re-asserting EYE: mid-fade (it gates on this flag).
    _eyes_should_be_on = False
    _eye_color = (0, 0, 0)
    _eyes_active = False
    _speaking = False
    _led_mode = "off"
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=False)
    send_command("FADEOFF")


def sleep() -> None:
    """Enter sleep LED state (eyes off, mouth slow red breathing)."""
    global _eye_color, _eyes_active, _led_mode, _eyes_should_be_on, _speaking
    _eye_color = (0, 0, 0)
    _eyes_active = False
    _eyes_should_be_on = False
    _speaking = False
    _led_mode = "sleep"
    _mirror_gui_head_led_state(mode=_led_mode, eye_color=_eye_color, eyes_active=False)
    send_command("SLEEP")


# ── Eye keep-alive heartbeat ─────────────────────────────────────────────────
#
# The head Arduino's serial link drops bytes during speech (FastLED.show()
# disables AVR interrupts while clocking 82 pixels), so the single post-speech
# ACTIVE re-arm can be lost — and nothing else re-asserts the eyes while running,
# leaving them dark until some later turn's re-arm happens to land. This low-rate
# daemon re-sends the current eye colour whenever Rex is awake (_eyes_should_be_on)
# and not mid-speech, so a dropped command self-heals within one interval. The tick
# itself is a pure passthrough when offline / off / sleeping / speaking, so it can
# never light eyes that are meant to be dark; the loop separately attempts a
# throttled reconnect while the port is down so a transient USB blip self-heals.

def _heartbeat_interval() -> float:
    try:
        return max(0.2, float(getattr(config, "HEAD_LED_HEARTBEAT_INTERVAL_SECS", 1.5)))
    except Exception:
        return 1.5


def _heartbeat_tick() -> None:
    """One keep-alive pass: re-assert the eyes if Rex is awake and quiet.

    The firmware treats an EYE: arriving in its OFF mode as "awake again" and
    resumes the mouth idle glow too, so this heartbeat also self-heals the glow
    after a firmware reboot mid-session (e.g. a USB blip + auto-reconnect).
    """
    global _eye_color
    with _lock:
        if not _serial_online_locked() or _speaking or not _eyes_should_be_on:
            return
        color = _eye_color if any(_eye_color) else _default_running_color()
        if not any(color):
            return
        _eye_color = color  # persist a defaulted colour so re-arm paths agree
    r, g, b = color
    # Outside the lock: send_command takes _lock itself (plain, non-reentrant).
    send_command(f"EYE:{r},{g},{b}")


def _attempt_reconnect() -> None:
    """Throttled attempt to reopen the head Arduino after it dropped offline.

    Runs from the heartbeat loop while the port is down. On success, connect()
    resets the drop/timeout state and the next _heartbeat_tick() re-asserts the
    eye colour, so a transient USB blip recovers on its own instead of leaving
    the head LEDs dark until the next full restart.
    """
    global _last_reconnect_attempt_at
    if not bool(getattr(config, "HEAD_LED_AUTO_RECONNECT", True)):
        return
    now = time.monotonic()
    try:
        interval = max(1.0, float(getattr(config, "HEAD_LED_RECONNECT_INTERVAL_SECS", 10.0)))
    except Exception:
        interval = 10.0
    if now - _last_reconnect_attempt_at < interval:
        return
    _last_reconnect_attempt_at = now
    _log.info("Head Arduino offline — attempting reconnect on %s ...", ARDUINO_HEAD_PORT)
    connect()


def _heartbeat_loop() -> None:
    while not _heartbeat_stop.is_set():
        if _heartbeat_stop.wait(_heartbeat_interval()):
            break
        try:
            if not _serial_online():
                _attempt_reconnect()
            _heartbeat_tick()
        except Exception as exc:  # never let the keep-alive thread die
            _log.debug("Head LED heartbeat tick failed: %s", exc)


def _start_heartbeat() -> None:
    global _heartbeat_thread
    if not bool(getattr(config, "HEAD_LED_HEARTBEAT_ENABLED", True)):
        return
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop, daemon=True, name="head-led-heartbeat"
    )
    _heartbeat_thread.start()


def _stop_heartbeat() -> None:
    global _heartbeat_thread
    _heartbeat_stop.set()
    thread = _heartbeat_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)
    _heartbeat_thread = None
