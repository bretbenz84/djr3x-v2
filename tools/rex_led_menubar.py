#!/usr/bin/env python3
"""
rex_led_menubar.py — macOS menu bar LED animation console for the head + chest
Arduino boards.

A small always-on menu bar app (rumps/Cocoa, sibling of rex_servo_menubar.py)
titled "LED Control". The dropdown has two areas — Head and Chest — with one
clickable button per animation each board's firmware supports (the newline
commands documented in arduino/head_nano/head_nano.ino and
arduino/chest_nano/chest_nano.ino). Click a button and the command goes
straight down the wire, so you can audition any animation while the robot is
off. No dependency on the project config (which refuses to import without API
keys); ports come from .env directly.

Head speak animations are an equalizer driven by a SPEAK_LEVEL:{0-255} stream
that main.py normally derives from live TTS audio. Clicking a head Speak button
therefore also starts a synthetic level wave (a few overlapping sines) so the
mouth actually dances; any other head button stops the wave (Speak Stop sends
SPEAK_STOP first). Chest speak patterns animate autonomously — no stream needed.

How it shares the serial ports with main.py (ports are exclusive-open):
  Same dormant pattern as the battery meter and servo console. main.py holds
  the single-instance flock for its whole lifetime; each zone worker polls it
  ~1×/s:
    - lock held  → close the port (main.py owns the LEDs), buttons inert,
                   status row shows "Rex is running"
    - lock free  → reopen the port and the buttons go live
  Opening either port toggles DTR and reboots that Arduino (both boards reset
  on open on this Mac), so each worker waits ~2 s after opening before it will
  send anything.

Run directly for debugging:
    venv/bin/python tools/rex_led_menubar.py
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# utils.single_instance must be importable WITHOUT the heavy project config
# (mirrors the battery/servo apps — this process must start even when
# apikeys.py would fail).
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | rex_led | %(levelname)s | %(message)s",
)
log = logging.getLogger("rex_led")

_LOCK_POLL_SECS = 1.0
_BAUD = 115200                 # config.HEAD_ARDUINO_BAUD == CHEST_ARDUINO_BAUD
_BOOT_WAIT_SECS = 2.0          # DTR toggle on open reboots the Arduino
_SPEAK_LEVEL_HZ = 30.0         # synthetic level stream rate for head speak demos

# Animation buttons per zone: (menu label, wire command). Mirrors each
# firmware's handleCommand() — keep in sync if a sketch gains a mode.
_HEAD_ANIMATIONS: list[tuple[str, str]] = [
    ("Idle (eyes breathe + mouth glow)", "IDLE"),
    ("Active (steady eyes + glow)", "ACTIVE"),
    ("Speak — neutral", "SPEAK:neutral"),
    ("Speak — happy", "SPEAK:happy"),
    ("Speak — excited", "SPEAK:excited"),
    ("Speak — sad", "SPEAK:sad"),
    ("Speak — angry", "SPEAK:angry"),
    ("Speak — curious", "SPEAK:curious"),
    ("Speak Stop", "SPEAK_STOP"),
    ("Sleep (red mouth breathing)", "SLEEP"),
    ("Charge glow (demo 50%)", "CHARGE:50"),
    ("Fade Off (~4s power-down)", "FADEOFF"),
    ("Off", "OFF"),
]

_CHEST_ANIMATIONS: list[tuple[str, str]] = [
    ("Startup (short-circuit intro)", "STARTUP"),
    ("Idle (RandomBlocks2)", "IDLE"),
    ("Active (bright RandomBlocks2)", "ACTIVE"),
    ("Speak — neutral", "SPEAK:neutral"),
    ("Speak — happy (gold + confetti)", "SPEAK:happy"),
    ("Speak — excited (racing gold/red)", "SPEAK:excited"),
    ("Speak — sad (slow blue sighs)", "SPEAK:sad"),
    ("Speak — angry (red alert)", "SPEAK:angry"),
    ("Speak Stop", "SPEAK_STOP"),
    ("Compliment flash (white/blue)", "COMPLIMENT"),
    ("Next built-in pattern", "NEXT"),
    ("Sleep (dim red breathing)", "SLEEP"),
    ("Battery meter (demo 50%, charging)", "CHARGE:50:1"),
    ("Fade Off (~4s power-down)", "FADEOFF"),
    ("Off", "OFF"),
]


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


def _zone_port(env_key: str) -> str:
    env = _read_env_file()
    return (os.environ.get(env_key) or env.get(env_key) or "").strip()


def _rex_running() -> bool:
    try:
        from utils import single_instance
        return single_instance.is_held_by_other()
    except Exception as exc:
        log.debug("single_instance check failed: %s", exc)
        return False


# ── Per-zone serial worker ─────────────────────────────────────────────────────
# One worker thread per board. Buttons enqueue a command; the worker owns the
# port, drops to dormant while main.py runs, and reconnects on failure. The head
# worker additionally streams synthetic SPEAK_LEVEL values while a head speak
# animation is active (see module docstring).

class _Zone:
    def __init__(self, name: str, env_key: str, *, synth_speak_levels: bool):
        self.name = name
        self.env_key = env_key
        self.synth_speak_levels = synth_speak_levels
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._mode = "connecting"       # connecting | live | dormant | no_port
        self._detail = "starting…"
        self._speaking_since: float | None = None

    # UI thread → worker
    def enqueue(self, cmd: str) -> None:
        with self._lock:
            if self._mode != "live":
                return                  # inert while Rex owns the port
            self._queue.append(cmd)

    def status(self) -> tuple[str, str]:
        with self._lock:
            return self._mode, self._detail

    def _set(self, mode: str, detail: str) -> None:
        with self._lock:
            self._mode = mode
            self._detail = detail

    def _synth_level(self, now: float) -> int:
        """A lively fake voice envelope: overlapping sines, clamped to 0–255."""
        t = now - (self._speaking_since or now)
        v = (0.55
             + 0.30 * math.sin(2 * math.pi * 2.1 * t)
             + 0.25 * math.sin(2 * math.pi * 4.7 * t + 1.3)
             + 0.15 * math.sin(2 * math.pi * 0.5 * t))
        return max(0, min(255, int(v * 255)))

    def worker(self, stop: threading.Event) -> None:
        import serial

        ser = None
        ready_at = 0.0

        def _close():
            nonlocal ser
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
            with self._lock:
                self._queue.clear()
                self._speaking_since = None

        while not stop.is_set():
            port = _zone_port(self.env_key)
            if not port:
                _close()
                self._set("no_port", f"{self.env_key} not set in .env")
                stop.wait(5.0)
                continue

            if _rex_running():
                if ser is not None:
                    _close()
                    log.info("%s: Rex is running — port released (dormant).", self.name)
                self._set("dormant", "Rex is running")
                stop.wait(_LOCK_POLL_SECS)
                continue

            if ser is None:
                try:
                    ser = serial.Serial(port, _BAUD, timeout=0.2, exclusive=True)
                except Exception as exc:
                    self._set("connecting", f"waiting for board on {port}")
                    log.debug("%s: open %s failed: %s", self.name, port, exc)
                    stop.wait(2.0)
                    continue
                # The open just rebooted the board; hold fire until it's up.
                ready_at = time.monotonic() + _BOOT_WAIT_SECS
                log.info("%s: connected on %s (waiting %.1fs for boot).",
                         self.name, port, _BOOT_WAIT_SECS)

            now = time.monotonic()
            if now < ready_at:
                self._set("connecting", "board rebooting…")
                stop.wait(min(0.2, ready_at - now))
                continue
            self._set("live", f"live on {port}")

            with self._lock:
                pending = self._queue[:]
                self._queue.clear()
            try:
                for cmd in pending:
                    if self.synth_speak_levels:
                        # Track speak state so the level wave starts/stops with
                        # the command that changes it. Speak Stop and every
                        # non-speak mode both end the stream.
                        if cmd.startswith("SPEAK:"):
                            with self._lock:
                                self._speaking_since = time.monotonic()
                        else:
                            with self._lock:
                                self._speaking_since = None
                    ser.write((cmd + "\n").encode())
                    ser.flush()
                    log.info("%s: sent %s", self.name, cmd)
                with self._lock:
                    speaking = self._speaking_since is not None
                if speaking:
                    ser.write(f"SPEAK_LEVEL:{self._synth_level(time.monotonic())}\n".encode())
            except Exception as exc:
                log.info("%s: write failed (%s) — reopening.", self.name, exc)
                _close()
                continue

            stop.wait(1.0 / _SPEAK_LEVEL_HZ if speaking else 0.1)

        _close()


# ── Menu bar app ───────────────────────────────────────────────────────────────

def run_app() -> int:
    try:
        import rumps
    except ImportError:
        log.error("rumps not installed in venv — run: venv/bin/pip install rumps")
        return 1

    zones = [
        _Zone("head", "ARDUINO_HEAD_PORT", synth_speak_levels=True),
        _Zone("chest", "ARDUINO_CHEST_PORT", synth_speak_levels=False),
    ]
    animations = {"head": _HEAD_ANIMATIONS, "chest": _CHEST_ANIMATIONS}
    stop = threading.Event()

    class RexLedApp(rumps.App):
        def __init__(self):
            super().__init__("R3XLed", title="💡 LED Control",
                             quit_button="Quit LED Control")
            self._status: dict[str, rumps.MenuItem] = {}
            menu: list = []
            for zone in zones:
                header = rumps.MenuItem(f"— {zone.name.upper()} —",
                                        callback=lambda _: None)
                status = rumps.MenuItem("status", callback=lambda _: None)
                self._status[zone.name] = status
                menu += [header, status]
                for label, cmd in animations[zone.name]:
                    menu.append(rumps.MenuItem(label, callback=self._make_cb(zone, cmd)))
                menu.append(None)
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
            self._refresh(None)

        def _make_cb(self, zone: _Zone, cmd: str):
            def _cb(_item):
                zone.enqueue(cmd)
            return _cb

        def _refresh(self, _timer):
            for zone in zones:
                mode, detail = zone.status()
                line = {
                    "live": f"● {detail}",
                    "dormant": "⏸  Rex is running — buttons inert",
                    "connecting": f"… {detail}",
                    "no_port": f"✕ {detail}",
                }.get(mode, detail)
                self._status[zone.name].title = line

    for zone in zones:
        threading.Thread(target=zone.worker, args=(stop,), daemon=True,
                         name=f"rex-led-{zone.name}").start()
    log.info("LED Control menu bar app online (head=%s, chest=%s).",
             _zone_port("ARDUINO_HEAD_PORT") or "<unset>",
             _zone_port("ARDUINO_CHEST_PORT") or "<unset>")
    try:
        RexLedApp().run()
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(run_app())
