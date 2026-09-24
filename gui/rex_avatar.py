"""DJ R-3X avatar state for the optional dashboard.

RexAvatar here holds the snapshot-driven state only — smoothed servo targets,
the head-LED eye state and blink cycle, speech state and the chest-LED mode; the
3D subclass in gui/rex_avatar_3d.py renders it. The three chest LED pods mirror
the real chest Arduino at mode level (hardware/leds_chest → gui bridge): the
firmware animates autonomously, so chest_render_state re-creates each mode's
pattern — random blocks in idle, emotion-colored flicker while speaking, a
bottom-up sweep at startup, the contiguous charge meter, and the one-shot
compliment flash.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

import config


def normalize_servo(channel_or_name, value) -> float:
    """Normalize a servo value to 0.0..1.0 using config.SERVO_CHANNELS."""
    name = _servo_name(channel_or_name)
    if name is None:
        return 0.5
    cfg = config.SERVO_CHANNELS[name]
    lo = float(cfg["min"])
    hi = float(cfg["max"])
    if hi <= lo:
        return 0.5
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float(cfg["neutral"])
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


class RexAvatar(QWidget):
    def __init__(self, parent=None, *, show_background: bool = True) -> None:
        super().__init__(parent)
        self._show_background = bool(show_background)
        # Boot pose matches the powered-off robot: visor rolled fully DOWN over the
        # face. Real servo positions stream in from the first hardware command
        # (hardware/servos._record_servo_positions → gui bridge), so the avatar
        # mirrors the physical visor rolling up while the program loads.
        self._target: dict[str, float] = _boot_norms()
        self._current: dict[str, float] = dict(self._target)
        self._eye_state: dict[str, Any] = {
            "mode": "off",
            "eye_color": (0, 0, 0),
            "eyes_active": False,
            "updated_at": 0.0,
        }
        self._speech_state: dict[str, Any] = {
            "speaking": False,
            "audio_path": None,
            "updated_at": 0.0,
        }
        # Chest-panel LED mirror (mode-level; see chest_render_state). Boot state
        # matches the powered-off robot: panels dark until the STARTUP command.
        self._chest_state: dict[str, Any] = {
            "mode": "off",
            "emotion": None,
            "soc": None,
            "charging": False,
            "flash_at": 0.0,
            "updated_at": 0.0,
        }
        self._last_eye_event_at = 0.0
        self._blink_state = "open"
        self._blink_timer = time.monotonic()
        self._blink_interval = random.uniform(2.0, 8.0)
        self._blink_duration = 0.0
        self._is_second_blink = False
        self._idle_phase = 0.0
        self._last_blink_tick = time.monotonic()
        if not self._show_background:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAutoFillBackground(False)
        if self._show_background:
            # The avatar scales proportionally, so a soft minimum + preferred
            # sizeHint lets tight layouts (e.g. with the system-log strip)
            # shrink the avatar instead of clipping it.
            self.setMinimumSize(280, 240)
        else:
            self.setMinimumSize(1, 1)

    def sizeHint(self):  # noqa: N802 - Qt override
        from PySide6.QtCore import QSize

        return QSize(430, 400) if self._show_background else QSize(1, 1)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        ws = snapshot.get("world_state") or {}
        self_state = ws.get("self_state") or ws.get("self") or {}
        positions = dict(snapshot.get("servo_positions") or {})
        positions.update(self_state.get("servo_positions") or {})
        # Only track servos the runtime has actually reported. Missing data must NOT
        # default to neutral — that would snap the visor open at launch before the
        # first real servo command, instead of holding the powered-off boot pose.
        for name in config.SERVO_CHANNELS:
            if name in positions:
                self._target[name] = normalize_servo(name, positions[name])
        eye_state = snapshot.get("head_led_state") or {}
        if eye_state:
            self._eye_state.update(eye_state)
            event_at = float(eye_state.get("updated_at") or 0.0)
            if event_at != self._last_eye_event_at:
                self._last_eye_event_at = event_at
                self._reset_blink_cycle()
        speech_state = snapshot.get("speech_state") or {}
        if speech_state:
            self._speech_state.update(speech_state)
        chest_state = snapshot.get("chest_led_state") or {}
        if chest_state:
            self._chest_state.update(chest_state)
        self.update()

    # ── Animation state ─────────────────────────────────────────────────────

    def _tick_eye_animation(self) -> None:
        now = time.monotonic()
        dt = max(0.0, min(0.25, now - self._last_blink_tick))
        self._last_blink_tick = now
        if self._eye_state.get("mode") == "idle":
            self._idle_phase += dt * 0.8
        if not bool(self._eye_state.get("eyes_active")) or not any(_eye_color(self._eye_state)):
            self._blink_state = "open"
            self._is_second_blink = False
            self._blink_timer = now
            return

        if self._blink_state == "open":
            if now - self._blink_timer >= self._blink_interval:
                self._blink_state = "closed"
                self._blink_timer = now
                self._blink_duration = random.uniform(0.10, 0.40)
        elif self._blink_state == "closed":
            if now - self._blink_timer >= self._blink_duration:
                self._blink_timer = now
                if not self._is_second_blink and random.random() < 0.10:
                    self._blink_state = "double_wait"
                    self._blink_duration = random.uniform(0.20, 0.40)
                else:
                    self._blink_state = "open"
                    self._is_second_blink = False
                    self._blink_interval = random.uniform(2.0, 8.0)
        elif self._blink_state == "double_wait":
            if now - self._blink_timer >= self._blink_duration:
                self._blink_state = "closed"
                self._is_second_blink = True
                self._blink_timer = now
                self._blink_duration = random.uniform(0.10, 0.40)

    def _eye_brightness(self) -> float:
        if self._eye_state.get("mode") == "idle":
            return 0.30 + 0.35 * (1.0 + math.sin(self._idle_phase))
        return 1.0

    def _reset_blink_cycle(self) -> None:
        self._blink_state = "open"
        self._blink_timer = time.monotonic()
        self._blink_interval = random.uniform(2.0, 8.0)
        self._blink_duration = 0.0
        self._is_second_blink = False

    def _is_speaking(self) -> bool:
        return bool(self._speech_state.get("speaking"))

    def _smooth(self) -> None:
        smoothing = max(0.01, min(1.0, float(getattr(config, "GUI_AVATAR_SMOOTHING", 0.25))))
        for name, target in self._target.items():
            current = self._current.get(name, target)
            self._current[name] = current + (target - current) * smoothing


# Palettes lifted from arduino/chest_nano/chest_nano.ino so the avatar shows the
# SAME colors the physical panels do (screen-brightened where the firmware runs
# LEDs dim). SmallLEDColors: dim red x3 / dim white x4 / dim blue x2 — the ladder
# flicker. BlockLEDColors: cRED / cWHITE / cGOLD / cBLUE — the square blocks.
_CHEST_SMALL_COLORS: tuple[tuple[int, int, int], ...] = (
    (208, 44, 40), (208, 44, 40), (208, 44, 40),            # cRED2 (x3 weight)
    (196, 205, 210), (196, 205, 210), (196, 205, 210), (196, 205, 210),  # cWHITE2 (x4)
    (64, 88, 224), (64, 88, 224),                           # cBLUE2 (x2)
)
_CHEST_BLOCK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 56, 40),      # cRED
    (235, 240, 245),    # cWHITE
    (255, 221, 136),    # cGOLD
    (80, 120, 255),     # cBLUE
)
# SPEAK:<emotion> palettes (ladder, squares), matching the firmware patterns:
# excited = racing red-orange bars/blocks, sad = slow blue sighs, angry = solid
# red alert, happy = gold/white confetti over the normal blocks.
_CHEST_EMOTION_PALETTES: dict[str, tuple[tuple, tuple]] = {
    "excited": (((255, 70, 16), (255, 120, 30)), ((255, 90, 0), (255, 40, 20))),
    "sad": (((40, 70, 220), (25, 45, 170)), ((40, 70, 220), (25, 45, 170))),
    "angry": (((255, 42, 30),), ((255, 42, 30),)),
    "happy": (((255, 200, 80), (255, 255, 255)), _CHEST_BLOCK_COLORS),
}
# Charge-gauge gradient anchors (gaugeAnchorColor, screen-brightened ~1.8x):
# red at empty through orange/yellow/green to blue at full.
_CHEST_GAUGE_ANCHORS: tuple[tuple[int, int, int], ...] = (
    (234, 0, 0), (234, 40, 0), (234, 81, 0), (216, 180, 0),
    (108, 198, 27), (0, 216, 54), (0, 148, 162), (0, 81, 255),
)
_CHEST_FLASH_SECS = 2.2


def _chest_gauge_color(frac: float) -> tuple[int, int, int]:
    """Color at 0..1 along the charge gauge (blend between gradient anchors)."""
    frac = max(0.0, min(1.0, frac))
    pos = frac * (len(_CHEST_GAUGE_ANCHORS) - 1)
    i = int(pos)
    if i >= len(_CHEST_GAUGE_ANCHORS) - 1:
        return _CHEST_GAUGE_ANCHORS[-1]
    f = pos - i
    a, b = _CHEST_GAUGE_ANCHORS[i], _CHEST_GAUGE_ANCHORS[i + 1]
    return tuple(int(a[c] + (b[c] - a[c]) * f) for c in range(3))  # type: ignore[return-value]


def _pick(palette: tuple, rnd: float) -> tuple[int, int, int]:
    return palette[min(len(palette) - 1, int(rnd * len(palette)))]


def _prand(step: int, *salts: int) -> float:
    """Deterministic 0..1 pseudo-random per (time-step, salts) — gives the panels
    firmware-style random-block flicker without per-LED state."""
    x = (step * 2654435761) & 0xFFFFFFFF
    for s in salts:
        x ^= (s * 40503 + 0x9E3779B9 + ((x << 6) & 0xFFFFFFFF) + (x >> 2)) & 0xFFFFFFFF
    return ((x >> 8) & 1023) / 1023.0


def chest_render_state(state: dict[str, Any], now: float) -> dict[str, Any]:
    """Pure mapping: mirrored chest-LED mode → render parameters.

    Returns {on, brightness 0..1, rate (flicker steps/sec; 0 = static), fill
    (None, or 0..1 lighting the ladders bottom-up), ladder/squares (color
    palettes the lit LEDs draw from — the firmware's own colors), gauge (True →
    ladder LEDs use the charge-gauge gradient by position), charging, flash}.
    `now` is wall-clock time.time() — bridge timestamps are wall-clock.
    """
    mode = str(state.get("mode") or "off").strip().lower()
    emotion = str(state.get("emotion") or "").strip().lower()
    updated = float(state.get("updated_at") or 0.0)
    flash_at = float(state.get("flash_at") or 0.0)
    out: dict[str, Any] = {
        "on": True,
        "brightness": 1.0,
        "rate": 1.5,
        "fill": None,
        "ladder": _CHEST_SMALL_COLORS,
        "squares": _CHEST_BLOCK_COLORS,
        "gauge": False,
        "charging": False,
        "flash": 0.0 < (now - flash_at) < _CHEST_FLASH_SECS,
    }
    if mode == "off":
        out["on"] = False
    elif mode == "fadeoff":
        # Firmware ramps brightness to black over ~4s autonomously.
        remaining = 1.0 - (now - updated) / 4.0
        out["brightness"] = max(0.0, min(1.0, remaining))
        out["on"] = out["brightness"] > 0.0
    elif mode == "sleep":
        out.update(brightness=0.3, rate=0.3, fill=0.15)
    elif mode == "active":
        out["rate"] = 3.5
    elif mode == "startup":
        out.update(rate=6.0, fill=(now * 1.5) % 1.0)
    elif mode == "speak":
        ladder, squares = _CHEST_EMOTION_PALETTES.get(
            emotion, (_CHEST_SMALL_COLORS, _CHEST_BLOCK_COLORS)
        )
        out.update(
            ladder=ladder,
            squares=squares,
            rate=2.5 if emotion == "sad" else 9.0,
        )
    elif mode == "charge":
        soc = state.get("soc")
        out.update(
            rate=1.0,
            fill=max(0.0, min(1.0, (int(soc) if soc is not None else 0) / 100.0)),
            gauge=True,
            charging=bool(state.get("charging")),
        )
    # any unknown mode (incl. "idle") keeps the idle defaults above
    return out


def _neutral_norms() -> dict[str, float]:
    return {
        name: normalize_servo(name, cfg["neutral"])
        for name, cfg in config.SERVO_CHANNELS.items()
    }


def _boot_norms() -> dict[str, float]:
    """Powered-off pose: neutral everywhere except the visor, which is rolled fully
    down over the face (servo min → norm 0.0) exactly like the real robot at rest."""
    norms = _neutral_norms()
    norms["visor"] = 0.0
    return norms


def _eye_color(eye_state: dict[str, Any]) -> tuple[int, int, int]:
    value = eye_state.get("eye_color") or (0, 0, 0)
    if isinstance(value, dict):
        value = (value.get("r", 0), value.get("g", 0), value.get("b", 0))
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return (0, 0, 0)
    try:
        return tuple(max(0, min(255, int(v))) for v in value[:3])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return (0, 0, 0)


def _servo_name(name_or_channel) -> str | None:
    if isinstance(name_or_channel, str):
        lowered = name_or_channel.strip().lower()
        if lowered in config.SERVO_CHANNELS:
            return lowered
        if lowered.isdigit():
            name_or_channel = int(lowered)
        else:
            return None
    try:
        channel = int(name_or_channel)
    except (TypeError, ValueError):
        return None
    for name, cfg in config.SERVO_CHANNELS.items():
        if int(cfg["ch"]) == channel:
            return name
    return None
