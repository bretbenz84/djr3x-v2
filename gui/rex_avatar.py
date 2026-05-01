"""Simplified 2D Rex avatar for the optional dashboard."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QTransform
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


def servo_to_angle(name, value) -> float:
    """Map a servo value to a dashboard-friendly visual angle in degrees."""
    name = _servo_name(name) or str(name)
    norm = normalize_servo(name, value)
    mapping = {
        "neck": (-35.0, 35.0),
        "headtilt": (18.0, -18.0),
        "visor": (-18.0, 18.0),
        "heroarm": (-55.0, 55.0),
        "elbow": (55.0, -45.0),
        "hand": (-60.0, 60.0),
        "pokerarm": (-25.0, 25.0),
    }
    lo, hi = mapping.get(name, (-30.0, 30.0))
    return lo + (hi - lo) * norm


def servo_to_offset(name, value) -> float:
    """Map a servo value to a compact visual offset in pixels."""
    name = _servo_name(name) or str(name)
    norm = normalize_servo(name, value)
    if name == "headlift":
        return (0.5 - norm) * 58.0
    if name == "neck":
        return (norm - 0.5) * 42.0
    if name == "visor":
        return norm * 22.0
    return (norm - 0.5) * 30.0


class RexAvatar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._target: dict[str, float] = _neutral_norms()
        self._current: dict[str, float] = dict(self._target)
        self._last_paint = time.monotonic()
        self._eye_state: dict[str, Any] = {
            "mode": "off",
            "eye_color": (0, 0, 0),
            "eyes_active": False,
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
        self._sprites = _load_sprites()
        self.setMinimumSize(430, 470)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        ws = snapshot.get("world_state") or {}
        self_state = ws.get("self_state") or ws.get("self") or {}
        positions = dict(snapshot.get("servo_positions") or {})
        positions.update(self_state.get("servo_positions") or {})
        for name, cfg in config.SERVO_CHANNELS.items():
            value = positions.get(name, cfg["neutral"])
            self._target[name] = normalize_servo(name, value)
        eye_state = snapshot.get("head_led_state") or {}
        if eye_state:
            self._eye_state.update(eye_state)
            event_at = float(eye_state.get("updated_at") or 0.0)
            if event_at != self._last_eye_event_at:
                self._last_eye_event_at = event_at
                self._reset_blink_cycle()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        self._smooth()
        self._tick_eye_animation()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07111a"))
        self._draw_grid(painter)

        if self._sprites:
            self._draw_sprite_avatar(painter)
            painter.end()
            return

        w = self.width()
        h = self.height()
        cx = w * 0.48 + servo_to_offset("neck", self._value("neck")) * 0.35
        base_y = h * 0.50 + servo_to_offset("headlift", self._value("headlift")) * 0.55

        self._draw_base(painter, cx, base_y)
        self._draw_body(painter, cx, base_y)
        self._draw_left_arm(painter, cx, base_y)
        self._draw_right_arm(painter, cx, base_y)
        self._draw_head(painter, cx, base_y)

        painter.end()

    def _draw_sprite_avatar(self, painter: QPainter) -> None:
        w = self.width()
        h = self.height()
        cx = w * 0.50
        body = self._sprites["body"]
        head = self._sprites["head"]
        right_arm = self._sprites.get("right_arm")
        poker_arm = self._sprites.get("poker_arm")

        body_h = min(h * 0.76, w * 1.08)
        body_w = body_h * body.width() / max(1, body.height())
        body_rect = QRectF(cx - body_w / 2.0, h * 0.14, body_w, body_h)

        neck_norm = self._current.get("neck", 0.5)
        lift = servo_to_offset("headlift", self._value("headlift")) * 0.78
        pitch = servo_to_angle("headtilt", self._value("headtilt")) / 18.0
        neck_x = (neck_norm - 0.5) * w * 0.13
        head_y_shift = lift + pitch * 22.0
        head_squash = 1.0 - abs(pitch) * 0.12
        head_shear = pitch * 0.05

        if poker_arm is not None:
            self._draw_pivoted_sprite(
                painter,
                poker_arm,
                QPointF(body_rect.left() + body_rect.width() * 0.52, body_rect.top() + body_rect.height() * 0.31),
                QPointF(30, 120),
                body_rect.width() * 0.92,
                servo_to_angle("pokerarm", self._value("pokerarm")) * 0.55 - 8,
            )
        if right_arm is not None:
            self._draw_pivoted_sprite(
                painter,
                right_arm,
                QPointF(body_rect.left() + body_rect.width() * 0.67, body_rect.top() + body_rect.height() * 0.23),
                QPointF(60, 155),
                body_rect.width() * 0.52,
                servo_to_angle("heroarm", self._value("heroarm")) * 0.55 - 10,
            )

        painter.drawPixmap(body_rect, body, QRectF(body.rect()))

        neck_top = QPointF(cx + neck_x, body_rect.top() + body_rect.height() * 0.15 + lift * 0.35)
        neck_bottom = QPointF(cx, body_rect.top() + body_rect.height() * 0.28)
        painter.setPen(QPen(QColor("#15181b"), 13, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(neck_bottom, neck_top)
        painter.setPen(QPen(QColor("#30363b"), 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(neck_bottom, neck_top)

        head_w = body_rect.width() * 0.72
        head_h = head_w * head.height() / max(1, head.width()) * head_squash
        head_cx = cx + neck_x + servo_to_offset("neck", self._value("neck")) * 0.45
        head_rect = QRectF(
            head_cx - head_w / 2.0,
            body_rect.top() - head_h * 0.30 + head_y_shift,
            head_w,
            head_h,
        )

        painter.save()
        painter.translate(head_rect.center())
        painter.shear(head_shear, 0.0)
        painter.drawPixmap(
            QRectF(-head_rect.width() / 2.0, -head_rect.height() / 2.0, head_rect.width(), head_rect.height()),
            head,
            QRectF(head.rect()),
        )
        self._draw_visor_overlay(painter, head_rect.size().width(), head_rect.size().height())
        self._draw_eye_overlay(painter, head_rect.size().width(), head_rect.size().height())
        painter.restore()

    def _draw_pivoted_sprite(
        self,
        painter: QPainter,
        sprite: QPixmap,
        target_pivot: QPointF,
        source_pivot: QPointF,
        target_width: float,
        angle: float,
    ) -> None:
        scale = target_width / max(1, sprite.width())
        painter.save()
        painter.translate(target_pivot)
        painter.rotate(angle)
        painter.scale(scale, scale)
        painter.drawPixmap(QPointF(-source_pivot.x(), -source_pivot.y()), sprite)
        painter.restore()

    def _draw_visor_overlay(self, painter: QPainter, head_w: float, head_h: float) -> None:
        visor_open = normalize_servo("visor", self._value("visor"))
        closed = 1.0 - visor_open
        visor_y = -head_h * 0.39 + closed * head_h * 0.18
        visor_rect = QRectF(-head_w * 0.30, visor_y, head_w * 0.60, head_h * 0.20)

        painter.setPen(QPen(QColor("#8b4216"), 2))
        painter.setBrush(QColor(218, 122, 31, 235))
        painter.drawChord(visor_rect, 0, 180 * 16)

        painter.setPen(QPen(QColor(255, 226, 197, 185), 2))
        for offset in (-0.18, -0.09, 0.0, 0.09, 0.18):
            x = offset * head_w
            painter.drawLine(
                QPointF(x, visor_y + head_h * 0.02),
                QPointF(x - head_w * 0.025, visor_y + head_h * 0.16),
            )

    def _draw_eye_overlay(self, painter: QPainter, head_w: float, head_h: float) -> None:
        color = _eye_color(self._eye_state)
        active = bool(self._eye_state.get("eyes_active")) and any(color)
        openness = 1.0 if active and self._blink_state != "closed" else 0.0
        brightness = self._eye_brightness() if active else 0.0
        radius_x = head_w * 0.043
        radius_y = head_h * 0.056
        centers = (
            QPointF(-head_w * 0.113, -head_h * 0.095),
            QPointF(head_w * 0.113, -head_h * 0.095),
        )
        painter.setPen(QPen(QColor("#08101c"), 2))
        for center in centers:
            painter.setBrush(QColor(5, 8, 12, 235))
            painter.drawEllipse(center, radius_x * 1.25, radius_y * 1.25)
            if openness <= 0.0:
                painter.setPen(QPen(QColor("#38414b"), 2))
                painter.drawLine(
                    QPointF(center.x() - radius_x * 1.1, center.y()),
                    QPointF(center.x() + radius_x * 1.1, center.y()),
                )
                painter.setPen(QPen(QColor("#08101c"), 2))
                continue
            r, g, b = (int(v * brightness) for v in color)
            painter.setBrush(QColor(r, g, b, 245))
            painter.drawEllipse(center, radius_x, radius_y)
            painter.setPen(QPen(QColor(min(255, r + 70), min(255, g + 70), min(255, b + 70), 210), 1))
            for shrink in (0.45, 0.75):
                painter.drawEllipse(center, radius_x * shrink, radius_y * shrink)
            painter.setPen(QPen(QColor("#08101c"), 2))

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

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(27, 50, 70, 105), 1))
        step = 18
        for x in range(14, self.width(), step):
            painter.drawLine(x, 14, x, self.height() - 14)
        for y in range(14, self.height(), step):
            painter.drawLine(14, y, self.width() - 14, y)

    def _draw_base(self, painter: QPainter, cx: float, base_y: float) -> None:
        base_top = base_y + 95
        painter.setPen(QPen(QColor("#3b2c1b"), 2))
        painter.setBrush(QColor("#b5631d"))
        painter.drawEllipse(QPointF(cx, base_top + 22), 96, 28)
        painter.setBrush(QColor("#d77a23"))
        painter.drawRoundedRect(QRectF(cx - 88, base_top - 7, 176, 34), 10, 10)
        painter.setBrush(QColor("#2d3033"))
        painter.drawEllipse(QPointF(cx, base_top + 8), 22, 22)
        painter.setPen(QPen(QColor("#7b8790"), 2))
        for i in range(12):
            angle = math.radians(i * 30)
            painter.drawLine(
                QPointF(cx + math.cos(angle) * 8, base_top + 8 + math.sin(angle) * 8),
                QPointF(cx + math.cos(angle) * 18, base_top + 8 + math.sin(angle) * 18),
            )

    def _smooth(self) -> None:
        smoothing = max(0.01, min(1.0, float(getattr(config, "GUI_AVATAR_SMOOTHING", 0.25))))
        for name, target in self._target.items():
            current = self._current.get(name, target)
            self._current[name] = current + (target - current) * smoothing

    def _value(self, name: str) -> int:
        cfg = config.SERVO_CHANNELS[name]
        norm = self._current.get(name, 0.5)
        return int(cfg["min"] + (cfg["max"] - cfg["min"]) * norm)

    def _draw_body(self, painter: QPainter, cx: float, base_y: float) -> None:
        body = QRectF(cx - 60, base_y - 10, 120, 110)
        painter.setPen(QPen(QColor("#1b2025"), 2))
        painter.setBrush(QColor("#c46b1f"))
        painter.drawRoundedRect(body, 18, 18)
        painter.setBrush(QColor("#202429"))
        painter.drawRoundedRect(QRectF(cx - 66, base_y + 16, 132, 50), 9, 9)
        painter.setPen(QPen(QColor("#9da7ad"), 4))
        for x in range(-42, 43, 16):
            painter.drawLine(QPointF(cx + x, base_y + 20), QPointF(cx + x, base_y + 62))
        painter.setPen(QPen(QColor("#5d6d78"), 2))
        painter.setBrush(QColor("#14191d"))
        painter.drawRoundedRect(QRectF(cx - 36, base_y - 1, 72, 24), 5, 5)
        painter.setPen(QPen(QColor("#3e9bff"), 2))
        painter.drawText(QRectF(cx - 30, base_y + 1, 60, 19), Qt.AlignmentFlag.AlignCenter, "7A11")
        painter.setPen(QPen(QColor("#6e7b8f"), 8))
        painter.drawLine(QPointF(cx, base_y - 8), QPointF(cx, base_y - 42))

    def _draw_head(self, painter: QPainter, cx: float, base_y: float) -> None:
        neck_angle = servo_to_angle("neck", self._value("neck"))
        tilt_angle = servo_to_angle("headtilt", self._value("headtilt"))
        head_x = cx + math.sin(math.radians(neck_angle)) * 28.0
        head_y = base_y - 94

        painter.save()
        painter.translate(head_x, head_y)
        painter.rotate(tilt_angle)

        head = QRectF(-78, -52, 156, 94)
        painter.setPen(QPen(QColor("#222930"), 2))
        painter.setBrush(QColor("#2e3338"))
        painter.drawRoundedRect(head, 14, 14)
        painter.setBrush(QColor("#d87920"))
        painter.drawChord(QRectF(-70, -70, 140, 76), 0, 180 * 16)
        painter.setPen(QPen(QColor("#f5d2b4"), 3))
        for x in range(-42, 43, 18):
            painter.drawLine(QPointF(x, -63), QPointF(x - 8, -24))

        painter.setBrush(QColor("#384455"))
        painter.drawRoundedRect(QRectF(-56, -25, 112, 36), 8, 8)

        visor_open = normalize_servo("visor", self._value("visor"))
        shutter_y = -20 - visor_open * 23
        painter.setBrush(QColor("#0b1118"))
        painter.drawRoundedRect(QRectF(-52, shutter_y, 104, 18), 6, 6)
        painter.setPen(QPen(QColor("#17365a"), 3))
        painter.setBrush(QColor("#1a63e6"))
        painter.drawEllipse(QPointF(-28, -7), 14, 14)
        painter.drawEllipse(QPointF(28, -7), 14, 14)
        painter.setPen(QPen(QColor("#7dbdff"), 1))
        for r in (5, 9):
            painter.drawEllipse(QPointF(-28, -7), r, r)
            painter.drawEllipse(QPointF(28, -7), r, r)

        painter.setPen(QPen(QColor("#7c8796"), 4))
        painter.drawLine(QPointF(-42, 30), QPointF(42, 30))
        painter.setPen(QPen(QColor("#5e6978"), 2))
        for x in range(-35, 41, 14):
            painter.drawLine(QPointF(x, 24), QPointF(x, 36))

        painter.restore()

    def _draw_right_arm(self, painter: QPainter, cx: float, base_y: float) -> None:
        shoulder = QPointF(cx + 68, base_y + 16)
        hero = math.radians(servo_to_angle("heroarm", self._value("heroarm")) + 25)
        elbow_angle = math.radians(servo_to_angle("elbow", self._value("elbow")))
        upper_len = 58
        lower_len = 50
        elbow = QPointF(
            shoulder.x() + math.cos(hero) * upper_len,
            shoulder.y() + math.sin(hero) * upper_len,
        )
        wrist_angle = hero + elbow_angle
        wrist = QPointF(
            elbow.x() + math.cos(wrist_angle) * lower_len,
            elbow.y() + math.sin(wrist_angle) * lower_len,
        )
        self._draw_limb(painter, shoulder, elbow, wrist, QColor("#d77a23"))
        painter.save()
        painter.translate(wrist)
        painter.rotate(servo_to_angle("hand", self._value("hand")))
        painter.setPen(QPen(QColor("#11161a"), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for angle in (-35, 0, 35):
            painter.save()
            painter.rotate(angle)
            painter.drawLine(QPointF(0, 0), QPointF(0, -34))
            painter.restore()
        painter.restore()

    def _draw_left_arm(self, painter: QPainter, cx: float, base_y: float) -> None:
        shoulder = QPointF(cx - 68, base_y + 24)
        angle = math.radians(180 + servo_to_angle("pokerarm", self._value("pokerarm")) * 0.8)
        elbow = QPointF(shoulder.x() + math.cos(angle) * 48, shoulder.y() + math.sin(angle) * 48)
        wrist = QPointF(elbow.x() - 36, elbow.y() + 14)
        self._draw_limb(painter, shoulder, elbow, wrist, QColor("#9aa5ad"))

    def _draw_limb(
        self,
        painter: QPainter,
        shoulder: QPointF,
        elbow: QPointF,
        wrist: QPointF,
        color: QColor,
    ) -> None:
        painter.setPen(QPen(QColor("#20262b"), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(shoulder, elbow)
        painter.drawLine(elbow, wrist)
        painter.setPen(QPen(color, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(shoulder, elbow)
        painter.drawLine(elbow, wrist)
        painter.setBrush(QColor("#3a4249"))
        painter.setPen(QPen(QColor("#101418"), 2))
        for point in (shoulder, elbow, wrist):
            painter.drawEllipse(point, 6, 6)

    def _draw_servo_values(self, painter: QPainter) -> None:
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor("#9fb0c4"))
        lines = []
        for name in ("neck", "headlift", "headtilt", "visor", "heroarm", "elbow", "hand", "pokerarm"):
            lines.append(f"{name}: {self._value(name)}")
        painter.drawText(
            QRectF(12, 12, 140, self.height() - 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "\n".join(lines),
        )


def _neutral_norms() -> dict[str, float]:
    return {
        name: normalize_servo(name, cfg["neutral"])
        for name, cfg in config.SERVO_CHANNELS.items()
    }


def _load_sprites() -> dict[str, QPixmap]:
    root = Path(__file__).resolve().parent.parent / "assets" / "gui"
    paths = {
        "head": root / "rex_head.png",
        "body": root / "rex_body_base.png",
        "right_arm": root / "rex_right_arm.png",
        "poker_arm": root / "rex_poker_arm.png",
    }
    sprites = {name: QPixmap(str(path)) for name, path in paths.items() if path.exists()}
    if not sprites.get("head") or sprites["head"].isNull():
        return {}
    if not sprites.get("body") or sprites["body"].isNull():
        return {}
    return {name: pix for name, pix in sprites.items() if not pix.isNull()}


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
