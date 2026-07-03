"""Articulated 2D DJ R-3X avatar for the optional dashboard.

Drawn procedurally with QPainter (no sprites) to match the real robot: striped
orange dome visor that slides over the eyes, blue carry handle and ear pods,
binocular LED eyes, ribbed vocoder chin, coil-spring neck on a lift pole, the
orange ring / black bellows / ribbed drum / flared bell torso stack on a domed
base — plus the two articulated arms (hero arm with elbow+claw at the top of the
torso, poker arm at the base).
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

import config

# Design-space canvas the robot is drawn in; scaled to fit the widget.
_DW = 460.0
_DH = 640.0

# Palette sampled from the reference figure.
_ORANGE = QColor("#d97a1f")
_ORANGE_HI = QColor("#f09a3e")
_ORANGE_LO = QColor("#a2551a")
_ORANGE_EDGE = QColor("#6f3a10")
_CREAM = QColor("#e9e2d4")
_SILVER = QColor("#9aa0a6")
_SILVER_HI = QColor("#c3c8cc")
_SILVER_LO = QColor("#5f656b")
_GUNMETAL = QColor("#3a3f44")
_DARK = QColor("#23272b")
_NEAR_BLACK = QColor("#15181b")
_BLUE = QColor("#2e5f9d")
_BLUE_DK = QColor("#1d3f6e")


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
    if name == "visor":
        return norm * 22.0
    return (norm - 0.5) * 30.0


def servo_to_yaw(name, value) -> float:
    """Map neck servo value to -1.0..1.0 yaw for front-view perspective."""
    if _servo_name(name) != "neck":
        return 0.0
    return max(-1.0, min(1.0, (normalize_servo("neck", value) - 0.5) * 2.0))


class RexAvatar(QWidget):
    def __init__(self, parent=None, *, show_background: bool = True, show_grid: bool = True) -> None:
        super().__init__(parent)
        self._show_background = bool(show_background)
        self._show_grid = bool(show_grid)
        self._target: dict[str, float] = _neutral_norms()
        self._current: dict[str, float] = dict(self._target)
        self._last_paint = time.monotonic()
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
        self._last_eye_event_at = 0.0
        self._blink_state = "open"
        self._blink_timer = time.monotonic()
        self._blink_interval = random.uniform(2.0, 8.0)
        self._blink_duration = 0.0
        self._is_second_blink = False
        self._idle_phase = 0.0
        self._mouth_phase = 0.0
        self._last_blink_tick = time.monotonic()
        if not self._show_background:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAutoFillBackground(False)
        if self._show_background:
            # Painting scales proportionally, so a soft minimum + preferred
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
        speech_state = snapshot.get("speech_state") or {}
        if speech_state:
            self._speech_state.update(speech_state)
        self.update()

    # ── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        self._smooth()
        self._tick_eye_animation()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._show_background:
            painter.fillRect(self.rect(), QColor("#07111a"))
        if self._show_grid:
            self._draw_grid(painter)
        self._draw_robot(painter)
        painter.end()

    def _draw_robot(self, painter: QPainter) -> None:
        w = float(self.width())
        h = float(self.height())
        fill = 1.00 if not self._show_background else 0.94
        s = min(w / _DW, h / _DH) * fill
        painter.save()
        painter.translate(w / 2.0, (h - _DH * s) / 2.0)
        painter.scale(s, s)
        # Design coords: x centered on 0, y 0..640 top→bottom.

        speaking = self._is_speaking()
        speech_bob = (math.sin(self._mouth_phase * 0.72) * 3.0) if speaking else 0.0
        # servo_to_offset: max lift → -29 (head UP), min → +29 (head DOWN).
        lift = servo_to_offset("headlift", self._value("headlift"))
        neck_top_y = 224.0 + lift * 0.85 + speech_bob

        self._draw_base(painter)
        self._draw_torso(painter)
        self._draw_poker_arm(painter)
        self._draw_middle_arm(painter)
        self._draw_hero_arm(painter)
        self._draw_neck(painter, neck_top_y)
        self._draw_head(painter, neck_top_y)
        painter.restore()

    # ── Base + torso ────────────────────────────────────────────────────────

    def _draw_base(self, painter: QPainter) -> None:
        # Flared dome foot with round ports and a center vent, on a squat ring.
        grad = QLinearGradient(0, 505, 0, 620)
        grad.setColorAt(0.0, _ORANGE_HI)
        grad.setColorAt(0.55, _ORANGE)
        grad.setColorAt(1.0, _ORANGE_LO)
        painter.setPen(QPen(_ORANGE_EDGE, 3))
        painter.setBrush(grad)
        dome = QPainterPath()
        dome.moveTo(-150, 596)
        dome.cubicTo(-150, 520, -70, 502, 0, 502)
        dome.cubicTo(70, 502, 150, 520, 150, 596)
        dome.closeSubpath()
        painter.drawPath(dome)
        # bottom ring
        painter.setBrush(QColor("#b3641c"))
        painter.drawRoundedRect(QRectF(-128, 592, 256, 26), 9, 9)
        painter.setBrush(_ORANGE_LO)
        painter.drawRoundedRect(QRectF(-112, 614, 224, 10), 5, 5)

        # round ports (angled onto the dome slope)
        painter.setPen(QPen(_NEAR_BLACK, 2))
        for x, y, rx, ry in ((-96, 549, 17, 14), (96, 549, 17, 14), (130, 585, 10, 12)):
            painter.setBrush(QColor("#41464b"))
            painter.drawEllipse(QPointF(x, y), rx, ry)
            painter.setBrush(QColor("#23272b"))
            painter.drawEllipse(QPointF(x, y), rx * 0.62, ry * 0.62)

        # center vent with concentric rings + bracket greebles
        painter.setBrush(QColor("#4b5054"))
        painter.drawEllipse(QPointF(0, 568), 24, 22)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#8d959b"), 2))
        for r in (6, 11, 16):
            painter.drawEllipse(QPointF(0, 568), r, r * 0.92)
        painter.setPen(QPen(QColor("#e8ddc9"), 3))
        for sx in (-1, 1):
            x0 = sx * 44
            painter.drawLine(QPointF(x0, 556), QPointF(x0 + sx * 12, 556))
            painter.drawLine(QPointF(x0, 556), QPointF(x0, 580))
            painter.drawLine(QPointF(x0, 580), QPointF(x0 + sx * 12, 580))
        # small hatch high on the dome
        painter.setPen(QPen(_ORANGE_EDGE, 2))
        painter.setBrush(QColor("#c06a1c"))
        painter.drawRoundedRect(QRectF(-17, 518, 34, 13), 3, 3)

    def _draw_torso(self, painter: QPainter) -> None:
        # Grey pedestal column between bell and base.
        grad = QLinearGradient(-28, 0, 28, 0)
        grad.setColorAt(0.0, QColor("#565b60"))
        grad.setColorAt(0.5, QColor("#7d838a"))
        grad.setColorAt(1.0, QColor("#4a4f54"))
        painter.setPen(QPen(_NEAR_BLACK, 2))
        painter.setBrush(grad)
        painter.drawRect(QRectF(-28, 452, 56, 78))
        painter.setPen(QPen(QColor("#2c3034"), 2))
        painter.setBrush(QColor("#5d6268"))
        painter.drawRoundedRect(QRectF(-13, 492, 26, 26), 3, 3)
        painter.setBrush(QColor("#33383c"))
        painter.drawEllipse(QPointF(0, 481), 4, 4)

        # Flared orange bell.
        bell = QPainterPath()
        bell.moveTo(-75, 424)
        bell.cubicTo(-82, 452, -112, 462, -118, 478)
        bell.lineTo(118, 478)
        bell.cubicTo(112, 462, 82, 452, 75, 424)
        bell.closeSubpath()
        grad = QLinearGradient(0, 424, 0, 478)
        grad.setColorAt(0.0, _ORANGE_HI)
        grad.setColorAt(1.0, _ORANGE_LO)
        painter.setPen(QPen(_ORANGE_EDGE, 3))
        painter.setBrush(grad)
        painter.drawPath(bell)
        painter.setPen(QPen(_ORANGE_EDGE, 2))
        painter.setBrush(QColor("#c06a1c"))
        painter.drawRoundedRect(QRectF(-21, 448, 42, 16), 3, 3)

        # Ribbed grey drum.
        grad = QLinearGradient(-105, 0, 105, 0)
        grad.setColorAt(0.0, QColor("#565c62"))
        grad.setColorAt(0.5, QColor("#8b9198"))
        grad.setColorAt(1.0, QColor("#4b5157"))
        painter.setPen(QPen(_NEAR_BLACK, 3))
        painter.setBrush(grad)
        painter.drawRoundedRect(QRectF(-105, 378, 210, 48), 10, 10)
        painter.setPen(QPen(QColor("#c9cdd1"), 4))
        for x in range(-88, 89, 16):
            painter.drawLine(QPointF(x, 386), QPointF(x, 418))
        painter.setPen(QPen(QColor("#2c3034"), 2))
        painter.setBrush(QColor("#787f86"))
        painter.drawRoundedRect(QRectF(54, 388, 34, 28), 4, 4)

        # Black accordion bellows.
        painter.setPen(Qt.PenStyle.NoPen)
        y = 336.0
        for i in range(6):
            wobble = 96 + (4 if i % 2 else 0)
            painter.setBrush(QColor("#191c1f") if i % 2 else QColor("#26292d"))
            painter.drawRoundedRect(QRectF(-wobble, y, wobble * 2, 9.4), 4.5, 4.5)
            y += 7.2
        # Upper orange ring with the blue glyph plate.
        grad = QLinearGradient(0, 300, 0, 340)
        grad.setColorAt(0.0, _ORANGE_HI)
        grad.setColorAt(1.0, _ORANGE_LO)
        painter.setPen(QPen(_ORANGE_EDGE, 3))
        painter.setBrush(grad)
        painter.drawRoundedRect(QRectF(-85, 300, 170, 40), 12, 12)
        painter.setPen(QPen(QColor("#0c1116"), 2))
        painter.setBrush(QColor("#0d1319"))
        painter.drawRoundedRect(QRectF(-46, 308, 92, 22), 4, 4)
        painter.setPen(QPen(QColor("#57a8ff"), 2))
        font = painter.font()
        font.setPointSizeF(13.0)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(-46, 308, 92, 22), Qt.AlignmentFlag.AlignCenter, "7Δ4⊪")

        # Grey platter the neck spring sits on.
        grad = QLinearGradient(0, 288, 0, 302)
        grad.setColorAt(0.0, QColor("#a7abb0"))
        grad.setColorAt(1.0, QColor("#5c6167"))
        painter.setPen(QPen(_NEAR_BLACK, 2))
        painter.setBrush(grad)
        painter.drawEllipse(QPointF(0, 296), 92, 15)
        painter.setBrush(QColor("#6d7378"))
        painter.drawEllipse(QPointF(0, 292), 70, 10)

    # ── Neck + head ─────────────────────────────────────────────────────────

    def _draw_neck(self, painter: QPainter, neck_top_y: float) -> None:
        # Lift pole.
        painter.setPen(QPen(QColor("#2b2f33"), 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(0, 292), QPointF(0, neck_top_y + 6))
        painter.setPen(QPen(QColor("#7d838a"), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(0, 292), QPointF(0, neck_top_y + 6))
        # Coil spring stretches with the lift.
        top = neck_top_y + 18
        bottom = 290.0
        coils = 5
        span = max(10.0, bottom - top)
        painter.setPen(QPen(QColor("#1a1d20"), 6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(coils):
            cy = top + span * (i + 0.5) / coils
            painter.drawEllipse(QPointF(0, cy), 24, max(4.0, span / coils * 0.44))

    def _draw_head(self, painter: QPainter, neck_top_y: float) -> None:
        # headtilt is PITCH (looking down at his toes / up at the sky), not a roll:
        # front-on it reads as the face features sliding down/up with a vertical
        # foreshorten squash. pitch +1 = looking down, -1 = looking up. Physical
        # convention (consciousness gaze search): tilt servo MAX = looking DOWN,
        # while servo_to_angle maps max → -18 — hence the negation.
        pitch = -servo_to_angle("headtilt", self._value("headtilt")) / 18.0
        # The avatar faces the camera, so his right = screen LEFT → mirror the yaw.
        yaw = -servo_to_yaw("neck", self._value("neck"))
        yaw_scale = 1.0 - abs(yaw) * 0.16
        yaw_shear = yaw * 0.10
        face_shift = yaw * 13.0
        face_dy = pitch * 15.0
        visor_open = normalize_servo("visor", self._value("visor"))
        # The visor is a full dome shell riding OUTSIDE the crown: rolled up it shows
        # only its top edge above the head; rolled fully down it covers the whole face
        # (robot off), with just the vocoder poking out below.
        visor_drop = (1.0 - visor_open) * 82.0

        painter.save()
        painter.translate(0, neck_top_y + pitch * 8.0)
        painter.scale(yaw_scale, 1.0 - abs(pitch) * 0.12)
        painter.shear(yaw_shear, 0.0)

        # Ear pods (behind the face plate).
        for sx in (-1, 1):
            cx = sx * 104
            grad = QLinearGradient(cx - 20, 0, cx + 20, 0)
            grad.setColorAt(0.0, QColor("#3a6cb0"))
            grad.setColorAt(1.0, _BLUE_DK)
            painter.setPen(QPen(QColor("#122c4e"), 2))
            painter.setBrush(grad)
            painter.drawEllipse(QPointF(cx, -52), 25, 37)
            painter.setPen(QPen(QColor("#0f2340"), 2))
            for i in range(5):
                yy = -76 + i * 12
                half = 24 * math.sqrt(max(0.05, 1 - ((yy + 52) / 37.0) ** 2))
                painter.drawLine(QPointF(cx - half, yy), QPointF(cx + half, yy))
            painter.setPen(QPen(_NEAR_BLACK, 2))
            painter.setBrush(QColor("#5b6167"))
            painter.drawEllipse(QPointF(sx * 121, -52), 8, 24)

        # Crown dome — same silver as the forehead and CONNECTED to it (one shell):
        # the chord's flat bottom lands just under the face-plate top so the two
        # read as a single continuous head. Side caps belong to this shell.
        for sx in (-1, 1):
            cap = QPainterPath()
            cap.moveTo(sx * 66, -122)
            cap.lineTo(sx * 100, -122)
            cap.lineTo(sx * 88, -148)
            cap.closeSubpath()
            painter.setPen(QPen(QColor("#26292d"), 2))
            painter.setBrush(QColor("#4a4f54"))
            painter.drawPath(cap)
        # Flat, low crown — the real head is squat; the dome is a shallow curve.
        crown_rect = QRectF(-97, -160, 194, 124)
        grad = QLinearGradient(0, -160, 0, -98)
        grad.setColorAt(0.0, _SILVER_HI)
        grad.setColorAt(0.6, _SILVER)
        grad.setColorAt(1.0, _SILVER_LO)
        painter.setPen(QPen(QColor("#25292d"), 3))
        painter.setBrush(grad)
        painter.drawChord(crown_rect, 0, 180 * 16)

        # Face plate.
        grad = QLinearGradient(0, -100, 0, 0)
        grad.setColorAt(0.0, _SILVER_HI)
        grad.setColorAt(0.5, _SILVER)
        grad.setColorAt(1.0, _SILVER_LO)
        painter.setPen(QPen(QColor("#25292d"), 3))
        painter.setBrush(grad)
        painter.drawRoundedRect(QRectF(-95, -102, 190, 102), 20, 20)
        # darker slate band around the eyes
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(58, 63, 68, 220))
        painter.drawRoundedRect(QRectF(-82, -88, 164, 62), 14, 14)
        # chin lip
        painter.setBrush(QColor("#a7abb0"))
        painter.drawRoundedRect(QRectF(-56, -16 + face_dy * 0.6, 112, 13), 5, 5)

        # Eyes.
        self._draw_eyes(painter, face_shift, face_dy)

        # Nose vent.
        nose = QPainterPath()
        nose.moveTo(-9 + face_shift, -34 + face_dy)
        nose.lineTo(9 + face_shift, -34 + face_dy)
        nose.lineTo(14 + face_shift, -14 + face_dy)
        nose.lineTo(-14 + face_shift, -14 + face_dy)
        nose.closeSubpath()
        painter.setPen(QPen(QColor("#33383c"), 2))
        painter.setBrush(QColor("#6d7378"))
        painter.drawPath(nose)

        # Vocoder chin (speaking EQ lives here).
        self._draw_vocoder(painter, face_shift, face_dy)

        # Orange visor: a FULL dome shell riding OUTSIDE the crown (see the real
        # robot). Rolled up, only its top edge shows above the head (the crescent
        # look); rolled fully down it slides in front of the whole face — eyes
        # hidden, vocoder poking out below. Bottom edge bows down slightly.
        painter.save()
        painter.translate(0, visor_drop + face_dy * 0.45)
        outer = QRectF(-104, -172, 208, 108)   # shallow top arc; ends at (±104, -118)
        visor = QPainterPath()
        visor.arcMoveTo(outer, 180)
        visor.arcTo(outer, 180, -180)          # over the crown, left end → right end
        visor.quadTo(0, -100, -104, -118)      # gently down-bowed bottom edge
        visor.closeSubpath()
        grad = QLinearGradient(0, -172, 0, -100)
        grad.setColorAt(0.0, _ORANGE_HI)
        grad.setColorAt(1.0, _ORANGE_LO)
        painter.setPen(QPen(_ORANGE_EDGE, 2.5))
        painter.setBrush(grad)
        painter.drawPath(visor)
        painter.restore()

        # Blue carry handle — WIDE, square-sided, rooted at the ear muffs (posts at
        # the ear-pod centerlines), with the orange visor fitting INSIDE the frame.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        handle = QPainterPath()
        handle.moveTo(-104, -92)
        handle.lineTo(-104, -186)
        handle.quadTo(0, -204, 104, -186)
        handle.lineTo(104, -92)
        pen = QPen(_BLUE_DK, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.drawPath(handle)
        pen = QPen(_BLUE, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.drawPath(handle)
        # grey corner brackets at the squared shoulders (see figure)
        painter.setPen(QPen(QColor("#2b2f33"), 2))
        painter.setBrush(QColor("#5b6167"))
        for sx in (-1, 1):
            painter.drawRoundedRect(QRectF(sx * 104 - 9, -194, 18, 16), 3, 3)

        painter.restore()

    def _draw_eyes(self, painter: QPainter, face_shift: float, face_dy: float = 0.0) -> None:
        color = _eye_color(self._eye_state)
        active = bool(self._eye_state.get("eyes_active")) and any(color)
        open_eye = active and self._blink_state != "closed"
        brightness = self._eye_brightness() if active else 0.0
        for sx in (-1, 1):
            center = QPointF(sx * 42 + face_shift, -58 + face_dy)
            painter.setPen(QPen(QColor("#2b2f33"), 2))
            painter.setBrush(QColor("#b9bdc1"))
            painter.drawEllipse(center, 27, 27)
            painter.setBrush(QColor("#14181c"))
            painter.drawEllipse(center, 21, 21)
            if not open_eye:
                painter.setPen(QPen(QColor("#3f464d"), 3))
                painter.drawLine(
                    QPointF(center.x() - 13, center.y()), QPointF(center.x() + 13, center.y())
                )
                continue
            r, g, b = (int(v * brightness) for v in color)
            glow = QRadialGradient(center, 18)
            glow.setColorAt(0.0, QColor(min(255, r + 130), min(255, g + 130), min(255, b + 130)))
            glow.setColorAt(0.55, QColor(r, g, b))
            glow.setColorAt(1.0, QColor(int(r * 0.25), int(g * 0.25), int(b * 0.35)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(center, 17, 17)
            painter.setPen(QPen(QColor(min(255, r + 70), min(255, g + 70), min(255, b + 70), 200), 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for shrink in (0.42, 0.72):
                painter.drawEllipse(center, 17 * shrink, 17 * shrink)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 215))
            painter.drawEllipse(QPointF(center.x() - 5, center.y() - 6), 3.4, 3.4)

    def _draw_vocoder(self, painter: QPainter, face_shift: float, face_dy: float = 0.0) -> None:
        speaking = self._is_speaking()
        rect = QRectF(-27 + face_shift, -12 + face_dy, 54, 46)
        painter.setPen(QPen(QColor("#0c0e10"), 3))
        painter.setBrush(QColor("#1c1f22"))
        painter.drawRoundedRect(rect, 9, 9)
        rib_count = 5
        gap = rect.height() * 0.055
        rib_h = (rect.height() - gap * (rib_count + 1)) / rib_count
        for idx in range(rib_count):
            phase = math.sin(self._mouth_phase + idx * 0.92)
            if speaking:
                level = 0.55 + 0.45 * phase
                alpha = int(70 + 165 * max(0.0, min(1.0, level)))
                color = QColor(72, 160, 255, alpha) if idx % 2 else QColor(255, 166, 48, alpha)
                rib_w = rect.width() * (0.58 + 0.30 * max(0.0, phase))
            else:
                color = QColor("#34383c")
                rib_w = rect.width() * 0.78
            rib = QRectF(
                rect.center().x() - rib_w / 2.0,
                rect.top() + gap + idx * (rib_h + gap),
                rib_w,
                max(1.5, rib_h),
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(rib, rib_h * 0.4, rib_h * 0.4)

    # ── Arms ────────────────────────────────────────────────────────────────

    def _draw_hero_arm(self, painter: QPainter) -> None:
        """Articulated arm mounted on the ring at the top of the torso."""
        mount = QPointF(88, 308)
        # heroarm max raises the arm (toward horizontal), min hangs it down the torso.
        shoulder_deg = 38.0 - servo_to_angle("heroarm", self._value("heroarm")) * 0.85
        elbow_deg = -52.0 - servo_to_angle("elbow", self._value("elbow")) * 0.9
        upper_len, fore_len = 84.0, 74.0
        a1 = math.radians(shoulder_deg)
        elbow = QPointF(mount.x() + math.cos(a1) * upper_len, mount.y() + math.sin(a1) * upper_len)
        a2 = math.radians(shoulder_deg + elbow_deg)
        wrist = QPointF(elbow.x() + math.cos(a2) * fore_len, elbow.y() + math.sin(a2) * fore_len)

        # mount plate on the torso ring
        painter.setPen(QPen(_ORANGE_EDGE, 2))
        painter.setBrush(_ORANGE)
        painter.drawEllipse(mount, 15, 15)
        painter.setBrush(QColor("#8c4a12"))
        painter.drawEllipse(mount, 7, 7)

        self._capsule(painter, mount, elbow, 15, QColor("#565b60"))
        self._joint(painter, elbow, 12)
        self._capsule(painter, elbow, wrist, 12, QColor("#6d7378"))
        # orange wrist cuff near the claw
        cuff_from = QPointF(
            wrist.x() - math.cos(a2) * 20, wrist.y() - math.sin(a2) * 20
        )
        self._capsule(painter, cuff_from, wrist, 13, _ORANGE, edge=_ORANGE_EDGE)

        # claw
        hand_deg = math.degrees(a2) + servo_to_angle("hand", self._value("hand")) * 0.6
        painter.save()
        painter.translate(wrist)
        painter.rotate(hand_deg)
        painter.setPen(QPen(QColor("#2b2f33"), 2))
        painter.setBrush(QColor("#9aa0a6"))
        painter.drawEllipse(QPointF(4, 0), 8, 8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#b9bdc1"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        upper = QPainterPath(QPointF(8, -5))
        upper.quadTo(30, -20, 40, -6)
        painter.drawPath(upper)
        lower = QPainterPath(QPointF(8, 5))
        lower.quadTo(30, 20, 40, 6)
        painter.drawPath(lower)
        painter.setPen(QPen(QColor("#7d838a"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(8, 0), QPointF(24, 0))
        painter.restore()

    def _draw_middle_arm(self, painter: QPainter) -> None:
        """Static vestigial arm from the figure (no servo on the real build).

        Anchored to the SCREEN-LEFT edge of the grey ribbed drum: a short fixed
        pole sticks straight out to the left, the upper arm hangs straight down
        from it, and the elbow folds the ribbed tool forearm back UP alongside
        the torso — matching the reference figure."""
        pole_root = QPointF(-103, 400)   # left edge of the ribbed drum
        pole_end = QPointF(-130, 400)
        elbow = QPointF(-130, 458)       # straight down from the pole
        hand = QPointF(-142, 352)        # elbow folds the forearm back up

        # fixed mounting pole (with a small orange collar at the drum)
        self._capsule(painter, pole_root, pole_end, 11, QColor("#565b60"))
        self._capsule(painter, pole_root, QPointF(-112, 400), 13, _ORANGE, edge=_ORANGE_EDGE)
        self._joint(painter, pole_end, 9)
        # upper arm straight down, elbow at the bottom
        self._capsule(painter, pole_end, elbow, 12, QColor("#4a4f54"))
        self._joint(painter, elbow, 10)
        # small idler disc hanging under the elbow (see figure)
        painter.setPen(QPen(QColor("#15181b"), 2))
        painter.setBrush(QColor("#5b6167"))
        painter.drawEllipse(QPointF(-130, 474), 8, 8)
        # forearm folded back UP, carrying the ribbed clipper tool
        self._capsule(painter, elbow, hand, 11, QColor("#6d7378"))
        painter.save()
        painter.translate(hand)
        painter.rotate(math.degrees(math.atan2(hand.y() - elbow.y(), hand.x() - elbow.x())) + 90)
        # ribbed tool head: grey block with horizontal ribs and a rounded cap
        tool = QRectF(-11, -54, 22, 54)
        painter.setPen(QPen(QColor("#2b2f33"), 2))
        painter.setBrush(QColor("#83898f"))
        painter.drawRoundedRect(tool, 5, 5)
        painter.setPen(QPen(QColor("#3f464d"), 2))
        for i in range(6):
            yy = tool.top() + 8 + i * 8
            painter.drawLine(QPointF(tool.left() + 3, yy), QPointF(tool.right() - 3, yy))
        painter.setPen(QPen(QColor("#2b2f33"), 2))
        painter.setBrush(QColor("#9aa0a6"))
        painter.drawEllipse(QPointF(0, 0), 7, 7)
        painter.restore()

    def _draw_poker_arm(self, painter: QPainter) -> None:
        """Simple tool arm mounted on the SCREEN-RIGHT of the bottom torso ring;
        sweeps left/right with the pokerarm servo."""
        mount = QPointF(98, 450)
        swing = servo_to_angle("pokerarm", self._value("pokerarm"))
        a1 = math.radians(-2.0 - swing * 0.9)
        seg1 = 60.0
        elbow = QPointF(mount.x() + math.cos(a1) * seg1, mount.y() + math.sin(a1) * seg1)
        a2 = a1 + math.radians(16.0)
        seg2 = 54.0
        tip = QPointF(elbow.x() + math.cos(a2) * seg2, elbow.y() + math.sin(a2) * seg2)

        painter.setPen(QPen(_ORANGE_EDGE, 2))
        painter.setBrush(_ORANGE)
        painter.drawEllipse(mount, 12, 12)
        self._capsule(painter, mount, elbow, 12, QColor("#4a4f54"))
        self._joint(painter, elbow, 9)
        self._capsule(painter, elbow, tip, 10, QColor("#565b60"))
        # little tool head: stylus + crossbars
        painter.save()
        painter.translate(tip)
        painter.rotate(math.degrees(a2))
        painter.setPen(QPen(_ORANGE_EDGE, 2))
        painter.setBrush(_ORANGE)
        painter.drawRoundedRect(QRectF(-4, -8, 14, 16), 3, 3)
        painter.setPen(QPen(QColor("#b9bdc1"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(10, -14), QPointF(22, 14))
        painter.drawLine(QPointF(10, 14), QPointF(22, -14))
        painter.restore()

    def _capsule(
        self,
        painter: QPainter,
        p1: QPointF,
        p2: QPointF,
        width: float,
        fill: QColor,
        edge: QColor | None = None,
    ) -> None:
        painter.setPen(QPen(edge or QColor("#1d2125"), width + 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(p1, p2)
        painter.setPen(QPen(fill, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(p1, p2)

    def _joint(self, painter: QPainter, center: QPointF, radius: float) -> None:
        painter.setPen(QPen(QColor("#15181b"), 2))
        painter.setBrush(QColor("#787f86"))
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(QColor("#33383c"))
        painter.drawEllipse(center, radius * 0.42, radius * 0.42)

    # ── Animation state ─────────────────────────────────────────────────────

    def _tick_eye_animation(self) -> None:
        now = time.monotonic()
        dt = max(0.0, min(0.25, now - self._last_blink_tick))
        self._last_blink_tick = now
        if self._is_speaking():
            self._mouth_phase += dt * 14.0
        else:
            self._mouth_phase += dt * 2.0
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

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(27, 50, 70, 105), 1))
        step = 18
        for x in range(14, self.width(), step):
            painter.drawLine(x, 14, x, self.height() - 14)
        for y in range(14, self.height(), step):
            painter.drawLine(14, y, self.width() - 14, y)

    def _smooth(self) -> None:
        smoothing = max(0.01, min(1.0, float(getattr(config, "GUI_AVATAR_SMOOTHING", 0.25))))
        for name, target in self._target.items():
            current = self._current.get(name, target)
            self._current[name] = current + (target - current) * smoothing

    def _value(self, name: str) -> int:
        cfg = config.SERVO_CHANNELS[name]
        norm = self._current.get(name, 0.5)
        return int(cfg["min"] + (cfg["max"] - cfg["min"]) * norm)


def _neutral_norms() -> dict[str, float]:
    return {
        name: normalize_servo(name, cfg["neutral"])
        for name, cfg in config.SERVO_CHANNELS.items()
    }


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
