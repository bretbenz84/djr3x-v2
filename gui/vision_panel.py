"""Camera preview panel for the optional DJ-R3X dashboard."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class VisionPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame = None
        self._people: list[dict[str, Any]] = []
        self._scene_description = ""
        self._last_frame_at = 0.0
        self.setMinimumSize(360, 360)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._frame = snapshot.get("frame")
        ws = snapshot.get("world_state") or {}
        self._people = list(ws.get("people") or [])
        env = ws.get("environment") or {}
        self._scene_description = (
            snapshot.get("scene_description")
            or env.get("description")
            or ""
        )
        if self._frame is not None:
            self._last_frame_at = time.monotonic()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07111a"))

        content = self.rect().adjusted(16, 16, -16, -16)
        frame_rect = QRectF(content.adjusted(0, 0, 0, -42))
        image_rect = QRectF()

        if self._frame is None:
            self._draw_placeholder(painter, frame_rect)
        else:
            image = _bgr_frame_to_qimage(self._frame)
            if image is None:
                self._draw_placeholder(painter, frame_rect)
            else:
                image_rect = _scaled_rect(image.width(), image.height(), frame_rect)
                painter.drawImage(image_rect, image)
                self._draw_people(painter, image_rect, image.width(), image.height())

        self._draw_timestamp(painter, frame_rect)
        self._draw_camera_meta(painter, content)
        painter.end()

    def _draw_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#274564"), 1))
        painter.setBrush(QColor("#0b1622"))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor("#9badbf"))
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            "Camera unavailable / software-only mode.",
        )

    def _draw_people(
        self,
        painter: QPainter,
        image_rect: QRectF,
        frame_w: int,
        frame_h: int,
    ) -> None:
        if not self._people or frame_w <= 0 or frame_h <= 0:
            return

        sx = image_rect.width() / float(frame_w)
        sy = image_rect.height() / float(frame_h)
        for idx, person in enumerate(self._people):
            label = _person_label(person)
            expression = _person_expression(person)
            color = QColor("#75ef63")
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            box = _person_box(person)
            if box is not None:
                x, y, w, h = box
                rect = QRectF(
                    image_rect.left() + x * sx,
                    image_rect.top() + y * sy,
                    w * sx,
                    h * sy,
                )
                painter.drawRect(rect)
                text_anchor = rect.topLeft() + QPointF(0, -6)
            else:
                point = _person_point(person, frame_w, frame_h)
                if point is None:
                    point = (frame_w * (0.3 + 0.2 * idx), frame_h * 0.45)
                px = image_rect.left() + point[0] * sx
                py = image_rect.top() + point[1] * sy
                painter.setBrush(color)
                painter.drawEllipse(QPointF(px, py), 6, 6)
                text_anchor = QPointF(px + 8, py - 8)

            _draw_label(painter, text_anchor, label, expression, color)

    def _draw_timestamp(self, painter: QPainter, frame_rect: QRectF) -> None:
        if self._last_frame_at <= 0:
            return
        text = time.strftime("%I:%M:%S %p").lstrip("0")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        box = QRectF(
            frame_rect.right() - metrics.horizontalAdvance(text) - 18,
            frame_rect.top() + 8,
            metrics.horizontalAdvance(text) + 12,
            28,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(7, 17, 26, 215))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor("#aebccc"))
        painter.drawText(
            box,
            Qt.AlignmentFlag.AlignCenter,
            text,
        )

    def _draw_camera_meta(self, painter: QPainter, content: QRectF) -> None:
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        y = content.bottom() - 10
        x = content.left() + 8
        parts = [("Camera:", "#5396ff"), ("USB Camera", "#c5d0dc"), ("•", "#5396ff"), ("30 FPS", "#c5d0dc"), ("•", "#5396ff")]
        if self._frame is not None:
            arr = np.asarray(self._frame)
            parts.append((f"{arr.shape[1]}x{arr.shape[0]}", "#c5d0dc"))
        else:
            parts.append(("No Signal", "#c5d0dc"))
        for text, color in parts:
            painter.setPen(QColor(color))
            painter.drawText(QPointF(x, y), text)
            x += painter.fontMetrics().horizontalAdvance(text) + 12


def _bgr_frame_to_qimage(frame) -> QImage | None:
    try:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    except Exception:
        return None


def _scaled_rect(image_w: int, image_h: int, bounds: QRectF) -> QRectF:
    if image_w <= 0 or image_h <= 0:
        return QRectF(bounds)
    scale = min(bounds.width() / image_w, bounds.height() / image_h)
    w = image_w * scale
    h = image_h * scale
    return QRectF(
        bounds.left() + (bounds.width() - w) / 2.0,
        bounds.top() + (bounds.height() - h) / 2.0,
        w,
        h,
    )


def _person_label(person: dict[str, Any]) -> str:
    for key in ("name", "face_id", "voice_id"):
        value = person.get(key)
        if value:
            return str(value)
    return "Unknown"


def _person_details(person: dict[str, Any]) -> str:
    parts = []
    for key in ("engagement", "distance_zone", "pose"):
        value = person.get(key)
        if value:
            parts.append(str(value).replace("_", " "))
    return " / ".join(parts)


def _person_expression(person: dict[str, Any]) -> str:
    for key in (
        "expression",
        "mood",
        "emotion",
        "affect",
        "face_expression",
        "face_mood",
        "facial_expression",
    ):
        value = person.get(key)
        if isinstance(value, dict):
            value = value.get("mood") or value.get("expression") or value.get("affect")
        if value:
            text = str(value).strip().lower().replace("_", " ")
            return text
    return "neutral"


def _person_box(person: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = (
        person.get("face_box")
        or person.get("bounding_box")
        or person.get("bbox")
        or person.get("box")
    )
    if isinstance(box, dict):
        box = (
            box.get("x"),
            box.get("y"),
            box.get("w") or box.get("width"),
            box.get("h") or box.get("height"),
        )
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _person_point(
    person: dict[str, Any],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float] | None:
    pos = person.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        x = float(pos[0])
        y = float(pos[1])
    except (TypeError, ValueError):
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x * frame_w, y * frame_h)
    return (x, y)


def _draw_label(
    painter: QPainter,
    anchor: QPointF,
    label: str,
    details: str,
    color: QColor,
) -> None:
    text = label if not details else f"{label}  {details}"
    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    width = min(max(metrics.horizontalAdvance(text) + 14, 70), 360)
    height = 24
    x = max(8.0, min(anchor.x(), painter.device().width() - width - 8))
    y = max(8.0, anchor.y())
    rect = QRectF(x, y, width, height)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(52, 136, 49, 210))
    painter.drawRoundedRect(rect, 2, 2)
    painter.setPen(QColor("#e9ffe6"))
    painter.drawText(
        rect.adjusted(7, 0, -7, 0),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        metrics.elidedText(text, Qt.TextElideMode.ElideRight, width - 14),
    )
