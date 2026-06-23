"""Camera preview panel for the optional DJ-R3X dashboard."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

import config
from gui.live_face_tracker import LiveFaceBoxTracker

# Frames older than this read as a dropout: the meta line shows "STALE Xs" and
# the held image is dimmed so a frozen frame isn't mistaken for a live one.
_CAMERA_STALE_SECS = 2.0

# Body-pose wireframe (MediaPipe 33-point skeleton). Edges connect named landmarks;
# only drawn when both endpoints are present and visible. Coordinates in pose_keypoints
# are normalized (0..1) over the full frame, mapped straight onto the displayed image.
# NOTE: the hand points below are COARSE — the Pose model gives only ONE pinky/index/thumb
# point per hand (landmarks 17-22), not an articulated finger skeleton. They extend the
# wireframe past the wrist as a small fan. A real 21-point finger skeleton would need the
# separate MediaPipe Hand Landmarker (see docs/junecodereview / project notes).
_SKELETON_COLOR = "#36d9ff"   # cyan — distinct from face boxes (green) / animals (amber)
_SKELETON_MIN_VIS = 0.3
_POSE_EDGES = (
    # face
    ("LEFT_EAR", "LEFT_EYE"), ("LEFT_EYE", "NOSE"),
    ("NOSE", "RIGHT_EYE"), ("RIGHT_EYE", "RIGHT_EAR"),
    # arms
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    # hands (coarse: wrist → thumb/index/pinky + a knuckle line, so the skeleton no
    # longer stops dead at the wrist; not real fingers — see the note above)
    ("LEFT_WRIST", "LEFT_THUMB"), ("LEFT_WRIST", "LEFT_INDEX"),
    ("LEFT_WRIST", "LEFT_PINKY"), ("LEFT_INDEX", "LEFT_PINKY"),
    ("RIGHT_WRIST", "RIGHT_THUMB"), ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_PINKY"), ("RIGHT_INDEX", "RIGHT_PINKY"),
    # shoulders + torso
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    # legs
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
)
_POSE_JOINTS = (
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    # coarse hand points (Pose landmarks 17-22): one pinky/index/thumb dot per hand.
    "LEFT_THUMB", "LEFT_INDEX", "LEFT_PINKY",
    "RIGHT_THUMB", "RIGHT_INDEX", "RIGHT_PINKY",
)


class VisionPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame = None
        self._people: list[dict[str, Any]] = []
        self._animals: list[dict[str, Any]] = []
        self._scene_description = ""
        self._last_frame_at = 0.0
        self._camera_stats: dict[str, Any] = {}
        self._face_tracker = LiveFaceBoxTracker()
        self.setMinimumSize(360, 260)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._frame = snapshot.get("frame")
        ws = snapshot.get("world_state") or {}
        people = list(ws.get("people") or [])
        self._people = self._face_tracker.update(self._frame, people)
        self._animals = [
            dict(animal)
            for animal in (ws.get("animals") or [])
            if isinstance(animal, dict)
        ]
        env = ws.get("environment") or {}
        self._scene_description = (
            snapshot.get("scene_description")
            or env.get("description")
            or ""
        )
        self._camera_stats = snapshot.get("camera_stats") or {}
        if self._frame is not None:
            self._last_frame_at = time.monotonic()
        self.update()

    def _camera_stale_secs(self) -> float | None:
        """Seconds since the camera last captured a frame, or None if unknown."""
        last_at = self._camera_stats.get("last_frame_monotonic")
        if last_at is None:
            return None
        return max(0.0, time.monotonic() - float(last_at))

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
                self._draw_pose_skeletons(painter, image_rect)
                self._draw_animals(painter, image_rect, image.width(), image.height())
                self._draw_people(painter, image_rect, image.width(), image.height())
                stale = self._camera_stale_secs()
                if stale is not None and stale > _CAMERA_STALE_SECS:
                    # Dim a frozen frame so it doesn't read as live video.
                    painter.fillRect(image_rect, QColor(7, 17, 26, 150))

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
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
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

    def _draw_pose_skeletons(self, painter: QPainter, image_rect: QRectF) -> None:
        """Overlay a live body-pose wireframe for each person that has landmarks."""
        if not getattr(config, "GUI_POSE_WIREFRAME_ENABLED", True):
            return
        if not self._people or image_rect.isEmpty():
            return
        # Clip to the displayed video rect so limbs running off the edge are cut at the
        # frame boundary instead of bleeding into the panel's letterbox/border.
        painter.save()
        painter.setClipRect(image_rect)
        try:
            for person in self._people:
                keypoints = person.get("pose_keypoints")
                if isinstance(keypoints, dict) and keypoints:
                    _draw_one_skeleton(painter, image_rect, keypoints)
        finally:
            painter.restore()

    def _draw_animals(
        self,
        painter: QPainter,
        image_rect: QRectF,
        frame_w: int,
        frame_h: int,
    ) -> None:
        if not self._animals or frame_w <= 0 or frame_h <= 0:
            return

        sx = image_rect.width() / float(frame_w)
        sy = image_rect.height() / float(frame_h)
        color = QColor("#f0c45a")
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for idx, animal in enumerate(self._animals):
            label = _animal_label(animal, idx)
            details = _animal_details(animal)
            box = _animal_box(animal)
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
                point = _animal_point(animal, frame_w, frame_h)
                px = image_rect.left() + point[0] * sx
                py = image_rect.top() + point[1] * sy
                painter.setBrush(color)
                painter.drawEllipse(QPointF(px, py), 6, 6)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                text_anchor = QPointF(px + 8, py - 8)

            _draw_label(painter, text_anchor, label, details, color)

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

        stats = self._camera_stats or {}
        label = str(stats.get("label") or "").strip() or "Camera"
        accent = "#5396ff"
        value = "#c5d0dc"
        amber = "#f0c45a"

        parts: list[tuple[str, str]] = [("Camera:", accent), (label, value), ("•", accent)]

        # Freshness/rate: tell the truth instead of a hardcoded "30 FPS".
        stale = self._camera_stale_secs()
        if stale is None:
            parts.append(("No Signal", amber))
        elif stale > _CAMERA_STALE_SECS:
            parts.append((f"STALE {stale:.1f}s", amber))
        else:
            fps = stats.get("fps")
            parts.append((f"{float(fps):.0f} FPS" if fps else "— FPS", value))
        parts.append(("•", accent))

        # Resolution: prefer the live frame, fall back to the reported stat.
        resolution = stats.get("resolution")
        if self._frame is not None:
            arr = np.asarray(self._frame)
            parts.append((f"{arr.shape[1]}x{arr.shape[0]}", value))
        elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            parts.append((f"{int(resolution[0])}x{int(resolution[1])}", value))
        else:
            parts.append(("No Signal", value))

        for text, color in parts:
            painter.setPen(QColor(color))
            painter.drawText(QPointF(x, y), text)
            x += painter.fontMetrics().horizontalAdvance(text) + 12


def _skeleton_point(
    keypoints: dict[str, Any],
    name: str,
    image_rect: QRectF,
) -> QPointF | None:
    """Map a normalized (x, y, visibility) landmark onto the displayed image, or None
    if it's missing / below the visibility floor."""
    value = keypoints.get(name)
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        x, y, vis = float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None
    if vis < _SKELETON_MIN_VIS:
        return None
    # Landmarks are normalized; clamp lightly so a just-offscreen joint still anchors a line.
    x = min(max(x, -0.1), 1.1)
    y = min(max(y, -0.1), 1.1)
    return QPointF(
        image_rect.left() + x * image_rect.width(),
        image_rect.top() + y * image_rect.height(),
    )


def _draw_one_skeleton(
    painter: QPainter,
    image_rect: QRectF,
    keypoints: dict[str, Any],
) -> None:
    color = QColor(_SKELETON_COLOR)
    points = {name: _skeleton_point(keypoints, name, image_rect) for name in _POSE_JOINTS}

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(color, 2))
    for a, b in _POSE_EDGES:
        pa = points.get(a) or _skeleton_point(keypoints, a, image_rect)
        pb = points.get(b) or _skeleton_point(keypoints, b, image_rect)
        if pa is not None and pb is not None:
            painter.drawLine(pa, pb)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    for point in points.values():
        if point is not None:
            painter.drawEllipse(point, 3.0, 3.0)


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
    for key in ("face_expression", "facial_expression"):
        value = person.get(key)
        if isinstance(value, dict):
            value = value.get("expression") or value.get("mood") or value.get("affect")
        if value:
            text = str(value).strip().lower().replace("_", " ")
            return text
    for key in ("face_mood", "expression", "mood", "emotion", "affect"):
        value = person.get(key)
        if isinstance(value, dict):
            value = value.get("mood") or value.get("expression") or value.get("affect")
        if value:
            text = str(value).strip().lower().replace("_", " ")
            return text
    return ""


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


def _animal_label(animal: dict[str, Any], idx: int) -> str:
    species = str(animal.get("species") or "").strip()
    return species.title() if species else f"Animal {idx + 1}"


def _animal_details(animal: dict[str, Any]) -> str:
    parts = []
    confidence = animal.get("confidence")
    try:
        if confidence is not None and confidence != "":
            score = float(confidence)
            if 0.0 <= score <= 1.0:
                parts.append(f"{score * 100.0:.0f}%")
            else:
                parts.append(str(confidence))
    except (TypeError, ValueError):
        if confidence:
            parts.append(str(confidence).replace("_", " "))
    source = str(animal.get("source") or "").strip()
    if source:
        parts.append(source.replace("_", " "))
    return " / ".join(parts)


def _animal_box(animal: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = (
        animal.get("box")
        or animal.get("animal_box")
        or animal.get("bounding_box")
        or animal.get("bbox")
    )
    if isinstance(box, dict):
        box = (
            box.get("x") or box.get("origin_x"),
            box.get("y") or box.get("origin_y"),
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


def _animal_point(
    animal: dict[str, Any],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    position = animal.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            x = float(position[0])
            y = float(position[1])
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return (x * frame_w, y * frame_h)
            return (x, y)
        except (TypeError, ValueError):
            pass

    text = str(position or "").strip().lower()
    if "left" in text:
        x = frame_w * 0.25
    elif "right" in text:
        x = frame_w * 0.75
    else:
        x = frame_w * 0.5

    if "upper" in text or "top" in text:
        y = frame_h * 0.25
    elif "lower" in text or "bottom" in text or "foreground" in text:
        y = frame_h * 0.75
    else:
        y = frame_h * 0.5
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
    brush = QColor(color)
    brush.setAlpha(210)
    painter.setBrush(brush)
    painter.drawRoundedRect(rect, 2, 2)
    painter.setPen(QColor("#e9ffe6"))
    painter.drawText(
        rect.adjusted(7, 0, -7, 0),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        metrics.elidedText(text, Qt.TextElideMode.ElideRight, width - 14),
    )
