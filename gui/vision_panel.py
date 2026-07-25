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
        self._objects: list[dict[str, Any]] = []
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
        self._objects = [
            dict(obj)
            for obj in (ws.get("objects") or [])
            if isinstance(obj, dict)
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

        # The holo panel behind provides the backdrop; the video owns the full area
        # (the old 42px caption strip is folded into a small in-frame status chip).
        content = self.rect().adjusted(12, 6, -12, -10)
        frame_rect = QRectF(content)
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
                self._draw_pose_skeletons(painter, image_rect, image.width(), image.height())
                self._draw_objects(painter, image_rect, image.width(), image.height())
                self._draw_occlusion_zones(painter, image_rect)
                self._draw_animals(painter, image_rect, image.width(), image.height())
                self._draw_people(painter, image_rect, image.width(), image.height())
                stale = self._camera_stale_secs()
                if stale is not None and stale > _CAMERA_STALE_SECS:
                    # Dim a frozen frame so it doesn't read as live video.
                    painter.fillRect(image_rect, QColor(7, 17, 26, 150))

        self._draw_timestamp(painter, frame_rect)
        self._draw_camera_meta(painter, image_rect if not image_rect.isEmpty() else frame_rect)
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
            if not _slot_is_drawable_person(person):
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

    def _draw_pose_skeletons(
        self, painter: QPainter, image_rect: QRectF, frame_w: int = 0, frame_h: int = 0
    ) -> None:
        """Overlay a live body-pose wireframe for each person that has landmarks.

        Coherence guard (GUI_POSE_REQUIRE_FACE): only draw a skeleton for a slot that has
        a VISIBLE face whose centre sits near the pose's head. This kills the two failures
        from the JT run — phantom wireframes drawn "above us" (no face there) and a
        mis-bound wireframe drawn over the WRONG person (pose head far from the slot's
        face). Set GUI_POSE_REQUIRE_FACE=False to draw every detected pose (old behavior).
        """
        if not getattr(config, "GUI_POSE_WIREFRAME_ENABLED", True):
            return
        if not self._people or image_rect.isEmpty():
            return
        require_face = bool(getattr(config, "GUI_POSE_REQUIRE_FACE", True))
        max_dist = float(getattr(config, "GUI_POSE_FACE_COHERENCE_DIST", 0.20))
        # Clip to the displayed video rect so limbs running off the edge are cut at the
        # frame boundary instead of bleeding into the panel's letterbox/border.
        painter.save()
        painter.setClipRect(image_rect)
        try:
            for person in self._people:
                keypoints = person.get("pose_keypoints")
                if not (isinstance(keypoints, dict) and keypoints):
                    continue
                if require_face and not _pose_face_coherent(
                    person, keypoints, frame_w, frame_h, max_dist
                ):
                    continue
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

    def _draw_occlusion_zones(self, painter: QPainter, image_rect: QRectF) -> None:
        """Dim dashed outlines of the self-occlusion zones (Rex's own eye stalks in
        front of the lens; config.CAMERA_SELF_OCCLUSION_ZONES) so the mask can be
        aligned against the live feed by eye. Object detections mostly inside a zone
        are suppressed at the source (vision/animal_detector)."""
        zones = getattr(config, "CAMERA_SELF_OCCLUSION_ZONES", None) or []
        if not zones or not getattr(config, "GUI_OBJECT_BOXES_ENABLED", True):
            return
        if not getattr(config, "GUI_OCCLUSION_ZONES_VISIBLE", True):
            return
        # Readability, not decoration: at the original 1-px dash on 27% alpha these
        # outlines measured ~11% contrast against the feed and were invisible in
        # practice (owner 2026-07-24: "I'm not seeing that blocked off anymore" —
        # while the mask itself was working fine). The zone is now filled and
        # labelled so it's obvious WHERE detection is disabled and whether the
        # rectangles still line up with the eye stalks after a camera move.
        alpha_fill = int(getattr(config, "GUI_OCCLUSION_ZONE_FILL_ALPHA", 48))
        alpha_edge = int(getattr(config, "GUI_OCCLUSION_ZONE_EDGE_ALPHA", 190))
        pen = QPen(QColor(185, 140, 255, alpha_edge), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        for zx0, zy0, zx1, zy1 in zones:
            rect = QRectF(
                image_rect.left() + zx0 * image_rect.width(),
                image_rect.top() + zy0 * image_rect.height(),
                (zx1 - zx0) * image_rect.width(),
                (zy1 - zy0) * image_rect.height(),
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(185, 140, 255, alpha_fill))
            painter.drawRect(rect)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # One label for the whole mask — repeating it per zone just adds clutter.
        first = zones[0]
        painter.setPen(QPen(QColor(205, 175, 255, 220), 1))
        painter.drawText(
            QPointF(
                image_rect.left() + first[0] * image_rect.width() + 6,
                image_rect.top() + first[1] * image_rect.height() + 14,
            ),
            "no-detect (eye stalks)",
        )

    def _draw_objects(
        self,
        painter: QPainter,
        image_rect: QRectF,
        frame_w: int,
        frame_h: int,
    ) -> None:
        if not getattr(config, "GUI_OBJECT_BOXES_ENABLED", True):
            return
        if not self._objects or frame_w <= 0 or frame_h <= 0:
            return

        sx = image_rect.width() / float(frame_w)
        sy = image_rect.height() / float(frame_h)
        color = QColor("#b98cff")  # violet — distinct from faces/animals/poses
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for idx, obj in enumerate(self._objects):
            label = _object_label(obj, idx)
            details = _animal_details(obj)  # confidence/source render identically
            box = _animal_box(obj)          # objects carry the same (x, y, w, h) box
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
                point = _animal_point(obj, frame_w, frame_h)
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

    def _draw_camera_meta(self, painter: QPainter, frame_rect: QRectF) -> None:
        """Tiny in-frame status chip: live FPS, or an honest STALE / NO SIGNAL warning.

        The old caption strip (camera name + resolution) was noise; the chip keeps the
        one thing worth glancing at — is the feed live and how fast."""
        stale = self._camera_stale_secs()
        if stale is None:
            text, color = "● NO SIGNAL", "#f0c45a"
        elif stale > _CAMERA_STALE_SECS:
            text, color = f"● STALE {stale:.1f}s", "#f0c45a"
        else:
            fps = (self._camera_stats or {}).get("fps")
            text, color = (f"● {float(fps):.0f} FPS" if fps else "● LIVE"), "#75ef63"

        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 16
        chip = QRectF(frame_rect.left() + 6, frame_rect.bottom() - 26, width, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(4, 10, 17, 200))
        painter.drawRoundedRect(chip, 2, 2)
        painter.setPen(QColor(color))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)


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


def _slot_is_drawable_person(person: dict[str, Any]) -> bool:
    """True if a world-state slot should get a GUI person marker.

    A pose-only phantom — person_db_id None, NO face box, NO identity — is not a person to
    draw: with no face box it falls to the dot branch and renders a bogus "Unknown" marker at
    the pose's nose, right over the real person's correctly-labelled face (the close-up
    regression from POSE_MAX_PEOPLE>1, where MediaPipe hallucinates a stray skeleton). Draw a
    marker only for a slot with a real detected face box OR a known identity — mirroring the
    visible-face gate the unknown-COUNTING consumers already use (social_scene, llm; a6f01bd).
    """
    if _person_box(person) is not None:
        return True  # a real detected face box — draw it even if the face is unidentified
    return bool(person.get("name") or person.get("face_id") or person.get("voice_id"))


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


def _pose_head_norm(keypoints: dict) -> tuple[float, float] | None:
    """Normalized (x, y) of the pose's head: the NOSE if present, else the shoulder
    midpoint. Keypoint values are (x, y, visibility) in [0, 1]."""
    nose = keypoints.get("NOSE")
    if isinstance(nose, (list, tuple)) and len(nose) >= 2:
        return (float(nose[0]), float(nose[1]))
    ls, rs = keypoints.get("LEFT_SHOULDER"), keypoints.get("RIGHT_SHOULDER")
    if (isinstance(ls, (list, tuple)) and len(ls) >= 2
            and isinstance(rs, (list, tuple)) and len(rs) >= 2):
        return ((float(ls[0]) + float(rs[0])) / 2.0, (float(ls[1]) + float(rs[1])) / 2.0)
    return None


def _pose_face_coherent(
    person: dict, keypoints: dict, frame_w: int, frame_h: int, max_dist: float
) -> bool:
    """True if this slot's pose belongs to its visible face: the slot has a visible
    face box and the pose head sits within max_dist (normalized) of the face centre.
    Rejects phantom poses (no face) and mis-bound poses (head far from the slot's face)."""
    if person.get("face_visible") is False or person.get("face_missing"):
        return False
    if frame_w <= 0 or frame_h <= 0:
        return False
    box = _person_box(person)
    if box is None:
        return False
    head = _pose_head_norm(keypoints)
    if head is None:
        return False
    fx = (box[0] + box[2] / 2.0) / float(frame_w)
    fy = (box[1] + box[3] / 2.0) / float(frame_h)
    return ((head[0] - fx) ** 2 + (head[1] - fy) ** 2) ** 0.5 <= max_dist


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


def _object_label(obj: dict[str, Any], idx: int) -> str:
    label = str(obj.get("label") or obj.get("class") or obj.get("name") or "").strip()
    return label.title() if label else f"Object {idx + 1}"


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
