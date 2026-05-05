"""Lightweight live face-box tracking between recognition updates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional for GUI polish.
    cv2 = None  # type: ignore


@dataclass
class _Track:
    key: str
    box: tuple[float, float, float, float]
    source_box: tuple[float, float, float, float]
    prev_gray: np.ndarray
    points: Optional[np.ndarray]
    last_update_at: float


class LiveFaceBoxTracker:
    """
    Track recognized face boxes on live camera frames without re-running dlib.

    The runtime's true identity/face detection still comes from world_state.
    This class only keeps visual overlays and gaze-control inputs current
    between those slower recognition updates.
    """

    def __init__(self, *, stale_secs: float = 0.75) -> None:
        self._tracks: dict[str, _Track] = {}
        self._stale_secs = max(0.0, float(stale_secs))

    def update(
        self,
        frame,
        people: list[dict[str, Any]],
        *,
        now: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if cv2 is None or frame is None:
            return list(people or [])

        gray = _to_gray(frame)
        if gray is None:
            return list(people or [])

        now_mono = time.monotonic() if now is None else float(now)
        frame_h, frame_w = gray.shape[:2]
        output: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for idx, person in enumerate(people or []):
            item = dict(person)
            key = _person_key(item, idx)
            seen_keys.add(key)
            source_box = _person_box(item)
            visible_source = (
                source_box is not None
                and item.get("face_visible") is not False
                and not item.get("face_missing")
            )

            track = self._tracks.get(key)
            if visible_source and (
                track is None or _box_changed(source_box, track.source_box)
            ):
                track = self._seed_track(key, gray, source_box, now_mono)
            elif track is not None:
                track = self._advance_track(track, gray, frame_w, frame_h, now_mono)

            if track is not None and (now_mono - track.last_update_at) <= self._stale_secs:
                item["face_box"] = track.box
                item["position"] = (
                    int(track.box[0] + track.box[2] / 2.0),
                    int(track.box[1] + track.box[3] / 2.0),
                )
                item["face_visible"] = True
                item["face_missing"] = False
                item["gui_live_tracked"] = True
                item["live_tracked"] = True
            output.append(item)

        for key, track in list(self._tracks.items()):
            if key not in seen_keys and (now_mono - track.last_update_at) > self._stale_secs:
                self._tracks.pop(key, None)

        return output

    def _seed_track(
        self,
        key: str,
        gray: np.ndarray,
        box: tuple[float, float, float, float],
        now_mono: float,
    ) -> _Track:
        clamped = _clamp_box(box, gray.shape[1], gray.shape[0])
        points = _points_for_box(gray, clamped)
        track = _Track(
            key=key,
            box=clamped,
            source_box=clamped,
            prev_gray=gray.copy(),
            points=points,
            last_update_at=now_mono,
        )
        self._tracks[key] = track
        return track

    def _advance_track(
        self,
        track: _Track,
        gray: np.ndarray,
        frame_w: int,
        frame_h: int,
        now_mono: float,
    ) -> Optional[_Track]:
        if track.points is None or len(track.points) < 3:
            track.points = _points_for_box(track.prev_gray, track.box)
        if track.points is None or len(track.points) < 3:
            self._tracks.pop(track.key, None)
            return None

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            track.prev_gray,
            gray,
            track.points.astype(np.float32),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if next_pts is None or status is None:
            self._tracks.pop(track.key, None)
            return None

        good_new = next_pts[status.reshape(-1) == 1].reshape(-1, 2)
        good_old = track.points[status.reshape(-1) == 1].reshape(-1, 2)
        if len(good_new) < 3:
            self._tracks.pop(track.key, None)
            return None

        delta = np.median(good_new - good_old, axis=0)
        dx = float(delta[0])
        dy = float(delta[1])
        if abs(dx) > frame_w * 0.20 or abs(dy) > frame_h * 0.20:
            self._tracks.pop(track.key, None)
            return None

        x, y, w, h = track.box
        new_box = _clamp_box((x + dx, y + dy, w, h), frame_w, frame_h)
        track.box = new_box
        track.prev_gray = gray.copy()
        track.points = good_new.reshape(-1, 1, 2)
        if len(track.points) < 8:
            refreshed = _points_for_box(gray, new_box)
            if refreshed is not None:
                track.points = refreshed
        track.last_update_at = now_mono
        self._tracks[track.key] = track
        return track


def _to_gray(frame) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            return np.ascontiguousarray(arr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    return None


def _person_key(person: dict[str, Any], idx: int) -> str:
    for key in ("person_db_id", "face_id", "name", "id"):
        value = person.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"idx:{idx}"


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


def _box_changed(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> bool:
    if a is None or b is None:
        return a != b
    return any(abs(float(x) - float(y)) > 2.0 for x, y in zip(a, b))


def _clamp_box(
    box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    w = max(2.0, min(w, float(frame_w)))
    h = max(2.0, min(h, float(frame_h)))
    x = max(0.0, min(x, max(0.0, float(frame_w) - w)))
    y = max(0.0, min(y, max(0.0, float(frame_h) - h)))
    return (x, y, w, h)


def _points_for_box(
    gray: np.ndarray,
    box: tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    x, y, w, h = [int(round(v)) for v in box]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(gray.shape[1], x + max(2, w))
    y1 = min(gray.shape[0], y + max(2, h))
    if x1 <= x0 or y1 <= y0:
        return None

    roi = gray[y0:y1, x0:x1]
    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=24,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if points is not None and len(points) >= 3:
        points[:, 0, 0] += x0
        points[:, 0, 1] += y0
        return points.astype(np.float32)

    grid: list[list[list[float]]] = []
    for gx in np.linspace(x0 + 3, x1 - 3, 4):
        for gy in np.linspace(y0 + 3, y1 - 3, 4):
            grid.append([[float(gx), float(gy)]])
    if len(grid) < 3:
        return None
    return np.asarray(grid, dtype=np.float32)
