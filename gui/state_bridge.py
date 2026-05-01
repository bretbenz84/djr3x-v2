"""Thread-safe bridge between the Rex runtime and the optional Qt dashboard."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a project dependency.
    np = None  # type: ignore

import config


class GUIDashboardBridge:
    """Owns copied runtime state for Qt to poll without touching live objects."""

    def __init__(self, max_lines: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._world_state: dict[str, Any] = {}
        self._servo_positions: dict[str, int] = {}
        self._head_led_state: dict[str, Any] = {
            "mode": "off",
            "eye_color": (0, 0, 0),
            "eyes_active": False,
            "updated_at": time.time(),
        }
        self._scene_description: str = ""
        self._conversation_lines: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(max_lines or getattr(config, "GUI_CONVERSATION_LOG_MAX_LINES", 300)))
        )
        self._line_seq = 0
        self._updated_at = time.time()

    def update_frame(self, frame) -> None:
        """Store a copy of the latest BGR camera frame, or clear it with None."""
        with self._lock:
            if frame is None:
                self._frame = None
            elif np is not None and hasattr(frame, "copy"):
                self._frame = frame.copy()
            else:
                self._frame = copy.deepcopy(frame)
            self._updated_at = time.time()

    def update_world_state_snapshot(self, world_state_snapshot: dict[str, Any]) -> None:
        snapshot = copy.deepcopy(world_state_snapshot or {})
        env = snapshot.get("environment") or {}
        self_state = snapshot.get("self_state") or snapshot.get("self") or {}
        servo_positions = self_state.get("servo_positions") or {}

        with self._lock:
            self._world_state = snapshot
            if servo_positions:
                self._servo_positions.update({
                    str(name): int(value)
                    for name, value in servo_positions.items()
                    if _looks_int(value)
                })
            description = env.get("description")
            if description:
                self._scene_description = str(description)
            self._updated_at = time.time()

    def update_servo_position(self, name, value) -> None:
        """Store an intended/current servo position by name or channel."""
        servo_name = _servo_name(name)
        if servo_name is None or not _looks_int(value):
            return
        with self._lock:
            self._servo_positions[servo_name] = int(value)
            self._updated_at = time.time()

    def update_head_led_state(
        self,
        *,
        mode: Optional[str] = None,
        eye_color: Optional[tuple[int, int, int]] = None,
        eyes_active: Optional[bool] = None,
    ) -> None:
        """Mirror high-level head LED state for the Qt avatar."""
        with self._lock:
            if mode is not None:
                self._head_led_state["mode"] = str(mode)
            if eye_color is not None:
                r, g, b = (max(0, min(255, int(v))) for v in eye_color)
                self._head_led_state["eye_color"] = (r, g, b)
            if eyes_active is not None:
                self._head_led_state["eyes_active"] = bool(eyes_active)
            self._head_led_state["updated_at"] = time.time()
            self._updated_at = time.time()

    def add_conversation_line(self, speaker: str, text: str, kind: str = "user") -> None:
        text = (text or "").strip()
        if not text:
            return
        kind = (kind or "system").strip().lower()
        if kind not in {"user", "rex", "system"}:
            kind = "system"
        speaker = (speaker or "").strip() or {
            "user": "Unknown",
            "rex": "Rex",
            "system": "System",
        }[kind]
        with self._lock:
            self._line_seq += 1
            self._conversation_lines.append({
                "seq": self._line_seq,
                "ts": time.time(),
                "speaker": speaker,
                "text": text,
                "kind": kind,
            })
            self._updated_at = time.time()

    def set_scene_description(self, text: str) -> None:
        with self._lock:
            self._scene_description = (text or "").strip()
            self._updated_at = time.time()

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None and hasattr(self._frame, "copy") else None
            return {
                "frame": frame,
                "world_state": copy.deepcopy(self._world_state),
                "servo_positions": dict(self._servo_positions),
                "head_led_state": copy.deepcopy(self._head_led_state),
                "scene_description": self._scene_description,
                "conversation_lines": list(self._conversation_lines),
                "updated_at": self._updated_at,
            }


def _servo_name(name_or_channel) -> Optional[str]:
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
        if int(cfg.get("ch", -1)) == channel:
            return name
    return None


def _looks_int(value) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


gui_bridge = GUIDashboardBridge()


def get_bridge() -> GUIDashboardBridge:
    return gui_bridge
