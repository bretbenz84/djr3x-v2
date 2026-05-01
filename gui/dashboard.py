"""Optional PySide6 dashboard for DJ-R3X."""

from __future__ import annotations

import argparse
import logging
import math
import random
import signal
import sys
import time
from typing import Any, Callable, Optional

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QSlider,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - exercised when dependency missing.
    raise RuntimeError(f"PySide6 is unavailable: {exc}") from exc

import numpy as np

import config
from gui.conversation_panel import ConversationPanel
from gui.jeopardy_panel import JeopardyPanel
from gui.rex_avatar import RexAvatar, normalize_servo, servo_to_angle, servo_to_offset
from gui.state_bridge import GUIDashboardBridge, gui_bridge
from gui.vision_panel import VisionPanel

_log = logging.getLogger(__name__)


class DashboardWindow(QMainWindow):
    def __init__(
        self,
        bridge: GUIDashboardBridge,
        *,
        shutdown_callback: Optional[Callable[[], None]] = None,
        demo: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._shutdown_callback = shutdown_callback
        self._demo = demo
        self._closing_from_shutdown = False
        self._last_frame_time = time.monotonic()
        self._frame_counter = 0
        self._fps = 0.0

        self.setWindowTitle(getattr(config, "GUI_WINDOW_TITLE", "DJ-R3X Controller"))
        self.resize(1280, 840)
        self.setMinimumSize(1100, 740)

        self.vision = VisionPanel()
        self.scene = VisionDescriptionPanel()
        self.avatar = RexAvatar()
        self.servos = ServoPositionsPanel()
        self.conversation = ConversationPanel()
        self.conversation.set_submit_callback(
            lambda text: self._bridge.add_conversation_line("Human", text, "user")
        )
        self.footer = FooterBar()
        self.jeopardy = JeopardyPanel()
        self.connection = QLabel("●  Connected")
        self.connection.setObjectName("connectionLabel")

        root = QWidget()
        root.setObjectName("root")
        self._shell = QVBoxLayout(root)
        self._shell.setContentsMargins(14, 8, 14, 14)
        self._shell.setSpacing(12)

        self._top_bar = QWidget()
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._top_bar.setLayout(top)
        lights = QLabel("●  ●  ●")
        lights.setObjectName("trafficLights")
        top.addWidget(lights)
        title = QLabel("DJ-R3X Controller")
        title.setObjectName("windowTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(title, 1)
        top.addWidget(self.connection)
        self._shell.addWidget(self._top_bar)

        columns = QGridLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setHorizontalSpacing(12)
        columns.setVerticalSpacing(12)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(ChromePanel("1", "VISION", self.vision), 7)
        left.addWidget(ChromePanel("", "OPENAI VISION DESCRIPTION", self.scene), 5)
        left_box = QWidget()
        left_box.setLayout(left)

        center = ChromePanel("☵", "CONVERSATION LOG", self.conversation)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        avatar_panel = ChromePanel("3", "R3X AVATAR", self.avatar)
        servo_panel = ChromePanel("", "SERVO POSITIONS", self.servos)
        servo_panel.setMaximumHeight(270)
        right.addWidget(avatar_panel, 1)
        right.addWidget(servo_panel, 0)
        right_box = QWidget()
        right_box.setLayout(right)

        columns.addWidget(left_box, 0, 0)
        columns.addWidget(center, 0, 1)
        columns.addWidget(right_box, 0, 2)
        columns.setColumnStretch(0, 11)
        columns.setColumnStretch(1, 10)
        columns.setColumnStretch(2, 17)
        dashboard_page = QWidget()
        dashboard_page.setLayout(columns)

        self._main_stack = QStackedWidget()
        self._main_stack.addWidget(dashboard_page)
        self._main_stack.addWidget(self.jeopardy)
        self._shell.addWidget(self._main_stack, 1)
        self._shell.addWidget(self.footer)

        self.setCentralWidget(root)
        self.setStyleSheet(_STYLE)

        fps = max(1, int(getattr(config, "GUI_FPS", 20) or 20))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(1, int(1000 / fps)))

        if demo:
            self._demo_timer = QTimer(self)
            self._demo_timer.timeout.connect(lambda: _advance_demo(self._bridge))
            self._demo_timer.start(250)
        else:
            self._demo_timer = None

    def close_from_shutdown(self) -> None:
        self._closing_from_shutdown = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._closing_from_shutdown and self._shutdown_callback is not None:
            try:
                self._shutdown_callback()
            except Exception as exc:
                _log.warning("GUI shutdown callback failed: %s", exc)
        super().closeEvent(event)

    def _tick(self) -> None:
        snapshot = self._bridge.get_snapshot()
        self.vision.set_snapshot(snapshot)
        self.scene.set_snapshot(snapshot)
        self.avatar.set_snapshot(snapshot)
        self.servos.set_snapshot(snapshot)
        self.conversation.set_snapshot(snapshot)
        self.jeopardy.set_snapshot(snapshot)
        game_state = snapshot.get("game_state") or {}
        jeopardy_active = game_state.get("active_game") == "jeopardy"
        if jeopardy_active:
            self._main_stack.setCurrentWidget(self.jeopardy)
        else:
            self._main_stack.setCurrentIndex(0)
        self._top_bar.setVisible(not jeopardy_active)
        self.footer.setVisible(not jeopardy_active)
        if jeopardy_active:
            self._shell.setContentsMargins(0, 0, 0, 0)
            self._shell.setSpacing(0)
        else:
            self._shell.setContentsMargins(14, 8, 14, 14)
            self._shell.setSpacing(12)

        self._frame_counter += 1
        now = time.monotonic()
        if now - self._last_frame_time >= 1.0:
            self._fps = self._frame_counter / (now - self._last_frame_time)
            self._frame_counter = 0
            self._last_frame_time = now

        ws = snapshot.get("world_state") or {}
        self.footer.set_snapshot(snapshot, self._fps)
        connected = "●  Connected" if snapshot.get("updated_at") else "●  Waiting"
        self.connection.setText(connected)

        if not self._demo:
            try:
                import state as state_module
                from state import State

                if state_module.is_state(State.SHUTDOWN):
                    self.close_from_shutdown()
            except Exception:
                pass


class ChromePanel(QFrame):
    def __init__(self, index: str, title: str, content: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("chromePanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(22, 18, 18, 16)
        header.setSpacing(12)
        if index:
            badge = QLabel(index)
            badge.setObjectName("panelBadge")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.addWidget(badge)
        label = QLabel(title)
        label.setObjectName("panelTitle")
        header.addWidget(label)
        header.addStretch(1)
        layout.addLayout(header)

        separator = QFrame()
        separator.setObjectName("panelSeparator")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        layout.addWidget(content, 1)


class VisionDescriptionPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._description = ""
        self.setMinimumHeight(220)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        ws = snapshot.get("world_state") or {}
        env = ws.get("environment") or {}
        self._description = (
            snapshot.get("scene_description")
            or env.get("description")
            or "Vision description will appear after Rex has a scene read."
        )
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07111a"))

        x = 18
        y = 28
        painter.setPen(QColor("#dbe7f3"))
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x, y, "R3X sees:")

        font.setBold(False)
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(QColor("#d8e2ee"))
        text_rect = self.rect().adjusted(18, 52, -18, -18)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self._description,
        )
        painter.end()


class ServoPositionsPanel(QWidget):
    _ORDER = ("neck", "headlift", "headtilt", "visor", "elbow", "hand", "pokerarm", "heroarm")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._state_labels: dict[str, QLabel] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(9)

        visual_row = 0
        for row, name in enumerate(self._ORDER):
            if row == 4:
                line = QFrame()
                line.setObjectName("panelSeparator")
                line.setFixedHeight(1)
                layout.addWidget(line, visual_row, 0, 1, 4)
                visual_row += 1

            label = QLabel(_servo_label(name))
            label.setObjectName("servoName")
            layout.addWidget(label, visual_row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setEnabled(False)
            slider.setRange(0, 1000)
            slider.setObjectName("servoSlider")
            layout.addWidget(slider, visual_row, 1)
            self._sliders[name] = slider

            value = QLabel("")
            value.setObjectName("servoValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(value, visual_row, 2)
            self._value_labels[name] = value

            state = QLabel("")
            state.setObjectName("servoState")
            state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(state, visual_row, 3)
            self._state_labels[name] = state
            visual_row += 1

        layout.setColumnStretch(1, 1)
        self.setMinimumHeight(190)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        ws = snapshot.get("world_state") or {}
        self_state = ws.get("self_state") or ws.get("self") or {}
        positions = dict(snapshot.get("servo_positions") or {})
        positions.update(self_state.get("servo_positions") or {})

        for name in self._ORDER:
            cfg = config.SERVO_CHANNELS[name]
            raw = int(positions.get(name, cfg["neutral"]))
            norm = normalize_servo(name, raw)
            self._sliders[name].setValue(int(norm * 1000))
            self._value_labels[name].setText(str(raw))
            self._state_labels[name].setText(_servo_state(name, raw))


class FooterBar(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("footerBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 10)
        layout.setSpacing(18)
        self.status = QLabel()
        self.battery = QLabel()
        self.system = QLabel()
        self.settings = QPushButton("⚙  Settings")
        self.settings.setObjectName("settingsButton")
        for widget in (self.status, self.battery, self.system):
            widget.setObjectName("footerText")
        layout.addWidget(self.status)
        layout.addStretch(1)
        layout.addWidget(self.battery)
        layout.addStretch(1)
        layout.addWidget(self.system)
        layout.addStretch(1)
        layout.addWidget(self.settings)

    def set_snapshot(self, snapshot: dict[str, Any], fps: float) -> None:
        ws = snapshot.get("world_state") or {}
        state = str(ws.get("state") or "IDLE").upper()
        self_state = ws.get("self_state") or ws.get("self") or {}
        battery = self_state.get("battery_voltage") or ws.get("battery_voltage") or "12.4V"
        health = self_state.get("body_state") or ws.get("system") or "Nominal"
        suffix = f"  GUI {fps:.0f} FPS" if bool(getattr(config, "GUI_SHOW_FPS", False)) else ""
        self.status.setText(f'R3X Status: <span style="color:#4dde67;font-weight:800;">{state}</span>{suffix}')
        self.battery.setText(f'Battery: {battery}  <span style="color:#4dde67;font-weight:800;">▰ 75%</span>')
        self.system.setText(f'System: <span style="color:#4dde67;font-weight:800;">{health}</span>')


def run_dashboard(
    bridge: GUIDashboardBridge = gui_bridge,
    *,
    shutdown_callback: Optional[Callable[[], None]] = None,
    demo: bool = False,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = DashboardWindow(
        bridge,
        shutdown_callback=shutdown_callback,
        demo=demo,
    )
    window.show()

    def _sigint(_signum, _frame) -> None:
        if shutdown_callback is not None:
            shutdown_callback()
        window.close_from_shutdown()

    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint)
    try:
        return int(app.exec())
    finally:
        signal.signal(signal.SIGINT, old_handler)


def _advance_demo(bridge: GUIDashboardBridge) -> None:
    now = time.monotonic()
    frame = _demo_frame(now)
    people = [
        {
            "id": "person_1",
            "face_id": "person 0.94",
            "face_box": (78, 112, 235, 285),
            "expression": _demo_expression(now, 0),
            "engagement": "tracking",
            "distance_zone": "social",
            "pose": "standing",
        },
        {
            "id": "person_2",
            "face_id": "person 0.91",
            "face_box": (606, 132, 205, 268),
            "expression": _demo_expression(now, 1),
            "engagement": "tracking",
            "distance_zone": "social",
            "pose": "walking away",
        },
    ]
    servo_positions = {}
    for name, cfg in config.SERVO_CHANNELS.items():
        phase = {
            "neck": 0.0,
            "headlift": 0.7,
            "headtilt": 1.4,
            "visor": 2.1,
            "elbow": 3.5,
            "hand": 4.2,
            "pokerarm": 4.9,
            "heroarm": 2.8,
        }.get(name, random.random())
        norm = 0.5 + math.sin(now * 0.55 + phase) * 0.28
        servo_positions[name] = int(cfg["min"] + (cfg["max"] - cfg["min"]) * norm)

    bridge.update_frame(frame)
    bridge.update_world_state_snapshot({
        "state": "IDLE",
        "people": people,
        "environment": {
            "description": (
                "An indoor office or lounge space with two people. Person 1 is a man "
                "wearing a black shirt and dark pants, standing on the left side of "
                "the room, facing away. Person 2 is a woman wearing a light colored "
                "sweater and jeans, walking away on the right side of the room. There "
                "is a couch on the left, a coffee table with a plant, desks and chairs "
                "in the background, and a poster on the far wall."
            ),
        },
        "self_state": {
            "servo_positions": servo_positions,
            "body_state": "Nominal",
            "battery_voltage": "12.4V",
        },
    })
    if getattr(_advance_demo, "_led_seeded", False) is False:
        _advance_demo._led_seeded = True  # type: ignore[attr-defined]
        bridge.update_head_led_state(mode="idle", eye_color=(45, 115, 255), eyes_active=True)

    if getattr(_advance_demo, "_seeded", False) is False:
        _advance_demo._seeded = True  # type: ignore[attr-defined]
        samples = [
            ("Human", "Hey R3X, how are you doing today?", "user"),
            ("R3X", "I'm functioning within normal parameters! Systems nominal and ready to assist.", "rex"),
            ("Human", "What do you see right now?", "user"),
            ("R3X", "I see an indoor office space with two people, furniture, and computer equipment.", "rex"),
            ("Human", "Can you wave hello?", "user"),
            ("R3X", "Certainly!", "rex"),
            ("R3X", "*waves right arm*", "rex"),
            ("Human", "Nice! What's on the poster?", "user"),
            ("R3X", "The poster appears to be colorful sci-fi or fantasy artwork with a character in the center.", "rex"),
        ]
        for speaker, text, kind in samples:
            bridge.add_conversation_line(speaker, text, kind)


def _demo_frame(now: float) -> np.ndarray:
    h, w = 540, 820
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (54, 58, 58)
    frame[:120, :] = (68, 72, 73)
    for x in range(0, w, 82):
        frame[:120, x:x + 2] = (82, 86, 88)
    for y in range(0, 120, 30):
        frame[y:y + 2, :] = (82, 86, 88)
    frame[170:410, 0:240] = (52, 58, 61)
    frame[250:270, 260:520] = (65, 70, 72)
    frame[270:390, 305:470] = (36, 42, 43)
    frame[0:h, :, :] = np.clip(frame + np.linspace(14, -22, h, dtype=np.int16)[:, None, None], 0, 255)
    _draw_demo_person(frame, 150 + int(math.sin(now * 0.4) * 3), 280, (20, 24, 28), (168, 118, 86))
    _draw_demo_person(frame, 700, 270, (178, 170, 150), (45, 48, 56))
    return frame


def _demo_expression(now: float, offset: int) -> str:
    labels = ("neutral", "smiling", "curious", "focused", "surprised")
    return labels[(int(now / 2.5) + offset) % len(labels)]


def _draw_demo_person(frame: np.ndarray, cx: int, cy: int, shirt: tuple[int, int, int], hair: tuple[int, int, int]) -> None:
    y0 = max(0, cy - 150)
    y1 = min(frame.shape[0], cy + 150)
    x0 = max(0, cx - 44)
    x1 = min(frame.shape[1], cx + 44)
    frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1] - 8, 0)
    frame[cy - 96:cy + 70, cx - 36:cx + 36] = shirt
    frame[cy + 70:cy + 148, cx - 34:cx - 8] = (32, 42, 54)
    frame[cy + 70:cy + 148, cx + 8:cx + 34] = (32, 42, 54)
    frame[cy - 136:cy - 100, cx - 20:cx + 20] = (132, 92, 70)
    frame[cy - 150:cy - 122, cx - 24:cx + 24] = hair


def _servo_label(name: str) -> str:
    cfg = config.SERVO_CHANNELS[name]
    labels = {
        "neck": "Neck",
        "headlift": "Headlift",
        "headtilt": "Headtilt",
        "visor": "Visor",
        "elbow": "Elbow",
        "hand": "Hand",
        "pokerarm": "Pokerarm",
        "heroarm": "Heroarm",
    }
    return f"{labels.get(name, name.title())} ({cfg['ch']})"


def _servo_state(name: str, value: int) -> str:
    if name == "visor":
        return "Open" if normalize_servo(name, value) >= 0.45 else "Closed"
    if name == "headlift":
        return f"{servo_to_offset(name, value):+.0f}mm"
    return f"{servo_to_angle(name, value):+.0f}°"


_STYLE = """
QWidget#root {
    background: #07111a;
    color: #d9e3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
QLabel#trafficLights {
    color: #ff6159;
    font-size: 15px;
    font-weight: 800;
}
QLabel#windowTitle {
    color: #aab5c1;
    font-size: 15px;
    font-weight: 800;
}
QLabel#connectionLabel {
    color: #45d85e;
    font-size: 13px;
}
QFrame#chromePanel, QFrame#footerBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b1824, stop:1 #08111a);
    border: 1px solid #255484;
    border-radius: 7px;
}
QFrame#panelSeparator {
    background: rgba(66, 105, 145, 0.48);
    border: none;
}
QLabel#panelBadge {
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    border-radius: 5px;
    background: #3b7fd9;
    color: white;
    font-size: 18px;
    font-weight: 900;
}
QLabel#panelTitle {
    color: #4e94ff;
    font-size: 18px;
    font-weight: 900;
}
QTextBrowser#conversationLog {
    background: #07111a;
    color: #d9e3ee;
    border: none;
}
QLineEdit#messageEntry {
    min-height: 40px;
    padding: 0 14px;
    background: #111b27;
    color: #e0e9f2;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-size: 13px;
}
QPushButton#primaryButton {
    min-height: 40px;
    padding: 0 18px;
    background: #326bbe;
    color: white;
    border: 1px solid #4e8be4;
    border-radius: 5px;
    font-weight: 800;
}
QPushButton#settingsButton {
    padding: 7px 12px;
    background: #111b27;
    color: #e0e8f0;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 700;
}
QLabel#footerText, QLabel#servoName, QLabel#servoValue, QLabel#servoState {
    color: #d6e0ea;
    font-size: 13px;
}
QLabel#servoName {
    font-weight: 700;
}
QLabel#servoValue, QLabel#servoState {
    color: #b8c3d0;
}
QSlider#servoSlider::groove:horizontal {
    height: 4px;
    background: #25303b;
    border-radius: 2px;
}
QSlider#servoSlider::sub-page:horizontal {
    background: #4d8dea;
    border-radius: 2px;
}
QSlider#servoSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #3f7fd8;
}
QScrollBar:vertical {
    background: #07111a;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #657384;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DJ-R3X optional GUI dashboard")
    parser.add_argument("--demo", action="store_true", help="run dashboard with simulated state")
    args = parser.parse_args(argv)
    if not args.demo:
        print("Use --demo to run the dashboard outside main.py.", file=sys.stderr)
    return run_dashboard(gui_bridge, demo=args.demo)


if __name__ == "__main__":
    raise SystemExit(main())
