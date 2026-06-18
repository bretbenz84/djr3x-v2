"""Optional PySide6 dashboard for DJ-R3X."""

from __future__ import annotations

import argparse
import html
import logging
import math
import random
import signal
import sys
import time
from typing import Any, Callable, Optional

try:
    from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QSlider,
        QStackedWidget,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - exercised when dependency missing.
    raise RuntimeError(f"PySide6 is unavailable: {exc}") from exc

import numpy as np

import config
from gui.conversation_panel import ConversationPanel
from gui.jeopardy_panel import JeopardyPanel
from gui.log_panel import LogPanel
from gui.rex_avatar import RexAvatar, normalize_servo, servo_to_angle, servo_to_offset
from gui.state_bridge import GUIDashboardBridge, gui_bridge
from gui.vision_panel import VisionPanel

_log = logging.getLogger(__name__)

# Live hardware-connection indicators shown in the top bar (key, label, tooltip).
_DEVICE_SPECS = (
    ("chest", "Chest LEDs", "Chest LEDs — Arduino Nano (ARDUINO_CHEST_PORT)"),
    ("head", "Head LEDs", "Head LEDs — Arduino Uno (ARDUINO_HEAD_PORT)"),
    ("maestro", "Maestro", "Pololu Maestro servo controller (MAESTRO_PORT)"),
    ("motor", "ESP32 Motor", "ESP32 motor controller — drive base (MOTION_ESP32_PORT)"),
)


def _device_status_color(enabled: bool, connected: bool) -> str:
    if not enabled:
        return "#5b6b7d"   # gray  — not configured / disabled
    if connected:
        return "#45d85e"   # green — connected
    return "#ff6b5e"       # red   — configured but offline


class DashboardWindow(QMainWindow):
    def __init__(
        self,
        bridge: GUIDashboardBridge,
        *,
        shutdown_callback: Optional[Callable[[], None]] = None,
        text_submit_callback: Optional[Callable[[str], None]] = None,
        sleep_callback: Optional[Callable[[], None]] = None,
        wake_callback: Optional[Callable[[], None]] = None,
        demo: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._shutdown_callback = shutdown_callback
        self._sleep_callback = sleep_callback
        self._wake_callback = wake_callback
        self._demo = demo
        self._closing_from_shutdown = False
        self._shutdown_requested = False
        self._is_asleep = False
        self._last_pause_glyph = None
        self._last_sleep_label = None

        self.setWindowTitle(getattr(config, "GUI_WINDOW_TITLE", "DJ-R3X Controller"))
        self.resize(1280, 840)
        self.setMinimumSize(1100, 740)

        self.vision = VisionPanel()
        self.scene = VisionDescriptionPanel()
        self.avatar = RexAvatar()
        self.servos = ServoPositionsPanel()
        self.conversation = ConversationPanel()
        if text_submit_callback is not None:
            self.conversation.set_submit_callback(text_submit_callback)
        else:
            self.conversation.set_submit_callback(
                lambda text: self._bridge.add_conversation_line("Human", text, "user")
            )
        self.jeopardy = JeopardyPanel()
        self.syslog = LogPanel()
        self.connection = QLabel("●  Booting…")
        self.connection.setObjectName("connectionLabel")
        self._last_status_text = ""
        self.state_badge = QLabel("—")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._last_badge_text = ""

        root = QWidget()
        root.setObjectName("root")
        self._shell = QVBoxLayout(root)
        self._shell.setContentsMargins(14, 8, 14, 14)
        self._shell.setSpacing(12)

        self._top_bar = QWidget()
        # 3-column grid so the title can be centered against the FULL window width
        # (it spans all columns), independent of how wide the left controls or the
        # right connection label are.
        top = QGridLayout(self._top_bar)
        top.setContentsMargins(0, 0, 0, 0)

        # Left cluster: state badge + control buttons, hugging the left edge.
        left_cluster = QWidget()
        cluster = QHBoxLayout(left_cluster)
        cluster.setContentsMargins(0, 0, 0, 0)
        cluster.setSpacing(0)
        cluster.addWidget(self.state_badge)
        self.memory_banks_btn = QPushButton("🧠  Memory Banks")
        self.memory_banks_btn.setObjectName("memoryBanksButton")
        self.memory_banks_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.memory_banks_btn.clicked.connect(self._open_memory_banks)
        cluster.addWidget(self.memory_banks_btn)

        # Play/Pause — shows the pause glyph while running (click to pause). Uses the
        # same INTERACTION_PAUSED mechanism the Memory Banks editor uses.
        self._pause_btn = QPushButton("⏸")
        self._pause_btn.setObjectName("topControlButton")
        self._pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_btn.setToolTip("Pause the robot")
        self._pause_btn.clicked.connect(self._toggle_pause)
        cluster.addWidget(self._pause_btn)

        # Sleep / Wake — label flips with the runtime state.
        self._sleep_btn = QPushButton("Sleep R3X")
        self._sleep_btn.setObjectName("topControlButton")
        self._sleep_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sleep_btn.setToolTip("Put DJ-R3X to sleep")
        self._sleep_btn.clicked.connect(self._toggle_sleep)
        cluster.addWidget(self._sleep_btn)

        # Shut Down — confirmed, then exits the program.
        self._shutdown_btn = QPushButton("⏻  Shut Down")
        self._shutdown_btn.setObjectName("topShutdownButton")
        self._shutdown_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._shutdown_btn.setToolTip("Shut down the program")
        self._shutdown_btn.clicked.connect(self._confirm_shutdown)
        cluster.addWidget(self._shutdown_btn)

        title = QLabel("DJ-R3X Controller")
        title.setObjectName("windowTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Spans the whole bar on top of the side groups; let clicks fall through to
        # the buttons beneath it.
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Right cluster: live device-connection indicators, then the overall
        # connection status — hugging the right edge (after the centered title,
        # before "Booting…"). Colors update live in _tick.
        right_cluster = QWidget()
        rc = QHBoxLayout(right_cluster)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(14)
        self._device_status: dict[str, QLabel] = {}
        self._last_device_color: dict[str, str] = {}
        for key, label_text, tip in _DEVICE_SPECS:
            lbl = QLabel()
            lbl.setObjectName("deviceStatus")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setToolTip(tip)
            lbl.setText(f'<span style="color:#5b6b7d;">●</span>&nbsp;{label_text}')
            rc.addWidget(lbl)
            self._device_status[key] = lbl
        rc.addWidget(self.connection)

        top.addWidget(left_cluster, 0, 0,
                      Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(right_cluster, 0, 2,
                      Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(title, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        top.setColumnStretch(0, 0)
        top.setColumnStretch(1, 1)
        top.setColumnStretch(2, 0)
        self._shell.addWidget(self._top_bar)
        self._memory_banks_window = None

        columns = QGridLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setHorizontalSpacing(12)
        columns.setVerticalSpacing(12)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(ChromePanel("", "VISION", self.vision), 4)
        left.addWidget(ChromePanel("", "OPENAI VISION + DLIB STATE", self.scene), 8)
        left_box = QWidget()
        left_box.setLayout(left)

        center = ChromePanel("☵", "CONVERSATION LOG", self.conversation)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        avatar_panel = ChromePanel("", "R3X AVATAR", self.avatar)
        servo_panel = ChromePanel("", "SERVO POSITIONS", self.servos)
        servo_panel.setMinimumHeight(410)
        servo_panel.setMaximumHeight(470)
        right.addWidget(avatar_panel, 1)
        right.addWidget(servo_panel, 0)
        right_box = QWidget()
        right_box.setLayout(right)
        # The avatar paints proportionally and the servo panel is fixed-width, so
        # the right column never needs to be the widest. Cap it and hand the
        # surplus to the conversation log (center), which was cramped before.
        right_box.setMaximumWidth(640)

        columns.addWidget(left_box, 0, 0)
        columns.addWidget(center, 0, 1)
        columns.addWidget(right_box, 0, 2)
        columns.setColumnStretch(0, 12)
        columns.setColumnStretch(1, 16)
        columns.setColumnStretch(2, 12)
        columns_box = QWidget()
        columns_box.setLayout(columns)

        log_title = "SYSTEM LOG"
        try:
            from utils.logging import active_log_path

            log_title = f"SYSTEM LOG — {active_log_path().name}"
        except Exception:
            pass
        syslog_panel = ChromePanel("≣", log_title, self.syslog)
        # Min low enough that the strip yields to the 3-column grid on small
        # windows instead of starving it (grid panels have their own minimums).
        syslog_panel.setMinimumHeight(120)
        syslog_panel.setMaximumHeight(280)

        page = QVBoxLayout()
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(12)
        page.addWidget(columns_box, 1)
        page.addWidget(syslog_panel, 0)
        dashboard_page = QWidget()
        dashboard_page.setLayout(page)

        self._main_stack = QStackedWidget()
        self._main_stack.addWidget(dashboard_page)
        self._main_stack.addWidget(self.jeopardy)
        self._shell.addWidget(self._main_stack, 1)

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

    def _open_memory_banks(self) -> None:
        """Open the Memory Banks editor in its own window. Pauses robot audio output
        while open (handled by the window); closing it leaves the program running."""
        try:
            existing = self._memory_banks_window
            if existing is not None and existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return
            from gui.memory_banks import MemoryBanksWindow
            self._memory_banks_window = MemoryBanksWindow(self)
            self._memory_banks_window.show()
            self._memory_banks_window.raise_()
            self._memory_banks_window.activateWindow()
        except Exception as exc:
            _log.warning("failed to open Memory Banks window: %s", exc)

    # ── Top-bar robot controls ────────────────────────────────────────────────
    def _sync_top_controls(self, paused: bool, asleep: bool) -> None:
        """Reflect the live pause/sleep state on the play-pause and sleep buttons,
        so they stay correct even when changed by voice or the Memory Banks editor."""
        glyph = "▶" if paused else "⏸"
        if glyph != self._last_pause_glyph:
            self._last_pause_glyph = glyph
            self._pause_btn.setText(glyph)
            self._pause_btn.setToolTip("Resume the robot" if paused else "Pause the robot")
        label = "Wake R3X" if asleep else "Sleep R3X"
        if label != self._last_sleep_label:
            self._last_sleep_label = label
            self._sleep_btn.setText(label)
            self._sleep_btn.setToolTip("Wake DJ-R3X" if asleep else "Put DJ-R3X to sleep")

    def _toggle_pause(self) -> None:
        paused = not bool(getattr(config, "INTERACTION_PAUSED", False))
        config.INTERACTION_PAUSED = paused
        _log.info("[dashboard] interaction %s via pause button", "paused" if paused else "resumed")
        self._sync_top_controls(paused, self._is_asleep)   # immediate feedback

    def _toggle_sleep(self) -> None:
        if self._is_asleep:
            if self._wake_callback is not None:
                self._wake_callback()
        else:
            if self._sleep_callback is not None:
                self._sleep_callback()
        # The button label follows the real runtime state via _tick.

    def _confirm_shutdown(self) -> None:
        dlg = QDialog(self)
        dlg.setObjectName("confirmDialog")
        dlg.setWindowTitle("Shut Down DJ-R3X")
        dlg.setModal(True)
        dlg.setStyleSheet(_STYLE + _CONFIRM_EXTRA)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 22, 24, 18)
        lay.setSpacing(18)
        msg = QLabel("Shut down DJ-R3X?\nThis will stop the program.")
        msg.setObjectName("confirmText")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(msg)
        row = QHBoxLayout()
        row.setSpacing(12)
        no_btn = QPushButton("No")
        no_btn.setObjectName("confirmNo")
        no_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        no_btn.clicked.connect(dlg.reject)
        yes_btn = QPushButton("Yes, shut down")
        yes_btn.setObjectName("confirmYes")
        yes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        yes_btn.clicked.connect(dlg.accept)
        row.addStretch(1)
        row.addWidget(no_btn)
        row.addWidget(yes_btn)
        lay.addLayout(row)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            _log.info("[dashboard] shutdown confirmed via top-bar button")
            if self._shutdown_callback is not None:
                self._shutdown_callback()

    def _device_states(self) -> "dict[str, tuple[bool, bool]]":
        """Live (enabled, connected) per device. Read directly from the hardware
        modules each tick so the indicators reflect connect/disconnect."""
        states = {k: (False, False) for k, _l, _t in _DEVICE_SPECS}
        try:
            from utils.config_loader import (
                SERVOS_ENABLED, HEAD_LEDS_ENABLED, CHEST_LEDS_ENABLED, MOTION_PORT_SET,
            )
            from hardware import servos, leds_head, leds_chest, motion

            def _ok(fn) -> bool:
                try:
                    return bool(fn())
                except Exception:
                    return False

            states["chest"] = (bool(CHEST_LEDS_ENABLED), _ok(leds_chest.connected))
            states["head"] = (bool(HEAD_LEDS_ENABLED), _ok(leds_head.connected))
            states["maestro"] = (bool(SERVOS_ENABLED), _ok(servos.connected))
            states["motor"] = (
                bool(MOTION_PORT_SET and getattr(config, "MOTION_ENABLED", True)),
                _ok(motion.connected),
            )
        except Exception:
            pass
        return states

    def _update_device_status(self) -> None:
        states = self._device_states()
        for key, label_text, _tip in _DEVICE_SPECS:
            enabled, conn = states.get(key, (False, False))
            color = _device_status_color(enabled, conn)
            if self._last_device_color.get(key) == color:
                continue
            self._last_device_color[key] = color
            lbl = self._device_status.get(key)
            if lbl is not None:
                lbl.setText(f'<span style="color:{color};">●</span>&nbsp;{label_text}')

    def close_from_shutdown(self) -> None:
        self._closing_from_shutdown = True
        try:
            if self._memory_banks_window is not None:
                self._memory_banks_window.close()
        except Exception:
            pass
        self._stop_timers()
        self.close()

    def request_shutdown(self) -> None:
        if self._shutdown_requested:
            self.close_from_shutdown()
            return
        self._shutdown_requested = True
        if self._shutdown_callback is not None:
            try:
                self._shutdown_callback()
            except Exception as exc:
                _log.warning("GUI shutdown callback failed: %s", exc)
        self.close_from_shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._stop_timers()
        if (
            not self._closing_from_shutdown
            and not self._shutdown_requested
            and self._shutdown_callback is not None
        ):
            self._shutdown_requested = True
            try:
                self._shutdown_callback()
            except Exception as exc:
                _log.warning("GUI shutdown callback failed: %s", exc)
        super().closeEvent(event)

    def _stop_timers(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._demo_timer is not None and self._demo_timer.isActive():
            self._demo_timer.stop()

    def _tick(self) -> None:
        # Check SHUTDOWN before reading the snapshot: the startup worker marks
        # the bridge "failed" strictly BEFORE setting SHUTDOWN, so when the
        # check below is true a failed boot is already visible in the snapshot.
        shutdown_requested = not self._demo and self._runtime_shutdown_requested()
        snapshot = self._bridge.get_snapshot()
        status = str(snapshot.get("controller_status") or "").strip().lower()
        if shutdown_requested and status != "failed":
            self.close_from_shutdown()
            return
        # On a failed boot the window stays open so the red status and the
        # system-log panel remain readable; closing it resumes teardown.
        self.vision.set_snapshot(snapshot)
        self.scene.set_snapshot(snapshot)
        self.avatar.set_snapshot(snapshot)
        self.servos.set_snapshot(snapshot)
        self.conversation.set_snapshot(snapshot)
        self.syslog.set_snapshot(snapshot)
        self.jeopardy.set_snapshot(snapshot)
        game_state = snapshot.get("game_state") or {}
        jeopardy_active = game_state.get("active_game") == "jeopardy"
        if not jeopardy_active and status == "starting":
            # --jeopardy launch: open on the game page during boot rather than
            # showing the diagnostic dashboard to the audience.
            jeopardy_active = (
                str(snapshot.get("startup_game_intent") or "") == "jeopardy"
            )
        if jeopardy_active:
            self._main_stack.setCurrentWidget(self.jeopardy)
        else:
            self._main_stack.setCurrentIndex(0)
        self._top_bar.setVisible(not jeopardy_active)
        if jeopardy_active:
            self._shell.setContentsMargins(0, 0, 0, 0)
            self._shell.setSpacing(0)
        else:
            self._shell.setContentsMargins(14, 8, 14, 14)
            self._shell.setSpacing(12)

        if status == "starting":
            text, color = "●  Booting…", "#f0c45a"
        elif status == "failed":
            text, color = "●  Startup failed — see system log", "#ff6b5e"
        elif snapshot.get("updated_at"):
            text, color = "●  Connected", "#45d85e"
        else:
            text, color = "●  Waiting", "#8d9aab"
        if text != self._last_status_text:
            self._last_status_text = text
            self.connection.setText(text)
            self.connection.setStyleSheet(f"color: {color}; font-size: 13px;")

        self._update_device_status()

        ws = snapshot.get("world_state") or {}
        speaking = bool((snapshot.get("speech_state") or {}).get("speaking"))
        paused = bool(getattr(config, "INTERACTION_PAUSED", False))
        self._is_asleep = str(ws.get("state") or "").strip().upper() == "SLEEP"
        self._sync_top_controls(paused, self._is_asleep)
        badge_text, badge_color = _state_badge_spec(ws.get("state"), speaking, paused)
        if badge_text != self._last_badge_text:
            self._last_badge_text = badge_text
            self.state_badge.setText(badge_text)
            self.state_badge.setStyleSheet(
                f"color:{badge_color}; border:1px solid {badge_color};"
                " border-radius:5px; padding:3px 13px;"
                " font-size:14px; font-weight:900;"
            )

    def _runtime_shutdown_requested(self) -> bool:
        try:
            import state as state_module
            from state import State

            return bool(state_module.is_state(State.SHUTDOWN))
        except Exception:
            return False


def _state_badge_spec(state_value: Any, speaking: bool, paused: bool = False) -> tuple[str, str]:
    """Map the runtime State (+ live speech / pause) to a top-bar badge label and color.

    SPEAKING overlays whichever conversational state is active; PAUSE wins over it;
    SLEEP/SHUTDOWN win over PAUSE. world_state['state'] is set by main's GUI bridge
    sync; `paused` reflects config.INTERACTION_PAUSED (the Memory Banks / pause-button
    mechanism)."""
    s = str(state_value or "").strip().upper()
    if s == "SHUTDOWN":
        return "SHUTDOWN", "#ff6b5e"
    if s == "SLEEP":
        return "SLEEP", "#8d9aab"
    if paused:
        return "PAUSE", "#e8a13c"
    if speaking:
        return "SPEAKING", "#ff9b21"
    if s == "ACTIVE":
        return "ACTIVE", "#45d85e"
    if s == "QUIET":
        return "QUIET", "#f0c45a"
    if s == "IDLE":
        return "IDLE", "#5396ff"
    return "—", "#5b6b7d"


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
        self._last_html = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(0)
        self._body = QTextBrowser()
        self._body.setObjectName("visionDescription")
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setOpenExternalLinks(False)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._body, 1)
        self.setMinimumHeight(300)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        rendered = _vision_state_html(snapshot)
        if rendered != self._last_html:
            self._last_html = rendered
            self._body.setHtml(rendered)


def _vision_state_html(snapshot: dict[str, Any]) -> str:
    ws = snapshot.get("world_state") or {}
    env = ws.get("environment") or {}
    description = (
        snapshot.get("scene_description")
        or env.get("description")
        or "Vision description will appear after Rex has a scene read."
    )
    people = [
        dict(person)
        for person in (ws.get("people") or [])
        if isinstance(person, dict)
    ]
    animals = [
        dict(animal)
        for animal in (ws.get("animals") or [])
        if isinstance(animal, dict)
    ]
    self_state = ws.get("self_state") or ws.get("self") or {}
    face_tracking = self_state.get("face_tracking") or {}

    visible_count = sum(1 for person in people if _face_visible(person))
    known_count = sum(
        1
        for person in people
        if _face_visible(person) and _person_known(person)
    )
    unknown_count = max(0, visible_count - known_count)
    summary = " / ".join(
        [
            _count_label(len(people), "slot"),
            _count_label(visible_count, "visible face"),
            f"{known_count} known",
            f"{unknown_count} unknown",
            _count_label(len(animals), "animal"),
        ]
    )

    tracking_html = _tracking_html(face_tracking)
    if people:
        people_html = "".join(
            _person_dlib_html(idx, person)
            for idx, person in enumerate(people, start=1)
        )
    else:
        people_html = '<p class="empty">No dlib face slots yet.</p>'
    animals_html = _animals_html(animals)

    return f"""
<html>
<head>
<style>
body {{
  margin: 0;
  background: #07111a;
  color: #d9e3ee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 12px;
}}
.section {{
  margin: 0 0 14px 0;
}}
.eyebrow {{
  color: #4e94ff;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0;
  text-transform: uppercase;
}}
.description {{
  margin-top: 5px;
  color: #d8e2ee;
  font-size: 12px;
  line-height: 1.35;
}}
.summary {{
  margin: 5px 0 8px 0;
  color: #aebccc;
  font-weight: 700;
}}
.face {{
  margin: 8px 0 0 0;
  padding: 8px 0 0 0;
  border-top: 1px solid #233b55;
}}
.face-title {{
  color: #e7f0fa;
  font-size: 12px;
  font-weight: 900;
}}
.status-visible {{
  color: #75ef63;
}}
.status-held {{
  color: #f0c45a;
}}
.status-missing {{
  color: #ff8a72;
}}
table.kv {{
  margin-top: 4px;
  border-collapse: collapse;
}}
td.key {{
  padding: 1px 14px 1px 0;
  color: #6fa0dc;
  font-weight: 800;
  white-space: nowrap;
}}
td.value {{
  padding: 1px 0;
  color: #ccd8e5;
}}
.empty {{
  margin: 6px 0 0 0;
  color: #91a4b8;
}}
</style>
</head>
<body>
  <div class="section">
    <div class="eyebrow">Vision Summary</div>
    <div class="description">{_html(description)}</div>
  </div>
  <div class="section">
    <div class="eyebrow">dlib + Expression State</div>
    <div class="summary">{_html(summary)}</div>
    {tracking_html}
    {people_html}
  </div>
  <div class="section">
    <div class="eyebrow">Local Object State</div>
    {animals_html}
  </div>
</body>
</html>
"""


def _tracking_html(face_tracking: dict[str, Any]) -> str:
    if not isinstance(face_tracking, dict) or not face_tracking:
        return '<p class="empty">Face tracker idle.</p>'

    if face_tracking.get("locked"):
        status = "locked"
    elif face_tracking.get("holding_lost_lock"):
        status = "holding lost lock"
    elif face_tracking.get("searching"):
        status = "searching"
    elif face_tracking.get("visible"):
        status = "visible"
    else:
        status = "idle"

    lost_age = _coerce_float(face_tracking.get("lost_age_secs"))
    rows = [
        ("tracking", status),
        ("target", face_tracking.get("lock_key")),
        ("person id", face_tracking.get("person_id")),
        (
            "visible",
            _yes_no(face_tracking.get("visible"))
            if face_tracking.get("visible") is not None
            else None,
        ),
        ("lost age", _format_age(lost_age) if lost_age is not None else None),
        ("search", face_tracking.get("search_reason")),
        ("search pose", face_tracking.get("search_pose")),
    ]
    return '<div class="face">' + _kv_table(rows) + "</div>"


def _person_dlib_html(idx: int, person: dict[str, Any]) -> str:
    label = _person_display_name(person, idx)
    status, status_class = _person_face_status(person)
    rows = [
        ("status", status),
        ("db id", person.get("person_db_id")),
        ("face id", person.get("face_id")),
        ("voice id", person.get("voice_id")),
        ("box", _format_box(person)),
        ("center", _format_position(person.get("position"))),
        ("face width", _format_face_fraction(person.get("face_box_fraction"))),
        ("distance", _clean_text(person.get("distance_zone"))),
        ("approach", _clean_text(person.get("approach_vector"))),
        ("pose", _clean_text(person.get("pose"))),
        ("gesture", _clean_text(person.get("gesture"))),
        ("engagement", _clean_text(person.get("engagement"))),
        ("expression", _format_expression(person)),
        ("mood", _format_mood(person)),
        ("last seen", _last_seen_label(person)),
    ]
    return (
        '<div class="face">'
        f'<div class="face-title">{_html(label)} '
        f'<span class="{status_class}">[{_html(status)}]</span></div>'
        f"{_kv_table(rows)}"
        "</div>"
    )


def _animals_html(animals: list[dict[str, Any]]) -> str:
    if not animals:
        return '<p class="empty">No local animals detected.</p>'
    return "".join(
        _animal_html(idx, animal)
        for idx, animal in enumerate(animals, start=1)
    )


def _animal_html(idx: int, animal: dict[str, Any]) -> str:
    species = _clean_text(animal.get("species")) or f"animal {idx}"
    rows = [
        ("species", species),
        ("position", _clean_text(animal.get("position"))),
        ("box", _format_box(animal)),
        ("confidence", _format_confidence(animal.get("confidence"))),
        ("furred", _yes_no(animal.get("furred")) if animal.get("furred") is not None else None),
        ("source", _clean_text(animal.get("source"))),
        ("last seen", _animal_last_seen_label(animal)),
    ]
    return (
        '<div class="face">'
        f'<div class="face-title">{_html(species.title())} '
        '<span class="status-visible">[animal]</span></div>'
        f"{_kv_table(rows)}"
        "</div>"
    )


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    rendered = []
    for key, value in rows:
        text = _clean_text(value)
        if not text:
            continue
        rendered.append(
            "<tr>"
            f'<td class="key">{_html(key)}</td>'
            f'<td class="value">{_html(text)}</td>'
            "</tr>"
        )
    if not rendered:
        return '<p class="empty">No live face details yet.</p>'
    return '<table class="kv">' + "".join(rendered) + "</table>"


def _person_display_name(person: dict[str, Any], idx: int) -> str:
    for key in ("name", "face_id", "voice_id", "id"):
        value = _clean_text(person.get(key))
        if value:
            return value
    return f"person {idx}"


def _person_face_status(person: dict[str, Any]) -> tuple[str, str]:
    if person.get("face_missing"):
        return ("missing", "status-missing")
    if person.get("face_visible") is True:
        return ("visible", "status-visible")
    if person.get("face_visible") is False:
        return ("held", "status-held")
    if (
        person.get("face_box")
        or person.get("bounding_box")
        or person.get("bbox")
        or person.get("box")
    ):
        return ("visible", "status-visible")
    return ("tracked", "status-held")


def _face_visible(person: dict[str, Any]) -> bool:
    status, _ = _person_face_status(person)
    return status == "visible"


def _person_known(person: dict[str, Any]) -> bool:
    return bool(person.get("person_db_id") or _clean_text(person.get("face_id")))


def _format_box(person: dict[str, Any]) -> str:
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
        return ""
    nums = [_coerce_float(value) for value in box[:4]]
    if any(value is None for value in nums):
        return ""
    x, y, w, h = [float(value) for value in nums]
    return f"{x:.0f},{y:.0f} {w:.0f}x{h:.0f}px"


def _format_position(position: Any) -> str:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return ""
    x = _coerce_float(position[0])
    y = _coerce_float(position[1])
    if x is None or y is None:
        return ""
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return f"{x * 100.0:.0f}%, {y * 100.0:.0f}%"
    return f"{x:.0f}, {y:.0f}px"


def _format_face_fraction(value: Any) -> str:
    fraction = _coerce_float(value)
    if fraction is None:
        return ""
    return f"{max(0.0, fraction) * 100.0:.1f}% of frame"


def _format_mood(person: dict[str, Any]) -> str:
    for key in ("face_mood", "mood", "emotion", "affect"):
        value = person.get(key)
        if isinstance(value, dict):
            mood = _clean_text(
                value.get("mood") or value.get("expression") or value.get("affect")
            )
            confidence = _coerce_float(value.get("confidence"))
            notes = _clean_text(value.get("notes"))
            source = _clean_text(value.get("source"))
            parts = [mood]
            if confidence is not None:
                parts.append(f"{confidence * 100.0:.0f}%")
            if notes:
                parts.append(notes)
            if source:
                parts.append(source.replace("_", " "))
            return " / ".join(part for part in parts if part)
        text = _clean_text(value)
        if text:
            return text
    return ""


def _format_expression(person: dict[str, Any]) -> str:
    for key in ("face_expression", "facial_expression"):
        value = person.get(key)
        if not isinstance(value, dict):
            continue
        expression = _clean_text(value.get("expression") or value.get("affect"))
        mood = _clean_text(value.get("mood"))
        confidence = _coerce_float(value.get("confidence"))
        notes = _clean_text(value.get("notes"))
        source = _clean_text(value.get("source"))
        parts = []
        if expression:
            parts.append(expression)
        if mood and mood != expression:
            parts.append(mood)
        if confidence is not None:
            parts.append(f"{confidence * 100.0:.0f}%")
        if notes:
            parts.append(notes)
        if source:
            parts.append(source.replace("_", " "))
        if parts:
            return " / ".join(parts)

    text = _clean_text(person.get("expression"))
    if text:
        return text
    return ""


def _last_seen_label(person: dict[str, Any]) -> str:
    age = _coerce_float(person.get("face_last_seen_age_secs"))
    if age is None:
        timestamp = _coerce_float(person.get("face_last_seen_at"))
        if timestamp is not None and timestamp > 1_000_000_000:
            age = max(0.0, time.time() - timestamp)
    if age is None:
        return ""
    return _format_age(age)


def _animal_last_seen_label(animal: dict[str, Any]) -> str:
    timestamp = _coerce_float(animal.get("last_seen"))
    if timestamp is None or timestamp <= 1_000_000_000:
        return ""
    return _format_age(time.time() - timestamp)


def _format_confidence(value: Any) -> str:
    score = _coerce_float(value)
    if score is None:
        return _clean_text(value)
    if 0.0 <= score <= 1.0:
        return f"{score * 100.0:.0f}%"
    return f"{score:.3g}"


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    if seconds < 0.75:
        return "now"
    if seconds < 10.0:
        return f"{seconds:.1f}s ago"
    if seconds < 60.0:
        return f"{seconds:.0f}s ago"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.0f}m ago"
    return f"{seconds / 3600.0:.1f}h ago"


def _count_label(count: int, noun: str) -> str:
    suffix = "" if count == 1 or noun.endswith("s") else "s"
    return f"{count} {noun}{suffix}"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text.replace("_", " ")


def _html(value: Any) -> str:
    return html.escape(_clean_text(value))


class ServoPositionsPanel(QWidget):
    _ORDER = ("neck", "headlift", "headtilt", "visor", "elbow", "hand", "pokerarm", "heroarm")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._state_labels: dict[str, QLabel] = {}
        self._manual_override = False
        self._updating_snapshot = False

        layout = QGridLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(9)

        visual_row = 0
        self._override_button = QPushButton("Manual Servo Override")
        self._override_button.setObjectName("servoOverrideButton")
        self._override_button.setCheckable(True)
        self._override_button.setToolTip(
            "Freeze program-driven servo motion and drive servos directly with the sliders."
        )
        self._override_button.toggled.connect(self._set_manual_override)
        layout.addWidget(self._override_button, visual_row, 0, 1, 4)
        visual_row += 1

        self._motivator_button = QPushButton("🕹  Motivator Control")
        self._motivator_button.setObjectName("servoOverrideButton")
        self._motivator_button.setToolTip(
            "Open a joystick console to drive the motion base (motivator) by hand."
        )
        self._motivator_button.clicked.connect(self._open_motivator)
        layout.addWidget(self._motivator_button, visual_row, 0, 1, 4)
        visual_row += 1
        self._motivator_dialog: Optional["MotivatorControlDialog"] = None

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
            cfg = config.SERVO_CHANNELS[name]
            slider.setRange(int(cfg["min"]), int(cfg["max"]))
            slider.setObjectName("servoSlider")
            slider.valueChanged.connect(
                lambda value, servo_name=name: self._manual_slider_changed(servo_name, value)
            )
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
        self.setMinimumHeight(360)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        ws = snapshot.get("world_state") or {}
        self_state = ws.get("self_state") or ws.get("self") or {}
        positions = dict(snapshot.get("servo_positions") or {})
        positions.update(self_state.get("servo_positions") or {})
        manual_override = bool(
            snapshot.get(
                "manual_servo_override",
                self_state.get("manual_servo_override", self._manual_override),
            )
        )
        self._apply_manual_override_ui(manual_override)

        self._updating_snapshot = True
        for name in self._ORDER:
            cfg = config.SERVO_CHANNELS[name]
            raw = int(positions.get(name, cfg["neutral"]))
            self._sliders[name].blockSignals(True)
            self._sliders[name].setValue(max(int(cfg["min"]), min(int(cfg["max"]), raw)))
            self._sliders[name].blockSignals(False)
            self._value_labels[name].setText(str(raw))
            self._state_labels[name].setText(_servo_state(name, raw))
        self._updating_snapshot = False

    def _apply_manual_override_ui(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._manual_override = enabled
        if self._override_button.isChecked() != enabled:
            self._override_button.blockSignals(True)
            self._override_button.setChecked(enabled)
            self._override_button.blockSignals(False)
        self._override_button.setProperty("active", enabled)
        self._override_button.style().unpolish(self._override_button)
        self._override_button.style().polish(self._override_button)
        for slider in self._sliders.values():
            slider.setEnabled(enabled)

    def _set_manual_override(self, enabled: bool) -> None:
        try:
            from hardware import servos

            servos.set_manual_override_enabled(bool(enabled))
            enabled = servos.manual_override_enabled()
        except Exception as exc:
            _log.warning("Manual servo override toggle failed: %s", exc)
            enabled = False
        self._apply_manual_override_ui(enabled)

    def _manual_slider_changed(self, name: str, value: int) -> None:
        if self._updating_snapshot or not self._manual_override:
            return
        cfg = config.SERVO_CHANNELS[name]
        raw = max(int(cfg["min"]), min(int(cfg["max"]), int(value)))
        self._value_labels[name].setText(str(raw))
        self._state_labels[name].setText(_servo_state(name, raw))
        try:
            from hardware import servos

            servos.set_manual_servo(int(cfg["ch"]), raw)
        except Exception as exc:
            _log.warning("Manual servo slider update failed for %s: %s", name, exc)

    def _open_motivator(self) -> None:
        if self._motivator_dialog is None:
            self._motivator_dialog = MotivatorControlDialog(self)
        dlg = self._motivator_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()


class JoystickWidget(QWidget):
    """Self-centering analog stick drawn like a game-controller thumbstick.

    Emits normalized coordinates in [-1, 1]: x right-positive, y UP-positive
    (screen Y is inverted internally). Magnitude (distance from center) ramps the
    speed: 0 at center, 1.0 at the edge. Snaps back to center on release."""

    moved = Signal(float, float)
    released = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._kx = 0.0   # knob pixel offset from center
        self._ky = 0.0
        self._x = 0.0    # normalized value
        self._y = 0.0
        self._dragging = False

    def value(self) -> "tuple[float, float]":
        return (self._x, self._y)

    def _geom(self) -> "tuple[float, float, float, float, float]":
        size = float(min(self.width(), self.height()))
        R = size / 2.0 - 8.0
        rk = R * 0.34
        travel = max(1.0, R - rk)
        return self.width() / 2.0, self.height() / 2.0, R, rk, travel

    def _set_from_point(self, px: float, py: float) -> None:
        cx, cy, _R, _rk, travel = self._geom()
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)
        if d > travel and d > 0:
            dx *= travel / d
            dy *= travel / d
        self._kx, self._ky = dx, dy
        self._x = dx / travel
        self._y = -dy / travel        # invert: up is positive
        self.update()
        self.moved.emit(self._x, self._y)

    def _recenter(self) -> None:
        self._kx = self._ky = 0.0
        self._x = self._y = 0.0
        self.update()
        self.moved.emit(0.0, 0.0)

    def mousePressEvent(self, e) -> None:
        self._dragging = True
        self._set_from_point(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e) -> None:
        if self._dragging:
            self._set_from_point(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e) -> None:
        self._dragging = False
        self._recenter()
        self.released.emit()

    def paintEvent(self, _e) -> None:
        cx, cy, R, rk, _travel = self._geom()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Base bezel.
        p.setPen(QPen(QColor(70, 80, 95), 2))
        p.setBrush(QBrush(QColor(28, 33, 41)))
        p.drawEllipse(QRectF(cx - R, cy - R, 2 * R, 2 * R))
        # Inner well.
        well = R * 0.82
        p.setPen(QPen(QColor(48, 56, 68), 1))
        p.setBrush(QBrush(QColor(18, 22, 28)))
        p.drawEllipse(QRectF(cx - well, cy - well, 2 * well, 2 * well))

        # Direction ticks (N/E/S/W).
        p.setPen(QPen(QColor(90, 150, 200, 160), 2))
        for ang in (0, 90, 180, 270):
            a = math.radians(ang)
            x1, y1 = cx + R * 0.70 * math.cos(a), cy + R * 0.70 * math.sin(a)
            x2, y2 = cx + R * 0.90 * math.cos(a), cy + R * 0.90 * math.sin(a)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # F / B / L / R labels.
        p.setPen(QColor(150, 175, 200))
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        ac = Qt.AlignmentFlag.AlignCenter
        p.drawText(QRectF(cx - 16, cy - R * 0.92 - 9, 32, 18), ac, "FWD")
        p.drawText(QRectF(cx - 16, cy + R * 0.92 - 9, 32, 18), ac, "BACK")
        p.drawText(QRectF(cx - R * 0.92 - 16, cy - 9, 32, 18), ac, "L")
        p.drawText(QRectF(cx + R * 0.92 - 16, cy - 9, 32, 18), ac, "R")

        # Knob with a 3D radial gradient.
        kx, ky = cx + self._kx, cy + self._ky
        grad = QRadialGradient(kx - rk * 0.3, ky - rk * 0.35, rk * 1.5)
        grad.setColorAt(0.0, QColor(130, 215, 255))
        grad.setColorAt(1.0, QColor(36, 86, 134))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor(190, 235, 255), 2))
        p.drawEllipse(QRectF(kx - rk, ky - rk, 2 * rk, 2 * rk))
        # Specular highlight.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 70))
        p.drawEllipse(QRectF(kx - rk * 0.45, ky - rk * 0.6, rk * 0.5, rk * 0.38))
        p.end()


class DistancePhotoreceptorsWidget(QWidget):
    """Top-down 'radar' of the drive base's 8 radial distance photoreceptors.

    Front is up. Four long-range VL53L1X at the cardinals (F / B / L / R) plus four
    short-range VL53L0X at the 45° diagonals (FL / FR / RL / RR) — a spatial sense of
    the room. Each cone's reach scales with the measured distance and is colored by
    zone — green clear, amber slow (< MOTION_SLOW_ZONE_M), red stop (< MOTION_STOP_ZONE_M)
    — with dashed reference rings at those thresholds. A -1 reading (sensor error / no
    return) draws a faint stub. Read-only: call set_readings() from the telemetry tick.

    NOTE: while the ToF subsystem is a firmware stub (MOTION_TOF_PRESENT=0) every sensor
    reads a constant 'clear'; the panel shows that honestly until the sensors are wired
    and the firmware is built with ToF on. There is no down/cliff sensor in this layout."""

    _FOV_DEG = 25.0                       # ToF cone ~25° (VL53L0X / VL53L1X similar)
    # (bearing°, telemetry key, label) — screen convention: 0 = 3 o'clock, CCW+, front = up = 90°.
    # 8 sensors at 45° steps: cardinals are long-range VL53L1X, diagonals short VL53L0X.
    _BEAMS = (
        (90.0,  "front", "F"),
        (135.0, "fl",    "FL"),
        (180.0, "left",  "L"),
        (225.0, "rl",    "RL"),
        (270.0, "rear",  "B"),
        (315.0, "rr",    "RR"),
        (0.0,   "right", "R"),
        (45.0,  "fr",    "FR"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tof: dict = {}
        self._zone = None
        self._blocked = None
        self._live = False
        self._stop_m = float(getattr(config, "MOTION_STOP_ZONE_M", 0.25))
        self._slow_m = float(getattr(config, "MOTION_SLOW_ZONE_M", 0.60))
        self._max_m = max(self._slow_m * 1.5, 4.0)        # display reach (m) — VL53L1X room scale

    def set_readings(self, tof_mm, zone=None, blocked=None) -> None:
        self._tof = dict(tof_mm or {})
        self._zone = (str(zone).lower() if zone else None)
        self._blocked = (str(blocked).lower() if blocked else None)
        self._live = True
        self.update()

    def clear(self) -> None:
        self._tof = {}
        self._zone = self._blocked = None
        self._live = False
        self.update()

    def _mm(self, key):
        try:
            v = self._tof.get(key)
            return None if v is None else int(v)
        except (TypeError, ValueError):
            return None

    def _beam_color(self, mm, alpha=255):
        if mm is None or mm < 0:
            return QColor(95, 105, 120, alpha)                  # error / no data
        d = mm / 1000.0
        if d <= self._stop_m:  return QColor(235, 70, 60, alpha)    # STOP
        if d <= self._slow_m:  return QColor(240, 180, 60, alpha)   # SLOW
        return QColor(70, 200, 130, alpha)                          # CLEAR

    def paintEvent(self, _e) -> None:
        w, h = float(self.width()), float(self.height())
        band = 0.0                                    # no down/cliff sensor in this layout
        top_h = h - band
        cx = w / 2.0
        cy = top_h / 2.0 + 6.0
        maxR = min(w, top_h) / 2.0 - 16.0
        if maxR < 24:
            return
        body_r = maxR * 0.20
        reach = maxR - body_r
        dim = 1.0 if self._live else 0.35

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        def pt(bearing_deg, r):
            a = math.radians(bearing_deg)
            return QPointF(cx + r * math.cos(a), cy - r * math.sin(a))

        def radius_for(mm):
            if mm is None or mm < 0:
                return body_r + reach * 0.12
            d = max(0.0, min(mm / 1000.0, self._max_m))
            return body_r + (d / self._max_m) * reach

        # Cones (filled translucent wedges from the body outward).
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        for bearing, key, label in self._BEAMS:
            mm = self._mm(key)
            r = radius_for(mm)
            p.setBrush(QBrush(self._beam_color(mm, int(150 * dim))))
            p.setPen(QPen(self._beam_color(mm, int(255 * dim)), 1))
            p.drawPie(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                      int((bearing - self._FOV_DEG / 2.0) * 16), int(self._FOV_DEG * 16))
            txt = "—" if mm is None else ("err" if mm < 0
                  else (f"{mm}" if mm < 1000 else f"{mm / 1000.0:.1f}m"))
            lp = pt(bearing, r + 15)
            p.setPen(QColor(185, 205, 225, int(255 * dim)))
            p.drawText(QRectF(lp.x() - 28, lp.y() - 9, 56, 18),
                       Qt.AlignmentFlag.AlignCenter, f"{label} {txt}")

        # Reference rings (over the cones): max range + slow + stop thresholds.
        p.setBrush(Qt.BrushStyle.NoBrush)
        for val_m, col in ((self._max_m, QColor(70, 80, 95)),
                           (self._slow_m, QColor(150, 120, 50)),
                           (self._stop_m, QColor(150, 60, 55))):
            rr = body_r + (min(val_m, self._max_m) / self._max_m) * reach
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawEllipse(QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr))

        # Robot body + forward chevron (points up).
        p.setPen(QPen(QColor(120, 140, 165, int(255 * dim)), 2))
        p.setBrush(QBrush(QColor(30, 36, 46, int(255 * dim))))
        p.drawEllipse(QRectF(cx - body_r, cy - body_r, 2 * body_r, 2 * body_r))
        p.setPen(QPen(QColor(130, 215, 255, int(255 * dim)), 2))
        p.drawLine(QPointF(cx - body_r * 0.45, cy), QPointF(cx, cy - body_r * 0.55))
        p.drawLine(QPointF(cx, cy - body_r * 0.55), QPointF(cx + body_r * 0.45, cy))

        # Zone badge (top-left chip).
        zone = self._zone or ("clear" if self._live else "—")
        zcol = {"clear": QColor(70, 200, 130), "slow": QColor(240, 180, 60),
                "stop": QColor(235, 70, 60), "cliff": QColor(235, 70, 60)}.get(
                    zone, QColor(120, 130, 145))
        ztxt = f"ZONE {zone.upper()}"
        if self._blocked and self._blocked not in ("none", "—"):
            ztxt += f" · blk {self._blocked}"
        f.setPointSize(8); f.setBold(True); p.setFont(f)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(zcol.red(), zcol.green(), zcol.blue(), int(55 * dim)))
        p.drawRoundedRect(QRectF(6, 6, 158, 19), 4, 4)
        p.setPen(zcol)
        p.drawText(QRectF(6, 6, 158, 19), Qt.AlignmentFlag.AlignCenter, ztxt)

        if not self._live:
            p.setPen(QColor(150, 165, 185))
            f.setPointSize(10); f.setBold(True); p.setFont(f)
            p.drawText(QRectF(0, cy - 12, w, 24), Qt.AlignmentFlag.AlignCenter, "no link")
        p.end()


class MotivatorControlDialog(QDialog):
    """Joystick console to drive the motion base by hand, with live ESP32 readout.

    Mixing (arcade): forward = stick-up, turn = stick-right.
      left motor  = forward + turn      right motor = forward - turn
    so UP = both forward, DOWN = both back, LEFT = left back/right forward,
    RIGHT = left forward/right back. Sent as a `drive` command (lin m/s, ang rad/s)
    that the ESP32 mixes to the wheels; refreshed at 10 Hz so the deadman stays fed."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Motivator Control")
        self.setObjectName("motivatorDialog")
        self.setModal(False)
        # A top-level QDialog doesn't inherit the main window's stylesheet, so apply
        # the dashboard theme (_STYLE) plus the Motivator-specific rules here.
        self.setStyleSheet(_STYLE + _MOTIVATOR_EXTRA)
        self.resize(760, 700)
        self._x = 0.0
        self._y = 0.0
        self._engaged = False     # only drive after the operator has touched the stick
        self._was_driving = False  # tracks displaced->centered so release sends one stop
        self._cmd_lin = 0.0       # ramped command actually sent — slews toward the stick
        self._cmd_ang = 0.0       # so a release eases to a stop instead of braking hard

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self._conn = QLabel("…")
        self._conn.setObjectName("motivatorConn")
        self._conn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conn.setWordWrap(True)
        root.addWidget(self._conn)

        # Two columns: drive controls on the left, sensing + telemetry on the right.
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        self.joystick = JoystickWidget()
        self.joystick.moved.connect(self._on_move)
        self.joystick.released.connect(self._on_release)
        left.addWidget(self.joystick, 1)

        cmd = self._section("COMMANDED (this console)")
        self._lbl_left = self._row(cmd, "Left motor")
        self._lbl_right = self._row(cmd, "Right motor")
        self._lbl_lin = self._row(cmd, "Linear")
        self._lbl_ang = self._row(cmd, "Angular")
        left.addWidget(cmd["frame"])
        body.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(12)
        photo = QFrame()
        photo.setObjectName("chromePanel")
        photo_lay = QVBoxLayout(photo)
        photo_lay.setContentsMargins(12, 10, 12, 10)
        photo_lay.setSpacing(8)
        photo_title = QLabel("DISTANCE PHOTORECEPTORS")
        photo_title.setObjectName("panelTitle")
        photo_lay.addWidget(photo_title)
        self._photoreceptors = DistancePhotoreceptorsWidget()
        photo_lay.addWidget(self._photoreceptors, 1)
        right.addWidget(photo, 1)

        fb = self._section("ESP32 FEEDBACK")
        self._fb_state = self._row(fb, "State")
        self._fb_owner = self._row(fb, "Owner / gamepad")
        self._fb_zone = self._row(fb, "Zone / blocked")
        self._fb_odom = self._row(fb, "Odom lin / ang")
        self._fb_pose = self._row(fb, "Pose x / y / θ")
        self._fb_tof = self._row(fb, "ToF F/B/L/R")
        self._fb_tof2 = self._row(fb, "ToF FL/FR/RL/RR")
        self._fb_batt = self._row(fb, "Battery")
        self._fb_fault = self._row(fb, "Fault / errs")
        right.addWidget(fb["frame"])
        body.addLayout(right, 1)

        self._stop_btn = QPushButton("■  STOP")
        self._stop_btn.setObjectName("motivatorStop")
        self._stop_btn.clicked.connect(self._stop)
        root.addWidget(self._stop_btn)

        self._send_timer = QTimer(self)
        self._send_timer.timeout.connect(self._tick_send)
        self._tel_timer = QTimer(self)
        self._tel_timer.timeout.connect(self._tick_telemetry)

    # ---- small builders ----
    def _section(self, title: str) -> dict:
        frame = QFrame()
        frame.setObjectName("chromePanel")
        lay = QGridLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setVerticalSpacing(4)
        lay.setHorizontalSpacing(10)
        head = QLabel(title)
        head.setObjectName("panelTitle")
        lay.addWidget(head, 0, 0, 1, 2)
        return {"lay": lay, "frame": frame, "row": 1}

    def _row(self, section: dict, label: str) -> QLabel:
        lay = section["lay"]
        r = section["row"]
        section["row"] = r + 1
        name = QLabel(label)
        name.setObjectName("servoName")
        val = QLabel("—")
        val.setObjectName("servoValue")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(name, r, 0)
        lay.addWidget(val, r, 1)
        lay.setColumnStretch(0, 1)
        return val

    # ---- mixing ----
    def _mix(self) -> "tuple[float, float, float, float]":
        fwd, turn = self._y, self._x
        left = max(-1.0, min(1.0, fwd + turn))
        right = max(-1.0, min(1.0, fwd - turn))
        max_lin = float(getattr(config, "MOTION_MAX_LINEAR_MS", 0.25))
        max_ang = math.radians(float(getattr(config, "MOTION_MAX_ANGULAR_DEG_S", 60.0)))
        lin = fwd * max_lin
        ang = -turn * max_ang      # stick-left (x<0) -> +ang = CCW/left turn (REP-103)
        return left, right, lin, ang

    def _on_move(self, x: float, y: float) -> None:
        if x != 0.0 or y != 0.0:
            self._engaged = True
        self._x, self._y = x, y
        left, right, lin, ang = self._mix()
        self._lbl_left.setText(f"{left * 100:+.0f}%")
        self._lbl_right.setText(f"{right * 100:+.0f}%")
        self._lbl_lin.setText(f"{lin:+.2f} m/s")
        self._lbl_ang.setText(f"{math.degrees(ang):+.0f}°/s")

    def _on_release(self) -> None:
        self._on_move(0.0, 0.0)

    # ---- timers ----
    def _tick_send(self) -> None:
        try:
            from intelligence import motion_controller as mc
            if not mc.available():
                return
            # Target velocity from the current stick position (0 when centered/released).
            if math.hypot(self._x, self._y) > 0.02:    # deadzone — ignore resting jitter
                _l, _r, lin_t, ang_t = self._mix()
            else:
                lin_t = ang_t = 0.0
            # Slew the commanded velocity toward the target: gentle on the way up, faster
            # but never abrupt on the way down, so releasing the stick eases to a stop
            # instead of braking hard enough to topple a tall base. (The STOP button and
            # closing the console still stop immediately — they zero the ramp directly.)
            dt = (self._send_timer.interval() / 1000.0) or 0.1
            max_lin = float(getattr(config, "MOTION_MAX_LINEAR_MS", 0.25))
            max_ang = math.radians(float(getattr(config, "MOTION_MAX_ANGULAR_DEG_S", 60.0)))
            up = max(0.05, float(getattr(config, "MOTION_MANUAL_RAMP_UP_SECS", 1.2)))
            down = max(0.05, float(getattr(config, "MOTION_MANUAL_RAMP_DOWN_SECS", 0.5)))
            self._cmd_lin = mc.ramp_toward(self._cmd_lin, lin_t, max_lin * dt / up, max_lin * dt / down)
            self._cmd_ang = mc.ramp_toward(self._cmd_ang, ang_t, max_ang * dt / up, max_ang * dt / down)
            # Readout reflects what's actually commanded (the ramped value), incl. the
            # smooth ramp-down after release.
            self._lbl_lin.setText(f"{self._cmd_lin:+.2f} m/s")
            self._lbl_ang.setText(f"{math.degrees(self._cmd_ang):+.0f}°/s")

            moving = abs(self._cmd_lin) > 1e-3 or abs(self._cmd_ang) > 1e-3
            target_active = abs(lin_t) > 1e-3 or abs(ang_t) > 1e-3
            if moving or target_active:
                mc.drive_manual(self._cmd_lin, self._cmd_ang)   # 10 Hz refresh feeds the deadman
                self._was_driving = True
            elif self._was_driving:
                mc.stop()                      # fully ramped down -> one clean idle
                self._was_driving = False
        except Exception:
            pass

    def _tick_telemetry(self) -> None:
        try:
            from hardware import motion
            connected = motion.connected()
            tel = motion.telemetry() if connected else None
        except Exception:
            connected, tel = False, None

        if connected:
            self._conn.setText("ESP32 connected — this console holds manual control while open")
        else:
            self._conn.setText("ESP32 disconnected — set MOTION_ESP32_PORT and run main.py --gui")
        self._conn.setProperty("ok", bool(connected))
        self._conn.style().unpolish(self._conn)
        self._conn.style().polish(self._conn)

        if not tel:
            for lbl in (self._fb_state, self._fb_owner, self._fb_zone, self._fb_odom,
                        self._fb_pose, self._fb_tof, self._fb_tof2, self._fb_batt, self._fb_fault):
                lbl.setText("—")
            self._photoreceptors.clear()
            return

        odom = tel.get("odom") or {}
        tof = tel.get("tof_mm") or {}

        def g(d, k, default=0.0):
            try:
                return float(d.get(k, default))
            except (TypeError, ValueError):
                return default

        self._fb_state.setText(str(tel.get("state", "—")))
        self._fb_owner.setText(f"{tel.get('owner', '—')} / {tel.get('gamepad', '—')}")
        self._fb_zone.setText(f"{tel.get('zone', '—')} / {tel.get('blocked_dir', '—')}")
        self._fb_odom.setText(f"{g(odom, 'lin'):+.2f} m/s / {math.degrees(g(odom, 'ang')):+.0f}°/s")
        self._fb_pose.setText(
            f"{g(odom, 'x'):+.2f} / {g(odom, 'y'):+.2f} / {math.degrees(g(odom, 'theta')):+.0f}°"
        )
        self._fb_tof.setText(
            f"{tof.get('front', '—')} / {tof.get('rear', '—')} / {tof.get('left', '—')} / {tof.get('right', '—')} mm")
        self._fb_tof2.setText(
            f"{tof.get('fl', '—')} / {tof.get('fr', '—')} / {tof.get('rl', '—')} / {tof.get('rr', '—')} mm")
        self._fb_batt.setText(f"{g(tel, 'batt_mv') / 1000.0:.2f} V")
        self._fb_fault.setText(f"{tel.get('fault') or 'none'} / errs {tel.get('errs', 0)}")
        self._photoreceptors.set_readings(tof, tel.get("zone"), tel.get("blocked_dir"))

    def _stop(self) -> None:
        # Immediate stop (unlike releasing the stick, which ramps down): zero the ramp
        # state directly so the next tick sends nothing, then command a controlled stop.
        self.joystick._recenter()
        self._x = self._y = 0.0
        self._cmd_lin = self._cmd_ang = 0.0
        self._was_driving = False
        self._on_move(0.0, 0.0)
        try:
            from intelligence import motion_controller as mc
            mc.stop()
        except Exception:
            pass

    # ---- lifecycle ----
    def showEvent(self, e) -> None:
        self._engaged = False
        self._was_driving = False
        self._cmd_lin = self._cmd_ang = 0.0
        self._x = self._y = 0.0
        self._on_move(0.0, 0.0)
        if not self._send_timer.isActive():
            self._send_timer.start(100)
        if not self._tel_timer.isActive():
            self._tel_timer.start(150)
        self._tick_telemetry()
        super().showEvent(e)

    def closeEvent(self, e) -> None:
        self._send_timer.stop()
        self._tel_timer.stop()
        try:
            from intelligence import motion_controller as mc
            mc.stop()
        except Exception:
            pass
        super().closeEvent(e)


def run_dashboard(
    bridge: GUIDashboardBridge = gui_bridge,
    *,
    shutdown_callback: Optional[Callable[[], None]] = None,
    text_submit_callback: Optional[Callable[[str], None]] = None,
    sleep_callback: Optional[Callable[[], None]] = None,
    wake_callback: Optional[Callable[[], None]] = None,
    demo: bool = False,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = DashboardWindow(
        bridge,
        shutdown_callback=shutdown_callback,
        text_submit_callback=text_submit_callback,
        sleep_callback=sleep_callback,
        wake_callback=wake_callback,
        demo=demo,
    )
    # Fill the available desktop (normal zoomed window) — NOT macOS native
    # full-screen, so the menu bar / other windows stay reachable.
    window.showMaximized()

    def _sigint(_signum, _frame) -> None:
        QTimer.singleShot(0, window.request_shutdown)

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
            "face_mood": {
                "mood": _demo_expression(now, 0),
                "confidence": 0.86,
                "notes": "demo",
            },
            "engagement": "tracking",
            "distance_zone": "social",
            "approach_vector": "stationary",
            "face_box_fraction": 235 / 820,
            "face_visible": True,
            "face_missing": False,
            "face_last_seen_at": time.time(),
            "pose": "standing",
        },
        {
            "id": "person_2",
            "face_id": "person 0.91",
            "face_box": (606, 132, 205, 268),
            "expression": _demo_expression(now, 1),
            "face_mood": {
                "mood": _demo_expression(now, 1),
                "confidence": 0.72,
                "notes": "demo",
            },
            "engagement": "tracking",
            "distance_zone": "social",
            "approach_vector": "departing",
            "face_box_fraction": 205 / 820,
            "face_visible": True,
            "face_missing": False,
            "face_last_seen_at": time.time(),
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
    bridge.update_camera_stats(
        label="Demo Camera",
        fps=24.0,
        seq=int(now * 24),
        last_frame_monotonic=now,
        resolution=(820, 540),
    )
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
            "face_tracking": {
                "locked": True,
                "visible": True,
                "holding_lost_lock": False,
                "searching": False,
                "lock_key": "person_1",
                "person_id": None,
                "lost_age_secs": 0.0,
            },
        },
    })
    if getattr(_advance_demo, "_led_seeded", False) is False:
        _advance_demo._led_seeded = True  # type: ignore[attr-defined]
        bridge.update_head_led_state(mode="idle", eye_color=(45, 115, 255), eyes_active=True)

    _advance_demo._log_n = getattr(_advance_demo, "_log_n", 0) + 1  # type: ignore[attr-defined]
    if _advance_demo._log_n % 4 == 0:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        bridge.add_log_line(
            f"{stamp} | demo.heartbeat                 | INFO     | demo tick {_advance_demo._log_n}"
        )

    if getattr(_advance_demo, "_seeded", False) is False:
        _advance_demo._seeded = True  # type: ignore[attr-defined]
        bridge.update_controller_status("online")
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
QLabel#windowTitle {
    color: #aab5c1;
    font-size: 15px;
    font-weight: 800;
}
QLabel#connectionLabel {
    color: #45d85e;
    font-size: 13px;
}
QLabel#deviceStatus {
    color: #9fb6cc;
    font-size: 12px;
    font-weight: 700;
}
QLabel#stateBadge {
    color: #5b6b7d;
    border: 1px solid #2b4562;
    border-radius: 5px;
    padding: 3px 13px;
    font-size: 14px;
    font-weight: 900;
}
QFrame#chromePanel {
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
QTextBrowser#visionDescription {
    background: #07111a;
    color: #d9e3ee;
    border: none;
}
QPlainTextEdit#systemLog {
    background: #050d14;
    color: #9fb6cc;
    border: none;
    selection-background-color: #244f89;
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
QPushButton#servoOverrideButton {
    min-height: 34px;
    padding: 0 12px;
    background: #111b27;
    color: #dbe7f3;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 800;
}
QPushButton#memoryBanksButton {
    min-height: 30px;
    padding: 0 14px;
    margin-left: 10px;
    background: #15212f;
    color: #aee0ff;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 700;
}
QPushButton#memoryBanksButton:hover {
    background: #1d2f44;
    border: 1px solid #65a2ff;
}
QPushButton#topControlButton {
    min-height: 30px;
    padding: 0 14px;
    margin-left: 8px;
    background: #15212f;
    color: #aee0ff;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 700;
}
QPushButton#topControlButton:hover {
    background: #1d2f44;
    border: 1px solid #65a2ff;
}
QPushButton#topShutdownButton {
    min-height: 30px;
    padding: 0 14px;
    margin-left: 8px;
    background: #2a1416;
    color: #ffb3ab;
    border: 1px solid #5e2a2a;
    border-radius: 5px;
    font-weight: 700;
}
QPushButton#topShutdownButton:hover {
    background: #4a1d1d;
    border: 1px solid #d05a5a;
    color: #ffffff;
}
QPushButton#servoOverrideButton[active="true"] {
    background: #244f89;
    color: #ffffff;
    border: 1px solid #65a2ff;
}
QLabel#servoName, QLabel#servoValue, QLabel#servoState {
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
    height: 2px;
    background: #203040;
    border: none;
    border-radius: 1px;
}
QSlider#servoSlider::sub-page:horizontal {
    background: transparent;
    border: none;
}
QSlider#servoSlider::handle:horizontal {
    width: 13px;
    height: 13px;
    margin: -6px 0;
    border-radius: 6px;
    background: #4d8dea;
}
QSlider#servoSlider:disabled::groove:horizontal {
    background: #182637;
}
QSlider#servoSlider:disabled::handle:horizontal {
    background: #526171;
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


# Motivator Control dialog theme — extends _STYLE (a top-level QDialog does not
# inherit the main window's stylesheet) with the dialog background + its widgets.
_MOTIVATOR_EXTRA = """
QDialog#motivatorDialog {
    background: #07111a;
    color: #d9e3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
QLabel#motivatorConn {
    color: #8aa0b6;
    font-size: 12px;
    font-weight: 700;
    padding: 2px;
}
QLabel#motivatorConn[ok="true"] { color: #45d85e; }
QLabel#motivatorConn[ok="false"] { color: #e0a23a; }
QPushButton#motivatorStop {
    min-height: 38px;
    background: #7a1f1f;
    color: #ffffff;
    border: 1px solid #a23a3a;
    border-radius: 6px;
    font-weight: 900;
    font-size: 14px;
}
QPushButton#motivatorStop:hover {
    background: #9a2a2a;
    border: 1px solid #d05a5a;
}
"""


# Shutdown confirmation dialog — dashboard-themed yes/no.
_CONFIRM_EXTRA = """
QDialog#confirmDialog {
    background: #0b1824;
    color: #d9e3ee;
    border: 1px solid #255484;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
QLabel#confirmText {
    color: #dbe7f3;
    font-size: 15px;
    font-weight: 700;
}
QPushButton#confirmNo {
    min-height: 34px;
    padding: 0 20px;
    background: #15212f;
    color: #aee0ff;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 700;
}
QPushButton#confirmNo:hover {
    background: #1d2f44;
    border: 1px solid #65a2ff;
}
QPushButton#confirmYes {
    min-height: 34px;
    padding: 0 20px;
    background: #7a1f1f;
    color: #ffffff;
    border: 1px solid #a23a3a;
    border-radius: 5px;
    font-weight: 800;
}
QPushButton#confirmYes:hover {
    background: #9a2a2a;
    border: 1px solid #d05a5a;
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
