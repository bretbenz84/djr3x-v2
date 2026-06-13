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


class DashboardWindow(QMainWindow):
    def __init__(
        self,
        bridge: GUIDashboardBridge,
        *,
        shutdown_callback: Optional[Callable[[], None]] = None,
        text_submit_callback: Optional[Callable[[str], None]] = None,
        demo: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._shutdown_callback = shutdown_callback
        self._demo = demo
        self._closing_from_shutdown = False
        self._shutdown_requested = False

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

        root = QWidget()
        root.setObjectName("root")
        self._shell = QVBoxLayout(root)
        self._shell.setContentsMargins(14, 8, 14, 14)
        self._shell.setSpacing(12)

        self._top_bar = QWidget()
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._top_bar.setLayout(top)
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
        left.addWidget(ChromePanel("1", "VISION", self.vision), 4)
        left.addWidget(ChromePanel("", "OPENAI VISION + DLIB STATE", self.scene), 8)
        left_box = QWidget()
        left_box.setLayout(left)

        center = ChromePanel("☵", "CONVERSATION LOG", self.conversation)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        avatar_panel = ChromePanel("3", "R3X AVATAR", self.avatar)
        servo_panel = ChromePanel("", "SERVO POSITIONS", self.servos)
        servo_panel.setMinimumHeight(350)
        servo_panel.setMaximumHeight(380)
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

    def close_from_shutdown(self) -> None:
        self._closing_from_shutdown = True
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

    def _runtime_shutdown_requested(self) -> bool:
        try:
            import state as state_module
            from state import State

            return bool(state_module.is_state(State.SHUTDOWN))
        except Exception:
            return False


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
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(11)

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
        self.setMinimumHeight(285)

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


def run_dashboard(
    bridge: GUIDashboardBridge = gui_bridge,
    *,
    shutdown_callback: Optional[Callable[[], None]] = None,
    text_submit_callback: Optional[Callable[[str], None]] = None,
    demo: bool = False,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = DashboardWindow(
        bridge,
        shutdown_callback=shutdown_callback,
        text_submit_callback=text_submit_callback,
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DJ-R3X optional GUI dashboard")
    parser.add_argument("--demo", action="store_true", help="run dashboard with simulated state")
    args = parser.parse_args(argv)
    if not args.demo:
        print("Use --demo to run the dashboard outside main.py.", file=sys.stderr)
    return run_dashboard(gui_bridge, demo=args.demo)


if __name__ == "__main__":
    raise SystemExit(main())
