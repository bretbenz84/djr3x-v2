"""Auto-scrolling system-log console for the optional dashboard.

Renders the app-log lines that utils.logging mirrors into the GUI bridge —
the same formatted records written to the active logs/djr3x*.log file.
"""

from __future__ import annotations

from typing import Any

import config

from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QTextOption
from PySide6.QtWidgets import QFrame, QPlainTextEdit, QVBoxLayout, QWidget

# Foreground color per log level. INFO stays the muted default; warnings and
# errors pop so they're scannable in a wall of INFO lines.
_LEVEL_COLORS = {
    "DEBUG": "#6c7f93",
    "INFO": "#9fb6cc",
    "WARNING": "#f0c45a",
    "ERROR": "#ff6b5e",
    "CRITICAL": "#ff4d4d",
}
_DEFAULT_LOG_COLOR = "#9fb6cc"


class LogPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_seq = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(0)

        self._view = QPlainTextEdit()
        self._view.setObjectName("systemLog")
        self._view.setReadOnly(True)
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        self._view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self._view.setMaximumBlockCount(
            max(1, int(getattr(config, "GUI_LOG_PANEL_MAX_LINES", 600) or 600))
        )
        self._view.setPlaceholderText("Waiting for log output…")
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(11)
        self._view.setFont(font)
        layout.addWidget(self._view, 1)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        lines = snapshot.get("log_lines") or []
        fresh = [line for line in lines if int(line.get("seq") or 0) > self._last_seq]
        if not fresh:
            return
        self._last_seq = int(fresh[-1].get("seq") or self._last_seq)

        scrollbar = self._view.verticalScrollBar()
        # Stick to the tail unless the user has scrolled up to read history.
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 6

        # Append each line with a per-level color via a cursor (preserves the
        # log format's exact spacing, unlike HTML, and setMaximumBlockCount
        # trims old blocks off the top for us).
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        document = self._view.document()
        for line in fresh:
            fmt = QTextCharFormat()
            level = str(line.get("level") or "INFO").upper()
            fmt.setForeground(QColor(_LEVEL_COLORS.get(level, _DEFAULT_LOG_COLOR)))
            if not document.isEmpty():
                cursor.insertBlock()
            cursor.insertText(str(line.get("text") or ""), fmt)

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
