"""Auto-scrolling system-log console for the optional dashboard.

Renders the app-log lines that utils.logging mirrors into the GUI bridge —
the same formatted records written to the active logs/djr3x*.log file.
"""

from __future__ import annotations

from typing import Any

import config

from PySide6.QtGui import QFont, QTextOption
from PySide6.QtWidgets import QFrame, QPlainTextEdit, QVBoxLayout, QWidget


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
        self._view.appendPlainText("\n".join(str(line.get("text") or "") for line in fresh))
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
