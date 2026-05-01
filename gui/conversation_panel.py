"""Conversation log panel for the optional dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ConversationPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_seq = -1
        self._submit_callback: Optional[Callable[[str], None]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._log = QTextBrowser()
        self._log.setObjectName("conversationLog")
        self._log.setOpenExternalLinks(False)
        self._log.setFrameShape(QFrame.Shape.NoFrame)
        self._log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._log, 1)

        entry_row = QHBoxLayout()
        entry_row.setContentsMargins(0, 0, 0, 0)
        entry_row.setSpacing(8)

        self._entry = QLineEdit()
        self._entry.setObjectName("messageEntry")
        self._entry.setPlaceholderText("Type a message...")
        self._entry.returnPressed.connect(self._submit)
        entry_row.addWidget(self._entry, 1)

        self._send = QPushButton("Send")
        self._send.setObjectName("primaryButton")
        self._send.clicked.connect(self._submit)
        entry_row.addWidget(self._send)

        layout.addLayout(entry_row)
        self.setMinimumSize(310, 420)

    def set_submit_callback(self, callback: Callable[[str], None]) -> None:
        self._submit_callback = callback

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        lines = list(snapshot.get("conversation_lines") or [])
        last_seq = lines[-1].get("seq", -1) if lines else -1
        if last_seq == self._last_seq:
            return

        scrollbar = self._log.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 6
        self._log.setHtml(_format_lines(lines))
        self._last_seq = last_seq
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _submit(self) -> None:
        text = self._entry.text().strip()
        if not text:
            return
        self._entry.clear()
        if self._submit_callback is not None:
            self._submit_callback(text)


def _format_lines(lines: list[dict[str, Any]]) -> str:
    if not lines:
        return """
        <html><body>
        <div class="empty">Conversation log waiting for the first exchange.</div>
        </body></html>
        """

    items = []
    for line in lines[-80:]:
        items.append(_format_line(line))
    return f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: #07111a;
            color: #d8e4f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 14px;
            line-height: 1.42;
        }}
        .entry {{
            border-top: 1px solid rgba(76, 118, 164, 0.36);
            padding: 13px 0 14px 0;
        }}
        .entry:first-child {{
            border-top: none;
            padding-top: 0;
        }}
        .meta {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 5px;
        }}
        .speaker {{
            font-weight: 800;
        }}
        .speaker.user {{
            color: #5396ff;
        }}
        .speaker.rex {{
            color: #ff9b21;
        }}
        .speaker.system {{
            color: #43d66f;
        }}
        .time {{
            color: #8d9aab;
            text-align: right;
            white-space: nowrap;
        }}
        .text {{
            color: #e2e9f1;
        }}
        .empty {{
            color: #73859a;
            padding: 24px 4px;
        }}
    </style>
    </head>
    <body>{"".join(items)}</body>
    </html>
    """


def _format_line(line: dict[str, Any]) -> str:
    ts = line.get("ts")
    try:
        stamp = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        stamp = "--:--:--"

    speaker = str(line.get("speaker") or "System").strip()
    text = _escape(str(line.get("text") or "").strip())
    kind = str(line.get("kind") or "").strip().lower()
    if kind not in {"user", "rex", "system"}:
        kind = "rex" if speaker.lower() in {"rex", "r3x"} else "system"
    label = "Human" if kind == "user" else ("R3X" if kind == "rex" else speaker)
    return f"""
    <div class="entry">
        <table class="meta"><tr>
            <td class="speaker {kind}">{_escape(label)}</td>
            <td class="time">{stamp}</td>
        </tr></table>
        <div class="text">{text}</div>
    </div>
    """


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )
