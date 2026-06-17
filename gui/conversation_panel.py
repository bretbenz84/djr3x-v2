"""Conversation log panel for the optional dashboard."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, QTimer
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
        # Capture intent BEFORE the re-render: was the view pinned to the newest
        # line, and where was the reader otherwise? setHtml resets the scrollbar
        # to the top, so these must be read first.
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 6
        prev_value = scrollbar.value()

        self._log.setHtml(_format_lines(lines))
        self._last_seq = last_seq

        if at_bottom:
            # Keep the latest text anchored to the bottom and let older lines
            # scroll up out of view. setHtml relays the document out lazily, so
            # the scrollbar's maximum can still be stale on this pass — pin once
            # now and again after layout settles, otherwise the newest line ends
            # up below the fold (the "I have to scroll down to see it" bug).
            self._scroll_to_bottom()
            QTimer.singleShot(0, self._scroll_to_bottom)
        else:
            # Reader has scrolled up through history — only new lines were
            # appended below, so hold their position across the full re-render
            # instead of snapping them back to the top.
            scrollbar.setValue(min(prev_value, scrollbar.maximum()))

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._log.verticalScrollBar()
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
    if kind == "user":
        label = _user_label(speaker)
    elif kind == "rex":
        label = "R3X"
    else:
        label = speaker
    return f"""
    <div class="entry">
        <table class="meta"><tr>
            <td class="speaker {kind}">{_escape(label)}</td>
            <td class="time">{stamp}</td>
        </tr></table>
        <div class="text">{text}</div>
    </div>
    """


_GENERIC_USER_LABELS = {"", "human", "user", "unknown", "unknown speaker"}
_ANON_VOICE_RE = re.compile(r"unknown[_ ]voice[_ ]?(\d+)$")


def _user_label(speaker: str) -> str:
    """Display name for a human turn.

    The identity pipeline already resolves WHO spoke and passes that name into
    the conversation bridge, so show it: Rex's best guess at the person he is
    talking to (e.g. "Bret Benziger", "JT"). A distinct-but-unidentified voice
    slot (``unknown_voice_2``) becomes a friendly ``Guest 2``; a turn with no
    identity at all falls back to the generic "Human"."""
    s = (speaker or "").strip()
    low = s.lower()
    if low in _GENERIC_USER_LABELS:
        return "Human"
    m = _ANON_VOICE_RE.match(low)
    if m:
        return f"Guest {m.group(1)}"
    return s


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )
