"""Shared droid console theme: worn olive metal, warm insignia orange,
cream typography, and restrained cyan telemetry. Panel chrome is painted
locally; backdrop texture is cached so live views stay inexpensive.
"""

from __future__ import annotations

import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ── Palette ──────────────────────────────────────────────────────────────────
SPACE_BLACK = "#101311"     # window / page background
PANEL_TOP = "#252a26"       # panel gradient top
PANEL_BOTTOM = "#171c1b"    # panel gradient bottom
BORDER = "#53584c"          # panel border
BORDER_DIM = "#313a34"
ORANGE = "#e9a35a"          # R3X orange — titles/accents
ORANGE_DIM = "#916643"
AMBER = "#ffb21e"           # jeopardy gold / highlight
BLUE = "#8cc7c4"            # holo blue — data
BLUE_DIM = "#475c59"
TEXT = "#f0eadb"
TEXT_DIM = "#b3b4a5"
GOOD = "#45d85e"
WARN = "#f0c45a"
BAD = "#ff6b5e"

_CUT = 14.0  # corner cut size for panels


def panel_path(rect: QRectF, cut: float = _CUT) -> QPainterPath:
    """Angular SW-console outline: cut top-left + bottom-right corners."""
    c = min(cut, rect.width() * 0.25, rect.height() * 0.25)
    p = QPainterPath()
    p.moveTo(rect.left() + c, rect.top())
    p.lineTo(rect.right(), rect.top())
    p.lineTo(rect.right(), rect.bottom() - c)
    p.lineTo(rect.right() - c, rect.bottom())
    p.lineTo(rect.left(), rect.bottom())
    p.lineTo(rect.left(), rect.top() + c)
    p.closeSubpath()
    return p


def paint_panel_chrome(painter: QPainter, rect: QRectF, *, header_h: float = 0.0) -> None:
    """Fill + border + texture for one holo panel (used by HoloPanel and the
    jeopardy painter so both match exactly)."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = panel_path(rect)

    grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    grad.setColorAt(0.0, QColor(PANEL_TOP))
    grad.setColorAt(1.0, QColor(PANEL_BOTTOM))
    painter.setPen(QPen(QColor(BORDER), 1.6))
    painter.setBrush(grad)
    painter.drawPath(path)

    # Fine brushed metal; damage stays on the frame, away from readable content.
    painter.save()
    painter.setClipPath(path)
    painter.setPen(QPen(QColor(255, 247, 217, 3), 1))
    y = rect.top() + 3
    while y < rect.bottom():
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        y += 5
    rng = random.Random(1977)
    for _ in range(32):
        x = rng.uniform(rect.left() + 8, rect.right() - 8)
        y = rng.choice((rect.top() + rng.uniform(2, 5), rect.bottom() - rng.uniform(2, 7)))
        painter.setPen(QPen(QColor(213, 197, 165, rng.randint(12, 45)), 1))
        painter.drawLine(QPointF(x, y), QPointF(x + rng.uniform(2, 12), y - 1))
    # header band
    if header_h > 0:
        band = QLinearGradient(rect.topLeft(), QPointF(rect.left(), rect.top() + header_h))
        band.setColorAt(0.0, QColor(255, 255, 255, 14))
        band.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), header_h), band)
        painter.setPen(QPen(QColor(BLUE_DIM), 1))
        painter.drawLine(
            QPointF(rect.left() + 6, rect.top() + header_h),
            QPointF(rect.right() - 6, rect.top() + header_h),
        )
    painter.restore()

    # orange corner brackets (top-left along the cut, bottom-right)
    painter.setPen(QPen(QColor(ORANGE), 2))
    c = _CUT
    painter.drawLine(QPointF(rect.left() + c + 1, rect.top() + 1), QPointF(rect.left() + c + 22, rect.top() + 1))
    painter.drawLine(QPointF(rect.left() + 1, rect.top() + c + 1), QPointF(rect.left() + 1, rect.top() + c + 22))
    painter.drawLine(QPointF(rect.right() - c - 22, rect.bottom() - 1), QPointF(rect.right() - c - 1, rect.bottom() - 1))
    painter.drawLine(QPointF(rect.right() - 1, rect.bottom() - c - 22), QPointF(rect.right() - 1, rect.bottom() - c - 1))
    # vent slashes top-right
    painter.setPen(QPen(QColor(ORANGE_DIM), 2))
    for i in range(3):
        x = rect.right() - 18 - i * 8
        painter.drawLine(QPointF(x, rect.top() + 6), QPointF(x - 6, rect.top() + 16))


def title_font(size: int = 13, spacing: float = 2.4) -> QFont:
    font = QFont()
    font.setPointSize(size)
    font.setWeight(QFont.Weight.Black)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


class HoloPanel(QFrame):
    """Cut-corner console panel with an orange title strip and glyph accents.

    Drop-in replacement for the old ChromePanel(index, title, content)."""

    HEADER_H = 44

    def __init__(self, index: str, title: str, content: QWidget, parent=None) -> None:
        super().__init__(parent)

        self.setObjectName("holoPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 10)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(6, 2, 22, 2)
        header.setSpacing(10)
        label = QLabel(title.upper())
        label.setObjectName("panelTitle")
        label.setFont(title_font(12, 1.2))
        header.addWidget(label)
        header.addStretch(1)
        glyphs = QLabel(index or _aurebesh_tag(title))
        glyphs.setObjectName("panelGlyphs")
        header.addWidget(glyphs)
        head_box = QWidget()
        head_box.setLayout(header)
        head_box.setFixedHeight(self.HEADER_H - 6)
        layout.addWidget(head_box)
        layout.addSpacing(4)
        layout.addWidget(content, 1)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        paint_panel_chrome(painter, rect, header_h=float(self.HEADER_H))
        painter.end()


def _aurebesh_tag(title: str) -> str:
    """Deterministic little glyph strip per panel — reads as in-universe signage."""
    glyphs = "⌐¬⌈⌉⌊⌋⊦⊪≡∆"
    rng = random.Random(hash(title) & 0xFFFF)
    return "".join(rng.choice(glyphs) for _ in range(4))


class ServoGauge(QSlider):
    """Custom-painted servo slider: dark track, orange fill to the current
    position, amber handle. QSS sub-page/add-page rendering is glitchy across
    Qt styles, so this paints the gauge directly — enabled = manual override
    (draggable, bright), disabled = live readout (dimmed fill)."""

    _TRACK_H = 5.0
    _HANDLE_W = 9.0
    _HANDLE_H = 15.0

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        cy = self.height() / 2.0
        span = max(1, self.maximum() - self.minimum())
        frac = (self.value() - self.minimum()) / span
        pad = self._HANDLE_W / 2.0 + 1
        x = pad + frac * (w - pad * 2)

        track = QRectF(pad, cy - self._TRACK_H / 2.0, w - pad * 2, self._TRACK_H)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#142230"))
        painter.drawRoundedRect(track, 2, 2)
        fill = QRectF(track.left(), track.top(), max(0.0, x - track.left()), track.height())
        painter.setBrush(QColor(ORANGE) if self.isEnabled() else QColor("#9c5e1b"))
        painter.drawRoundedRect(fill, 2, 2)
        handle = QRectF(x - self._HANDLE_W / 2.0, cy - self._HANDLE_H / 2.0,
                        self._HANDLE_W, self._HANDLE_H)
        painter.setBrush(QColor(AMBER) if self.isEnabled() else QColor("#c98636"))
        painter.drawRoundedRect(handle, 2, 2)
        painter.end()

    # Direct click-to-position (the default QStyle hit test doesn't know our
    # custom handle geometry).
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.isEnabled():
            self._set_from_x(event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.isEnabled() and event.buttons() & Qt.MouseButton.LeftButton:
            self._set_from_x(event.position().x())
        super().mouseMoveEvent(event)

    def _set_from_x(self, x: float) -> None:
        pad = self._HANDLE_W / 2.0 + 1
        w = max(1.0, self.width() - pad * 2)
        frac = min(1.0, max(0.0, (x - pad) / w))
        self.setValue(round(self.minimum() + frac * (self.maximum() - self.minimum())))


class StarfieldBackdrop(QWidget):
    """Page background: deep space gradient + deterministic starfield + faint grid.
    Cheap: rendered to a cached pixmap per size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cache: QPixmap | None = None
        self._cache_size = (0, 0)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        size = (self.width(), self.height())
        if self._cache is None or self._cache_size != size:
            self._cache = self._render(size)
            self._cache_size = size
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.end()

    def _render(self, size: tuple[int, int]) -> QPixmap:
        w, h = max(1, size[0]), max(1, size[1])
        pix = QPixmap(w, h)
        painter = QPainter(pix)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#071019"))
        grad.setColorAt(0.5, QColor(SPACE_BLACK))
        grad.setColorAt(1.0, QColor("#03070c"))
        painter.fillRect(0, 0, w, h, grad)
        # faint nebula glow top-left
        neb = QRadialGradient(QPointF(w * 0.18, h * 0.06), max(w, h) * 0.55)
        neb.setColorAt(0.0, QColor(46, 95, 157, 26))
        neb.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, h, neb)
        # deterministic starfield
        rng = random.Random(3263827)
        for _ in range(int(w * h / 6200)):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            b = rng.randint(60, 170)
            r = rng.choice((1, 1, 1, 2))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(b, b + 10, min(255, b + 30), rng.randint(90, 170)))
            painter.drawEllipse(QPointF(x, y), r * 0.6, r * 0.6)
        painter.end()
        return pix


# ── Shared stylesheet ────────────────────────────────────────────────────────
STYLE = f"""
QWidget#root {{
    background: transparent;
    color: {TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
QMainWindow {{ background: {SPACE_BLACK}; }}
QLabel#windowTitle {{
    color: {ORANGE};
    font-size: 16px;
    font-weight: 900;
    letter-spacing: 3px;
}}
QLabel#windowSubtitle {{
    color: {TEXT_DIM};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#connectionLabel {{ color: {GOOD}; font-size: 13px; }}
QLabel#deviceStatus {{ color: #9fb6cc; font-size: 12px; font-weight: 700; }}
QLabel#stateBadge {{
    color: #5b6b7d;
    border: 1px solid {BLUE_DIM};
    border-radius: 2px;
    padding: 3px 13px;
    font-size: 14px;
    font-weight: 900;
}}
QLabel#panelTitle {{ color: {ORANGE}; font-size: 13px; font-weight: 900; }}
QLabel#panelGlyphs {{ color: {BLUE_DIM}; font-size: 12px; font-weight: 900; letter-spacing: 2px; }}
QTextBrowser#conversationLog, QTextBrowser#visionDescription {{
    background: transparent;
    color: {TEXT};
    border: none;
}}
QPlainTextEdit#systemLog {{
    background: transparent;
    color: #9fb6cc;
    border: none;
    selection-background-color: #244f89;
}}
QLineEdit#messageEntry {{
    min-height: 36px;
    padding: 0 14px;
    background: #0d1926;
    color: #e0e9f2;
    border: 1px solid {BLUE_DIM};
    border-radius: 2px;
    font-size: 13px;
}}
QLineEdit#messageEntry:focus {{ border: 1px solid {BLUE}; }}
QPushButton#primaryButton {{
    min-height: 36px;
    padding: 0 18px;
    background: #17334f;
    color: #cfe5ff;
    border: 1px solid {BLUE};
    border-radius: 2px;
    font-weight: 800;
    letter-spacing: 1px;
}}
QPushButton#primaryButton:hover {{ background: #244f89; color: white; }}
QPushButton#servoOverrideButton {{
    min-height: 26px;
    padding: 0 10px;
    background: #0d1926;
    color: #cfe0f1;
    border: 1px solid {BLUE_DIM};
    border-radius: 2px;
    font-weight: 800;
    font-size: 11px;
}}
QPushButton#servoOverrideButton:hover {{ border: 1px solid {BLUE}; }}
QPushButton#servoOverrideButton[active="true"] {{
    background: #4a2c0e;
    color: #ffd9a8;
    border: 1px solid {ORANGE};
}}
QPushButton#memoryBanksButton, QPushButton#topControlButton {{
    min-height: 28px;
    padding: 0 13px;
    margin-left: 8px;
    background: #0d1926;
    color: #aee0ff;
    border: 1px solid {BLUE_DIM};
    border-radius: 2px;
    font-weight: 700;
}}
QPushButton#memoryBanksButton:hover, QPushButton#topControlButton:hover {{
    background: #1d2f44;
    border: 1px solid {BLUE};
}}
QPushButton#topShutdownButton {{
    min-height: 28px;
    padding: 0 13px;
    margin-left: 8px;
    background: #2a1416;
    color: #ffb3ab;
    border: 1px solid #5e2a2a;
    border-radius: 2px;
    font-weight: 700;
}}
QPushButton#topShutdownButton:hover {{ background: #4a1d1d; border: 1px solid #d05a5a; color: #ffffff; }}
QLabel#servoName {{ color: #cfd9e4; font-size: 11px; font-weight: 800; }}
QLabel#servoValue {{ color: {BLUE}; font-size: 11px; font-weight: 700; }}
QLabel#servoState {{ color: {TEXT_DIM}; font-size: 11px; }}
/* Servo gauges are custom-painted (theme.ServoGauge) — no QSS needed. */
QScrollBar:vertical {{ background: transparent; width: 9px; }}
QScrollBar::handle:vertical {{ background: #33506b; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BLUE_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""

# Dialog add-on: shared by shutdown confirm, motivator, and any modal.
DIALOG_STYLE = STYLE + f"""
QDialog {{
    background: {SPACE_BLACK};
    color: {TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    border: 1px solid {BORDER};
}}
QLabel#confirmText {{ color: #dbe7f3; font-size: 15px; font-weight: 700; }}
QFrame#chromePanel, QWidget#chromePanel {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {PANEL_TOP}, stop:1 {PANEL_BOTTOM});
    border: 1px solid {BORDER};
    border-radius: 2px;
}}
QLabel#dialogTitle {{ color: {ORANGE}; font-size: 14px; font-weight: 900; letter-spacing: 2px; }}
QPushButton#confirmNo {{
    min-height: 34px; padding: 0 20px;
    background: #0d1926; color: #aee0ff;
    border: 1px solid {BLUE_DIM}; border-radius: 2px; font-weight: 700;
}}
QPushButton#confirmNo:hover {{ background: #1d2f44; border: 1px solid {BLUE}; }}
QPushButton#confirmYes {{
    min-height: 34px; padding: 0 20px;
    background: #7a1f1f; color: #ffffff;
    border: 1px solid #a23a3a; border-radius: 2px; font-weight: 800;
}}
QPushButton#confirmYes:hover {{ background: #9a2a2a; border: 1px solid #d05a5a; }}
QLabel#motivatorConn {{ color: #8aa0b6; font-size: 12px; font-weight: 700; padding: 2px; }}
QLabel#motivatorConn[ok="true"] {{ color: {GOOD}; }}
QLabel#motivatorConn[ok="false"] {{ color: {WARN}; }}
QPushButton#motivatorStop {{
    min-height: 64px;
    background: #7a1f1f; color: #ffffff;
    border: 2px solid #a23a3a; border-radius: 4px;
    font-weight: 900; font-size: 26px; letter-spacing: 4px;
}}
QPushButton#motivatorStop:hover {{ background: #9a2a2a; border: 1px solid #d05a5a; }}
"""


class RebelMark(QWidget):
    """Painted insignia plate; no external image or font dependency."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(46, 46)
        self.setToolTip("Alliance field operations")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setWindow(0, 0, 100, 100)
        p.setPen(QPen(QColor("#976454"), 2))
        p.setBrush(QColor("#2b3029"))
        p.drawEllipse(QRectF(3, 3, 94, 94))
        bird = QPainterPath()
        bird.moveTo(50, 13)
        bird.cubicTo(60, 31, 62, 43, 58, 57)
        bird.cubicTo(72, 53, 83, 40, 82, 24)
        bird.cubicTo(106, 60, 78, 88, 50, 88)
        bird.cubicTo(22, 88, -6, 60, 18, 24)
        bird.cubicTo(17, 40, 28, 53, 42, 57)
        bird.cubicTo(38, 43, 40, 31, 50, 13)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#c56a4f"))
        p.drawPath(bird)
        p.end()


class RebelBackdrop(StarfieldBackdrop):
    """Cached, weathered console housing with recessed seams and worn paint."""

    def _render(self, size):
        w, h = max(1, size[0]), max(1, size[1])
        pix = QPixmap(w, h)
        p = QPainter(pix)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0, QColor("#34382f"))
        grad.setColorAt(.4, QColor("#1c231f"))
        grad.setColorAt(1, QColor("#121917"))
        p.fillRect(0, 0, w, h, grad)
        rng = random.Random(1977)
        for _ in range(int(w*h/170)):
            x, y = rng.randrange(w), rng.randrange(h)
            p.setPen(QColor(218, 207, 174, rng.randrange(3, 15)))
            p.drawLine(x, y, x + rng.randrange(1, 5), y)
        for y in (103, h-32):
            p.setPen(QPen(QColor("#080e0c"), 2))
            p.drawLine(12, y, w-12, y)
            p.setPen(QColor("#475045"))
            p.drawLine(12, y+2, w-12, y+2)
        p.setPen(QPen(QColor("#b86d47"), 3))
        p.drawLine(18, 3, 128, 3)
        p.setPen(QPen(QColor("#b4aa8e"), 3))
        p.drawLine(136, 3, 174, 3)
        for x in (7, w-7):
            for y in (8, h-9):
                p.setPen(QPen(QColor("#687065"), 1))
                p.setBrush(QColor("#151b17"))
                p.drawEllipse(QPointF(x, y), 3, 3)
                p.drawLine(x-1, y+1, x+1, y-1)
        p.end()
        return pix


REBEL_STYLE = """
QLabel#windowTitle { color: #eee6ce; font-size: 18px; letter-spacing: 2px; }
QLabel#windowSubtitle { color: #acae96; font-size: 9px; letter-spacing: 1.5px; }
QLabel#deviceStatus { color: #b7c3b7; font-size: 11px; }
QLabel#connectionLabel { font-size: 12px; }
QLabel#panelTitle { font-size: 12px; letter-spacing: 1.5px; }
QLabel#panelGlyphs { color: #95947e; font-size: 10px; }
QLabel#consoleFootnote { color: #a8ad99; font-size: 10px; letter-spacing: 1px; }
QPushButton#memoryBanksButton, QPushButton#topControlButton {
    background: #303b34; color: #ece5cf; border-color: #606a58;
    min-height: 30px; padding: 0 10px;
}
QPushButton#memoryBanksButton:hover, QPushButton#topControlButton:hover {
    background: #465448; border-color: #c8ba8d;
}
QPushButton#servoOverrideButton { background: #28352f; color: #dddcc8; border-color: #667361; }
QPushButton#servoOverrideButton:checked { background: #63472a; border-color: #e9a35a; }
QLabel#servoName { color: #dfdfcc; }
QLabel#servoValue { color: #a3d6ce; }
QLineEdit#messageEntry {
    min-height: 42px; font-size: 15px; color: #f0eadb;
    background: #111d19; border-color: #6d7c65; padding: 0 10px;
}
QLineEdit#messageEntry:focus { border-color: #e9a35a; }
QPushButton#primaryButton {
    min-height: 42px; padding: 0 14px; color: #181c17;
    background: #c99d61; border-color: #e9bf85;
}
QPushButton#primaryButton:hover { background: #edbc7c; color: #111711; }
QTabWidget#sensorTabs { background: #1b2520; }
QTabWidget#sensorTabs::pane { border: 0; background: transparent; }
QTabBar { background: #1b2520; }
QTabBar::tab {
    color: #a9b3a6; background: #1b2520; border-bottom: 2px solid #3b4a40;
    padding: 9px 8px; font-size: 10px; font-weight: 700;
}
QTabBar::tab:selected { color: #f0dbc0; background: #334137; border-bottom-color: #e9a35a; }
QTabBar::tab:hover { color: #ffffff; background: #3a493e; }
QSplitter#dashboardColumns::handle { background: transparent; }
QSplitter#dashboardColumns::handle:hover { background: #7d7853; }
QScrollBar::handle:vertical { background: #637461; }
"""
