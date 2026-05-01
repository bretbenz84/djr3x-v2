"""Jeopardy game view for the optional DJ-R3X dashboard."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from gui.rex_avatar import RexAvatar


class JeopardyPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state: dict[str, Any] = {}
        self._avatar = RexAvatar(self, show_background=False, show_grid=False)
        self._avatar.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setMinimumSize(980, 640)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._state = dict(snapshot.get("game_state") or {})
        self._avatar.set_snapshot(snapshot)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_background(painter)

        bounds = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        left_w = min(300.0, bounds.width() * 0.25)
        side_w = min(190.0, bounds.width() * 0.14)
        players_h = min(250.0, max(190.0, bounds.height() * 0.29))
        gap = 12.0

        mascot = QRectF(bounds.left(), bounds.top(), left_w, bounds.height() - players_h - gap)
        board = QRectF(
            mascot.right() + gap,
            bounds.top(),
            bounds.width() - left_w - side_w - gap * 2,
            bounds.height() - players_h - gap,
        )
        side = QRectF(board.right() + gap, bounds.top(), side_w, board.height())
        players = QRectF(bounds.left(), board.bottom() + gap, bounds.width(), players_h)

        self._draw_mascot(painter, mascot)
        self._draw_board(painter, board)
        self._draw_side_controls(painter, side)
        self._draw_players(painter, players)
        self._draw_current_clue(painter, board)
        painter.end()

    def _draw_background(self, painter: QPainter) -> None:
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, QColor("#030914"))
        grad.setColorAt(0.55, QColor("#071a2c"))
        grad.setColorAt(1, QColor("#02060c"))
        painter.fillRect(self.rect(), grad)

    def _draw_mascot(self, painter: QPainter, rect: QRectF) -> None:
        self._metal_panel(painter, rect, radius=10)
        title = QRectF(rect.left() + 18, rect.top() + 18, rect.width() - 36, 120)
        painter.setPen(QColor("#f3f7ff"))
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(title, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "WELCOME TO")
        font.setPointSize(30)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#ffb21e"))
        painter.drawText(title.adjusted(0, 30, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "JEOPARDY!")
        font.setPointSize(16)
        painter.setFont(font)
        painter.setPen(QColor("#f3f7ff"))
        prompt = _phase_prompt(self._state)
        painter.drawText(
            title.adjusted(0, 78, 0, 34),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            prompt,
        )

        bot = rect.adjusted(4, 138, -4, -88)
        self._avatar.setGeometry(bot.toRect())
        self._avatar.raise_()

        logo = QRectF(rect.left() + 30, rect.bottom() - 82, rect.width() - 60, 56)
        self._blue_panel(painter, logo)
        painter.setPen(QColor("#ffb21e"))
        font.setPointSize(26)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(logo, Qt.AlignmentFlag.AlignCenter, "JEOPARDY!")

    def _draw_rex_badge(self, painter: QPainter, rect: QRectF) -> None:
        cx = rect.center().x()
        top = rect.top()
        painter.setPen(QPen(QColor("#1d2833"), 4))
        painter.setBrush(QColor("#4d5155"))
        painter.drawRoundedRect(QRectF(cx - 86, top + 48, 172, 80), 14, 14)
        painter.setBrush(QColor("#dc7d22"))
        painter.drawChord(QRectF(cx - 74, top + 20, 148, 74), 0, 180 * 16)
        painter.setBrush(QColor("#0c1b37"))
        for ex in (cx - 34, cx + 34):
            painter.drawEllipse(QRectF(ex - 17, top + 68, 34, 34))
            painter.setBrush(QColor("#1d6cff"))
            painter.drawEllipse(QRectF(ex - 11, top + 74, 22, 22))
            painter.setBrush(QColor("#0c1b37"))
        painter.setPen(QPen(QColor("#15181b"), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(rect.center().x(), top + 122, rect.center().x(), top + 178)
        painter.setPen(QPen(QColor("#272b30"), 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(cx - 82, top + 180, cx - 145, top + 210)
        painter.drawLine(cx + 82, top + 180, cx + 145, top + 130)
        painter.setBrush(QColor("#db7c22"))
        painter.setPen(QPen(QColor("#623715"), 3))
        painter.drawRoundedRect(QRectF(cx - 92, top + 168, 184, 86), 18, 18)
        painter.setBrush(QColor("#30343a"))
        painter.drawRoundedRect(QRectF(cx - 100, top + 210, 200, 70), 10, 10)
        painter.setPen(QPen(QColor("#bcc0bd"), 4))
        for x in range(-70, 71, 24):
            painter.drawLine(cx + x, top + 220, cx + x, top + 268)

    def _draw_board(self, painter: QPainter, rect: QRectF) -> None:
        self._metal_panel(painter, rect, radius=10)
        inner = rect.adjusted(18, 18, -18, -18)
        cats = list(self._state.get("categories") or [])
        values = list(self._state.get("values") or [200, 400, 600, 800, 1000])
        if not cats:
            cats = [{"name": name, "remaining_values": values} for name in ["SCIENCE", "HISTORY", "LITERATURE", "POP CULTURE", "MATH"]]
        cats = cats[:6]
        col_count = max(1, len(cats))
        row_count = max(1, len(values))
        header_h = min(86.0, inner.height() * 0.16)
        cell_gap = 7.0
        col_w = (inner.width() - cell_gap * (col_count - 1)) / col_count
        row_h = (inner.height() - header_h - cell_gap * row_count) / row_count

        for col, category in enumerate(cats):
            x = inner.left() + col * (col_w + cell_gap)
            header = QRectF(x, inner.top(), col_w, header_h)
            self._blue_panel(painter, header)
            painter.setPen(QColor("#f4f6ff"))
            font = QFont()
            header_text = str(category.get("name") or "CATEGORY").upper()
            longest = max((len(part) for part in header_text.split()), default=len(header_text))
            font.setPointSize(max(9, int(min(15, col_w / max(7.0, longest * 0.72)))))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                header.adjusted(8, 4, -8, -4),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                header_text,
            )

            remaining = {int(v) for v in category.get("remaining_values") or []}
            for row, value in enumerate(values):
                cell = QRectF(
                    x,
                    inner.top() + header_h + cell_gap + row * (row_h + cell_gap),
                    col_w,
                    row_h,
                )
                available = int(value) in remaining
                self._blue_panel(painter, cell, dim=not available)
                painter.setPen(QColor("#ffb21e") if available else QColor("#1e3c6d"))
                font.setPointSize(max(14, int(min(23, row_h * 0.36, col_w / 5.4))))
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(cell.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignCenter, f"${int(value)}" if available else "")

    def _draw_current_clue(self, painter: QPainter, board: QRectF) -> None:
        clue = self._state.get("current_clue") or {}
        if not clue or self._state.get("phase") != "awaiting_answer":
            return
        card = QRectF(board.left() + board.width() * 0.10, board.top() + board.height() * 0.22, board.width() * 0.80, board.height() * 0.48)
        painter.setPen(QPen(QColor("#8291a2"), 3))
        painter.setBrush(QColor(4, 17, 44, 242))
        painter.drawRoundedRect(card, 8, 8)
        painter.setPen(QColor("#ffb21e"))
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        title = f"{clue.get('category', 'Category')} for ${clue.get('value', '')}"
        painter.drawText(card.adjusted(20, 14, -20, -20), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, title)
        painter.setPen(QColor("#f5f7ff"))
        font.setPointSize(24)
        painter.setFont(font)
        painter.drawText(
            card.adjusted(34, 70, -34, -28),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            str(clue.get("clue") or ""),
        )

    def _draw_players(self, painter: QPainter, rect: QRectF) -> None:
        self._metal_panel(painter, rect, radius=0)
        player_area = rect.adjusted(12, 12, -12, -12)
        players = list(self._state.get("players") or [])
        max_players = 4
        current = int(self._state.get("current_player_idx", 0) or 0)
        colors = ["#006fe8", "#d01818", "#32a326", "#8b24d2"]
        gap = 14.0
        card_w = (player_area.width() - gap * (max_players - 1)) / max_players
        for idx in range(max_players):
            player = players[idx] if idx < len(players) else {"name": f"PLAYER {idx + 1}", "score": 0}
            card = QRectF(player_area.left() + idx * (card_w + gap), player_area.top(), card_w, player_area.height())
            active = idx == current and bool(players)
            self._draw_player_card(painter, card, player, colors[idx], active=active, enrolled=idx < len(players))

    def _draw_player_card(self, painter: QPainter, rect: QRectF, player: dict, color: str, *, active: bool, enrolled: bool) -> None:
        self._metal_panel(painter, rect, radius=6, active=active)
        name_h = 42.0
        label = QRectF(rect.left() + 20, rect.top() + 12, rect.width() - 40, name_h)
        painter.setPen(QPen(QColor("#14191f"), 2))
        painter.setBrush(QColor(color if enrolled else "#1c355a"))
        painter.drawRoundedRect(label, 6, 6)
        painter.setPen(QColor("#f8fbff"))
        font = QFont()
        font.setPointSize(max(11, int(min(18, label.width() / 9))))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(label.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter, str(player.get("name") or "PLAYER").upper())

        portrait = QRectF(rect.left() + 18, label.bottom() + 12, rect.width() - 36, rect.height() - 108)
        grad = QLinearGradient(portrait.topLeft(), portrait.bottomRight())
        grad.setColorAt(0, QColor(color).darker(260))
        grad.setColorAt(1, QColor(color).darker(120) if enrolled else QColor("#05101d"))
        painter.setPen(QPen(QColor(color if enrolled else "#39516d"), 2))
        painter.setBrush(grad)
        painter.drawRoundedRect(portrait, 7, 7)
        self._draw_silhouette(painter, portrait, QColor(color if enrolled else "#2d587c"))

        score = QRectF(rect.left() + 18, rect.bottom() - 70, rect.width() - 36, 54)
        painter.setPen(QPen(QColor("#222830"), 2))
        painter.setBrush(QColor("#020304"))
        painter.drawRoundedRect(score, 5, 5)
        painter.setPen(QColor("#f7f7f7"))
        font.setPointSize(26)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(score, Qt.AlignmentFlag.AlignCenter, f"${int(player.get('score', 0) or 0)}")

    def _draw_silhouette(self, painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        cx = rect.center().x()
        head_r = min(rect.width(), rect.height()) * 0.17
        painter.drawEllipse(QRectF(cx - head_r, rect.top() + rect.height() * 0.26, head_r * 2, head_r * 2))
        body = QPainterPath()
        body.moveTo(cx, rect.top() + rect.height() * 0.54)
        body.cubicTo(rect.left() + rect.width() * 0.22, rect.top() + rect.height() * 0.62, rect.left() + rect.width() * 0.16, rect.bottom() - 8, rect.left() + rect.width() * 0.16, rect.bottom() - 8)
        body.lineTo(rect.right() - rect.width() * 0.16, rect.bottom() - 8)
        body.cubicTo(rect.right() - rect.width() * 0.16, rect.bottom() - 8, rect.right() - rect.width() * 0.22, rect.top() + rect.height() * 0.62, cx, rect.top() + rect.height() * 0.54)
        painter.drawPath(body)

    def _draw_side_controls(self, painter: QPainter, rect: QRectF) -> None:
        self._metal_panel(painter, rect, radius=9)
        self._draw_round_box(painter, rect.adjusted(16, 20, -16, -20), compact=True)

    def _draw_round_box(self, painter: QPainter, rect: QRectF, *, compact: bool = False) -> None:
        round_no = int(self._state.get("round", 1) or 1)
        if compact:
            top = QRectF(rect.left(), rect.top(), rect.width(), 96)
            self._blue_panel(painter, top, dim=True)
            font = QFont()
            font.setPointSize(17)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("#f8fbff"))
            painter.drawText(top.adjusted(4, 8, -4, -4), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "ROUND")
            font.setPointSize(30)
            painter.setFont(font)
            painter.setPen(QColor("#ffb21e"))
            painter.drawText(top.adjusted(4, 38, -4, -4), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, str(round_no))
            labels = ["SCOREBOARD", "RULES", "NEW GAME"]
            y = top.bottom() + 16
            for label in labels:
                button = QRectF(rect.left() + 8, y, rect.width() - 16, 48)
                self._blue_panel(painter, button)
                painter.setPen(QColor("#f8fbff"))
                font.setPointSize(max(10, int(min(14, button.width() / 9.8))))
                painter.setFont(font)
                painter.drawText(button, Qt.AlignmentFlag.AlignCenter, label)
                y += 62
            return

        painter.setPen(QColor("#d8e0ea"))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        phase = str(self._state.get("phase") or "idle").replace("_", " ").upper()
        painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"ROUND {round_no}  ·  {phase}")

    def _metal_panel(self, painter: QPainter, rect: QRectF, *, radius: float = 8, active: bool = False) -> None:
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0, QColor("#5e6468"))
        grad.setColorAt(0.45, QColor("#25292c"))
        grad.setColorAt(1, QColor("#111417"))
        painter.setPen(QPen(QColor("#9aa1a6") if active else QColor("#343a40"), 2))
        painter.setBrush(grad)
        painter.drawRoundedRect(rect, radius, radius)

    def _blue_panel(self, painter: QPainter, rect: QRectF, *, dim: bool = False) -> None:
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0, QColor("#053b92") if not dim else QColor("#06162d"))
        grad.setColorAt(1, QColor("#001748") if not dim else QColor("#030914"))
        painter.setPen(QPen(QColor("#0a0f18"), 2))
        painter.setBrush(grad)
        painter.drawRoundedRect(rect, 3, 3)


def _phase_prompt(state: dict[str, Any]) -> str:
    phase = state.get("phase")
    if phase == "awaiting_players":
        return "ADD CONTESTANTS."
    if phase == "voice_enroll":
        return "VOICE CHECK."
    if phase == "awaiting_answer":
        return "ANSWER THE CLUE."
    return "CHOOSE A CATEGORY AND VALUE."
