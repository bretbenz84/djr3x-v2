"""
gui/memory_banks.py — the "Memory Banks" editor window for the DJ-R3X dashboard.

A separate (non-modal) window that browses and EDITS both memory databases:
  • Rex's own episodic memories (rex.db) — view / edit / delete each memory.
  • The people Rex knows (people.db) — create / delete people, and per person edit core
    fields and add / edit / delete their stored facts, interests, and preferences.

While the window is open the robot's audio output is PAUSED (config.AUDIO_OUTPUT_SUPPRESSED)
so Rex doesn't talk over you mid-edit; the prior value is restored when it closes.
Closing this window does NOT close the program.

All data access goes through memory.admin (failure-safe, unit-tested separately).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config
from memory import admin

_log = logging.getLogger(__name__)

_ROLE_ID = Qt.ItemDataRole.UserRole


# Matches the main dashboard's palette/typography (gui/dashboard.py _STYLE): deep-navy
# ground, gradient panels, accent-blue headers, Apple system font.
_MEMORY_BANKS_STYLE = """
QMainWindow { background: #07111a; }
QWidget#memBankRoot {
    background: #07111a;
    color: #d9e3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel { color: #d9e3ee; }
QLabel#memSection {
    color: #4e94ff;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 1px;
    padding: 8px 2px 4px 2px;
}
QLabel#memMeta { color: #7c8a99; font-size: 12px; }
QLabel#memBanner {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b1824, stop:1 #08111a);
    color: #ffd479;
    border: 1px solid #6b5a1f;
    border-radius: 7px;
    padding: 7px 12px;
    font-weight: 700;
}
QListWidget, QTableWidget, QPlainTextEdit {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b1824, stop:1 #08111a);
    color: #d9e3ee;
    border: 1px solid #255484;
    border-radius: 7px;
    selection-background-color: #244f89;
    selection-color: #ffffff;
}
QListWidget::item { padding: 5px 6px; }
QListWidget::item:selected, QTableWidget::item:selected { background: #244f89; color: #ffffff; }
QTableWidget { gridline-color: #173049; }
QTableView { background: #0b1824; }
QHeaderView { background: #0e1d2b; border: none; }
QHeaderView::section {
    background: #0e1d2b;
    color: #9fb6cc;
    border: none;
    border-right: 1px solid #1c3247;
    border-bottom: 1px solid #255484;
    padding: 5px 8px;
    font-weight: 700;
}
QTableCornerButton::section { background: #0e1d2b; border: none; }
QLineEdit, QDoubleSpinBox {
    min-height: 28px;
    padding: 2px 10px;
    background: #111b27;
    color: #e0e9f2;
    border: 1px solid #2b4562;
    border-radius: 5px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus { border: 1px solid #65a2ff; }
QPushButton {
    min-height: 30px;
    padding: 4px 14px;
    background: #15212f;
    color: #dbe7f3;
    border: 1px solid #2b4562;
    border-radius: 5px;
    font-weight: 700;
}
QPushButton:hover { background: #1d2f44; border: 1px solid #65a2ff; }
QPushButton:pressed { background: #244f89; }
QPushButton#primary { background: #326bbe; color: #ffffff; border: 1px solid #4e8be4; }
QPushButton#primary:hover { background: #3b7fd9; border: 1px solid #65a2ff; }
QScrollBar:vertical { background: #07111a; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #244f89; border-radius: 5px; min-height: 24px; }
QScrollBar:horizontal { background: #07111a; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #244f89; border-radius: 5px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QStatusBar { color: #9fb6cc; background: #07111a; }
"""


def _section_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("memSection")
    return lab


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #1c3247;")
    return line


# Bold-red styling for destructive actions — the pale text-only red read as greyed-out.
_DANGER_QSS = (
    "QPushButton {"
    " min-height:30px; padding:4px 14px; background:#c62828; color:#ffffff;"
    " font-weight:800; border:1px solid #ff5252; border-radius:5px; }"
    "QPushButton:hover { background:#e53935; border:1px solid #ff8a80; }"
    "QPushButton:pressed { background:#8e0000; }"
)


def _danger_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(_DANGER_QSS)
    return btn


class MemoryBanksWindow(QMainWindow):
    """Memory browser/editor. Pauses robot audio output while open."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("R3X — Memory Banks")
        self.resize(1080, 720)
        self.setMinimumSize(820, 560)
        # Independent top-level window: closing it must NOT close the dashboard/app.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)

        # ── Pause the robot while editing ────────────────────────────────────
        # INTERACTION_PAUSED is the TRUE pause: it halts the conversation engine (no
        # transcription, responses, idle banter, or "are you there?" reactions — so no
        # wasted LLM calls). AUDIO_OUTPUT_SUPPRESSED additionally mutes any already-queued
        # audio. Both are restored to their prior values on close.
        self._prior_output_suppressed = bool(getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False))
        self._prior_interaction_paused = bool(getattr(config, "INTERACTION_PAUSED", False))
        config.INTERACTION_PAUSED = True
        config.AUDIO_OUTPUT_SUPPRESSED = True
        _log.info("[memory_banks] opened — conversation engine paused")

        self._current_person_id = None
        self._build_ui()
        self.reload_all()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("memBankRoot")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(10)

        banner = QLabel(
            "⏸  Conversation PAUSED while the Memory Banks are open — Rex won't listen, "
            "respond, or speak until you close this."
        )
        banner.setObjectName("memBanner")
        outer.addWidget(banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close_btn = QPushButton("Close (keep robot running)")
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

        self.setCentralWidget(root)
        self.setStyleSheet(_MEMORY_BANKS_STYLE)
        self.statusBar()  # create the status bar so toasts share the themed styling

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        # Rex's own memories
        col.addWidget(_section_label("R3X'S OWN MEMORIES"))
        self.rex_list = QListWidget()
        self.rex_list.itemSelectionChanged.connect(self._on_rex_selected)
        col.addWidget(self.rex_list, 5)
        rex_bar = QHBoxLayout()
        refresh_rex = QPushButton("Refresh")
        refresh_rex.clicked.connect(self.reload_rex_memories)
        rex_bar.addWidget(refresh_rex)
        rex_bar.addStretch(1)
        col.addLayout(rex_bar)

        col.addWidget(_hline())

        # People Rex knows
        col.addWidget(_section_label("PEOPLE R3X KNOWS"))
        self.people_list = QListWidget()
        self.people_list.itemSelectionChanged.connect(self._on_person_selected)
        col.addWidget(self.people_list, 5)
        ppl_bar = QHBoxLayout()
        new_person = QPushButton("+ New Person")
        new_person.clicked.connect(self._create_person)
        refresh_ppl = QPushButton("Refresh")
        refresh_ppl.clicked.connect(self.reload_people)
        ppl_bar.addWidget(new_person)
        ppl_bar.addWidget(refresh_ppl)
        ppl_bar.addStretch(1)
        col.addLayout(ppl_bar)

        return panel

    def _build_detail_panel(self) -> QWidget:
        self.detail = QStackedWidget()

        # 0 — placeholder
        placeholder = QLabel("Select a memory or a person on the left to view and edit it.")
        placeholder.setObjectName("memMeta")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.addWidget(placeholder)

        self.detail.addWidget(self._build_rex_editor())
        self.detail.addWidget(self._build_person_editor())
        return self.detail

    def _build_rex_editor(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(_section_label("EDIT MEMORY"))
        self.rex_meta = QLabel("")
        self.rex_meta.setObjectName("memMeta")
        v.addWidget(self.rex_meta)
        v.addWidget(QLabel("Kind:"))
        self.rex_kind = QLineEdit()
        v.addWidget(self.rex_kind)
        v.addWidget(QLabel("Memory (first-person):"))
        self.rex_summary = QPlainTextEdit()
        v.addWidget(self.rex_summary, 1)
        sal_row = QHBoxLayout()
        sal_row.addWidget(QLabel("Salience (0–1):"))
        self.rex_salience = QDoubleSpinBox()
        self.rex_salience.setRange(0.0, 1.0)
        self.rex_salience.setSingleStep(0.1)
        sal_row.addWidget(self.rex_salience)
        sal_row.addStretch(1)
        v.addLayout(sal_row)
        btns = QHBoxLayout()
        save = QPushButton("Save Memory")
        save.setObjectName("primary")
        save.clicked.connect(self._save_rex_memory)
        delete = _danger_button("Delete Memory")
        delete.clicked.connect(self._delete_rex_memory)
        btns.addWidget(save)
        btns.addWidget(delete)
        btns.addStretch(1)
        v.addLayout(btns)
        return w

    def _build_person_editor(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Header: name + nickname + person actions
        v.addWidget(_section_label("PERSON"))
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.p_name = QLineEdit()
        name_row.addWidget(self.p_name, 2)
        name_row.addWidget(QLabel("Nickname:"))
        self.p_nick = QLineEdit()
        name_row.addWidget(self.p_nick, 1)
        v.addLayout(name_row)

        self.p_meta = QLabel("")
        self.p_meta.setObjectName("memMeta")
        v.addWidget(self.p_meta)

        person_btns = QHBoxLayout()
        save_person = QPushButton("Save Person")
        save_person.setObjectName("primary")
        save_person.clicked.connect(self._save_person)
        del_person = _danger_button("Delete Person")
        del_person.clicked.connect(self._delete_person)
        person_btns.addWidget(save_person)
        person_btns.addWidget(del_person)
        person_btns.addStretch(1)
        v.addLayout(person_btns)

        v.addWidget(_hline())

        # Facts table (full CRUD)
        v.addWidget(_section_label("FACTS"))
        self.facts_table = QTableWidget(0, 4)
        self.facts_table.setHorizontalHeaderLabels(["Category", "Key", "Value", "Importance"])
        self.facts_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.facts_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        v.addWidget(self.facts_table, 3)
        fact_btns = QHBoxLayout()
        add_fact = QPushButton("Add Fact")
        add_fact.clicked.connect(self._add_fact_row)
        del_fact = _danger_button("Delete Selected Fact")
        del_fact.clicked.connect(self._delete_selected_fact)
        save_facts = QPushButton("Save Facts")
        save_facts.setObjectName("primary")
        save_facts.clicked.connect(self._save_facts)
        fact_btns.addWidget(add_fact)
        fact_btns.addWidget(del_fact)
        fact_btns.addWidget(save_facts)
        fact_btns.addStretch(1)
        v.addLayout(fact_btns)

        v.addWidget(_hline())

        # Interests + preferences (view + delete)
        lists_row = QHBoxLayout()
        int_col = QVBoxLayout()
        int_col.addWidget(_section_label("INTERESTS"))
        self.interests_list = QListWidget()
        int_col.addWidget(self.interests_list)
        del_int = _danger_button("Delete Selected Interest")
        del_int.clicked.connect(self._delete_selected_interest)
        int_col.addWidget(del_int)
        lists_row.addLayout(int_col)

        pref_col = QVBoxLayout()
        pref_col.addWidget(_section_label("PREFERENCES"))
        self.prefs_list = QListWidget()
        pref_col.addWidget(self.prefs_list)
        del_pref = _danger_button("Delete Selected Preference")
        del_pref.clicked.connect(self._delete_selected_preference)
        pref_col.addWidget(del_pref)
        lists_row.addLayout(pref_col)
        v.addLayout(lists_row, 2)

        return w

    # ── Loading ──────────────────────────────────────────────────────────────
    def reload_all(self) -> None:
        self.reload_rex_memories()
        self.reload_people()

    def reload_rex_memories(self) -> None:
        self.rex_list.clear()
        for mem in admin.list_rex_memories():
            date = str(mem.get("created_at") or "")[:16]
            kind = mem.get("kind") or "other"
            summary = (mem.get("summary") or "").replace("\n", " ")
            item = QListWidgetItem(f"[{kind}] {summary}   ·   {date}")
            item.setData(_ROLE_ID, int(mem["id"]))
            self.rex_list.addItem(item)

    def reload_people(self) -> None:
        self.people_list.clear()
        for person in admin.list_people():
            tier = person.get("friendship_tier") or "stranger"
            visits = person.get("visit_count") or 0
            label = f"{person.get('name') or '(unnamed)'}  —  {tier}, {visits} visit(s)"
            item = QListWidgetItem(label)
            item.setData(_ROLE_ID, int(person["id"]))
            self.people_list.addItem(item)

    # ── Rex memory editing ───────────────────────────────────────────────────
    def _selected_id(self, widget: QListWidget):
        items = widget.selectedItems()
        return items[0].data(_ROLE_ID) if items else None

    def _on_rex_selected(self) -> None:
        mem_id = self._selected_id(self.rex_list)
        if mem_id is None:
            return
        self.people_list.clearSelection()
        mem = next((m for m in admin.list_rex_memories() if m["id"] == mem_id), None)
        if not mem:
            return
        self._editing_rex_id = int(mem_id)
        self.rex_meta.setText(
            f"#{mem['id']}  ·  {mem.get('created_at','')}"
            + (f"  ·  about {mem.get('person_name')}" if mem.get("person_name") else "")
        )
        self.rex_kind.setText(mem.get("kind") or "")
        self.rex_summary.setPlainText(mem.get("summary") or "")
        self.rex_salience.setValue(float(mem.get("salience") or 0.5))
        self.detail.setCurrentIndex(1)

    def _save_rex_memory(self) -> None:
        mem_id = getattr(self, "_editing_rex_id", None)
        if mem_id is None:
            return
        admin.update_rex_memory(
            mem_id,
            summary=self.rex_summary.toPlainText(),
            kind=self.rex_kind.text(),
            salience=self.rex_salience.value(),
        )
        self.reload_rex_memories()
        self._toast("Memory saved.")

    def _delete_rex_memory(self) -> None:
        mem_id = getattr(self, "_editing_rex_id", None)
        if mem_id is None:
            return
        if not self._confirm("Delete this memory?", "This permanently removes the memory."):
            return
        admin.delete_rex_memory(mem_id)
        self._editing_rex_id = None
        self.detail.setCurrentIndex(0)
        self.reload_rex_memories()

    # ── Person editing ───────────────────────────────────────────────────────
    def _create_person(self) -> None:
        name, ok = QInputDialog.getText(self, "New Person", "Name:")
        if not ok or not name.strip():
            return
        new_id = admin.create_person(name.strip())
        if new_id is None:
            QMessageBox.warning(self, "New Person", "That name was rejected. Try a real name.")
            return
        self.reload_people()
        self._select_person_in_list(new_id)

    def _select_person_in_list(self, person_id: int) -> None:
        for i in range(self.people_list.count()):
            if self.people_list.item(i).data(_ROLE_ID) == person_id:
                self.people_list.setCurrentRow(i)
                return

    def _on_person_selected(self) -> None:
        person_id = self._selected_id(self.people_list)
        if person_id is None:
            return
        self.rex_list.clearSelection()
        self._load_person(int(person_id))

    def _load_person(self, person_id: int) -> None:
        detail = admin.get_person_detail(person_id)
        if not detail:
            return
        self._current_person_id = person_id
        person = detail["person"]
        self.p_name.setText(person.get("name") or "")
        self.p_nick.setText(person.get("nickname") or "")
        self.p_meta.setText(
            f"#{person_id}  ·  tier {person.get('friendship_tier','?')}  ·  "
            f"warmth {float(person.get('warmth_score') or 0):.2f}  ·  "
            f"antagonism {float(person.get('antagonism_score') or 0):.2f}  ·  "
            f"{person.get('visit_count') or 0} visit(s)"
        )

        # Facts table
        self.facts_table.setRowCount(0)
        for fact in detail["facts"]:
            self._append_fact_row(
                fact_id=int(fact["id"]),
                category=fact.get("category") or "",
                key=fact.get("key") or "",
                value=fact.get("value") or "",
                importance=float(fact.get("importance") or 0.5),
            )

        # Interests
        self.interests_list.clear()
        for it in detail["interests"]:
            label = f"{it.get('name','?')} ({it.get('interest_strength') or 'medium'})"
            li = QListWidgetItem(label)
            li.setData(_ROLE_ID, int(it["id"]))
            self.interests_list.addItem(li)

        # Preferences
        self.prefs_list.clear()
        for pr in detail["preferences"]:
            label = f"{pr.get('domain','')}.{pr.get('preference_type','')}: {pr.get('value') or pr.get('key') or ''}"
            li = QListWidgetItem(label)
            li.setData(_ROLE_ID, int(pr["id"]))
            self.prefs_list.addItem(li)

        self.detail.setCurrentIndex(2)

    def _append_fact_row(self, *, fact_id, category, key, value, importance) -> None:
        row = self.facts_table.rowCount()
        self.facts_table.insertRow(row)
        cat_item = QTableWidgetItem(category)
        cat_item.setData(_ROLE_ID, fact_id)  # None for a new, unsaved row
        self.facts_table.setItem(row, 0, cat_item)
        self.facts_table.setItem(row, 1, QTableWidgetItem(key))
        self.facts_table.setItem(row, 2, QTableWidgetItem(value))
        self.facts_table.setItem(row, 3, QTableWidgetItem(f"{importance:.2f}"))

    def _add_fact_row(self) -> None:
        if self._current_person_id is None:
            return
        self._append_fact_row(fact_id=None, category="other", key="", value="", importance=0.5)
        self.facts_table.editItem(self.facts_table.item(self.facts_table.rowCount() - 1, 1))

    def _cell(self, row: int, col: int) -> str:
        item = self.facts_table.item(row, col)
        return item.text().strip() if item else ""

    def _commit_open_cell_editor(self) -> None:
        """Commit an in-progress fact-cell edit before reading the table. On macOS a
        push-button doesn't take focus on click, so clicking "Save Facts" leaves the
        cell editor open and uncommitted — the value would be read stale otherwise."""
        table = self.facts_table
        if table.state() != QAbstractItemView.State.EditingState:
            return
        editor = table.focusWidget()
        if editor is None or editor in (table, table.viewport()):
            editor = table.viewport().findChild(QLineEdit)
        if editor is not None:
            try:
                table.itemDelegate().commitData.emit(editor)
            except Exception:
                pass

    def _save_facts(self) -> None:
        if self._current_person_id is None:
            return
        self._commit_open_cell_editor()
        for row in range(self.facts_table.rowCount()):
            cat_item = self.facts_table.item(row, 0)
            fact_id = cat_item.data(_ROLE_ID) if cat_item else None
            category = self._cell(row, 0) or "other"
            key = self._cell(row, 1)
            value = self._cell(row, 2)
            try:
                importance = float(self._cell(row, 3) or 0.5)
            except ValueError:
                importance = 0.5
            if not key or not value:
                continue
            if fact_id is None:
                admin.add_person_fact(
                    self._current_person_id, category, key, value, importance=importance
                )
            else:
                admin.update_fact(
                    int(fact_id), category=category, key=key, value=value, importance=importance
                )
        self._load_person(self._current_person_id)
        self._toast("Facts saved.")

    def _delete_selected_fact(self) -> None:
        row = self.facts_table.currentRow()
        if row < 0:
            return
        cat_item = self.facts_table.item(row, 0)
        fact_id = cat_item.data(_ROLE_ID) if cat_item else None
        if fact_id is not None:
            admin.delete_fact(int(fact_id))
        self.facts_table.removeRow(row)

    def _delete_selected_interest(self) -> None:
        iid = self._selected_id(self.interests_list)
        if iid is not None:
            admin.delete_interest(int(iid))
            if self._current_person_id is not None:
                self._load_person(self._current_person_id)

    def _delete_selected_preference(self) -> None:
        pid = self._selected_id(self.prefs_list)
        if pid is not None:
            admin.delete_preference(int(pid))
            if self._current_person_id is not None:
                self._load_person(self._current_person_id)

    def _save_person(self) -> None:
        if self._current_person_id is None:
            return
        admin.update_person_fields(
            self._current_person_id,
            name=self.p_name.text(),
            nickname=self.p_nick.text(),
        )
        self.reload_people()
        self._load_person(self._current_person_id)
        self._toast("Person saved.")

    def _delete_person(self) -> None:
        if self._current_person_id is None:
            return
        name = self.p_name.text() or "this person"
        if not self._confirm(
            f"Delete {name}?",
            "This permanently removes the person and ALL of their facts, interests, "
            "preferences, and voice/face data.",
        ):
            return
        admin.delete_person(self._current_person_id)
        self._current_person_id = None
        self.detail.setCurrentIndex(0)
        self.reload_people()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _confirm(self, title: str, body: str) -> bool:
        return (
            QMessageBox.question(
                self, title, body,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _toast(self, msg: str) -> None:
        try:
            self.statusBar().showMessage(msg, 2500)
        except Exception:
            pass

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Restore the robot to whatever pause/mute state it was in before we opened.
        config.AUDIO_OUTPUT_SUPPRESSED = self._prior_output_suppressed
        config.INTERACTION_PAUSED = self._prior_interaction_paused
        _log.info("[memory_banks] closed — conversation engine resumed")
        super().closeEvent(event)
