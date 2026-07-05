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
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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


# Shared Star Wars console theme (gui/theme.py): space-black ground, angular 2px
# corners, R3X-orange section headers, holo-blue data/borders.
_MEMORY_BANKS_STYLE = """
QMainWindow { background: #040a11; }
QWidget#memBankRoot {
    background: #040a11;
    color: #d9e3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel { color: #d9e3ee; }
QLabel#memSection {
    color: #e08428;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 2px;
    padding: 8px 2px 4px 2px;
}
QLabel#memMeta { color: #7c8a99; font-size: 12px; }
QLabel#memBanner {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b1722, stop:1 #060e16);
    color: #ffb21e;
    border: 1px solid #8c5316;
    border-radius: 2px;
    padding: 7px 12px;
    font-weight: 700;
}
QListWidget, QTableWidget, QPlainTextEdit {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b1722, stop:1 #060e16);
    color: #d9e3ee;
    border: 1px solid #24486b;
    border-radius: 2px;
    selection-background-color: #244f89;
    selection-color: #ffffff;
}
QListWidget::item { padding: 5px 6px; }
QListWidget::item:selected, QTableWidget::item:selected { background: #244f89; color: #ffffff; }
QTableWidget {
    gridline-color: #14283c;
    font-size: 13px;
    alternate-background-color: #0a1420;
}
QLabel#memHint { color: #6f8296; font-size: 11px; padding: 0 2px; }
QTableView { background: #0b1722; }
QHeaderView { background: #0c1826; border: none; }
QHeaderView::section {
    background: #0c1826;
    color: #e08428;
    border: none;
    border-right: 1px solid #14283c;
    border-bottom: 1px solid #8c5316;
    padding: 5px 8px;
    font-weight: 800;
    letter-spacing: 1px;
}
QTableCornerButton::section { background: #0c1826; border: none; }
QLineEdit, QDoubleSpinBox {
    min-height: 28px;
    padding: 2px 10px;
    background: #0d1926;
    color: #e0e9f2;
    border: 1px solid #2b4a66;
    border-radius: 2px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus { border: 1px solid #4e94ff; }
QComboBox {
    min-height: 24px;
    padding: 1px 6px;
    background: #0d1926;
    color: #e0e9f2;
    border: 1px solid #2b4a66;
    border-radius: 2px;
}
QComboBox:focus { border: 1px solid #4e94ff; }
QComboBox QAbstractItemView {
    background: #0c1826;
    color: #d9e3ee;
    border: 1px solid #24486b;
    selection-background-color: #244f89;
    selection-color: #ffffff;
}
QPushButton {
    min-height: 30px;
    padding: 4px 14px;
    background: #0d1926;
    color: #cfe0f1;
    border: 1px solid #2b4a66;
    border-radius: 2px;
    font-weight: 700;
}
QPushButton:hover { background: #1d2f44; border: 1px solid #4e94ff; }
QPushButton:pressed { background: #244f89; }
QPushButton#primary { background: #4a2c0e; color: #ffd9a8; border: 1px solid #e08428; }
QPushButton#primary:hover { background: #6b3f14; border: 1px solid #ffb21e; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #33506b; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #2b4a66; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #33506b; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QStatusBar { color: #9fb6cc; background: #040a11; }
/* Pop-up dialogs (confirm/prompt/warning) are top-level windows that do NOT inherit
   this window's stylesheet, so they render with the white OS default unless this sheet
   is applied to them directly — see _themed_messagebox / the New Person prompt. */
QMessageBox, QInputDialog, QDialog {
    background: #0b1722;
    color: #d9e3ee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
}
QMessageBox QLabel, QInputDialog QLabel { color: #d9e3ee; }
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
    " min-height:30px; padding:4px 14px; background:#7a1f1f; color:#ffffff;"
    " font-weight:800; border:1px solid #a23a3a; border-radius:2px; }"
    "QPushButton:hover { background:#9a2a2a; border:1px solid #d05a5a; }"
    "QPushButton:pressed { background:#5e1414; }"
)


def _danger_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(_DANGER_QSS)
    return btn


# Unsaved-state chip styles — the single at-a-glance answer to "do I need to save?".
_CHIP_DIRTY_QSS = (
    "QLabel { background:#4a2c0e; color:#ffb21e; border:1px solid #e08428;"
    " border-radius:2px; padding:3px 10px; font-weight:800; letter-spacing:1px; }"
)
_CHIP_CLEAN_QSS = (
    "QLabel { background:transparent; color:#5f7186; border:1px solid #1c3247;"
    " border-radius:2px; padding:3px 10px; font-weight:700; letter-spacing:1px; }"
)
# Amber tint for edited/new (not yet saved) fact rows.
_UNSAVED_ROW_BRUSH_RGB = (74, 44, 14)


class MemoryBanksWindow(QMainWindow):
    """Memory browser/editor. Pauses robot audio output while open."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("R3X — Memory Banks")
        self.setMinimumSize(820, 560)
        # Open large by default: ~90% of the available screen, centered.
        try:
            screen = self.screen() or QGuiApplication.primaryScreen()
            avail = screen.availableGeometry()
            w = int(avail.width() * 0.9)
            h = int(avail.height() * 0.9)
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        except Exception:
            self.resize(1280, 860)
        # Independent top-level window: closing it must NOT close the dashboard/app.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)

        # ── Pause the robot while editing ────────────────────────────────────
        # INTERACTION_PAUSED is the TRUE pause: it halts the conversation engine (no
        # transcription, responses, idle banter, or "are you there?" reactions — so no
        # wasted LLM calls). AUDIO_OUTPUT_SUPPRESSED additionally mutes any already-queued
        # audio. Both are restored to their prior values on close.
        # KNOWN GAP (junecodereview finding #8): only the AUDIO loop honors this today —
        # interaction.submit_text() (the GUI/CLI text path) does NOT check INTERACTION_PAUSED,
        # so typing into the dashboard while this window is open still generates a reply and
        # can write memories. Fix is a one-line early-return in submit_text().
        self._prior_output_suppressed = bool(getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False))
        self._prior_interaction_paused = bool(getattr(config, "INTERACTION_PAUSED", False))
        config.INTERACTION_PAUSED = True
        config.AUDIO_OUTPUT_SUPPRESSED = True
        _log.info("[memory_banks] opened — conversation engine paused")

        self._current_person_id = None
        self._editing_rex_id = None
        # Unsaved-work tracking. _loading suppresses dirty signals while an editor is
        # being POPULATED (programmatic setText/setItem fire the same signals as typing).
        self._loading = False
        self._rex_dirty = False
        self._person_dirty = False
        # Last committed selection (list widget, row) — restored when the user cancels
        # a navigation away from unsaved changes.
        self._nav_anchor = None
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

        # State the save model plainly so nobody has to guess it from behavior.
        hint = QLabel(
            "HOW SAVING WORKS:   Deletions apply IMMEDIATELY (after a confirmation).   "
            "Edits and new rows are held until you press SAVE — unsaved work is marked "
            "in amber, and switching away will ask what to do with it."
        )
        hint.setObjectName("memHint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        # Stretch factors only shape RESIZES; without initial sizes the left column
        # opens at its size hint (~320px) and the detail pane is a wasteland.
        splitter.setSizes([460, 820])

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

        # 0 — placeholder (themed empty state, not a bare line of grey text)
        placeholder = QLabel(
            "<div style='text-align:center;'>"
            "<span style='font-size:44px; color:#24486b;'>⌈ 🧠 ⌉</span><br/><br/>"
            "<span style='font-size:15px; font-weight:800; color:#e08428; letter-spacing:2px;'>"
            "MEMORY BANKS</span><br/><br/>"
            "<span style='color:#8ba0b5;'>Select one of R3X's memories or a person on the left<br/>"
            "to view and edit it.</span></div>"
        )
        placeholder.setObjectName("memMeta")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.addWidget(placeholder)

        self.detail.addWidget(self._build_rex_editor())
        self.detail.addWidget(self._build_person_editor())
        return self.detail

    def _build_rex_editor(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        head = QHBoxLayout()
        head.addWidget(_section_label("EDIT MEMORY"))
        head.addStretch(1)
        self.rex_chip = QLabel("ALL SAVED")
        self.rex_chip.setStyleSheet(_CHIP_CLEAN_QSS)
        head.addWidget(self.rex_chip)
        v.addLayout(head)
        self.rex_meta = QLabel("")
        self.rex_meta.setObjectName("memMeta")
        v.addWidget(self.rex_meta)
        v.addWidget(QLabel("Kind:"))
        self.rex_kind = QLineEdit()
        self.rex_kind.textEdited.connect(lambda _t: self._mark_rex_dirty())
        v.addWidget(self.rex_kind)
        v.addWidget(QLabel("Memory (first-person):"))
        self.rex_summary = QPlainTextEdit()
        self.rex_summary.textChanged.connect(self._mark_rex_dirty)  # guarded by _loading
        v.addWidget(self.rex_summary, 1)
        sal_row = QHBoxLayout()
        sal_row.addWidget(QLabel("Salience (0–1):"))
        self.rex_salience = QDoubleSpinBox()
        self.rex_salience.setRange(0.0, 1.0)
        self.rex_salience.setSingleStep(0.1)
        self.rex_salience.valueChanged.connect(lambda _v: self._mark_rex_dirty())
        sal_row.addWidget(self.rex_salience)
        sal_row.addStretch(1)
        v.addLayout(sal_row)
        btns = QHBoxLayout()
        self.rex_save_btn = QPushButton("Save Memory")
        self.rex_save_btn.setObjectName("primary")
        self.rex_save_btn.clicked.connect(self._save_rex_memory)
        self.rex_save_btn.setEnabled(False)   # nothing to save until something changes
        delete = _danger_button("Delete Memory  (immediate)")
        delete.clicked.connect(self._delete_rex_memory)
        btns.addWidget(self.rex_save_btn)
        btns.addWidget(delete)
        btns.addStretch(1)
        v.addLayout(btns)
        return w

    def _build_person_editor(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Header: name + nickname + person actions + the unsaved-state chip
        head = QHBoxLayout()
        head.addWidget(_section_label("PERSON"))
        head.addStretch(1)
        self.person_chip = QLabel("ALL SAVED")
        self.person_chip.setStyleSheet(_CHIP_CLEAN_QSS)
        head.addWidget(self.person_chip)
        v.addLayout(head)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.p_name = QLineEdit()
        self.p_name.textEdited.connect(lambda _t: self._mark_person_dirty())
        name_row.addWidget(self.p_name, 2)
        name_row.addWidget(QLabel("Nickname:"))
        self.p_nick = QLineEdit()
        self.p_nick.textEdited.connect(lambda _t: self._mark_person_dirty())
        name_row.addWidget(self.p_nick, 1)
        v.addLayout(name_row)

        self.p_meta = QLabel("")
        self.p_meta.setObjectName("memMeta")
        v.addWidget(self.p_meta)

        person_btns = QHBoxLayout()
        self.person_save_btn = QPushButton("Save All Changes")
        self.person_save_btn.setObjectName("primary")
        self.person_save_btn.setToolTip(
            "Saves the name/nickname AND every edited or added fact row in one click."
        )
        self.person_save_btn.clicked.connect(self._save_person_and_facts)
        self.person_save_btn.setEnabled(False)
        del_person = _danger_button("Delete Person  (immediate)")
        del_person.clicked.connect(self._delete_person)
        person_btns.addWidget(self.person_save_btn)
        person_btns.addWidget(del_person)
        person_btns.addStretch(1)
        v.addLayout(person_btns)

        v.addWidget(_hline())

        # Biometrics — one compact line: status + the two clear actions.
        bio_row = QHBoxLayout()
        bio_row.addWidget(_section_label("BIOMETRICS"))
        self.bio_label = QLabel("")
        self.bio_label.setObjectName("memMeta")
        bio_row.addWidget(self.bio_label, 1)
        self.clear_face_btn = _danger_button("Clear Face Data")
        self.clear_face_btn.clicked.connect(lambda: self._clear_biometric("face"))
        self.clear_voice_btn = _danger_button("Clear Voiceprint")
        self.clear_voice_btn.clicked.connect(lambda: self._clear_biometric("voice"))
        bio_row.addWidget(self.clear_face_btn)
        bio_row.addWidget(self.clear_voice_btn)
        v.addLayout(bio_row)

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
        # Category and Key are both guided dropdowns (editable, so custom text is still
        # allowed). Category drives how a fact decays / how important it is; the Key menu
        # changes to match the chosen category (e.g. relationship → boss/coworker/mentor).
        self.facts_table.horizontalHeaderItem(0).setToolTip(
            "Category controls how the fact is remembered (e.g. birthday/identity/"
            "relationship never fade; family/pet/preference are high-importance). Pick "
            "from the list, or type a custom one."
        )
        self.facts_table.horizontalHeaderItem(1).setToolTip(
            "The KIND of thing (the VALUE holds the specifics). The menu matches the "
            "category — e.g. relationship → boss / coworker / mentor; family → nephew / "
            "spouse. Editable, so you can type your own."
        )
        # Readability: taller rows (the combos need air), real starting column widths
        # (Key used to squeeze into ~90px and truncate), alternating row stripes, and
        # edits tracked per-row so unsaved rows can be tinted amber.
        self.facts_table.setAlternatingRowColors(True)
        self.facts_table.verticalHeader().setDefaultSectionSize(34)
        self.facts_table.setColumnWidth(0, 150)
        self.facts_table.setColumnWidth(1, 240)
        self.facts_table.setColumnWidth(3, 100)
        self.facts_table.itemChanged.connect(self._on_fact_item_changed)
        v.addWidget(self.facts_table, 5)
        fact_btns = QHBoxLayout()
        add_fact = QPushButton("+ Add Fact")
        add_fact.clicked.connect(self._add_fact_row)
        del_fact = _danger_button("Delete Selected Fact  (immediate)")
        del_fact.clicked.connect(self._delete_selected_fact)
        fact_btns.addWidget(add_fact)
        fact_btns.addWidget(del_fact)
        fact_btns.addStretch(1)
        fact_note = QLabel("edited / new rows are amber until saved")
        fact_note.setObjectName("memHint")
        fact_btns.addWidget(fact_note)
        v.addLayout(fact_btns)

        v.addWidget(_hline())

        # Interests + preferences (view + delete; deletes are immediate)
        lists_row = QHBoxLayout()
        int_col = QVBoxLayout()
        int_col.addWidget(_section_label("INTERESTS"))
        self.interests_list = QListWidget()
        self.interests_list.setWordWrap(True)
        int_col.addWidget(self.interests_list)
        del_int = _danger_button("Delete Selected Interest  (immediate)")
        del_int.clicked.connect(self._delete_selected_interest)
        int_col.addWidget(del_int)
        lists_row.addLayout(int_col)

        pref_col = QVBoxLayout()
        pref_col.addWidget(_section_label("PREFERENCES"))
        self.prefs_list = QListWidget()
        self.prefs_list.setWordWrap(True)
        pref_col.addWidget(self.prefs_list)
        del_pref = _danger_button("Delete Selected Preference  (immediate)")
        del_pref.clicked.connect(self._delete_selected_preference)
        pref_col.addWidget(del_pref)
        lists_row.addLayout(pref_col)
        v.addLayout(lists_row, 3)

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

    # ── Unsaved-work tracking ────────────────────────────────────────────────
    def _mark_rex_dirty(self) -> None:
        if self._loading or self._editing_rex_id is None:
            return
        self._rex_dirty = True
        self._refresh_dirty_ui()

    def _mark_person_dirty(self) -> None:
        if self._loading or self._current_person_id is None:
            return
        self._person_dirty = True
        self._refresh_dirty_ui()

    def _clear_dirty(self) -> None:
        self._rex_dirty = False
        self._person_dirty = False
        self._refresh_dirty_ui()

    def _refresh_dirty_ui(self) -> None:
        """One place drives every 'do I need to save?' signal: the chips, the save
        buttons' enabled state, and the window-title asterisk."""
        for chip, btn, dirty in (
            (getattr(self, "rex_chip", None), getattr(self, "rex_save_btn", None), self._rex_dirty),
            (getattr(self, "person_chip", None), getattr(self, "person_save_btn", None), self._person_dirty),
        ):
            if chip is not None:
                chip.setText("●  UNSAVED — press SAVE" if dirty else "ALL SAVED")
                chip.setStyleSheet(_CHIP_DIRTY_QSS if dirty else _CHIP_CLEAN_QSS)
            if btn is not None:
                btn.setEnabled(dirty)
        dirty_any = self._rex_dirty or self._person_dirty
        self.setWindowTitle("R3X — Memory Banks" + ("  *UNSAVED*" if dirty_any else ""))

    def _confirm_leave(self) -> bool:
        """Called before navigating away from (or closing over) unsaved changes.
        Returns True when it is OK to proceed (saved or deliberately discarded)."""
        if not (self._rex_dirty or self._person_dirty):
            return True
        box = self._themed_messagebox(
            QMessageBox.Icon.Warning,
            "Unsaved changes",
            "You have unsaved edits here. Save them before switching?",
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            if self._rex_dirty:
                self._save_rex_memory()
            if self._person_dirty:
                self._save_person_and_facts()
        else:
            self._clear_dirty()   # deliberate discard
        return True

    def _restore_nav_anchor(self) -> None:
        """Put the selection back where it was (used when a guard is cancelled)."""
        if not self._nav_anchor:
            return
        widget, row = self._nav_anchor
        other = self.people_list if widget is self.rex_list else self.rex_list
        for lst in (widget, other):
            lst.blockSignals(True)
        other.clearSelection()
        if 0 <= row < widget.count():
            widget.setCurrentRow(row)
        for lst in (widget, other):
            lst.blockSignals(False)

    # ── Rex memory editing ───────────────────────────────────────────────────
    def _selected_id(self, widget: QListWidget):
        items = widget.selectedItems()
        return items[0].data(_ROLE_ID) if items else None

    def _on_rex_selected(self) -> None:
        mem_id = self._selected_id(self.rex_list)
        if mem_id is None:
            return
        if int(mem_id) != (self._editing_rex_id or -1) and not self._confirm_leave():
            self._restore_nav_anchor()
            return
        self.people_list.clearSelection()
        mem = next((m for m in admin.list_rex_memories() if m["id"] == mem_id), None)
        if not mem:
            return
        self._editing_rex_id = int(mem_id)
        self._loading = True
        try:
            self.rex_meta.setText(
                f"#{mem['id']}  ·  {mem.get('created_at','')}"
                + (f"  ·  about {mem.get('person_name')}" if mem.get("person_name") else "")
            )
            self.rex_kind.setText(mem.get("kind") or "")
            self.rex_summary.setPlainText(mem.get("summary") or "")
            self.rex_salience.setValue(float(mem.get("salience") or 0.5))
        finally:
            self._loading = False
        self._clear_dirty()
        self._nav_anchor = (self.rex_list, self.rex_list.currentRow())
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
        self._rex_dirty = False
        self._refresh_dirty_ui()
        # Reload the list without disturbing the current editing session.
        self.rex_list.blockSignals(True)
        self.reload_rex_memories()
        self.rex_list.blockSignals(False)
        self._toast("Memory saved.")

    def _delete_rex_memory(self) -> None:
        mem_id = getattr(self, "_editing_rex_id", None)
        if mem_id is None:
            return
        if not self._confirm("Delete this memory?", "This permanently removes the memory."):
            return
        admin.delete_rex_memory(mem_id)
        self._editing_rex_id = None
        self._clear_dirty()
        self.detail.setCurrentIndex(0)
        self.reload_rex_memories()

    # ── Person editing ───────────────────────────────────────────────────────
    def _create_person(self) -> None:
        # Build the prompt by hand (not QInputDialog.getText) so the dashboard theme
        # applies — the static helper spawns an unstyled white dialog.
        dlg = QInputDialog(self)
        dlg.setWindowTitle("New Person")
        dlg.setLabelText("Name:")
        dlg.setStyleSheet(_MEMORY_BANKS_STYLE)
        if not dlg.exec():
            return
        name = dlg.textValue()
        if not name.strip():
            return
        new_id = admin.create_person(name.strip())
        if new_id is None:
            self._warn("New Person", "That name was rejected. Try a real name.")
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
        if int(person_id) != (self._current_person_id or -1) and not self._confirm_leave():
            self._restore_nav_anchor()
            return
        self.rex_list.clearSelection()
        self._load_person(int(person_id))
        self._nav_anchor = (self.people_list, self.people_list.currentRow())

    def _load_person(self, person_id: int) -> None:
        detail = admin.get_person_detail(person_id)
        if not detail:
            return
        self._current_person_id = person_id
        self._loading = True
        try:
            person = detail["person"]
            self.p_name.setText(person.get("name") or "")
            self.p_nick.setText(person.get("nickname") or "")
            self.p_meta.setText(
                f"#{person_id}  ·  tier {person.get('friendship_tier','?')}  ·  "
                f"warmth {float(person.get('warmth_score') or 0):.2f}  ·  "
                f"antagonism {float(person.get('antagonism_score') or 0):.2f}  ·  "
                f"{person.get('visit_count') or 0} visit(s)"
            )

            # Biometrics — can Rex recognize this person by face / voice?
            bio = detail.get("biometrics") or {}
            face_n = int(bio.get("face") or 0)
            voice_n = int(bio.get("voice") or 0)
            self.bio_label.setText(
                f"Face ID: {'✓ ' + str(face_n) + ' stored' if face_n else '✗ none'}"
                f"        Voiceprint: {'✓ ' + str(voice_n) + ' stored' if voice_n else '✗ none'}"
            )
            self.clear_face_btn.setEnabled(face_n > 0)
            self.clear_voice_btn.setEnabled(voice_n > 0)

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

            # Interests — "name · strength", notes in the tooltip.
            self.interests_list.clear()
            for it in detail["interests"]:
                label = f"{it.get('name','?')}   ·   {it.get('interest_strength') or 'medium'}"
                li = QListWidgetItem(label)
                li.setData(_ROLE_ID, int(it["id"]))
                notes = (it.get("notes") or "").strip()
                if notes:
                    li.setToolTip(notes)
                self.interests_list.addItem(li)

            # Preferences — "domain · type — value", full text in the tooltip.
            self.prefs_list.clear()
            for pr in detail["preferences"]:
                val = pr.get("value") or pr.get("key") or ""
                label = f"{pr.get('domain','?')} · {pr.get('preference_type','?')}  —  {val}"
                li = QListWidgetItem(label)
                li.setData(_ROLE_ID, int(pr["id"]))
                li.setToolTip(label)
                self.prefs_list.addItem(li)
        finally:
            self._loading = False
        self._clear_dirty()
        self.detail.setCurrentIndex(2)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        text = (text or "").strip()
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(text)
        # Editable combos scroll their line-edit to the cursor (the END of the text),
        # so "preference" rendered as "erence" in a narrow cell. Snap back to the start.
        le = combo.lineEdit()
        if le is not None:
            le.setCursorPosition(0)

    def _make_category_combo(self, category: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # editable so a custom category is still possible
        combo.addItems(admin.FACT_CATEGORIES)
        self._set_combo_text(combo, category or "preference")
        return combo

    def _make_key_combo(self, category: str, key: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # editable: keys are free text; the list is just the menu
        combo.addItems(admin.suggested_keys_for_category(category))
        self._set_combo_text(combo, key)
        return combo

    def _repopulate_key_combo(self, key_combo: QComboBox, category: str) -> None:
        """When the category changes, swap the key menu to match — keeping any text the
        user already typed so a half-entered key isn't lost."""
        current = key_combo.currentText().strip()
        key_combo.blockSignals(True)
        key_combo.clear()
        key_combo.addItems(admin.suggested_keys_for_category(category))
        self._set_combo_text(key_combo, current)
        key_combo.blockSignals(False)

    def _append_fact_row(self, *, fact_id, category, key, value, importance) -> None:
        row = self.facts_table.rowCount()
        self.facts_table.insertRow(row)
        # Category + Key are dropdown cell widgets; the fact id rides on the Value item.
        cat_combo = self._make_category_combo(category)
        key_combo = self._make_key_combo(category, key)
        cat_combo.currentTextChanged.connect(
            lambda text, kc=key_combo: self._repopulate_key_combo(kc, text)
        )
        # Any combo edit marks the ROW unsaved (amber) + the page dirty. The row is
        # resolved at FIRE time from the widget (a captured index would go stale the
        # moment a row above it is deleted).
        cat_combo.currentTextChanged.connect(lambda _t, cb=cat_combo: self._note_fact_widget_edited(cb))
        key_combo.currentTextChanged.connect(lambda _t, cb=key_combo: self._note_fact_widget_edited(cb))
        self.facts_table.setCellWidget(row, 0, cat_combo)
        self.facts_table.setCellWidget(row, 1, key_combo)
        value_item = QTableWidgetItem(value)
        value_item.setData(_ROLE_ID, fact_id)  # None for a new, unsaved row
        value_item.setToolTip(value)           # full text survives a narrow column
        self.facts_table.setItem(row, 2, value_item)
        self.facts_table.setItem(row, 3, QTableWidgetItem(f"{importance:.2f}"))

    def _note_fact_widget_edited(self, widget) -> None:
        """Combo-edit hook: resolve the widget's CURRENT row, then mark it edited."""
        if self._loading:
            return
        for row in range(self.facts_table.rowCount()):
            if (self.facts_table.cellWidget(row, 0) is widget
                    or self.facts_table.cellWidget(row, 1) is widget):
                self._note_fact_row_edited(row)
                return

    def _note_fact_row_edited(self, row: int) -> None:
        if self._loading:
            return
        self._tint_fact_row(row)
        self._mark_person_dirty()

    def _on_fact_item_changed(self, item: QTableWidgetItem) -> None:
        # Guard both the load-time population and our own setToolTip/setBackground
        # writes below (item mutations re-emit itemChanged — would recurse).
        if self._loading or getattr(self, "_tinting", False):
            return
        self._tinting = True
        try:
            if item.column() == 2:
                item.setToolTip(item.text())
        finally:
            self._tinting = False
        self._note_fact_row_edited(item.row())

    def _tint_fact_row(self, row: int) -> None:
        """Amber-tint a row's editable cells: the visual 'this row isn't saved yet'."""
        if getattr(self, "_tinting", False):
            return
        self._tinting = True
        try:
            brush = QBrush(QColor(*_UNSAVED_ROW_BRUSH_RGB))
            for col in (2, 3):
                item = self.facts_table.item(row, col)
                if item is not None:
                    item.setBackground(brush)
        finally:
            self._tinting = False

    def _add_fact_row(self) -> None:
        if self._current_person_id is None:
            return
        self._append_fact_row(fact_id=None, category="preference", key="", value="", importance=0.5)
        row = self.facts_table.rowCount() - 1
        self._note_fact_row_edited(row)        # a new row IS unsaved work
        key_combo = self.facts_table.cellWidget(row, 1)
        if isinstance(key_combo, QComboBox):
            key_combo.setFocus()
            key_combo.showPopup()  # open the key menu so it's obvious what to choose

    def _cell(self, row: int, col: int) -> str:
        item = self.facts_table.item(row, col)
        return item.text().strip() if item else ""

    def _combo_text(self, row: int, col: int) -> str:
        widget = self.facts_table.cellWidget(row, col)
        return widget.currentText().strip() if isinstance(widget, QComboBox) else ""

    def _category_text(self, row: int) -> str:
        return self._combo_text(row, 0)

    def _key_text(self, row: int) -> str:
        return self._combo_text(row, 1)

    def _fact_id(self, row: int):
        value_item = self.facts_table.item(row, 2)
        return value_item.data(_ROLE_ID) if value_item else None

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
            fact_id = self._fact_id(row)
            category = self._category_text(row) or "other"
            key = self._key_text(row)
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
        fact_id = self._fact_id(row)
        if fact_id is not None:
            # A persisted fact — deleting is immediate and permanent; say so.
            value = self._cell(row, 2)
            if not self._confirm(
                "Delete this fact?",
                f"“{value or self._key_text(row)}” will be removed immediately "
                "(deletions don't wait for Save).",
            ):
                return
            admin.delete_fact(int(fact_id))
        self.facts_table.removeRow(row)   # an unsaved new row just disappears, no prompt

    def _delete_selected_interest(self) -> None:
        iid = self._selected_id(self.interests_list)
        if iid is None:
            return
        label = self.interests_list.selectedItems()[0].text()
        if not self._confirm(
            "Delete this interest?",
            f"“{label}” will be removed immediately (deletions don't wait for Save).",
        ):
            return
        admin.delete_interest(int(iid))
        if self._current_person_id is not None:
            self._reload_person_preserving_edits()

    def _delete_selected_preference(self) -> None:
        pid = self._selected_id(self.prefs_list)
        if pid is None:
            return
        label = self.prefs_list.selectedItems()[0].text()
        if not self._confirm(
            "Delete this preference?",
            f"“{label}” will be removed immediately (deletions don't wait for Save).",
        ):
            return
        admin.delete_preference(int(pid))
        if self._current_person_id is not None:
            self._reload_person_preserving_edits()

    def _reload_person_preserving_edits(self) -> None:
        """Refresh after an interest/preference delete WITHOUT discarding unsaved fact
        edits: a full _load_person would rebuild the facts table and silently drop
        them (the old behavior — exactly the trap this rework removes). If facts are
        clean, do the full reload; if dirty, refresh only the interests/prefs lists."""
        pid = self._current_person_id
        if pid is None:
            return
        if not self._person_dirty:
            self._load_person(pid)
            return
        detail = admin.get_person_detail(pid)
        if not detail:
            return
        self._loading = True
        try:
            self.interests_list.clear()
            for it in detail["interests"]:
                label = f"{it.get('name','?')}   ·   {it.get('interest_strength') or 'medium'}"
                li = QListWidgetItem(label)
                li.setData(_ROLE_ID, int(it["id"]))
                notes = (it.get("notes") or "").strip()
                if notes:
                    li.setToolTip(notes)
                self.interests_list.addItem(li)
            self.prefs_list.clear()
            for pr in detail["preferences"]:
                val = pr.get("value") or pr.get("key") or ""
                label = f"{pr.get('domain','?')} · {pr.get('preference_type','?')}  —  {val}"
                li = QListWidgetItem(label)
                li.setData(_ROLE_ID, int(pr["id"]))
                li.setToolTip(label)
                self.prefs_list.addItem(li)
        finally:
            self._loading = False

    def _save_person_and_facts(self) -> None:
        """The one Save button: person fields + every edited/new fact row together."""
        if self._current_person_id is None:
            return
        self._save_person()
        self._save_facts()

    def _save_person(self) -> None:
        if self._current_person_id is None:
            return
        admin.update_person_fields(
            self._current_person_id,
            name=self.p_name.text(),
            nickname=self.p_nick.text(),
        )
        # Refresh the people list (name may have changed) without bouncing selection.
        self.people_list.blockSignals(True)
        self.reload_people()
        self._select_person_in_list(self._current_person_id)
        self.people_list.blockSignals(False)
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
        self._clear_dirty()
        self.detail.setCurrentIndex(0)
        self.reload_people()

    def _clear_biometric(self, kind: str) -> None:
        if self._current_person_id is None:
            return
        label = "face data" if kind == "face" else "voiceprint"
        if not self._confirm(
            f"Clear {label}?",
            f"This deletes the stored {label} for this person — Rex will no longer "
            f"recognize them by {'face' if kind == 'face' else 'voice'} until it is "
            "re-learned. Useful if it was enrolled wrong.",
        ):
            return
        admin.clear_biometrics(self._current_person_id, kind)
        self._load_person(self._current_person_id)
        self._toast(f"Cleared {label}.")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _themed_messagebox(self, icon: "QMessageBox.Icon", title: str, body: str) -> QMessageBox:
        """A QMessageBox carrying this window's dark theme. The native
        QMessageBox.question/.warning helpers spawn an unstyled top-level dialog
        that renders white and doesn't match the dashboard, so build it by hand
        and apply _MEMORY_BANKS_STYLE."""
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(body)
        box.setStyleSheet(_MEMORY_BANKS_STYLE)
        return box

    def _confirm(self, title: str, body: str) -> bool:
        box = self._themed_messagebox(QMessageBox.Icon.Question, title, body)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _warn(self, title: str, body: str) -> None:
        box = self._themed_messagebox(QMessageBox.Icon.Warning, title, body)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _toast(self, msg: str) -> None:
        try:
            self.statusBar().showMessage(msg, 2500)
        except Exception:
            pass

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Same guard as switching: closing over unsaved edits asks first.
        if not self._confirm_leave():
            event.ignore()
            return
        # Restore the robot to whatever pause/mute state it was in before we opened.
        config.AUDIO_OUTPUT_SUPPRESSED = self._prior_output_suppressed
        config.INTERACTION_PAUSED = self._prior_interaction_paused
        _log.info("[memory_banks] closed — conversation engine resumed")
        super().closeEvent(event)
