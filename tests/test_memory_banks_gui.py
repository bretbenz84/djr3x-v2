"""
Smoke test for the Memory Banks editor window — it constructs, loads both databases,
pauses robot audio output while open, and restores it on close. Runs headless via the
Qt 'offscreen' platform; skips cleanly if PySide6 / a Qt platform isn't available.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Force headless BEFORE any QApplication is created so the full test sweep never tries
# to open a real window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import config
from memory import database as db
from memory import rex_db

try:
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from gui.memory_banks import MemoryBanksWindow
    _GUI_OK = True
except Exception:  # pragma: no cover - environment without a usable Qt platform
    _GUI_OK = False


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class MemoryBanksWindowSmokeTest(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        people_path = base / "people.db"
        with sqlite3.connect(people_path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")
        self._p_people = mock.patch.object(db, "_DB_FILE", people_path)
        self._p_rex = mock.patch.object(config, "REX_DB_PATH", str(base / "rex.db"))
        self._p_people.start()
        self._p_rex.start()
        rex_db.ensure_schema()
        rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, salience, session_id) "
            "VALUES (?,?,?,?,?)",
            ("2026-06-14 10:00:00", "animal", "I saw a dog.", 0.5, "r1"),
        )
        self._prior = getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)

    def tearDown(self):
        config.AUDIO_OUTPUT_SUPPRESSED = self._prior
        self._p_rex.stop()
        self._p_people.stop()
        self._tmp.cleanup()

    def test_window_builds_pauses_output_and_lists_both_dbs(self):
        config.AUDIO_OUTPUT_SUPPRESSED = False
        w = MemoryBanksWindow()
        try:
            self.assertTrue(config.AUDIO_OUTPUT_SUPPRESSED)  # paused while open
            self.assertEqual(w.rex_list.count(), 1)          # Rex memory listed
            self.assertEqual(w.people_list.count(), 1)       # person listed
            w._select_person_in_list(1)
            self.assertEqual(w.p_name.text(), "Bret Benziger")
            self.assertEqual(w.detail.currentIndex(), 2)     # person editor shown
        finally:
            w.close()
        self.assertFalse(config.AUDIO_OUTPUT_SUPPRESSED)      # restored on close

    def test_close_restores_prior_suppressed_state(self):
        config.AUDIO_OUTPUT_SUPPRESSED = True  # already suppressed before opening
        w = MemoryBanksWindow()
        self.assertTrue(config.AUDIO_OUTPUT_SUPPRESSED)
        w.close()
        self.assertTrue(config.AUDIO_OUTPUT_SUPPRESSED)       # left as it was, not forced off

    def test_save_facts_commits_an_open_cell_edit(self):
        # The reported bug: editing a fact cell and clicking Save Facts without first
        # clicking elsewhere lost the edit (the cell editor hadn't committed).
        from PySide6.QtWidgets import QAbstractItemView, QLineEdit
        from memory import admin
        admin.add_person_fact(1, "preference", "favorite_color", "teal")
        w = MemoryBanksWindow()
        try:
            w._select_person_in_list(1)
            # Find the Value cell for the fact and open its editor (as a double-click would).
            self.assertEqual(w.facts_table.rowCount(), 1)
            index = w.facts_table.model().index(0, 2)  # column 2 = Value
            w.facts_table.edit(index)
            self.assertEqual(
                w.facts_table.state(), QAbstractItemView.State.EditingState
            )
            editor = w.facts_table.viewport().findChild(QLineEdit)
            editor.setText("cyan")
            # Save WITHOUT clicking out first — the fix must commit the editor.
            w._save_facts()
        finally:
            w.close()
        facts = admin.get_person_detail(1)["facts"]
        self.assertEqual([f["value"] for f in facts], ["cyan"])


if __name__ == "__main__":
    unittest.main()
