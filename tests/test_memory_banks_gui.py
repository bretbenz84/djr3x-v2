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
        self._prior_pause = getattr(config, "INTERACTION_PAUSED", False)

    def tearDown(self):
        config.AUDIO_OUTPUT_SUPPRESSED = self._prior
        config.INTERACTION_PAUSED = self._prior_pause
        self._p_rex.stop()
        self._p_people.stop()
        self._tmp.cleanup()

    def test_window_builds_pauses_engine_and_lists_both_dbs(self):
        config.AUDIO_OUTPUT_SUPPRESSED = False
        config.INTERACTION_PAUSED = False
        w = MemoryBanksWindow()
        try:
            self.assertTrue(config.INTERACTION_PAUSED)       # conversation engine paused
            self.assertTrue(config.AUDIO_OUTPUT_SUPPRESSED)  # audio muted too
            self.assertEqual(w.rex_list.count(), 1)          # Rex memory listed
            self.assertEqual(w.people_list.count(), 1)       # person listed
            w._select_person_in_list(1)
            self.assertEqual(w.p_name.text(), "Bret Benziger")
            self.assertEqual(w.detail.currentIndex(), 2)     # person editor shown
        finally:
            w.close()
        self.assertFalse(config.INTERACTION_PAUSED)          # resumed on close
        self.assertFalse(config.AUDIO_OUTPUT_SUPPRESSED)

    def test_close_restores_prior_paused_state(self):
        config.INTERACTION_PAUSED = True  # already paused before opening
        config.AUDIO_OUTPUT_SUPPRESSED = True
        w = MemoryBanksWindow()
        self.assertTrue(config.INTERACTION_PAUSED)
        w.close()
        self.assertTrue(config.INTERACTION_PAUSED)           # left as it was, not forced off
        self.assertTrue(config.AUDIO_OUTPUT_SUPPRESSED)

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

    def test_category_and_key_are_dropdowns_and_drive_saved_values(self):
        from PySide6.QtWidgets import QComboBox
        from memory import admin
        w = MemoryBanksWindow()
        try:
            w._select_person_in_list(1)
            w._add_fact_row()  # new blank row, category defaults to "preference"
            row = w.facts_table.rowCount() - 1
            cat = w.facts_table.cellWidget(row, 0)
            key = w.facts_table.cellWidget(row, 1)
            self.assertIsInstance(cat, QComboBox)
            self.assertIsInstance(key, QComboBox)
            # Category dropdown is the recognized categories.
            self.assertEqual([cat.itemText(i) for i in range(cat.count())], admin.FACT_CATEGORIES)
            self.assertEqual(cat.currentText(), "preference")
            # Pick category=family, then fill key/value, then save.
            cat.setCurrentText("family")
            key.setCurrentText("nephew")
            w.facts_table.item(row, 2).setText("Wade")
            w._save_facts()
        finally:
            w.close()
        facts = {f["key"]: f["category"] for f in admin.get_person_detail(1)["facts"]}
        self.assertEqual(facts.get("nephew"), "family")

    def test_biometrics_status_is_shown(self):
        from memory import database as db
        # Give person 1 a face encoding (no voiceprint).
        db.execute(
            "INSERT INTO biometrics (person_id, type, encoding, created_at) "
            "VALUES (?, ?, ?, ?)",
            (1, "face", b"\x00\x01", "2026-06-14 10:00:00"),
        )
        w = MemoryBanksWindow()
        try:
            w._select_person_in_list(1)
            text = w.bio_label.text()
            self.assertIn("Face ID", text)
            self.assertIn("Voiceprint", text)
            self.assertIn("1 stored", text)              # the face encoding
            self.assertTrue(w.clear_face_btn.isEnabled())   # has face → enabled
            self.assertFalse(w.clear_voice_btn.isEnabled()) # no voiceprint → disabled
        finally:
            w.close()

    def test_key_menu_follows_the_category(self):
        # The user's report: choosing "relationship" left the key a blank box. The key
        # menu must now offer the relationship keys (boss, coworker, …).
        from PySide6.QtWidgets import QComboBox
        from memory import admin
        w = MemoryBanksWindow()
        try:
            w._select_person_in_list(1)
            w._add_fact_row()
            row = w.facts_table.rowCount() - 1
            cat = w.facts_table.cellWidget(row, 0)
            key = w.facts_table.cellWidget(row, 1)
            cat.setCurrentText("relationship")  # triggers the key menu to repopulate
            key_options = [key.itemText(i) for i in range(key.count())]
            self.assertEqual(key_options, admin.suggested_keys_for_category("relationship"))
            self.assertIn("boss", key_options)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
