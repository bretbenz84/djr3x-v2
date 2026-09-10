"""A bystander naming Jeff must not become a new person or Jeff's voice."""
from contextlib import ExitStack
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from memory import database, people
from memory.name_validation import extract_referred_person_name, normalize_person_name
from setup_assets import DB_SCHEMA


class PersonReferenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.path = Path(directory) / 'people.db'
        with sqlite3.connect(self.path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.executemany('INSERT INTO people(id, name, nickname) VALUES (?, ?, ?)', [
                (1, 'Bret Benziger', ''), (5, 'Jeffrey Benziger', 'Jeff')])
        self.stack.enter_context(patch.object(database, '_DB_FILE', self.path))

    def test_clauses_cannot_be_stored_as_names(self):
        for clause in ("That's Jeff Benziger.", "That’s Jeff Benziger.",
                       "He's Jeff.", "She's Jane.", "Their name is Alex.",
                       "It's Bret.", "You're Bret.", "I'm Bret."):
            with self.subTest(clause=clause):
                self.assertIsNone(normalize_person_name(clause))
                self.assertEqual(people.find_or_create_person(clause), (None, False))
        for name in ("Bret Benziger", "Jeffrey Benziger", "Sean O'Neill", "Ann-Marie", "He Li"):
            self.assertEqual(normalize_person_name(name), name)

    def test_extracts_referent_without_calling_it_a_self_introduction(self):
        from intelligence import interaction as I
        for text in ("That's Jeff Benziger.", "That’s Jeff Benziger.",
                     "That is Jeff Benziger.", "He's Jeff Benziger.",
                     "His name is Jeff Benziger.", "No, that's Jeff Benziger."):
            with self.subTest(text=text):
                self.assertEqual(extract_referred_person_name(text), 'Jeff Benziger')
                self.assertIsNone(I._extract_introduced_name(text, allow_bare_name=True))
                self.assertTrue(I._identity_prompt_reply_names_third_party(text))
                self.assertFalse(I._looks_like_direct_offscreen_identity_answer(text, 'Jeff Benziger'))
        for text in ("I'm Jeff Benziger.", "My name is Jeff Benziger.", "Jeff Benziger."):
            self.assertIsNone(extract_referred_person_name(text))
            self.assertEqual(I._extract_introduced_name(text, allow_bare_name=True), 'Jeff Benziger')

    def test_nickname_with_surname_reuses_existing_person(self):
        for name in ('Jeff', 'Jeff Benziger', 'jeff benziger', 'Jeffrey Benziger'):
            with self.subTest(name=name):
                self.assertEqual(people.find_person_by_name(name)['id'], 5)
                self.assertEqual(people.find_or_create_person(name), (5, False))
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM people').fetchone()[0], 2)
            self.assertEqual(conn.execute('SELECT count(*) FROM person_aliases').fetchone()[0], 0)
        self.assertIsNone(people.find_person_by_name('Jeff Davis'))

    def test_shared_nickname_requires_disambiguation(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO people(id,name,nickname) VALUES (6,'Jefferson Davis','Jeff')")
        self.assertIsNone(people.find_person_by_name('Jeff'))
        self.assertEqual(people.find_person_by_name('Jeff Benziger')['id'], 5)
        self.assertEqual(people.find_person_by_name('Jeff Davis')['id'], 6)
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE people SET name='Jeffery Benziger' WHERE id=6")
        self.assertIsNone(people.find_person_by_name('Jeff Benziger'))

    def test_reference_lookup_never_creates_people_from_unknown_names_or_remarks(self):
        from intelligence import interaction as I
        with patch.object(I, '_turn_transcript_trusted', return_value=True), \
             patch.object(people, 'find_or_create_person') as create:
            self.assertEqual(I._resolve_prompted_person_reference("That's Jeff Benziger."),
                             'Got it — Jeff. Thanks for clearing that up.')
            for text in ("That's correct.", "That's Pat Smith.", "That's not his name."):
                self.assertIsNone(I._resolve_prompted_person_reference(text))
            create.assert_not_called()
        with patch.object(I, '_turn_transcript_trusted', return_value=False):
            self.assertIsNone(I._resolve_prompted_person_reference("That's Jeff Benziger."))
