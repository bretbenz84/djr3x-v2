import importlib
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class ConfigLoaderNoAudioValidationTests(unittest.TestCase):
    def _import_config_loader_with(self, fake_apikeys, no_audio: str):
        saved_loader = sys.modules.pop("utils.config_loader", None)
        utils_pkg = importlib.import_module("utils")
        had_attr = hasattr(utils_pkg, "config_loader")
        old_attr = getattr(utils_pkg, "config_loader", None)
        if had_attr:
            delattr(utils_pkg, "config_loader")

        try:
            with (
                mock.patch.dict(sys.modules, {"apikeys": fake_apikeys}),
                mock.patch.dict(os.environ, {"DJR3X_NO_AUDIO_MODE": no_audio}),
            ):
                return importlib.import_module("utils.config_loader")
        finally:
            sys.modules.pop("utils.config_loader", None)
            if saved_loader is not None:
                sys.modules["utils.config_loader"] = saved_loader
            if had_attr:
                setattr(utils_pkg, "config_loader", old_attr)
            elif hasattr(utils_pkg, "config_loader"):
                delattr(utils_pkg, "config_loader")

    def test_noaudio_allows_placeholder_elevenlabs_key(self):
        fake_apikeys = types.SimpleNamespace(
            OPENAI_API_KEY="sk-test-valid",
            ELEVENLABS_API_KEY="your-elevenlabs-api-key-here",
        )

        module = self._import_config_loader_with(fake_apikeys, "1")

        self.assertEqual(module.OPENAI_API_KEY, "sk-test-valid")
        self.assertEqual(module.ELEVENLABS_API_KEY, "your-elevenlabs-api-key-here")

    def test_audio_mode_still_requires_elevenlabs_key(self):
        fake_apikeys = types.SimpleNamespace(
            OPENAI_API_KEY="sk-test-valid",
            ELEVENLABS_API_KEY="your-elevenlabs-api-key-here",
        )

        with self.assertRaisesRegex(RuntimeError, "ELEVENLABS_API_KEY"):
            self._import_config_loader_with(fake_apikeys, "")


class MemoryWipeTransactionTests(unittest.TestCase):
    def _create_partial_memory_db(self, path: Path) -> None:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute(
                "CREATE TABLE biometrics (id INTEGER PRIMARY KEY, person_id INTEGER, type TEXT)"
            )
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Daniel')")
            conn.execute("INSERT INTO biometrics (person_id, type) VALUES (1, 'voice')")

    def _counts(self, path: Path) -> tuple[int, int]:
        with sqlite3.connect(path) as conn:
            people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
            biometrics = conn.execute("SELECT COUNT(*) FROM biometrics").fetchone()[0]
        return people, biometrics

    def test_delete_person_rolls_back_when_any_table_delete_fails(self):
        from memory import people

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "people.db"
            self._create_partial_memory_db(db_path)

            with mock.patch.object(people.db, "_DB_FILE", db_path):
                with self.assertRaises(sqlite3.OperationalError):
                    people.delete_person(1)

            self.assertEqual(self._counts(db_path), (1, 1))

    def test_delete_all_people_rolls_back_when_any_table_delete_fails(self):
        from memory import people

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "people.db"
            self._create_partial_memory_db(db_path)

            with mock.patch.object(people.db, "_DB_FILE", db_path):
                with self.assertRaises(sqlite3.OperationalError):
                    people.delete_all_people()

            self.assertEqual(self._counts(db_path), (1, 1))


class PeopleNameValidationTests(unittest.TestCase):
    def _create_people_db(self, path: Path) -> None:
        from setup_assets import DB_SCHEMA

        with sqlite3.connect(path) as conn:
            conn.executescript(DB_SCHEMA)

    def test_name_validator_rejects_sentence_fragments_but_allows_single_names(self):
        from memory.name_validation import normalize_person_name

        bad_names = [
            "What Chances OF IT ME Gonna Sit There",
            "Straight Down A Little Bit Funny No",
            "Don't Break IN His New Bret Down",
            "Tell People About IT",
            "The Manual Override",
            "I Know",
            "No",
            "Nope",
            "Nah",
        ]
        for candidate in bad_names:
            self.assertIsNone(normalize_person_name(candidate), candidate)

        self.assertEqual(normalize_person_name("Bret"), "Bret")
        self.assertEqual(normalize_person_name("bret benziger"), "Bret Benziger")
        self.assertEqual(normalize_person_name("JT"), "JT")

    def test_people_memory_aliases_first_name_and_blocks_fuzzy_duplicate(self):
        from memory import people

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "people.db"
            self._create_people_db(db_path)

            with mock.patch.object(people.db, "_DB_FILE", db_path):
                person_id = people.enroll_person("Bret Benziger")
                self.assertIsNotNone(person_id)
                self.assertIsNone(people.enroll_person("What Chances OF IT ME Gonna Sit There"))

                first_match = people.find_potential_person_match("Bret")
                self.assertEqual(first_match["match_type"], "first_name")
                self.assertEqual(first_match["person"]["id"], person_id)

                self.assertTrue(people.add_alias(person_id, "Bret"))
                self.assertEqual(people.find_person_by_name("Bret")["id"], person_id)

                fuzzy_match = people.find_potential_person_match("Bret Bensinger")
                self.assertEqual(fuzzy_match["match_type"], "fuzzy")
                self.assertEqual(fuzzy_match["person"]["id"], person_id)

                created_id, created = people.find_or_create_person("Bret Bensinger")
                self.assertIsNone(created_id)
                self.assertFalse(created)

                with sqlite3.connect(db_path) as conn:
                    count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
                self.assertEqual(count, 1)

    def test_rename_preserves_previous_single_name_as_alias(self):
        from memory import people

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "people.db"
            self._create_people_db(db_path)

            with mock.patch.object(people.db, "_DB_FILE", db_path):
                person_id = people.enroll_person("Bret")
                self.assertTrue(people.rename_person(person_id, "Bret Benziger"))

                self.assertEqual(people.find_person_by_name("Bret")["id"], person_id)
                with sqlite3.connect(db_path) as conn:
                    row = conn.execute(
                        "SELECT source FROM person_aliases WHERE alias_norm = 'bret'"
                    ).fetchone()
                self.assertEqual(row[0], "previous_name")


class PersonMemoryRoutingTests(unittest.TestCase):
    def test_router_keeps_known_named_person_topic_as_memory_query(self):
        from intelligence import action_router, person_memory_targets

        decision = action_router.ActionDecision(
            action="memory.query",
            confidence=0.90,
            args={},
            reason="person memory question",
        )

        with mock.patch.object(
            person_memory_targets,
            "_load_known_person_names",
            return_value=["Daniel Benziger"],
        ):
            routed = action_router._apply_context_overrides(
                decision,
                "What do you know about Daniel?",
                {},
            )

        self.assertEqual(routed.action, "memory.query")

    def test_known_person_helper_ignores_sentence_like_stored_names(self):
        from intelligence import person_memory_targets

        with mock.patch.object(
            person_memory_targets,
            "_load_known_person_names",
            return_value=["What Chances Of It", "Daniel Benziger"],
        ):
            self.assertFalse(
                person_memory_targets.references_person_memory_target(
                    "What do you know about Star Trek?"
                )
            )
            self.assertTrue(
                person_memory_targets.references_person_memory_target(
                    "What do you know about Daniel?"
                )
            )

    def test_intent_classifier_allows_known_named_person_memory_topic(self):
        from intelligence import intent_classifier, person_memory_targets

        with (
            mock.patch.object(
                person_memory_targets,
                "_load_known_person_names",
                return_value=["Daniel Benziger"],
            ),
            mock.patch.object(intent_classifier.config, "INTENT_CLASSIFIER_LLM_FALLBACK_ENABLED", True),
            mock.patch.object(intent_classifier, "_classify_with_llm", return_value="query_memory"),
        ):
            intent = intent_classifier.classify("What do you know about Daniel?")

        self.assertEqual(intent, "query_memory")

    def test_general_subject_topic_is_not_a_memory_query(self):
        # "tell me about <subject>" must fall through to a general LLM answer, not query_memory.
        # Regression: the bare "me" in the "tell me" opener made every subject look like a person
        # query, so "Tell me about Star Tours" was answered "I don't have memory for that person yet".
        from intelligence import intent_classifier, person_memory_targets

        with mock.patch.object(
            person_memory_targets, "_load_known_person_names", return_value=["Daniel Benziger"]
        ):
            for text in (
                "Tell me about Star Tours",
                "Can you tell me about Star Tours and Disneyland and how you used to pilot it?",
                "Tell me a joke",
                "Explain how hyperdrives work",
            ):
                self.assertFalse(
                    intent_classifier._memory_query_allowed(text),
                    f"{text!r} should not route to query_memory",
                )
                self.assertEqual(intent_classifier._deterministic_label(text), "general", text)


class AudioStreamBufferLockTests(unittest.TestCase):
    def test_callback_snapshot_and_flush_use_buffer_lock(self):
        from audio import stream

        class TrackingLock:
            def __init__(self):
                self.locked = False

            def __enter__(self):
                self.locked = True

            def __exit__(self, exc_type, exc, tb):
                self.locked = False

        class GuardedBuffer:
            def __init__(self, lock):
                self.lock = lock
                self.items = []

            def append(self, item):
                if not self.lock.locked:
                    raise AssertionError("append happened without buffer lock")
                self.items.append(item)

            def __iter__(self):
                if not self.lock.locked:
                    raise AssertionError("snapshot happened without buffer lock")
                return iter(self.items)

            def clear(self):
                if not self.lock.locked:
                    raise AssertionError("clear happened without buffer lock")
                self.items.clear()

        lock = TrackingLock()
        guarded = GuardedBuffer(lock)

        with (
            mock.patch.object(stream, "_buf_lock", lock),
            mock.patch.object(stream, "_buf", guarded),
            mock.patch.object(stream, "_input_channels", 1),
        ):
            stream._callback(np.ones((2, 1), dtype=np.float32), 2, None, None)
            self.assertEqual(stream.get_full_buffer().tolist(), [1.0, 1.0])
            stream.flush()

        self.assertEqual(guarded.items, [])


if __name__ == "__main__":
    unittest.main()
