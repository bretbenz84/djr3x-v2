"""
Tests for Rex's episodic memory (memory/episodes.py + memory/rex_db.py).

The CRITICAL property: the test suite must never create or populate a real rex.db.
Writes to the DEFAULT path are gated under the test runner; a test that points
REX_DB_PATH at a temp file opts back IN to real I/O (so the writer can be exercised).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import episodes, rex_db


class EpisodicGateTest(unittest.TestCase):
    """Under `python -m unittest`, writes to the DEFAULT rex.db must be suppressed —
    the suite never touches the real file."""

    def test_default_path_is_suppressed_under_test_runner(self):
        self.assertTrue(rex_db._under_test_runner())
        self.assertTrue(rex_db.writes_suppressed())  # default path + under runner

    def test_robot_entrypoint_is_not_suppressed(self):
        # The gate must NOT over-suppress: on `python main.py` (no unittest/pytest in
        # argv, no pytest env), writes must actually happen on the default path.
        import os
        with (
            mock.patch.object(rex_db.sys, "argv", ["main.py"]),
            mock.patch.dict(os.environ, {}, clear=False),  # restored after the block
        ):
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            os.environ.pop("DJR3X_EPISODIC_TEST_OPT_IN", None)
            self.assertFalse(rex_db._under_test_runner())
            self.assertFalse(rex_db.writes_suppressed())

    def test_record_on_default_path_is_a_noop_and_creates_no_file(self):
        default = rex_db._default_db_path()
        existed_before = default.exists()
        self.assertIsNone(episodes.record_person_seen(1, "Bret"))
        self.assertIsNone(episodes.record_animal("dog"))
        self.assertIsNone(episodes.record_conversation_summary("anything"))
        # The real rex.db must not have been created by these calls.
        self.assertEqual(default.exists(), existed_before)

    def test_read_on_default_path_creates_no_file_and_returns_empty(self):
        # READS must be gated too — connection() mkdir+connect would create the real
        # file before any per-query check. (Regression guard for the review blocker.)
        default = rex_db._default_db_path()
        existed_before = default.exists()
        self.assertEqual(episodes.recent_episodes(10), [])
        self.assertEqual(episodes.count(), 0)
        self.assertEqual(episodes.episodes_on_date("2026-06-05"), [])
        self.assertEqual(rex_db.fetchall("SELECT 1"), [])
        self.assertEqual(default.exists(), existed_before)

    def test_kill_switch_suppresses_even_with_temp_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rex.db"
            with (
                mock.patch.object(config, "REX_DB_PATH", str(p)),
                mock.patch.object(config, "EPISODIC_MEMORY_ENABLED", False),
            ):
                self.assertIsNone(episodes.record_person_seen(1, "Bret"))
                self.assertFalse(p.exists())


class EpisodicWriteReadTest(unittest.TestCase):
    """With REX_DB_PATH pointed at a temp file, the writer/reader round-trips."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "rex.db"
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(self._path))
        self._patch.start()
        episodes.reset_session("run-test")  # path-aware schema cache re-ensures per temp DB

    def tearDown(self):
        self._patch.stop()
        episodes.reset_session(None)
        self._tmp.cleanup()

    def test_temp_path_opts_in_to_real_io(self):
        self.assertFalse(rex_db.writes_suppressed())  # non-default path

    def test_round_trip_all_kinds(self):
        self.assertIsNotNone(episodes.record_person_seen(1, "Bret"))
        self.assertIsNotNone(episodes.record_made_laugh(1, "Bret", kind="laugh"))
        self.assertIsNotNone(episodes.record_animal("dog", position="lower right"))
        self.assertIsNotNone(episodes.record_scene("I looked around: cluttered workshop"))
        self.assertIsNotNone(episodes.record_conversation_summary(
            "Bret and I talked about his robot DJ.", people=[{"person_id": 1, "name": "Bret"}]))

        self.assertEqual(episodes.count(), 5)
        rows = episodes.recent_episodes(10)
        kinds = {r["kind"] for r in rows}
        self.assertEqual(kinds, {"person_seen", "made_laugh", "animal", "scene", "conversation_summary"})
        # Each has a timestamp, a first-person summary, and a session id.
        for r in rows:
            self.assertTrue(r["created_at"])
            self.assertTrue(r["summary"].strip())
            self.assertEqual(r["session_id"], "run-test")

    def test_summary_phrasing_and_fields(self):
        episodes.record_person_seen(7, "Jeff")
        episodes.record_made_laugh(7, "Jeff", kind="smile")
        episodes.record_animal("owl")  # vowel → "an owl"
        rows = {r["kind"]: r for r in episodes.recent_episodes(10)}
        self.assertEqual(rows["person_seen"]["summary"], "I saw Jeff.")
        self.assertEqual(rows["person_seen"]["person_id"], 7)
        self.assertEqual(rows["made_laugh"]["summary"], "I made Jeff smile.")
        self.assertEqual(rows["made_laugh"]["person_id"], 7)  # the hook passes _person_db_id(person)
        self.assertEqual(rows["animal"]["summary"], "I saw an owl.")

    def test_detail_is_stored_as_json(self):
        import json
        episodes.record_animal("cat", position="by the door")
        row = episodes.recent_episodes(1)[0]
        self.assertEqual(json.loads(row["detail"]) ["species"], "cat")

    def test_episodes_on_date_filters_by_day(self):
        from datetime import date
        episodes.record_scene("a scene today")
        today = date.today().strftime("%Y-%m-%d")
        self.assertEqual(len(episodes.episodes_on_date(today)), 1)
        self.assertEqual(len(episodes.episodes_on_date("1999-01-01")), 0)

    def test_blank_summary_is_dropped(self):
        self.assertIsNone(episodes.record_episode("scene", "   "))
        self.assertEqual(episodes.count(), 0)


class StartupImageCaptionTest(unittest.TestCase):
    """One cheap GPT caption of Rex's first look at the room, once per run → rex.db.
    The GPT call is gated like every episodic write (never fires under the suite)."""

    def setUp(self):
        from intelligence import consciousness as c
        c._startup_image_captured = False

    def tearDown(self):
        from intelligence import consciousness as c
        c._startup_image_captured = False

    def test_quick_caption_is_empty_without_a_frame_and_makes_no_call(self):
        from vision import scene
        with (
            mock.patch("vision.camera.get_frame", return_value=None),
            mock.patch.object(scene, "_call_gpt4o") as gpt,
        ):
            self.assertEqual(scene.quick_caption(), "")
            gpt.assert_not_called()

    def test_quick_caption_returns_the_gpt_text_for_a_frame(self):
        from vision import scene
        with mock.patch.object(scene, "_call_gpt4o", return_value="A cluttered workshop, one person.") as gpt:
            self.assertEqual(scene.quick_caption(object()), "A cluttered workshop, one person.")
            gpt.assert_called_once()

    def test_startup_hook_is_gated_under_test_runner_no_gpt_call(self):
        from intelligence import consciousness as c
        from vision import scene
        with mock.patch.object(scene, "quick_caption") as qc:
            c._capture_startup_image_episode(object())  # non-None frame
            qc.assert_not_called()                      # episodes._suppressed() → no GPT call
            self.assertTrue(c._startup_image_captured)  # but it's a one-shot, so it latches

    def test_startup_hook_waits_for_a_real_frame(self):
        from intelligence import consciousness as c
        c._capture_startup_image_episode(None)
        self.assertFalse(c._startup_image_captured)     # no frame yet → don't latch

    def test_startup_hook_is_one_shot(self):
        from intelligence import consciousness as c
        from vision import scene
        c._startup_image_captured = True
        with mock.patch.object(scene, "quick_caption") as qc:
            c._capture_startup_image_episode(object())
            qc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
