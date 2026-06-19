"""
Capture-quality fixes for Rex's episodic memory: scenes are attributed to recognized
people (face match), generic unattended room scans are gated out, "I saw X" is logged
once per session not per tick, and a made-laugh moment carries its topic. Higher-quality
captures are the precondition for topic-relevant recall.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import episodes, rex_db


# ── Pure helpers (no DB) ─────────────────────────────────────────────────────────

class VisibleKnownPeopleTest(unittest.TestCase):
    def test_resolves_ids_and_names_from_snapshot(self):
        from vision import face
        snapshot = {"people": [
            {"person_db_id": 1, "face_visible": True},
            {"person_db_id": None},                       # unknown → skipped
            {"person_db_id": 2, "face_missing": True},    # not visible → skipped
            {"person_db_id": 1, "face_visible": True},    # dup id → skipped
        ]}
        with mock.patch.object(
            face.people, "get_person",
            side_effect=lambda pid: {"name": {1: "Bret", 2: "Jeff"}.get(pid)},
        ):
            self.assertEqual(face.visible_known_people(snapshot), [(1, "Bret")])
            # names view derives from the same resolver
            self.assertEqual(face.visible_known_names(snapshot), ["Bret"])


class SceneNotabilityHelpersTest(unittest.TestCase):
    def test_caption_differs_when_no_prior(self):
        from intelligence import episodic_hooks as eh
        self.assertTrue(eh._caption_materially_differs("", "a cluttered workshop"))

    def test_near_identical_captions_are_not_notable(self):
        from intelligence import episodic_hooks as eh
        a = "a tidy room with white walls and soft lighting"
        b = "a tidy room with white walls and a soft light"
        self.assertFalse(eh._caption_materially_differs(a, b))

    def test_clearly_different_captions_are_notable(self):
        from intelligence import episodic_hooks as eh
        a = "a tidy office with white walls and a desk"
        b = "a crowded outdoor patio at night with string lights"
        self.assertTrue(eh._caption_materially_differs(a, b))

    def test_sole_known_person_only_when_unambiguous(self):
        from intelligence import episodic_hooks as eh
        with mock.patch.object(eh, "_visible_known_people", return_value=[(1, "Bret")]):
            self.assertEqual(eh._sole_known_person(), (1, "Bret"))
        with mock.patch.object(eh, "_visible_known_people", return_value=[(1, "Bret"), (2, "Jeff")]):
            self.assertEqual(eh._sole_known_person(), (None, None))
        with mock.patch.object(eh, "_visible_known_people", return_value=[]):
            self.assertEqual(eh._sole_known_person(), (None, None))


class PersonSeenDedupTest(unittest.TestCase):
    def setUp(self):
        from intelligence import episodic_hooks as eh
        eh._person_seen_this_session.clear()

    def test_logs_once_per_session_per_person(self):
        from intelligence import episodic_hooks as eh
        with mock.patch.object(episodes, "record_person_seen") as rec, \
             mock.patch.object(episodes, "_session", return_value="run-x"):
            eh.person_seen(1, "Bret")
            eh.person_seen(1, "Bret")
            eh.person_seen(1, "Bret")
            eh.person_seen(2, "Jeff")
        self.assertEqual(rec.call_count, 2)  # Bret once + Jeff once, not 4


# ── Round-trip through a temp rex.db (opts in to real I/O) ────────────────────────

class CaptureAttributionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(Path(self._tmp.name) / "rex.db"))
        self._patch.start()
        episodes.reset_session("run-test")

    def tearDown(self):
        self._patch.stop()
        episodes.reset_session(None)
        self._tmp.cleanup()

    def test_scene_attributed_to_person_when_present(self):
        episodes.record_scene("I looked around: Bret at his desk", person_id=1, person_name="Bret")
        row = episodes.recent_episodes(1)[0]
        self.assertEqual(row["person_id"], 1)
        self.assertEqual(row["person_name"], "Bret")
        self.assertGreater(row["salience"], 0.4)  # attributed scenes rank a touch higher

    def test_anonymous_scene_unattributed(self):
        episodes.record_scene("When I powered up, I saw: a tidy room")
        row = episodes.recent_episodes(1)[0]
        self.assertIsNone(row["person_id"])
        self.assertAlmostEqual(row["salience"], 0.4, places=3)

    def test_made_laugh_carries_topic(self):
        episodes.record_made_laugh(1, "Bret", kind="laugh", topic="his fantasy team")
        row = episodes.recent_episodes(1)[0]
        self.assertEqual(row["summary"], "I made Bret laugh about his fantasy team.")

    def test_made_laugh_ignores_placeholder_topic(self):
        episodes.record_made_laugh(1, "Bret", kind="smile", topic="current exchange")
        row = episodes.recent_episodes(1)[0]
        self.assertEqual(row["summary"], "I made Bret smile.")


if __name__ == "__main__":
    unittest.main()
