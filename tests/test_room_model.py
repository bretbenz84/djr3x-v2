"""Room model — persistent object permanence (rex.db room_objects) + its two payoffs.

room_model records which objects Rex has seen over time (one row per label, sighting_count
+ first/last_seen). It powers a novelty-aware visual-curiosity prompt and a conservatively-
gated "wait, that's new" change-detection reaction. Writes are test-suppressed on the default
path exactly like episodes; pointing REX_DB_PATH at a temp file opts into real I/O.
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import config


class _Profile:
    def __init__(self, suppress_proactive=False, user_mid_sentence=False, interaction_busy=False):
        self.suppress_proactive = suppress_proactive
        self.user_mid_sentence = user_mid_sentence
        self.interaction_busy = interaction_busy


class RoomModelSuppressionTest(unittest.TestCase):
    def test_record_is_a_noop_and_creates_no_file_on_default_path(self):
        from memory import room_model, rex_db

        default = rex_db._default_db_path()
        existed = default.exists()
        room_model.record_objects([{"label": "chair", "position": "center"}])
        self.assertEqual(default.exists(), existed)  # no real rex.db created
        self.assertEqual(room_model.label_sightings({"chair"}), {"chair": 0})
        self.assertEqual(room_model.established_count(1), 0)

    def test_disabled_flag_suppresses_writes_even_on_temp_path(self):
        from memory import room_model

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rex.db"
            with mock.patch.object(config, "REX_DB_PATH", str(p)), \
                 mock.patch.object(config, "ROOM_MODEL_ENABLED", False):
                room_model.record_objects([{"label": "chair", "position": "center"}])
                self.assertFalse(p.exists())


class RoomModelDbTest(unittest.TestCase):
    """REX_DB_PATH → temp file opts into real I/O; the round-trip is exercised."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(Path(self._tmp.name) / "rex.db"))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_record_upserts_and_label_sightings_counts(self):
        from memory import room_model

        room_model.record_objects([{"label": "Chair", "position": "center"}, {"label": "cup", "position": "left"}])
        room_model.record_objects([{"label": "chair", "position": "right"}])  # bump chair, move it
        counts = room_model.label_sightings({"chair", "cup", "lamp"})
        self.assertEqual(counts["chair"], 2)   # label lowercased + upserted across positions
        self.assertEqual(counts["cup"], 1)
        self.assertEqual(counts["lamp"], 0)    # never seen

    def test_same_label_twice_in_one_call_counts_once(self):
        from memory import room_model

        room_model.record_objects([{"label": "book", "position": "a"}, {"label": "book", "position": "b"}])
        self.assertEqual(room_model.label_sightings({"book"})["book"], 1)

    def test_established_count_is_the_fixture_baseline(self):
        from memory import room_model

        for _ in range(5):
            room_model.record_objects([{"label": "desk", "position": "center"}])
        room_model.record_objects([{"label": "plant", "position": "left"}])
        self.assertEqual(room_model.established_count(5), 1)  # only desk (5) clears the bar
        self.assertEqual(room_model.established_count(1), 2)  # desk + plant


class StepRoomChangeTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness as c

        self.c = c
        self._reset()
        self.addCleanup(self._reset)

    def _reset(self):
        self.c._room_change_state["count"] = 0.0
        self.c._room_change_state["last_at"] = 0.0
        self.c._room_change_remarked.clear()

    def _run(self, objects, *, established=10, sightings=None, can_speak=True, profile=None):
        c = self.c
        captured = {}
        with mock.patch("memory.room_model.established_count", return_value=established), \
             mock.patch("memory.room_model.label_sightings", return_value=dict(sightings or {})), \
             mock.patch.object(c, "_can_proactive_speak", return_value=can_speak), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: captured.update(line=line, kw=k) or True):
            c._step_room_change({"objects": objects}, profile or _Profile())
        return captured

    def test_fires_on_a_genuinely_new_object(self):
        out = self._run([{"label": "guitar", "position": "left"}], established=10, sightings={"guitar": 3})
        self.assertIn("guitar", out.get("line", ""))
        self.assertEqual(out["kw"].get("purpose"), "room_change")
        self.assertEqual(self.c._room_change_state["count"], 1)
        self.assertIn("guitar", self.c._room_change_remarked)

    def test_no_baseline_means_no_remark(self):
        # established (1) < ROOM_CHANGE_MIN_BASELINE (4): Rex doesn't know the room yet.
        self.assertNotIn("line", self._run([{"label": "guitar"}], established=1, sightings={"guitar": 3}))

    def test_a_fixture_is_not_new(self):
        self.assertNotIn("line", self._run([{"label": "chair"}], established=10, sightings={"chair": 999}))

    def test_a_one_frame_misread_is_not_new(self):
        self.assertNotIn("line", self._run([{"label": "blip"}], established=10, sightings={"blip": 1}))

    def test_already_remarked_label_is_skipped(self):
        self.c._room_change_remarked.add("guitar")
        self.assertNotIn("line", self._run([{"label": "guitar"}], established=10, sightings={"guitar": 3}))

    def test_session_cap(self):
        self.c._room_change_state["count"] = float(config.ROOM_CHANGE_SESSION_CAP)
        self.assertNotIn("line", self._run([{"label": "guitar"}], established=10, sightings={"guitar": 3}))

    def test_cooldown(self):
        self.c._room_change_state["last_at"] = time.monotonic()
        self.assertNotIn("line", self._run([{"label": "guitar"}], established=10, sightings={"guitar": 3}))

    def test_disabled_flag(self):
        with mock.patch.object(config, "ROOM_CHANGE_REMARK_ENABLED", False):
            self.assertNotIn("line", self._run([{"label": "guitar"}], established=10, sightings={"guitar": 3}))

    def test_blocked_speech_does_not_consume_the_cap(self):
        out = self._run([{"label": "guitar"}], established=10, sightings={"guitar": 3}, can_speak=False)
        self.assertNotIn("line", out)
        self.assertEqual(self.c._room_change_state["count"], 0.0)

    def test_label_deduped_even_when_the_enqueue_races_and_fails(self):
        # _can_proactive_speak passed but the enqueue returned False: the label must still
        # be de-duped (so a flickering false positive can't re-fire) WITHOUT burning the cap.
        c = self.c
        with mock.patch("memory.room_model.established_count", return_value=10), \
             mock.patch("memory.room_model.label_sightings", return_value={"guitar": 3}), \
             mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_speak_async", return_value=False):
            c._step_room_change({"objects": [{"label": "guitar"}]}, _Profile())
        self.assertIn("guitar", c._room_change_remarked)
        self.assertEqual(c._room_change_state["count"], 0.0)


class VisualCuriosityNoveltyTest(unittest.TestCase):
    def _line(self, objects, sightings):
        from intelligence import consciousness as c

        with mock.patch.object(c.world_state, "get", return_value=objects), \
             mock.patch("memory.room_model.label_sightings", return_value=sightings):
            return c._visual_curiosity_objects_line()

    def test_novel_object_is_floated_to_front_and_flagged(self):
        objs = [
            {"label": "chair", "position": "center", "confidence": 0.9},
            {"label": "guitar", "position": "left", "confidence": 0.5},
        ]
        line = self._line(objs, {"chair": 500, "guitar": 1})
        self.assertIn("guitar is NEW", line)
        self.assertLess(line.index("guitar"), line.index("chair"))  # floated despite lower confidence

    def test_no_note_when_everything_is_a_fixture(self):
        line = self._line([{"label": "chair", "confidence": 0.9}], {"chair": 500})
        self.assertNotIn("NEW to the room", line)
        self.assertIn("chair", line)


if __name__ == "__main__":
    unittest.main()


class RoomChangeAddresseeTest(unittest.TestCase):
    """Person present -> ask about the new object (owner feedback 2026-07-06: the
    sandwich deserved 'what kind?' not 'A wild sandwich appears'); alone -> the
    canned observational line."""

    def _snapshot(self, people):
        return {"people": people}

    def test_known_visible_person_gets_first_name(self):
        from unittest import mock
        from intelligence import consciousness as c
        snap = self._snapshot([{"face_visible": True, "person_db_id": 1}])
        with mock.patch("memory.people.get_person",
                        return_value={"name": "Bret Benziger"}):
            self.assertEqual(c._room_change_addressee(snap), "Bret")

    def test_unknown_visible_person_still_asked(self):
        from intelligence import consciousness as c
        snap = self._snapshot([{"face_visible": True, "person_db_id": None}])
        self.assertEqual(c._room_change_addressee(snap), "them")

    def test_empty_room_returns_none(self):
        from intelligence import consciousness as c
        self.assertIsNone(c._room_change_addressee(self._snapshot([])))
        # pose-only phantom without a visible face doesn't count as an addressee
        self.assertIsNone(c._room_change_addressee(
            self._snapshot([{"face_visible": False, "pose": "x"}])))
