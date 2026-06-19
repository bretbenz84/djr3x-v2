"""
Visual curiosity question: when the conversation lulls with a known person present, Rex
can take ONE look and warmly ask about a specific real detail he sees (their shirt
graphic, an item, their hair) — the way real people show interest. Person-focused,
safety-railed, boundary-aware, cooldowned.
"""

from __future__ import annotations

import unittest
from unittest import mock


class _SyncThread:
    """Runs the target inline so do_*'s off-tick worker is testable."""
    def __init__(self, target=None, **kw):
        self._t = target

    def start(self):
        if self._t:
            self._t()


class DescribePersonDetailTest(unittest.TestCase):
    def _detail(self, raw):
        from vision import scene
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            return scene.describe_person_detail("frame", name="Bret")

    def test_returns_friendly_detail(self):
        self.assertEqual(self._detail("a band t-shirt with white lettering"),
                         "a band t-shirt with white lettering")

    def test_hair_is_allowed(self):
        self.assertEqual(self._detail("blonde hair"), "blonde hair")

    def test_none_becomes_empty(self):
        self.assertEqual(self._detail("NONE"), "")

    def test_banned_category_dropped(self):
        for raw in ("looks overweight", "appears elderly", "darker skin tone", "very attractive"):
            self.assertEqual(self._detail(raw), "", raw)

    def test_none_frame(self):
        from vision import scene
        self.assertEqual(scene.describe_person_detail(None), "")

    def test_api_error_is_safe(self):
        from vision import scene
        with mock.patch.object(scene, "_call_gpt4o", side_effect=RuntimeError):
            self.assertEqual(scene.describe_person_detail("frame"), "")


class VisualCuriosityTargetTest(unittest.TestCase):
    def _target(self, people, engaged_ids=()):
        from intelligence import idle_behaviors as ib
        with mock.patch("intelligence.consciousness.is_engaged_with",
                        side_effect=lambda pid: pid in engaged_ids), \
             mock.patch("intelligence.consciousness._first_name",
                        side_effect=lambda fid, default="there": fid or default):
            return ib._visual_curiosity_target({"people": people})

    def test_prefers_engaged_person(self):
        people = [
            {"person_db_id": 1, "face_id": "Bret", "face_visible": True},
            {"person_db_id": 2, "face_id": "Jeff", "face_visible": True},
        ]
        pid, name = self._target(people, engaged_ids={2})
        self.assertEqual((pid, name), (2, "Jeff"))

    def test_sole_visible_known_person(self):
        pid, name = self._target([{"person_db_id": 1, "face_id": "Bret", "face_visible": True}])
        self.assertEqual((pid, name), (1, "Bret"))

    def test_ambiguous_crowd_no_target(self):
        people = [
            {"person_db_id": 1, "face_id": "Bret", "face_visible": True},
            {"person_db_id": 2, "face_id": "Jeff", "face_visible": True},
        ]
        self.assertEqual(self._target(people), (None, None))  # no one engaged → ambiguous

    def test_unknown_person_no_target(self):
        self.assertEqual(self._target([{"person_db_id": None, "face_visible": True}]), (None, None))


class DoVisualCuriosityQuestionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import idle_behaviors as ib
        self.ib = ib
        ib._last_visual_curiosity_at = 0.0

    def tearDown(self):
        self.ib._last_visual_curiosity_at = 0.0

    def _snapshot(self):
        return {"people": [{"person_db_id": 1, "face_id": "Bret", "face_visible": True}]}

    def _run(self, *, detail="a band tee with white lettering", can_speak=True):
        captured = {}
        ib = self.ib
        with mock.patch("intelligence.consciousness.is_engaged_with", return_value=True), \
             mock.patch("intelligence.consciousness._first_name", return_value="Bret"), \
             mock.patch("intelligence.consciousness._can_proactive_speak", return_value=can_speak), \
             mock.patch("intelligence.consciousness._generate_and_speak",
                        side_effect=lambda d, **k: captured.update(directive=d, kw=k)), \
             mock.patch("vision.camera.get_frame", return_value="frame"), \
             mock.patch("vision.scene.describe_person_detail", return_value=detail), \
             mock.patch("memory.boundaries.is_blocked", return_value=False), \
             mock.patch.object(ib.threading, "Thread", _SyncThread):
            ib.do_visual_curiosity_question(self._snapshot())
        return captured

    def test_asks_about_the_seen_detail(self):
        out = self._run()
        self.assertIn("band tee", out["directive"])
        self.assertIn("Bret", out["directive"])
        self.assertEqual(out["kw"].get("purpose"), "visual_curiosity")

    def test_no_detail_means_silence(self):
        self.assertEqual(self._run(detail=""), {})  # vision found nothing askable

    def test_give_space_gate_blocks(self):
        # can_proactive_speak False (e.g. sober window after a heavy moment) → no question.
        self.assertEqual(self._run(can_speak=False), {})

    def test_disabled_flag_no_op(self):
        import config
        with mock.patch.object(config, "VISUAL_CURIOSITY_ENABLED", False):
            self.assertEqual(self._run(), {})

    def test_cooldown_blocks_second_call(self):
        self.assertTrue(self._run())          # first fires
        self.assertEqual(self._run(), {})     # within cooldown → blocked

    def test_boundary_blocks(self):
        ib = self.ib
        with mock.patch("intelligence.consciousness.is_engaged_with", return_value=True), \
             mock.patch("intelligence.consciousness._first_name", return_value="Bret"), \
             mock.patch("memory.boundaries.is_blocked", return_value=True), \
             mock.patch("intelligence.consciousness._generate_and_speak") as speak, \
             mock.patch.object(ib.threading, "Thread", _SyncThread):
            ib.do_visual_curiosity_question(self._snapshot())
        speak.assert_not_called()


class DispatcherWiringTest(unittest.TestCase):
    def test_offered_when_people_present(self):
        from intelligence import consciousness as c
        snap = {"people": [{"person_db_id": 1, "face_id": "Bret", "face_visible": True}]}
        with mock.patch.object(c, "_room_looks_empty", return_value=False):
            choices, weights = c._idle_micro_behavior_choices(snap)
        self.assertIn("visual_curiosity_question", choices)
        self.assertEqual(len(choices), len(weights))


if __name__ == "__main__":
    unittest.main()
