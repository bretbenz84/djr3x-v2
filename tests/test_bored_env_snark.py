"""
Tests for the bored environmental snark idle behavior (intelligence/consciousness.py):
when Rex is idle and bored he looks around and invents snark about the room — a dull-room
complaint, a faux-clueless question about an object, a clutter jab, a snobby art opinion,
or a plea to be taken somewhere livelier — grounded in what he actually sees.
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness as c
from intelligence import idle_behaviors as ib


class _SyncThread:
    """Runs the target synchronously on start() so we can test the off-tick _task."""
    def __init__(self, target=None, daemon=None, name=None, args=()):
        self._target = target
        self._args = args

    def start(self):
        if self._target:
            self._target(*self._args)


class ModePickTest(unittest.TestCase):
    def test_no_objects_limits_to_object_free_modes(self):
        for _ in range(50):
            self.assertIn(ib._pick_bored_env_snark_mode([]), ("complaint", "relocate"))

    def test_objects_unlock_object_modes(self):
        seen = {ib._pick_bored_env_snark_mode(["a black chair", "empty boxes"]) for _ in range(200)}
        # All object-dependent modes should be reachable when there are objects.
        self.assertTrue({"naive_question", "clutter", "art_opinion"} <= seen)

    def test_relocate_dropped_when_a_person_is_present(self):
        # "wheel me somewhere with actual life forms" is tone-deaf when someone's here.
        seen = {ib._pick_bored_env_snark_mode(["a chair"], present_name="Bret") for _ in range(200)}
        self.assertNotIn("relocate", seen)
        seen_alone = {ib._pick_bored_env_snark_mode(["a chair"]) for _ in range(200)}
        self.assertIn("relocate", seen_alone)

    def test_present_name_read_from_snapshot(self):
        present = {"people": [{"person_db_id": 1, "name": "Bret Benziger", "face_visible": True}]}
        self.assertEqual(ib._bored_snark_present_name(present), "Bret")
        self.assertIsNone(ib._bored_snark_present_name({"people": []}))
        # An unknown (no person_db_id) face doesn't count as a known present person.
        self.assertIsNone(ib._bored_snark_present_name({"people": [{"person_db_id": None}]}))


class PromptTest(unittest.TestCase):
    def test_prompt_includes_scene_and_objects_and_is_one_line(self):
        p = ib._bored_env_snark_prompt("naive_question", "a dim cluttered office", ["a black chair", "empty boxes"])
        self.assertIn("a dim cluttered office", p)
        self.assertIn("black chair", p)
        self.assertIn("One line only", p)
        self.assertIn("never invent an object", p)

    def test_each_mode_has_distinct_instruction(self):
        s = "a room"
        objs = ["art on the wall", "boxes"]
        prompts = {m: ib._bored_env_snark_prompt(m, s, objs)
                   for m in ("complaint", "naive_question", "clutter", "art_opinion", "relocate")}
        # Mode-specific cues present.
        self.assertIn("don't know what it is", prompts["naive_question"])
        self.assertIn("tidy", prompts["clutter"].lower())
        self.assertIn("art", prompts["art_opinion"].lower())
        self.assertIn("life forms", prompts["relocate"].lower())
        self.assertEqual(len(set(prompts.values())), 5)  # all distinct

    def test_present_prompt_forbids_empty_room_framing(self):
        p = ib._bored_env_snark_prompt("complaint", "a tidy room", ["a chair"], present_name="Bret")
        self.assertIn("Bret", p)
        self.assertIn("do NOT claim the room is empty", p)
        # The phrase appears only inside the FORBID instruction, never as an ask.
        self.assertIn("never ask to be taken somewhere with 'life forms'", p)


class FireTest(unittest.TestCase):
    def setUp(self):
        ib._last_bored_env_snark_at = 0.0

    def tearDown(self):
        ib._last_bored_env_snark_at = 0.0

    def _fire(self, *, now=1000.0, details=None, can_speak=True, locked=False):
        if details is None:
            details = {"overall_summary": "a dim cluttered home office",
                       "notable_details": ["a black chair", "empty cardboard boxes", "abstract art"]}
        gen = mock.MagicMock()
        with (
            mock.patch.object(c.time, "monotonic", return_value=now),
            mock.patch.object(c.threading, "Thread", _SyncThread),
            mock.patch.object(c, "_can_proactive_speak", return_value=can_speak),
            mock.patch.object(c, "_face_tracking_has_fresh_lock", return_value=locked),
            mock.patch.object(ib, "do_ambient_scan") as scan,
            mock.patch.object(c, "_generate_and_speak", gen),
            mock.patch("vision.camera.get_frame", return_value=object()),
            mock.patch("vision.scene.describe_scene_detailed", return_value=details),
        ):
            ib.do_bored_environment_snark({})
        return gen, scan

    def test_fires_a_visual_curiosity_line_grounded_in_objects(self):
        gen, _ = self._fire()
        gen.assert_called_once()
        # Empty-room room riffs use the dedicated boredom purpose so Lean's
        # person-present visual-curiosity suppression cannot discard them.
        self.assertEqual(gen.call_args.kwargs.get("purpose"), "boredom")
        prompt = gen.call_args.args[0]
        self.assertTrue(any(o in prompt for o in ("black chair", "boxes", "abstract art")))

    def test_looks_around_when_not_locked_on_a_face(self):
        _, scan = self._fire(locked=False)
        scan.assert_called_once()

    def test_skips_look_around_when_fixed_on_someone(self):
        _, scan = self._fire(locked=True)
        scan.assert_not_called()

    def test_disabled_is_a_noop(self):
        with mock.patch.object(config, "BORED_ENV_SNARK_ENABLED", False), \
             mock.patch.object(c, "_generate_and_speak") as gen, \
             mock.patch.object(c.threading, "Thread", _SyncThread):
            ib.do_bored_environment_snark({})
        gen.assert_not_called()

    def test_cooldown_blocks_rapid_refire(self):
        gen1, _ = self._fire(now=1000.0)
        gen1.assert_called_once()
        # Second call within the cooldown window does nothing.
        gen2, _ = self._fire(now=1000.0 + config.BORED_ENV_SNARK_COOLDOWN_SECS - 1.0)
        gen2.assert_not_called()

    def test_falls_back_to_cheap_scene_when_detailed_empty(self):
        gen = mock.MagicMock()
        with (
            mock.patch.object(c.time, "monotonic", return_value=2000.0),
            mock.patch.object(c.threading, "Thread", _SyncThread),
            mock.patch.object(c, "_can_proactive_speak", return_value=True),
            mock.patch.object(c, "_face_tracking_has_fresh_lock", return_value=True),
            mock.patch.object(c, "_generate_and_speak", gen),
            mock.patch("vision.camera.get_frame", return_value=object()),
            mock.patch("vision.scene.describe_scene_detailed", return_value={}),
            mock.patch("vision.scene.describe_scene", return_value="a quiet beige room"),
        ):
            ib.do_bored_environment_snark({})
        gen.assert_called_once()
        self.assertIn("a quiet beige room", gen.call_args.args[0])

    def test_no_scene_no_speech(self):
        gen = mock.MagicMock()
        with (
            mock.patch.object(c.time, "monotonic", return_value=3000.0),
            mock.patch.object(c.threading, "Thread", _SyncThread),
            mock.patch.object(c, "_can_proactive_speak", return_value=True),
            mock.patch.object(c, "_face_tracking_has_fresh_lock", return_value=True),
            mock.patch.object(c, "_generate_and_speak", gen),
            mock.patch("vision.camera.get_frame", return_value=object()),
            mock.patch("vision.scene.describe_scene_detailed", return_value={}),
            mock.patch("vision.scene.describe_scene", return_value=""),
        ):
            ib.do_bored_environment_snark({})
        gen.assert_not_called()


class ChoiceWiringTest(unittest.TestCase):
    def test_bored_snark_is_an_idle_choice(self):
        # Present in the people / non-empty / empty-allowed branches.
        with mock.patch.object(c, "_room_looks_empty", return_value=False):
            choices, weights = c._idle_micro_behavior_choices({"people": [{"id": "p1"}]})
        self.assertIn("bored_env_snark", choices)
        self.assertEqual(len(choices), len(weights))


if __name__ == "__main__":
    unittest.main()
