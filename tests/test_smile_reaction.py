import unittest
from types import SimpleNamespace
from unittest import mock


class SmileReactionTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.c = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_watch = consciousness._smile_reaction_watch
        self.old_last_smile_reaction_at = consciousness._last_smile_reaction_at
        with consciousness._engaged_lock:
            self.old_engaged_person_id = consciousness._engaged_person_id
            self.old_engaged_last_touch_at = consciousness._engaged_last_touch_at
            self.old_recent_engaged_person_id = consciousness._recent_engaged_person_id
            self.old_recent_engaged_touch_at = consciousness._recent_engaged_touch_at
            consciousness._engaged_person_id = None
            consciousness._engaged_last_touch_at = 0.0
            consciousness._recent_engaged_person_id = None
            consciousness._recent_engaged_touch_at = 0.0
        with consciousness._smile_reaction_lock:
            consciousness._smile_reaction_watch = None
        consciousness._last_smile_reaction_at = 0.0

    def tearDown(self):
        c = self.c
        c.world_state.update("people", self.old_people)
        with c._smile_reaction_lock:
            c._smile_reaction_watch = self.old_watch
        c._last_smile_reaction_at = self.old_last_smile_reaction_at
        with c._engaged_lock:
            c._engaged_person_id = self.old_engaged_person_id
            c._engaged_last_touch_at = self.old_engaged_last_touch_at
            c._recent_engaged_person_id = self.old_recent_engaged_person_id
            c._recent_engaged_touch_at = self.old_recent_engaged_touch_at

    def _person(self, expression="neutral", confidence=0.9):
        return {
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 80, 200, 220),
            "face_expression": {
                "expression": expression,
                "mood": "happy" if expression == "smile" else expression,
                "confidence": confidence,
                "source": "mediapipe_face_landmarker",
                "blendshapes": {
                    "mouthSmileLeft": confidence if expression == "smile" else 0.02,
                    "mouthSmileRight": confidence if expression == "smile" else 0.02,
                },
            },
        }

    def _quip_item(self, seq=42):
        return SimpleNamespace(
            seq=seq,
            text="Great, another flawless decision from carbon-based management.",
            audio_path=None,
            tag=None,
        )

    def test_snarky_rex_line_arms_watch_for_neutral_visible_person(self):
        c = self.c
        c.world_state.update("people", [self._person("neutral", 0.92)])

        c._note_rex_speech_item_started(self._quip_item())

        self.assertIsNotNone(c._smile_reaction_watch)
        self.assertEqual(c._smile_reaction_watch["person_key"], "db:1")
        self.assertEqual(c._smile_reaction_watch["baseline_expression"], "neutral")

    def test_watch_not_armed_when_person_is_already_smiling(self):
        c = self.c
        c.world_state.update("people", [self._person("smile", 0.78)])

        c._note_rex_speech_item_started(self._quip_item())

        self.assertIsNone(c._smile_reaction_watch)

    def test_smile_after_rex_speaks_feeds_reaction_awareness(self):
        # Owner rework 2026-08-05: the DEFAULT confirmed-smile path no longer speaks
        # a canned interjection — it records first-person awareness so Rex's NEXT
        # generated line can enjoy that the joke landed. The canned speaker must
        # stay silent, the watch must clear, and the cooldown must still arm (a
        # held smile must not re-mint the awareness every tick).
        from intelligence import reaction_awareness
        c = self.c
        item = self._quip_item()
        c.world_state.update("people", [self._person("neutral", 0.92)])
        c._note_rex_speech_item_started(item)
        c._note_rex_speech_item_done(item)

        c.world_state.update("people", [self._person("smile", 0.76)])
        snapshot = c.world_state.snapshot()

        reaction_awareness.clear()
        self.addCleanup(reaction_awareness.clear)
        saved_cooldown = c._last_smile_reaction_at
        self.addCleanup(lambda: setattr(c, "_last_smile_reaction_at", saved_cooldown))
        c._last_smile_reaction_at = 0.0
        with (
            mock.patch.object(c.config, "SMILE_REACTION_MIN_DELAY_SECS", 0.0),
            mock.patch.object(c, "_speak_smile_reaction", return_value=True) as speak,
        ):
            c._step_smile_reaction(snapshot, mock.Mock())

        speak.assert_not_called()
        active = reaction_awareness.active()
        self.assertIsNotNone(active)
        self.assertEqual(active["kind"], "smile")
        self.assertEqual(active["person_id"], 1)
        # The quip that landed rides along so the awareness can reference it.
        self.assertIn("flawless decision", active["trigger_text"])
        self.assertIsNone(c._smile_reaction_watch)
        self.assertGreater(c._last_smile_reaction_at, 0.0)

    def test_legacy_flag_restores_the_canned_interjection(self):
        c = self.c
        item = self._quip_item()
        c.world_state.update("people", [self._person("neutral", 0.92)])
        c._note_rex_speech_item_started(item)
        c._note_rex_speech_item_done(item)

        c.world_state.update("people", [self._person("smile", 0.76)])
        snapshot = c.world_state.snapshot()

        with (
            mock.patch.object(c.config, "SMILE_REACTION_MIN_DELAY_SECS", 0.0),
            mock.patch.object(c.config, "SMILE_REACTION_CANNED_LINES_ENABLED", True),
            mock.patch.object(c, "_speak_smile_reaction", return_value=True) as speak,
        ):
            c._step_smile_reaction(snapshot, mock.Mock())

        speak.assert_called_once()
        self.assertIsNone(c._smile_reaction_watch)

    def test_raw_blendshapes_never_trigger_when_classifier_says_neutral(self):
        # THE "comedy validated on a non-smile" fix: MediaPipe over-reads mouthSmile on a
        # resting face at the robot's camera angle. The baseline-corrected CLASSIFIER label
        # is the sole trigger — a neutral label with screaming-hot raw blendshapes must not
        # count as a smile, for the discrete reaction or the expression-kind nominator.
        c = self.c
        person = self._person("neutral", 0.9)
        person["face_expression"]["blendshapes"] = {
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.83,
        }
        self.assertFalse(c._person_is_smiling(person))
        kind, _score = c._person_reactable_expression(person)
        self.assertIsNone(kind)

    def test_low_confidence_smile_does_not_trigger_reaction(self):
        c = self.c
        item = self._quip_item()
        c.world_state.update("people", [self._person("neutral", 0.92)])
        c._note_rex_speech_item_started(item)
        c._note_rex_speech_item_done(item)

        c.world_state.update("people", [self._person("smile", 0.20)])
        snapshot = c.world_state.snapshot()

        with (
            mock.patch.object(c.config, "SMILE_REACTION_MIN_DELAY_SECS", 0.0),
            mock.patch.object(c, "_speak_smile_reaction", return_value=True) as speak,
        ):
            c._step_smile_reaction(snapshot, mock.Mock())

        speak.assert_not_called()

    def test_questions_do_not_arm_smile_reaction_watch(self):
        c = self.c
        c.world_state.update("people", [self._person("neutral", 0.92)])
        item = SimpleNamespace(
            seq=43,
            text="Want me to play something?",
            audio_path=None,
            tag=None,
        )

        c._note_rex_speech_item_started(item)

        self.assertIsNone(c._smile_reaction_watch)


if __name__ == "__main__":
    unittest.main()
