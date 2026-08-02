"""vision.snapshot — the confirm-gated scene-memory capture (built 2026-08-02;
previously a stub whose 'say yes, remember this scene' offer led nowhere)."""

import unittest
from unittest import mock

from intelligence import interaction


class SceneSnapshotConfirmationTest(unittest.TestCase):
    def setUp(self):
        self._saved = interaction._pending_scene_snapshot
        interaction._pending_scene_snapshot = None

    def tearDown(self):
        interaction._pending_scene_snapshot = self._saved

    def _arm(self, person_id=1):
        with mock.patch.object(interaction.consciousness, "begin_response_wait"):
            line = interaction._offer_scene_snapshot(person_id)
        return line

    def test_offer_arms_slot_and_speaks_confirmation_ask(self):
        line = self._arm()
        self.assertIn("remember this scene", line)
        self.assertIsNotNone(interaction._pending_scene_snapshot)

    def test_yes_executes_the_capture(self):
        self._arm(person_id=7)
        with mock.patch.object(
            interaction, "_execute_scene_snapshot", return_value="Locked in."
        ) as ex:
            resp = interaction._handle_scene_snapshot_confirmation(
                "Yes, remember this scene.", 7
            )
        ex.assert_called_once_with(7)
        self.assertEqual(resp, "Locked in.")
        self.assertIsNone(interaction._pending_scene_snapshot)

    def test_no_declines_without_saving(self):
        self._arm()
        with mock.patch.object(interaction, "_execute_scene_snapshot") as ex:
            resp = interaction._handle_scene_snapshot_confirmation("No, don't.", 1)
        ex.assert_not_called()
        self.assertIn("othing saved", resp)
        self.assertIsNone(interaction._pending_scene_snapshot)

    def test_unrelated_turn_lapses_silently(self):
        self._arm()
        resp = interaction._handle_scene_snapshot_confirmation(
            "What's the weather tomorrow?", 1
        )
        self.assertIsNone(resp)
        self.assertIsNone(interaction._pending_scene_snapshot)

    def test_expired_slot_is_ignored(self):
        self._arm()
        interaction._pending_scene_snapshot["asked_at"] -= 999.0
        resp = interaction._handle_scene_snapshot_confirmation("yes", 1)
        self.assertIsNone(resp)
        self.assertIsNone(interaction._pending_scene_snapshot)

    def test_no_slot_no_op(self):
        self.assertIsNone(
            interaction._handle_scene_snapshot_confirmation("yes", 1)
        )


class SceneSnapshotExecutorTest(unittest.TestCase):
    def test_capture_caption_and_record(self):
        frame = object()
        with (
            mock.patch("vision.camera.capture_current_gaze", return_value=frame),
            mock.patch("vision.face.visible_known_names", return_value=["Bret"]),
            mock.patch("vision.scene.quick_caption",
                       return_value="Bret's cluttered workshop, warm light.") as qc,
            mock.patch("memory.episodes.record_scene") as rec,
            mock.patch.object(interaction, "_tool_router_person_name",
                              return_value="Bret"),
        ):
            resp = interaction._execute_scene_snapshot(1)
        qc.assert_called_once_with(frame, known_people=["Bret"])
        rec.assert_called_once()
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["person_id"], 1)
        self.assertEqual(kwargs["person_name"], "Bret")
        self.assertEqual(kwargs["detail"]["source"], "scene_snapshot")
        self.assertIn("Bret's cluttered workshop", resp)

    def test_no_frame_is_an_honest_refusal(self):
        with (
            mock.patch("vision.camera.capture_current_gaze", return_value=None),
            mock.patch("vision.camera.get_frame", return_value=None),
            mock.patch("memory.episodes.record_scene") as rec,
        ):
            resp = interaction._execute_scene_snapshot(1)
        rec.assert_not_called()
        self.assertIn("no scene saved", resp)

    def test_empty_caption_saves_nothing(self):
        with (
            mock.patch("vision.camera.capture_current_gaze", return_value=object()),
            mock.patch("vision.face.visible_known_names", return_value=[]),
            mock.patch("vision.scene.quick_caption", return_value=""),
            mock.patch("memory.episodes.record_scene") as rec,
        ):
            resp = interaction._execute_scene_snapshot(None)
        rec.assert_not_called()
        self.assertIn("nothing saved", resp)


if __name__ == "__main__":
    unittest.main()
