"""Departure suppression — no questions at an empty room.

Field bug (2026-07-11): "I'm gonna leave the room now" matched neither the closure
nor the farewell pattern, so the farewell-departure latch never armed and the lean
lull impulse kept asking the departed person questions for 2+ minutes. Three fixes
under test here:

1. end_thread patterns cover the "leave / step out / be right back / to bed" family.
2. The farewell-departure latch closes person-facing proactive purposes while
   keeping the empty-room/boredom purposes alive (the desired behavior: boredom bit
   yes, questions at nobody no).
3. interaction._lean_impulse_person_present — the presence backstop that stands the
   lull impulse down when the target is neither visible nor recently heard (covers
   silent, un-announced walkouts too).
"""

import time
import unittest
from unittest import mock

from intelligence import end_thread


class FarewellPatternTest(unittest.TestCase):
    def setUp(self):
        end_thread.clear()

    def tearDown(self):
        end_thread.clear()

    def _says(self, text):
        end_thread.clear()
        end_thread.note_user_turn(text, person_id=1)

    def test_leave_the_room_arms_farewell(self):
        # The exact field-logged phrase that slipped through.
        self._says("I'm gonna leave the room now")
        self.assertTrue(end_thread.recent_farewell())

    def test_leave_family_phrases_arm_farewell(self):
        for phrase in (
            "I'm going to leave",
            "I'm gonna step out",
            "leaving the room",
            "leaving now",
            "be right back",
            "I'll be back",
            "I'm going to bed",
            "heading to bed",
            "gotta go",
            "I have to leave",
        ):
            self._says(phrase)
            self.assertTrue(
                end_thread.recent_farewell(), f"farewell should arm for {phrase!r}"
            )

    def test_non_departures_do_not_arm(self):
        for phrase in (
            "I love this room",
            "leave it on the table",           # "leave" without departure shape
            "the concert was great",
        ):
            self._says(phrase)
            self.assertFalse(
                end_thread.recent_farewell(), f"farewell should NOT arm for {phrase!r}"
            )

    def test_question_never_arms(self):
        # Questions start a new thread and clear closure state entirely.
        self._says("what happens if I leave the room?")
        self.assertFalse(end_thread.recent_farewell())


class FarewellLatchTest(unittest.TestCase):
    def setUp(self):
        end_thread.clear()

    def tearDown(self):
        end_thread.clear()

    def test_latch_blocks_person_purposes_allows_empty_room(self):
        end_thread.note_user_turn("I'm gonna leave the room now", person_id=1)
        self.assertTrue(end_thread.note_farewell_departure())
        self.assertTrue(end_thread.is_conversation_closed())
        self.assertTrue(end_thread.is_grace_active())
        # Person-facing purposes are muzzled...
        self.assertFalse(end_thread.can_proactive_purpose("visual_curiosity"))
        self.assertFalse(end_thread.can_proactive_purpose("emotional_checkin"))
        self.assertFalse(end_thread.can_proactive_purpose("small_talk"))
        # ...but the empty-room/boredom arc stays alive.
        self.assertTrue(end_thread.can_proactive_purpose("idle_monologue"))
        self.assertTrue(end_thread.can_proactive_purpose("ambient_observation"))

    def test_latch_needs_recent_farewell(self):
        # No goodbye on record: a camera departure alone must not latch.
        self.assertFalse(end_thread.note_farewell_departure())
        self.assertFalse(end_thread.is_conversation_closed())

    def test_return_clears_latch(self):
        end_thread.note_user_turn("gotta go", person_id=1)
        end_thread.note_farewell_departure()
        self.assertTrue(end_thread.is_conversation_closed())
        end_thread.note_presence_return()
        self.assertFalse(end_thread.is_conversation_closed())
        self.assertFalse(end_thread.is_grace_active())


class LeanImpulsePresenceTest(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        self._saved = interaction._last_user_content_at

    def tearDown(self):
        self.interaction._last_user_content_at = self._saved

    def test_visible_person_is_present(self):
        self.interaction._last_user_content_at = 0.0
        with mock.patch.object(
            self.interaction.world_state, "get",
            return_value=[{"person_db_id": 1, "face_id": "Bret"}],
        ):
            self.assertTrue(self.interaction._lean_impulse_person_present(1))

    def test_recently_heard_is_present_even_off_camera(self):
        self.interaction._last_user_content_at = time.monotonic() - 10.0
        with mock.patch.object(self.interaction.world_state, "get", return_value=[]):
            self.assertTrue(self.interaction._lean_impulse_person_present(1))

    def test_gone_and_silent_is_absent(self):
        # Not visible AND last heard beyond the window -> stand down.
        self.interaction._last_user_content_at = time.monotonic() - 10_000.0
        with mock.patch.object(self.interaction.world_state, "get", return_value=[]):
            self.assertFalse(self.interaction._lean_impulse_person_present(1))

    def test_other_person_visible_does_not_count_for_target(self):
        self.interaction._last_user_content_at = time.monotonic() - 10_000.0
        with mock.patch.object(
            self.interaction.world_state, "get",
            return_value=[{"person_db_id": 2, "face_id": "Someone Else"}],
        ):
            self.assertFalse(self.interaction._lean_impulse_person_present(1))

    def test_state_error_fails_open(self):
        self.interaction._last_user_content_at = time.monotonic() - 10_000.0
        with mock.patch.object(
            self.interaction.world_state, "get", side_effect=RuntimeError("boom"),
        ):
            self.assertTrue(self.interaction._lean_impulse_person_present(1))


if __name__ == "__main__":
    unittest.main()
