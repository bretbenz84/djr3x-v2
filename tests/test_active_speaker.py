"""
Active-speaker detection — COMMIT 1 (scaffold): the world-state write/latch/read
contract + the VAD accessor. The detection layers (head-pose, lip-energy,
arbitration) land in later commits and get their own tests; here we pin the
race-prone plumbing: is_speaking is written to exactly one slot, keyed by
person_db_id (stable across a slot resize), the recent-speaker latch survives the
post-turn gap, and a concurrent identity write never clobbers a speaker write.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from world_state import world_state
from vision import active_speaker as A


class VadAccessorTest(unittest.TestCase):
    def test_is_user_speaking_reflects_vad_state(self):
        from awareness.situation import assessor

        assessor.set_vad_active(True)
        self.assertTrue(assessor.is_user_speaking())
        assessor.set_vad_active(False)
        self.assertFalse(assessor.is_user_speaking())


class _WorldStateFixture(unittest.TestCase):
    def setUp(self):
        self._saved = world_state.get("people")
        A.reset()

    def tearDown(self):
        world_state.update("people", self._saved)
        A.reset()


class WriteSpeakerFieldsTest(_WorldStateFixture):
    def _people(self):
        return [
            {"id": "p1", "person_db_id": 1, "face_visible": True},
            {"id": "p2", "person_db_id": 2, "face_visible": True},
        ]

    def test_winner_by_person_db_id_sets_exactly_one(self):
        out = A._write_speaker_fields(
            self._people(), winner_pid=2, winner_slot=None, confidence=0.8, now=100.0
        )
        self.assertFalse(out[0]["is_speaking"])
        self.assertTrue(out[1]["is_speaking"])
        self.assertEqual(out[1]["speaking_confidence"], 0.8)
        self.assertEqual(out[1]["speaking_updated_at"], 100.0)
        self.assertEqual(sum(1 for s in out if s["is_speaking"]), 1)

    def test_winner_by_slot_index_when_pid_unknown(self):
        people = [{"id": "p1", "person_db_id": None}, {"id": "p2", "person_db_id": None}]
        out = A._write_speaker_fields(
            people, winner_pid=None, winner_slot=0, confidence=0.5, now=100.0
        )
        self.assertTrue(out[0]["is_speaking"])
        self.assertFalse(out[1]["is_speaking"])

    def test_no_winner_clears_all(self):
        out = A._write_speaker_fields(
            self._people(), winner_pid=None, winner_slot=None, confidence=0.0, now=100.0
        )
        self.assertEqual(sum(1 for s in out if s["is_speaking"]), 0)


class PublishAndLatchTest(_WorldStateFixture):
    def test_publish_writes_world_state_and_latches(self):
        world_state.update("people", [{"id": "p1", "person_db_id": 7, "face_visible": True}])
        A._publish_speaker(winner_pid=7, winner_slot=0, confidence=0.9)
        self.assertTrue(world_state.get("people")[0]["is_speaking"])
        latched = A.recent_visual_speaker(max_age_secs=10.0)
        self.assertIsNotNone(latched)
        self.assertEqual(latched["person_db_id"], 7)

    def test_latch_expires_with_age(self):
        world_state.update("people", [{"id": "p1", "person_db_id": 7, "face_visible": True}])
        A._publish_speaker(winner_pid=7, winner_slot=0, confidence=0.9, now=time.time() - 100.0)
        self.assertIsNone(A.recent_visual_speaker(max_age_secs=3.0))   # too old
        self.assertIsNotNone(A.recent_visual_speaker(max_age_secs=200.0))

    def test_clearing_does_not_touch_latch(self):
        # A winner publishes (latches); a later no-winner cycle clears the live
        # field but the latch must survive for the post-turn voice read.
        world_state.update("people", [{"id": "p1", "person_db_id": 7, "face_visible": True}])
        A._publish_speaker(winner_pid=7, winner_slot=0, confidence=0.9)
        A._publish_speaker(winner_pid=None, winner_slot=None, confidence=0.0)
        self.assertFalse(world_state.get("people")[0]["is_speaking"])   # live cleared
        self.assertIsNotNone(A.recent_visual_speaker(max_age_secs=10.0))  # latch kept


class CurrentSpeakerTest(_WorldStateFixture):
    def test_returns_fresh_visible_highest_confidence(self):
        now = time.time()
        world_state.update("people", [
            {"id": "p1", "person_db_id": None, "face_visible": True,
             "is_speaking": True, "speaking_confidence": 0.3, "speaking_updated_at": now},
            {"id": "p2", "person_db_id": None, "face_visible": True,
             "is_speaking": True, "speaking_confidence": 0.7, "speaking_updated_at": now},
        ])
        cur = A.current_speaker()
        self.assertIsNotNone(cur)
        self.assertEqual(cur["speaking_confidence"], 0.7)

    def test_ignores_stale_and_invisible(self):
        now = time.time()
        world_state.update("people", [
            {"id": "p1", "person_db_id": None, "face_visible": True,
             "is_speaking": True, "speaking_confidence": 0.9, "speaking_updated_at": now - 100.0},
            {"id": "p2", "person_db_id": None, "face_visible": False,
             "is_speaking": True, "speaking_confidence": 0.9, "speaking_updated_at": now},
        ])
        self.assertIsNone(A.current_speaker())

    def test_resolves_name_via_people_db(self):
        now = time.time()
        world_state.update("people", [
            {"id": "p1", "person_db_id": 5, "face_visible": True,
             "is_speaking": True, "speaking_confidence": 0.8, "speaking_updated_at": now},
        ])
        with mock.patch("memory.people.get_person", return_value={"name": "Dana"}):
            cur = A.current_speaker()
        self.assertEqual(cur["name"], "Dana")


class NonClobberTest(_WorldStateFixture):
    def test_speaker_write_and_identity_write_coexist(self):
        # An active-speaker write and a concurrent identity re-bind on the same
        # slot must both survive (the is_speaking analogue of the pose/identity
        # flicker test). Proves the two writers interleave under the shared lock.
        world_state.update("people", [{"id": "p1", "person_db_id": 9, "face_visible": True}])

        def write_speaker():
            for _ in range(200):
                A._publish_speaker(winner_pid=9, winner_slot=0, confidence=0.8)

        def write_identity():
            for _ in range(200):
                def _apply(people):
                    if people:
                        people[0]["face_id"] = "Bret Benziger"
                        return people
                    return None
                world_state.mutate("people", _apply)

        a = threading.Thread(target=write_speaker)
        b = threading.Thread(target=write_identity)
        a.start(); b.start(); a.join(); b.join()

        slot = world_state.get("people")[0]
        self.assertTrue(slot["is_speaking"])          # speaker write survived
        self.assertEqual(slot["face_id"], "Bret Benziger")  # identity write survived


class DisabledFlagTest(_WorldStateFixture):
    def test_update_is_noop_when_disabled(self):
        world_state.update("people", [{"id": "p1", "person_db_id": 1, "face_visible": True}])
        with mock.patch.object(A.config, "ACTIVE_SPEAKER_ENABLED", False):
            A.update([{"slot_idx": 0, "person_db_id": 1, "jaw_open": 0.5, "yaw": 0.0, "ts": time.time()}], True)
        self.assertNotIn("is_speaking", {k for s in world_state.get("people") for k in s if s.get(k)})


if __name__ == "__main__":
    unittest.main()
