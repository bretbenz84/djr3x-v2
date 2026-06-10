"""Speaker-gaze acquisition regression tests.

Replay of the wild-head-movement failure from logs/djr3x-2026-06-09-19-03-31.log:
the startup "find anyone" scan (person_id=None, unknown_voice=False) could never
match a candidate in _candidate_matches_speaker_gaze, so _speaker_gaze_note_acquired
never fired and search_requested stayed armed for the whole 13.5s search window —
even after Rex locked onto Bret and greeted him. The next ≥0.45s face-detection
blip relaunched full-room waypoint snaps (right_low / left_down at search servo
speed), swinging the neck to its rail and the lift up from the seated greet pose.

Policy now: an "anyone" scan is satisfied by ANY visible face; specific-person and
unknown-voice intents keep their strict matching.
"""

import unittest


def _consciousness():
    from intelligence import consciousness

    return consciousness


class CandidateMatchTests(unittest.TestCase):
    def test_anyone_scan_matches_any_candidate(self):
        c = _consciousness()
        intent = {"person_id": None, "unknown_voice": False, "reason": "startup"}
        known = {"key": "db:1", "person_id": 1}
        unknown = {"key": "track:3", "person_id": None}
        self.assertTrue(c._candidate_matches_speaker_gaze(known, intent))
        self.assertTrue(c._candidate_matches_speaker_gaze(unknown, intent))

    def test_specific_person_intent_unchanged(self):
        c = _consciousness()
        intent = {"person_id": 1, "unknown_voice": False, "reason": "speech"}
        self.assertTrue(
            c._candidate_matches_speaker_gaze({"key": "db:1", "person_id": 1}, intent)
        )
        self.assertFalse(
            c._candidate_matches_speaker_gaze({"key": "db:2", "person_id": 2}, intent)
        )
        self.assertFalse(
            c._candidate_matches_speaker_gaze({"key": "t:0", "person_id": None}, intent)
        )

    def test_unknown_voice_intent_unchanged(self):
        c = _consciousness()
        intent = {"person_id": None, "unknown_voice": True, "reason": "speech"}
        self.assertTrue(
            c._candidate_matches_speaker_gaze({"key": "t:0", "person_id": None}, intent)
        )
        self.assertFalse(
            c._candidate_matches_speaker_gaze({"key": "db:1", "person_id": 1}, intent)
        )

    def test_no_intent_matches_nothing(self):
        c = _consciousness()
        self.assertFalse(
            c._candidate_matches_speaker_gaze({"key": "db:1", "person_id": 1}, None)
        )


class StartupScanAcquisitionTests(unittest.TestCase):
    """Finding any face stands the startup room scan down."""

    def setUp(self):
        self.c = _consciousness()
        with self.c._speaker_gaze_lock:
            self._saved = dict(self.c._speaker_gaze_intent)
        self.addCleanup(self._restore)

    def _restore(self):
        with self.c._speaker_gaze_lock:
            self.c._speaker_gaze_intent.clear()
            self.c._speaker_gaze_intent.update(self._saved)

    def test_any_face_acquisition_clears_startup_search(self):
        c = self.c
        # Arm the startup "find anyone" scan the way request_face_acquisition_scan does.
        with c._speaker_gaze_lock:
            c._speaker_gaze_intent.clear()
            c._speaker_gaze_intent.update({
                "person_id": None,
                "unknown_voice": False,
                "reason": "startup",
                "requested_at": 100.0,
                "search_requested": True,
                "search_started_at": 100.0,
                "last_search_at": 0.0,
                "search_index": 2,
                "search_plan": [(0.5, 0.6)],
                "search_plan_index": 1,
                "waypoint_committed_at": 100.0,
                "waypoint_pose": "right_low",
                "acquired_at": 0.0,
            })
        candidate = {"key": "db:1", "person_id": 1, "area": 14400}
        # The tracking step's acquisition gate: matches → note acquired.
        self.assertTrue(c._candidate_matches_speaker_gaze(candidate, {"person_id": None, "unknown_voice": False, "reason": "startup"}))
        c._speaker_gaze_note_acquired(candidate)
        with c._speaker_gaze_lock:
            self.assertFalse(c._speaker_gaze_intent["search_requested"])
            self.assertIsNone(c._speaker_gaze_intent["search_plan"])
            self.assertGreater(float(c._speaker_gaze_intent["acquired_at"]), 0.0)


if __name__ == "__main__":
    unittest.main()
