"""Presence, identity and pet-name guards that the 2026-08-20 run walked through.

Three separate ways Rex mis-modelled who was in the room during one continuous
24-minute visit (logs/djr3x-2026-08-20-20-06-34.log).
"""

import time
import unittest
from unittest import mock

import config
from intelligence import consciousness as C


class SilentDepartureOrderingTests(unittest.TestCase):
    """Bret "left" twice without moving.

    _step_presence_tracking checked the silent-departure timeout BEFORE the two
    guards that exist to stop exactly this, so `likely_still_present` and
    `_face_tracking_recently_held_person` could only defer the spoken quip — the
    visit was closed out regardless. Both triggers in the field were Rex's OWN
    motion: an explore drive Bret had just asked for (20:10:50), then a head turn
    (20:29:24). Each wrote a visit_departure episode, and the second was followed
    by a welcome-back line to someone who had never left.
    """

    def setUp(self):
        self._saved = {
            "visible": set(C._visible_people),
            "last_seen": dict(C._last_seen),
            "pending": dict(C._pending_departure_keys),
            "missing": dict(C._first_missing_at),
            "visit": dict(C._visit_started_at),
        }
        C._visible_people.clear()
        C._last_seen.clear()
        C._pending_departure_keys.clear()
        C._first_missing_at.clear()
        C._visit_started_at.clear()

    def tearDown(self):
        C._visible_people.clear(); C._visible_people.update(self._saved["visible"])
        C._last_seen.clear(); C._last_seen.update(self._saved["last_seen"])
        C._pending_departure_keys.clear()
        C._pending_departure_keys.update(self._saved["pending"])
        C._first_missing_at.clear(); C._first_missing_at.update(self._saved["missing"])
        C._visit_started_at.clear(); C._visit_started_at.update(self._saved["visit"])

    def _run(self, *, now, likely_still_present, staged_at, departed_at=None):
        """One presence tick with person 1 staged as pending-departure and no face
        in the snapshot (Rex has looked away / driven off)."""
        departed_at = staged_at if departed_at is None else departed_at
        C._pending_departure_keys[1] = (departed_at, "Bret Benziger", 1, staged_at)
        C._first_missing_at[1] = departed_at
        C._visit_started_at[1] = departed_at - 180.0
        profile = mock.Mock(
            suppress_proactive=False, interaction_busy=False,
            user_mid_sentence=False, likely_still_present=likely_still_present,
            apparent_departure=False,
        )
        with (
            mock.patch.object(C.time, "monotonic", return_value=now),
            mock.patch.object(C.config, "PRESENCE_DEPARTURE_COOLDOWN_SECS", 30),
            mock.patch.object(C, "_face_tracking_recently_held_person",
                              return_value=False),
            mock.patch.object(C, "_should_fire_presence", return_value=False),
            mock.patch.object(C.episodic_hooks, "visit_departure") as departure,
        ):
            C._step_presence_tracking({"people": [], "crowd": {"count": 0}}, profile)
        return departure

    def test_a_talking_person_is_not_logged_as_departed(self):
        """The field case: face gone because Rex drove away, but Bret is talking."""
        departure = self._run(now=200.0, likely_still_present=True, staged_at=100.0)
        departure.assert_not_called()
        self.assertIn(1, C._pending_departure_keys,
                      "the pending entry must survive so a REAL departure can fire")

    def test_the_stage_stamp_is_refreshed_so_a_pause_is_not_a_departure(self):
        """Without re-stamping, the window ages while they talk and resolves the
        instant they stop."""
        self._run(now=200.0, likely_still_present=True, staged_at=100.0)
        self.assertAlmostEqual(C._pending_departure_keys[1][3], 200.0, places=3)

    def test_a_genuinely_absent_person_still_resolves(self):
        departure = self._run(now=200.0, likely_still_present=False, staged_at=100.0)
        departure.assert_called_once()
        self.assertNotIn(1, C._pending_departure_keys)

    def test_a_never_clearing_still_here_signal_cannot_haunt_forever(self):
        """Moving the guards up removed the accidental backstop against an entry
        re-staging every tick (the 2026-07-11 empty-room bug)."""
        cap = float(config.PRESENCE_STILL_HERE_MAX_HOLD_SECS)
        departure = self._run(now=1000.0 + cap + 10.0, likely_still_present=True,
                              staged_at=1000.0, departed_at=1000.0)
        departure.assert_called_once()
        self.assertNotIn(1, C._pending_departure_keys)


class IdentityPromptDropoutTests(unittest.TestCase):
    """JT — greeted by name at 20:11:43 — was queued for "I don't know you yet"
    at 20:13:32, after ~36 s of degraded recognition, while face tracking was
    locked on db:4 the whole time. It went unspoken only because the in-flight
    latch went stale first."""

    def setUp(self):
        C._solo_unknown_since = 0.0
        self._lock = dict(C._face_tracking_lock)
        self.addCleanup(lambda: C._face_tracking_lock.update(self._lock))

    def test_known_lock_vetoes_the_stranger_prompt(self):
        C._face_tracking_lock["person_id"] = 4
        C._face_tracking_lock["last_seen_at"] = time.monotonic()
        with mock.patch.object(C, "_pending_identity_prompt") as pending:
            C._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
            pending.is_set.assert_not_called()
        self.assertEqual(C._solo_unknown_since, 0.0,
                         "the grace timer must reset, not keep counting toward a prompt")

    def test_stale_lock_does_not_veto(self):
        """A genuine stranger after the known person really left must still be asked."""
        C._face_tracking_lock["person_id"] = 4
        C._face_tracking_lock["last_seen_at"] = time.monotonic() - 3600.0
        self.assertFalse(C._known_face_recently_locked())

    def test_no_lock_does_not_veto(self):
        C._face_tracking_lock["person_id"] = None
        self.assertFalse(C._known_face_recently_locked())

    def test_check_never_raises(self):
        C._face_tracking_lock["person_id"] = "not-an-id"
        self.assertFalse(C._known_face_recently_locked())


class FurryPetNameTests(unittest.TestCase):
    """RF-DETR flips the one physical dog to "cat" on a large minority of frames.
    Both confirmed-pet maps are keyed by the DETECTOR label, so under "cat" there
    was no name, the named line pool was skipped, and Rex called Max "the creature"
    (20:24:19) long after Bret had confirmed the name."""

    def setUp(self):
        C._animal_confirmed_pet.clear()
        C._animal_confirmed_species.clear()
        self.addCleanup(C._animal_confirmed_pet.clear)
        self.addCleanup(C._animal_confirmed_species.clear)

    def test_name_survives_a_dog_to_cat_flip(self):
        C._animal_confirmed_pet["dog"] = "Max"
        name, species = C._confirmed_pet_for("cat")
        self.assertEqual(name, "Max")
        self.assertEqual(species, "dog", "must report what Max actually IS")

    def test_true_species_is_preferred_over_either_label(self):
        C._animal_confirmed_pet["cat"] = "Max"       # confirmed while misread
        C._animal_confirmed_species["cat"] = "dog"   # facts DB knows better
        self.assertEqual(C._confirmed_pet_for("dog"), ("Max", "dog"))

    def test_exact_key_still_wins(self):
        C._animal_confirmed_pet["dog"] = "Max"
        C._animal_confirmed_pet["cat"] = "Toby"
        self.assertEqual(C._confirmed_pet_for("cat")[0], "Toby")

    def test_non_furry_species_stay_strictly_keyed(self):
        """A visiting bird is genuinely a different animal — no name bleed."""
        C._animal_confirmed_pet["dog"] = "Max"
        self.assertEqual(C._confirmed_pet_for("bird"), (None, None))

    def test_display_species_uses_the_resolved_identity(self):
        C._animal_confirmed_pet["dog"] = "Max"
        self.assertEqual(C._animal_display_species("cat"), "dog named Max")

    def test_unconfirmed_animal_is_unchanged(self):
        self.assertEqual(C._animal_display_species("cat"), "cat")
        self.assertEqual(C._confirmed_pet_for("cat"), (None, None))


class PoseGuardLogThrottleTests(unittest.TestCase):
    """742 of 7515 lines in the run — ~10% of the log, 14% of its tail — were one
    stationary wall photo being correctly rejected over and over. The rejection is
    right; logging every instance at INFO is what hampered the analysis."""

    def setUp(self):
        C._last_pose_guard_log_at = 0.0
        C._pose_guard_suppressed = 0

    def test_repeat_drops_are_collapsed(self):
        with mock.patch.object(C._log, "info") as info:
            for _ in range(50):
                C._note_pose_guard_drop(1600.0, 500.0, 1, False)
            self.assertEqual(info.call_count, 1, "unthrottled phantom-face spam")

    def test_next_window_reports_the_suppressed_count(self):
        with mock.patch.object(C._log, "info") as info:
            C._note_pose_guard_drop(1600.0, 500.0, 1, False)
            for _ in range(9):
                C._note_pose_guard_drop(1600.0, 500.0, 1, False)
            C._last_pose_guard_log_at = time.monotonic() - 999.0
            C._note_pose_guard_drop(1600.0, 500.0, 1, False)
            self.assertEqual(info.call_count, 2)
            self.assertIn("+9 more", info.call_args[0][0] % info.call_args[0][1:])

    def test_throttle_can_be_disabled(self):
        with (
            mock.patch.object(config, "POSE_FACE_GUARD_LOG_INTERVAL_SECS", 0.0),
            mock.patch.object(C._log, "info") as info,
        ):
            for _ in range(5):
                C._note_pose_guard_drop(1600.0, 500.0, 1, False)
            self.assertEqual(info.call_count, 5)


if __name__ == "__main__":
    unittest.main()
