"""Dynamic animal presence bit (owner 2026-08-03).

Replaces the flat 5-minute animal cooldowns: first sighting reacts, a REAL
departure (out of frame past a grace window — frame flicker doesn't count)
followed by a return earns an escalating return joke ("womp rat energy" →
"doing laps" → "get it a badge"), and pacing (min gap between spoken remarks +
a per-run session cap + a long-absence reset) keeps the bit from becoming a
doorbell.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import consciousness as C


def _snapshot(*species):
    return {"animals": [{"species": s, "position": "lower right", "furred": True}
                        for s in species]}


class _PresenceCase(unittest.TestCase):
    def setUp(self):
        self._saved_presence = dict(C._animal_presence)
        self._saved_pending = dict(C._pending_animal_arrivals)
        self._saved_seen = set(C._animal_seen_signatures)
        C._animal_presence.clear()
        C._pending_animal_arrivals.clear()
        C._animal_seen_signatures.clear()
        self._ls = mock.patch.object(C, "_last_snapshot", {"animals": []})
        self._ls.start()
        self.addCleanup(self._ls.stop)

    def tearDown(self):
        C._animal_presence.clear()
        C._animal_presence.update(self._saved_presence)
        C._pending_animal_arrivals.clear()
        C._pending_animal_arrivals.update(self._saved_pending)
        C._animal_seen_signatures.clear()
        C._animal_seen_signatures.update(self._saved_seen)

    def _tick(self, *species):
        C._stage_animal_arrivals(_snapshot(*species))

    def _dog(self):
        return C._animal_presence.get("dog")

    def _pending_kind(self):
        p = C._pending_animal_arrivals.get("dog")
        return (p or {}).get("kind")


class ArrivalAndFlickerTest(_PresenceCase):
    def test_first_sighting_stages_arrival(self):
        self._tick("dog")
        self.assertEqual(self._pending_kind(), "arrival")
        self.assertTrue(self._dog()["present"])
        self.assertEqual(self._dog()["return_count"], 0)

    def test_flicker_below_grace_is_not_a_departure(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        # Dog drops out of frame for a moment (well under the grace window)...
        self._tick()
        self.assertTrue(self._dog()["present"], "flicker must not read as leaving")
        # ...and pops back in: still present, no return joke staged.
        self._tick("dog")
        self.assertEqual(C._pending_animal_arrivals, {})
        self.assertEqual(self._dog()["return_count"], 0)

    def test_absence_past_grace_marks_departure(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        self._dog()["last_seen_at"] = time.monotonic() - 120  # unseen for 2 min
        self._tick()
        self.assertFalse(self._dog()["present"])
        self.assertTrue(self._dog()["departed_at"])


class ReturnJokeTest(_PresenceCase):
    def setUp(self):
        super().setUp()
        # The pet-name guess ("is that Max?") reads the REAL people DB — on the
        # robot Mac that finds Bret's dogs and swaps the line. These tests are
        # about the generic pools, so hold the guess off.
        p = mock.patch.object(config, "ANIMAL_PET_NAME_GUESS_ENABLED", False, create=True)
        p.start()
        self.addCleanup(p.stop)
        C._animal_guessed_pet.clear()
        self.addCleanup(C._animal_guessed_pet.clear)

    def _depart(self, *, remark_ago=300.0, away_secs=300.0):
        rec = self._dog()
        rec["present"] = False
        rec["departed_at"] = time.monotonic() - away_secs
        rec["last_remark_at"] = time.monotonic() - remark_ago

    def test_departure_then_return_stages_the_return_joke(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        self._depart()
        self._tick("dog")
        self.assertEqual(self._pending_kind(), "return")
        self.assertEqual(C._pending_animal_arrivals["dog"]["return_count"], 1)

    def test_repeat_returns_escalate_the_count(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        for expected in (1, 2, 3):
            self._depart()
            self._tick("dog")
            self.assertEqual(
                C._pending_animal_arrivals["dog"]["return_count"], expected)
            C._pending_animal_arrivals.clear()

    def test_return_lines_escalate_by_count(self):
        for count, pool in (
            (1, C._ANIMAL_RETURN_LINES_FIRST),
            (2, C._ANIMAL_RETURN_LINES_SECOND),
            (5, C._ANIMAL_RETURN_LINES_MANY),
        ):
            animal = {"species": "dog", "furred": True,
                      "kind": "return", "return_count": count}
            frame, line = C._animal_reaction_frame_and_line(animal)
            self.assertIn(line, pool, f"count={count}")
            self.assertNotEqual(frame.affect, "surprised",
                                "a return is a clocked pattern, not a surprise")

    def test_non_furry_returns_use_the_generic_pool(self):
        animal = {"species": "bird", "kind": "return", "return_count": 1}
        _frame, line = C._animal_reaction_frame_and_line(animal)
        self.assertIn(line, C._ANIMAL_RETURN_LINES_GENERIC)

    def test_arrival_reaction_unchanged_for_furry_companions(self):
        animal = {"species": "dog", "furred": True}
        frame, line = C._animal_reaction_frame_and_line(animal)
        self.assertIn(line, C._FURRY_ANIMAL_REACTION_LINES)
        self.assertEqual(frame.affect, "surprised")


class PacingTest(_PresenceCase):
    def _departed_dog(self, **overrides):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        rec = self._dog()
        rec.update({"present": False,
                    "departed_at": time.monotonic() - 300,
                    "last_remark_at": time.monotonic() - 300})
        rec.update(overrides)

    def test_min_gap_since_last_remark_stays_silent(self):
        self._departed_dog(last_remark_at=time.monotonic() - 5)
        self._tick("dog")
        self.assertEqual(C._pending_animal_arrivals, {},
                         "a return seconds after the last remark must not speak")
        # State still tracked — the silent return counts.
        self.assertEqual(self._dog()["return_count"], 1)

    def test_session_cap_spends_the_bit(self):
        cap = int(getattr(config, "ANIMAL_REMARK_SESSION_CAP", 4))
        self._departed_dog(remarks_spoken=cap)
        self._tick("dog")
        self.assertEqual(C._pending_animal_arrivals, {},
                         "beyond the session cap the bit is spent")

    def test_long_absence_resets_to_fresh_arrival(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        rec = self._dog()
        rec["present"] = False
        rec["return_count"] = 3
        rec["departed_at"] = time.monotonic() - (
            float(getattr(config, "ANIMAL_FRESH_ARRIVAL_AFTER_SECS", 1800.0)) + 60)
        self._tick("dog")
        self.assertEqual(self._pending_kind(), "arrival",
                         "after hours away, 'back again' would read weird")
        self.assertEqual(self._dog()["return_count"], 0)


class CrossSpeciesFlipTest(_PresenceCase):
    """One pet, two labels (field 2026-08-13): RF-DETR flip-flopped Max between
    "dog" and "cat" — the per-species ledger minted a second fresh arrival and
    Rex announced the same "small furry lifeform" twice in 20 seconds."""

    def setUp(self):
        super().setUp()
        self._saved_reacted = dict(C._animal_species_reacted_at)
        C._animal_species_reacted_at.clear()
        self.addCleanup(self._restore_reacted)

    def _restore_reacted(self):
        C._animal_species_reacted_at.clear()
        C._animal_species_reacted_at.update(self._saved_reacted)

    def test_species_flip_after_spoken_arrival_stays_silent(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        C._animal_species_reacted_at["dog"] = time.monotonic() - 12
        self._tick("cat", "dog")  # detector now labels the same pet both ways
        self.assertNotIn("cat", C._pending_animal_arrivals)
        self.assertTrue(C._animal_presence["cat"]["present"],
                        "presence is still tracked, just silently")

    def test_sibling_pending_mutes_the_second_label(self):
        self._tick("dog")  # dog arrival staged, not yet spoken
        self._tick("cat", "dog")
        self.assertIn("dog", C._pending_animal_arrivals)
        self.assertNotIn("cat", C._pending_animal_arrivals)

    def test_pending_furry_remark_dropped_once_sibling_speaks(self):
        # The morning failure mode: the cat line waited out the output gate for
        # 8s and fired AFTER the dog line (and after the human had already
        # introduced the pet by name).
        self._tick("cat")
        self.assertIn("cat", C._pending_animal_arrivals)
        C._animal_species_reacted_at["dog"] = time.monotonic() - 5
        with mock.patch.object(C, "_speak_async", return_value=True) as speak:
            fired = C._fire_pending_animal_arrival_reaction()
        self.assertFalse(fired)
        speak.assert_not_called()
        self.assertNotIn("cat", C._pending_animal_arrivals,
                         "a muted pending must be dropped, not retried")

    def test_cooldown_expiry_lets_a_genuinely_new_furry_speak(self):
        cooldown = float(getattr(
            config, "ANIMAL_FURRY_CROSS_SPECIES_REMARK_COOLDOWN_SECS", 180.0))
        C._animal_species_reacted_at["dog"] = time.monotonic() - (cooldown + 30)
        self._tick("cat")
        self.assertIn("cat", C._pending_animal_arrivals)

    def test_non_furry_arrivals_are_not_muted(self):
        C._animal_species_reacted_at["dog"] = time.monotonic() - 5
        C._stage_animal_arrivals(
            {"animals": [{"species": "bird", "position": "upper left"}]})
        self.assertIn("bird", C._pending_animal_arrivals,
                      "a bird is not a relabeled dog")

    def test_same_species_return_is_not_its_own_sibling(self):
        self._tick("dog")
        C._pending_animal_arrivals.clear()
        C._animal_species_reacted_at["dog"] = time.monotonic() - 130
        rec = self._dog()
        rec["present"] = False
        rec["departed_at"] = time.monotonic() - 300
        rec["last_remark_at"] = time.monotonic() - 300
        self._tick("dog")
        self.assertEqual(self._pending_kind(), "return",
                         "the guard is cross-species only")


class SpokenLedgerTest(_PresenceCase):
    def test_on_spoke_bumps_the_ledger_and_only_arrivals_hit_the_diary(self):
        self._tick("dog")
        rec = self._dog()
        rec["present"] = False
        rec["departed_at"] = time.monotonic() - 300
        C._pending_animal_arrivals.clear()
        self._tick("dog")  # stages the return

        spoken = []

        def fake_speak(line, affect, **kwargs):
            spoken.append(line)
            on_spoke = kwargs.get("on_spoke")
            if on_spoke:
                on_spoke()
            return True

        with mock.patch.object(C, "_speak_async", side_effect=fake_speak), \
             mock.patch.object(C.episodic_hooks, "animal") as diary:
            fired = C._fire_pending_animal_arrival_reaction()
        self.assertTrue(fired)
        self.assertEqual(len(spoken), 1)
        self.assertEqual(rec["remarks_spoken"], 1)
        self.assertGreater(rec["last_remark_at"], 0.0)
        self.assertEqual(C._pending_animal_arrivals, {})
        diary.assert_not_called()  # returns don't spam the diary; arrivals do

    def test_losing_the_governor_race_does_not_burn_the_cap(self):
        self._tick("dog")
        rec = self._dog()
        with mock.patch.object(C, "_speak_async", return_value=False):
            C._fire_pending_animal_arrival_reaction()
        self.assertEqual(rec["remarks_spoken"], 0)
        self.assertIn("dog", C._pending_animal_arrivals,
                      "an unspoken remark stays pending")


if __name__ == "__main__":
    unittest.main()


class ReacquireAndConfirmTest(_PresenceCase):
    """Field 2026-08-19 22:25: Rex wandered, oriented back to the couch, and
    greeted the cat that never moved with 'the furry lifeform is BACK. Still
    calling it Max until Bret tells me otherwise' — a departure that didn't
    happen, hedged with a name the owner had ALREADY confirmed, in the third
    person to the owner's face. Three fixes under test: base-motion gaps are
    re-sights (not returns), confirmed names retire the hedge, and the
    unconfirmed hedge speaks to the owner in the second person."""

    def setUp(self):
        super().setUp()
        self._pose = {"v": (0.0, 0.0, 0.0)}
        self._bp = mock.patch.object(C, "_base_pose",
                                     side_effect=lambda: self._pose["v"])
        self._bp.start()
        self.addCleanup(self._bp.stop)
        C._animal_guessed_pet.clear()
        C._animal_confirmed_pet.clear()
        self.addCleanup(C._animal_guessed_pet.clear)
        self.addCleanup(C._animal_confirmed_pet.clear)

    def _depart_dog(self):
        rec = self._dog()
        rec["last_seen_at"] = time.monotonic() - (
            float(config.ANIMAL_DEPARTURE_GRACE_SECS) + 1.0)
        self._tick()          # empty frame past grace -> departed
        self.assertFalse(rec["present"])
        C._pending_animal_arrivals.clear()   # the arrival remark is history

    def test_base_motion_gap_is_a_resight_not_a_return(self):
        self._tick("dog")
        self._depart_dog()
        self._pose["v"] = (0.0, 0.0, 1.0)    # Rex turned ~57 deg in the gap
        self._tick("dog")
        pending = C._pending_animal_arrivals.get("dog")
        self.assertIsNotNone(pending)
        self.assertTrue(pending.get("reacquired"))
        self.assertEqual(self._dog()["return_count"], 0)   # no round trip counted

    def test_still_base_gap_is_a_real_return(self):
        self._tick("dog")
        self._depart_dog()
        self._tick("dog")                    # pose unchanged: the DOG left
        pending = C._pending_animal_arrivals.get("dog")
        self.assertIsNotNone(pending)
        self.assertFalse(pending.get("reacquired", False))
        self.assertEqual(self._dog()["return_count"], 1)

    def test_reacquired_nonfurry_stays_silent(self):
        C._stage_animal_arrivals({"animals": [{"species": "bird"}]})
        rec = C._animal_presence["bird"]
        rec["last_seen_at"] = time.monotonic() - (
            float(config.ANIMAL_DEPARTURE_GRACE_SECS) + 1.0)
        self._tick()
        C._pending_animal_arrivals.clear()
        self._pose["v"] = (0.5, 0.0, 0.0)    # Rex drove half a metre
        C._stage_animal_arrivals({"animals": [{"species": "bird"}]})
        self.assertNotIn("bird", C._pending_animal_arrivals)

    def test_resight_line_names_the_confirmed_pet_without_back(self):
        C._animal_confirmed_pet["dog"] = "Max"
        animal = {"species": "dog", "furred": True, "kind": "return",
                  "return_count": 0, "reacquired": True}
        with mock.patch.object(C.random, "choice", side_effect=lambda s: s[0]):
            _, line = C._animal_reaction_frame_and_line(animal)
        self.assertIn("Max", line)
        self.assertNotIn("back", line.lower())

    def test_confirmed_return_drops_every_hedge(self):
        C._animal_confirmed_pet["dog"] = "Max"
        animal = {"species": "dog", "furred": True, "kind": "return",
                  "return_count": 1}
        with mock.patch.object(C.random, "choice", side_effect=lambda s: s[0]):
            _, line = C._animal_reaction_frame_and_line(animal)
        self.assertIn("Max", line)
        for hedge in ("otherwise", "whoever", "probably"):
            self.assertNotIn(hedge, line.lower())

    def test_unconfirmed_hedge_speaks_to_the_owner(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max", ())
        animal = {"species": "dog", "furred": True, "kind": "return",
                  "return_count": 1}
        with mock.patch.object(C.random, "choice", side_effect=lambda s: s[0]):
            _, line = C._animal_reaction_frame_and_line(animal)
        self.assertIn("you tell me otherwise", line)
        self.assertNotIn("Bret tells", line)

    # ── the answer capture ──
    def test_affirmative_answer_confirms_the_guess(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max", ("Toby",))
        C.note_pet_guess_answer("Yeah, it's Max.")
        self.assertEqual(C._animal_confirmed_pet.get("dog"), "Max")

    def test_bare_yes_confirms_too(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max", ())
        C.note_pet_guess_answer("Yep.")
        self.assertEqual(C._animal_confirmed_pet.get("dog"), "Max")

    def test_correction_confirms_the_sibling(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max", ("Toby",))
        C.note_pet_guess_answer("No, that's Toby.")
        self.assertEqual(C._animal_confirmed_pet.get("dog"), "Toby")

    def test_plain_denial_drops_the_guess(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max", ())
        C.note_pet_guess_answer("No, that's not him.")
        self.assertNotIn("dog", C._animal_confirmed_pet)
        self.assertNotIn("dog", C._animal_guessed_pet)
