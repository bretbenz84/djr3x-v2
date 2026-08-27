"""'Is that Max?' — the furry-arrival remark asks by pet NAME when someone Rex saw
recently has told him about a pet (owner note 2026-08-18). Covers the pet-fact
reader (memory.facts.get_pets), the owner candidate ladder, the line choice
(same species / two pets / species mismatch / nobody), and the named return line.
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness as C
from memory import facts


def _fact(key, value, conf=0.55):
    return {"key": key, "value": value, "confidence": conf, "category": "pet"}


class GetPetsTest(unittest.TestCase):
    def _pets(self, rows):
        with mock.patch.object(facts, "get_facts_by_category", return_value=rows):
            return facts.get_pets(1)

    def test_reads_the_field_key_zoo(self):
        rows = [
            _fact("pet_name", "Max"), _fact("dog", "Max"), _fact("dog_name_1", "Max"),
            _fact("dog_name_2", "Toby"), _fact("dog_age_1", "3.5 years"),
            _fact("dog_condition_1", "blind"), _fact("pet_age", "9 years"),
            _fact("pet_condition", "blind"),
        ]
        pets = self._pets(rows)
        self.assertEqual([p["name"] for p in pets], ["Max", "Toby"])
        self.assertEqual({p["species"] for p in pets}, {"dog"})

    def test_name_inside_a_sentence_value(self):
        pets = self._pets([_fact("pet", "a cat named Pixel"), _fact("pet_note", "she is fluffy")])
        self.assertEqual(pets, [{"name": "Pixel", "species": "cat", "confidence": 0.55}])

    def test_species_unknown_stays_pet(self):
        pets = self._pets([_fact("pet_name", "Biscuit")])
        self.assertEqual(pets[0]["species"], "pet")

    def test_no_name_no_pet(self):
        self.assertEqual(self._pets([_fact("dog_age", "3"), _fact("pet_condition", "blind")]), [])

    def test_best_attested_first(self):
        pets = self._pets([_fact("dog_name_1", "Max", 0.4), _fact("cat", "Pixel", 0.9)])
        self.assertEqual([p["name"] for p in pets], ["Pixel", "Max"])


class _GuessCase(unittest.TestCase):
    def setUp(self):
        C._animal_guessed_pet.clear()
        C._animal_species_contradicted.clear()
        self.addCleanup(C._animal_guessed_pet.clear)
        self.addCleanup(C._animal_species_contradicted.clear)
        self._patches = [
            mock.patch.object(config, "ANIMAL_PET_NAME_GUESS_ENABLED", True, create=True),
            mock.patch.object(config, "ANIMAL_PET_GUESS_LINES", ("{first}, is that {name}?",), create=True),
            mock.patch.object(config, "ANIMAL_PET_GUESS_TWO_LINES", ("{first}, {name} or {alt}?",), create=True),
            mock.patch.object(config, "ANIMAL_SPECIES_MISMATCH_ASK_LINES",
                              ("{first}, who is this one?",), create=True),
            mock.patch.object(config, "ANIMAL_PET_RETURN_LINES", ("{name} is back, {first}.",), create=True),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _owners(self, *pairs):
        return mock.patch.object(C, "_pet_owner_candidates", return_value=list(pairs))

    def _pets(self, table):
        return mock.patch.object(facts, "get_pets", side_effect=lambda pid: list(table.get(pid, [])))


class GuessLineTest(_GuessCase):
    def test_same_species_single_pet(self):
        with self._owners((1, "Bret Benziger")), self._pets({1: [{"name": "Max", "species": "dog", "confidence": 0.6}]}):
            self.assertEqual(C._pet_name_guess_line("dog"), "Bret, is that Max?")
        # (owner_first, pet_name, alt_names, name->true-species) — alts feed the
        # answer capture ("no, that's Toby" confirms the sibling); the species
        # map lets a confirmation correct a detector misread (cat -> dog).
        self.assertEqual(C._animal_guessed_pet["dog"],
                         ("Bret", "Max", (), {"Max": "dog"}))

    def test_two_pets_same_species(self):
        table = {1: [{"name": "Max", "species": "dog", "confidence": 0.6},
                     {"name": "Toby", "species": "dog", "confidence": 0.5}]}
        with self._owners((1, "Bret Benziger")), self._pets(table):
            self.assertEqual(C._pet_name_guess_line("dog"), "Bret, Max or Toby?")

    def test_species_mismatch_asks_without_naming_a_pet(self):
        # Field 2026-08-27 13:35:47: detector said cat, the DB says Max is a dog,
        # and Rex guessed "Max" anyway with "Ignore the part of me that said cat."
        with self._owners((1, "Bret Benziger")), self._pets({1: [{"name": "Max", "species": "dog", "confidence": 0.6}]}):
            line = C._pet_name_guess_line("cat")
        self.assertEqual(line, "Bret, who is this one?")
        self.assertNotIn("Max", line)
        self.assertNotIn("cat", line)
        # No guess stored, so no later return line can call the dog "Max".
        self.assertNotIn("cat", C._animal_guessed_pet)

    def test_species_mismatch_is_not_recorded_as_that_species(self):
        with self._owners((1, "Bret Benziger")), self._pets({1: [{"name": "Max", "species": "dog", "confidence": 0.6}]}):
            C._pet_name_guess_line("cat")
        self.assertEqual(C._animal_display_species("cat"), "animal")

    def test_species_the_db_never_pinned_down_still_gets_the_name(self):
        with self._owners((1, "Bret Benziger")), self._pets({1: [{"name": "Biscuit", "species": "pet", "confidence": 0.6}]}):
            self.assertEqual(C._pet_name_guess_line("cat"), "Bret, is that Biscuit?")

    def test_nobody_recent_with_pets_falls_back(self):
        with self._owners((4, "JT")), self._pets({4: []}):
            self.assertIsNone(C._pet_name_guess_line("dog"))
        with self._owners():
            self.assertIsNone(C._pet_name_guess_line("dog"))

    def test_first_owner_with_pets_wins(self):
        table = {4: [], 1: [{"name": "Max", "species": "dog", "confidence": 0.6}]}
        with self._owners((4, "JT"), (1, "Bret Benziger")), self._pets(table):
            self.assertEqual(C._pet_name_guess_line("dog"), "Bret, is that Max?")

    def test_disabled(self):
        with mock.patch.object(config, "ANIMAL_PET_NAME_GUESS_ENABLED", False, create=True), \
             self._owners((1, "Bret")), self._pets({1: [{"name": "Max", "species": "dog", "confidence": 0.6}]}):
            self.assertIsNone(C._pet_name_guess_line("dog"))

    def test_fish_is_not_a_furry_guess(self):
        with self._owners((1, "Bret")), self._pets({1: [{"name": "Nemo", "species": "fish", "confidence": 0.9}]}):
            self.assertIsNone(C._pet_name_guess_line("dog"))


class ArrivalAndReturnWiringTest(_GuessCase):
    def test_furry_arrival_uses_the_guess(self):
        with mock.patch.object(C, "_pet_name_guess_line", return_value="Bret, is that Max?"):
            _frame, line = C._animal_reaction_frame_and_line({"species": "dog", "furred": True})
        self.assertEqual(line, "Bret, is that Max?")

    def test_furry_arrival_without_guess_uses_pool(self):
        with mock.patch.object(C, "_pet_name_guess_line", return_value=None):
            _frame, line = C._animal_reaction_frame_and_line({"species": "dog", "furred": True})
        self.assertIn(line, C._FURRY_ANIMAL_REACTION_LINES)

    def test_return_uses_the_guessed_name(self):
        C._animal_guessed_pet["dog"] = ("Bret", "Max")
        _frame, line = C._animal_reaction_frame_and_line(
            {"species": "dog", "furred": True, "kind": "return", "return_count": 1})
        self.assertEqual(line, "Max is back, Bret.")

    def test_return_without_guess_uses_ladder(self):
        _frame, line = C._animal_reaction_frame_and_line(
            {"species": "dog", "furred": True, "kind": "return", "return_count": 1})
        self.assertIn(line, C._ANIMAL_RETURN_LINES_FIRST)


class OwnerCandidatesTest(unittest.TestCase):
    def test_ladder_order_and_dedupe(self):
        with mock.patch.object(C.episodic_hooks, "_visible_known_people", return_value=[(4, "JT")]), \
             mock.patch.object(C, "get_recent_engagement", return_value={"person_id": 1, "name": "Bret"}), \
             mock.patch("memory.people.recently_seen_people",
                        return_value=[{"id": 1, "name": "Bret", "age_secs": 30.0},
                                      {"id": 5, "name": "Jennifer", "age_secs": 200.0}]):
            self.assertEqual(C._pet_owner_candidates(900.0),
                             [(4, "JT"), (1, "Bret"), (5, "Jennifer")])


class SpeciesMismatchNeverDisclaimsTest(_GuessCase):
    """Field 2026-08-27 13:35:47, on Bret's DOG: "Small furry lifeform... Bret,
    that's Max, right? Ignore the part of me that said cat." A contradicted
    species must not carry a name, must not poison the return lines, and must
    never voice Rex disclaiming his own detector."""

    def test_no_line_ever_disclaims_the_detector(self):
        pool = tuple(getattr(config, "ANIMAL_SPECIES_MISMATCH_ASK_LINES", ()) or ())
        self.assertTrue(pool)
        for line in pool:
            self.assertNotIn("{name}", line)
            self.assertNotIn("{species}", line)
            self.assertNotIn("ignore", line.lower())
            self.assertNotIn("classifier", line.lower())

    def test_two_contradicted_pets_still_only_ask(self):
        table = {1: [{"name": "Max", "species": "dog", "confidence": 0.6},
                     {"name": "Toby", "species": "dog", "confidence": 0.5}]}
        with self._owners((1, "Bret Benziger")), self._pets(table):
            line = C._pet_name_guess_line("cat")
        self.assertEqual(line, "Bret, who is this one?")
        self.assertNotIn("cat", C._animal_guessed_pet)

    def test_matching_species_is_untouched(self):
        with self._owners((1, "Bret Benziger")), self._pets({1: [{"name": "Toby", "species": "dog", "confidence": 0.6}]}):
            self.assertEqual(C._pet_name_guess_line("dog"), "Bret, is that Toby?")
        self.assertEqual(C._animal_guessed_pet["dog"][1], "Toby")
        self.assertEqual(C._animal_display_species("dog"), "dog")

    def test_non_furry_pet_still_declines_entirely(self):
        with self._owners((1, "Bret")), self._pets({1: [{"name": "Nemo", "species": "fish", "confidence": 0.9}]}):
            self.assertIsNone(C._pet_name_guess_line("cat"))
        self.assertNotIn("cat", C._animal_species_contradicted)


if __name__ == "__main__":
    unittest.main()
