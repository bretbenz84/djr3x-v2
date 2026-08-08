"""ASR-garbled surnames must match the real person, not mint a phantom.

Live incident 2026-08-08: Bret spoke while eating, his voice fell to an
unknown slot, and when Rex asked who he was, ASR heard "Bret Bender". The
whole-string SequenceMatcher ratio against "Bret Benziger" is 0.833 — 0.007
under the 0.84 names_are_similar bar — so every fuzzy tier and the
did-you-mean confirmation were skipped, and a clean trusted transcript would
have minted a duplicate person. The token-aware full_names_are_similar check
compares first name and surname separately to close that gap.
"""

import unittest

from memory.name_validation import full_names_are_similar


class GarbledSurnameMatcherTest(unittest.TestCase):
    def test_garbled_surname_matches(self):
        self.assertTrue(full_names_are_similar("Bret Bender", "Bret Benziger"))
        self.assertTrue(full_names_are_similar("Brett Benzinger", "Bret Benziger"))

    def test_genuinely_different_surnames_do_not_match(self):
        self.assertFalse(full_names_are_similar("Bret Smith", "Bret Jones"))
        self.assertFalse(full_names_are_similar("Exudica Royale", "Exudica Marbles"))

    def test_different_first_names_do_not_match(self):
        self.assertFalse(full_names_are_similar("John Bender", "Bret Benziger"))

    def test_single_token_and_identical_names_defer_to_other_tiers(self):
        # Single tokens are the first_name/fuzzy_first_name tiers' job; exact
        # matches are the exact tier's job. This check must stay out of the way.
        self.assertFalse(full_names_are_similar("Bret", "Bret Benziger"))
        self.assertFalse(full_names_are_similar("Bret Benziger", "Bret Benziger"))


class GarbledSurnameLookupTest(unittest.TestCase):
    def test_find_potential_person_match_returns_fuzzy(self):
        """The people-layer lookup must surface the near-miss as a 'fuzzy'
        match so find_or_create_person refuses silent creation and the
        did-you-mean confirmation flow fires."""
        from unittest import mock

        from memory import people

        stored = {
            "id": 1,
            "name": "Bret Benziger",
            "face_count": 2,
            "voice_count": 5,
            "visit_count": 13,
            "familiarity_score": 5.0,
        }
        with mock.patch.object(people, "find_person_by_name", return_value=None), \
                mock.patch.object(people, "_person_aliases_available", return_value=False), \
                mock.patch.object(people.db, "fetchall", return_value=[stored]):
            match = people.find_potential_person_match("Bret Bender")
        self.assertIsNotNone(match)
        self.assertEqual(match["match_type"], "fuzzy")
        self.assertEqual(match["person"]["name"], "Bret Benziger")

    def test_find_or_create_refuses_near_miss_creation(self):
        from unittest import mock

        from memory import people

        with mock.patch.object(
            people,
            "find_potential_person_match",
            return_value={
                "match_type": "fuzzy",
                "person": {"id": 1, "name": "Bret Benziger"},
                "candidate_name": "Bret Bender",
            },
        ), mock.patch.object(people, "enroll_person") as enroll:
            person_id, created = people.find_or_create_person("Bret Bender")
        self.assertIsNone(person_id)
        self.assertFalse(created)
        enroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
