"""Relationship-toned return + departure reactions.

Named arrivals already scale tone via _greeting_profile; the previously-flat RETURN and
DEPARTURE reactions now scale too via _presence_relationship_tone, which reuses the
reply path's llm._relationship_tone_rule (warmth/antagonism/tier) so a needling friend
gets a sharper send-off and a near-stranger stays plain.
"""

import unittest
from unittest import mock

from intelligence import consciousness as c


class PresenceRelationshipToneTest(unittest.TestCase):
    def _tone(self, person):
        with mock.patch("memory.people.get_person", return_value=person):
            return c._presence_relationship_tone(7)

    def test_non_int_id_returns_empty(self):
        self.assertEqual(c._presence_relationship_tone(None), "")
        self.assertEqual(c._presence_relationship_tone("person_1"), "")

    def test_disabled_flag_returns_empty(self):
        with mock.patch.object(c.config, "PRESENCE_RELATIONSHIP_TONE_ENABLED", False):
            self.assertEqual(
                self._tone({"name": "Dave", "antagonism_score": 0.9}), ""
            )

    def test_missing_person_returns_empty(self):
        with mock.patch("memory.people.get_person", return_value=None):
            self.assertEqual(c._presence_relationship_tone(7), "")

    def test_sparring_relationship_gets_a_sharper_directive(self):
        tone = self._tone({
            "name": "Dave", "warmth_score": 0.1, "antagonism_score": 0.6,
            "trust_score": 0.3, "friendship_tier": "friend",
        })
        self.assertNotEqual(tone, "")
        self.assertIn("needle", tone.lower())

    def test_close_friend_gets_a_warm_non_sparring_directive(self):
        tone = self._tone({
            "name": "Bret", "warmth_score": 0.85, "antagonism_score": 0.0,
            "trust_score": 0.7, "friendship_tier": "close_friend",
        })
        self.assertNotEqual(tone, "")
        self.assertNotIn("needle", tone.lower())

    def test_neutral_stranger_stays_plain(self):
        tone = self._tone({
            "name": "Sam", "warmth_score": 0.0, "antagonism_score": 0.0,
            "trust_score": 0.0, "friendship_tier": "stranger",
        })
        self.assertEqual(tone, "")


if __name__ == "__main__":
    unittest.main()
