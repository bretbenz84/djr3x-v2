"""Field log 2026-08-22-20-12: Joy answered Rex's who's-that ask with
"This is Exudica, A.K.A. Joy." and stayed unknown_voice_1 for the whole visit,
while Bret's 3-word "Yeah, that's Max" minted unknown_voice_2.

Three fixes pinned here:
  * an "X, a.k.a. Y" self-introduction parses as name X with alias Y;
  * the challenged unknown voice answering with its own name is NOT a third-party
    introduction by the last engaged person;
  * a person-linked voice signature resolves at the warm bar while that person
    has been a confident speaker in this session.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import interaction


class AkaAliasSplitTest(unittest.TestCase):
    def test_field_utterance_splits_into_name_and_alias(self):
        head, alias = interaction._split_aka_alias("This is Exudica, A.K.A. Joy.")
        self.assertEqual(head, "This is Exudica.")
        self.assertEqual(alias, "Joy")

    def test_also_known_as_phrasing(self):
        head, alias = interaction._split_aka_alias("I'm Joy, also known as Exudica")
        self.assertEqual(head, "I'm Joy.")
        self.assertEqual(alias, "Exudica")

    def test_no_alias_tail_is_untouched(self):
        self.assertEqual(interaction._split_aka_alias("This is JT"), ("This is JT", None))
        self.assertEqual(interaction._split_aka_alias(""), ("", None))

    def test_challenged_self_id_sees_through_the_alias_tail(self):
        self.assertEqual(
            interaction._challenged_self_identified_name("This is Exudica, A.K.A. Joy."),
            "Exudica",
        )
        self.assertIsNone(interaction._challenged_self_identified_name("This is amazing"))


class ChallengedVoiceAnswersItselfTest(unittest.TestCase):
    def setUp(self):
        self._saved = interaction._pending_offscreen_identify
        interaction._pending_offscreen_identify = {
            "asked_at": time.monotonic(),
            "anonymous_speaker_label": "unknown_voice_1",
        }

    def tearDown(self):
        interaction._pending_offscreen_identify = self._saved

    def test_name_only_self_intro_from_challenged_voice_is_an_answer(self):
        self.assertTrue(interaction._challenged_voice_is_answering_whos_that(
            "This is Exudica, A.K.A. Joy.", "unknown_voice_1"))

    def test_relationship_intro_is_still_an_introduction(self):
        self.assertFalse(interaction._challenged_voice_is_answering_whos_that(
            "This is my brother Wade", "unknown_voice_1"))

    def test_a_different_anonymous_voice_is_not_the_answerer(self):
        self.assertFalse(interaction._challenged_voice_is_answering_whos_that(
            "This is JT", "unknown_voice_2"))

    def test_no_pending_ask_means_no_guard(self):
        interaction._pending_offscreen_identify = None
        self.assertFalse(interaction._challenged_voice_is_answering_whos_that(
            "This is JT", "unknown_voice_1"))

    def test_expired_ask_means_no_guard(self):
        interaction._pending_offscreen_identify["asked_at"] = time.monotonic() - 10_000
        self.assertFalse(interaction._challenged_voice_is_answering_whos_that(
            "This is JT", "unknown_voice_1"))


class PersonWarmSignatureResolveTest(unittest.TestCase):
    COLD_SEEN = "2026-08-01T00:00:00+00:00"

    def setUp(self):
        self._saved = dict(interaction._last_confident_voice_at)
        interaction._last_confident_voice_at.clear()

    def tearDown(self):
        interaction._last_confident_voice_at.clear()
        interaction._last_confident_voice_at.update(self._saved)

    def test_cold_signature_of_a_recently_confident_speaker_resolves(self):
        # Bret: matched 0.89 on his prints five minutes ago, 0.760 on his linked
        # signature now (the field shape).
        interaction._last_confident_voice_at[1] = time.monotonic() - 300
        with mock.patch.object(config, "VOICE_SIGNATURE_RESOLVE_PERSON_MIN_SCORE", 0.85, create=True), \
             mock.patch.object(config, "VOICE_SIGNATURE_RESOLVE_WARM_MIN_SCORE", 0.70, create=True):
            self.assertTrue(interaction._signature_resolves_to_person(
                0.760, self.COLD_SEEN, person_id=1))

    def test_cold_signature_of_a_silent_person_still_needs_the_strict_bar(self):
        with mock.patch.object(config, "VOICE_SIGNATURE_RESOLVE_PERSON_MIN_SCORE", 0.85, create=True), \
             mock.patch.object(config, "VOICE_SIGNATURE_RESOLVE_WARM_MIN_SCORE", 0.70, create=True):
            self.assertFalse(interaction._signature_resolves_to_person(
                0.760, self.COLD_SEEN, person_id=3))

    def test_person_warmth_expires_with_the_warm_window(self):
        interaction._last_confident_voice_at[1] = time.monotonic() - 5_000
        with mock.patch.object(config, "VOICE_SIGNATURE_WARM_WINDOW_SECS", 900.0, create=True):
            self.assertFalse(interaction._signature_resolves_to_person(
                0.760, self.COLD_SEEN, person_id=1))

    def test_below_warm_bar_never_resolves(self):
        interaction._last_confident_voice_at[1] = time.monotonic()
        self.assertFalse(interaction._signature_resolves_to_person(
            0.65, self.COLD_SEEN, person_id=1))


if __name__ == "__main__":
    unittest.main()
