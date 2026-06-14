"""
Fixes from the 2026-06-14 15:54 run:
  A. A LAMP was misdetected as a "bird" (≥0.45 for ~19s) and made Rex announce a
     phantom "creature cameo", then churned the governor ~100s. Exotic species now
     need a higher confidence bar, and arrivals require persistence across scans.
  B. A departure fired while a distracted user (camera turned away, gone quiet) was
     still present. The confirm grace was lengthened.
  C. "it came out so good." (a real on-topic line about a photo) was dropped as
     background crosstalk because the regex matched "came out". Removed that pattern;
     the engaged 1-on-1 partner's lines are also never dropped as crosstalk.
  D. analyze_sentiment crashed on an empty/non-JSON reply, silently losing the turn's
     sentiment. JSON output is now forced and parsed leniently.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config


class ExoticAnimalThresholdTest(unittest.TestCase):
    def test_companions_keep_base_bar_exotics_need_more(self):
        from vision import animal_detector as ad
        base = float(config.LOCAL_ANIMAL_DETECTION_SCORE_THRESHOLD)
        exotic = float(config.LOCAL_ANIMAL_EXOTIC_SCORE_THRESHOLD)
        self.assertEqual(ad._accept_threshold_for("dog"), base)
        self.assertEqual(ad._accept_threshold_for("cat"), base)
        self.assertEqual(ad._accept_threshold_for("bird"), max(base, exotic))
        self.assertGreater(ad._accept_threshold_for("bird"), ad._accept_threshold_for("dog"))


class AnimalPersistenceTest(unittest.TestCase):
    def setUp(self):
        from vision import scene
        self.scene = scene
        scene._animal_confirm_streak.clear()

    def tearDown(self):
        self.scene._animal_confirm_streak.clear()

    def test_single_scan_is_not_confirmed(self):
        out = self.scene._confirm_persistent_animals([{"species": "bird"}])
        self.assertEqual(out, [])  # need >= 2 consecutive scans

    def test_consecutive_scans_confirm(self):
        self.scene._confirm_persistent_animals([{"species": "dog"}])
        out = self.scene._confirm_persistent_animals([{"species": "dog"}])
        self.assertEqual([a["species"] for a in out], ["dog"])

    def test_flicker_never_confirms(self):
        for animals in ([{"species": "bird"}], [], [{"species": "bird"}], []):
            out = self.scene._confirm_persistent_animals(animals)
            self.assertEqual(out, [])  # the gap resets the streak every time


class CrosstalkPartnerGuardTest(unittest.TestCase):
    def test_came_out_is_not_crosstalk(self):
        from intelligence import interaction as ix
        self.assertFalse(ix._looks_like_background_crosstalk("it came out so good."))

    def test_engaged_partner_turn_bypasses_crosstalk(self):
        from intelligence import interaction as ix
        with mock.patch.object(ix, "_primary_session_person_id", return_value=1), \
             mock.patch.object(ix, "_session_exchange_count", 3):
            self.assertTrue(ix._is_engaged_partner_turn(1, 0.66))   # confident partner
            self.assertFalse(ix._is_engaged_partner_turn(2, 0.66))  # not the partner
            self.assertFalse(ix._is_engaged_partner_turn(1, 0.20))  # too low to be sure

    def test_no_active_conversation_keeps_guard_strict(self):
        from intelligence import interaction as ix
        with mock.patch.object(ix, "_primary_session_person_id", return_value=1), \
             mock.patch.object(ix, "_session_exchange_count", 0):
            self.assertFalse(ix._is_engaged_partner_turn(1, 0.9))


class DepartureGraceTest(unittest.TestCase):
    def test_departure_confirm_grace_was_lengthened(self):
        # Was 20s; a distracted, camera-turned-away user was still present when it fired.
        self.assertGreaterEqual(config.PRESENCE_DEPARTURE_CONFIRM_SECS, 35.0)


class LenientJsonTest(unittest.TestCase):
    def test_plain_fenced_and_prose(self):
        from intelligence import llm
        self.assertEqual(llm._lenient_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(llm._lenient_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm._lenient_json_object('sure: {"a": 1} done'), {"a": 1})

    def test_empty_and_garbage_return_none(self):
        from intelligence import llm
        self.assertIsNone(llm._lenient_json_object(""))
        self.assertIsNone(llm._lenient_json_object("not json at all"))

    def test_analyze_sentiment_survives_empty_reply(self):
        from intelligence import llm
        fake = mock.Mock()
        fake.choices = [mock.Mock(message=mock.Mock(content=""))]
        with mock.patch.object(llm._client.chat.completions, "create", return_value=fake):
            result = llm.analyze_sentiment("the photo came out great")
        # No crash; defaults returned.
        self.assertFalse(result["is_insult"])
        self.assertEqual(result["emotion_detected"], "neutral")


if __name__ == "__main__":
    unittest.main()
