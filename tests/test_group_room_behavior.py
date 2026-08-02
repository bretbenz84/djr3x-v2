"""
tests/test_group_room_behavior.py — 3-person room fixes (field 2026-08-02
13:48): pet-directed speech guard, group-chatter gating for KNOWN speakers,
and the species-level animal announce cooldown.
"""

import time
import unittest
from unittest import mock

from intelligence import interaction as I


class PetDirectedSpeechTest(unittest.TestCase):
    def test_field_cases_are_pet_directed(self):
        # "Come here, Max" fired motion_agency.request_come — the robot drove
        # at the speaker off a dog command.
        for text in (
            "Oh, that's a good one. Come here, Max. Max, come here. No, uh huh. "
            "My small furry first one.",
            "Max, go lay down. Go lay.",
            "Lay down. Lay down.",
            "Sit. Stay.",
            "Go lay down.",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(I._pet_directed_speech(text))

    def test_real_rex_commands_pass(self):
        for text in (
            "Come here.",
            "Rex, come here.",
            "Turn around.",
            "Turn to your left a little and then back up four feet.",
            "What's the weather like?",
            "I got Max a new collar yesterday",
        ):
            with self.subTest(text=text):
                self.assertIsNone(I._pet_directed_speech(text))

    def test_kill_switch(self):
        with mock.patch.object(
            I.config, "PET_DIRECTED_SPEECH_GUARD_ENABLED", False, create=True
        ):
            self.assertIsNone(I._pet_directed_speech("Lay down. Lay down."))


class GroupChatterDirectedEvidenceTest(unittest.TestCase):
    def test_directed_turns_get_replies(self):
        for text, kind in (
            ("Hey Rex, how does my ass look in these shorts?", "name_mention"),
            ("What's the weather like?", "weather_query"),
            ("Turn to your left.", "motion_command"),
            ("Come here.", "motion_command"),
            ("Can you tell me about recent news?", "second_person_ask"),
        ):
            with self.subTest(text=text):
                self.assertEqual(I._group_chatter_directed_evidence(text), kind)

    def test_cross_talk_is_listened_to_not_answered(self):
        for text in (
            "That's why they run around upstairs. They ain't got no AC.",
            "I hate Elon Musk.",
            "Little Dick Energy.",
            "I was actually talking to myself.",
            "Not, they didn't mind doing that with the Epstein files.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(I._group_chatter_directed_evidence(text))


class SplicedEchoCoverageTest(unittest.TestCase):
    def test_stale_plus_fresh_line_splice_rejected(self):
        # Field 13:56:14: "On my way. Brad, daringly specific..." — one line
        # 20s old (outside the 12s ratio window), one 10s old. The coverage
        # check uses its own wider window.
        with I._recent_rex_lines_lock:
            saved = list(I._recent_rex_lines)
            I._recent_rex_lines.clear()
            now = time.monotonic()
            I._recent_rex_lines.append(
                (I._normalize_echo_text("On my way."), now - 20))
            I._recent_rex_lines.append(
                (I._normalize_echo_text(
                    "Brad, daringly specific. Toss me a last name to go with it."),
                 now - 10))
        try:
            self.assertTrue(I._looks_like_own_echo(
                "On my way. Brad, daringly specific. Toss me a last name to "
                "go with it."))
        finally:
            with I._recent_rex_lines_lock:
                I._recent_rex_lines.clear()
                I._recent_rex_lines.extend(saved)


class AnimalSpeciesCooldownTest(unittest.TestCase):
    def test_same_species_new_position_stays_quiet_in_window(self):
        from intelligence import consciousness as C
        saved = dict(C._animal_species_reacted_at)
        C._animal_species_reacted_at.clear()
        C._pending_animal_arrivals.clear()
        try:
            C._animal_species_reacted_at["dog"] = time.monotonic()
            snapshot = {"animals": [{"species": "dog", "position": "left"}]}
            with mock.patch.object(C, "_last_snapshot", {"animals": []}):
                C._stage_animal_arrivals(snapshot)
            self.assertEqual(C._pending_animal_arrivals, {})
            # Window expired → the dog is announceable again.
            C._animal_species_reacted_at["dog"] = time.monotonic() - 9999
            with mock.patch.object(C, "_last_snapshot", {"animals": []}):
                C._stage_animal_arrivals(snapshot)
            self.assertEqual(len(C._pending_animal_arrivals), 1)
        finally:
            C._pending_animal_arrivals.clear()
            C._animal_species_reacted_at.clear()
            C._animal_species_reacted_at.update(saved)


if __name__ == "__main__":
    unittest.main()
