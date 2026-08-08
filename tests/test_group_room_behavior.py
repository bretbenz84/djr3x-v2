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

    def test_bare_no_passes_while_rex_awaits_an_answer(self):
        # Field 2026-08-07 18:20: "did you land on one?" → "No." was eaten as
        # pet_only_command — the answer to Rex's own question.
        with mock.patch.object(
            I.consciousness, "is_waiting_for_response", return_value=True
        ):
            self.assertIsNone(I._pet_directed_speech("No."))
            self.assertIsNone(I._pet_directed_speech("Sit."))

    def test_bare_no_is_pet_directed_outside_the_window(self):
        with mock.patch.object(
            I.consciousness, "is_waiting_for_response", return_value=False
        ):
            self.assertEqual(I._pet_directed_speech("No."), "pet_only_command")

    def test_pet_name_branch_still_fires_inside_the_window(self):
        with mock.patch.object(
            I.consciousness, "is_waiting_for_response", return_value=True
        ):
            self.assertIsNotNone(I._pet_directed_speech("Max, come here."))


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
    def test_same_species_new_position_stays_quiet_while_present(self):
        """The protected behavior from the flat species cooldown, re-expressed in
        the presence-ledger model: a dog roaming the room (new positions, brief
        out-of-frame flicker) is ONE animal already remarked on — no re-announce.
        A REAL departure past the grace window followed by a return is the one
        thing that speaks again (as a return joke, not a re-announcement)."""
        from intelligence import consciousness as C
        saved_presence = dict(C._animal_presence)
        C._animal_presence.clear()
        C._pending_animal_arrivals.clear()
        try:
            now = time.monotonic()
            # Dog already present + announced this run.
            C._animal_presence["dog"] = {
                "present": True, "first_seen_at": now - 60, "last_seen_at": now,
                "departed_at": None, "return_count": 0,
                "remarks_spoken": 1, "last_remark_at": now - 60,
            }
            snapshot = {"animals": [{"species": "dog", "position": "left"}]}
            with mock.patch.object(C, "_last_snapshot", {"animals": []}):
                C._stage_animal_arrivals(snapshot)
            self.assertEqual(C._pending_animal_arrivals, {})
            # Real departure (past grace) then a sighting → the return bit fires.
            C._animal_presence["dog"]["present"] = False
            C._animal_presence["dog"]["departed_at"] = now - 300
            C._animal_presence["dog"]["last_remark_at"] = now - 300
            with mock.patch.object(C, "_last_snapshot", {"animals": []}):
                C._stage_animal_arrivals(snapshot)
            self.assertEqual(len(C._pending_animal_arrivals), 1)
            self.assertEqual(C._pending_animal_arrivals["dog"]["kind"], "return")
        finally:
            C._pending_animal_arrivals.clear()
            C._animal_presence.clear()
            C._animal_presence.update(saved_presence)


if __name__ == "__main__":
    unittest.main()
