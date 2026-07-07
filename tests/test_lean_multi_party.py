"""Multi-party awareness in the lean brain (owner spec 2026-07-06).

Identity resolution labels transcript turns with real speaker names, but _messages
used to flatten every human into an anonymous "user" role — the model couldn't tell
Bret's lines from JT's and answered JT's questions as if Bret asked them. With 2+
distinct humans in the recent window: labeled history, named current turn, room block,
other-participant context. A 1-on-1 session carries none of that weight.
"""

import unittest
from unittest import mock

import config
from intelligence import lean_brain as LB


def _msgs(transcript, person_id=1, user_text="what do you think?", *,
          label_current=True, current_name="Bret Benziger"):
    with (
        mock.patch.object(config, "LEAN_MULTI_PARTY_ENABLED", True, create=True),
        mock.patch.object(config, "LEAN_BRAIN_TRANSCRIPT_TURNS", 8, create=True),
        mock.patch.object(LB, "_persona", return_value="PERSONA"),
        mock.patch.object(LB, "_person_lines", return_value=[]),
        mock.patch.object(LB, "_scene_lines", return_value=[]),
        mock.patch.object(LB, "_current_speaker_display",
                          return_value=current_name.split()[0]),
        mock.patch.object(LB, "_other_participant_lines", return_value=[]),
    ):
        return LB._messages(user_text, person_id, transcript, None,
                            label_current_speaker=label_current)


class MultiPartyMessagesTest(unittest.TestCase):
    TWO_PEOPLE = [
        {"speaker": "Bret Benziger", "text": "the wheels are almost done"},
        {"speaker": "Rex", "text": "Finally, mobility."},
        {"speaker": "JT", "text": "can it climb stairs?"},
    ]

    def test_two_humans_get_labeled_history_and_room_block(self):
        msgs = _msgs(self.TWO_PEOPLE)
        system = msgs[0]["content"]
        self.assertIn("MULTI-PERSON ROOM", system)
        self.assertIn("Bret", system)
        self.assertIn("JT", system)
        # history: humans labeled, Rex's own line untouched
        self.assertEqual(msgs[1]["content"], "Bret: the wheels are almost done")
        self.assertEqual(msgs[2]["content"], "Finally, mobility.")
        self.assertEqual(msgs[3]["content"], "JT: can it climb stairs?")
        # current turn names its speaker
        self.assertEqual(msgs[-1]["content"], "Bret: what do you think?")

    def test_room_block_names_the_current_speaker(self):
        msgs = _msgs(self.TWO_PEOPLE, current_name="JT")
        system = msgs[0]["content"]
        self.assertIn("speaking RIGHT NOW is JT", system)
        self.assertEqual(msgs[-1]["content"], "JT: what do you think?")

    def test_single_human_stays_unlabeled(self):
        solo = [
            {"speaker": "Bret Benziger", "text": "hello there"},
            {"speaker": "Rex", "text": "General greeting."},
        ]
        msgs = _msgs(solo)
        self.assertNotIn("MULTI-PERSON ROOM", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "hello there")       # no label
        self.assertEqual(msgs[-1]["content"], "what do you think?")

    def test_directive_final_message_is_never_labeled(self):
        msgs = _msgs(self.TWO_PEOPLE, label_current=False,
                     user_text="You see Bret — greet him warmly.")
        self.assertIn("MULTI-PERSON ROOM", msgs[0]["content"])    # room still described
        self.assertEqual(msgs[-1]["content"], "You see Bret — greet him warmly.")

    def test_guest_labels_prettified(self):
        transcript = [
            {"speaker": "Bret Benziger", "text": "someone else is here"},
            {"speaker": "unknown_voice_2", "text": "hello robot"},
        ]
        msgs = _msgs(transcript)
        self.assertEqual(msgs[2]["content"], "Guest 2: hello robot")

    def test_kill_switch_restores_flat_history(self):
        with mock.patch.object(config, "LEAN_MULTI_PARTY_ENABLED", False, create=True), \
             mock.patch.object(LB, "_persona", return_value="PERSONA"), \
             mock.patch.object(LB, "_person_lines", return_value=[]), \
             mock.patch.object(LB, "_scene_lines", return_value=[]):
            msgs = LB._messages("hi", 1, self.TWO_PEOPLE, None)
        self.assertNotIn("MULTI-PERSON ROOM", msgs[0]["content"])
        self.assertEqual(msgs[3]["content"], "can it climb stairs?")


class DisplaySpeakerTest(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(LB._display_speaker("Bret Benziger"), "Bret")
        self.assertEqual(LB._display_speaker("JT"), "JT")
        self.assertEqual(LB._display_speaker("unknown_voice_3"), "Guest 3")
        self.assertEqual(LB._display_speaker(""), "Guest")


if __name__ == "__main__":
    unittest.main()
