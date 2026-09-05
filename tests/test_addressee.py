"""Addressee decision (Lean Brain 2B, third decision): was the line said TO Rex?

Field 2026-09-05 00:41 — Rex asked Bret "Which room is this?"; JT asked Bret
"Are you gonna watch both movies?"; the dialogue act bound JT's question as the
answer to Rex's and Rex answered it as his own. Covers: the deterministic hint,
the dialogue-act targeted-frame fix, the optional stay-quiet tool being attached
only when the hint asks, the Lean prompt line, and the silent executor.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import addressee as A


def _hint(text, **kw):
    base = dict(speaker_pid=2, speaker_known=True, speaker_uncertain=False,
                humans_in_window=1, engaged_pid=1, last_frame_target_pid=None,
                last_frame_target_name=None, last_frame_is_question=False)
    base.update(kw)
    return A.assess(text, **base)


class HintTest(unittest.TestCase):
    def test_one_on_one_is_always_to_rex(self):
        h = _hint("are you gonna watch both movies?", humans_in_window=1, speaker_pid=1)
        self.assertEqual(h.status, "to_rex")
        self.assertFalse(h.offer_stay_quiet)
        self.assertEqual(h.prompt_line(), "")

    def test_name_mention_is_to_rex_even_in_a_crowd(self):
        h = _hint("Rex, are you gonna watch both movies?", humans_in_window=3, speaker_pid=None,
                  speaker_known=False, last_frame_target_pid=1, last_frame_target_name="Bret")
        self.assertEqual(h.status, "to_rex")

    def test_command_is_to_rex(self):
        h = _hint("turn left", humans_in_window=2, command_parsed=True)
        self.assertEqual(h.status, "to_rex")

    def test_field_case_is_likely_side(self):
        # Rex just asked Bret (pid 1) a question; an unknown voice asks a question.
        h = _hint("Are you gonna watch both movies?", speaker_pid=None, speaker_known=False,
                  humans_in_window=2, engaged_pid=1, last_frame_target_pid=1,
                  last_frame_target_name="Bret", last_frame_is_question=True)
        self.assertEqual(h.status, "likely_side")
        self.assertTrue(h.offer_stay_quiet)
        line = h.prompt_line()
        self.assertIn("PROBABLY not said to you", line)
        self.assertIn("aimed at Bret", line)
        self.assertIn("conversation_stay_quiet", line)

    def test_two_humans_statement_is_uncertain(self):
        h = _hint("I think the second one is better.", speaker_pid=2, humans_in_window=2)
        self.assertEqual(h.status, "uncertain")
        self.assertIn("2 people have spoken recently", h.reasons[0])
        self.assertIn("may not have been said to you", h.prompt_line())

    def test_uncertain_speaker_alone_opens_the_question(self):
        h = _hint("what do you think?", speaker_uncertain=True, humans_in_window=1)
        self.assertEqual(h.status, "uncertain")

    def test_disabled_flag_makes_everything_to_rex(self):
        with mock.patch.object(config, "ADDRESSEE_JUDGMENT_ENABLED", False, create=True):
            h = _hint("Are you gonna watch both?", speaker_pid=None, speaker_known=False,
                      humans_in_window=2, last_frame_target_pid=1)
        self.assertEqual(h.status, "to_rex")


class DialogueActTargetedFrameTest(unittest.TestCase):
    def setUp(self):
        from intelligence import dialogue_act as DA
        DA.clear()
        self.addCleanup(DA.clear)
        self.DA = DA

    def test_third_party_question_is_not_bound_to_rex_question_for_another(self):
        DA = self.DA
        DA.note_rex_turn("Excellent choice. Cheating death and still a better plan.", source="memory_hint")
        DA.note_rex_turn("Which room is this?", source="lean_impulse", target_person_id=1,
                         target_name="Bret", expected_reply_types=["answer", "statement"])
        d = DA.classify("Are you gonna watch both movies?", {}, person_id=None)
        self.assertEqual(d.label, "general_chat")
        self.assertIn("aimed at someone else", d.reason)
        self.assertFalse(d.skip_action_router)

    def test_target_person_still_binds(self):
        DA = self.DA
        DA.note_rex_turn("Which room is this?", source="lean_impulse", target_person_id=1,
                         target_name="Bret", expected_reply_types=["answer", "statement"])
        d = DA.classify("This is the bedroom.", {}, person_id=1)
        self.assertEqual(d.label, "answer_to_rex")

    def test_untargeted_frame_binds_anyone(self):
        DA = self.DA
        DA.note_rex_turn("How was your weekend?", source="lean_impulse")
        d = DA.classify("It was pretty good actually.", {}, person_id=None)
        self.assertEqual(d.label, "answer_to_rex")


class OptionalToolTest(unittest.TestCase):
    def test_stay_quiet_only_when_requested(self):
        from intelligence import tool_router as TR
        names = lambda tools: {t["function"]["name"] for t in (tools or [])}
        with mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", True, create=True):
            self.assertNotIn("conversation_stay_quiet", names(TR.live_reply_tools()))
            self.assertIn("conversation_stay_quiet",
                          names(TR.live_reply_tools(optional={"conversation.stay_quiet"})))
        self.assertIn("conversation.stay_quiet", TR.live_actions())
        self.assertEqual(TR.resolve_tool_call("conversation_stay_quiet", "{}"),
                         ("conversation.stay_quiet", {}))


class LeanWiringTest(unittest.TestCase):
    def test_messages_carry_addressee_note(self):
        from intelligence import lean_brain as LB
        with (
            mock.patch.object(LB, "_persona", return_value="PERSONA"),
            mock.patch.object(LB, "_person_lines", return_value=[]),
            mock.patch.object(LB, "_scene_lines", return_value=[]),
            mock.patch.object(LB, "_context_lines", return_value=[]),
            mock.patch.object(LB, "_current_speaker_display", return_value="Guest"),
        ):
            msgs = LB._messages("Are you gonna watch both?", None, [], None,
                                addressee_note="ADDRESSEE CHECK: maybe not you.")
        self.assertIn("ADDRESSEE CHECK", msgs[0]["content"])

    def test_stream_reply_attaches_optional_tool_only_with_hint(self):
        from intelligence import lean_brain as LB, tool_router as TR
        seen = {}

        def fake_create(client, **kwargs):
            seen["extra"] = kwargs.get("extra")
            return iter([])

        hint = A.AddresseeHint("likely_side", ["two people"], "Bret")
        with (
            mock.patch.object(LB, "_messages", return_value=[{"role": "system", "content": "x"},
                                                             {"role": "user", "content": "y"}]),
            mock.patch("intelligence.connectivity.is_offline", return_value=False),
            mock.patch.object(LB.llm_compat, "create", side_effect=fake_create),
            mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", True, create=True),
        ):
            list(LB.stream_reply("y", None, addressee=hint))
            with_hint = {t["function"]["name"] for t in (seen["extra"] or {}).get("tools", [])}
            list(LB.stream_reply("y", None, addressee=None))
            without = {t["function"]["name"] for t in (seen["extra"] or {}).get("tools", [])}
        self.assertIn("conversation_stay_quiet", with_hint)
        self.assertNotIn("conversation_stay_quiet", without)


class ExecutorTest(unittest.TestCase):
    def test_stay_quiet_returns_empty_and_records(self):
        from intelligence import interaction as I
        I._tool_routed_path.clear()
        with (
            mock.patch("intelligence.decision_ledger.record") as rec,
            mock.patch.object(I, "_speak_blocking") as speak,
        ):
            out = I._execute_tool_routed_action("conversation.stay_quiet", {},
                                                "Are you gonna watch both movies?", None)
        self.assertEqual(out, "")
        speak.assert_not_called()
        rec.assert_called_once()
        self.assertEqual(rec.call_args.args[0], "stayed_quiet")
        self.assertEqual(I._consume_tool_routed_path(), "tool_router.conversation.stay_quiet")


class InteractionHintTest(unittest.TestCase):
    def test_assess_turn_addressee_field_case(self):
        from intelligence import interaction as I, dialogue_act as DA
        from memory import conversations as conv
        DA.clear(); conv.clear_transcript()
        self.addCleanup(DA.clear); self.addCleanup(conv.clear_transcript)
        conv.add_to_transcript("Bret Benziger", "The Wrath of Khan.")
        conv.add_to_transcript("Rex", "Excellent choice.")
        DA.note_rex_turn("Which room is this?", source="lean_impulse", target_person_id=1,
                         target_name="Bret")
        with (
            mock.patch.object(I, "_turn_speaker_uncertain", return_value=False),
            mock.patch.object(I.command_parser, "parse", return_value=None),
        ):
            hint = I._assess_turn_addressee(
                "Are you gonna watch both movies?", person_id=None, text_input=False,
                recent_engagement={"person_id": 1, "name": "Bret Benziger"})
        self.assertEqual(hint.status, "likely_side")
        self.assertEqual(hint.target_name, "Bret")

    def test_typed_input_is_to_rex(self):
        from intelligence import interaction as I
        hint = I._assess_turn_addressee("hello", person_id=None, text_input=True, recent_engagement=None)
        self.assertEqual(hint.status, "to_rex")


if __name__ == "__main__":
    unittest.main()
