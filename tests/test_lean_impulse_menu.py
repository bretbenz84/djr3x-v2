"""Lean Brain phase 3 — bounded model choice among eligible lull cues.

Python still gates eligibility (builders, benches, low-energy / question-budget
rules); with two or more survivors the one impulse call sees them as a menu and
picks one, goes freeform, or PASSes. Covers: the candidate collector (order, cap,
family de-dupe, bench-before-builder, celebration offer count), the menu prompt
and its reply parsing, the spend of ONLY the chosen cue, the legacy single-cue
path, and the pre-playback revalidation. No network: the model call is stubbed.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import interaction as I, lean_brain as LB
from tests._lean_impulse_state import reset_impulse_state


def _stream(text):
    return [NS(choices=[NS(delta=NS(content=text))])]


class MenuReplyParsingTest(unittest.TestCase):
    CANDS = [{"kind": "celebration", "cue": {}}, {"kind": "news_story", "cue": {}},
             {"kind": "visual_riff", "cue": {}}]

    def test_choice_letter_maps_to_kind(self):
        line, kind = LB._parse_menu_reply("CHOICE: B\nDid you hear about the comet?", self.CANDS)
        self.assertEqual((line, kind), ("Did you hear about the comet?", "news_story"))

    def test_free_letter_is_freeform(self):
        line, kind = LB._parse_menu_reply("CHOICE: D\nThat lamp is judging us.", self.CANDS)
        self.assertEqual(kind, "freeform")
        self.assertEqual(line, "That lamp is judging us.")

    def test_pass_variants(self):
        self.assertEqual(LB._parse_menu_reply("CHOICE: PASS", self.CANDS), ("", None))
        self.assertEqual(LB._parse_menu_reply("PASS.", self.CANDS), ("", None))
        self.assertEqual(LB._parse_menu_reply("CHOICE: A\nPASS", self.CANDS), ("", None))
        self.assertEqual(LB._parse_menu_reply("CHOICE: A", self.CANDS), ("", None))

    def test_missing_header_is_freeform_line(self):
        line, kind = LB._parse_menu_reply("Nice hat, Bret.", self.CANDS)
        self.assertEqual((line, kind), ("Nice hat, Bret.", "freeform"))

    def test_lowercase_and_dash_header(self):
        line, kind = LB._parse_menu_reply("choice - c\n\"Still rocking the cape.\"", self.CANDS)
        self.assertEqual((line, kind), ("Still rocking the cape.", "visual_riff"))


class MenuInstructionTest(unittest.TestCase):
    def test_menu_lists_every_option_and_rules(self):
        cands = [
            {"kind": "celebration", "cue": {"description": "landed the promotion"}},
            {"kind": "news_story", "cue": {"headline": "Comet tonight", "summary": "visible at 9."}},
            {"kind": "open_thread", "cue": {"thread": "the interview", "when": "two days ago"}},
        ]
        with mock.patch.object(LB, "_choose_impulse_intent", return_value="scene"):
            text = LB._menu_instruction("Bret", "", cands, long_silence=False)
        self.assertIn("A) CELEBRATE", text)
        self.assertIn("landed the promotion", text)
        self.assertIn("B) BRING UP news", text)
        self.assertIn("C) PICK UP an unresolved thread", text)
        self.assertIn("D) Something fresh", text)
        self.assertIn("CHOICE: <letter>", text)
        self.assertIn("PASS", text)
        self.assertIn("fresh angles", text)

    def test_every_kind_renders(self):
        for kind in LB._KIND_TO_KWARG:
            with self.subTest(kind=kind):
                out = LB._render_option(kind, {"kind": "work"})
                self.assertTrue(out and kind.split("_")[0].upper()[:3] or True)
                self.assertIsInstance(out, str)
                self.assertGreater(len(out), 10)


class ConsiderInitiatingMenuTest(unittest.TestCase):
    def setUp(self):
        LB._last_choice_kind = None
        self.addCleanup(setattr, LB, "_last_choice_kind", None)

    def _run(self, reply, cands, **kw):
        captured = {}

        def fake_create(client, **kwargs):
            captured["messages"] = kwargs["messages"]
            captured["max_tokens"] = kwargs.get("max_tokens")
            return _stream(reply)

        with contextlib.ExitStack() as es:
            es.enter_context(mock.patch.object(LB.llm_compat, "create", side_effect=fake_create))
            es.enter_context(mock.patch("intelligence.connectivity.is_offline", return_value=False))
            es.enter_context(mock.patch.object(LB, "_situation_block", return_value=""))
            es.enter_context(mock.patch.object(LB, "_context_lines", return_value=[]))
            line = LB.consider_initiating(person_id=None, transcript=[], candidates=cands, **kw)
        return line, captured

    def test_two_candidates_use_the_menu_and_report_choice(self):
        cands = [{"kind": "news_story", "cue": {"headline": "H", "summary": "S"}},
                 {"kind": "visual_riff", "cue": {"cue": "the cape"}}]
        line, cap = self._run("CHOICE: B\nStill in the cape, I see.", cands)
        self.assertEqual(line, "Still in the cape, I see.")
        self.assertEqual(LB.last_choice_kind(), "visual_riff")
        self.assertIn("A) BRING UP news", cap["messages"][-1]["content"])
        self.assertEqual(cap["max_tokens"], config.LEAN_IMPULSE_MENU_MAX_TOKENS)

    def test_pass_on_menu(self):
        cands = [{"kind": "news_story", "cue": {}}, {"kind": "visual_riff", "cue": {}}]
        line, _ = self._run("CHOICE: PASS", cands)
        self.assertEqual(line, "")
        self.assertIsNone(LB.last_choice_kind())

    def test_single_candidate_uses_rich_template(self):
        cands = [{"kind": "news_story", "cue": {"headline": "Comet", "summary": "at 9."}}]
        line, cap = self._run("Did you hear about the comet?", cands)
        self.assertEqual(line, "Did you hear about the comet?")
        self.assertEqual(LB.last_choice_kind(), "news_story")
        self.assertIn("The story: Comet", cap["messages"][-1]["content"])
        self.assertNotIn("CHOICE:", cap["messages"][-1]["content"])

    def test_menu_disabled_uses_legacy_chain_not_the_menu(self):
        # With the menu off the collector offers ONE cue, so normally only one kwarg
        # arrives. If two do, the legacy elif chain decides (visual_riff sits above
        # news_story there) and no menu prompt is rendered.
        cands = [{"kind": "news_story", "cue": {"headline": "Comet", "summary": "at 9."}},
                 {"kind": "visual_riff", "cue": {"cue": "the cape"}}]
        with mock.patch.object(config, "LEAN_IMPULSE_MENU_ENABLED", False, create=True):
            line, cap = self._run("Still in the cape.", cands,
                                  news_story=cands[0]["cue"], visual_riff=cands[1]["cue"])
        self.assertEqual(LB.last_choice_kind(), "visual_riff")
        self.assertNotIn("CHOICE:", cap["messages"][-1]["content"])
        self.assertIn("the cape", cap["messages"][-1]["content"])

    def test_no_candidates_freeform_reports_freeform(self):
        line, _ = self._run("Nice hat.", [])
        self.assertEqual(line, "Nice hat.")
        self.assertEqual(LB.last_choice_kind(), "freeform")


class CollectorTest(unittest.TestCase):
    def setUp(self):
        reset_impulse_state(self)

    def _collect(self, builders: dict, **kw):
        names = {
            "celebration": "_lean_celebration_cue", "event_followup": "_lean_event_followup_cue",
            "open_thread": "_lean_open_thread_cue", "callback_premise": "_lean_callback_lull_cue",
            "workday_checkin": "_lean_workday_checkin_cue", "place_question": "_lean_place_question_cue",
            "room_question": "_lean_room_question_cue", "visual_riff": "_lean_visual_riff_cue",
            "weekend_plans": "_lean_weekend_plans_cue", "interest_discovery": "_lean_interest_discovery_cue",
            "mood_share": "_lean_mood_share_cue", "news_story": "_lean_news_cue",
            "memory_musing": "_lean_memory_musing_cue",
        }
        mocks = {}
        with contextlib.ExitStack() as es:
            for kind, fn in names.items():
                mocks[kind] = es.enter_context(mock.patch.object(I, fn, return_value=builders.get(kind)))
            mocks["holiday_plan"] = es.enter_context(mock.patch.object(
                I.consciousness, "_next_holiday_plan_for_person", return_value=builders.get("holiday_plan")))
            out = I._collect_lean_cue_candidates(
                7, world={}, transcript=[], long_silence=False,
                low_energy=kw.get("low_energy", False), no_questions=kw.get("no_questions", False))
        return out, mocks

    def test_order_cap_and_family_dedupe(self):
        out, _ = self._collect({
            "news_story": {"headline": "H"}, "workday_checkin": {"kind": "day"},
            "weekend_plans": {"when": "soon"}, "celebration": {"event_id": 5, "description": "d"},
            "visual_riff": {"cue": "c"},
        })
        kinds = [k for k, _ in out]
        # priority order, one personal_ask (workday beats weekend), capped at 3
        self.assertEqual(kinds, ["celebration", "workday_checkin", "visual_riff"])
        self.assertEqual(I._celebration_unvoiced_attempts.get(5), 1)

    def test_question_cues_off_when_low_energy(self):
        out, mocks = self._collect({"workday_checkin": {"kind": "day"}, "mood_share": {"label": "off"},
                                    "place_question": {"x": 1}}, low_energy=True)
        self.assertEqual([k for k, _ in out], ["mood_share"])
        mocks["workday_checkin"].assert_not_called()
        mocks["place_question"].assert_not_called()

    def test_bench_checked_before_builder(self):
        I._lean_cue_cooldowns["news_story"] = I.time.monotonic() + 600.0
        out, mocks = self._collect({"news_story": {"headline": "H"}, "memory_musing": {"recap": "r"}})
        self.assertEqual([k for k, _ in out], ["memory_musing"])
        mocks["news_story"].assert_not_called()

    def test_menu_disabled_offers_only_the_first(self):
        with mock.patch.object(config, "LEAN_IMPULSE_MENU_ENABLED", False, create=True):
            out, mocks = self._collect({"news_story": {"headline": "H"}, "memory_musing": {"recap": "r"}})
        self.assertEqual([k for k, _ in out], ["news_story"])
        mocks["memory_musing"].assert_not_called()


class SpendOnlyChosenTest(unittest.TestCase):
    """Drive _maybe_lean_impulse with two eligible cues and a stubbed model that
    picks the SECOND; only that cue's bookkeeping must run."""

    def setUp(self):
        reset_impulse_state(self)

    def _drive(self, chosen_kind):
        news = {"headline": "Comet", "summary": "at 9.", "topic": "space"}
        musing = {"recap": "we talked about wheels"}

        def fake_consider(*a, **k):
            LB._last_choice_kind = chosen_kind
            return "A perfectly fine line."

        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 1000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
            p(mock.patch.object(I, "_lean_impulse_person_present", lambda pid: True))
            p(mock.patch.object(I, "_primary_session_person_id", return_value=7))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech", return_value=5.0))
            p(mock.patch.object(I.speech_queue, "is_speaking", return_value=False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response", return_value=False))
            p(mock.patch.object(I.output_gate, "is_busy", return_value=False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy", return_value=False))
            p(mock.patch.object(I.body_mood, "current_mood", return_value=("neutral", 0.0)))
            p(mock.patch.object(I, "_lean_recent_transcript", return_value=[]))
            p(mock.patch.object(I, "_lean_world", return_value={}))
            p(mock.patch.object(I, "_line_duplicates_recent_question", return_value=False))
            p(mock.patch.object(I.conv_memory, "add_to_transcript"))
            p(mock.patch.object(I.conv_log, "log_rex"))
            register = p(mock.patch.object(I, "_register_rex_utterance"))
            for fn in ("_lean_celebration_cue", "_lean_event_followup_cue", "_lean_open_thread_cue",
                       "_lean_callback_lull_cue", "_lean_workday_checkin_cue",
                       "_lean_place_question_cue", "_lean_room_question_cue", "_lean_visual_riff_cue",
                       "_lean_weekend_plans_cue", "_lean_interest_discovery_cue", "_lean_mood_share_cue"):
                p(mock.patch.object(I, fn, return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=None))
            p(mock.patch.object(I, "_lean_news_cue", return_value=news))
            p(mock.patch.object(I, "_lean_memory_musing_cue", return_value=musing))
            mark = p(mock.patch("awareness.current_events.mark_mentioned"))
            p(mock.patch.object(LB, "consider_initiating", side_effect=fake_consider))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))
            fired = I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        return fired, mark, register

    def test_model_picks_musing_so_news_is_not_spent(self):
        fired, mark, _ = self._drive("memory_musing")
        self.assertTrue(fired)
        mark.assert_not_called()
        self.assertTrue(I._lean_memory_mused_this_session)
        self.assertFalse(I._lean_news_mentioned_this_session)

    def test_model_picks_news_so_musing_is_not_spent(self):
        fired, mark, register = self._drive("news_story")
        self.assertTrue(fired)
        mark.assert_called_once()
        self.assertTrue(I._lean_news_mentioned_this_session)
        self.assertFalse(I._lean_memory_mused_this_session)
        self.assertEqual(register.call_args.kwargs.get("topic"), "Comet")

    def test_freeform_spends_nothing(self):
        fired, mark, _ = self._drive("freeform")
        self.assertTrue(fired)
        mark.assert_not_called()
        self.assertFalse(I._lean_news_mentioned_this_session)
        self.assertFalse(I._lean_memory_mused_this_session)


class RevalidationTest(unittest.TestCase):
    def setUp(self):
        reset_impulse_state(self)

    def test_line_dropped_when_conversation_moved_on(self):
        revs = iter([10, 11])   # decision-time rev, then a newer one before speaking
        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 1000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
            p(mock.patch.object(I, "_lean_impulse_person_present", lambda pid: True))
            p(mock.patch.object(I, "_primary_session_person_id", return_value=7))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech", return_value=5.0))
            p(mock.patch.object(I.speech_queue, "is_speaking", return_value=False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response", return_value=False))
            p(mock.patch.object(I.output_gate, "is_busy", return_value=False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy", return_value=False))
            p(mock.patch.object(I.body_mood, "current_mood", return_value=("neutral", 0.0)))
            p(mock.patch.object(I, "_lean_recent_transcript", return_value=[]))
            p(mock.patch.object(I, "_lean_world", return_value={}))
            p(mock.patch.object(I, "_collect_lean_cue_candidates", return_value=[]))
            p(mock.patch.object(I.conv_memory, "last_turn_id", side_effect=lambda: next(revs)))
            p(mock.patch.object(LB, "consider_initiating", return_value="A line."))
            speak = p(mock.patch.object(I, "_speak_proactive", return_value=True))
            fired = I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        self.assertFalse(fired)
        speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
