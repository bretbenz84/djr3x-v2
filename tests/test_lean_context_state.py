"""Lean Brain phases 1 / 2 / 2B (first slices) — conversation state reaches Lean.

- conversation_state: corrections, body-action outcomes keyed by seq, pending
  questions per target, session clear via topic_thread.clear().
- brain_context: arc line + widened transcript window (every turn the arc has not
  covered, capped), presence notes pass-through.
- topic_thread: the arc cursor is now the highest transcript turn_id covered;
  fixtures without turn_id still work (1-based fallback).
- lean_brain: _system_prompt and consider_initiating carry the context lines;
  _messages uses the widened window.
- semantic: one inline budget per retrieval; misses past the deadline fall back
  to keyword and are queued for background prewarm.
- plan_intent: the local qwen confirm is skipped under the Lean brain.
- interaction._note_voice_bearing: a failed read clears the stored bearing.
- motion_controller: refusals and done frames land in conversation_state.

No hardware, no network. Model calls are mocked at the dispatch seams.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import config


class ConversationStateTest(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_state as cs, dialogue_act
        cs.clear()
        dialogue_act.clear()
        self.cs = cs
        self.addCleanup(cs.clear)
        self.addCleanup(dialogue_act.clear)

    def test_corrections_render_and_expire(self):
        self.cs.note_correction("name", "Their name is JT, not Brad", person_id=3)
        lines = self.cs.render_lines(3)
        self.assertEqual(len(lines), 1)
        self.assertIn("CORRECTIONS", lines[0])
        self.assertIn("JT, not Brad", lines[0])
        with mock.patch.object(config, "CONVERSATION_STATE_CORRECTION_TTL_SECS", 0.0):
            time.sleep(0.01)
            self.assertEqual(self.cs.render_lines(3), [])

    def test_action_lifecycle_by_seq(self):
        self.cs.note_action_issued(41, "turn", "left 90°")
        self.assertIn("still in progress", self.cs.render_lines(None)[0])
        self.cs.note_action_result(41, "blocked")
        line = self.cs.render_lines(None)[0]
        self.assertIn("turn left 90°", line)
        self.assertIn("blocked", line)
        self.assertIn("obstacle", line)
        self.cs.note_action_result(999, "completed")   # unknown seq ignored
        self.assertIn("blocked", self.cs.render_lines(None)[0])

    def test_refusal_reason_is_spelled_out(self):
        self.cs.note_action_refused("turn", "swing_blocked")
        line = self.cs.render_lines(None)[0]
        self.assertIn("REFUSED", line)
        self.assertIn("sweep into something", line)
        self.cs.clear()
        self.cs.note_action_refused("move", "tof_fault")
        self.assertIn("depth sensing", self.cs.render_lines(None)[0])

    def test_pending_question_for_another_target_only(self):
        from intelligence import dialogue_act
        dialogue_act.note_rex_turn("JT, how was the game?", target_person_id=7,
                                   target_name="JT")
        # The current speaker IS the target → the incoming message answers it.
        self.assertEqual(self.cs.pending_question_lines(7), [])
        # Someone else is speaking → it stays pending for JT.
        lines = self.cs.pending_question_lines(3)
        self.assertEqual(len(lines), 1)
        self.assertIn("You asked JT", lines[0])
        self.assertIn("does not answer it for JT", lines[0])
        # Non-questions never render.
        dialogue_act.clear()
        dialogue_act.note_rex_turn("Nice hat.", target_person_id=7, target_name="JT")
        self.assertEqual(self.cs.pending_question_lines(3), [])

    def test_kill_switch(self):
        self.cs.note_correction("x", "anything")
        with mock.patch.object(config, "LEAN_CONTEXT_STATE_ENABLED", False):
            self.assertEqual(self.cs.render_lines(None), [])

    def test_topic_thread_clear_clears_state(self):
        from intelligence import topic_thread
        self.cs.note_correction("x", "anything")
        topic_thread.clear()
        self.assertEqual(self.cs.recent_corrections(), [])


class BrainContextTest(unittest.TestCase):
    def setUp(self):
        from intelligence import topic_thread
        topic_thread.clear()
        self.addCleanup(topic_thread.clear)

    def _rows(self, n, start=1):
        return [{"speaker": "Bret" if i % 2 else "Rex", "text": f"t{i}", "turn_id": i}
                for i in range(start, start + n)]

    def test_window_is_base_without_arc(self):
        from intelligence import brain_context as bc
        rows = self._rows(30)
        win = bc.transcript_window(rows, base_keep=8, max_keep=20)
        self.assertEqual([r["turn_id"] for r in win], list(range(23, 31)))

    def test_window_widens_to_uncovered_turns(self):
        from intelligence import brain_context as bc, topic_thread as tt
        rows = self._rows(30)
        with tt._arc_lock:
            tt._arc_summary = "Topics: wheels"
            tt._arc_cursor = 16
        win = bc.transcript_window(rows, base_keep=8, max_keep=20)
        self.assertEqual([r["turn_id"] for r in win], list(range(17, 31)))   # 14 turns
        with tt._arc_lock:
            tt._arc_cursor = 2
        win = bc.transcript_window(rows, base_keep=8, max_keep=20)
        self.assertEqual(len(win), 20)                                       # capped
        with tt._arc_lock:
            tt._arc_cursor = 29
        win = bc.transcript_window(rows, base_keep=8, max_keep=20)
        self.assertEqual(len(win), 8)                                        # never below base

    def test_arc_line_flattens_summary(self):
        from intelligence import brain_context as bc, topic_thread as tt
        self.assertEqual(bc.arc_lines(), [])
        with tt._arc_lock:
            tt._arc_summary = "Topics: wheels\nOpen threads: stairs"
        line = bc.arc_lines()[0]
        self.assertIn("Topics: wheels · Open threads: stairs", line)
        self.assertIn("recent message wins", line)

    def test_lines_include_presence_notes(self):
        from intelligence import brain_context as bc, consciousness
        with mock.patch.object(consciousness, "presence_notes",
                               return_value=["Bret is out of frame because YOU moved"],
                               create=True):
            out = bc.lines(1)
        self.assertTrue(any("YOU moved" in l for l in out))


class ArcTurnIdCursorTest(unittest.TestCase):
    def setUp(self):
        from intelligence import topic_thread
        topic_thread.clear()
        self.addCleanup(topic_thread.clear)

    def _refresh(self, transcript, returns="Topics: a\nShared: b\nMood: c\nUsed up: d\nOpen threads: e"):
        from intelligence import topic_thread as tt
        with (
            mock.patch.object(tt, "_arc_enabled", return_value=True),
            mock.patch.object(tt, "_arc_generate", return_value=returns),
            mock.patch("memory.conversations.get_session_transcript", return_value=transcript),
        ):
            return tt._arc_refresh_core()

    def test_cursor_is_highest_turn_id(self):
        from intelligence import topic_thread as tt
        rows = [{"speaker": "Bret", "text": "hi", "turn_id": 101},
                {"speaker": "Rex", "text": "yo", "turn_id": 102}]
        self.assertTrue(self._refresh(rows))
        self.assertEqual(tt.arc_covered_through(), 102)
        # Nothing new → no refresh.
        self.assertFalse(self._refresh(rows))
        rows.append({"speaker": "Bret", "text": "more", "turn_id": 103})
        self.assertTrue(self._refresh(rows))
        self.assertEqual(tt.arc_covered_through(), 103)

    def test_fixture_without_turn_ids_uses_positions(self):
        from intelligence import topic_thread as tt
        rows = [{"speaker": "Bret", "text": "hi"}, {"speaker": "Rex", "text": "yo"}]
        self.assertTrue(self._refresh(rows))
        self.assertEqual(tt.arc_covered_through(), 2)

    def test_reset_smaller_ids_restart_cursor(self):
        from intelligence import topic_thread as tt
        self.assertTrue(self._refresh([{"speaker": "Bret", "text": "hi", "turn_id": 50}]))
        # A transcript whose ids are all below the cursor is a reset under us.
        self.assertTrue(self._refresh([{"speaker": "Bret", "text": "new", "turn_id": 3}]))
        self.assertEqual(tt.arc_covered_through(), 3)


class LeanPromptCarriesContextTest(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_state as cs, topic_thread
        cs.clear()
        topic_thread.clear()
        self.addCleanup(cs.clear)
        self.addCleanup(topic_thread.clear)

    def test_system_prompt_has_context_lines(self):
        from intelligence import lean_brain as LB, conversation_state as cs
        cs.note_correction("name", "Their name is JT, not Brad")
        cs.note_action_refused("turn", "swing_blocked")
        with (
            mock.patch.object(LB, "_persona", return_value="PERSONA"),
            mock.patch.object(LB, "_person_lines", return_value=[]),
            mock.patch.object(LB, "_scene_lines", return_value=[]),
            mock.patch.object(LB, "_room_belief_lines", return_value=[]),
            mock.patch.object(LB, "_mood_lines", return_value=[]),
            mock.patch.object(LB, "_pride_lines", return_value=[]),
            mock.patch.object(LB, "_homie_lines", return_value=[]),
            mock.patch.object(LB, "_taste_lines", return_value=[]),
            mock.patch.object(LB, "_reaction_lines", return_value=[]),
            mock.patch.object(LB, "_cadence_lines", return_value=[]),
            mock.patch("intelligence.consciousness.presence_notes", return_value=[], create=True),
        ):
            prompt = LB._system_prompt(1, None, None, user_text="hey")
        self.assertIn("JT, not Brad", prompt)
        self.assertIn("REFUSED", prompt)

    def test_messages_use_widened_window(self):
        from intelligence import lean_brain as LB, topic_thread as tt
        rows = [{"speaker": "Bret" if i % 2 else "Rex", "text": f"t{i}", "turn_id": i}
                for i in range(1, 21)]
        with tt._arc_lock:
            tt._arc_summary = "Topics: x"
            tt._arc_cursor = 8
        with (
            mock.patch.object(config, "LEAN_BRAIN_TRANSCRIPT_TURNS", 8),
            mock.patch.object(config, "LEAN_BRAIN_TRANSCRIPT_TURNS_MAX", 20, create=True),
            mock.patch.object(LB, "_system_prompt", return_value="SYS"),
            mock.patch.object(LB, "_current_speaker_display", return_value="Bret"),
        ):
            msgs = LB._messages("now?", 1, rows, None)
        # system + 12 uncovered turns (9..20) + the new user turn
        self.assertEqual(len(msgs), 1 + 12 + 1)
        self.assertEqual(msgs[1]["content"], "t9")

    def test_impulse_system_content_carries_context(self):
        from intelligence import lean_brain as LB, conversation_state as cs
        cs.note_correction("presence", "They said they are still here")
        captured = {}

        def _fake_create(client, **kwargs):
            captured["messages"] = kwargs["messages"]
            return iter([])

        with (
            mock.patch.object(LB, "_persona", return_value="PERSONA"),
            mock.patch.object(LB, "_mood_lines", return_value=[]),
            mock.patch.object(LB, "_reaction_lines", return_value=[]),
            mock.patch.object(LB, "_cadence_lines", return_value=[]),
            mock.patch.object(LB, "_situation_block", return_value=""),
            mock.patch("intelligence.consciousness.presence_notes", return_value=[], create=True),
            mock.patch("intelligence.connectivity.is_offline", return_value=False),
            mock.patch.object(LB.llm_compat, "create", _fake_create),
        ):
            LB.consider_initiating(None, transcript=[], world=None)
        self.assertIn("still here", captured["messages"][0]["content"])


class SemanticBudgetTest(unittest.TestCase):
    def setUp(self):
        from memory import semantic
        semantic.reset_cache()
        self.addCleanup(semantic.reset_cache)

    def test_past_deadline_falls_back_and_queues_prewarm(self):
        from memory import semantic
        with (
            mock.patch.object(semantic, "_embed", side_effect=AssertionError("inline embed")),
            mock.patch.object(semantic, "_ensure_prewarm_worker") as worker,
            semantic.turn_budget(0.0),
        ):
            self.assertIsNone(semantic._embed_candidate("blue whales"))
        worker.assert_called()
        self.assertIn("blue whales", semantic._prewarm_pending)

    def test_within_budget_embeds_with_capped_timeout(self):
        import numpy as np
        from memory import semantic
        seen = {}

        def _fake_request(text, *, timeout=None):
            seen["timeout"] = timeout
            return np.array([1.0, 0.0], dtype=np.float32)

        with (
            mock.patch.object(semantic, "_request_embedding", side_effect=_fake_request),
            mock.patch.object(semantic, "_healthy", return_value=True),
            semantic.turn_budget(0.2),
        ):
            vec = semantic._topic_vector({"orcas"})
        self.assertIsNotNone(vec)
        self.assertLessEqual(seen["timeout"], 0.2)
        # Outside a budget the cap is gone and _embed(text) keeps its one-arg shape.
        with mock.patch.object(semantic, "_request_embedding", side_effect=_fake_request):
            semantic._embed("plain")
        self.assertIsNone(seen["timeout"])

    def test_no_budget_means_old_behavior(self):
        import numpy as np
        from memory import semantic
        with mock.patch.object(semantic, "_embed",
                               return_value=np.array([1.0, 0.0], dtype=np.float32)) as emb:
            semantic._embed_candidate("dolphins")
        emb.assert_called_once_with("dolphins")

    def test_retrieval_wraps_scoring_in_budget(self):
        from memory import retrieval
        with (
            mock.patch.object(config, "MEMORY_SEMANTIC_RECALL_ENABLED", True),
            mock.patch("memory.semantic.turn_budget") as budget,
            mock.patch("memory.facts.get_prompt_worthy_facts", return_value=[]),
            mock.patch("memory.interests.get_interests_for_prompt", return_value=[]),
        ):
            budget.return_value.__enter__ = lambda *a: None
            budget.return_value.__exit__ = lambda *a: False
            retrieval.retrieve_person_memory(1, topic_tokens={"x"})
        budget.assert_called_once()


class PlanIntentLeanGateTest(unittest.TestCase):
    def test_qwen_confirm_off_under_lean(self):
        from intelligence import plan_intent
        with (
            mock.patch.object(config, "PLAN_INTENT_QWEN_CONFIRM_ENABLED", True),
            mock.patch.object(config, "LEAN_BRAIN_ENABLED", True),
        ):
            self.assertFalse(plan_intent._qwen_confirm_enabled())
        with (
            mock.patch.object(config, "PLAN_INTENT_QWEN_CONFIRM_ENABLED", True),
            mock.patch.object(config, "LEAN_BRAIN_ENABLED", False),
        ):
            self.assertTrue(plan_intent._qwen_confirm_enabled())


class StaleBearingTest(unittest.TestCase):
    def test_failed_read_clears_previous_bearing(self):
        from intelligence import interaction as I
        I._last_voice_bearing = {"bearing_deg": 40.0, "at": time.monotonic(), "share": 0.9}
        self.addCleanup(setattr, I, "_last_voice_bearing", None)
        with (
            mock.patch("hardware.flex_doa.available", return_value=True),
            mock.patch("hardware.flex_doa.bearing_between", return_value=None),
        ):
            self.assertIsNone(I._note_voice_bearing(1.0, 2.0))
        self.assertIsNone(I._last_voice_bearing)
        self.assertIsNone(I._recent_voice_bearing())

    def test_good_read_is_stamped_with_utterance_start(self):
        from intelligence import interaction as I
        self.addCleanup(setattr, I, "_last_voice_bearing", None)
        res = {"bearing_deg": 10.0, "raw_deg": 80.0, "cluster_n": 5, "n": 6,
               "spread_deg": 4.0, "share": 0.8, "clusters": []}
        with (
            mock.patch("hardware.flex_doa.available", return_value=True),
            mock.patch("hardware.flex_doa.bearing_between", return_value=dict(res)),
            mock.patch("hardware.flex_doa.describe_clusters", return_value=""),
            mock.patch("intelligence.motion_agency.note_voice_bearing"),
        ):
            out = I._note_voice_bearing(5.0, 6.5)
        self.assertEqual(out["utterance_t0"], 5.0)
        self.assertIs(I._last_voice_bearing, out)


class MotionOutcomesRecordedTest(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_state as cs
        cs.clear()
        self.addCleanup(cs.clear)

    def test_suppressed_records_refusal(self):
        from intelligence import motion_controller as mc, conversation_state as cs
        with mock.patch.object(mc, "_user_commanded_fx", return_value=False):
            mc._suppressed("turn", "swing_blocked")
        acts = cs.recent_actions()
        self.assertEqual(acts[0]["status"], "refused")
        self.assertEqual(acts[0]["reason"], "swing_blocked")

    def test_done_frame_updates_issued_action(self):
        from intelligence import motion_controller as mc, conversation_state as cs
        mc._note_issued(77, "turn", "left 90°")
        with (
            mock.patch.object(mc, "_finish_swing_escape"),
            mock.patch.object(mc, "_fx_drive_loop_stop"),
            mock.patch.object(mc, "_fx"),
            mock.patch.object(mc, "_maybe_announce_blocked"),
            mock.patch.object(mc, "_handle_turn_verification_done"),
        ):
            mc._on_motion_done({"seq": 77, "result": "blocked"})
        self.assertEqual(cs.recent_actions()[0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
