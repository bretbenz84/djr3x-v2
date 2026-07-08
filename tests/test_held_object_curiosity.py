"""Person-oriented curiosity: the held-object remark + adaptive re-engage wait.

Two behaviors from the 2026-07-08 owner feedback ("comment on objects I'm holding
more often" + "continue the conversation sooner — humans don't like dead silence"):

  1. consciousness._step_held_object_remark — an event-driven "what's that you're
     drinking?" that fires once a near_person object has persisted in-hand, in a lull,
     bounded by de-dup + cooldown + a low session cap. Needs no room-model baseline.

  2. interaction._maybe_lean_impulse — the flow-quiet gate now SHORTENS when Rex's
     last line was a closed statement (the exchange stalled on him) vs a question
     (the floor-hold already governs that wait).
"""

import unittest
from unittest import mock

import config


class _Profile:
    def __init__(self, suppress_proactive=False, user_mid_sentence=False, interaction_busy=False):
        self.suppress_proactive = suppress_proactive
        self.user_mid_sentence = user_mid_sentence
        self.interaction_busy = interaction_busy


class StepHeldObjectRemarkTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness as c
        self.c = c
        self._reset()
        self.addCleanup(self._reset)

    def _reset(self):
        self.c._held_object_state["count"] = 0.0
        self.c._held_object_state["last_at"] = 0.0
        self.c._held_object_remarked.clear()
        self.c._held_object_first_seen.clear()

    def _run(self, objects, *, can_speak=True, profile=None, preseen_secs=None):
        """Run the step. preseen_secs pre-ages every near_person label's first-seen
        stamp so the MIN_HOLD persistence gate is already satisfied (the common case
        under test); pass None to exercise the just-appeared (not-yet-held) path."""
        c = self.c
        captured = {}
        if preseen_secs is not None:
            import time
            old = time.monotonic() - preseen_secs
            for o in objects:
                if isinstance(o, dict) and o.get("near_person") and o.get("label"):
                    c._held_object_first_seen[str(o["label"]).strip().lower()] = old
        with mock.patch.object(c, "_can_proactive_speak", return_value=can_speak), \
             mock.patch.object(c, "_generate_and_speak_presence",
                               side_effect=lambda prompt, **k: captured.update(prompt=prompt, kw=k) or True):
            c._step_held_object_remark({"objects": objects}, profile or _Profile())
        return captured

    def test_fires_for_a_persisted_held_object(self):
        out = self._run(
            [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
            preseen_secs=10.0,
        )
        self.assertIn("cup", out.get("prompt", ""))
        self.assertIn("Bret", out.get("prompt", ""))
        self.assertEqual(out["kw"].get("purpose"), "held_object_remark")
        self.assertEqual(self.c._held_object_state["count"], 1)
        self.assertIn("cup", self.c._held_object_remarked)

    def test_just_appeared_object_waits_for_min_hold(self):
        # First sighting this tick — not held long enough yet, so no remark.
        out = self._run([{"label": "cup", "near_person": True, "near_person_name": "Bret"}])
        self.assertNotIn("prompt", out)
        # ...but it IS now being tracked, so a later tick past MIN_HOLD will fire.
        self.assertIn("cup", self.c._held_object_first_seen)

    def test_background_object_never_fires(self):
        out = self._run([{"label": "chair", "position": "background"}], preseen_secs=10.0)
        self.assertNotIn("prompt", out)

    def test_already_remarked_label_skipped(self):
        self.c._held_object_remarked.add("cup")
        out = self._run(
            [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
            preseen_secs=10.0,
        )
        self.assertNotIn("prompt", out)

    def test_session_cap(self):
        self.c._held_object_state["count"] = float(config.HELD_OBJECT_REMARK_SESSION_CAP)
        out = self._run(
            [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
            preseen_secs=10.0,
        )
        self.assertNotIn("prompt", out)

    def test_cooldown_blocks_second_remark(self):
        import time
        self.c._held_object_state["last_at"] = time.monotonic()  # just fired
        out = self._run(
            [{"label": "book", "near_person": True, "near_person_name": "Bret"}],
            preseen_secs=10.0,
        )
        self.assertNotIn("prompt", out)

    def test_suppressed_when_not_a_lull(self):
        out = self._run(
            [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
            can_speak=False, preseen_secs=10.0,
        )
        self.assertNotIn("prompt", out)

    def test_user_mid_sentence_blocks(self):
        out = self._run(
            [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
            profile=_Profile(user_mid_sentence=True), preseen_secs=10.0,
        )
        self.assertNotIn("prompt", out)

    def test_kill_switch(self):
        with mock.patch.object(config, "HELD_OBJECT_REMARK_ENABLED", False, create=True):
            out = self._run(
                [{"label": "cup", "near_person": True, "near_person_name": "Bret"}],
                preseen_secs=10.0,
            )
        self.assertNotIn("prompt", out)

    def test_dropped_object_clears_first_seen(self):
        objs = [{"label": "cup", "near_person": True, "near_person_name": "Bret"}]
        self._run(objs)                       # cup tracked
        self.assertIn("cup", self.c._held_object_first_seen)
        self._run([{"label": "chair", "position": "bg"}])  # cup gone from frame
        self.assertNotIn("cup", self.c._held_object_first_seen)


class RegisterUtteranceQuestionFlagTest(unittest.TestCase):
    """_register_rex_utterance tracks whether Rex's last line handed the user a turn."""

    def setUp(self):
        from intelligence import interaction as I
        self.I = I

    def _register(self, text):
        # Isolate the flag write from the heavier side effects.
        with mock.patch.object(self.I, "repair_moves"), \
             mock.patch.object(self.I, "comedy_modes"), \
             mock.patch.object(self.I, "consciousness"), \
             mock.patch.object(self.I, "topic_thread", create=True):
            try:
                self.I._register_rex_utterance(text)
            except Exception:
                pass
        return self.I._last_rex_line_was_question

    def test_question_sets_flag_true(self):
        self.assertTrue(self._register("What are you drinking over there?"))

    def test_closed_statement_sets_flag_false(self):
        self.assertFalse(self._register("Good. Try not to make it a personality."))


class AdaptiveReengageWaitTest(unittest.TestCase):
    """The flow-quiet gate shortens after a dead-end statement so Rex bridges the
    awkward silence sooner, while still waiting the full window after a question."""

    def _impulse_reached_brain(self, *, last_was_question, quiet):
        """Drive _maybe_lean_impulse past its gates with a controlled `quiet` and
        return whether it reached lean_brain.consider_initiating (i.e. the flow gate
        let it through). All non-flow gates are neutralized."""
        from contextlib import ExitStack
        from intelligence import interaction as I
        import time

        I._interrupted.clear()
        with ExitStack() as es:
            p = es.enter_context
            for name, val in (
                ("LEAN_BRAIN_ENABLED", True), ("LEAN_IMPULSE_ENABLED", True),
                ("LEAN_IMPULSE_QUIET_SECS", 4.0), ("LEAN_IMPULSE_FLOW_WINDOW_SECS", 120.0),
                ("LEAN_IMPULSE_FLOW_QUIET_SECS", 14.0),
                ("LEAN_IMPULSE_FLOW_QUIET_AFTER_STATEMENT_SECS", 7.0),
                ("LEAN_IMPULSE_REENGAGE_SECS", 40.0),
            ):
                p(mock.patch.object(config, name, val, create=True))
            for name in ("_game_suppresses_conversation", "_directed_context_fresh",
                         "_proactive_line_recently_fired", "_suppress_proactive_after_heavy"):
                p(mock.patch.object(I, name, return_value=False))
            p(mock.patch.object(I, "_primary_session_person_id", return_value=1))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech", return_value=quiet))
            p(mock.patch.object(I.speech_queue, "is_speaking", return_value=False))
            p(mock.patch.object(I.output_gate, "is_busy", return_value=False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response", return_value=False))
            p(mock.patch.object(I, "_lean_recent_transcript", return_value=[]))
            p(mock.patch.object(I, "_lean_world", return_value={}))
            brain = p(mock.patch("intelligence.lean_brain.consider_initiating", return_value=""))
            I._last_user_content_at = time.monotonic()   # user spoke recently → flow window active
            I._floor_held_until = 0.0
            I._last_lean_impulse_at = 0.0
            I._consecutive_lean_impulses = 0
            I._last_rex_line_was_question = last_was_question
            I._maybe_lean_impulse(idle_for=quiet, effective_idle_timeout=90.0)
            return brain.called

    def test_statement_reengages_sooner(self):
        # 8s quiet: past the 7s after-statement threshold, under the 14s question one.
        self.assertTrue(self._impulse_reached_brain(last_was_question=False, quiet=8.0))

    def test_question_still_waits_the_full_window(self):
        self.assertFalse(self._impulse_reached_brain(last_was_question=True, quiet=8.0))

    def test_question_reengages_after_full_window(self):
        self.assertTrue(self._impulse_reached_brain(last_was_question=True, quiet=15.0))


if __name__ == "__main__":
    unittest.main()
