"""
The "he sits there quiet so much" package (owner 2026-08-05).

The lull gauntlet is ~15 gates deep, each added for a documented over-talking gripe —
individually defensible, multiplicatively brutal. Three targeted softenings plus the
telemetry to tune the rest from live data instead of guesswork:

  1. A model PASS re-arms only HALF the pacing window. The anchor arms on every
     consult (so the model isn't hammered), but PASS-praising instructions meant each
     polite shrug bought a FULL window of guaranteed silence — two chained PASSes on
     the 40s re-engage path was 2+ minutes of dead air with the person sitting there.
  2. +1 unanswered-line allowance when the person is VISIBLY on camera. Sitting in
     plain view is soft permission for one more try; the base cap stands for
     voice-only, where silence more plausibly means they left.
  3. Probe snooze 600s -> 240s (config-only; asserted here so a revert is loud).
  4. Every consult records ONE outcome (spoken / watched_pass / dropped_* / gate
     name); transitions log at INFO and a session summary answers "why was he quiet".
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import interaction as I
from tests._lean_impulse_state import reset_impulse_state


class PassRearmTests(unittest.TestCase):

    def setUp(self) -> None:
        reset_impulse_state(self)

    def _consult_reaching_pass(self, *, consecutive: int = 0, quiet: float = 60.0):
        """Drive one consult far enough to get a model PASS back, with a controlled
        clock, and return the resulting _last_lean_impulse_at anchor."""
        import contextlib
        from intelligence import lean_brain
        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_REENGAGE_SECS", 40.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 1000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
            p(mock.patch.object(I, "_lean_impulse_person_present", lambda pid: True))
            p(mock.patch.object(I, "_primary_session_person_id", return_value=7))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech",
                                return_value=quiet))
            p(mock.patch.object(I.speech_queue, "is_speaking", return_value=False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response",
                                return_value=False))
            p(mock.patch.object(I.output_gate, "is_busy", return_value=False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy", return_value=False))
            p(mock.patch.object(I.body_mood, "current_mood", return_value=("neutral", 0.0)))
            p(mock.patch.object(I, "_lean_recent_transcript", return_value=[]))
            p(mock.patch.object(I, "_lean_world", return_value={}))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person",
                                return_value=None))
            for name in ("_lean_celebration_cue", "_lean_event_followup_cue",
                         "_lean_open_thread_cue", "_lean_callback_lull_cue",
                         "_lean_workday_checkin_cue", "_lean_place_question_cue",
                         "_lean_room_question_cue", "_lean_visual_riff_cue",
                         "_lean_weekend_plans_cue", "_lean_interest_discovery_cue",
                         "_lean_mood_share_cue", "_lean_news_cue",
                         "_lean_memory_musing_cue"):
                p(mock.patch.object(I, name, return_value=None))
            p(mock.patch.object(lean_brain, "consider_initiating", return_value=""))
            I._consecutive_lean_impulses = consecutive
            I._last_lean_impulse_at = 0.0
            I._last_user_content_at = 0.0
            I._floor_held_until = 0.0
            I._last_proactive_line_at = 0.0
            self.assertFalse(I._maybe_lean_impulse(idle_for=5.0,
                                                   effective_idle_timeout=600.0))
            return I._last_lean_impulse_at

    def test_a_pass_rearms_only_half_the_reengage_window(self):
        # quiet=60 >= REENGAGE_SECS=40 → long-silence mode. Old behavior anchored at
        # now (next eligible consult in a full 40s); the fix backdates the anchor by
        # half the window so the next try comes at now+20.
        anchor = self._consult_reaching_pass(quiet=60.0)
        self.assertAlmostEqual(anchor, 1000.0 - 40.0 * 0.5, places=3)

    def test_a_pass_rearms_only_half_the_fast_cooldown(self):
        # quiet=10 < REENGAGE → fast mode, window = 12s * (1 + n). With n=0 the
        # anchor backdates by 6s.
        anchor = self._consult_reaching_pass(quiet=10.0)
        self.assertAlmostEqual(anchor, 1000.0 - 12.0 * 0.5, places=3)

    def test_fraction_one_restores_the_old_full_window(self):
        with mock.patch.object(config, "LEAN_IMPULSE_PASS_REARM_FRACTION", 1.0):
            anchor = self._consult_reaching_pass(quiet=60.0)
        self.assertAlmostEqual(anchor, 1000.0, places=3)

    def test_pass_outcome_is_counted(self):
        self._consult_reaching_pass(quiet=60.0)
        self.assertEqual(I._impulse_outcome_counts.get("watched_pass"), 1)


class VisibleBonusTests(unittest.TestCase):

    def setUp(self) -> None:
        reset_impulse_state(self)

    def test_person_visible_now_reads_world_state(self):
        with mock.patch.object(I.world_state, "get",
                               return_value=[{"person_db_id": 7}]):
            self.assertTrue(I._person_visible_now(7))
            self.assertFalse(I._person_visible_now(8))

    def test_person_visible_now_fails_closed(self):
        # It only grants a BONUS line, so a state hiccup must deny, not grant.
        with mock.patch.object(I.world_state, "get",
                               side_effect=RuntimeError("state gone")):
            self.assertFalse(I._person_visible_now(7))

    def test_the_unanswered_cap_wiring_grants_the_visible_bonus(self):
        # Source-level guard on the gate: base allowance + visible bonus, and the
        # outcome name the telemetry reports for it.
        import inspect
        src = inspect.getsource(I._maybe_lean_impulse)
        self.assertIn("LEAN_IMPULSE_MAX_UNANSWERED_VISIBLE_BONUS", src)
        self.assertIn("_person_visible_now(person_id)", src)
        self.assertIn('_impulse_blocked("unanswered_cap")', src)


class TelemetryTests(unittest.TestCase):

    def setUp(self) -> None:
        reset_impulse_state(self)

    def test_outcomes_accumulate_per_reason(self):
        for reason in ("cooldown", "cooldown", "watched_pass"):
            if reason == "watched_pass":
                I._impulse_outcome(reason)
            else:
                self.assertFalse(I._impulse_blocked(reason))
        self.assertEqual(I._impulse_outcome_counts["cooldown"], 2)
        self.assertEqual(I._impulse_outcome_counts["watched_pass"], 1)

    def test_only_transitions_log_at_info(self):
        # The consult loop ticks ~1/s; logging every repeat of the same gate would
        # flood the log. A CHANGE of blocking gate is the informative event.
        with mock.patch.object(I, "_log") as log:
            I._impulse_blocked("cooldown")
            I._impulse_blocked("cooldown")
            I._impulse_blocked("cooldown")
            I._impulse_blocked("flow")
        infos = [c for c in log.info.call_args_list
                 if "impulse blocked" in str(c.args[0])]
        self.assertEqual(len(infos), 2)     # cooldown once, flow once

    def test_session_summary_logs_sorted_counts_and_resets(self):
        I._impulse_blocked("cooldown")
        I._impulse_blocked("cooldown")
        I._impulse_outcome("spoken")
        with mock.patch.object(I, "_log") as log:
            I._log_impulse_session_summary()
        summary = str(log.info.call_args)
        self.assertIn("impulse session summary", summary)
        self.assertIn("cooldown=2", summary)
        self.assertIn("spoken=1", summary)
        self.assertEqual(I._impulse_outcome_counts, {})
        # An empty session logs nothing.
        with mock.patch.object(I, "_log") as log:
            I._log_impulse_session_summary()
        log.info.assert_not_called()

    def test_every_gate_records_an_outcome(self):
        # Source-level completeness guard: any bare `return False` inside the gate
        # region of _maybe_lean_impulse is a gate the telemetry can't see. The only
        # allowed bare returns are the enabled-flags short-circuit (not a lull
        # decision), the probe delegation, and the instrumented PASS/drop paths.
        import inspect
        src = inspect.getsource(I._maybe_lean_impulse)
        gate_region = src[: src.index("_winning_kind")].splitlines()
        bare = []
        for i, ln in enumerate(gate_region):
            stripped = ln.strip()
            if not stripped.startswith("return False"):
                continue        # skips the comment that merely MENTIONS return False
            context = "\n".join(gate_region[max(0, i - 2): i + 1])
            if "_impulse_blocked(" in context or "_impulse_outcome(" in context:
                continue        # instrumented on this or an adjacent line
            bare.append(gate_region[max(0, i - 1)].strip() or stripped)
        instrumented = sum("_impulse_blocked(" in ln for ln in gate_region)
        self.assertGreaterEqual(instrumented, 17)
        # The ONE allowed bare return is the enabled-flags short-circuit at the top —
        # feature off is not a lull decision worth counting.
        self.assertLessEqual(len(bare), 1, f"uninstrumented gates appeared: {bare}")


class ConfigDefaultsTests(unittest.TestCase):

    def test_probe_snooze_shortened(self):
        # 600s (10 min of total silence for missing one 30s window) was the harshest
        # single gate in the gauntlet. A revert should be a deliberate decision.
        self.assertEqual(float(config.ENGAGEMENT_PROBE_NO_ANSWER_SNOOZE_SECS), 240.0)

    def test_pass_rearm_fraction_is_a_real_fraction(self):
        frac = float(config.LEAN_IMPULSE_PASS_REARM_FRACTION)
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0)

    def test_visible_bonus_default(self):
        self.assertEqual(int(config.LEAN_IMPULSE_MAX_UNANSWERED_VISIBLE_BONUS), 1)


if __name__ == "__main__":
    unittest.main()
