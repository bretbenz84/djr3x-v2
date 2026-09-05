"""
The lull-cue drop-bench actually benches EVERY cue kind (review find 2026-08-05).

When a generated lull line is dropped (near-duplicate question, banned topic, bit
repeat), `_strike_lean_cue(kind)` benches that cue kind for
LEAN_CUE_DROP_COOLDOWN_SECS so the next consult reaches lower cues instead of
regenerating the same doomed line. But the bench is only real if the ladder actually
consults `_lean_cue_blocked(kind)` before offering the cue again — and originally
only the six top-tier cues did. For the seven lower-tier kinds the strike wrote a
cooldown nothing ever read, so the ladder cheerfully re-offered the benched cue on
the very next consult.

These tests drive `_maybe_lean_impulse` for each lower-tier kind and assert that a
benched kind's BUILDER IS NEVER CALLED (the checks sit before the builder runs, not
after — several builders arm their own pacing/probability state at lookup time, and a
benched lookup would silently spend it), and that an expired bench lets it run again.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import config
from intelligence import interaction, lean_brain
from tests._lean_impulse_state import reset_impulse_state

# Every lower-tier cue kind -> the builder the ladder calls for it. The kind strings
# must match the `_winning_kind` tuple in _maybe_lean_impulse — that is what
# _strike_lean_cue records, so a typo here would test a bench nothing writes.
_LOWER_TIER = {
    "place_question": "_lean_place_question_cue",
    "room_question": "_lean_room_question_cue",
    "visual_riff": "_lean_visual_riff_cue",
    "weekend_plans": "_lean_weekend_plans_cue",
    "interest_discovery": "_lean_interest_discovery_cue",
    "news_story": "_lean_news_cue",
    "memory_musing": "_lean_memory_musing_cue",
}

_ALL_BUILDERS = (
    "_lean_celebration_cue", "_lean_event_followup_cue", "_lean_open_thread_cue",
    "_lean_callback_lull_cue", "_lean_workday_checkin_cue",
    "_lean_mood_share_cue", *_LOWER_TIER.values(),
)


class DropBenchLadderTests(unittest.TestCase):

    def setUp(self) -> None:
        reset_impulse_state(self)

    def _drive(self, benched: str | None, builders: dict) -> dict:
        """Run one _maybe_lean_impulse consult with every gate opened, all builders
        stubbed to None except `builders` overrides. Returns {name: Mock}."""
        I = interaction
        mocks: dict = {}
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
            p(mock.patch.object(I, "_register_rex_utterance"))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person",
                                return_value=None))
            for name in _ALL_BUILDERS:
                ret = builders.get(name)
                mocks[name] = p(mock.patch.object(I, name, return_value=ret))
            p(mock.patch.object(lean_brain, "consider_initiating",
                                return_value="A perfectly serviceable line."))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            # time.monotonic is patched to 1000.0, so bench relative to that.
            if benched is not None:
                I._lean_cue_cooldowns[benched] = 1000.0 + 300.0
            I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        return mocks

    # The cue payloads only need to be truthy dicts — consider_initiating is stubbed.
    _PAYLOAD = {"anything": "truthy"}

    def test_a_benched_lower_tier_cue_builder_is_never_called(self):
        for kind, builder in _LOWER_TIER.items():
            with self.subTest(kind=kind):
                reset_impulse_state(self)
                mocks = self._drive(benched=kind, builders={builder: self._PAYLOAD})
                mocks[builder].assert_not_called()

    def test_an_unbenched_cue_builder_runs_and_wins(self):
        for kind, builder in _LOWER_TIER.items():
            with self.subTest(kind=kind):
                reset_impulse_state(self)
                mocks = self._drive(benched=None, builders={builder: self._PAYLOAD})
                mocks[builder].assert_called_once()

    def test_an_expired_bench_lets_the_cue_run_again(self):
        interaction._lean_cue_cooldowns["news_story"] = 999.0   # < mocked now (1000.0)
        mocks = self._drive(benched=None,
                            builders={"_lean_news_cue": self._PAYLOAD})
        mocks["_lean_news_cue"].assert_called_once()

    def test_benching_one_kind_lets_the_next_cue_through(self):
        # The whole point of the bench: a dropped news line must not block the lull —
        # the ladder should fall through to the diary musing instead.
        mocks = self._drive(
            benched="news_story",
            builders={"_lean_news_cue": self._PAYLOAD,
                      "_lean_memory_musing_cue": {"recap": "a thing happened"}},
        )
        mocks["_lean_news_cue"].assert_not_called()
        mocks["_lean_memory_musing_cue"].assert_called_once()

    def test_bench_kind_strings_match_the_winning_kind_tuple(self):
        # _strike_lean_cue writes whatever _winning_kind says; the ladder reads what
        # these tests say. A rename on either side silently disconnects them.
        import inspect
        # Phase 3: the ladder became _collect_lean_cue_candidates — every kind is
        # listed in its spec table and the bench is consulted (by kind) before the
        # builder runs; the spend site keys the same strings off `chosen`.
        src = inspect.getsource(interaction._collect_lean_cue_candidates)
        spend = inspect.getsource(interaction._maybe_lean_impulse)
        self.assertIn("_lean_cue_blocked(kind)", src)
        for kind in _LOWER_TIER:
            with self.subTest(kind=kind):
                self.assertIn(f'("{kind}",', src)
                self.assertIn(f'chosen == "{kind}"', spend)


if __name__ == "__main__":
    unittest.main()


class RejectedLineFeedbackTests(unittest.TestCase):
    """Dead-draw feedback (field 2026-08-19 22:51: the same blue-light bit was
    generated and culled twice in twenty seconds — paid for both times, spoken
    never, and with no winning cue there was nothing to bench). A post-generation
    cull now (a) pays the FULL pacing window and (b) feeds the culled line back
    into the next generation as a dead premise."""

    def setUp(self) -> None:
        reset_impulse_state(self)

    def _drive(self, *, line, bit_repeat, now=1000.0):
        I = interaction
        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.config, "PROACTIVE_LINE_MIN_GAP_SECS", 6.0))
            p(mock.patch.object(I.time, "monotonic", lambda: now))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
            p(mock.patch.object(I, "_lean_impulse_person_present", lambda pid: True))
            p(mock.patch.object(I, "_primary_session_person_id", return_value=7))
            p(mock.patch.object(I.speech_queue, "seconds_since_last_speech",
                                return_value=5.0))
            p(mock.patch.object(I.speech_queue, "is_speaking", return_value=False))
            p(mock.patch.object(I.consciousness, "is_waiting_for_response",
                                return_value=False))
            p(mock.patch.object(I.output_gate, "is_busy", return_value=False))
            p(mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False))
            p(mock.patch.object(I, "_suppress_proactive_after_heavy",
                                return_value=False))
            p(mock.patch.object(I.body_mood, "current_mood",
                                return_value=("neutral", 0.0)))
            p(mock.patch.object(I, "_lean_recent_transcript", return_value=[]))
            p(mock.patch.object(I, "_lean_world", return_value={}))
            p(mock.patch.object(I, "_line_duplicates_recent_question",
                                return_value=False))
            p(mock.patch.object(I.conv_memory, "add_to_transcript"))
            p(mock.patch.object(I.conv_log, "log_rex"))
            p(mock.patch.object(I, "_register_rex_utterance"))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person",
                                return_value=None))
            for name in _ALL_BUILDERS:
                p(mock.patch.object(I, name, return_value=None))
            p(mock.patch.object(I, "_bit_ledger_blocks", return_value=bit_repeat))
            ci = p(mock.patch.object(lean_brain, "consider_initiating",
                                     return_value=line))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))
            I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        return ci

    def test_bit_repeat_drop_files_a_dead_draw_and_pays_the_full_window(self):
        self._drive(line="The blue light is auditing your choices.", bit_repeat=True)
        self.assertIn("The blue light is auditing your choices.",
                      list(interaction._lean_rejected_lines))
        # Full pacing window from the drop — NOT a 20-second re-roll.
        self.assertEqual(interaction._last_lean_impulse_at, 1000.0)

    def test_next_generation_receives_the_dead_draws(self):
        self._drive(line="The blue light is auditing your choices.", bit_repeat=True)
        interaction._last_lean_impulse_at = 0.0     # cooldown elapsed
        ci = self._drive(line="Something fresh.", bit_repeat=False, now=2000.0)
        rejected = ci.call_args.kwargs.get("rejected_lines")
        self.assertIn("The blue light is auditing your choices.", rejected)

    def test_a_spoken_line_is_not_filed_as_a_dead_draw(self):
        self._drive(line="Something fresh.", bit_repeat=False)
        self.assertEqual(list(interaction._lean_rejected_lines), [])

    def test_dead_draws_are_accepted_and_pass_safe(self):
        # The public seam tolerates the new kwarg end-to-end (the LLM call inside
        # fails closed to "" in the test env — a PASS, never an exception).
        out = lean_brain.consider_initiating(
            7, transcript=[], world={}, quiet_secs=5.0,
            rejected_lines=["The blue light is auditing your choices."],
        )
        self.assertIsInstance(out, str)
