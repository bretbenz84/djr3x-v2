"""Remembered event/plan follow-ups as a Lean-owned lull cue.

The old silence-fill `memory_followup` behavior ("how did the interview go?")
went dark under the lean brain (its purpose is suppressed). This revives it as a
cue into the single lull speaker, sourced NON-destructively from
`memory.events.get_pending_followups`, de-duped against the shared
`_fired_followup_event_ids`, and arming the normal awaiting-resolution loop so the
person's next reply closes the event. Upcoming-event anticipation is deliberately
NOT handled here (that lives in the greeting-time `_pick_anticipated_event` path).
"""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import interaction, lean_brain


def _one_chunk_stream(text):
    """A fake OpenAI stream yielding one delta with `text`."""
    return [NS(choices=[NS(delta=NS(content=text))])]


class _FollowupStateCase(unittest.TestCase):
    """Snapshot + reset the shared per-session follow-up state around each test."""

    def setUp(self):
        self._saved = (
            interaction._last_followup_exchange,
            interaction._last_followup_at,
            set(interaction._fired_followup_event_ids),
        )
        interaction._last_followup_exchange = -(10 ** 9)
        interaction._last_followup_at = 0.0
        interaction._fired_followup_event_ids = set()

    def tearDown(self):
        (
            interaction._last_followup_exchange,
            interaction._last_followup_at,
            interaction._fired_followup_event_ids,
        ) = self._saved


class EventFollowupCueSelectionTest(_FollowupStateCase):
    """`_lean_event_followup_cue`: pick one due, un-asked, named past plan — or None."""

    def _cue(self, pending, *, cadence=True, enabled=True, person_id=7):
        with mock.patch.object(interaction, "_memory_followup_cadence_allows", return_value=cadence), \
             mock.patch.object(interaction.events_memory, "get_pending_followups", return_value=pending), \
             mock.patch.object(config, "LEAN_EVENT_FOLLOWUP_ENABLED", enabled):
            return interaction._lean_event_followup_cue(person_id)

    def test_returns_first_eligible_past_event(self):
        cue = self._cue([
            {"id": 12, "event_name": "job interview", "event_date": "2026-07-01"},
            {"id": 13, "event_name": "dentist", "event_date": "2026-07-02"},
        ])
        self.assertEqual(
            cue,
            {"event_id": 12, "event_name": "job interview", "kind": "past", "dated": True},
        )

    def test_dated_flag_reflects_event_date_presence(self):
        dated = self._cue([{"id": 1, "event_name": "the gala", "event_date": "2026-07-01"}])
        self.assertTrue(dated["dated"])
        undated = self._cue([{"id": 2, "event_name": "redo the kitchen", "event_date": None}])
        self.assertFalse(undated["dated"])

    def test_skips_already_fired_events(self):
        interaction._fired_followup_event_ids = {12}
        cue = self._cue([
            {"id": 12, "event_name": "job interview", "event_date": "2026-07-01"},
            {"id": 13, "event_name": "dentist", "event_date": "2026-07-02"},
        ])
        self.assertEqual(cue["event_id"], 13)

    def test_skips_events_without_a_name(self):
        cue = self._cue([
            {"id": 12, "event_name": "", "event_date": "2026-07-01"},
            {"id": 13, "event_name": "  ", "event_date": "2026-07-02"},
            {"id": 14, "event_name": "art show", "event_date": "2026-07-03"},
        ])
        self.assertEqual(cue["event_id"], 14)

    def test_none_when_nothing_pending(self):
        self.assertIsNone(self._cue([]))

    def test_none_when_all_fired(self):
        interaction._fired_followup_event_ids = {12, 13}
        cue = self._cue([
            {"id": 12, "event_name": "job interview"},
            {"id": 13, "event_name": "dentist"},
        ])
        self.assertIsNone(cue)

    def test_none_when_cadence_disallows(self):
        # The shared FOLLOWUP_* clamp gates this cue too, so a lull follow-up can't
        # stack on top of a reactive one — and get_pending_followups isn't even read.
        with mock.patch.object(interaction.events_memory, "get_pending_followups") as getter:
            cue = self._cue([{"id": 12, "event_name": "x"}], cadence=False)
        self.assertIsNone(cue)
        getter.assert_not_called()

    def test_none_when_kill_switch_off(self):
        self.assertIsNone(self._cue([{"id": 12, "event_name": "x"}], enabled=False))

    def test_none_when_no_person(self):
        self.assertIsNone(self._cue([{"id": 12, "event_name": "x"}], person_id=None))

    def test_failsafe_none_on_db_error(self):
        with mock.patch.object(interaction, "_memory_followup_cadence_allows", return_value=True), \
             mock.patch.object(interaction.events_memory, "get_pending_followups",
                               side_effect=RuntimeError("db down")), \
             mock.patch.object(config, "LEAN_EVENT_FOLLOWUP_ENABLED", True):
            self.assertIsNone(interaction._lean_event_followup_cue(7))

    def test_skips_rows_with_unparseable_id(self):
        cue = self._cue([
            {"id": None, "event_name": "ghost plan"},
            {"id": "oops", "event_name": "typo plan"},
            {"id": 20, "event_name": "real plan"},
        ])
        self.assertEqual(cue["event_id"], 20)


class LeanEventFollowupInstructionTest(unittest.TestCase):
    """The lean instruction carries the remembered plan and demands one question."""

    def test_instruction_asks_how_it_went_without_menu(self):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("So — how'd the big job interview shake out?")

        cue = {"event_id": 12, "event_name": "job interview", "kind": "past"}
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(
                person_id=None, transcript=[], event_followup=cue,
            )
        self.assertTrue(line.endswith("?"))
        instruction = captured[0]
        self.assertIn("job interview", instruction)
        self.assertIn("how it went", instruction)
        self.assertIn("MUST ask the one question", instruction)
        # It is NOT the generic fresh-angles impulse.
        self.assertNotIn("fresh angles", instruction)

    def test_event_cue_beats_callback_and_visual_riff(self):
        # Priority in consider_initiating: holiday > event > callback > visual_riff.
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("How'd the recital go?")

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(
                person_id=None,
                transcript=[],
                event_followup={"event_id": 1, "event_name": "piano recital", "kind": "past"},
                callback_premise={"premise": "the haunted stapler"},
                visual_riff={"cue": "a familiar visual detail: red scarf"},
            )
        instruction = captured[0]
        self.assertIn("piano recital", instruction)
        self.assertNotIn("haunted stapler", instruction)
        self.assertNotIn("red scarf", instruction)

    def test_clause_falls_back_on_empty_name(self):
        clause = lean_brain._event_followup_clause({})
        self.assertIn("that thing they had going on", clause)
        self.assertIn("how it went", clause)


class SpokenEventFollowupWiringTest(_FollowupStateCase):
    """End-to-end through `_maybe_lean_impulse`: a spoken event follow-up arms the
    awaiting-resolution loop, purges the reactive queue, and uses the memory_followup
    frame. Mirrors the holiday-cue spoken test."""

    def _run_impulse(self, *, spoke=True, dup=False):
        I = interaction
        saved = {
            "last_user_content_at": I._last_user_content_at,
            "consecutive_lean_impulses": I._consecutive_lean_impulses,
            "last_lean_impulse_at": I._last_lean_impulse_at,
            "last_proactive_line_at": I._last_proactive_line_at,
            "floor_held_until": I._floor_held_until,
        }
        for k, v in saved.items():
            self.addCleanup(lambda k=k, v=v: setattr(I, f"_{k}", v))
        I._last_user_content_at = 0.0
        I._consecutive_lean_impulses = 0
        I._last_lean_impulse_at = 0.0
        I._last_proactive_line_at = 0.0
        I._floor_held_until = 0.0
        I._interrupted.clear()

        cue = {"event_id": 55, "event_name": "kitchen remodel", "kind": "past"}
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
            p(mock.patch.object(I, "_line_duplicates_recent_question", return_value=dup))
            p(mock.patch.object(I.conv_memory, "add_to_transcript"))
            p(mock.patch.object(I.conv_log, "log_rex"))
            # No celebration / holiday cue → event cue gets its turn.
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=None))
            p(mock.patch.object(I, "_lean_event_followup_cue", return_value=cue))
            register = p(mock.patch.object(I, "_register_rex_utterance"))
            arm = p(mock.patch.object(I, "set_awaiting_followup_event"))
            purge = p(mock.patch.object(I.consciousness, "_pending_followups_lock_remove"))
            # If the event cue wins, these later cues must never be consulted.
            cb = p(mock.patch.object(I, "_lean_callback_lull_cue", return_value=None))
            riff = p(mock.patch.object(I, "_lean_visual_riff_cue", return_value=None))
            p(mock.patch.object(
                lean_brain, "consider_initiating",
                return_value="How'd the kitchen remodel turn out?",
            ))
            p(mock.patch.object(I, "_speak_proactive", return_value=spoke))

            fired = I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        return cue, fired, register, arm, purge, cb, riff

    def test_spoken_event_followup_arms_resolution_and_dedup(self):
        cue, fired, register, arm, purge, cb, riff = self._run_impulse()
        self.assertTrue(fired)
        # Awaiting-resolution loop armed with the event, and the reactive queue purged.
        arm.assert_called_once_with(7, 55, "kitchen remodel")
        purge.assert_called_once_with(7, 55)
        # The event cue won → callback / visual-riff cues were never consulted.
        cb.assert_not_called()
        riff.assert_not_called()
        # Registered with the memory_followup FRAME (not the generic lean_impulse one).
        self.assertEqual(register.call_count, 1)
        _args, kwargs = register.call_args
        self.assertEqual(kwargs.get("source"), "memory_followup")
        self.assertEqual(kwargs.get("topic"), "kitchen remodel")
        self.assertIn("status_update", kwargs.get("expected_reply_types") or [])

    def test_not_spoken_marks_nothing(self):
        # If the line is never actually spoken (_speak_proactive False), the event must NOT
        # be armed, purged, registered, or added to the shared anti-repeat set — otherwise a
        # follow-up Rex never voiced would be silently "used up".
        _cue, fired, register, arm, purge, _cb, _riff = self._run_impulse(spoke=False)
        self.assertFalse(fired)
        arm.assert_not_called()
        purge.assert_not_called()
        register.assert_not_called()
        self.assertNotIn(55, interaction._fired_followup_event_ids)

    def test_dropped_duplicate_marks_nothing(self):
        # A line dropped as a recent-question duplicate is likewise never marked.
        _cue, fired, register, arm, purge, _cb, _riff = self._run_impulse(dup=True)
        self.assertFalse(fired)
        arm.assert_not_called()
        purge.assert_not_called()
        register.assert_not_called()
        self.assertNotIn(55, interaction._fired_followup_event_ids)

    def test_spoken_followup_makes_reactive_path_skip_the_same_event(self):
        # Integration of the no-double-ask invariant: let the REAL set_awaiting_followup_event
        # run, seed the consciousness in-memory queue with event 55, fire the lull cue, then
        # assert the reactive _post_response path can no longer surface 55 (it's both purged
        # from the queue AND in the shared _fired anti-repeat set).
        I = interaction
        # Seed the reactive queue as if a startup follow-up had queued event 55.
        I.consciousness.set_pending_followup(7, {"id": 55, "event_name": "kitchen remodel"})
        self.addCleanup(lambda: I.consciousness._pending_followups.pop(7, None))
        saved_await = I._awaiting_followup_event
        self.addCleanup(lambda: setattr(I, "_awaiting_followup_event", saved_await))

        cue = {"event_id": 55, "event_name": "kitchen remodel", "kind": "past"}
        saved = {
            "last_user_content_at": I._last_user_content_at,
            "consecutive_lean_impulses": I._consecutive_lean_impulses,
            "last_lean_impulse_at": I._last_lean_impulse_at,
            "last_proactive_line_at": I._last_proactive_line_at,
            "floor_held_until": I._floor_held_until,
        }
        for k, v in saved.items():
            self.addCleanup(lambda k=k, v=v: setattr(I, f"_{k}", v))
        I._last_user_content_at = 0.0
        I._consecutive_lean_impulses = 0
        I._last_lean_impulse_at = 0.0
        I._last_proactive_line_at = 0.0
        I._floor_held_until = 0.0
        I._interrupted.clear()

        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 1000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
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
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=None))
            p(mock.patch.object(I, "_lean_event_followup_cue", return_value=cue))
            p(mock.patch.object(lean_brain, "consider_initiating",
                                return_value="How'd the kitchen remodel turn out?"))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        self.assertIn(55, I._fired_followup_event_ids)
        # The reactive path pops the queue; 55 was purged, so it can't re-ask it.
        self.assertIsNone(I.consciousness.get_pending_followup(7))

    def test_non_event_impulse_clears_a_stale_awaiting_followup(self):
        # Regression (adversarial review): a NON-event lean line (here a visual riff) must
        # drop any armed _awaiting_followup_event so the user's reply to the riff can't
        # mis-close an earlier event follow-up.
        I = interaction
        saved_await = I._awaiting_followup_event
        self.addCleanup(lambda: setattr(I, "_awaiting_followup_event", saved_await))
        I._awaiting_followup_event = {"person_id": 7, "event_id": 99, "event_name": "old plan"}

        saved = {
            "last_user_content_at": I._last_user_content_at,
            "consecutive_lean_impulses": I._consecutive_lean_impulses,
            "last_lean_impulse_at": I._last_lean_impulse_at,
            "last_proactive_line_at": I._last_proactive_line_at,
            "floor_held_until": I._floor_held_until,
        }
        for k, v in saved.items():
            self.addCleanup(lambda k=k, v=v: setattr(I, f"_{k}", v))
        I._last_user_content_at = 0.0
        I._consecutive_lean_impulses = 0
        I._last_lean_impulse_at = 0.0
        I._last_proactive_line_at = 0.0
        I._floor_held_until = 0.0
        I._interrupted.clear()

        with contextlib.ExitStack() as es:
            p = es.enter_context
            p(mock.patch.object(I.config, "LEAN_BRAIN_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_ENABLED", True))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_QUIET_SECS", 4.0))
            p(mock.patch.object(I.config, "LEAN_IMPULSE_COOLDOWN_SECS", 12.0))
            p(mock.patch.object(I.time, "monotonic", lambda: 1000.0))
            p(mock.patch.object(I, "_game_suppresses_conversation", return_value=False))
            p(mock.patch.object(I, "_directed_context_fresh", return_value=False))
            p(mock.patch.object(I.end_thread, "is_grace_active", return_value=False))
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
            # No cue wins → generic impulse line (a non-event, non-followup line).
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=None))
            p(mock.patch.object(I, "_lean_event_followup_cue", return_value=None))
            p(mock.patch.object(I, "_lean_callback_lull_cue", return_value=None))
            p(mock.patch.object(I, "_lean_visual_riff_cue", return_value=None))
            p(mock.patch.object(lean_brain, "consider_initiating",
                                return_value="What's the best thing you ate this week?"))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        self.assertIsNone(I._awaiting_followup_event)


class UndatedEventClauseTest(unittest.TestCase):
    """A dateless aspiration must NOT be asserted to have happened."""

    def test_dated_event_asserts_it_happened(self):
        clause = lean_brain._event_followup_clause(
            {"event_name": "job interview", "dated": True}
        )
        self.assertIn("almost certainly happened", clause)
        self.assertIn("how it went", clause)

    def test_undated_event_does_not_assert_completion(self):
        clause = lean_brain._event_followup_clause(
            {"event_name": "redo the kitchen", "dated": False}
        )
        self.assertNotIn("almost certainly happened", clause)
        self.assertIn("if they ever got to it", clause)

    def test_defaults_to_dated_wording_when_flag_absent(self):
        clause = lean_brain._event_followup_clause({"event_name": "the trip"})
        self.assertIn("almost certainly happened", clause)

    def test_holiday_cue_takes_priority_over_event(self):
        I = interaction
        saved = (I._last_user_content_at, I._consecutive_lean_impulses,
                 I._last_lean_impulse_at, I._last_proactive_line_at, I._floor_held_until)

        def _restore():
            (I._last_user_content_at, I._consecutive_lean_impulses,
             I._last_lean_impulse_at, I._last_proactive_line_at, I._floor_held_until) = saved
        self.addCleanup(_restore)
        I._last_user_content_at = 0.0
        I._consecutive_lean_impulses = 0
        I._last_lean_impulse_at = 0.0
        I._last_proactive_line_at = 0.0
        I._floor_held_until = 0.0
        I._interrupted.clear()

        holiday = {"name": "Juneteenth", "when": "this Friday", "date": "2026-06-19"}
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
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=holiday))
            p(mock.patch.object(I.consciousness, "_mark_holiday_plan_asked"))
            event_cue = p(mock.patch.object(I, "_lean_event_followup_cue"))
            arm = p(mock.patch.object(I, "set_awaiting_followup_event"))
            # A line that reads as a response-expecting question, so the holiday
            # "must be a question" guard passes and we reach the mark step.
            p(mock.patch.object(lean_brain, "consider_initiating", return_value="Got plans for Juneteenth?"))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        # Holiday won → the event cue is never even looked up, and nothing is armed.
        event_cue.assert_not_called()
        arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
