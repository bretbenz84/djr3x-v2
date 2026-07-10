"""Remembered good-news / celebration check-ins as a Lean-owned lull cue.

The lean rework broke a symmetry: hard-event / negative-affect check-ins (purpose
`emotional_checkin`) kept firing, but the positive branch (`celebration_checkin`)
was suppressed — so Rex would console bad news yet silently drop good news. This
revives celebrations as the TOP lull cue, faithful to the legacy
`_step_emotional_checkin` Trigger A2: sourced from
`emotional_events.get_due_celebrations`, gated by the shared per-session
`_emotional_checkin_fired` set, marking the event acknowledged + logging the rex.db
episode on speak.
"""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import consciousness, interaction, lean_brain


def _one_chunk_stream(text):
    """A fake OpenAI stream yielding one delta with `text`."""
    return [NS(choices=[NS(delta=NS(content=text))])]


class CelebrationCueSelectionTest(unittest.TestCase):
    """`_lean_celebration_cue`: pick one due, un-consoled celebration — or None."""

    def setUp(self):
        # Snapshot + clear the shared once-per-session emotional-check-in gate + attempt cap.
        self._saved_fired = set(consciousness._emotional_checkin_fired)
        self._saved_attempts = dict(interaction._celebration_unvoiced_attempts)
        consciousness._emotional_checkin_fired.clear()
        interaction._celebration_unvoiced_attempts.clear()
        self.addCleanup(lambda: (
            consciousness._emotional_checkin_fired.clear(),
            consciousness._emotional_checkin_fired.update(self._saved_fired),
            interaction._celebration_unvoiced_attempts.clear(),
            interaction._celebration_unvoiced_attempts.update(self._saved_attempts),
        ))

    def _cue(self, celebrations, *, person_id=7, enabled=True, empathy=True, proactive=True, crowd=1):
        with mock.patch.object(config, "LEAN_CELEBRATION_CHECKIN_ENABLED", enabled), \
             mock.patch.object(config, "EMPATHY_ENABLED", empathy), \
             mock.patch.object(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", proactive), \
             mock.patch.object(interaction, "_current_crowd_count", return_value=crowd), \
             mock.patch.object(interaction.emotional_events, "get_due_celebrations",
                               return_value=celebrations), \
             mock.patch.object(interaction.people_memory, "get_person",
                               return_value={"name": "Bret Benziger"}):
            return interaction._lean_celebration_cue(person_id)

    def test_returns_first_due_celebration(self):
        cue = self._cue([
            {"id": 4, "category": "career", "description": "got the new job", "valence": 0.9},
            {"id": 5, "category": "family", "description": "new puppy", "valence": 0.7},
        ])
        self.assertEqual(cue["event_id"], 4)
        self.assertEqual(cue["description"], "got the new job")
        self.assertEqual(cue["category"], "career")
        self.assertEqual(cue["first_name"], "Bret")
        self.assertEqual(cue["person_name"], "Bret Benziger")

    def test_skips_celebrations_without_a_description(self):
        cue = self._cue([
            {"id": 4, "category": "career", "description": "", "valence": 0.9},
            {"id": 5, "category": "family", "description": "  ", "valence": 0.7},
            {"id": 6, "category": "milestone", "description": "ran a marathon", "valence": 0.8},
        ])
        self.assertEqual(cue["event_id"], 6)

    def test_none_when_nothing_due(self):
        self.assertIsNone(self._cue([]))

    def test_suppressed_in_a_crowd(self):
        # Good news can be private (pregnancy/engagement) — don't announce it in a group,
        # mirroring the bad-news console path's crowd discretion. get_due_celebrations is
        # not even queried once the crowd guard trips.
        with mock.patch.object(config, "LEAN_CELEBRATION_CHECKIN_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_DISCRETION_IN_CROWD", True), \
             mock.patch.object(interaction, "_current_crowd_count", return_value=3), \
             mock.patch.object(interaction.emotional_events, "get_due_celebrations") as getter:
            cue = interaction._lean_celebration_cue(7)
        self.assertIsNone(cue)
        getter.assert_not_called()

    def test_fires_solo_even_with_discretion_on(self):
        cue = self._cue(
            [{"id": 4, "category": "new_baby", "description": "expecting a baby", "valence": 0.9}],
            crowd=1,
        )
        self.assertEqual(cue["event_id"], 4)

    def test_attempt_cap_steps_aside(self):
        # After the cap of unvoiced offers, the event is skipped so lower cues can run.
        cap = int(config.LEAN_CELEBRATION_MAX_UNVOICED_ATTEMPTS)
        interaction._celebration_unvoiced_attempts[4] = cap
        cue = self._cue([
            {"id": 4, "category": "career", "description": "the promotion", "valence": 0.9},
            {"id": 5, "category": "family", "description": "the new puppy", "valence": 0.7},
        ])
        # 4 is capped → falls through to 5.
        self.assertEqual(cue["event_id"], 5)

    def test_all_capped_returns_none(self):
        cap = int(config.LEAN_CELEBRATION_MAX_UNVOICED_ATTEMPTS)
        interaction._celebration_unvoiced_attempts.update({4: cap, 5: cap})
        cue = self._cue([
            {"id": 4, "description": "the promotion"},
            {"id": 5, "description": "the puppy"},
        ])
        self.assertIsNone(cue)

    def test_blocked_when_person_already_got_a_checkin(self):
        # A console (or prior celebration) this session set the shared gate → skip, and
        # get_due_celebrations is not even queried.
        consciousness._emotional_checkin_fired.add(7)
        with mock.patch.object(interaction.emotional_events, "get_due_celebrations") as getter, \
             mock.patch.object(config, "LEAN_CELEBRATION_CHECKIN_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", True):
            cue = interaction._lean_celebration_cue(7)
        self.assertIsNone(cue)
        getter.assert_not_called()

    def test_none_when_kill_switch_off(self):
        self.assertIsNone(self._cue([{"id": 4, "description": "x"}], enabled=False))

    def test_none_when_empathy_disabled(self):
        self.assertIsNone(self._cue([{"id": 4, "description": "x"}], empathy=False))

    def test_none_when_proactive_checkin_disabled(self):
        self.assertIsNone(self._cue([{"id": 4, "description": "x"}], proactive=False))

    def test_none_when_no_person(self):
        self.assertIsNone(self._cue([{"id": 4, "description": "x"}], person_id=None))

    def test_failsafe_none_on_db_error(self):
        with mock.patch.object(config, "LEAN_CELEBRATION_CHECKIN_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", True), \
             mock.patch.object(interaction.emotional_events, "get_due_celebrations",
                               side_effect=RuntimeError("db down")):
            self.assertIsNone(interaction._lean_celebration_cue(7))

    def test_survives_missing_person_name(self):
        with mock.patch.object(config, "LEAN_CELEBRATION_CHECKIN_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_ENABLED", True), \
             mock.patch.object(config, "EMPATHY_PROACTIVE_CHECKIN_ENABLED", True), \
             mock.patch.object(interaction.emotional_events, "get_due_celebrations",
                               return_value=[{"id": 9, "description": "big win", "category": "career"}]), \
             mock.patch.object(interaction.people_memory, "get_person", return_value=None):
            cue = interaction._lean_celebration_cue(7)
        self.assertEqual(cue["event_id"], 9)
        self.assertIsNone(cue["person_name"])
        self.assertEqual(cue["first_name"], "them")


class LeanCelebrationInstructionTest(unittest.TestCase):
    """The lean instruction celebrates the news and outranks every other cue."""

    def test_instruction_carries_the_news_and_forbids_pass(self):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("A new job — look at you, upgrading your meatbag existence.")

        cue = {"event_id": 4, "description": "got the new job", "category": "career"}
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(
                person_id=None, transcript=[], celebration=cue,
            )
        self.assertTrue(line)
        instruction = captured[0]
        self.assertIn("got the new job", instruction)
        self.assertIn("celebrate it", instruction)
        self.assertIn("do not reply PASS", instruction)
        self.assertNotIn("fresh angles", instruction)

    def test_celebration_beats_every_other_cue(self):
        # Priority: celebration > holiday > event > callback > visual.
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("Congrats!")

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(
                person_id=None,
                transcript=[],
                celebration={"event_id": 1, "description": "passed the bar exam", "category": "career"},
                holiday_plan={"name": "Juneteenth", "when": "Friday"},
                event_followup={"event_id": 2, "event_name": "job interview", "kind": "past"},
                callback_premise={"premise": "the haunted stapler"},
                visual_riff={"cue": "red scarf"},
            )
        instruction = captured[0]
        self.assertIn("passed the bar exam", instruction)
        self.assertNotIn("Juneteenth", instruction)
        self.assertNotIn("job interview", instruction)
        self.assertNotIn("haunted stapler", instruction)


class SpokenCelebrationWiringTest(unittest.TestCase):
    """End-to-end through `_maybe_lean_impulse`: a spoken celebration marks the event
    acknowledged, logs the episode, uses the celebration_checkin frame, and outranks the
    other cues (which are never even consulted)."""

    def setUp(self):
        self._saved_fired = set(consciousness._emotional_checkin_fired)
        self._saved_attempts = dict(interaction._celebration_unvoiced_attempts)
        consciousness._emotional_checkin_fired.clear()
        interaction._celebration_unvoiced_attempts.clear()
        self.addCleanup(lambda: (
            consciousness._emotional_checkin_fired.clear(),
            consciousness._emotional_checkin_fired.update(self._saved_fired),
            interaction._celebration_unvoiced_attempts.clear(),
            interaction._celebration_unvoiced_attempts.update(self._saved_attempts),
        ))

    def _run_impulse(self, *, spoke=True):
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

        cue = {
            "event_id": 42, "description": "landed the promotion", "category": "career",
            "person_name": "Bret Benziger", "first_name": "Bret",
        }
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
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=cue))
            # If celebration wins, none of the lower cues should be consulted.
            holiday = p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person"))
            event = p(mock.patch.object(I, "_lean_event_followup_cue"))
            cb = p(mock.patch.object(I, "_lean_callback_lull_cue"))
            riff = p(mock.patch.object(I, "_lean_visual_riff_cue"))
            register = p(mock.patch.object(I, "_register_rex_utterance"))
            ack = p(mock.patch.object(I.emotional_events, "mark_acknowledged"))
            hook = p(mock.patch("intelligence.episodic_hooks.celebration"))
            p(mock.patch.object(
                lean_brain, "consider_initiating",
                return_value="A promotion — try not to let the power go to your servos.",
            ))
            p(mock.patch.object(I, "_speak_proactive", return_value=spoke))

            fired = I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0)
        return cue, fired, register, ack, hook, holiday, event, cb, riff

    def test_spoken_celebration_marks_and_logs(self):
        cue, fired, register, ack, hook, holiday, event, cb, riff = self._run_impulse()
        self.assertTrue(fired)
        ack.assert_called_once_with(42)
        self.assertEqual(hook.call_count, 1)
        # The episode names the person and reads grammatically.
        _args, _kw = hook.call_args
        self.assertIn("Bret's good news", _args[2])
        # Celebration outranked everything → lower cues never consulted.
        holiday.assert_not_called()
        event.assert_not_called()
        cb.assert_not_called()
        riff.assert_not_called()
        # Registered under the celebration_checkin frame.
        self.assertEqual(register.call_count, 1)
        _a, kwargs = register.call_args
        self.assertEqual(kwargs.get("source"), "celebration_checkin")
        # A voiced celebration clears its unvoiced-attempt counter.
        self.assertNotIn(42, interaction._celebration_unvoiced_attempts)

    def test_offered_but_not_spoken_marks_nothing(self):
        # If the line is never actually spoken, the celebration must NOT be acknowledged or
        # logged — it stays available to retry — but the un-voiced attempt IS recorded so the
        # cap can eventually step it aside.
        _cue, fired, register, ack, hook, _h, _e, _c, _r = self._run_impulse(spoke=False)
        self.assertFalse(fired)
        ack.assert_not_called()
        hook.assert_not_called()
        register.assert_not_called()
        self.assertEqual(interaction._celebration_unvoiced_attempts.get(42), 1)


if __name__ == "__main__":
    unittest.main()
