"""Episodic "memory musing" as the lowest-priority Lean lull cue.

The old idle `do_memory_musing` beat (purpose `memory_musing`) went dark under the lean
brain — its governor candidate is suppressed and the lean impulse never consulted the
rex.db diary. This revives it as a cue into the single lull speaker: an occasional
"since I was last on" recollection, sourced from `episodic_recall.session_recap`, at the
LOWEST priority (only when no celebration/holiday/event/callback/visual-riff fires) and
capped at one per session (the recap is stable within a session, so a second would repeat).
"""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import interaction, lean_brain
from memory import episodic_recall


def _one_chunk_stream(text):
    return [NS(choices=[NS(delta=NS(content=text))])]


class MemoryMusingCueSelectionTest(unittest.TestCase):
    def setUp(self):
        self._saved_mused = interaction._lean_memory_mused_this_session
        interaction._lean_memory_mused_this_session = False
        self.addCleanup(
            lambda: setattr(interaction, "_lean_memory_mused_this_session", self._saved_mused)
        )

    def _cue(self, recap, *, person_id=7, enabled=True, recall=True, prob=0.5, roll=0.0):
        with mock.patch.object(config, "LEAN_MEMORY_MUSING_ENABLED", enabled), \
             mock.patch.object(config, "EPISODIC_RECALL_ENABLED", recall), \
             mock.patch.object(config, "EPISODIC_RECALL_SESSION_RECAP_PROBABILITY", prob), \
             mock.patch.object(interaction.random, "random", return_value=roll), \
             mock.patch.object(episodic_recall, "session_recap", return_value=recap):
            return interaction._lean_memory_musing_cue(person_id)

    def test_returns_recap_when_all_gates_pass(self):
        cue = self._cue("the room was cluttered; I played Trivia with Bret — scored 4 of 5.")
        self.assertEqual(cue, {"recap": "the room was cluttered; I played Trivia with Bret — scored 4 of 5."})

    def test_none_when_kill_switch_off(self):
        self.assertIsNone(self._cue("something", enabled=False))

    def test_none_when_recall_disabled(self):
        self.assertIsNone(self._cue("something", recall=False))

    def test_none_when_probability_fails(self):
        # roll >= prob → skip (subtle occasional spice).
        self.assertIsNone(self._cue("something", prob=0.5, roll=0.9))

    def test_none_when_recap_empty(self):
        self.assertIsNone(self._cue(None))
        self.assertIsNone(self._cue("   "))

    def test_none_when_no_person(self):
        self.assertIsNone(self._cue("something", person_id=None))

    def test_once_per_session(self):
        interaction._lean_memory_mused_this_session = True
        with mock.patch.object(episodic_recall, "session_recap") as recap:
            cue = self._cue("something")
        self.assertIsNone(cue)
        recap.assert_not_called()  # short-circuits before touching the diary

    def test_failsafe_none_on_error(self):
        with mock.patch.object(config, "LEAN_MEMORY_MUSING_ENABLED", True), \
             mock.patch.object(config, "EPISODIC_RECALL_ENABLED", True), \
             mock.patch.object(config, "EPISODIC_RECALL_SESSION_RECAP_PROBABILITY", 1.0), \
             mock.patch.object(interaction.random, "random", return_value=0.0), \
             mock.patch.object(episodic_recall, "session_recap", side_effect=RuntimeError("db")):
            self.assertIsNone(interaction._lean_memory_musing_cue(7))


class MemoryMusingInstructionTest(unittest.TestCase):
    def test_instruction_reminisces_without_menu(self):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("Still can't shake that Trivia run — four out of five, not bad for an organic.")

        cue = {"recap": "I played Trivia with Bret — scored 4 of 5."}
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(person_id=None, transcript=[], memory_musing=cue)
        self.assertTrue(line)
        instruction = captured[0]
        self.assertIn("I played Trivia with Bret", instruction)
        self.assertIn("MUSE", instruction)
        self.assertIn("do not reply PASS", instruction)
        self.assertNotIn("fresh angles", instruction)

    def test_memory_musing_is_lowest_priority(self):
        # Every other cue outranks it.
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("How'd the recital go?")

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(
                person_id=None,
                transcript=[],
                visual_riff={"cue": "a familiar visual detail: red scarf"},
                memory_musing={"recap": "I saw a dog once."},
            )
        instruction = captured[0]
        self.assertIn("red scarf", instruction)     # visual_riff wins
        self.assertNotIn("I saw a dog", instruction)  # musing did not


class SpokenMemoryMusingWiringTest(unittest.TestCase):
    """End-to-end through `_maybe_lean_impulse`: a spoken musing sets the once-per-session
    flag, and it's only consulted when no higher cue fires."""

    def setUp(self):
        self._saved = interaction._lean_memory_mused_this_session
        interaction._lean_memory_mused_this_session = False
        self.addCleanup(
            lambda: setattr(interaction, "_lean_memory_mused_this_session", self._saved)
        )

    def test_spoken_musing_sets_once_per_session_flag(self):
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

        cue = {"recap": "I made Bret laugh; the room was a mess."}
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
            # All higher cues absent → musing gets its turn.
            p(mock.patch.object(I, "_lean_celebration_cue", return_value=None))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=None))
            p(mock.patch.object(I, "_lean_event_followup_cue", return_value=None))
            p(mock.patch.object(I, "_lean_callback_lull_cue", return_value=None))
            p(mock.patch.object(I, "_lean_visual_riff_cue", return_value=None))
            musing = p(mock.patch.object(I, "_lean_memory_musing_cue", return_value=cue))
            p(mock.patch.object(lean_brain, "consider_initiating",
                                return_value="Still turning that mess of a room over in my circuits."))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        musing.assert_called_once()
        self.assertTrue(I._lean_memory_mused_this_session)

    def test_musing_not_consulted_when_a_higher_cue_wins(self):
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
            # A celebration wins → musing must never be consulted.
            p(mock.patch.object(I, "_lean_celebration_cue",
                                return_value={"event_id": 1, "description": "the promotion",
                                              "category": "career", "first_name": "Bret",
                                              "person_name": "Bret"}))
            p(mock.patch.object(I.emotional_events, "mark_acknowledged"))
            p(mock.patch("intelligence.episodic_hooks.celebration"))
            musing = p(mock.patch.object(I, "_lean_memory_musing_cue"))
            p(mock.patch.object(lean_brain, "consider_initiating", return_value="Congrats on the promotion!"))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        musing.assert_not_called()
        self.assertFalse(I._lean_memory_mused_this_session)


class MemoryMusingSessionResetTest(unittest.TestCase):
    """The once-per-session cap is only correct if _end_session clears the flag — otherwise a
    single musing mutes all future musings for the process lifetime. Pins that reset."""

    def test_end_session_clears_the_once_per_session_flag(self):
        I = interaction
        saved = I._lean_memory_mused_this_session
        self.addCleanup(lambda: setattr(I, "_lean_memory_mused_this_session", saved))
        I._lean_memory_mused_this_session = True
        # Empty transcript → the LIGHT early-return branch of _end_session (no per-person DB
        # summary/consolidation work). Its in-memory clears + try/except-wrapped calls are safe
        # in the test env; this exercises the reset without the heavy main path.
        with mock.patch.object(I.conv_memory, "get_session_transcript", return_value=[]):
            I._end_session()
        self.assertFalse(I._lean_memory_mused_this_session)


if __name__ == "__main__":
    unittest.main()
