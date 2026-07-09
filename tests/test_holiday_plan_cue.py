"""Holiday-plan cues: Lean owns the lull, while calendar dedupe stays per person/date."""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from awareness import holidays
from intelligence import consciousness, interaction, lean_brain


def _one_chunk_stream(text):
    return [NS(choices=[NS(delta=NS(content=text))])]


class HolidayLookupTest(unittest.TestCase):
    def setUp(self):
        self._asked = set(consciousness._holiday_plans_asked)
        consciousness._holiday_plans_asked.clear()

    def tearDown(self):
        consciousness._holiday_plans_asked.clear()
        consciousness._holiday_plans_asked.update(self._asked)

    def test_holiday_is_available_once_per_person_not_once_per_room(self):
        upcoming = [{
            "name": "Juneteenth National Independence Day",
            "date": "2026-06-19",
            "days_until": 2,
            "window": "minor",
        }]
        with (
            mock.patch.object(holidays, "upcoming_holidays", return_value=upcoming),
            mock.patch("memory.relationships.was_proactive_asked", return_value=False),
            mock.patch("memory.relationships.mark_proactive_asked"),
            mock.patch.object(config, "HOLIDAY_PLANS_INCLUDE_MINOR", True),
        ):
            cue_for_bret = consciousness._next_holiday_plan_for_person(1)
            self.assertTrue(cue_for_bret["when"])
            consciousness._mark_holiday_plan_asked(1, cue_for_bret)
            self.assertIsNone(consciousness._next_holiday_plan_for_person(1))
            # A new person has not been asked, so the same date remains fair game.
            self.assertEqual(
                consciousness._next_holiday_plan_for_person(2)["name"],
                "Juneteenth National Independence Day",
            )


class LeanHolidayCueTest(unittest.TestCase):
    def test_holiday_cue_requires_a_question_in_the_lean_instruction(self):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("Got plans for Juneteenth, or are you free-range that Friday?")

        cue = {
            "name": "Juneteenth National Independence Day",
            "when": "this Friday",
            "date": "2026-06-19",
        }
        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(
                person_id=None, transcript=[], holiday_plan=cue,
            )
        self.assertTrue(line.endswith("?"))
        self.assertIn("Juneteenth National Independence Day", captured[0])
        self.assertIn("MUST ask the question", captured[0])
        self.assertNotIn("fresh angles", captured[0])

    def test_spoken_lean_holiday_cue_marks_only_that_person(self):
        import intelligence.interaction as I

        cue = {"name": "Juneteenth", "when": "this Friday", "date": "2026-06-19"}
        original_state = {
            "last_user_content_at": I._last_user_content_at,
            "consecutive_lean_impulses": I._consecutive_lean_impulses,
            "last_lean_impulse_at": I._last_lean_impulse_at,
            "last_proactive_line_at": I._last_proactive_line_at,
            "floor_held_until": I._floor_held_until,
        }
        self.addCleanup(lambda: setattr(I, "_last_user_content_at", original_state["last_user_content_at"]))
        self.addCleanup(lambda: setattr(I, "_consecutive_lean_impulses", original_state["consecutive_lean_impulses"]))
        self.addCleanup(lambda: setattr(I, "_last_lean_impulse_at", original_state["last_lean_impulse_at"]))
        self.addCleanup(lambda: setattr(I, "_last_proactive_line_at", original_state["last_proactive_line_at"]))
        self.addCleanup(lambda: setattr(I, "_floor_held_until", original_state["floor_held_until"]))
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
            p(mock.patch.object(I, "_register_rex_utterance"))
            p(mock.patch.object(I.conv_memory, "add_to_transcript"))
            p(mock.patch.object(I.conv_log, "log_rex"))
            p(mock.patch.object(I.consciousness, "_next_holiday_plan_for_person", return_value=cue))
            marked = p(mock.patch.object(I.consciousness, "_mark_holiday_plan_asked"))
            p(mock.patch.object(
                lean_brain,
                "consider_initiating",
                return_value="Got plans for Juneteenth?",
            ))
            p(mock.patch.object(I, "_speak_proactive", return_value=True))

            self.assertTrue(I._maybe_lean_impulse(idle_for=5.0, effective_idle_timeout=60.0))

        marked.assert_called_once_with(7, cue)


class HolidayFallbackCalendarTest(unittest.TestCase):
    def setUp(self):
        self._cache = dict(holidays._cache)
        self._retry = dict(holidays._fetch_retry_after)
        holidays._cache.clear()
        holidays._fetch_retry_after.clear()

    def tearDown(self):
        holidays._cache.clear()
        holidays._cache.update(self._cache)
        holidays._fetch_retry_after.clear()
        holidays._fetch_retry_after.update(self._retry)

    def test_us_fallback_keeps_calendar_available_when_fetch_fails(self):
        with (
            mock.patch.object(holidays, "_fetch_year", return_value=[]),
            mock.patch.object(config, "HOLIDAY_COUNTRY_CODE", "US"),
        ):
            rows = holidays.get_holidays(2026)
        self.assertTrue(any(row["name"] == "Thanksgiving Day" for row in rows))
        self.assertTrue(any(row["name"] == "Juneteenth National Independence Day" for row in rows))


if __name__ == "__main__":
    unittest.main()
