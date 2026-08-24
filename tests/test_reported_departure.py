"""Third-person departure reports close the departed person's threads.

Field log 2026-08-23 18:26 (logs/djr3x-2026-08-23-18-15-49.log): Exudica said
"Work trip." and walked out; Rex asked "Where are they dragging you?" at the
empty doorway. The remaining guest then said "She left already." TWICE, and
finally "You're asking questions to somebody that already left." — each report
was swallowed by the open-question capture (the onboarding burst consumed the
first as its answer) and chained straight into the next interview question.

Now a departure report from a resolved speaker:
  * is consumed BEFORE the onboarding/answer captures ever see it,
  * closes the departed person's engagement (active AND recent) via
    consciousness.note_reported_departure, so attribution stops chaining,
  * arms a latch that makes _step_presence_tracking resolve the camera-confirmed
    absence silently (the exit was already acknowledged out loud),
  * closes an onboarding burst aimed at the departed person, and
  * gets a brief acknowledgement — never another question at the empty chair.

Pronoun rule: "she" resolves to the most recent NON-SPEAKER engaged person,
never the speaker — the speaker is describing someone else's exit.

Run: venv/bin/python -m unittest tests.test_reported_departure
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness as C
from intelligence import interaction as I


_PEOPLE = {
    1: {"id": 1, "name": "Bret Benziger"},
    3: {"id": 3, "name": "Exudica Marbles"},
    8: {"id": 8, "name": "Headed Home"},
}


def _fake_get_person(pid):
    return _PEOPLE.get(int(pid))


class _EngagementStateMixin(unittest.TestCase):
    """Save/restore the consciousness engagement + turn-history globals."""

    def setUp(self):
        super().setUp()
        self._saved = (
            C._engaged_person_id,
            C._engaged_last_touch_at,
            C._recent_engaged_person_id,
            C._recent_engaged_touch_at,
        )
        self._saved_turns = {
            pid: list(turns) for pid, turns in C._group_turn_speaker_times.items()
        }
        self._saved_reported = dict(C._reported_departure_at)
        C.clear_engagement()
        C._recent_engaged_person_id = None
        C._recent_engaged_touch_at = 0.0
        C._group_turn_speaker_times.clear()
        C._reported_departure_at.clear()
        patcher = mock.patch("memory.people.get_person", side_effect=_fake_get_person)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        (
            C._engaged_person_id,
            C._engaged_last_touch_at,
            C._recent_engaged_person_id,
            C._recent_engaged_touch_at,
        ) = self._saved
        C._group_turn_speaker_times.clear()
        for pid, turns in self._saved_turns.items():
            C._group_turn_speaker_times[pid] = C.deque(turns)
        C._reported_departure_at.clear()
        C._reported_departure_at.update(self._saved_reported)
        super().tearDown()


# ── Detection ──────────────────────────────────────────────────────────────────

class DetectionTests(unittest.TestCase):

    def test_field_utterances_are_reports(self):
        for text in (
            "She left already.",
            "She left already.",  # said twice in the field log; both must land
            "You're asking questions to somebody that already left.",
        ):
            self.assertIsNotNone(I._reported_departure_match(text), text)

    def test_pronoun_forms(self):
        for text, pronoun in (
            ("She left already.", "she"),
            ("He's gone.", "he"),
            ("They went home.", "they"),
            ("She already left.", "she"),
            ("They're gone.", "they"),
            ("He just left.", "he"),
            ("She took off.", "she"),
            ("No, she left already.", "she"),
        ):
            m = I._reported_departure_match(text)
            self.assertIsNotNone(m, text)
            self.assertEqual(m["kind"], "pronoun", text)
            self.assertEqual(m["pronoun"], pronoun, text)

    def test_named_forms(self):
        for text, name in (
            ("Exudica left already.", "Exudica"),
            ("Exudica Marbles went home.", "Exudica Marbles"),
            ("PJ is gone.", "PJ"),  # the aux must not ride along in the name
        ):
            m = I._reported_departure_match(text)
            self.assertIsNotNone(m, text)
            self.assertEqual(m["kind"], "named", text)
            self.assertEqual(m["name"], name, text)

    def test_non_departures_do_not_trigger(self):
        for text in (
            "She left her keys on the couch.",   # left + object: not an exit
            "He left me a note.",
            "They left the door open.",
            "I left already.",                    # first person → end_thread's job
            "I'm gonna leave.",
            "You left already.",
            "We left the party.",
            "Did she leave?",                     # a question is never a report
            "Work trip.",
            "That was a great trip.",
            "Left already.",                      # no subject, no referent
        ):
            self.assertIsNone(I._reported_departure_match(text), text)


# ── Pronoun referent: most recent NON-SPEAKER voice ────────────────────────────

class PronounReferentTests(_EngagementStateMixin):

    def test_resolves_the_most_recent_other_speaker(self):
        C.note_person_spoke(1)   # Bret, earlier
        C.note_person_spoke(3)   # Exudica, most recent
        other = C.most_recent_other_speaker(8)
        self.assertEqual(other["person_id"], 3)
        self.assertEqual(other["name"], "Exudica Marbles")

    def test_never_resolves_to_the_speaker(self):
        C.note_person_spoke(1)
        C.note_person_spoke(3)
        other = C.most_recent_other_speaker(3)
        self.assertEqual(other["person_id"], 1)

    def test_no_other_voice_means_no_referent(self):
        C.note_person_spoke(8)
        self.assertIsNone(C.most_recent_other_speaker(8))


# ── The handler: field scenario ────────────────────────────────────────────────

class ReportedDepartureHandlerTests(_EngagementStateMixin):
    """"She left already." from Headed Home (8) while Exudica (3) was engaged."""

    def setUp(self):
        super().setUp()
        self._saved_session = set(I._session_person_ids)
        self._saved_onboarding = I._pending_onboarding
        I._session_person_ids.clear()
        I._session_person_ids.update({1, 3, 8})
        I._pending_onboarding = None
        # Exudica spoke last and is the engaged partner (field state at 18:26:49).
        C.note_person_spoke(1)
        C.note_person_spoke(3)
        C.mark_engagement(3)
        unlearnable = mock.patch.object(I.conv_memory, "mark_last_human_turn_unlearnable")
        self._unlearnable = unlearnable.start()
        self.addCleanup(unlearnable.stop)

    def tearDown(self):
        I._session_person_ids.clear()
        I._session_person_ids.update(self._saved_session)
        I._pending_onboarding = self._saved_onboarding
        super().tearDown()

    def test_she_left_resolves_to_the_engaged_non_speaker_and_acks(self):
        ack = I._handle_reported_departure("She left already.", 8, "Headed Home")
        self.assertTrue(ack)
        self.assertNotIn("?", ack, "the ack must not re-open the interview")
        self.assertTrue(
            C.recent_reported_departure(3),
            "the presence silent-close latch must arm for the departed person",
        )

    def test_the_departed_persons_engagement_is_fully_closed(self):
        I._handle_reported_departure("She left already.", 8, "Headed Home")
        recent = C.get_recent_engagement() or {}
        self.assertNotEqual(
            recent.get("person_id"), 3,
            "recent engagement must stop chaining follow-ups to the departed person",
        )
        # ...and the reporter is the conversational partner now.
        self.assertEqual(recent.get("person_id"), 8)

    def test_the_meta_repeat_from_the_field_log_also_lands(self):
        ack = I._handle_reported_departure(
            "You're asking questions to somebody that already left.", 8, "Headed Home"
        )
        self.assertTrue(ack)
        self.assertTrue(C.recent_reported_departure(3))

    def test_named_report_resolves_against_session_people(self):
        with mock.patch.object(
            I.people_memory, "find_person_by_name",
            return_value={"id": 3, "name": "Exudica Marbles"},
        ):
            ack = I._handle_reported_departure("Exudica left already.", 1, "Bret Benziger")
        self.assertTrue(ack)
        self.assertIn("Exudica", ack)
        self.assertTrue(C.recent_reported_departure(3))

    def test_named_report_about_a_stranger_is_not_consumed(self):
        # "Alice left" about a DB person who was never here: release the turn.
        with mock.patch.object(
            I.people_memory, "find_person_by_name",
            return_value={"id": 99, "name": "Alice Elsewhere"},
        ):
            ack = I._handle_reported_departure("Alice left already.", 1, "Bret Benziger")
        self.assertIsNone(ack)
        self.assertFalse(C.recent_reported_departure(99))

    def test_an_unresolved_speaker_does_not_close_anyone(self):
        self.assertIsNone(I._handle_reported_departure("She left already.", None, None))
        self.assertFalse(C.recent_reported_departure(3))

    def test_a_lone_speaker_report_has_no_referent(self):
        # Only the speaker's own voice on record: "she" cannot mean the speaker.
        C._group_turn_speaker_times.clear()
        C.clear_engagement()
        C._recent_engaged_person_id = None
        C.note_person_spoke(8)
        C.mark_engagement(8)
        self.assertIsNone(
            I._handle_reported_departure("She left already.", 8, "Headed Home")
        )

    def test_an_onboarding_burst_aimed_at_the_departed_person_closes(self):
        I._pending_onboarding = {"person_id": 3, "asked_count": 1, "answered_count": 0}
        I._handle_reported_departure("She left already.", 8, "Headed Home")
        self.assertIsNone(I._pending_onboarding)

    def test_an_onboarding_burst_aimed_at_the_reporter_stays_open(self):
        # The field burst was for the REPORTER (person 8): it keeps its audience.
        I._pending_onboarding = {"person_id": 8, "asked_count": 1, "answered_count": 0}
        I._handle_reported_departure("She left already.", 8, "Headed Home")
        self.assertIsNotNone(I._pending_onboarding)

    def test_the_report_teaches_nothing_about_the_reporter(self):
        I._handle_reported_departure("She left already.", 8, "Headed Home")
        self._unlearnable.assert_called_once()

    def test_the_departed_persons_topic_thread_is_dropped(self):
        with mock.patch.object(I.topic_thread, "clear") as thread_clear:
            I._handle_reported_departure("She left already.", 8, "Headed Home")
        thread_clear.assert_called_once()

    def test_a_report_about_someone_else_keeps_the_reporters_thread(self):
        # Bret is engaged and mid-topic; Exudica (spoke earlier) left. Bret's
        # own live thread must survive the report.
        C.mark_engagement(1)
        with mock.patch.object(I.topic_thread, "clear") as thread_clear:
            ack = I._handle_reported_departure("She left already.", 1, "Bret Benziger")
        self.assertTrue(ack)
        self.assertTrue(C.recent_reported_departure(3))
        thread_clear.assert_not_called()


# ── Presence machinery: camera-confirmed absence closes silently ───────────────

class SilentPresenceCloseTests(unittest.TestCase):
    """A staged departure for a reported-gone person resolves with no quip —
    even past a stale still-here hold, same rule as the explicit-goodbye latch."""

    def setUp(self):
        self._saved = {
            "visible": set(C._visible_people),
            "last_seen": dict(C._last_seen),
            "pending": dict(C._pending_departure_keys),
            "missing": dict(C._first_missing_at),
            "visit": dict(C._visit_started_at),
            "reported": dict(C._reported_departure_at),
            "reacted": dict(C._last_departure_reaction_at),
        }
        C._visible_people.clear()
        C._last_seen.clear()
        C._pending_departure_keys.clear()
        C._first_missing_at.clear()
        C._visit_started_at.clear()
        C._reported_departure_at.clear()
        C._last_departure_reaction_at.clear()

    def tearDown(self):
        C._visible_people.clear(); C._visible_people.update(self._saved["visible"])
        C._last_seen.clear(); C._last_seen.update(self._saved["last_seen"])
        C._pending_departure_keys.clear()
        C._pending_departure_keys.update(self._saved["pending"])
        C._first_missing_at.clear(); C._first_missing_at.update(self._saved["missing"])
        C._visit_started_at.clear(); C._visit_started_at.update(self._saved["visit"])
        C._reported_departure_at.clear()
        C._reported_departure_at.update(self._saved["reported"])
        C._last_departure_reaction_at.clear()
        C._last_departure_reaction_at.update(self._saved["reacted"])

    def _run_tick(self, *, now, likely_still_present):
        # Missing for 60s, staged 10s ago — inside the quip window, so the
        # no-latch control case fires the spoken departure reaction.
        C._pending_departure_keys[3] = (now - 60.0, "Exudica Marbles", 3, now - 10.0)
        C._first_missing_at[3] = now - 60.0
        C._visit_started_at[3] = now - 300.0
        profile = mock.Mock(
            suppress_proactive=False, interaction_busy=False,
            user_mid_sentence=False, likely_still_present=likely_still_present,
            apparent_departure=True,
        )
        with (
            mock.patch.object(C.time, "monotonic", return_value=now),
            mock.patch.object(C, "_face_tracking_recently_held_person",
                              return_value=False),
            mock.patch.object(C, "_should_fire_presence", return_value=True),
            mock.patch.object(C, "_generate_and_speak_presence") as quip,
            mock.patch.object(C.episodic_hooks, "visit_departure") as departure,
        ):
            C._step_presence_tracking({"people": [], "crowd": {"count": 0}}, profile)
        return quip, departure

    def test_reported_departure_closes_the_visit_without_a_quip(self):
        C._reported_departure_at[3] = 950.0
        quip, departure = self._run_tick(now=1000.0, likely_still_present=False)
        quip.assert_not_called()
        departure.assert_called_once()
        self.assertNotIn(3, C._pending_departure_keys)
        self.assertIn(3, C._last_departure_reaction_at)

    def test_the_latch_beats_a_stale_still_here_hold(self):
        # Field shape: a lingering face-track/audio hold reads the departed
        # person as present; the human's word must close the visit anyway.
        C._reported_departure_at[3] = 950.0
        quip, departure = self._run_tick(now=1000.0, likely_still_present=True)
        quip.assert_not_called()
        departure.assert_called_once()
        self.assertNotIn(3, C._pending_departure_keys)

    def test_without_the_latch_a_departure_still_quips(self):
        quip, departure = self._run_tick(now=1000.0, likely_still_present=False)
        quip.assert_called_once()
        departure.assert_called_once()

    def test_an_expired_report_no_longer_suppresses(self):
        window = float(getattr(config, "REPORTED_DEPARTURE_WINDOW_SECS", 600.0))
        C._reported_departure_at[3] = 1000.0 - window - 5.0
        quip, departure = self._run_tick(now=1000.0, likely_still_present=False)
        quip.assert_called_once()

    def test_speaking_again_clears_the_latch(self):
        C._reported_departure_at[3] = 950.0
        C.mark_engagement(3)
        self.assertFalse(C.recent_reported_departure(3))


# ── Wiring: the report is consumed before any answer capture ───────────────────

class GateWiringTests(unittest.TestCase):
    """The full segment handler needs a heavyweight harness — asserted
    structurally, same as the address-gate and own-echo overrides."""

    def test_report_is_intercepted_before_onboarding_and_pending_qa(self):
        import inspect
        src = inspect.getsource(I._handle_speech_segment)
        report_idx = src.index("_handle_reported_departure(")
        self.assertLess(
            report_idx,
            src.index("onboarding_flow_active()"),
            "the onboarding burst swallowed 'She left already.' in the field — "
            "the report must be consumed first",
        )
        self.assertLess(report_idx, src.index("_maybe_capture_pending_qa("))
        self.assertLess(report_idx, src.index("_handle_onboarding_turn("))

    def test_the_acks_never_ask_a_question(self):
        for line in I._REPORTED_DEPARTURE_ACKS_NAMED:
            self.assertNotIn("?", line.format(name="Exudica"))
        for line in I._REPORTED_DEPARTURE_ACKS_GENERIC:
            self.assertNotIn("?", line)


if __name__ == "__main__":
    unittest.main()
