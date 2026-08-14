"""A pet that walks in during a commanded look is the ANSWER, not an interruption.

Field 2026-08-13 (logs/djr3x-2026-08-13-20-58-37.log, 21:02:14-21:03:01): the owner
said "look down and to your left" at his dog. The detector fired 3s later, the remark
was composed, and the action governor rejected it 46 consecutive times over 47 seconds
until the session ended. Rex never mentioned the dog.

Two of those 46 rejections named a SINGLE reason with Rex silent and idle:

    21:02:18  reasons=waiting_for_human_response
    21:02:47  reasons=situation_suppresses_proactive

The first is the gate that says "you asked a question, wait for the answer" — but the
answer had just arrived through his own eyes. The second is pacing. Neither is a
reason to stay quiet about the thing you were just told to look at.
"""

import time
import unittest
from unittest import mock

from awareness.situation import SituationAssessor, SituationProfile
from state import State
from intelligence.action_governor import ActionGovernor, CandidateMove


def _profile(**kw):
    """A SituationProfile with every field defaulted, overridden by kw."""
    base = dict(
        conversation_active=False,
        user_mid_sentence=False,
        rapid_exchange=False,
        child_present=False,
        apparent_departure=False,
        likely_still_present=False,
        social_mode="one_on_one",
        suppress_proactive=False,
        suppress_system_comments=False,
        force_family_safe=False,
        being_discussed=False,
        discussion_sentiment="neutral",
        interaction_busy=False,
        suppress_proactive_pacing_only=False,
    )
    base.update(kw)
    return SituationProfile(**base)


class SuppressProactiveDecompositionTest(unittest.TestCase):
    """suppress_proactive ORs three HARD conditions with two PACING ones. The new
    flag says "the pacing half is the whole reason" and nothing else."""

    def setUp(self):
        self.assessor = SituationAssessor()
        self.state = mock.patch("awareness.situation.state_module.get_state",
                                return_value=State.IDLE)
        self.state.start()
        self.addCleanup(self.state.stop)

    def _evaluate(self):
        return self.assessor.evaluate()

    def test_rex_just_spoke_is_pacing_only(self):
        self.assessor.set_rex_speaking(True)
        self.assessor.set_rex_speaking(False)   # stamps rex_stopped_at = now
        p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertTrue(p.suppress_proactive_pacing_only)

    def test_user_mid_sentence_is_not_pacing_only(self):
        self.assessor.set_vad_active(True)
        p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)

    def test_interaction_busy_is_not_pacing_only(self):
        self.assessor.set_interaction_busy(True)
        p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)

    def test_quiet_state_is_not_pacing_only(self):
        # The safety case: QUIET/SHUTDOWN must never look like mere pacing, or the
        # governor bypass below would let a reaction speak while Rex is off.
        with mock.patch("awareness.situation.state_module.get_state",
                        return_value=State.QUIET):
            p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)

    def test_shutdown_state_is_not_pacing_only(self):
        with mock.patch("awareness.situation.state_module.get_state",
                        return_value=State.SHUTDOWN):
            p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)

    def test_pacing_flag_is_never_true_without_suppression(self):
        p = self._evaluate()
        self.assertFalse(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)

    def test_hard_condition_wins_when_both_are_present(self):
        # Rex just stopped talking AND the user is now talking over him: the pacing
        # half is true, but it is NOT the whole reason.
        self.assessor.set_rex_speaking(True)
        self.assessor.set_rex_speaking(False)
        self.assessor.set_vad_active(True)
        p = self._evaluate()
        self.assertTrue(p.suppress_proactive)
        self.assertFalse(p.suppress_proactive_pacing_only)


class GovernorPacingBypassTest(unittest.TestCase):
    def setUp(self):
        from intelligence import action_governor as ag
        ag._recent_selected.clear()

    def _score(self, *, profile, metadata):
        governor = ActionGovernor()
        governor.start_cycle(profile=profile)
        governor.observe(CandidateMove(
            source="_step_proactive_reactions",
            purpose="world.animal_arrival",
            suggested_text="Well, hello, small furry lifeform.",
            metadata=metadata,
        ))
        decision = governor.finish_cycle()
        return decision.scored[0]

    def test_reactive_arrival_speaks_through_pacing_only_suppression(self):
        # cycle-244 in the field log: this was the ONE reason, Rex silent and idle.
        scored = self._score(
            profile=_profile(suppress_proactive=True,
                             suppress_proactive_pacing_only=True),
            metadata={"salient": True, "reactive": True},
        )
        self.assertNotIn("situation_suppresses_proactive", scored.reasons)
        self.assertFalse(scored.rejected)

    def test_salient_alone_still_yields_to_pacing(self):
        # The bypass is scoped to `reactive` on purpose — an ordinary salient
        # arrival keeps waiting its turn, so this is not a global over-talk change.
        scored = self._score(
            profile=_profile(suppress_proactive=True,
                             suppress_proactive_pacing_only=True),
            metadata={"salient": True},
        )
        self.assertIn("situation_suppresses_proactive", scored.reasons)
        self.assertTrue(scored.rejected)

    def test_reactive_still_blocked_when_a_person_is_talking(self):
        scored = self._score(
            profile=_profile(user_mid_sentence=True, suppress_proactive=True,
                             suppress_proactive_pacing_only=False),
            metadata={"salient": True, "reactive": True},
        )
        self.assertIn("situation_suppresses_proactive", scored.reasons)
        self.assertIn("user_mid_sentence", scored.reasons)
        self.assertTrue(scored.rejected)

    def test_reactive_still_blocked_mid_turn(self):
        scored = self._score(
            profile=_profile(interaction_busy=True, suppress_proactive=True,
                             suppress_proactive_pacing_only=False),
            metadata={"salient": True, "reactive": True},
        )
        self.assertIn("situation_suppresses_proactive", scored.reasons)
        self.assertTrue(scored.rejected)

    def test_reactive_still_blocked_while_rex_is_speaking(self):
        # The bypass touches ONE reason. Talking over himself stays impossible.
        scored = self._score(
            profile=_profile(suppress_proactive=True,
                             suppress_proactive_pacing_only=True),
            metadata={"salient": True, "reactive": True,
                      "speech_queue_speaking": True, "output_gate_busy": True},
        )
        self.assertTrue(scored.rejected)
        self.assertIn("speech_queue_speaking", scored.reasons)
        self.assertIn("output_gate_busy", scored.reasons)

    def test_reactive_cannot_speak_while_rex_is_off(self):
        # THE safety case for narrowing situation_suppresses_proactive. QUIET /
        # SLEEP / SHUTDOWN never depended on that reason: observe_governor_candidate
        # merges governor_speech_metadata() under the caller's flags, so every
        # governed candidate carries can_speak — and can_speak_false rejects with no
        # salient/reactive exemption. (_do_speak re-checks _can_speak() again before
        # enqueuing, so this is belt AND braces.) SLEEP proves the point on its own:
        # situation.py's state_suppresses is only (QUIET, SHUTDOWN), so in SLEEP the
        # reason this test narrows was already False today.
        scored = self._score(
            profile=_profile(suppress_proactive=True,
                             suppress_proactive_pacing_only=True),
            metadata={"salient": True, "reactive": True, "can_speak": False},
        )
        self.assertIn("can_speak_false", scored.reasons)
        self.assertTrue(scored.rejected)

    def test_can_speak_is_stamped_on_governed_candidates(self):
        # Guards the assumption above: if this merge ever stops happening, the
        # state guard silently disappears from the governor layer.
        from intelligence import speech_engine
        self.assertIn("can_speak", speech_engine.governor_speech_metadata())

    def test_profile_without_the_new_field_stays_conservative(self):
        # Hand-rolled fake profiles live in several test modules and in older call
        # sites; a missing field must read as "not pacing-only", i.e. still blocked.
        class _OldProfile:
            suppress_proactive = True

        scored = self._score(profile=_OldProfile(),
                             metadata={"salient": True, "reactive": True})
        self.assertIn("situation_suppresses_proactive", scored.reasons)
        self.assertTrue(scored.rejected)

    def test_reactive_also_clears_the_awaiting_reply_gate(self):
        # 21:02:18, the other single-reason rejection: Rex had asked "What am I
        # looking for?" and the answer walked into frame by itself.
        scored = self._score(
            profile=_profile(),
            metadata={"salient": True, "reactive": True,
                      "waiting_for_response": True},
        )
        self.assertNotIn("waiting_for_human_response", scored.reasons)
        self.assertFalse(scored.rejected)

    def test_non_reactive_arrival_still_waits_for_the_reply(self):
        scored = self._score(
            profile=_profile(),
            metadata={"salient": True, "waiting_for_response": True},
        )
        self.assertIn("waiting_for_human_response", scored.reasons)
        self.assertTrue(scored.rejected)


class DirectedLookContextIsTheSignalTest(unittest.TestCase):
    """The predicate is the interaction-side directed-look context, NOT the gaze
    hold — exploration and motion_agency take that hold for gazes nobody asked for."""

    def setUp(self):
        from intelligence import interaction as I
        self.I = I
        self._saved = dict(I._directed_look_context)
        I._reset_directed_look_context()
        self.addCleanup(lambda: I._directed_look_context.update(self._saved))

    def test_false_when_no_look_was_commanded(self):
        self.assertFalse(self.I.user_directed_look_active())

    def test_true_right_after_a_commanded_look(self):
        self.I._note_directed_look_context(direction="down_left")
        self.assertTrue(self.I.user_directed_look_active())

    def test_false_once_the_target_was_found(self):
        # found=True resets the context: Rex has answered, so a later pet is
        # ordinary proactive chatter again.
        self.I._note_directed_look_context(direction="down_left")
        self.I._note_directed_look_context(direction="down_left", found=True)
        self.assertFalse(self.I.user_directed_look_active())

    def test_false_once_the_window_lapses(self):
        import config
        self.I._note_directed_look_context(direction="down_left")
        ttl = float(getattr(config, "DIRECTED_LOOK_CONTEXT_WINDOW_SECS", 25.0))
        self.assertFalse(self.I.user_directed_look_active(time.monotonic() + ttl + 1.0))

    def test_consciousness_reads_it(self):
        from intelligence import consciousness as C
        self.assertFalse(C._answering_a_directed_look())
        self.I._note_directed_look_context(direction="down_left")
        self.assertTrue(C._answering_a_directed_look())

    def test_consciousness_predicate_survives_a_broken_import(self):
        from intelligence import consciousness as C
        with mock.patch.object(self.I, "user_directed_look_active",
                               side_effect=RuntimeError("boom")):
            self.assertFalse(C._answering_a_directed_look())


class ArrivalIsMarkedReactiveDuringALookTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness as C
        self.C = C
        self._saved_pending = dict(C._pending_animal_arrivals)
        C._pending_animal_arrivals.clear()
        C._pending_animal_arrivals["cat"] = {
            "species": "cat", "kind": "arrival", "return_count": 0,
            "first_seen_at": time.monotonic(), "last_seen_at": time.monotonic(),
        }
        self.addCleanup(self._restore)
        self.frame_and_line = mock.patch.object(
            C, "_animal_reaction_frame_and_line",
            return_value=(mock.Mock(affect="amused"), "Well, hello, small furry lifeform."),
        )
        self.frame_and_line.start()
        self.addCleanup(self.frame_and_line.stop)

    def _restore(self):
        self.C._pending_animal_arrivals.clear()
        self.C._pending_animal_arrivals.update(self._saved_pending)

    def _fire(self, *, answering):
        with mock.patch.object(self.C, "_answering_a_directed_look", return_value=answering), \
             mock.patch.object(self.C, "_speak_async", return_value=True) as speak:
            self.C._fire_pending_animal_arrival_reaction()
        return speak

    def test_reactive_when_a_look_is_open(self):
        speak = self._fire(answering=True)
        speak.assert_called_once()
        self.assertTrue(speak.call_args.kwargs.get("reactive"))
        self.assertTrue(speak.call_args.kwargs.get("force_salient"))

    def test_not_reactive_otherwise(self):
        # No commanded look → the arrival is unprompted, and keeps the old pacing.
        speak = self._fire(answering=False)
        speak.assert_called_once()
        self.assertFalse(speak.call_args.kwargs.get("reactive"))
        self.assertTrue(speak.call_args.kwargs.get("force_salient"))


class ReportSwallowsTheDuplicateRemarkTest(unittest.TestCase):
    """624bddf made a commanded look report the new view — and that report's vision
    call publishes what it saw into world_state["animals"], which is exactly what
    stages an arrival. The report doesn't race the remark, it CAUSES it. Without
    this guard, "a dog down there" is followed a second later by "Well, hello,
    small furry lifeform": one dog, two lines.

    Note the ordering that makes this work: the duplicate is staged AFTER the
    report, so a fence taken at the report's frame timestamp would miss every case
    it was written for. The mark is read at FIRE time instead.
    """

    def setUp(self):
        from intelligence import consciousness as C
        self.C = C
        self._saved = dict(C._pending_animal_arrivals)
        self._saved_mark = C._directed_look_reported_at
        C._pending_animal_arrivals.clear()
        C._directed_look_reported_at = 0.0
        self.addCleanup(self._restore)
        self.frame_and_line = mock.patch.object(
            C, "_animal_reaction_frame_and_line",
            return_value=(mock.Mock(affect="amused"), "Well, hello, small furry lifeform."),
        )
        self.frame_and_line.start()
        self.addCleanup(self.frame_and_line.stop)

    def _restore(self):
        self.C._pending_animal_arrivals.clear()
        self.C._pending_animal_arrivals.update(self._saved)
        self.C._directed_look_reported_at = self._saved_mark

    def _stage(self, species, at):
        self.C._pending_animal_arrivals[species] = {
            "species": species, "kind": "arrival", "return_count": 0,
            "first_seen_at": at, "last_seen_at": at,
        }

    def _fire(self):
        with mock.patch.object(self.C, "_answering_a_directed_look", return_value=True), \
             mock.patch.object(self.C, "_speak_async", return_value=True) as speak:
            self.C._fire_pending_animal_arrival_reaction()
        return speak

    def test_the_dog_the_report_just_described_stays_quiet(self):
        # The dominant path: staged AFTER the report, off the report's own vision.
        now = time.monotonic()
        self.C.note_directed_look_reported(now)
        self._stage("dog", now + 1.0)
        speak = self._fire()
        speak.assert_not_called()
        self.assertNotIn("dog", self.C._pending_animal_arrivals)

    def test_a_dog_the_local_detector_staged_first_also_stays_quiet(self):
        # The other ordering: the COCO detector beat the report to it.
        now = time.monotonic()
        self._stage("dog", now - 2.0)
        self.C.note_directed_look_reported(now)
        speak = self._fire()
        speak.assert_not_called()

    def test_a_pet_that_wanders_in_later_still_gets_its_line(self):
        # The case the whole fix exists for — and the reason the grace is a window
        # rather than "any arrival after a report is a duplicate forever".
        import config
        now = time.monotonic()
        self.C.note_directed_look_reported(now)
        grace = float(getattr(config, "DIRECTED_LOOK_REPORT_ANIMAL_GRACE_SECS", 6.0))
        self._stage("dog", now + grace + 1.0)
        speak = self._fire()
        speak.assert_called_once()
        self.assertTrue(speak.call_args.kwargs.get("reactive"))

    def test_no_report_means_no_suppression(self):
        now = time.monotonic()
        self._stage("dog", now)
        speak = self._fire()
        speak.assert_called_once()

    def test_suppression_is_silent_and_does_not_burn_the_session_cap(self):
        # Rex did not spend an animal remark here, so the ledger must not say he did.
        now = time.monotonic()
        self.C.note_directed_look_reported(now)
        self._stage("dog", now + 1.0)
        saved_species = dict(self.C._animal_species_reacted_at)
        saved_reacted = dict(self.C._animal_reacted_at)
        try:
            self._fire()
            self.assertEqual(self.C._animal_species_reacted_at, saved_species)
            self.assertEqual(self.C._animal_reacted_at, saved_reacted)
        finally:
            self.C._animal_species_reacted_at.clear()
            self.C._animal_species_reacted_at.update(saved_species)
            self.C._animal_reacted_at.clear()
            self.C._animal_reacted_at.update(saved_reacted)


if __name__ == "__main__":
    unittest.main()
