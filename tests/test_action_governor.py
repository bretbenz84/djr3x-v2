import unittest


class ActionGovernorScopeTests(unittest.TestCase):
    def test_governor_selects_proactive_candidate(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        governor.start_cycle()
        governor.observe(CandidateMove(
            source="_step_idle_micro_behavior",
            purpose="idle_monologue",
            suggested_text="Empty room joke.",
            priority=50,
        ))

        decision = governor.finish_cycle()

        self.assertEqual(decision.action, "speak")
        self.assertEqual(decision.selected.candidate.purpose, "idle_monologue")
        self.assertFalse(decision.selected.rejected)

    def test_governor_rejects_non_proactive_candidate(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        governor.start_cycle()
        governor.observe(CandidateMove(
            source="interaction._handle_router_takeover_action",
            purpose="humor.tell_joke",
            kind="user_turn",
            suggested_text="Tell one joke.",
            priority=100,
        ))

        decision = governor.finish_cycle()

        self.assertEqual(decision.action, "wait")
        self.assertIsNone(decision.selected)
        scored = decision.scored[0]
        self.assertTrue(scored.rejected)
        self.assertIn("non_proactive_candidate", scored.reasons)

    def test_candidate_default_kind_is_proactive(self):
        from intelligence.action_governor import CandidateMove, PROACTIVE_CANDIDATE_KIND

        candidate = CandidateMove(source="_step_small_talk", purpose="small_talk")

        self.assertEqual(candidate.kind, PROACTIVE_CANDIDATE_KIND)

    def test_lower_priority_candidate_records_skip_reason(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        governor.start_cycle()
        governor.observe(CandidateMove(
            source="_step_idle_micro_behavior",
            purpose="idle_monologue",
            suggested_text="Empty room joke.",
            priority=25,
            metadata={"topic_key": "idle-room"},
        ))
        governor.observe(CandidateMove(
            source="_step_emotional_checkin",
            purpose="emotional_checkin",
            suggested_text="Check in softly.",
            priority=100,
            metadata={"topic_key": "empathy"},
        ))

        decision = governor.finish_cycle()

        self.assertEqual(decision.action, "speak")
        self.assertEqual(decision.selected.candidate.purpose, "emotional_checkin")
        skipped = [
            item for item in decision.scored
            if item.candidate.purpose == "idle_monologue"
        ][0]
        self.assertFalse(skipped.rejected)
        self.assertFalse(skipped.selected)
        self.assertIn(
            "lower_priority_than_selected:emotional_checkin",
            skipped.skip_reasons,
        )

    def test_duplicate_topic_candidate_records_skip_reason(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        governor.start_cycle()
        governor.observe(CandidateMove(
            source="_step_group_lull",
            purpose="group_turn_invite",
            label="invite Jeff",
            suggested_text="Jeff, your move.",
            priority=70,
            target_person_id=42,
            metadata={"topic_key": "turn-invite:42"},
        ))
        governor.observe(CandidateMove(
            source="_step_group_turn_taking",
            purpose="group_turn_invite",
            label="invite Jeff again",
            suggested_text="Jeff, care to jump in?",
            priority=68,
            target_person_id=42,
            metadata={"topic_key": "turn-invite:42"},
        ))

        decision = governor.finish_cycle()

        duplicate = [
            item for item in decision.scored
            if item.candidate.source == "_step_group_turn_taking"
        ][0]
        self.assertTrue(duplicate.rejected)
        self.assertFalse(duplicate.selected)
        self.assertIn("duplicate_topic", duplicate.reasons)
        self.assertIn("duplicate_topic", duplicate.skip_reasons)

    def test_specific_gate_reasons_are_recorded(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        class Profile:
            user_mid_sentence = True
            interaction_busy = False
            suppress_proactive = False
            rapid_exchange = False
            conversation_active = False
            force_family_safe = True

        governor = ActionGovernor()
        governor.start_cycle(profile=Profile())
        governor.observe(CandidateMove(
            source="_step_idle_micro_behavior",
            purpose="idle_monologue",
            suggested_text="A definitely not-for-kids bit.",
            priority=80,
            metadata={
                "cooldown_active": True,
                "cooldown_reason": "idle_monologue_cooldown",
                "cooldown_remaining_secs": 4.25,
                "output_gate_busy": True,
                "family_safe": False,
            },
        ))

        decision = governor.finish_cycle()

        scored = decision.scored[0]
        self.assertEqual(decision.action, "wait")
        self.assertTrue(scored.rejected)
        self.assertIn("user_mid_sentence", scored.reasons)
        self.assertIn("child_present_family_safe_block", scored.reasons)
        self.assertIn("output_gate_busy", scored.reasons)
        self.assertIn("idle_monologue_cooldown_4.2s", scored.reasons)
        self.assertIn("output_gate_busy", scored.skip_reasons)

    def test_candidate_log_includes_skip_fields(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        with self.assertLogs("intelligence.action_governor", level="INFO") as logs:
            governor.start_cycle()
            governor.observe(CandidateMove(
                source="_step_idle_micro_behavior",
                purpose="idle_monologue",
                suggested_text="Empty room joke.",
                priority=25,
                metadata={"topic_key": "idle-room"},
            ))
            governor.observe(CandidateMove(
                source="_step_emotional_checkin",
                purpose="emotional_checkin",
                suggested_text="Check in softly.",
                priority=100,
                metadata={"topic_key": "empathy"},
            ))
            governor.finish_cycle()

        joined = "\n".join(logs.output)
        self.assertIn("selected=", joined)
        self.assertIn("skipped=True", joined)
        self.assertIn("skip_reasons=lower_priority_than_selected:emotional_checkin", joined)

    def test_weather_proactive_comment_is_named_candidate_type(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove

        governor = ActionGovernor()
        governor.start_cycle()
        governor.observe(CandidateMove(
            source="_step_proactive_reactions",
            purpose="weather.proactive_comment",
            label="weather changed",
            suggested_text="Weather feed says rain.",
        ))

        decision = governor.finish_cycle()

        self.assertEqual(decision.action, "speak")
        self.assertEqual(decision.selected.candidate.purpose, "weather.proactive_comment")
        self.assertGreaterEqual(decision.selected.score, 40)


class GovernorEnforcementTests(unittest.TestCase):
    """Enforce mode (consolidation): the governor becomes the single decider —
    candidates carry a deferred `speak_fn` and only the tick's WINNER's runs."""

    def test_enforcing_property_reflects_config_and_activates(self):
        from unittest import mock
        from intelligence import action_governor as ag
        gov = ag.ActionGovernor()
        with mock.patch.object(ag.config, "ACTION_GOVERNOR_ENFORCE", True):
            self.assertTrue(gov.enforcing)
            self.assertTrue(gov.active())  # enforcing keeps the cycle collecting
        with mock.patch.object(ag.config, "ACTION_GOVERNOR_ENFORCE", False):
            self.assertFalse(gov.enforcing)

    def test_only_winner_carries_the_runnable_speak_fn(self):
        from intelligence.action_governor import ActionGovernor, CandidateMove
        calls = []
        gov = ActionGovernor()
        gov.start_cycle()
        gov.observe(CandidateMove(
            source="_step_a", purpose="visual_curiosity", priority=50,
            speak_fn=lambda: calls.append("loser")))
        gov.observe(CandidateMove(
            source="_step_b", purpose="emotional_checkin", priority=100,
            speak_fn=lambda: calls.append("winner")))
        decision = gov.finish_cycle()
        self.assertEqual(decision.action, "speak")
        self.assertEqual(decision.selected.candidate.priority, 100)
        # The cycle resolver (consciousness._finish_governor_cycle) runs ONLY the
        # winner's speak_fn — the loser stays silent.
        decision.selected.candidate.speak_fn()
        self.assertEqual(calls, ["winner"])


class GovernorCrossThreadIntakeTests(unittest.TestCase):
    """Increment 2: a candidate from a non-consciousness thread (idle banter,
    memory follow-ups) is submitted to a shared buffer and drained into the next
    consciousness tick, so one decider arbitrates ALL proactive speech."""

    def _clear(self, ag):
        with ag._external_lock:
            ag._external_candidates.clear()

    def test_external_candidate_drained_and_arbitrated_when_enforcing(self):
        from unittest import mock
        from intelligence import action_governor as ag
        gov = ag.ActionGovernor()
        with mock.patch.object(ag.config, "ACTION_GOVERNOR_ENFORCE", True):
            self._clear(ag)
            gov.submit_external(ag.CandidateMove(
                source="interaction._maybe_idle_banter", purpose="idle_monologue",
                priority=50, speak_fn=lambda: None))
            gov.start_cycle()            # drains the external candidate into the cycle
            decision = gov.finish_cycle()
            self.assertIsNotNone(decision)
            self.assertEqual(decision.action, "speak")
            self.assertEqual(decision.selected.candidate.source,
                             "interaction._maybe_idle_banter")

    def test_submit_external_is_noop_when_not_enforcing(self):
        from unittest import mock
        from intelligence import action_governor as ag
        gov = ag.ActionGovernor()
        with mock.patch.object(ag.config, "ACTION_GOVERNOR_ENFORCE", False):
            self._clear(ag)
            gov.submit_external(ag.CandidateMove(source="x", purpose="idle_monologue"))
            with ag._external_lock:
                self.assertEqual(ag._external_candidates, [])

    def test_stale_external_candidates_are_dropped(self):
        from unittest import mock
        from intelligence import action_governor as ag
        gov = ag.ActionGovernor()
        with mock.patch.object(ag.config, "ACTION_GOVERNOR_ENFORCE", True):
            self._clear(ag)
            stale = ag.CandidateMove(source="x", purpose="idle_monologue", priority=50)
            stale.created_at -= (ag._EXTERNAL_CANDIDATE_TTL_SECS + 1.0)
            gov.submit_external(stale)
            gov.start_cycle()
            self.assertIsNone(gov.finish_cycle())  # stale dropped → empty cycle


if __name__ == "__main__":
    unittest.main()
