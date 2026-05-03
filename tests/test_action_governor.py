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


if __name__ == "__main__":
    unittest.main()
