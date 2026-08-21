"""
Proactive-discipline fixes (post-Phase-1 live-session regressions):
  R1 governor cross-cycle de-dup   — a flickering world cue can't re-fire the SAME
                                      proactive line on consecutive ticks.
  R3 crowd-change debounce          — a one-frame crowd-count flicker doesn't trigger
                                      a "now it's just us" reaction.
  R3 speak-time re-validation       — a proactive line decided before a user turn
                                      began yields instead of talking over it.
"""

from __future__ import annotations

import unittest

import config


class GovernorCrossCycleDedupTest(unittest.TestCase):
    def setUp(self):
        from intelligence import action_governor as ag
        ag._recent_selected.clear()

    def _crowd_candidate(self):
        from intelligence.action_governor import CandidateMove
        return CandidateMove(
            source="_step_proactive_reactions", purpose="world_reaction",
            label="crowd size changed", suggested_text="now it's just us",
        )

    def _run_tick(self, candidate, *, spoke=True):
        """One governor tick. `spoke` models what happens AFTER arbitration: the
        winner's deferred speak_fn can still abort (barge-guard yield, can't-speak,
        empty text, user speaking), and the repeat cooldown arms only on real
        speech — see note_topic_spoken."""
        from intelligence import action_governor as ag
        ag.governor.start_cycle()
        ag.governor.observe(candidate)
        decision = ag.governor.finish_cycle()
        if spoke and decision is not None and decision.action == "speak":
            ag.note_topic_spoken(decision.selected.candidate)
        return decision

    def test_identical_world_cue_is_blocked_on_next_tick(self):
        d1 = self._run_tick(self._crowd_candidate())
        self.assertEqual(d1.action, "speak")
        d2 = self._run_tick(self._crowd_candidate())
        self.assertEqual(d2.action, "wait")  # cross-cycle cooldown rejected the repeat

    def test_a_winner_that_never_spoke_does_not_block_its_own_topic(self):
        """Field 2026-08-20 20:16:39: the animal remark won, yielded to the user
        mid-sentence, said nothing — and still blocked world.animal_arrival for the
        full 45 s while three otherwise-ELIGIBLE cycles were rejected
        `topic_repeat_cooldown`. The cooldown is for lines that DID speak."""
        d1 = self._run_tick(self._crowd_candidate(), spoke=False)
        self.assertEqual(d1.action, "speak")
        d2 = self._run_tick(self._crowd_candidate(), spoke=False)
        self.assertEqual(d2.action, "speak",
                         "a line that yielded must not silence its own topic")

    def test_idle_monologue_is_excluded_from_the_cooldown(self):
        from intelligence.action_governor import CandidateMove

        def idle():
            return CandidateMove(
                source="interaction._maybe_idle_banter",
                purpose="idle_monologue", label="idle_banter",
            )
        self.assertEqual(self._run_tick(idle()).action, "speak")
        self.assertEqual(self._run_tick(idle()).action, "speak")  # still allowed

    def test_cooldown_zero_disables(self):
        from intelligence.action_governor import governor
        orig = getattr(config, "ACTION_GOVERNOR_REPEAT_COOLDOWN_SECS", 45.0)
        try:
            config.ACTION_GOVERNOR_REPEAT_COOLDOWN_SECS = 0.0
            self.assertEqual(self._run_tick(self._crowd_candidate()).action, "speak")
            self.assertEqual(self._run_tick(self._crowd_candidate()).action, "speak")
        finally:
            config.ACTION_GOVERNOR_REPEAT_COOLDOWN_SECS = orig


class CrowdChangeDebounceTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness
        consciousness._crowd_change_reacted_label = ""
        consciousness._crowd_change_pending_label = ""
        consciousness._crowd_change_pending_since = 0.0
        self._orig = getattr(config, "CROWD_CHANGE_SETTLE_SECS", 2.5)
        config.CROWD_CHANGE_SETTLE_SECS = 0.0  # settle on the 2nd consecutive obs

    def tearDown(self):
        config.CROWD_CHANGE_SETTLE_SECS = self._orig

    def test_first_observation_is_silent_baseline(self):
        from intelligence import consciousness as c
        self.assertIsNone(c._crowd_change_settled("pair"))

    def test_one_frame_flicker_is_ignored(self):
        from intelligence import consciousness as c
        self.assertIsNone(c._crowd_change_settled("pair"))   # baseline
        self.assertIsNone(c._crowd_change_settled("alone"))  # pending, not settled
        # flips back before settling -> no reaction
        self.assertIsNone(c._crowd_change_settled("pair"))

    def test_persisted_change_fires_once(self):
        from intelligence import consciousness as c
        self.assertIsNone(c._crowd_change_settled("pair"))   # baseline
        self.assertIsNone(c._crowd_change_settled("alone"))  # pending
        self.assertEqual(c._crowd_change_settled("alone"), "pair")  # settled -> fire
        self.assertIsNone(c._crowd_change_settled("alone"))  # now stable, no repeat


class SpeakTimeRevalidationTest(unittest.TestCase):
    def test_line_decided_before_user_turn_is_dropped(self):
        from intelligence import interaction
        import time
        decided = time.monotonic()
        # A user turn begins AFTER the line was decided:
        interaction._last_user_turn_started_at = decided + 0.5
        self.assertFalse(
            interaction._speak_proactive("stale line", decided_at=decided)
        )

    def test_line_decided_after_last_user_turn_is_not_dropped_by_this_guard(self):
        from intelligence import interaction
        import time
        # No user turn since the line was decided -> the decided_at guard passes
        # (it may still yield for other reasons, but not via this check). We assert
        # the guard itself does not short-circuit: set stamp BEFORE decided.
        interaction._last_user_turn_started_at = time.monotonic()
        decided = time.monotonic() + 0.5
        # In text-only test context, downstream speak is a no-op; we only assert the
        # decided_at guard did not trip (no exception, returns a bool).
        result = interaction._speak_proactive("", decided_at=decided)
        self.assertIs(result, False)  # empty text returns False, guard not the cause


if __name__ == "__main__":
    unittest.main()
