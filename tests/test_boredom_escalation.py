"""Tests for the boredom → sleep escalation (consciousness._step_boredom_escalation).

Left alone (no HUMAN interaction) for a while, Rex grumbles he's bored; after
BOREDOM_SLEEP_AFTER_SECS of being bored he dozes off into SLEEP. His own grumbling
must NOT reset the clock.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock


class BoredomEscalationTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.c = consciousness
        self._saved = {
            k: getattr(consciousness, k)
            for k in (
                "_boredom_started_at",
                "_last_boredom_comment_at",
                "_boredom_sleeping",
                "_boredom_loop_started_at",
                "_engaged_last_touch_at",
                "_recent_engaged_touch_at",
            )
        }
        self.profile = SimpleNamespace(
            suppress_proactive=False, suppress_system_comments=False
        )

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(self.c, k, v)

    def _set_human_idle(self, idle_secs):
        """Last human engagement was ``idle_secs`` ago."""
        now = time.monotonic()
        self.c._engaged_last_touch_at = 0.0
        self.c._recent_engaged_touch_at = now - idle_secs
        self.c._boredom_loop_started_at = now - idle_secs

    def _run_step(self, *, state=None):
        c = self.c
        state = state if state is not None else c.State.IDLE
        with (
            mock.patch.object(c, "state_module") as sm,
            mock.patch.object(c, "_speak_async") as speak,
            mock.patch.object(c, "_trigger_boredom_sleep") as sleep_trigger,
            mock.patch.object(c, "is_waiting_for_response", return_value=False),
            mock.patch.object(c, "_can_proactive_speak", return_value=True),
            mock.patch.object(c.config, "BOREDOM_ENABLED", True),
            mock.patch.object(c.config, "BOREDOM_ONSET_SECS", 150.0),
            mock.patch.object(c.config, "BOREDOM_SLEEP_AFTER_SECS", 600.0),
            mock.patch.object(c.config, "BOREDOM_COMMENT_INTERVAL_SECS_MIN", 0.0),
            mock.patch.object(c.config, "BOREDOM_COMMENT_INTERVAL_SECS_MAX", 0.0),
        ):
            sm.get_state.return_value = state
            c._step_boredom_escalation({}, self.profile)
        return speak, sleep_trigger

    def test_not_bored_when_recently_engaged(self):
        self.c._boredom_started_at = 0.0
        self.c._boredom_sleeping = False
        self._set_human_idle(5)  # engaged 5s ago; onset is 150
        speak, sleep_trigger = self._run_step()
        speak.assert_not_called()
        sleep_trigger.assert_not_called()
        self.assertEqual(self.c._boredom_started_at, 0.0)

    def test_grumbles_after_onset(self):
        self.c._boredom_started_at = 0.0
        self.c._last_boredom_comment_at = 0.0
        self.c._boredom_sleeping = False
        self._set_human_idle(200)  # idle 200s > onset 150
        speak, sleep_trigger = self._run_step()
        speak.assert_called_once()
        sleep_trigger.assert_not_called()
        self.assertGreater(self.c._boredom_started_at, 0.0)

    def test_dozes_off_after_sleep_threshold(self):
        self._set_human_idle(900)
        self.c._boredom_started_at = time.monotonic() - 601  # bored for >600s
        self.c._boredom_sleeping = False
        speak, sleep_trigger = self._run_step()
        sleep_trigger.assert_called_once()
        self.assertTrue(self.c._boredom_sleeping)
        self.assertEqual(self.c._boredom_started_at, 0.0)

    def test_engagement_resets_boredom(self):
        self.c._boredom_started_at = time.monotonic() - 60  # was bored
        self.c._boredom_sleeping = False
        self._set_human_idle(2)  # but a human just engaged
        speak, sleep_trigger = self._run_step()
        speak.assert_not_called()
        sleep_trigger.assert_not_called()
        self.assertEqual(self.c._boredom_started_at, 0.0)

    def test_no_boredom_outside_idle_state(self):
        self.c._boredom_started_at = time.monotonic() - 60
        self.c._boredom_sleeping = False
        self._set_human_idle(900)
        speak, sleep_trigger = self._run_step(state=self.c.State.ACTIVE)
        speak.assert_not_called()
        sleep_trigger.assert_not_called()
        self.assertEqual(self.c._boredom_started_at, 0.0)

    def test_human_idle_clock_ignores_rex_own_speech(self):
        c = self.c
        now = time.monotonic()
        c._engaged_last_touch_at = 0.0
        c._recent_engaged_touch_at = now - 300  # human silent for 5 min
        c._boredom_loop_started_at = now - 300
        if hasattr(c, "_last_proactive_speech_at"):
            c._last_proactive_speech_at = now - 1  # Rex grumbled 1s ago
        # Rex's own chatter must not reset the human-idle clock.
        self.assertGreaterEqual(c._human_idle_secs(now), 299.0)


if __name__ == "__main__":
    unittest.main()


class BoredomLeanResurrectionTest(BoredomEscalationTest):
    """Field regression 2026-07-07 (owner: 'this should already be coded'): the
    boredom arc rode purpose=idle_monologue / visual_curiosity, which the lean
    brain suppresses — the entire empty-room show (grumbles, room riff, doze-off
    lead-in) silently died when LEAN_BRAIN_ENABLED went live. The arc now uses a
    dedicated 'boredom' purpose exempt from lean suppression and the cadence
    clamp, and is gated to genuinely EMPTY rooms."""

    def test_grumble_uses_dedicated_boredom_purpose(self):
        self.c._boredom_started_at = 0.0
        self.c._last_boredom_comment_at = 0.0
        self.c._boredom_sleeping = False
        self._set_human_idle(200)
        speak, _ = self._run_step()
        self.assertEqual(speak.call_args.kwargs.get("purpose"), "boredom")

    def test_boredom_purpose_survives_lean_suppression(self):
        import config
        suppressed = getattr(config, "LEAN_SUPPRESSED_PROACTIVE_PURPOSES", set())
        self.assertNotIn("boredom", suppressed)
        self.assertNotIn("startup_empty_room", suppressed)  # empty-room one-shot
        clamped = tuple(getattr(config, "PROACTIVE_CADENCE_CLAMP_PURPOSES", ()))
        self.assertNotIn("boredom", clamped)  # self-paced, terminates in SLEEP

    def test_boredom_priority_clears_governor_floor(self):
        from intelligence.action_governor import _PURPOSE_PRIORITIES
        import config
        self.assertGreaterEqual(
            _PURPOSE_PRIORITIES.get("boredom", 0),
            int(getattr(config, "ACTION_GOVERNOR_MIN_SCORE", 20)),
        )

    def test_person_present_resets_the_arc(self):
        # Someone visibly in the room -> no grumbling AT them, clock cleared.
        self.c._boredom_started_at = time.monotonic() - 100.0
        self.c._boredom_sleeping = False
        self._set_human_idle(500)
        c = self.c
        with (
            mock.patch.object(c, "state_module") as sm,
            mock.patch.object(c, "_speak_async") as speak,
            mock.patch.object(c, "_trigger_boredom_sleep") as sleep_trigger,
            mock.patch.object(c, "is_waiting_for_response", return_value=False),
            mock.patch.object(c.config, "BOREDOM_ENABLED", True),
        ):
            sm.get_state.return_value = c.State.IDLE
            snapshot = {"people": [{"face_visible": True}], "crowd": {"count": 1}}
            c._step_boredom_escalation(snapshot, self.profile)
        speak.assert_not_called()
        sleep_trigger.assert_not_called()
        self.assertEqual(self.c._boredom_started_at, 0.0)
