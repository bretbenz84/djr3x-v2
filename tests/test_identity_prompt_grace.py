"""
Identity-prompt startup grace + silent-stranger recovery.

Grace: a known face reads as "unknown" for the tick or two recognition needs to
resolve at startup, so Rex fired "I don't know you yet — what's your name?" one tick
before recognizing Bret. _maybe_prompt_unknown_identity requires a solo-unknown face
to PERSIST past IDENTITY_PROMPT_UNKNOWN_GRACE_SECS before prompting.

Recovery (field log 2026-07-06-19-20): speak_async returns True on governor
SUBMISSION, not on speech. A governor-REJECTED candidate (e.g. the 5s
situation-suppression window right after ACTIVE->IDLE) never ran speak_fn/on_done, so
the in-flight latch stayed set and the 45s cooldown was armed — a silent unknown
visitor got NOTHING for the rest of the session. Now: the cooldown arms only when
the line actually speaks (on_spoke), a stale latch (>IDENTITY_PROMPT_INFLIGHT_STALE_SECS)
is cleared and the ask retried, and the ask is salient so it can fire during the
ACTIVE boot window.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from intelligence import consciousness as c


class IdentityPromptTestBase(unittest.TestCase):
    def setUp(self):
        self._reset()

    def tearDown(self):
        # The prompt path sets _identity_prompt_in_flight; its on_done (which clears it)
        # is mocked away, so reset module state to avoid leaking into other suites.
        self._reset()

    @staticmethod
    def _reset():
        c._solo_unknown_since = 0.0
        c._pending_identity_prompt.clear()
        c._identity_prompt_in_flight.clear()
        c._identity_prompt_in_flight_at = 0.0
        c._last_identity_prompt_at = 0.0

    def _ctx(self, state=None):
        return (
            mock.patch.object(c.state_module, "get_state",
                              return_value=state or c.State.IDLE),
            mock.patch.object(c, "_can_proactive_speak", return_value=True),
            mock.patch.object(c, "_speak_async", return_value=True),
        )


class IdentityPromptGraceTest(IdentityPromptTestBase):
    def test_first_unknown_tick_does_not_prompt(self):
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            c._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
            self.assertFalse(speak.called)          # within grace — no prompt
            self.assertGreater(c._solo_unknown_since, 0.0)  # grace clock started

    def test_known_face_resolving_resets_and_never_prompts(self):
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            c._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])   # tick 1: unknown
            c._maybe_prompt_unknown_identity(unknown_count=0, known_unique=["Bret Benziger"])  # tick 2: recognized
            self.assertEqual(c._solo_unknown_since, 0.0)
            self.assertFalse(speak.called)

    def test_persistent_unknown_past_grace_prompts(self):
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            c._solo_unknown_since = time.monotonic() - 99.0  # has persisted well past grace
            c._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
            self.assertTrue(speak.called)


class SilentStrangerRecoveryTest(IdentityPromptTestBase):
    """The 2026-07-06-19-20 failure chain, pinned."""

    def _prompt(self, speak):
        c._solo_unknown_since = time.monotonic() - 99.0
        c._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
        return speak

    def test_fires_during_active_state(self):
        # The boot window is ACTIVE for ~60s — the silent stranger must not wait it out.
        s, p, spk = self._ctx(state=c.State.ACTIVE)
        with s, p, spk as speak:
            self._prompt(speak)
            self.assertTrue(speak.called)

    def test_ask_is_salient(self):
        s, p, spk = self._ctx(state=c.State.ACTIVE)
        with s, p as can_speak, spk as speak:
            self._prompt(speak)
        can_speak.assert_called_with(salient=True)
        self.assertTrue(speak.call_args.kwargs.get("force_salient"))

    def test_rejected_candidate_does_not_burn_cooldown(self):
        # speak_async returning True = SUBMITTED, not spoken. Cooldown must only arm
        # via on_spoke (which a rejected candidate never calls).
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            self._prompt(speak)
            self.assertEqual(c._last_identity_prompt_at, 0.0)
            # When the line actually speaks, on_spoke arms it.
            speak.call_args.kwargs["on_spoke"]()
            self.assertGreater(c._last_identity_prompt_at, 0.0)

    def test_stale_inflight_latch_recovers_and_reasks(self):
        # Governor rejected a previous ask: latch set, nothing cleared it.
        c._identity_prompt_in_flight.set()
        c._identity_prompt_in_flight_at = time.monotonic() - 30.0  # > stale window
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            self._prompt(speak)
            self.assertTrue(speak.called)           # recovered and re-asked

    def test_fresh_inflight_latch_still_blocks(self):
        c._identity_prompt_in_flight.set()
        c._identity_prompt_in_flight_at = time.monotonic() - 1.0   # genuinely in flight
        s, p, spk = self._ctx()
        with s, p, spk as speak:
            self._prompt(speak)
            self.assertFalse(speak.called)


if __name__ == "__main__":
    unittest.main()
