"""
Identity-prompt startup grace. A known face reads as "unknown" for the tick or two
recognition needs to resolve at startup, so Rex fired "I don't know you yet — what's
your name?" one tick before recognizing Bret (whose name was already on the GUI
overlay). _maybe_prompt_unknown_identity now requires a solo-unknown face to PERSIST
past IDENTITY_PROMPT_UNKNOWN_GRACE_SECS before prompting.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from intelligence import consciousness as c


class IdentityPromptGraceTest(unittest.TestCase):
    def setUp(self):
        c._solo_unknown_since = 0.0
        c._pending_identity_prompt.clear()
        c._identity_prompt_in_flight.clear()
        c._last_identity_prompt_at = 0.0

    def tearDown(self):
        # The prompt path sets _identity_prompt_in_flight; its on_done (which clears it)
        # is mocked away, so reset module state to avoid leaking into other suites.
        c._solo_unknown_since = 0.0
        c._pending_identity_prompt.clear()
        c._identity_prompt_in_flight.clear()
        c._last_identity_prompt_at = 0.0

    def _ctx(self):
        return (
            mock.patch.object(c.state_module, "get_state", return_value=c.State.IDLE),
            mock.patch.object(c, "_can_proactive_speak", return_value=True),
            mock.patch.object(c, "_speak_async", return_value=True),
        )

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


if __name__ == "__main__":
    unittest.main()
