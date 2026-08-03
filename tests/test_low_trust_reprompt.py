"""Low-trust reprompt gating (field 2026-08-01/02: garbled decodes got bluffed
replies — "I'm not a cat." at logprob -1.88 was answered with a quip about
mystery voices instead of a human "sorry, what?").

The decision helper is pure enough to test directly: trust flag, word floor,
once-per-exchange cooldown, game suppression, and the motion/stop exemption.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import interaction


class LowTrustRepromptGateTest(unittest.TestCase):
    def setUp(self):
        interaction._last_low_trust_reprompt_at = 0.0
        interaction._last_low_trust_reprompt_line = ""
        self._game_patch = mock.patch.object(
            interaction, "_game_suppresses_conversation", return_value=False
        )
        self._game_patch.start()
        self.addCleanup(self._game_patch.stop)

    def _should(self, text, *, trusted=False, text_input=False):
        return interaction._should_reprompt_low_trust(
            text, trusted=trusted, text_input=text_input
        )

    def test_substantial_garble_triggers_reprompt(self):
        self.assertTrue(self._should("I'm not a cat."))
        self.assertTrue(self._should("You kill everybody."))

    def test_trusted_transcript_never_reprompts(self):
        self.assertFalse(self._should("I'm not a cat.", trusted=True))

    def test_typed_text_never_reprompts(self):
        self.assertFalse(self._should("I'm not a cat.", text_input=True))

    def test_short_backchannels_skip_reprompt(self):
        for text in ("Okay.", "Yeah.", "Scared.", "I'm sorry."):
            self.assertFalse(self._should(text), text)

    def test_once_per_exchange_cooldown(self):
        self.assertTrue(self._should("I'm not a cat."))
        interaction._arm_low_trust_reprompt_cooldown()
        # The repeat also came back low-trust: engage best-effort, don't loop.
        self.assertFalse(self._should("I'm still not a cat."))
        # After the cooldown lapses, a fresh garble may be re-asked.
        interaction._last_low_trust_reprompt_at = (
            time.monotonic()
            - float(getattr(config, "LOW_TRUST_REPROMPT_COOLDOWN_SECS", 120.0))
            - 1.0
        )
        self.assertTrue(self._should("A whole new garble arrived."))

    def test_game_mode_skips_reprompt(self):
        with mock.patch.object(
            interaction, "_game_suppresses_conversation", return_value=True
        ):
            self.assertFalse(self._should("I'm not a cat."))

    def test_motion_commands_execute_instead_of_reprompting(self):
        # An explicit drive command acts; asking "sorry, what?" would stall it.
        self.assertFalse(self._should("turn to your left a little bit"))

    def test_bare_stop_never_reprompts(self):
        self.assertFalse(self._should("stop stop stop"))

    def test_kill_switch(self):
        with mock.patch.object(config, "LOW_TRUST_REPROMPT_ENABLED", False):
            self.assertFalse(self._should("I'm not a cat."))

    def test_reprompt_lines_rotate(self):
        seen = {interaction._low_trust_reprompt_line() for _ in range(12)}
        self.assertGreater(len(seen), 1)
        picks = [interaction._low_trust_reprompt_line() for _ in range(8)]
        for a, b in zip(picks, picks[1:]):
            self.assertNotEqual(a, b, "consecutive reprompt lines must differ")


if __name__ == "__main__":
    unittest.main()
