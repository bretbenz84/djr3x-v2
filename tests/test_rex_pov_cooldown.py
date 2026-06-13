"""
Phase 0.4 — preoccupation spoken-cooldown (S1): after Rex actually utters his
current preoccupation, it must not be re-volunteered near-verbatim within the
cooldown window. Regression for the live double-utterance ("organics power down...
design flaw" twice in ~33s).
"""

from __future__ import annotations

import unittest


class PovSpokenCooldownTest(unittest.TestCase):
    def setUp(self):
        from intelligence import rex_pov
        rex_pov.clear()

    def tearDown(self):
        from intelligence import rex_pov
        rex_pov.clear()

    def _install_active(self):
        from intelligence import rex_pov
        rex_pov._active = rex_pov._ActivePov(
            seed_id="sleep-is-a-bug",
            pov="organics power down for a third of their lives",
            selected_at_exchange=0,
            context_sig=frozenset(),
        )

    def test_no_active_is_never_recently_spoken(self):
        from intelligence import rex_pov
        self.assertFalse(rex_pov.pov_recently_spoken())

    def test_unspoken_active_is_not_recently_spoken(self):
        from intelligence import rex_pov
        self._install_active()
        self.assertFalse(rex_pov.pov_recently_spoken())

    def test_spoken_is_within_cooldown_then_expires(self):
        from intelligence import rex_pov
        self._install_active()
        rex_pov.note_pov_spoken()
        self.assertTrue(rex_pov.pov_recently_spoken(window_secs=180.0))
        # A zero-length window means the cooldown has already elapsed.
        self.assertFalse(rex_pov.pov_recently_spoken(window_secs=0.0))

    def test_clear_resets_spoken_state(self):
        from intelligence import rex_pov
        self._install_active()
        rex_pov.note_pov_spoken()
        rex_pov.clear()
        self.assertFalse(rex_pov.pov_recently_spoken())


if __name__ == "__main__":
    unittest.main()
