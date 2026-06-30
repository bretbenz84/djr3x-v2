"""Idle-engagement rework (field gripe 2026-06-30: "if I don't say much, R3x only tries a few
times before he lets the conversation die — no jokes, no curiosity, no 'cat's got your tongue?'").

Three pieces: a playful silence-tease mode (previously the directives BANNED announcing the
silence), hobby-grounded re-engagement (use the person's stored Interest profile), and a warmer
give-up. The timing/persistence (presence-aware window + MAX_PER_STRETCH) is config-level.
"""

import unittest

from intelligence import interaction as ix


class IdleDirectiveTest(unittest.TestCase):
    def test_tease_branch_returns_a_silence_tease(self):
        directive, pov = ix._idle_banter_directive(False, True, "", tease=True)
        self.assertIn(directive, ix._IDLE_BANTER_TEASE_SILENCE)
        self.assertFalse(pov)
        # The tease is the ONE mode allowed to call out the dead air.
        self.assertIn("silence", directive.lower() + " ".join(ix._IDLE_BANTER_TEASE_SILENCE).lower())

    def test_tease_pool_has_variety(self):
        self.assertGreaterEqual(len(ix._IDLE_BANTER_TEASE_SILENCE), 2)

    def test_non_tease_stays_earnest(self):
        # No-live-topic ask leads with the Interest profile (hobby grounding).
        d0, _ = ix._idle_banter_directive(True, False, "", tease=False)
        self.assertIs(d0, ix._IDLE_BANTER_DIRECTIVES[0])
        self.assertIn("Interest profile", d0)
        self.assertIn("do not announce the silence", d0)   # earnest modes keep the ban

    def test_live_topic_ask_allows_hobby_pivot(self):
        d, _ = ix._idle_banter_directive(True, True, "", tease=False)
        self.assertIs(d, ix._IDLE_BANTER_LIVE_TOPIC_ASK)
        self.assertIn("Interest profile", d)               # may pivot to a known hobby

    def test_earnest_directives_still_ban_announcing_silence(self):
        # The tease is the ONLY exception; every other idle directive keeps the ban.
        for d in ix._IDLE_BANTER_DIRECTIVES:
            self.assertIn("announce the silence", d)
        self.assertIn("announce the silence", ix._IDLE_BANTER_LIVE_TOPIC_ASK)
        for d in ix._IDLE_BANTER_TEASE_SILENCE:
            self.assertNotIn("do not announce", d.lower())


class IdleConfigTest(unittest.TestCase):
    def test_persistence_and_tease_config(self):
        import config
        self.assertGreaterEqual(config.IDLE_BANTER_MAX_PER_STRETCH, 3)
        self.assertEqual(config.IDLE_BANTER_TEASE_SILENCE_AT, 2)
        self.assertGreater(config.PRESENT_REENGAGE_IDLE_TIMEOUT_SECS,
                           config.CONVERSATION_IDLE_TIMEOUT_SECS)
        # Warmer give-up options were added alongside the cold ones.
        self.assertGreaterEqual(len(config.IDLE_OUTRO_LINES), 5)


if __name__ == "__main__":
    unittest.main()
