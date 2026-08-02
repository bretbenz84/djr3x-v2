import unittest


class CommandParserBoundaryTests(unittest.TestCase):
    def test_reported_or_meta_command_phrases_do_not_parse(self):
        from intelligence import command_parser

        cases = [
            "I said, look down",
            "look down is a phrase",
            "remember that time we played trivia",
            "I asked you to set humor to maximum",
            "when you set humor to maximum it gets annoying",
            "I asked what is your sarcasm level",
            "call me when you arrive",
            "call me maybe",
            "I want to play trivia later",
            "can we play trivia later",
            "let's play it by ear",
            "play something later",
            "Nope, my partner is in the hospital.",
            "Actually, my partner is in the hospital.",
            # Stray game mentions in narration must NOT start a game (these used to fire
            # the canned games list / re-trigger start).
            "it was just the 20 questions idea that I had",
            "we used to play 20 questions",
            "I don't want to play 20 questions",
        ]

        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(command_parser.parse(text))

    def test_direct_command_variants_still_parse(self):
        from intelligence import command_parser

        cases = {
            "Rex, look down": "directed_look",
            "okay, look down": "directed_look",
            "Guess what? Stop playing with me. Look down.": "directed_look",
            "remember that Jennifer hates being called Jenny.": "memory_remember_fact",
            "set sarcasm to high please": "set_personality",
            "please set sarcasm to high": "set_personality",
            "could you set sarcasm to high": "set_personality",
            "what is your sarcasm level?": "query_personality",
            "please tell me what is your sarcasm level": "query_personality",
            "call me Bret": "rename_me",
            "I want to play trivia": "start_game",
            "play something upbeat": "dj_play_vibe",
            # Game-start must survive a leading clause and a bare game name (the routing
            # that previously fell through to the canned "here are my games" list).
            "let's play 20 questions": "start_game",
            "I'm good, but let's play 20 questions": "start_game",
            "okay so let's play trivia": "start_game",
            "20 questions": "start_game",
            "twenty questions": "start_game",
        }

        for text, command_key in cases.items():
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, command_key)


class ShutdownVsSleepSplitTests(unittest.TestCase):
    """Full shutdown (exit main.py) is distinct from sleep (stay alive)."""

    def test_standalone_shutdown_phrases(self):
        from intelligence import command_parser as cp

        for text in (
            "shut down",
            "shut down rex",
            "shutdown",
            "shutdown rex",
            "power down",
            "power down rex",
            "power off rex",
            "turn yourself off",
            "rex shut down please",
            "shut down now",
        ):
            with self.subTest(text=text):
                self.assertTrue(cp.is_standalone_shutdown_command(text))
                self.assertFalse(cp.is_standalone_sleep_command(text))
                match = cp.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "shutdown")

    def test_sleep_phrases_are_not_shutdown(self):
        from intelligence import command_parser as cp

        for text in ("go to sleep", "sleep", "rex go to sleep", "go to sleep please"):
            with self.subTest(text=text):
                self.assertTrue(cp.is_standalone_sleep_command(text))
                self.assertFalse(cp.is_standalone_shutdown_command(text))
                match = cp.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "sleep")

    def test_shutdown_narration_and_scoped_phrases_do_not_trigger_shutdown(self):
        from intelligence import command_parser as cp

        # Embedded narration / scoped "shut down X" must NOT power off the droid.
        for text in (
            "I had to shut down my old server yesterday",
            "can you shut down the music",
            "shut down the music",
            "the reactor will shut down if it overheats",
        ):
            with self.subTest(text=text):
                self.assertFalse(cp.is_standalone_shutdown_command(text))
                match = cp.parse(text)
                self.assertFalse(match is not None and match.command_key == "shutdown")

    def test_shutdown_wake_confirmation_accepts_whisper_homophones(self):
        """The acoustic shut_down wake model already heard the phrase; a
        transcript spelling it "Cut down." must confirm, not veto (live
        2026-07-30: wake at 0.945 was ignored and Rex quipped instead)."""
        from intelligence import command_parser as cp

        for text in ("Cut down.", "Shot down.", "shut town", "Shut down.", "shut it down"):
            with self.subTest(text=text):
                self.assertTrue(cp.is_shutdown_wake_confirmation(text))
        # Homophones stay wake-confirm-only, never general shutdown commands.
        self.assertFalse(cp.is_standalone_shutdown_command("Cut down."))

    def test_shutdown_wake_confirmation_still_rejects_non_shutdown(self):
        from intelligence import command_parser as cp

        for text in (
            "Look down.",
            "sit down",
            "don't shut down",
            "cut down the music",
            "I had to cut down a tree",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(cp.is_shutdown_wake_confirmation(text))

    def test_embedded_and_prefixed_shutdown_phrases(self):
        from intelligence import command_parser as cp

        # A direct "shut down" clause should fire even when trailing frustration
        # or another clause ("Shut up, Shut down" — the live-logged failure).
        for text in (
            "Shut up, Shut down",
            "shut up shut down",
            "wait, shut down",
            "stop talking and shut down",
            "okay shut down",
            "please just shut down",
            "shut up, shut down now",
        ):
            with self.subTest(text=text):
                self.assertTrue(cp.is_standalone_shutdown_command(text))
                match = cp.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "shutdown")

    def test_negated_and_interrogative_shutdown_phrases_do_not_trigger(self):
        from intelligence import command_parser as cp

        # Destructive action: negations/questions/hypotheticals must never fire.
        for text in (
            "don't shut down",
            "no don't shut down",
            "why would I shut you down",
            "why would I shut down rex",
            "should I shut down",
            "shut up",
            "wait",
        ):
            with self.subTest(text=text):
                self.assertFalse(cp.is_standalone_shutdown_command(text))

    def test_polite_shutdown_requests_accepted_by_tool_router_backstop(self):
        # Field 2026-08-02: "Can you shut down, please?" — the deterministic
        # classifier rejects "can you..." on purpose, but the LLM tool-router
        # path verifies with is_shutdown_request, which accepts polite direct
        # requests while still rejecting object-scoped/negated forms.
        from intelligence import command_parser as cp

        for text in (
            "Can you shut down, please?",
            "could you power off now",
            "would you shut down",
            "Rex, shut down.",
        ):
            with self.subTest(text=text):
                self.assertTrue(cp.is_shutdown_request(text))

        for text in (
            "can you shut down the music",
            "don't shut down",
            "why would I shut down",
            "should I shut down",
            "can you believe my server shut down",
        ):
            with self.subTest(text=text):
                self.assertFalse(cp.is_shutdown_request(text))

    def test_polite_sleep_requests(self):
        from intelligence import command_parser as cp

        self.assertTrue(cp.is_sleep_request("can you go to sleep, please?"))
        self.assertTrue(cp.is_sleep_request("go to sleep"))
        self.assertFalse(cp.is_sleep_request("I could not sleep last night"))
        self.assertFalse(cp.is_sleep_request("can you play a sleep playlist"))


if __name__ == "__main__":
    unittest.main()
