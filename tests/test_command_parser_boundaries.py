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


if __name__ == "__main__":
    unittest.main()
