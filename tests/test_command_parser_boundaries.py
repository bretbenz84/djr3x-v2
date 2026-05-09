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


if __name__ == "__main__":
    unittest.main()
