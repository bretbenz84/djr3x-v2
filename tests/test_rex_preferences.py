import unittest


class RexPreferenceTests(unittest.TestCase):
    def test_favorite_and_compare_queries_parse(self):
        from intelligence import rex_preferences

        favorite = rex_preferences.extract_preference_query("what's your favorite color?")
        compare = rex_preferences.extract_preference_query("do you prefer jazz or silence?")

        self.assertEqual(favorite["mode"], "favorite")
        self.assertEqual(favorite["domain"], "color")
        self.assertEqual(compare["mode"], "compare")
        self.assertEqual(compare["options"], ["jazz", "silence"])


if __name__ == "__main__":
    unittest.main()
