import unittest
from unittest import mock


class RexPreferenceTests(unittest.TestCase):
    def test_music_preference_has_stable_positive_motion(self):
        from intelligence import rex_preferences

        reply = rex_preferences.answer_preference_query("do you like music?")

        self.assertEqual(reply.text, "Mmhmm. Music gets the premium circuits.")
        self.assertEqual(reply.emotion, "excited")
        self.assertEqual(reply.body_beat, "giddy_wiggle")
        self.assertEqual(reply.stance, "strong_like")

    def test_silence_preference_uses_strong_negative_reaction(self):
        from intelligence import rex_preferences

        reply = rex_preferences.answer_preference_query("do you like silence?")

        self.assertEqual(reply.text, "Hell to the no. Silence is just a failed soundcheck.")
        self.assertEqual(reply.body_beat, "disgust_recoil")
        self.assertEqual(reply.emotion, "angry")

    def test_child_present_softens_strong_negative_wording(self):
        from intelligence import rex_preferences

        with mock.patch.object(
            rex_preferences.world_state,
            "get",
            return_value=[{"age_estimate": "child"}],
        ):
            reply = rex_preferences.answer_preference_query("do you like silence?")

        self.assertEqual(reply.text, "Absolutely not. Silence is just a failed soundcheck.")

    def test_unknown_topic_is_stable(self):
        from intelligence import rex_preferences

        one = rex_preferences.answer_preference_query("do you like hydrospanners?")
        two = rex_preferences.answer_preference_query("do you like hydrospanners?")

        self.assertEqual(one, two)

    def test_sensitive_group_topic_uses_boundary_reply(self):
        from intelligence import rex_preferences

        reply = rex_preferences.answer_preference_query("do you like religions?")

        self.assertEqual(reply.stance, "boundary")
        self.assertEqual(reply.body_beat, "thinking_tilt")
        self.assertIn("do not rate whole categories of people", reply.text)

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
