"""memory/fact_quality.py — the content-quality gate for stored memories.

Each case is input -> REJECT(reason) / KEEP. Both the real garbage found in Bret's
record (workflow wf_0d93bffd) and the GOOD-facts false-positive guard are covered:
a fix that drops a good fact erases a real memory on a live robot, so the KEEP tests
are as important as the REJECT ones.
"""

import unittest

from memory import fact_quality as fq


class TestRejectFact(unittest.TestCase):
    # ── REJECT (garbage) ──
    def test_tautology_bare_dad(self):        # facts #13, #47, #51
        self.assertEqual(fq.is_tautology("family", "family_member", "dad"),
                         "tautology_relation_noun")

    def test_tautology_dog_eq_key(self):      # fact #26
        self.assertEqual(fq.is_tautology("pet", "dog", "dog"), "tautology_key")

    def test_keep_age_category_child(self):   # identity age_category='child' is REAL
        # 'child'/'kid' are relation nouns but a legit enumerated identity value —
        # the relation-noun tautology is scoped to family/pet only.
        self.assertIsNone(fq.reject_fact("identity", "age_category", "child"))
        self.assertIsNone(fq.reject_fact("identity", "age_category", "kid"))

    def test_fragment_i_might(self):          # fact #40
        self.assertTrue(fq.reject_fact("family", "dad",
                        "I might go see my dad for the 4th of July"))

    def test_fragment_its_historical(self):   # fact #23 — word-boundary bug fixed
        self.assertEqual(fq.is_fragment("interest_note", "x",
                        "Its usually historical on me"), "sentence_fragment")

    def test_fragment_how_found_rex(self):    # fact #10
        self.assertTrue(fq.reject_fact("identity", "how_found_rex",
                        "you can take me because I created you"))

    def test_fragment_with_leading_filler(self):   # fact #53 'oh I love to fix things'
        self.assertEqual(fq.is_fragment("interest_note", "x",
                        "oh I love to fix things"), "sentence_fragment")

    def test_fiction_scene_as_fact(self):     # fact #20 (category='interest')
        self.assertTrue(fq.reject_fact("interest", "interest_scene",
                        "scene where the kids figure out it's their dad because he's peeing upright"))

    def test_negation_coney_island_with_utterance(self):  # fact #11 — needs utterance
        self.assertEqual(
            fq.reject_fact("hometown", "hometown", "Coney Island",
                           utterance="that's a place I've never been to, Coney Island",
                           source="inferred"),
            "negation_source")

    def test_hypothetical_source(self):
        self.assertTrue(fq.reject_fact("hometown", "hometown", "Paris",
                        utterance="imagine if I lived in Paris someday"))

    def test_verbatim_question_value(self):
        self.assertTrue(fq.reject_fact("other", "note", "are you gonna judge me?"))

    # ── KEEP (GOOD-facts guard — a fix that drops these is WRONG) ──
    def test_keep_hometown_sacramento(self):
        self.assertIsNone(fq.reject_fact("hometown", "hometown", "Sacramento",
                          utterance="I'm from Sacramento", source="explicit"))

    def test_keep_pet_name_rex(self):
        self.assertIsNone(fq.reject_fact("pet", "pet_name", "Rex",
                          utterance="I have a dog named Rex", source="explicit"))

    def test_keep_favorite_movie(self):
        self.assertIsNone(fq.reject_fact("preference", "favorite_movie", "Mrs. Doubtfire"))

    def test_keep_favorite_music(self):
        self.assertIsNone(fq.reject_fact("preference", "favorite_music",
                          "classical music and soundtracks"))

    def test_keep_dad_visit_plan_third_person(self):   # GOOD twin of the #40 garbage
        self.assertIsNone(fq.reject_fact("relationship", "dad_visit_4th_of_july",
                          "Bret plans to visit his dad for the 4th of July"))

    def test_keep_likes_pizza(self):                   # cross_table_dedup FP guard
        self.assertIsNone(fq.reject_fact("food", "likes", "likes pizza"))

    def test_keep_negated_worldview(self):             # speaker_misattrib FP guard
        # 'I am not religious' -> worldview MUST survive (scope excludes worldview)
        self.assertIsNone(fq.reject_fact("worldview", "religion", "atheist",
                          utterance="I'm not religious", source="inferred"))

    def test_keep_explicit_dislike(self):
        self.assertIsNone(fq.reject_fact("preference", "dislikes_country",
                          "dislikes country music", utterance="I hate country music"))

    def test_keep_no_bake_cookies(self):               # 'no' not in negation set
        self.assertIsNone(fq.reject_fact("preference", "favorite_food", "no-bake cookies"))

    def test_keep_i_love_beatles_distilled(self):
        # A distilled value from a good extractor is 'The Beatles' — passes.
        self.assertIsNone(fq.reject_fact("preference", "favorite_band", "The Beatles"))

    def test_keep_title_with_never_short(self):        # negation-set FP guard
        # A movie title with 'never', stored as a preference (out of negation scope).
        self.assertIsNone(fq.reject_fact("preference", "favorite_movie", "Never Let Me Go"))

    def test_keep_garden_plot_interest_fact(self):     # fiction 'plot' FP guard
        self.assertIsNone(fq.reject_fact("interest", "interest_garden", "garden plot"))


class TestRejectInterest(unittest.TestCase):
    def test_reject_fiction_scene_interest(self):      # interest #15
        self.assertTrue(fq.reject_interest(
            "scene where the kids figure out it's their dad because he's peeing upright"))

    def test_reject_rex_misattribution(self):          # interest #23
        self.assertEqual(fq.reject_interest("music",
            "Rex mentioned being obsessed with music, indicating Bret's interest"),
            "rex_misattribution")

    def test_keep_music_interest_no_rex_note(self):    # FP guard: bare 'music' is fine
        self.assertIsNone(fq.reject_interest("music", ""))

    def test_keep_astrophotography(self):
        self.assertIsNone(fq.reject_interest("astrophotography"))

    def test_keep_beethoven_symphony(self):            # 5 words, <=8 cap
        self.assertIsNone(fq.reject_interest("Beethoven's 7th Symphony Movement No. 2"))

    def test_keep_garden_plot_interest(self):          # 'plot' FP guard
        self.assertIsNone(fq.reject_interest("garden plot"))

    def test_clean_question_note_keeps_interest(self):  # interest #22 note
        self.assertEqual(fq.clean_interest_note(
            "I like pizza, are you gonna judge me for my pizza?"), "")

    def test_clean_keeps_real_note(self):
        self.assertEqual(fq.clean_interest_note("builds telescopes on weekends"),
                         "builds telescopes on weekends")


if __name__ == "__main__":
    unittest.main()
