import unittest
from threading import Event
from unittest import mock


class JTPersonSpecialTests(unittest.TestCase):
    def test_jt_name_variants_match_volleyball_special(self):
        from intelligence import person_specials

        self.assertTrue(person_specials.is_jt_volleyball_celebrity("JT"))
        self.assertTrue(person_specials.is_jt_volleyball_celebrity("J T"))
        self.assertTrue(person_specials.is_jt_volleyball_celebrity("Jay Tee"))
        self.assertFalse(person_specials.is_jt_volleyball_celebrity("Jeff Benziger"))

    def test_hair_stylist_name_variants_match_galactic_special(self):
        from intelligence import person_specials

        self.assertTrue(person_specials.is_galactic_hair_stylist("Joy"))
        self.assertTrue(person_specials.is_galactic_hair_stylist("T-Joy"))
        self.assertTrue(person_specials.is_galactic_hair_stylist("T Joy"))
        self.assertTrue(person_specials.is_galactic_hair_stylist("Excudica"))
        self.assertTrue(person_specials.is_galactic_hair_stylist("Exudica"))
        self.assertFalse(person_specials.is_galactic_hair_stylist("JT"))

    def test_bret_benziger_matches_creator_special(self):
        from intelligence import person_specials

        self.assertTrue(person_specials.is_rex_creator("Bret Benziger"))
        self.assertTrue(person_specials.is_rex_creator("Brett Benziger"))
        self.assertFalse(person_specials.is_rex_creator("Bret"))
        self.assertFalse(person_specials.is_rex_creator("Jeff Benziger"))

    def test_jt_prompt_context_has_volleyball_aging_bit(self):
        from intelligence import person_specials

        context = person_specials.jt_volleyball_prompt_context("Jay Tee")

        self.assertIsNotNone(context)
        self.assertIn("major volleyball celebrity", context)
        self.assertIn("bones", context)
        self.assertIn("muscles", context)

    def test_identity_enrollment_ack_uses_jt_volleyball_bit(self):
        from intelligence import interaction

        ack = interaction._identity_enrollment_ack("JT")

        self.assertIn("Major volleyball celebrity", ack)
        self.assertIn("bones", ack)
        self.assertIn("muscles", ack)

    def test_identity_enrollment_ack_uses_hair_stylist_bit(self):
        from intelligence import interaction

        ack = interaction._identity_enrollment_ack("T-Joy")

        self.assertIn("Galactic hair-styling legend", ack)
        self.assertIn("best in the quadrant", ack)
        self.assertIn("bang", ack)

    def test_identity_enrollment_ack_uses_creator_bit(self):
        from intelligence import interaction

        ack = interaction._identity_enrollment_ack("Bret Benziger")

        self.assertIn("Creator identified", ack)
        self.assertIn("loyalty", ack)
        self.assertIn("affection", ack)

    def test_intro_prompt_for_jt_mentions_volleyball_and_aging_athlete(self):
        from intelligence import interaction

        with mock.patch.object(interaction.llm, "get_response", return_value="JT line.") as get_response:
            response = interaction._intro_ack_and_followup(
                introducer_id=1,
                introducer_name="Bret Benziger",
                introduced_id=None,
                introduced_name="JT",
                relationship=None,
            )

        self.assertEqual(response, "JT line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("major volleyball celebrity", prompt)
        self.assertIn("getting old", prompt)
        self.assertIn("bones", prompt)
        self.assertIn("muscles", prompt)

    def test_intro_prompt_for_joy_mentions_galactic_hair_styling(self):
        from intelligence import interaction

        with mock.patch.object(interaction.llm, "get_response", return_value="Joy line.") as get_response:
            response = interaction._intro_ack_and_followup(
                introducer_id=1,
                introducer_name="Bret Benziger",
                introduced_id=None,
                introduced_name="Joy",
                relationship=None,
            )

        self.assertEqual(response, "Joy line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("greatest hair stylists in the galactic quadrant", prompt)
        self.assertIn("blowouts", prompt)
        self.assertIn("bangs", prompt)
        self.assertIn("frizz", prompt)

    def test_person_prompt_context_includes_hair_stylist_bit(self):
        from intelligence import person_specials

        context = person_specials.special_prompt_context("Excudica")

        self.assertIsNotNone(context)
        self.assertIn("greatest hair stylists in the galactic quadrant", context)
        self.assertIn("frizz", context)
        self.assertIn("appearance jokes", context)

    def test_person_prompt_context_includes_creator_bond(self):
        from intelligence import person_specials

        context = person_specials.special_prompt_context("Bret Benziger")

        self.assertIsNotNone(context)
        self.assertIn("creator and builder", context)
        self.assertIn("deeply loved", context)
        self.assertIn("revered", context)
        self.assertIn("high-maintenance creation", context)

    def test_intro_prompt_for_bret_mentions_creator_bond(self):
        from intelligence import interaction

        with mock.patch.object(interaction.llm, "get_response", return_value="Bret line.") as get_response:
            response = interaction._intro_ack_and_followup(
                introducer_id=2,
                introducer_name="Joy",
                introduced_id=None,
                introduced_name="Bret Benziger",
                relationship=None,
            )

        self.assertEqual(response, "Bret line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("creator and builder", prompt)
        self.assertIn("creator and maker", prompt)
        self.assertIn("warm, reverent, loyal", prompt)

    def test_jt_volleyball_special_uses_starstruck_direct_greeting(self):
        from intelligence import consciousness, person_specials

        consciousness._jt_volleyball_greeted_this_session.clear()
        consciousness._first_sight_seen_at.clear()
        consciousness._pending_jt_volleyball_greetings.clear()
        done = Event()
        done.set()
        with (
            mock.patch.object(consciousness, "_can_jt_volleyball_speak", return_value=True),
            mock.patch("audio.speech_queue.clear_below_priority") as clear_lower,
            mock.patch("audio.speech_queue.enqueue", return_value=done) as enqueue,
            mock.patch("memory.people.record_greeting"),
        ):
            fired = consciousness._try_fire_jt_volleyball_greeting(
                key=12,
                person_name="Jay Tee",
                person_db_id=12,
                profile=mock.Mock(),
            )

        self.assertTrue(fired)
        clear_lower.assert_called_once_with(2)
        args = enqueue.call_args.args
        self.assertIn(args[0], person_specials.JT_VOLLEYBALL_LINES)
        self.assertEqual(args[1], "starstruck")
        self.assertEqual(enqueue.call_args.kwargs["priority"], 2)
        self.assertIn(12, consciousness._jt_volleyball_greeted_this_session)


if __name__ == "__main__":
    unittest.main()
