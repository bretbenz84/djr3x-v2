"""The decision ledger (intelligence/decision_ledger.py): the ring, the "why?"
detector, the directive (record present / record empty / reply frames capped),
purpose humanizing, and the cheap instrumentation hooks that feed it."""

import time
import unittest
from unittest import mock

import config
from intelligence import decision_ledger as L


class RingTest(unittest.TestCase):
    def setUp(self):
        L.clear()
        self.addCleanup(L.clear)

    def test_record_and_recent_newest_first(self):
        L.record("a", "first thing")
        L.record("b", "second thing", said="hi there")
        items = L.recent()
        self.assertEqual([e["why"] for e in items], ["second thing", "first thing"])
        self.assertEqual(items[0]["said"], "hi there")

    def test_empty_why_ignored(self):
        L.record("a", "   ")
        self.assertEqual(L.recent(), [])

    def test_kinds_and_age_filter(self):
        L.record("a", "old")
        with L._lock:
            L._ring[-1]["mono"] -= 1000
        L.record("b", "new")
        self.assertEqual([e["why"] for e in L.recent(max_age_secs=60)], ["new"])
        self.assertEqual([e["why"] for e in L.recent(max_age_secs=5000, kinds={"a"})], ["old"])


class WhyQuestionTest(unittest.TestCase):
    def test_positive(self):
        for t in ("why did you do that?", "Why'd you turn around", "why are you looking over there",
                  "how come you said that", "what made you do the Carter thing", "what was that about",
                  "who were you looking for", "explain yourself", "why the impression?"):
            self.assertTrue(L.looks_like_why_question(t), t)

    def test_negative(self):
        for t in ("what's the weather", "do you like jazz", "why is the sky blue",
                  "tell me a joke", "why did the chicken cross the road"):
            self.assertFalse(L.looks_like_why_question(t), t)


class DirectiveTest(unittest.TestCase):
    def setUp(self):
        L.clear()
        self.addCleanup(L.clear)

    def test_not_a_why_question(self):
        L.record("head_wander", "I looked around")
        self.assertIsNone(L.why_directive("what time is it"))

    def test_empty_record_says_not_sure(self):
        d = L.why_directive("why did you turn?")
        self.assertIn("NO record", d)
        self.assertIn("not sure", d)

    def test_record_is_handed_over_with_ages(self):
        L.record("gaze_search", "I heard someone talking but couldn't see a face, so I turned my head")
        L.record("proactive_line", "I spotted an animal (animal arrival: dog)", said="Is that Max?")
        d = L.why_directive("why did you turn your head?")
        self.assertIn("ACTUAL record", d)
        self.assertIn("turned my head", d)
        self.assertIn('I said: "Is that Max?"', d)
        self.assertIn("just now", d)
        self.assertIn("not sure", d)   # the past-the-edge instruction is always there

    def test_reply_frames_capped_at_two(self):
        for i in range(5):
            L.record("reply_frame", f"frame {i}")
        L.record("head_wander", "I looked around")
        d = L.why_directive("why did you look away?")
        self.assertIn("frame 4", d)
        self.assertIn("frame 3", d)
        self.assertNotIn("frame 2", d)
        self.assertIn("looked around", d)

    def test_disabled(self):
        with mock.patch.object(config, "DECISION_LEDGER_ENABLED", False, create=True):
            self.assertIsNone(L.why_directive("why did you do that"))

    def test_age_phrase(self):
        self.assertEqual(L._age_phrase(3), "just now")
        self.assertEqual(L._age_phrase(33), "about 35 seconds ago")
        self.assertEqual(L._age_phrase(130), "about 2 minutes ago")


class PurposeTest(unittest.TestCase):
    def test_known_purpose_with_label(self):
        self.assertEqual(L.why_for_purpose("world.animal_arrival", "animal arrival: dog"),
                         "I spotted an animal (animal arrival: dog)")

    def test_unknown_purpose_humanized(self):
        self.assertEqual(L.why_for_purpose("some.new_thing"), "some new thing")

    def test_label_not_repeated(self):
        self.assertEqual(L.why_for_purpose("small_talk", "small talk"),
                         "the room was quiet and I made small talk")


class HooksTest(unittest.TestCase):
    def setUp(self):
        L.clear()
        self.addCleanup(L.clear)

    def test_speak_proactive_records_with_why(self):
        from intelligence import interaction as I
        with mock.patch.object(I, "_speak_blocking", return_value=True), \
             mock.patch.object(I, "_text_only_mode", True):
            self.assertTrue(I._speak_proactive("Quiet in here.", label="idle_banter"))
            self.assertTrue(I._speak_proactive("How was work?", label="lean_impulse",
                                               why="the room had been quiet about 40 seconds, and I checked in"))
        whys = [e["why"] for e in L.recent()]
        self.assertEqual(whys[0], "the room had been quiet about 40 seconds, and I checked in")
        self.assertEqual(whys[1], "the room was quiet and I made idle banter")

    def test_speak_proactive_dropped_line_not_recorded(self):
        from intelligence import interaction as I
        with mock.patch.object(I, "_speak_blocking", return_value=False), \
             mock.patch.object(I, "_text_only_mode", True):
            I._speak_proactive("Quiet in here.", label="idle_banter")
        self.assertEqual(L.recent(), [])

    def test_lean_impulse_why(self):
        from intelligence import interaction as I
        self.assertIn("event they'd mentioned", I._lean_impulse_why("event_followup", 42.0, False))
        self.assertIn("long silence", I._lean_impulse_why(None, 120.0, True))
        self.assertIn("whatever came to mind", I._lean_impulse_why(None, 20.0, False))

    def test_reply_frame_why(self):
        from intelligence import interaction as I
        frame = mock.Mock(purpose="answer_ack", allow_roast="sharp")
        mode = mock.Mock(key="deadpan", label="deadpan")
        claim = mock.Mock(premise="the paddleboard incident")
        why = I._reply_frame_why(frame, mode, claim)
        self.assertIn("'answer ack'", why)
        self.assertIn("deadpan", why)
        self.assertIn("sharp roast", why)
        self.assertIn("paddleboard", why)
        straight = mock.Mock(key="straight", label="straight")
        self.assertIn("played straight", I._reply_frame_why(mock.Mock(purpose="care", allow_roast="none"), straight, None))

    def test_pet_guess_records(self):
        from intelligence import consciousness as C
        with mock.patch.object(C, "_pet_owner_candidates", return_value=[(1, "Bret Benziger")]), \
             mock.patch("memory.facts.get_pets", return_value=[{"name": "Max", "species": "dog", "confidence": 0.6}]), \
             mock.patch.object(config, "ANIMAL_PET_NAME_GUESS_ENABLED", True, create=True):
            C._pet_name_guess_line("dog")
        C._animal_guessed_pet.clear()
        items = L.recent(kinds={"pet_guess"})
        self.assertEqual(len(items), 1)
        self.assertIn("Max", items[0]["why"])
        self.assertIn("Bret", items[0]["why"])

    def test_gaze_search_start_records_once(self):
        from intelligence import consciousness as C
        from world_state import world_state
        saved = world_state.get("self_state")
        try:
            C._record_face_tracking_state(locked=False, visible=False, searching=True,
                                          search_reason="speech", search_pose="left")
            C._record_face_tracking_state(locked=False, visible=False, searching=True,
                                          search_reason="speech", search_pose="right")
        finally:
            world_state.update("self_state", saved)
        items = L.recent(kinds={"gaze_search"})
        self.assertEqual(len(items), 1)
        self.assertIn("turned my head", items[0]["why"])


if __name__ == "__main__":
    unittest.main()
