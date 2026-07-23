"""
Unit tests for intelligence.place_questions — the learn-by-being-told room loop.

The perception.place_service is faked (recognizer present, controllable belief/state,
enroll records the name), so the ASK-eligibility and NAME-capture logic is exercised
with no model, camera, or DB.
"""

import unittest

from intelligence import place_questions as pq


class _FakeService:
    def __init__(self):
        self.enrolled = []
        self.belief = None      # current_place() return
        self.st = "idle"
        self.has_recognizer = True
        self.known_names = []   # place_names() return
        self.active_enroll = None  # enrolling_name() return
        self._next_id = 0

    def get_recognizer(self):
        return object() if self.has_recognizer else None

    def current_place(self):
        return self.belief

    def state(self):
        return self.st

    def enrolling_name(self):
        return self.active_enroll

    def place_names(self):
        return list(self.known_names)

    def enroll(self, name):
        self._next_id += 1
        self.enrolled.append(name)
        return self._next_id

    def belief_context(self):
        return {
            "belief": self.belief,
            "top": getattr(self, "top", []),
            "classification": None,
            "ambiguous": getattr(self, "ambiguous", False),
            "known_rooms": len(self.known_names),
            "enrolling": self.active_enroll,
            "age_s": 1.0,
        }


class PlaceQuestionsTest(unittest.TestCase):
    def setUp(self):
        pq.reset()
        self.svc = _FakeService()
        self._orig = pq._service
        pq._service = lambda: self.svc  # inject the fake
        self.addCleanup(lambda: setattr(pq, "_service", self._orig))
        self.addCleanup(pq.reset)

    # ── ASK eligibility ──
    def test_asks_when_unknown_and_available(self):
        q = pq.next_place_question()
        self.assertIsNotNone(q)
        self.assertIn("text", q)

    def test_no_ask_when_belief_known(self):
        self.svc.belief = {"name": "office", "place_id": 1}
        self.assertIsNone(pq.next_place_question())

    def test_no_ask_when_enrolling(self):
        self.svc.st = "collecting"
        self.assertIsNone(pq.next_place_question())

    def test_no_ask_when_service_down(self):
        self.svc.has_recognizer = False
        self.assertIsNone(pq.next_place_question())

    def test_cooldown_blocks_reask(self):
        pq.note_asked()
        self.assertIsNone(pq.next_place_question())  # just asked -> cooldown

    # ── NAME capture: volunteered (unlatched) ──
    def test_volunteered_declaration_with_room_word(self):
        cap = pq.maybe_capture_answer("this is the living room")
        self.assertEqual(cap["name"], "living room")
        self.assertEqual(self.svc.enrolled, ["living room"])

    def test_volunteered_were_in_the_kitchen(self):
        self.assertEqual(pq.maybe_capture_answer("we're in the kitchen")["name"], "kitchen")

    def test_volunteered_youre_in_the_office(self):
        self.assertEqual(pq.maybe_capture_answer("you're in the office now")["name"], "office")

    def test_longest_room_word_wins(self):
        self.assertEqual(
            pq.maybe_capture_answer("this is the master bedroom")["name"], "master bedroom")

    def test_person_intro_does_not_mint_a_room(self):
        self.assertIsNone(pq.maybe_capture_answer("this is Sarah"))
        self.assertEqual(self.svc.enrolled, [])

    def test_passing_mention_without_declaration_ignored(self):
        self.assertIsNone(pq.maybe_capture_answer("I really love the kitchen at my mom's place"))
        self.assertEqual(self.svc.enrolled, [])

    def test_custom_name_unlatched_is_rejected(self):
        # A custom name with no known room word and no ask pending -> too risky, ignore.
        self.assertIsNone(pq.maybe_capture_answer("this is the nook"))
        self.assertEqual(self.svc.enrolled, [])

    # ── NAME capture: answering Rex's question (latched) ──
    def test_latched_bare_answer(self):
        pq.note_asked()
        self.assertEqual(pq.maybe_capture_answer("the living room")["name"], "living room")

    def test_latched_single_word(self):
        pq.note_asked()
        self.assertEqual(pq.maybe_capture_answer("kitchen")["name"], "kitchen")

    def test_latched_custom_name_allowed(self):
        pq.note_asked()
        self.assertEqual(pq.maybe_capture_answer("the lab")["name"], "lab")

    def test_latched_non_answer_decrements_and_expires(self):
        pq.note_asked()
        self.assertIsNone(pq.maybe_capture_answer("I don't know"))
        self.assertIsNone(pq.maybe_capture_answer("nope"))
        self.assertIsNone(pq.maybe_capture_answer("dunno"))
        # latch spent (3 turns) -> a later bare noun is no longer taken as the answer
        self.assertIsNone(pq.maybe_capture_answer("basement"))
        self.assertEqual(self.svc.enrolled, [])

    def test_latched_question_back_is_not_an_answer(self):
        pq.note_asked()
        self.assertIsNone(pq.maybe_capture_answer("why do you want to know?"))

    # ── robot-lens review regressions ──
    def test_past_tense_never_enrolls(self):
        # "we'?re" used to match past-tense "were" — reminiscing enrolled the current view.
        self.assertIsNone(pq.maybe_capture_answer("when we were in the kitchen it smelled great"))
        self.assertIsNone(pq.maybe_capture_answer("remember when we were in the living room"))
        pq.note_asked()
        self.assertIsNone(pq.maybe_capture_answer("the den used to be a garage"))
        self.assertEqual(self.svc.enrolled, [])

    def test_latched_incidental_room_word_in_long_sentence_rejected(self):
        pq.note_asked()
        self.assertIsNone(pq.maybe_capture_answer("I told you about the kitchen twice already"))
        self.assertEqual(self.svc.enrolled, [])

    def test_latched_answer_prefixes_are_stripped(self):
        # "it's the nook" used to mint a room literally named "it's the nook".
        for utterance, want in [
            ("it's the nook", "nook"),
            ("its the nook", "nook"),            # Whisper drops apostrophes
            ("thats the snug", "snug"),
            ("well, it's the nook", "nook"),
        ]:
            pq.reset()
            pq.note_asked()
            cap = pq.maybe_capture_answer(utterance)
            self.assertIsNotNone(cap, utterance)
            self.assertEqual(cap["name"], want, utterance)

    def test_retelling_same_room_mid_capture_is_suppressed(self):
        self.svc.st = "collecting"
        self.svc.active_enroll = "living room"
        self.assertIsNone(pq.maybe_capture_answer("this is the living room"))
        self.assertEqual(self.svc.enrolled, [])   # no session restart, no double ack

    def test_correction_to_different_room_restarts(self):
        self.svc.st = "collecting"
        self.svc.active_enroll = "living room"
        cap = pq.maybe_capture_answer("no wait, this is the den")
        self.assertEqual(cap["name"], "den")      # different name = intentional restart

    def test_known_room_gets_recognition_ack(self):
        self.svc.known_names = ["living room"]
        cap = pq.maybe_capture_answer("this is the living room")
        self.assertTrue(cap["known"])
        self.assertIn("living room", pq.ack_line(cap))
        cap2 = pq.maybe_capture_answer("this is the garage")
        self.assertFalse(cap2["known"])

    # ── grounding clause ──
    def test_belief_clause_confident(self):
        self.svc.belief = {"name": "living room", "place_id": 1}
        self.svc.known_names = ["living room"]
        clause = pq.belief_clause()
        self.assertIn("living room", clause)
        self.assertIn("recognize", clause)

    def test_belief_clause_hedges_when_ambiguous(self):
        self.svc.belief = {"name": "living room", "place_id": 1}
        self.svc.known_names = ["living room", "dining room"]
        self.svc.ambiguous = True
        self.svc.top = [("living room", 0.85), ("dining room", 0.84)]
        clause = pq.belief_clause()
        self.assertIn("living room", clause)
        self.assertIn("dining room", clause)
        self.assertIn("hedge", clause.lower())

    def test_belief_clause_admits_not_knowing(self):
        self.svc.known_names = ["living room"]
        self.assertIn("don't recognize", pq.belief_clause())

    def test_belief_clause_no_rooms_taught(self):
        clause = pq.belief_clause()
        self.assertIn("don't know any rooms", clause)

    def test_belief_clause_silent_when_service_off(self):
        self.svc.has_recognizer = False
        # service returns None context when recognizer is missing
        self.svc.belief_context = lambda: None
        self.assertEqual(pq.belief_clause(), "")

    def test_belief_clause_mentions_enrollment(self):
        self.svc.active_enroll = "den"
        self.assertIn("memorizing", pq.belief_clause())

    # ── availability + ack ──
    def test_no_capture_when_service_down(self):
        self.svc.has_recognizer = False
        self.assertIsNone(pq.maybe_capture_answer("this is the kitchen"))

    def test_recently_captured_and_ack(self):
        cap = pq.maybe_capture_answer("this is the den")
        self.assertTrue(pq.recently_captured())
        self.assertEqual(pq.last_capture()["name"], "den")
        self.assertIn("den", pq.ack_line(cap))


if __name__ == "__main__":
    unittest.main()
