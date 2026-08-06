"""
Unit tests for intelligence.place_questions — the learn-by-being-told room loop.

The perception.place_service is faked (recognizer present, controllable belief/state,
enroll records the name), so the ASK-eligibility and NAME-capture logic is exercised
with no model, camera, or DB.
"""

import unittest

import config
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
        self.no_drive = {}      # name -> (on, reason)

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

    def set_no_drive(self, name, on, reason=None):
        if name not in self.known_names:
            return False
        self.no_drive[name] = (bool(on), reason)
        return True

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

    # ── field 2026-08-06: "Tell me more." became a room ──
    def test_latched_request_to_rex_is_not_a_room_name(self):
        """A reply aimed at Rex is never what a room is called. He asked which room
        this was, moved on to a news offer, and filed the reply to THAT as the room."""
        for utterance in ("Tell me more.", "tell me more about that", "go on",
                          "keep going", "say that again", "shut down", "play music",
                          "stop it", "come here", "turn left", "read it to me",
                          "anything else", "one more time", "explain that"):
            with self.subTest(utterance):
                pq.reset()
                pq.note_asked()
                self.assertIsNone(pq.maybe_capture_answer(utterance))
        self.assertEqual(self.svc.enrolled, [])

    def test_room_word_still_wins_over_the_request_veto(self):
        """The veto is first-word-only, and a known room word or a room head noun
        bypasses it — so a compound name starting with a vetoed token still lands."""
        for utterance, want in [("show room", "show room"),
                                ("play room", "play room"),
                                ("the back nook", "back nook"),
                                ("the shop", "shop"),
                                ("study", "study"),
                                ("playroom", "playroom")]:
            with self.subTest(utterance):
                pq.reset()
                pq.note_asked()
                cap = pq.maybe_capture_answer(utterance)
                self.assertIsNotNone(cap, utterance)
                self.assertEqual(cap["name"], want)

    def test_rex_changing_the_subject_disarms_the_latch(self):
        pq.note_asked("Which room is this, Bret?")
        pq.note_rex_line("Hey, did you hear about the AWS outage?", source="lean_impulse")
        self.assertIsNone(pq.maybe_capture_answer("the workshop"))
        self.assertEqual(self.svc.enrolled, [])

    def test_the_ask_does_not_disarm_its_own_latch(self):
        """note_asked() is called after the line is registered, but the exemption is
        by TEXT so the order can never matter."""
        line = "Which room is this, Bret?"
        pq.note_asked(line)
        pq.note_rex_line(line, source="lean_impulse")
        self.assertEqual(pq.maybe_capture_answer("the workshop")["name"], "workshop")

    def test_place_flow_lines_keep_the_latch(self):
        pq.note_asked("Which room is this?")
        pq.note_rex_line("Sorry — which room?", source="place_question")
        self.assertEqual(pq.maybe_capture_answer("the lab")["name"], "lab")

    def test_unlatched_declaration_is_unaffected_by_the_disarm(self):
        """Volunteered "this is the kitchen" never needed a latch and must still work."""
        pq.note_asked("Which room is this?")
        pq.note_rex_line("Something else entirely.", source="lean_impulse")
        self.assertEqual(pq.maybe_capture_answer("this is the kitchen")["name"], "kitchen")

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


class DenialTest(unittest.TestCase):
    """"This is not the workshop" must DROP the belief. Field 2026-07-24: it drew
    "Yep, the workshop. I recognize it." — a human standing in the room outranks a
    cosine score, and doubling down is the worst possible reply."""

    def setUp(self):
        pq.reset()
        self.svc = _FakeService()
        self.svc.known_names = ["workshop", "kitchen", "garage"]
        self.svc.belief = {"name": "workshop", "place_id": 1}
        self.svc.rejected = []
        self.svc.reject_belief = self._reject
        self._orig = pq._service
        pq._service = lambda: self.svc
        self.addCleanup(lambda: setattr(pq, "_service", self._orig))
        self.addCleanup(pq.reset)

    def _reject(self, name=None):
        self.svc.rejected.append(name)
        self.svc.belief = None
        return True

    def test_direct_denial_drops_the_belief(self):
        for text in ("This is not the workshop.",
                     "this isn't the workshop",
                     "that is not the workshop",
                     "you're not in the workshop"):
            self.setUp()
            out = pq.maybe_capture_denial(text)
            self.assertEqual(out, {"was": "workshop"}, text)
            self.assertEqual(self.svc.rejected, ["workshop"], text)

    def test_denial_of_a_different_room_is_ignored(self):
        self.assertIsNone(pq.maybe_capture_denial("this is not the garage"))
        self.assertEqual(self.svc.rejected, [])

    def test_opinions_and_unrelated_negations_are_not_denials(self):
        for text in ("I do not like the workshop",
                     "we are not done in the workshop",
                     "the workshop is great",
                     "this is not what I expected",
                     "no",
                     "What room are you in?"):
            self.assertIsNone(pq.maybe_capture_denial(text), text)
        self.assertEqual(self.svc.rejected, [])

    def test_no_belief_means_nothing_to_deny(self):
        self.svc.belief = None
        self.assertIsNone(pq.maybe_capture_denial("this is not the workshop"))
        self.assertEqual(self.svc.rejected, [])

    def test_denial_ack_invites_the_real_name(self):
        # Check EVERY template, not one random draw: the ack must always end in a
        # question so the correction leads straight into learning the real name.
        for template in config.PLACE_DENIAL_ACK_TEMPLATES:
            line = template.format(was="workshop")
            self.assertTrue(line.strip().endswith("?"), template)
            self.assertNotIn("{", line, template)     # no unfilled placeholders
        # The rendered line never argues the point.
        for _ in range(20):
            line = pq.denial_ack_line({"was": "workshop"})
            self.assertNotIn("recognize", line.lower())

    def test_a_normal_room_statement_still_enrolls(self):
        # The denial check must not swallow the ordinary teach path.
        self.assertIsNone(pq.maybe_capture_denial("this is the dining room"))
        cap = pq.maybe_capture_answer("this is the dining room")
        self.assertIsNotNone(cap)
        self.assertEqual(cap["name"], "dining room")


class ReplyGroundingTest(unittest.TestCase):
    """The room belief must reach the REPLY prompt, not just proactive lines.

    Field 2026-07-24: "What room are you in?" got "I'm in whatever room you're in,
    unfortunately" while the recognizer was scoring the enrolled workshop at
    0.83-0.87 all session. Cause: belief_clause() was wired into
    llm._summarize_world_state (the CLASSIC prompt, bypassed under ONE VOICE) and
    lean_brain._situation_block (the PROACTIVE path only) — never into the lean
    system prompt that actually answers questions.
    """

    def setUp(self):
        pq.reset()
        self.svc = _FakeService()
        self.svc.known_names = ["workshop"]
        self.svc.belief = {"name": "workshop", "place_id": 1}
        self._orig = pq._service
        pq._service = lambda: self.svc
        self.addCleanup(lambda: setattr(pq, "_service", self._orig))
        self.addCleanup(pq.reset)

    def test_lean_system_prompt_carries_the_room_belief(self):
        from intelligence import lean_brain
        prompt = lean_brain._system_prompt(None, {"time_of_day": "evening"})
        self.assertIn("workshop", prompt)

    def test_reply_messages_carry_the_room_belief(self):
        # The exact path a spoken question takes: _messages -> _system_prompt.
        from intelligence import lean_brain
        msgs = lean_brain._messages(
            "What room are you in?", None, [], {"time_of_day": "evening"},
        )
        self.assertIn("workshop", msgs[0]["content"])

    def test_no_room_line_when_service_is_not_running(self):
        # Feature off / service down must contribute NOTHING — silence must not
        # read as "he claims ignorance" (belief_clause returns "" in that case).
        pq._service = lambda: None
        from intelligence import lean_brain
        prompt = lean_brain._system_prompt(None, {"time_of_day": "evening"})
        self.assertNotIn("Room:", prompt)

    def test_unknown_room_is_stated_honestly_not_dodged(self):
        # Recognizer running but nothing recognized: he should be TOLD he doesn't
        # recognize the room, so he can say so instead of inventing a dodge.
        self.svc.belief = None
        self.svc.known_names = ["workshop"]
        from intelligence import lean_brain
        prompt = lean_brain._system_prompt(None, {"time_of_day": "evening"})
        self.assertIn("Room:", prompt)
        self.assertIn("don't recognize", prompt)


if __name__ == "__main__":
    unittest.main()


class DriveRuleTest(unittest.TestCase):
    """"This room has carpet" / "don't move in the workshop" — a standing rule about
    the FLOOR, filed against the room so it re-arms whenever he walks back in."""

    def setUp(self):
        pq.reset()
        self.svc = _FakeService()
        self.svc.known_names = ["workshop", "dining room"]
        self.svc.belief = {"name": "workshop", "place_id": 1}
        self._orig = pq._service
        pq._service = lambda: self.svc
        self.addCleanup(lambda: setattr(pq, "_service", self._orig))
        self.addCleanup(pq.reset)

    def test_the_owners_own_phrasings_all_land(self):
        for text in ("This room has carpet.", "This room is carpeted.",
                     "There's carpet in here.", "Don't try to move in this room.",
                     "Don't move in this room.", "Don't drive in here.",
                     "Don't move in the workshop.", "No driving in the workshop."):
            with self.subTest(text=text):
                self.svc.no_drive.clear()
                rule = pq.maybe_capture_drive_rule(text)
                self.assertIsNotNone(rule, "phrase did not parse")
                self.assertTrue(rule["no_drive"])
                self.assertTrue(rule["applied"])
                self.assertEqual(self.svc.no_drive["workshop"][0], True)

    def test_a_named_room_files_against_that_room_not_the_current_one(self):
        rule = pq.maybe_capture_drive_rule("Don't move in the dining room.")
        self.assertEqual(rule["name"], "dining room")
        self.assertNotIn("workshop", self.svc.no_drive)
        self.assertEqual(self.svc.no_drive["dining room"][0], True)
        # He's in the workshop, so this rule must not stop his wheels here and now.
        self.assertFalse(rule["current"])

    def test_a_rule_about_the_room_he_is_in_is_marked_current(self):
        self.assertTrue(pq.maybe_capture_drive_rule("This room has carpet.")["current"])
        self.assertTrue(
            pq.maybe_capture_drive_rule("Don't move in the workshop.")["current"])

    def test_carpet_is_recorded_as_the_reason(self):
        pq.maybe_capture_drive_rule("This room has carpet.")
        self.assertEqual(self.svc.no_drive["workshop"], (True, "carpet"))

    def test_the_rule_can_be_lifted_by_voice(self):
        pq.maybe_capture_drive_rule("This room has carpet.")
        rule = pq.maybe_capture_drive_rule("You can drive in here.")
        self.assertFalse(rule["no_drive"])
        self.assertEqual(self.svc.no_drive["workshop"][0], False)

    def test_hard_floors_lift_it_too(self):
        pq.maybe_capture_drive_rule("This room has carpet.")
        for text in ("This room has hardwood floors.", "No carpet in here.",
                     "It's fine to drive in the workshop."):
            with self.subTest(text=text):
                self.svc.no_drive["workshop"] = (True, "carpet")
                rule = pq.maybe_capture_drive_rule(text)
                self.assertIsNotNone(rule)
                self.assertFalse(rule["no_drive"])
                self.assertEqual(self.svc.no_drive["workshop"][0], False)

    def test_unrecognized_room_reports_but_cannot_file(self):
        # He has to stop NOW and say plainly that he can't remember it yet.
        self.svc.belief = None
        rule = pq.maybe_capture_drive_rule("This room has carpet.")
        self.assertTrue(rule["no_drive"])
        self.assertFalse(rule["applied"])
        self.assertTrue(rule["here"])
        self.assertIn("don't know which room", pq.drive_rule_ack_line(rule).lower())

    def test_room_he_has_never_enrolled_reports_but_cannot_file(self):
        rule = pq.maybe_capture_drive_rule("Don't move in the nursery.")
        self.assertEqual(rule["name"], "nursery")
        self.assertFalse(rule["applied"])
        self.assertEqual(self.svc.no_drive, {})

    def test_direction_words_are_not_rooms(self):
        # "don't move forward" must reach the motion router, not be filed as a rule
        # about a room called "forward".
        for text in ("don't move forward", "don't go closer", "don't move to the left",
                     "don't drive into the wall"):
            with self.subTest(text=text):
                self.assertIsNone(pq.maybe_capture_drive_rule(text))

    def test_unrelated_turns_are_left_alone(self):
        for text in ("Don't move.", "Come here.", "This is the workshop.",
                     "What room are you in?", "I love the carpet at my mom's house",
                     "Back when the den had carpet"):
            with self.subTest(text=text):
                self.assertIsNone(pq.maybe_capture_drive_rule(text))

    def test_a_carpet_statement_is_not_mined_for_a_room_name(self):
        # maybe_capture_answer would happily read "this room has carpet" as a
        # declaration; the drive rule must win, and interaction.py runs it first.
        self.assertIsNotNone(pq.maybe_capture_drive_rule("This room has carpet."))

    def test_belief_clause_warns_the_reply_model(self):
        self.svc.belief = {"name": "workshop", "place_id": 1,
                           "no_drive": True, "no_drive_reason": "carpet"}
        clause = pq.belief_clause()
        self.assertIn("workshop", clause)
        self.assertIn("not to drive", clause)
        self.assertIn("carpet", clause)

    def test_belief_clause_is_unchanged_without_a_rule(self):
        clause = pq.belief_clause()
        self.assertIn("workshop", clause)
        self.assertNotIn("not to drive", clause)
