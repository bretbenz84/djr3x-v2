"""
Regressions from the 2026-08-29 11:15 run (logs/djr3x-2026-08-29-11-15-50.log).

Bret was alone in the room. He said "PJ is actually in the living room right
now" and then "Say hi to PJ", which opened an intro voice-capture window on PJ
(person 7). Three writes followed, none of them true:

  1. 11:21:46 — Bret's own next sentence ("I didn't leave. I just turned
     around.") became PJ's voice print. Speaker-ID had Bret top at 0.604, but
     the open intro window suppresses the introducer's visible face, so identity
     resolved to "off-camera unknown" (person_id=None) and the introducer guard
     — which only fires on a RESOLVED person_id — never ran. `accepted_unknown`
     waved it straight through to enrollment (biometrics row 56 on person 7).
  2. 11:22:01 — Bret's correction "PJ is not here. This is Bret." was filed as
     the Bret<->PJ connection story on BOTH people at confidence 0.90
     (person_facts 132/133). should_capture_followup() takes any three words.
  3. 11:23:23 — answering Rex's "who just said that?" with "Bret Benziger said
     that." drew "Bret, good to meet you; try not to break anything expensive."
     — a first-meeting greeting for a best_friend of 63 visits, who was also the
     person Rex was already talking to.

Fixes under test: introductions.denies_introduction(); the denial short-circuit
+ print retraction in the voice-capture and follow-up windows; the
self-attribution and already-met branches of the off-camera identify ack.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from intelligence import interaction as I
from intelligence import introductions


class DeniesIntroductionTest(unittest.TestCase):
    """The predicate itself: denials out, real connection stories in."""

    def test_field_sentence_is_a_denial(self):
        self.assertTrue(
            introductions.denies_introduction(
                "PJ is not here. This is Bret.", introduced_name="PJ Thomas"
            )
        )

    def test_naming_yourself_as_someone_else_is_a_denial(self):
        self.assertTrue(
            introductions.denies_introduction("This is Bret.", introduced_name="PJ")
        )
        self.assertTrue(
            introductions.denies_introduction("I'm Bret", introduced_name="PJ")
        )

    def test_newcomer_naming_themself_is_not_a_denial(self):
        self.assertFalse(
            introductions.denies_introduction("This is PJ.", introduced_name="PJ")
        )

    def test_presence_denials_need_no_name(self):
        for text in (
            "He is not here right now",
            "She isn't here",
            "Nobody else is around",
            "Wrong person, buddy",
            "You've got the wrong guy",
            "That was me",
            "It's just me in here",
        ):
            with self.subTest(text=text):
                self.assertTrue(introductions.denies_introduction(text))

    def test_named_absence_and_negation(self):
        self.assertTrue(
            introductions.denies_introduction("That's not PJ", introduced_name="PJ")
        )
        self.assertTrue(
            introductions.denies_introduction("PJ isn't talking", introduced_name="PJ")
        )

    def test_real_connection_stories_still_capture(self):
        for text in (
            "We met at work about ten years ago",
            "She is my sister from Chicago",
            "We went to college together",
            "He is here to fix the sink",
            "PJ is my brother in law",
            "It is a long story involving a boat",
            "I am his brother",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    introductions.denies_introduction(text, introduced_name="PJ")
                )
                self.assertTrue(
                    introductions.should_capture_followup(text, introduced_name="PJ")
                )

    def test_should_capture_followup_rejects_the_denial(self):
        self.assertFalse(
            introductions.should_capture_followup(
                "PJ is not here. This is Bret.", introduced_name="PJ Thomas"
            )
        )


class IntroVoiceCaptureDenialTest(unittest.TestCase):
    """11:21:46 — the correction must not become the newcomer's voice print."""

    def _ctx(self, **kw):
        ctx = {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 7,
            "introduced_name": "PJ Thomas",
            "relationship": None,
            "asked_at": time.monotonic(),
        }
        ctx.update(kw)
        return ctx

    def test_denial_never_enrolls_and_clears_the_window(self):
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_pending_intro_followup", None), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(I.people_memory, "delete_biometric") as delete_bio, \
             mock.patch.object(I.llm, "get_response", return_value=None):
            resp = I._handle_intro_voice_capture(
                "PJ is not here. This is Bret.",
                audio,
                person_id=1,
                raw_best_id=1,
                speaker_score=0.811,
            )
        self.assertTrue(resp)
        self.assertFalse(enroll.called)
        # Nothing was enrolled by THIS window, so there is no row to retract.
        self.assertFalse(delete_bio.called)
        self.assertIsNone(I._pending_intro_voice_capture)
        self.assertIsNone(I._pending_intro_followup)

    def test_off_camera_unknown_denial_is_still_a_denial(self):
        """The field shape: the open window suppressed the introducer's face, so
        identity handed the turn over as person_id=None. `accepted_unknown` used
        to make that an automatic enrollment."""
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_pending_intro_followup", None), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(I.people_memory, "delete_biometric"), \
             mock.patch.object(I.llm, "get_response", return_value=None):
            resp = I._handle_intro_voice_capture(
                "That's not PJ, wrong person.",
                audio,
                person_id=None,
                raw_best_id=1,
                speaker_score=0.604,
            )
        self.assertTrue(resp)
        self.assertFalse(enroll.called)

    def test_ordinary_hello_still_enrolls_the_newcomer(self):
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_pending_intro_followup", None), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(
                 I.people_memory, "latest_biometric_id", return_value=56
             ), \
             mock.patch.object(I, "_bind_intro_visible_face_if_present"), \
             mock.patch.object(I.llm, "get_response", return_value="PJ! Welcome."), \
             mock.patch.object(I.consciousness, "mark_engagement"), \
             mock.patch.object(I.consciousness, "note_person_spoke"), \
             mock.patch.object(I.consciousness, "note_person_greeted_this_session"), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_heard"), \
             mock.patch.object(I.topic_thread, "note_user_turn"), \
             mock.patch.object(I.user_energy, "note_user_turn"):
            resp = I._handle_intro_voice_capture(
                "Hi Rex, nice to meet you.",
                audio,
                person_id=None,
                raw_best_id=None,
                speaker_score=0.30,
            )
            # Read the armed follow-up INSIDE the patch, before it is restored.
            followup = I._pending_intro_followup
        self.assertTrue(resp)
        enroll.assert_called_once()
        self.assertEqual(enroll.call_args.args[0], 7)
        # The row id rides forward so a correction on the NEXT turn can undo it.
        self.assertEqual((followup or {}).get("enrolled_voice_biometric_id"), 56)


class IntroWindowFieldSequenceTest(unittest.TestCase):
    """The two field turns back to back: enroll, then get corrected.

    Turn 12 takes the sample on the window's expectation (that band is genuinely
    ambiguous — an un-enrolled newcomer cross-matches the introducer at the same
    0.6 scores the introducer himself lands on a short clip, and blocking it
    outright is what stranded PJ in the 2026-08-23 run). Turn 13 is the human
    saying it was wrong, and that has to reach the row.
    """

    def test_enroll_then_deny_retracts_the_print(self):
        audio = np.zeros(16000, dtype=np.float32)
        ctx = {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 7,
            "introduced_name": "PJ",
            "relationship": None,
            "asked_at": time.monotonic(),
        }
        with mock.patch.object(I, "_pending_intro_voice_capture", ctx), \
             mock.patch.object(I, "_pending_intro_followup", None), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True), \
             mock.patch.object(
                 I.people_memory, "latest_biometric_id", return_value=56
             ), \
             mock.patch.object(I.people_memory, "delete_biometric") as delete_bio, \
             mock.patch.object(I.facts_memory, "add_fact") as add_fact, \
             mock.patch.object(I, "_bind_intro_visible_face_if_present"), \
             mock.patch.object(I.llm, "get_response", return_value="Filed."), \
             mock.patch.object(I.consciousness, "mark_engagement"), \
             mock.patch.object(I.consciousness, "note_person_spoke"), \
             mock.patch.object(I.consciousness, "note_person_greeted_this_session"), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_heard"), \
             mock.patch.object(I.topic_thread, "note_user_turn"), \
             mock.patch.object(I.user_energy, "note_user_turn"):
            # turn 12 — the sample lands on PJ
            I._handle_intro_voice_capture(
                "I didn't leave. I just turned around.",
                audio, person_id=None, raw_best_id=1, speaker_score=0.604,
            )
            self.assertIsNotNone(I._pending_intro_followup)
            # turn 13 — Bret corrects him
            resp = I._handle_intro_followup_answer("PJ is not here. This is Bret.")

        self.assertTrue(resp)
        self.assertFalse(add_fact.called)
        delete_bio.assert_called_once_with(56)


class IntroFollowupDenialTest(unittest.TestCase):
    """11:22:01 — the correction must not become their connection story."""

    def _ctx(self, **kw):
        ctx = {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 7,
            "introduced_name": "PJ",
            "relationship": None,
            "followup_kind": "connection_story",
            "asked_at": time.monotonic(),
            "enrolled_voice_biometric_id": 56,
        }
        ctx.update(kw)
        return ctx

    def test_denial_saves_no_fact_and_retracts_the_print(self):
        with mock.patch.object(I, "_pending_intro_followup", self._ctx()), \
             mock.patch.object(I, "_pending_intro_voice_capture", None), \
             mock.patch.object(I.facts_memory, "add_fact") as add_fact, \
             mock.patch.object(I.people_memory, "delete_biometric") as delete_bio, \
             mock.patch.object(I.llm, "get_response", return_value=None):
            resp = I._handle_intro_followup_answer("PJ is not here. This is Bret.")
        self.assertTrue(resp)
        self.assertFalse(add_fact.called)
        delete_bio.assert_called_once_with(56)
        self.assertIsNone(I._pending_intro_followup)

    def test_real_connection_story_still_stores(self):
        with mock.patch.object(I, "_pending_intro_followup", self._ctx()), \
             mock.patch.object(I, "_pending_intro_voice_capture", None), \
             mock.patch.object(I.facts_memory, "add_fact") as add_fact, \
             mock.patch.object(I.people_memory, "delete_biometric") as delete_bio, \
             mock.patch.object(I.llm, "get_response", return_value="Noted."):
            resp = I._handle_intro_followup_answer("We met at work ten years ago.")
        self.assertTrue(resp)
        self.assertEqual(add_fact.call_count, 2)
        self.assertFalse(delete_bio.called)


class OffscreenSelfAttributionTest(unittest.TestCase):
    """11:23:23 — "Bret Benziger said that." is not an introduction."""

    def _pending(self):
        return {
            "audio": np.zeros(16000, dtype=np.float32),
            "asked_at": time.monotonic(),
            "prior_engaged_id": 1,
            "prior_engaged_name": "Bret Benziger",
            "overheard_text": "This week's Star Trek episode was pretty good.",
            "anonymous_speaker_label": "unknown_voice_1",
        }

    def _run(
        self, text, *, extracted, speaker_pid, resolved_pid, created, previously_met
    ):
        """Drive the handler and return (consumed, ack, llm_prompt)."""
        with mock.patch.object(I, "_pending_offscreen_identify", self._pending()), \
             mock.patch.object(
                 I.llm, "extract_relationship_introduction",
                 return_value={"name": extracted, "relationship": None},
             ), \
             mock.patch.object(
                 I.people_memory, "find_potential_person_match", return_value=None
             ), \
             mock.patch.object(
                 I.people_memory, "find_or_create_person",
                 return_value=(resolved_pid, created),
             ), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True), \
             mock.patch.object(I, "_person_previously_met", return_value=previously_met), \
             mock.patch.object(I.speaker_id, "rank_speakers", return_value=[]), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=False), \
             mock.patch.object(I, "_bind_world_state_identity"), \
             mock.patch.object(I, "_retire_anonymous_speaker_slot"), \
             mock.patch.object(I, "_episodic_person_enrolled"), \
             mock.patch.object(I.people_memory, "update_familiarity"), \
             mock.patch.object(I.llm, "get_response", return_value=None) as get_response, \
             mock.patch.object(I, "_speak_blocking", return_value=True), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch.object(I, "_register_rex_utterance"):
            consumed, ack = I._handle_pending_offscreen_identify_reply(
                text,
                person_id=speaker_pid,
                person_name="Bret Benziger",
                audio_array=np.zeros(16000, dtype=np.float32),
                anonymous_speaker_label=None,
            )
        prompt = get_response.call_args[0][0] if get_response.call_args else ""
        return consumed, ack, prompt

    def _no_newcomer_run(self, text, extracted):
        """Same drive, but assert nothing was minted or enrolled."""
        with mock.patch.object(
            I.people_memory, "find_or_create_person", return_value=(1, False)
        ) as mint, mock.patch.object(I, "_safe_enroll_voice") as enroll:
            consumed, ack, prompt = self._run(
                text, extracted=extracted, speaker_pid=1, resolved_pid=1,
                created=False, previously_met=True,
            )
        return consumed, ack, prompt, mint, enroll

    def test_speaker_naming_themself_is_not_greeted(self):
        # The field name: the extractor handed back the bare first name, which
        # slipped the whole-string and fuzzy speaker-match guards.
        consumed, ack, prompt, mint, enroll = self._no_newcomer_run(
            "Bret Benziger said that.", "Bret"
        )
        self.assertTrue(consumed)
        lowered = (ack or "").lower()
        for banned in ("good to meet you", "nice to meet", "welcome"):
            self.assertNotIn(banned, lowered)
        self.assertIn("do not greet them", prompt.lower())
        self.assertFalse(mint.called)
        self.assertFalse(enroll.called)

    def test_that_was_me_is_the_same_answer(self):
        consumed, ack, prompt, mint, _enroll = self._no_newcomer_run(
            "That was me.", None
        )
        self.assertTrue(consumed)
        self.assertNotIn("welcome", (ack or "").lower())
        self.assertFalse(mint.called)

    def test_already_met_third_party_is_placed_not_welcomed(self):
        consumed, ack, prompt = self._run(
            "That was Jeremy.", extracted="Jeremy",
            speaker_pid=1, resolved_pid=4, created=False, previously_met=True,
        )
        self.assertTrue(consumed)
        self.assertNotIn("welcome", (ack or "").lower())
        self.assertIn("do not welcome them", prompt.lower())

    def test_genuine_newcomer_is_still_welcomed(self):
        consumed, ack, prompt = self._run(
            "That was Jade.", extracted="Jade",
            speaker_pid=1, resolved_pid=99, created=True, previously_met=False,
        )
        self.assertTrue(consumed)
        self.assertIn("welcome", (ack or "").lower())
        self.assertIn("welcome them", prompt.lower())
        self.assertNotIn("do not welcome", prompt.lower())

    def test_newcomers_own_enrollment_does_not_make_them_a_regular(self):
        """already_met has to be sampled BEFORE this turn enrolls them: the
        enroll + first_enrollment familiarity bump happen a few lines earlier, so
        reading the row afterwards makes every brand-new person look like a
        returning one and swallows their welcome."""
        # The real predicate, against a row that already carries this turn's print.
        with mock.patch.object(
            I.people_memory, "get_person",
            return_value={"visit_count": 0, "familiarity_score": 0.1},
        ), mock.patch.object(I.people_memory, "has_voice_biometric", return_value=True), \
             mock.patch.object(I.people_memory, "has_face_biometric", return_value=False), \
             mock.patch.object(I, "_pending_offscreen_identify", self._pending()), \
             mock.patch.object(
                 I.llm, "extract_relationship_introduction",
                 return_value={"name": "Jade", "relationship": None},
             ), \
             mock.patch.object(
                 I.people_memory, "find_potential_person_match", return_value=None
             ), \
             mock.patch.object(
                 I.people_memory, "find_or_create_person", return_value=(99, True)
             ), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True), \
             mock.patch.object(I.speaker_id, "rank_speakers", return_value=[]), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=False), \
             mock.patch.object(I, "_bind_world_state_identity"), \
             mock.patch.object(I, "_retire_anonymous_speaker_slot"), \
             mock.patch.object(I, "_episodic_person_enrolled"), \
             mock.patch.object(I.people_memory, "update_familiarity"), \
             mock.patch.object(I.llm, "get_response", return_value=None) as get_response, \
             mock.patch.object(I, "_speak_blocking", return_value=True), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch.object(I, "_register_rex_utterance"):
            consumed, ack = I._handle_pending_offscreen_identify_reply(
                "That was Jade.",
                person_id=1,
                person_name="Bret Benziger",
                audio_array=np.zeros(16000, dtype=np.float32),
                anonymous_speaker_label=None,
            )
        prompt = get_response.call_args[0][0]
        self.assertTrue(consumed)
        self.assertIn("welcome", (ack or "").lower())
        self.assertNotIn("do not welcome", prompt.lower())


class SpeakerNameGuardTest(unittest.TestCase):
    """_same_first_name closes the hole the field answer walked through."""

    def test_first_name_only_matches_the_speaker(self):
        self.assertTrue(I._same_first_name("Bret", "Bret Benziger"))
        self.assertTrue(I._same_first_name("Bret Benziger", "Bret"))

    def test_different_people_do_not_match(self):
        self.assertFalse(I._same_first_name("Jade", "Bret Benziger"))
        self.assertFalse(I._same_first_name("PJ Thomas", "Jeremy Thomas"))
        self.assertFalse(I._same_first_name(None, "Bret"))

    def test_filter_drops_a_bare_first_name_matching_the_speaker(self):
        name, _rel = I._filter_relationship_introduction_evidence(
            {"name": "Bret", "relationship": None},
            "Bret Benziger said that.",
            "Bret Benziger",
            source="offscreen_identify",
        )
        self.assertIsNone(name)


class SelfIntroductionIsNotAnIntroductionTest(unittest.TestCase):
    """"This is Bret." said BY Bret must not open an introduction."""

    def test_predicate_matches_the_introducer(self):
        self.assertTrue(I._introduced_name_is_the_introducer("Bret", "Bret Benziger"))
        self.assertTrue(
            I._introduced_name_is_the_introducer("Bret Benziger", "Bret Benziger")
        )

    def test_predicate_lets_a_real_third_party_through(self):
        self.assertFalse(I._introduced_name_is_the_introducer("PJ", "Bret Benziger"))
        self.assertFalse(I._introduced_name_is_the_introducer("", "Bret Benziger"))

    def test_parse_handler_refuses_and_files_nothing(self):
        parsed = introductions.IntroductionParse(
            is_introduction=True, name="Bret", subject_kind="person", confidence=0.8
        )
        with mock.patch.object(I, "_pending_introduction", None), \
             mock.patch.object(I, "_enroll_introduced_person") as enroll, \
             mock.patch.object(I, "_store_introduction_memories") as store:
            resp = I._handle_introduction_parse(
                parsed, introducer_id=1, introducer_name="Bret Benziger",
                visible_newcomer=False,
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)
        self.assertFalse(store.called)

    def test_nickname_resolving_to_the_introducer_is_refused_at_the_row(self):
        """The name check can miss (a nickname, a middle name); the id check in
        _enroll_introduced_person is the backstop."""
        with mock.patch.object(
            I.people_memory, "find_or_create_person", return_value=(1, False)
        ), mock.patch.object(I, "_store_introduction_memories") as store, \
             mock.patch.object(I, "_episodic_person_enrolled") as episodic:
            new_id = I._enroll_introduced_person(
                "Benzo", 1, "Bret Benziger", None, enroll_visible_face=False,
            )
        self.assertIsNone(new_id)
        self.assertFalse(store.called)
        self.assertFalse(episodic.called)

    def test_real_introduction_still_enrolls(self):
        parsed = introductions.IntroductionParse(
            is_introduction=True, name="Jade", subject_kind="person", confidence=0.8
        )
        with mock.patch.object(I, "_pending_introduction", None), \
             mock.patch.object(I.people_memory, "find_person_by_name", return_value=None), \
             mock.patch.object(
                 I, "_resolve_existing_visible_introduced_person", return_value=None
             ), \
             mock.patch.object(I, "_enroll_introduced_person", return_value=99) as enroll, \
             mock.patch.object(I, "_mark_single_name_for_later_last_name"), \
             mock.patch.object(I, "_intro_ack_and_followup", return_value="Hey Jade."):
            resp = I._handle_introduction_parse(
                parsed, introducer_id=1, introducer_name="Bret Benziger",
                visible_newcomer=True,
            )
        self.assertEqual(resp, "Hey Jade.")
        enroll.assert_called_once()


class PreviouslyMetTest(unittest.TestCase):
    """A row on file is not the same as having met someone."""

    def test_visits_count_as_met(self):
        with mock.patch.object(
            I.people_memory, "get_person",
            return_value={"visit_count": 63, "familiarity_score": 1.0},
        ):
            self.assertTrue(I._person_previously_met(1))

    def test_name_only_row_is_not_met(self):
        with mock.patch.object(
            I.people_memory, "get_person",
            return_value={"visit_count": 0, "familiarity_score": 0.0},
        ), mock.patch.object(I.people_memory, "has_voice_biometric", return_value=False), \
             mock.patch.object(I.people_memory, "has_face_biometric", return_value=False):
            self.assertFalse(I._person_previously_met(5))

    def test_enrolled_prints_count_as_met(self):
        with mock.patch.object(
            I.people_memory, "get_person",
            return_value={"visit_count": 0, "familiarity_score": 0.0},
        ), mock.patch.object(I.people_memory, "has_voice_biometric", return_value=True), \
             mock.patch.object(I.people_memory, "has_face_biometric", return_value=False):
            self.assertTrue(I._person_previously_met(7))

    def test_missing_person_is_not_met(self):
        self.assertFalse(I._person_previously_met(None))


if __name__ == "__main__":
    unittest.main()
