"""
What Rex is allowed to BELIEVE, and what he may bring up later unprompted.

All of this is one field failure, traced 2026-07-25 from
logs/conversation-2026-07-25-15-59-52.log, where four of six proactive lines were
about things that never happened. The chain ran:

  Whisper fabricates ("wine" -> "I'm going to split it.")
    -> nothing marks the transcript as a guess
    -> the shutdown summarizer records it, mixed with REX'S OWN jokes, as fact
    -> open_threads are mined from that
    -> the lull speaker asks about it days later
    -> the human corrects him
    -> the correction is summarized into NEW open_threads
    -> repeat, forever, with the loop supplying its own material.

Each layer below is one link in that chain.
"""

import unittest
from unittest import mock

import config


# ── 1. Whisper's confidence is read, and low confidence means "don't learn" ──

class TranscriptConfidenceTest(unittest.TestCase):
    def setUp(self):
        from audio import transcription
        self.T = transcription

    def test_transcript_is_still_a_string(self):
        # Every existing caller treats the return value as a str; the confidence
        # fields ride along without breaking any of them.
        t = self.T.Transcript("hello there", avg_logprob=-0.3, confident=True)
        self.assertEqual(t, "hello there")
        self.assertEqual(t.upper(), "HELLO THERE")
        self.assertEqual(len(t), 11)
        self.assertTrue(t.confident)

    def test_decode_stats_summarize_the_segments(self):
        stats = self.T._decode_stats({"segments": [
            {"avg_logprob": -0.4, "no_speech_prob": 0.1},
            {"avg_logprob": -0.6, "no_speech_prob": 0.7},
        ]})
        self.assertAlmostEqual(stats[0], -0.5)     # mean logprob
        self.assertAlmostEqual(stats[1], 0.7)      # WORST no-speech reading

    def test_missing_stats_are_treated_as_trustworthy(self):
        # The OpenAI fallback path returns no per-segment stats. Absent evidence is
        # not evidence of guessing — refusing to learn from every API turn would be
        # its own bug.
        self.assertEqual(self.T._decode_stats({}), (None, None))
        self.assertTrue(self.T._is_confident(None, None))

    def test_a_confident_decode_passes(self):
        self.assertTrue(self.T._is_confident(-0.35, 0.05))

    def test_guessing_is_caught(self):
        self.assertFalse(self.T._is_confident(-1.4, 0.05))   # logprob floor
        self.assertFalse(self.T._is_confident(-0.2, 0.92))   # it was probably silence

    def test_the_gate_is_deliberately_permissive(self):
        # Far-field SNR here is 13-15 dB and real speech scores badly often. This is
        # a LEARNING gate, not a hearing gate — a middling decode must still pass, or
        # Rex goes deaf to the owner across the room.
        self.assertTrue(self.T._is_confident(-0.8, 0.4))


# ── 2. Rex's own talk is never evidence about the human ─────────────────────

class DiaryAttributionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import llm
        self.llm = llm
        # The real 2026-07-24 turns behind "how did the baking turn out?"
        self.transcript = [
            {"speaker": "Bret Benziger", "text": "I might do some baking later."},
            {"speaker": "Rex", "text": "Baking? Your confidence sounds like a cry "
                                      "for help from the oven."},
        ]

    def test_rex_turns_are_marked_as_non_evidence(self):
        formatted = self.llm._format_transcript_attributed(self.transcript)
        rex_line = [l for l in formatted.splitlines() if "cry for help" in l][0]
        human_line = [l for l in formatted.splitlines() if "might do some baking" in l][0]
        self.assertIn("YOU, Rex", rex_line)
        self.assertIn("never evidence", rex_line)
        self.assertIn("HUMAN", human_line)

    def test_the_prompt_carries_the_attribution_rule(self):
        # Episode 450 recorded Rex's own "cry for help from the oven" tic as a fact
        # about Bret. The rule that forbids it has to actually reach the model.
        captured = {}

        class _Resp:
            choices = [mock.MagicMock(message=mock.MagicMock(
                content='{"remember": false, "note": "", "salience": 0.0, '
                        '"open_threads": []}'))]

        def _create(**kw):
            captured["prompt"] = kw["messages"][0]["content"]
            return _Resp()

        with mock.patch.object(self.llm._client.chat.completions, "create", _create):
            self.llm.generate_diary_entry(self.transcript, people_names=["Bret"])
        prompt = captured["prompt"]
        self.assertIn("ATTRIBUTION", prompt)
        self.assertIn("[HUMAN]", prompt)
        self.assertIn("cry for help", prompt)        # transcript is present...
        self.assertIn("YOU, Rex", prompt)            # ...but attributed to Rex


# ── 3. A correction closes a thread; it never opens one ─────────────────────

class DenialClosesThreadTest(unittest.TestCase):
    """The self-feeding loop. Episode 461 (2026-07-25 16:05) was written from the
    three turns where Bret corrected Rex, and filed BOTH of its open threads from
    them — so the next session would have opened with the same invented topics."""

    def setUp(self):
        from intelligence import llm
        self.llm = llm
        self.transcript = [
            {"speaker": "Rex", "text": "So how did the baking turn out, or did the oven win?"},
            {"speaker": "Bret Benziger", "text": "We didn't end up doing that."},
            {"speaker": "Rex", "text": "So, did Black Widow's pizza earn a comeback?"},
            {"speaker": "Bret Benziger", "text": "We got Mount Mike's Pizza, not Black Widow Pizza."},
            {"speaker": "Rex", "text": "So what actually changed in the routine?"},
            {"speaker": "Bret Benziger", "text": "I don't know what you're talking about."},
        ]

    def test_the_exact_threads_episode_461_filed_are_dropped(self):
        filed = ["whether they will attempt baking again",
                 "how the Mount Mike's Pizza turned out"]
        self.assertEqual(self.llm._filtered_open_threads(filed, self.transcript), [])

    def test_every_denial_shape_in_that_session_is_recognized(self):
        for text in ("We didn't end up doing that.",
                     "We got Mount Mike's Pizza, not Black Widow Pizza.",
                     "I don't know what you're talking about.",
                     "That's not right.", "I never said that."):
            with self.subTest(text=text):
                self.assertTrue(self.llm._human_denied_something(
                    [{"speaker": "Bret", "text": text}]))

    def test_a_denial_with_no_nouns_still_closes_its_topic(self):
        # "I don't know what you're talking about" names nothing — the subject lives
        # in the Rex question it answers, so that turn counts as denied too.
        self.assertEqual(
            self.llm._filtered_open_threads(["what changed in his routine"], self.transcript), [])

    def test_a_real_thread_survives_a_session_containing_denials(self):
        # Over-correcting here would be its own failure: deny one thing and Rex
        # forgets everything else you said.
        transcript = self.transcript + [
            {"speaker": "Bret Benziger", "text": "I did adopt a dog named Max last week."},
        ]
        self.assertEqual(
            self.llm._filtered_open_threads(
                ["how Max is settling in", "whether they will attempt baking again"],
                transcript),
            ["how Max is settling in"])

    def test_a_clean_session_is_untouched(self):
        clean = [{"speaker": "Bret", "text": "I'm finally booking the dentist tomorrow."},
                 {"speaker": "Rex", "text": "Brave."}]
        self.assertFalse(self.llm._human_denied_something(clean))
        self.assertEqual(
            self.llm._filtered_open_threads(["whether the dentist appointment happened"], clean),
            ["whether the dentist appointment happened"])

    def test_threads_are_still_capped(self):
        clean = [{"speaker": "Bret", "text": "Lots happened today."}]
        self.assertEqual(len(self.llm._filtered_open_threads(list("abcdef"), clean)), 3)


# ── 3b. A guess never mints a person or a room ──────────────────────────────

class GuessNeverEnrollsTest(unittest.TestCase):
    """The most expensive failure of the lot. An utterance decoded as "Spice it."
    created a PERSON named Spice — twice, one second apart — because Rex had just
    asked "what do I call you?" and the name latch takes the next short utterance.
    "This is the workshop room" was decoded "Shop room." and made a room by that
    name. Both are permanent, both attach biometrics, and both keep colliding with
    the real human or room forever after."""

    def test_person_enrollment_refuses_a_low_confidence_turn(self):
        from intelligence import interaction
        with mock.patch.object(interaction, "_turn_transcript_trusted", return_value=False):
            import numpy as np
            self.assertIsNone(interaction._enroll_new_person(
                "Spice", np.zeros(16000, dtype=np.float32)))

    def test_room_enrollment_refuses_a_low_confidence_turn(self):
        from intelligence import place_questions
        svc = mock.MagicMock()
        with mock.patch.object(place_questions, "_service", return_value=svc), \
                mock.patch.object(place_questions, "_transcript_trusted", return_value=False):
            self.assertIsNone(place_questions._enroll("shop room"))
        svc.enroll.assert_not_called()

    def test_room_enrollment_proceeds_when_the_transcript_is_trusted(self):
        from intelligence import place_questions
        svc = mock.MagicMock()
        svc.enroll.return_value = 7
        with mock.patch.object(place_questions, "_service", return_value=svc), \
                mock.patch.object(place_questions, "_transcript_trusted", return_value=True):
            self.assertEqual(place_questions._enroll("workshop"), 7)

    def test_trust_defaults_to_true_outside_a_turn(self):
        # Background/offline callers have no turn context; they must not be
        # silently prevented from ever enrolling anything.
        from intelligence.interaction import _turn_transcript_trusted
        self.assertTrue(_turn_transcript_trusted())


# ── 4. A camera guess is not a personal detail ──────────────────────────────

class VisionLabelBankingTest(unittest.TestCase):
    """All nine callback rows in the field DB came from the object detector. The
    worst filed the owner himself: "has Bret Benziger in their space"."""

    def setUp(self):
        from intelligence import exploration
        self.X = exploration

    def test_a_detected_person_is_not_a_possession(self):
        self.assertFalse(self.X._label_bankable("Bret Benziger", "Bret Benziger"))

    def test_a_known_person_is_rejected_even_when_someone_else_is_present(self):
        with mock.patch("memory.people.list_person_names",
                        return_value=["Bret Benziger", "Exudica Marbles"]):
            self.assertFalse(self.X._label_bankable("Exudica Marbles", "Bret Benziger"))

    def test_the_detectors_own_hedges_are_rejected(self):
        for label in ("Mysterious black object", "Mystery organic", "unknown item",
                      "unidentified shape", "a possible chair", "some thing"):
            with self.subTest(label=label):
                self.assertFalse(self.X._label_bankable(label, None))

    def test_labels_too_generic_to_mention_are_rejected(self):
        for label in ("object", "thing", "person", "wall", "background"):
            with self.subTest(label=label):
                self.assertFalse(self.X._label_bankable(label, None))

    def test_a_genuinely_personal_object_still_banks(self):
        for label in ("bicycle sculpture", "Forest mural", "Memorabilia", "Plant"):
            with self.subTest(label=label):
                self.assertTrue(self.X._label_bankable(label, None))

    def test_banked_vision_rows_record_that_rex_only_SAW_it(self):
        # 'explicit' was hardcoded, so a camera guess carried the same standing as
        # something the person said out loud.
        from memory import callbacks
        sess = mock.MagicMock(person_id=1, person_name="Bret Benziger")
        with mock.patch.object(callbacks, "bank", return_value=1) as bank, \
                mock.patch.object(config, "EXPLORE_BANK_CALLBACK_ENABLED", True, create=True):
            self.X._bank_callback(sess, {"name": "bicycle sculpture"})
        self.assertEqual(bank.call_args.kwargs["source"], "observed")

    def test_a_rejected_label_never_reaches_the_database(self):
        from memory import callbacks
        sess = mock.MagicMock(person_id=1, person_name="Bret Benziger")
        with mock.patch.object(callbacks, "bank") as bank, \
                mock.patch.object(config, "EXPLORE_BANK_CALLBACK_ENABLED", True, create=True):
            self.X._bank_callback(sess, {"name": "Bret Benziger"})
        bank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
