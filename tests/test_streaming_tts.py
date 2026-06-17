"""Tests for streaming-answer → TTS (intelligence/interaction streaming path).

Covers the pure helpers (sentence splitter, per-sentence governance) and an
orchestration test that the streaming path enqueues sentences in order, at one
priority (so the single speech-queue worker can never overlap them), applies the
one-question cap across the reply, and returns the full spoken text.
"""

import threading
import types
import unittest
from unittest import mock

from intelligence import interaction as I
from intelligence import social_frame as SF
from intelligence import comedy_modes as CM


def _frame(allow_question=True, allow_visual_comment=True, allow_roast="normal",
           purpose="general"):
    return types.SimpleNamespace(
        allow_question=allow_question,
        allow_visual_comment=allow_visual_comment,
        allow_roast=allow_roast,
        purpose=purpose,
        max_sentences=4,
        max_words=80,
    )


class GovernBackReferenceTest(unittest.TestCase):
    def _frame_noq(self):
        return types.SimpleNamespace(
            allow_question=False, allow_visual_comment=True, allow_roast="normal",
            purpose="interest", max_sentences=4, max_words=80, reason="test",
        )

    def test_keeps_question_that_anchors_a_back_reference(self):
        # Dropping the leading question would orphan "That's like..." into a
        # non-sequitur (live: the "bass drop" line). Keep the anchor.
        out = SF.govern_response(
            "A sassy personality for your robot? That's like adding a bass drop.",
            self._frame_noq(),
        )
        self.assertIn("sassy personality", out.text)
        self.assertIn("bass drop", out.text)

    def test_still_strips_a_disallowed_question_with_no_back_reference(self):
        out = SF.govern_response(
            "What's your favorite color? I like blue myself.",
            self._frame_noq(),
        )
        self.assertNotIn("?", out.text)
        self.assertIn("blue", out.text)


class CleanResponseTextTest(unittest.TestCase):
    def test_strips_just_remember_opener(self):
        from intelligence import llm
        # Leading.
        self.assertEqual(
            llm.clean_response_text("Just remember, you owe me a tune."),
            "You owe me a tune.",
        )
        # Mid-reply sentence opener (the live "Nice to meet you... Just remember, I'm..." case).
        self.assertEqual(
            llm.clean_response_text("Nice to meet you, Bret. Just remember, I'm not just a pretty interface."),
            "Nice to meet you, Bret. I'm not just a pretty interface.",
        )

    def test_leaves_legitimate_remember_intact(self):
        from intelligence import llm
        # "just remember" not used as a sentence-opening crutch is untouched.
        self.assertEqual(
            llm.clean_response_text("I just remember the old days fondly."),
            "I just remember the old days fondly.",
        )


class SplitStreamSentencesTest(unittest.TestCase):
    def test_emits_completed_sentence_keeps_remainder(self):
        ready, rest = I._split_stream_sentences("First full sentence. Second", 12)
        self.assertEqual(ready, ["First full sentence."])
        self.assertEqual(rest, "Second")

    def test_short_fragment_merges_forward(self):
        # "No." (3 chars) is below the floor, so it coalesces into the next one
        # instead of being spoken as a choppy one-word burst.
        ready, rest = I._split_stream_sentences("No. A bigger sentence follows. Tail", 12)
        self.assertEqual(ready, ["No. A bigger sentence follows."])
        self.assertEqual(rest, "Tail")

    def test_no_boundary_yet_returns_empty(self):
        ready, rest = I._split_stream_sentences("Half a sentence with no", 12)
        self.assertEqual(ready, [])
        self.assertEqual(rest, "Half a sentence with no")

    def test_trailing_punctuation_stays_until_whitespace(self):
        # The final sentence has no trailing whitespace, so it is treated as
        # still-in-progress and held for the tail flush (never split early).
        ready, rest = I._split_stream_sentences("All done here.", 12)
        self.assertEqual(ready, [])
        self.assertEqual(rest, "All done here.")

    def test_multiple_sentences(self):
        ready, rest = I._split_stream_sentences("Aaaaaaaaaaa. Bbbbbbbbbbb. C", 12)
        self.assertEqual(ready, ["Aaaaaaaaaaa.", "Bbbbbbbbbbb."])
        self.assertEqual(rest, "C")

    def test_question_and_ellipsis_boundaries(self):
        ready, _ = I._split_stream_sentences("Are you serious right now? Yes", 12)
        self.assertEqual(ready, ["Are you serious right now?"])
        ready, _ = I._split_stream_sentences("Wait... what is happening here? More", 12)
        self.assertEqual(ready, ["Wait... what is happening here?"])

    def test_abbreviation_is_not_a_sentence_boundary(self):
        # "Mrs. Doubtfire" must NOT split at the period in "Mrs." — that truncated the
        # title and TTS read a mangled run-on (live failure 2026-06-16).
        ready, rest = I._split_stream_sentences(
            "I love Mrs. Doubtfire. Such a classic. Next", 12
        )
        self.assertEqual(ready, ["I love Mrs. Doubtfire.", "Such a classic."])
        self.assertEqual(rest, "Next")
        # Dr./St./U.S. likewise stay intact.
        ready, _ = I._split_stream_sentences("Dr. Aphra is back. Good", 12)
        self.assertEqual(ready, ["Dr. Aphra is back."])


class TailIsSpeakableTest(unittest.TestCase):
    """End-of-stream tail handling. The model likes to trail off mid-clause with
    an ellipsis ("…the excitement of…"); that passes the terminal-punctuation
    check but the ellipsis is stripped for TTS, leaving a bare dangling fragment
    that lands as a hard cut-off. Such tails must be dropped (the complete earlier
    sentence still plays), while genuinely finished tails are kept."""

    def test_complete_sentences_are_speakable(self):
        for tail in [
            "That is hilarious.",
            "Best concert of my life!",
            "Visual reacquired. There you are, Bret.",
            "Well, this is awkward…",          # trail-off on a complete-enough word
            "the joy of contemplating the universe.",
        ]:
            with self.subTest(tail=tail):
                self.assertTrue(I._tail_is_speakable(tail))

    def test_ellipsis_trailoff_on_dangling_word_is_dropped(self):
        # The two live cut-offs and other dangling trail-offs.
        for tail in [
            "I guess the excitement of…",
            "I guess the excitement of...",
            "I mean, I've…",
            "Maybe it was the…",
            "I was just thinking about…",
        ]:
            with self.subTest(tail=tail):
                self.assertFalse(I._tail_is_speakable(tail))

    def test_unpunctuated_or_empty_tail_is_dropped(self):
        for tail in ["Glad to", "I guess the excitement of", "", "   ", None]:
            with self.subTest(tail=tail):
                self.assertFalse(I._tail_is_speakable(tail))


class CompleteSentencePrefixTest(unittest.TestCase):
    """The streaming safety-net fallback must NOT re-emit a max-token-truncated
    fragment ("Wow indeed! I" — the live cut-off). _complete_sentence_prefix trims
    to the last complete sentence (recovering short finished ones the min-chars
    merge skipped) and returns "" when nothing complete remains."""

    def test_drops_truncated_tail_keeps_short_finished_sentence(self):
        self.assertEqual(I._complete_sentence_prefix("Wow indeed! I"), "Wow indeed!")

    def test_keeps_fully_complete_reply(self):
        self.assertEqual(
            I._complete_sentence_prefix("That's great. Tell me more."),
            "That's great. Tell me more.",
        )

    def test_no_complete_sentence_returns_empty(self):
        for text in ["I mean, I've", "Hi there, friend, how", "", "   ", None]:
            with self.subTest(text=text):
                self.assertEqual(I._complete_sentence_prefix(text), "")


class GovernStreamSentenceTest(unittest.TestCase):
    def test_drops_disallowed_question(self):
        self.assertEqual(SF.govern_stream_sentence("Who are you?", _frame(allow_question=False)), "")

    def test_keeps_allowed_question(self):
        self.assertEqual(SF.govern_stream_sentence("Who are you?", _frame(allow_question=True)), "Who are you?")

    def test_keeps_normal_statement(self):
        self.assertEqual(SF.govern_stream_sentence("I am fine today.", _frame()), "I am fine today.")

    def test_empty_returns_empty(self):
        self.assertEqual(SF.govern_stream_sentence("   ", _frame()), "")

    def test_is_question_sentence(self):
        self.assertTrue(SF.is_question_sentence("Who are you?"))
        self.assertFalse(SF.is_question_sentence("I am fine."))


class PolishStreamSentenceTest(unittest.TestCase):
    def test_straight_mode_passthrough(self):
        mode = types.SimpleNamespace(key="straight")
        self.assertEqual(CM.polish_stream_sentence("Just a line.", mode), "Just a line.")

    def test_collapses_overexplained_joke_tail(self):
        mode = types.SimpleNamespace(key="dry")
        out = CM.polish_stream_sentence("That was a great landing. Get it? Because I crashed.", mode)
        self.assertEqual(out, "That was a great landing.")


class StreamingOrchestrationTest(unittest.TestCase):
    """Drive _stream_and_speak_sentences with a fake LLM stream + fake queue."""

    def setUp(self):
        I._interrupted.clear()

    def tearDown(self):
        I._interrupted.clear()

    def _run(self, chunks, frame, *, surprising=False):
        enqueued = []

        def fake_enqueue(text, emotion, *, priority=1, pre_beat_ms=0,
                         post_beat_ms=0, voice_settings=None, on_start=None,
                         log_text=True):
            enqueued.append({
                "text": text, "emotion": emotion, "priority": priority,
                "pre_beat_ms": pre_beat_ms, "post_beat_ms": post_beat_ms,
                "log_text": log_text,
            })
            done = threading.Event()
            done.set()  # pretend playback finished so the drain returns at once
            return done

        def fake_stream(user_text, person_id=None, agenda_directive=None):
            for c in chunks:
                yield c

        mode = types.SimpleNamespace(key="straight")
        with mock.patch.object(I.llm, "stream_response", fake_stream), \
             mock.patch.object(I.speech_queue, "enqueue", side_effect=fake_enqueue), \
             mock.patch.object(I.empathy, "get_delivery_overrides", return_value=None), \
             mock.patch.object(I, "_prefetch_stream_audio"), \
             mock.patch.object(I, "_apply_post_tts_handoff"), \
             mock.patch.object(I.time, "sleep"):
            full = I._stream_and_speak_sentences(
                "hi rex", 1, frame, mode, "directive",
                {"value": surprising}, None, threading.Event(),
            )
        return enqueued, full

    def test_sentences_enqueued_in_order_same_priority(self):
        chunks = ["Hello there, Bret. ", "Good to see you again. ", "What's new with you?"]
        enqueued, full = self._run(chunks, _frame())
        texts = [e["text"] for e in enqueued]
        self.assertEqual(texts, [
            "Hello there, Bret.",
            "Good to see you again.",
            "What's new with you?",
        ])
        # All same priority → the single queue worker plays them strictly in
        # order; overlap is impossible.
        self.assertTrue(all(e["priority"] == 1 for e in enqueued))
        # Only the final line may carry a post-beat (handled after drain, so 0 here).
        self.assertTrue(all(e["post_beat_ms"] == 0 for e in enqueued))
        # Per-sentence conversation logging is suppressed; the turn is logged once.
        self.assertTrue(all(e["log_text"] is False for e in enqueued))
        self.assertEqual(full, "Hello there, Bret. Good to see you again. What's new with you?")

    def test_one_question_cap_across_reply(self):
        chunks = ["Who exactly are you? ", "And where are you from?"]
        enqueued, _ = self._run(chunks, _frame(allow_question=True))
        texts = [e["text"] for e in enqueued]
        self.assertEqual(texts, ["Who exactly are you?"])  # second question dropped

    def test_disallowed_question_sentence_skipped(self):
        chunks = ["Are you really sure? ", "I can handle that for you."]
        enqueued, full = self._run(chunks, _frame(allow_question=False))
        texts = [e["text"] for e in enqueued]
        self.assertEqual(texts, ["I can handle that for you."])
        self.assertEqual(full, "I can handle that for you.")

    def test_all_sentences_governed_away_falls_back(self):
        # allow_question False + an all-question reply → every sentence dropped;
        # the safety net speaks one whole-reply govern fallback instead of leaving
        # Rex silent.
        chunks = ["Who are you? ", "Where are you from?"]
        with mock.patch.object(
                I.social_frame, "govern_response",
                return_value=types.SimpleNamespace(text="Right, moving on.")), \
             mock.patch.object(
                I.comedy_modes, "polish_response", side_effect=lambda t, m, **k: t), \
             mock.patch.object(I, "_speak_blocking") as speak_blocking:
            enqueued, full = self._run(chunks, _frame(allow_question=False))
        self.assertEqual(enqueued, [])          # nothing streamed through the queue
        speak_blocking.assert_called_once()
        self.assertEqual(full, "Right, moving on.")

    def test_surprise_sets_emotion_on_first_sentence(self):
        chunks = ["Well that is unexpected news. ", "Tell me more about it."]
        enqueued, _ = self._run(chunks, _frame(), surprising=True)
        self.assertEqual(enqueued[0]["emotion"], "surprised")

    def test_interrupted_before_stream_returns_quickly(self):
        I._interrupted.set()
        try:
            enqueued, full = self._run(["Anything at all here."], _frame())
        finally:
            I._interrupted.clear()
        self.assertEqual(enqueued, [])
        self.assertEqual(full, "")


if __name__ == "__main__":
    unittest.main()
