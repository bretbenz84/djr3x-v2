"""Two-chunk TTS (TTS_FIRST_SENTENCE_SPLIT_ENABLED) — the latency/consistency middle path.

The first sentence synthesizes the instant the LLM produces it; everything after is
ONE second generation (a single v3 voice seam per reply, vs per-sentence drift).
Drives _stream_and_speak_sentences directly with the token stream and speech queue
mocked, and asserts the chunking contract.
"""

import threading
import unittest
from unittest import mock

import config
from intelligence import interaction as I


class _Frame:
    purpose = "general"
    allow_roast = "no"


class _Comedy:
    key = "dry"


def _run(tokens, *, two_chunk=True):
    """Run the streamer over a canned token stream; return list of enqueue calls."""
    calls = []

    def _fake_enqueue(text, emotion, **kw):
        calls.append({"text": text, "emotion": emotion, **kw})
        ev = threading.Event()
        ev.set()
        return ev

    filler_stop = threading.Event()
    with (
        mock.patch.object(config, "LEAN_BRAIN_ENABLED", True, create=True),
        mock.patch.object(config, "SELF_EMOTION_CLASSIFY_ENABLED", False, create=True),
        mock.patch.object(config, "LLM_STREAMING_PREFETCH_ENABLED", False, create=True),
        mock.patch.object(config, "LLM_STREAMING_MIN_SENTENCE_CHARS", 5, create=True),
        mock.patch.object(I, "_reply_token_stream", return_value=iter(tokens)),
        mock.patch.object(I.speech_queue, "enqueue", side_effect=_fake_enqueue),
        mock.patch.object(I.empathy, "get_delivery_overrides", return_value=None),
        mock.patch.object(I.comedy_modes, "voice_settings_for_mode", return_value=None),
        mock.patch.object(I.conv_log, "log_rex_stream"),
        mock.patch.object(I.conv_log, "log_rex"),
        mock.patch.object(I.conv_log, "finish_rex_stream"),
        mock.patch.object(I, "_mark_first_response_queued"),
        mock.patch.object(I, "_await_streamed_speech", return_value=True),
        mock.patch.object(I, "_apply_post_tts_handoff"),
        mock.patch.object(I, "_latency_log"),
        mock.patch.object(I, "_play_event_body_beat"),
    ):
        spoken = I._stream_and_speak_sentences(
            "hi", 1, _Frame(), _Comedy(), "",
            {"value": None}, None, None, filler_stop,
            two_chunk=two_chunk,
        )
    return spoken, calls


class TwoChunkTest(unittest.TestCase):
    def test_reply_speaks_as_exactly_two_generations(self):
        spoken, calls = _run([
            "Well now. ", "That is a very ", "interesting thought. ",
            "Tell me more about it.",
        ])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["text"], "Well now.")           # first sentence, instant
        self.assertEqual(
            calls[1]["text"],
            "That is a very interesting thought. Tell me more about it.",
        )
        # the reply's single audio tag rides chunk 1; chunk 2 suppresses it
        self.assertFalse(calls[0].get("suppress_audio_tag"))
        self.assertTrue(calls[1].get("suppress_audio_tag"))
        self.assertEqual(spoken, "Well now. That is a very interesting thought. "
                                 "Tell me more about it.")

    def test_single_sentence_reply_is_one_generation_no_seam(self):
        spoken, calls = _run(["Just the one line here."])
        self.assertEqual(len(calls), 1)
        self.assertEqual(spoken, "Just the one line here.")

    def test_question_cap_holds_across_the_seam(self):
        # First sentence is a question; a second question in the remainder is dropped
        # (the one-question-per-reply cap must survive the chunk merge).
        spoken, calls = _run([
            "Want to hear a secret? ", "I run on gossip. ", "Should I tell you more?",
        ])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["text"], "Want to hear a secret?")
        self.assertEqual(calls[1]["text"], "I run on gossip.")   # second question gone

    def test_full_streaming_mode_still_speaks_per_sentence(self):
        _spoken, calls = _run(
            ["One here. ", "Two here. ", "Three here."], two_chunk=False,
        )
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
