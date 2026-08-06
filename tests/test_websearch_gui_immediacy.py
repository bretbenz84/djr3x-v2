"""
A searched answer must reach the transcript/GUI when the TEXT is ready, not when
Rex has finished saying it.

Field 2026-08-06: asking for more about a news story showed nothing in the GUI
for ~30 seconds. The answer rode the caller's post-return log in
`_handle_speech_segment`, but `_maybe_web_search_reply` speaks via
`_speak_blocking`, which returns only after PLAYBACK completes — so a long spoken
answer stayed invisible for its entire duration. The streaming reply path logs as
soon as the text exists (that is why ordinary replies appear instantly); this path
now matches it, and marks the text so the caller does not write it twice.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import interaction as I


class PreloggedMarkerTests(unittest.TestCase):

    def setUp(self) -> None:
        I._prelogged_response.clear()
        self.addCleanup(I._prelogged_response.clear)

    def test_absent_by_default(self):
        self.assertIsNone(I._consume_prelogged_response())

    def test_round_trip(self):
        I._note_prelogged_response("The eclipse is on August 12.")
        self.assertEqual(I._consume_prelogged_response(), "The eclipse is on August 12.")

    def test_is_one_shot(self):
        # The caller consumes it once; a later turn must not still see it.
        I._note_prelogged_response("something")
        I._consume_prelogged_response()
        self.assertIsNone(I._consume_prelogged_response())

    def test_empty_text_is_not_marked(self):
        for junk in ("", "   ", None):
            with self.subTest(junk=junk):
                I._note_prelogged_response(junk)
                self.assertIsNone(I._consume_prelogged_response())

    def test_a_new_turn_clears_a_stale_marker(self):
        # If a reply path errors after logging but before the caller consumes the
        # marker, it must not suppress a legitimate write on the NEXT turn.
        I._note_prelogged_response("stale from a previous turn")
        I._begin_user_turn()
        self.assertIsNone(I._consume_prelogged_response())


class CallerSuppressionTests(unittest.TestCase):
    """The caller writes the transcript only when the reply path did not."""

    def setUp(self) -> None:
        I._prelogged_response.clear()
        self.addCleanup(I._prelogged_response.clear)

    def _caller_would_write(self, response_text: str) -> bool:
        # Mirrors the guard in _handle_speech_segment.
        return I._consume_prelogged_response() != response_text

    def test_an_ordinary_reply_is_still_written_by_the_caller(self):
        self.assertTrue(self._caller_would_write("A normal streamed reply."))

    def test_a_prelogged_answer_is_not_written_twice(self):
        answer = "The eclipse is on August 12, over Greenland and Iceland."
        I._note_prelogged_response(answer)
        self.assertFalse(self._caller_would_write(answer))

    def test_a_different_reply_in_the_same_turn_still_gets_written(self):
        I._note_prelogged_response("the searched answer")
        self.assertTrue(self._caller_would_write("a different line entirely"))


class WiringTests(unittest.TestCase):

    def test_the_search_path_logs_before_it_speaks(self):
        # Order is the whole fix: the transcript write must precede _speak_blocking.
        import inspect
        src = inspect.getsource(I._maybe_web_search_reply)
        log_at = src.index("conv_log.log_rex(answer_text)")
        speak_at = src.index("_speak_blocking(answer_text")
        self.assertLess(log_at, speak_at,
                        "the answer must be logged BEFORE playback, not after")
        self.assertIn("_note_prelogged_response(answer_text)", src)

    def test_the_caller_consults_the_marker(self):
        import inspect
        src = inspect.getsource(I._handle_speech_segment)
        self.assertIn("_consume_prelogged_response() != response_text", src)

    def test_a_transcript_failure_cannot_block_the_answer(self):
        # Logging is best-effort — a GUI/transcript error must never cost the reply.
        import inspect
        src = inspect.getsource(I._maybe_web_search_reply)
        block = src[src.index("conv_memory.add_to_transcript(\"Rex\", answer_text)"):]
        self.assertIn("except Exception", block[:400])


if __name__ == "__main__":
    unittest.main()
