"""
Streaming-reply timeout. A stalled OpenAI stream (200 OK received, then the token
stream goes silent) used to block the turn on the SDK's 600s default — and because the
turn handler holds AEC mic-suppression until the reply finishes, Rex went deaf AND mute
for up to ten minutes (observed 2026-06-14, froze until force-quit). The client now has
a bounded default timeout and the streaming call a tighter per-read timeout; on timeout
the stream raises, a fallback line is yielded, and the turn COMPLETES so the mic releases.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import llm


_FALLBACK = "...my circuits are experiencing some turbulence. Try again."


def _chunk(text):
    """Minimal stand-in for an OpenAI streaming chunk: chunk.choices[0].delta.content."""
    delta = mock.Mock()
    delta.content = text
    choice = mock.Mock()
    choice.delta = delta
    ch = mock.Mock()
    ch.choices = [choice]
    return ch


class ClientTimeoutBoundedTest(unittest.TestCase):
    def test_client_default_timeout_is_not_the_600s_default(self):
        # The bug was the SDK's 600s default. The client must carry our bound.
        self.assertEqual(llm._client.timeout, config.LLM_REQUEST_TIMEOUT_SECS)
        self.assertLess(float(config.LLM_REQUEST_TIMEOUT_SECS), 600.0)

    def test_max_retries_configured(self):
        self.assertEqual(llm._client.max_retries, config.LLM_MAX_RETRIES)


class StreamResponseTimeoutTest(unittest.TestCase):
    def setUp(self):
        self._assemble = mock.patch.object(
            llm, "assemble_system_prompt", return_value="SYS"
        )
        self._assemble.start()

    def tearDown(self):
        self._assemble.stop()

    def test_streaming_call_passes_read_timeout(self):
        with mock.patch.object(llm._client.chat.completions, "create") as create:
            create.return_value = iter([_chunk("hello "), _chunk("there")])
            out = "".join(llm.stream_response("hi", person_id=1))
        self.assertEqual(out, "hello there")
        self.assertTrue(create.called)
        kwargs = create.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["timeout"], config.LLM_STREAM_TIMEOUT_SECS)

    def test_stall_before_first_token_yields_fallback_and_completes(self):
        # The exact field case: connect succeeds, then the stream raises (timeout).
        import openai

        def _raise(*a, **k):
            raise openai.APITimeoutError(request=mock.Mock())

        with mock.patch.object(llm._client.chat.completions, "create", side_effect=_raise):
            out = list(llm.stream_response("hi", person_id=1))
        # The generator completes (does not hang) and emits the fallback line so the
        # turn finishes and AEC suppression is released downstream.
        self.assertEqual(out, [_FALLBACK])

    def test_stall_midstream_yields_partial_then_fallback(self):
        import openai

        def _gen():
            yield _chunk("Well, look ")
            raise openai.APITimeoutError(request=mock.Mock())

        with mock.patch.object(llm._client.chat.completions, "create", return_value=_gen()):
            out = list(llm.stream_response("hi", person_id=1))
        self.assertEqual(out[0], "Well, look ")
        self.assertEqual(out[-1], _FALLBACK)


if __name__ == "__main__":
    unittest.main()
