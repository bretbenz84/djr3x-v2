"""20:35 field run: context-prompt decodes disappeared as 'silence'."""
import unittest
from unittest import mock
import numpy as np
from audio import transcription as T
from intelligence import interaction as I, repair_moves


class RejectedAudioTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(repair_moves.clear)
        repair_moves.clear()
        for patch in [mock.patch.object(T, "_context_backoff_until", 0.),
                      mock.patch.object(I, "_last_low_trust_reprompt_at", 0.),
                      mock.patch.object(I, "_game_suppresses_conversation", return_value=False)]:
            patch.start()
            self.addCleanup(patch.stop)

    def rejected(self):
        return T.Transcript("", confident=False, backend="qwen3_asr", rejection_reason="context-echo")

    def test_prompt_rejection_retains_reason_and_temporarily_disables_bias(self):
        audio = np.zeros(32000, dtype=np.float32)
        with mock.patch.object(T, "_qwen_backend_selected", return_value=True), \
             mock.patch.object(T, "_qwen_ready", return_value=True), \
             mock.patch.object(T, "_QWEN_LOAD_FAILED", False), \
             mock.patch.object(T, "_qwen_transcribe", side_effect=[
                 ("This audio is one side of a live spoken conversation.", 0.), ("Okay.", -.75)]), \
             mock.patch("openai.OpenAI") as api:
            out = T.transcribe(audio)
        self.assertFalse(out)
        self.assertFalse(out.confident)
        self.assertEqual(out.rejection_reason, "context-echo")
        self.assertIsNone(T._asr_context_prompt())
        with mock.patch.object(T.time, "monotonic", return_value=T._context_backoff_until + 1):
            self.assertTrue(T._asr_context_prompt())
        api.assert_not_called()

    def test_audio_processing_preserves_empty_transcript_metadata(self):
        out = self.rejected()
        with mock.patch.object(I.transcription, "transcribe", return_value=out), \
             mock.patch.object(I.speaker_id, "voiced_secs", return_value=1.), \
             mock.patch.object(I.speaker_id, "buffer_secs", return_value=2.), \
             mock.patch.object(I.speaker_id, "rank_speakers", return_value=[]), \
             mock.patch.object(I.speaker_id, "window_evidence", return_value=[]):
            result = I._process_audio(np.zeros(32000, dtype=np.float32))
        self.assertIs(result[0], out)

    def gate(self, *, text=None, voiced=1., duration=2., started=100., playback=90., **kwargs):
        with mock.patch.object(I, "_last_scan_secs", {"voiced": voiced, "buffer": duration}), \
             mock.patch.object(I, "_utterance_observations", {"started_at": started}), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at", return_value=playback):
            return I._should_reprompt_rejected_audio(
                self.rejected() if text is None else text,
                require_trusted=kwargs.get("require_trusted", False), from_idle=kwargs.get("from_idle", False))

    def test_substantial_post_playback_voice_can_request_repeat(self):
        self.assertTrue(self.gate())

    def test_empty_silence_and_playback_residue_stay_quiet(self):
        for args in [{"text": ""}, {"voiced": .42, "duration": 5.68},
                     {"started": 90.5}, {"started": None}, {"require_trusted": True},
                     {"from_idle": True}, {"voiced": .8, "duration": 8.}]:
            with self.subTest(args=args):
                self.assertFalse(self.gate(**args))

    def test_reprompt_cooldown_and_phantom_standdown_are_preserved(self):
        I._arm_low_trust_reprompt_cooldown()
        self.assertFalse(self.gate())
        I._last_low_trust_reprompt_at = 0.
        repair_moves.phantom_audio_response()
        self.assertFalse(self.gate())

    def test_actual_handler_reprompts_without_inventing_heard_words(self):
        from tests.test_voice_learning import RuntimeTests
        fixture = RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        real_reprompt_gate = I._should_reprompt_low_trust
        fixture._mock_speech_pipeline()
        audio = fixture.prepare()
        with mock.patch.object(I, "_process_audio", return_value=(self.rejected(), 1, "Bret", .8, .3, .07)), \
             mock.patch.object(I, "_should_reprompt_low_trust", real_reprompt_gate), \
             mock.patch.object(I, "_last_scan_secs", {"voiced": 1., "buffer": 2.}), \
             mock.patch.object(I.consciousness, "begin_response_wait"):
            I._handle_speech_segment(audio)
        self.assertIn("couldn't make out", I._speak_blocking.call_args.args[0])
        I.conv_log.log_heard.assert_not_called()
        I._reply_token_stream.assert_not_called()
