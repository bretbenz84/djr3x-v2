"""
tests/test_reply_emotion_voice.py — the reply's OWN emotion outranks the comedy
stance's timbre (owner call 2026-09-02).

Field 2026-09-02 00:35:21: "Are you excited about your new microphone?" → stance
"dry acknowledgment" (chosen before the model wrote a word) → deadpan profile
(stability 0.66, style 0.20, speed 0.97) → the model wrote "[excited] Very, Bret —
..." → ElevenLabs got an [excited] tag wrapped in deadpan settings, and the owner
heard a flat "yes". Drives the streaming speaker with the token stream and speech
queue mocked, the way tests/test_two_chunk_tts.py does.
"""

import threading
import unittest
from unittest import mock

import config
from intelligence import interaction as I

DEADPAN = {"stability": 0.66, "style": 0.20, "similarity_boost": 0.82, "speed": 0.97}


class _Frame:
    purpose = "answer"
    allow_roast = "sharp"


class _Comedy:
    key = "dry_ack"


def _run(tokens, *, comedy_voice=DEADPAN, self_emotion=None, empathy=None,
         override_on=True, two_chunk=False):
    calls = []

    def _fake_enqueue(text, emotion, **kw):
        calls.append({"text": text, "emotion": emotion, **kw})
        ev = threading.Event()
        ev.set()
        return ev

    filler_stop = threading.Event()
    patches = [
        mock.patch.object(config, "LEAN_BRAIN_ENABLED", True, create=True),
        mock.patch.object(config, "LLM_STREAMING_PREFETCH_ENABLED", False, create=True),
        mock.patch.object(config, "LLM_STREAMING_MIN_SENTENCE_CHARS", 5, create=True),
        mock.patch.object(config, "REPLY_EMOTION_OVERRIDES_COMEDY_VOICE", override_on, create=True),
        mock.patch.object(config, "SELF_EMOTION_CLASSIFY_ENABLED", self_emotion is not None, create=True),
        mock.patch.object(I, "_reply_token_stream", return_value=iter(tokens)),
        mock.patch.object(I.speech_queue, "enqueue", side_effect=_fake_enqueue),
        mock.patch.object(I.empathy, "get_delivery_overrides", return_value=empathy),
        mock.patch.object(I.comedy_modes, "voice_settings_for_mode", return_value=comedy_voice),
        mock.patch.object(I.conv_log, "log_rex_stream"),
        mock.patch.object(I.conv_log, "log_rex"),
        mock.patch.object(I.conv_log, "finish_rex_stream"),
        mock.patch.object(I, "_mark_first_response_queued"),
        mock.patch.object(I, "_await_streamed_speech", return_value=True),
        mock.patch.object(I, "_apply_post_tts_handoff"),
        mock.patch.object(I, "_latency_log"),
        mock.patch.object(I, "_play_event_body_beat"),
        mock.patch.object(I, "_set_body_mood"),
    ]
    if self_emotion is not None:
        patches.append(mock.patch.object(I.llm, "classify_self_emotion", return_value=self_emotion))
    for p in patches:
        p.start()
    try:
        I._stream_and_speak_sentences(
            "are you excited about the new mics", 1, _Frame(), _Comedy(), "",
            {"value": None}, None, None, filler_stop, two_chunk=two_chunk,
        )
    finally:
        for p in reversed(patches):
            p.stop()
    return calls


def _energetic():
    from intelligence import emotion_orchestrator
    return emotion_orchestrator.voice_settings_for_emotion("excited")


class TagOverridesComedyProfileTest(unittest.TestCase):
    def test_field_case_excited_tag_beats_deadpan_on_the_first_sentence(self):
        calls = _run(["[excited] Very, Bret — I'm hoping this one lands. ",
                      "Less crash report, more upgrade."])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["emotion"], "excited")
        self.assertEqual(calls[0]["voice_settings"], _energetic())
        self.assertNotEqual(calls[0]["voice_settings"], DEADPAN)
        # The whole reply carries it, not just the tagged sentence.
        self.assertEqual(calls[1]["voice_settings"], _energetic())

    def test_untagged_neutral_reply_keeps_the_stance_timbre(self):
        calls = _run(["Sure. ", "It's a shelf, apparently."])
        for c in calls:
            self.assertEqual(c["emotion"], "neutral")
            self.assertEqual(c["voice_settings"], DEADPAN)

    def test_a_stance_tag_is_not_a_feeling(self):
        calls = _run(["[sarcastic] Sure, a shelf. ", "Very convincing."])
        self.assertEqual(calls[0]["emotion"], "neutral")
        self.assertEqual(calls[0]["voice_settings"], DEADPAN)

    def test_self_emotion_read_lifts_the_rest_of_the_reply(self):
        calls = _run(["Oh this is good. ", "The array lands tomorrow. ", "Finally."],
                     self_emotion="excited")
        # The classifier is started on sentence one and read from sentence two on.
        self.assertEqual(calls[0]["voice_settings"], DEADPAN)
        self.assertEqual(calls[1]["voice_settings"], _energetic())
        self.assertEqual(calls[2]["voice_settings"], _energetic())

    def test_empathy_delivery_still_wins(self):
        grief = {"mode": "support", "emotion": "sad", "voice_settings": {"stability": 0.7, "style": 0.1}}
        calls = _run(["[excited] I'm so sorry. ", "Take your time."], empathy=grief)
        for c in calls:
            self.assertEqual(c["voice_settings"], grief["voice_settings"])

    def test_kill_switch_keeps_the_old_behaviour(self):
        calls = _run(["[excited] Very, Bret."], override_on=False)
        self.assertEqual(calls[0]["voice_settings"], DEADPAN)
        self.assertEqual(calls[0]["emotion"], "excited")   # eyes/motion still follow the tag

    def test_two_chunk_mode_lifts_the_remainder_too(self):
        calls = _run(["[excited] Very, Bret. ", "Less crash report, ", "more upgrade."],
                     two_chunk=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["voice_settings"], _energetic())
        self.assertEqual(calls[1]["voice_settings"], _energetic())

    def test_no_comedy_profile_and_no_emotion_leaves_none(self):
        calls = _run(["Sure."], comedy_voice=None)
        self.assertIsNone(calls[0]["voice_settings"])


class TagEmotionReadTest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(I._reply_emotion_from_tag("[excited] Very, Bret."), "excited")
        self.assertEqual(I._reply_emotion_from_tag("Well [curious] what is that?"), "curious")
        self.assertEqual(I._reply_emotion_from_tag("[laughs] Classic."), "happy")
        self.assertIsNone(I._reply_emotion_from_tag("[sarcastic] Sure."))
        self.assertIsNone(I._reply_emotion_from_tag("[mischievously] Sure."))
        self.assertIsNone(I._reply_emotion_from_tag("no tag here"))
        self.assertIsNone(I._reply_emotion_from_tag(""))

    def test_first_tag_wins(self):
        self.assertEqual(I._reply_emotion_from_tag("[sarcastic] Sure. [excited] Wait."), "excited")


if __name__ == "__main__":
    unittest.main()
