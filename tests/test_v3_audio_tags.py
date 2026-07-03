"""Eleven v3 audio-tag injection (audio/tts.py) — deterministic affect -> tag, transcript-clean.

Tags shape delivery at synthesis but must NEVER reach the transcript/log. The mapping is:
comedy_mode (sarcasm/mischief) wins, else the reply emotion, else nothing (neutral/sincere).
Stability is pinned to Creative on any tagged line (high stability mutes tags per the v3 docs).
"""

import unittest
from unittest import mock

import config
from audio import tts


class ResolveTagTest(unittest.TestCase):
    def test_comedy_mode_wins(self):
        self.assertEqual(tts.resolve_audio_tag("neutral", "smug_superiority"), "sarcastic")
        self.assertEqual(tts.resolve_audio_tag("neutral", "friendly_roast"), "sarcastic")
        self.assertEqual(tts.resolve_audio_tag("neutral", "appliance_conspiracy"), "mischievously")
        self.assertEqual(tts.resolve_audio_tag("neutral", "self_own"), "snorts")

    def test_emotion_fallback(self):
        self.assertEqual(tts.resolve_audio_tag("excited", None), "excited")
        self.assertEqual(tts.resolve_audio_tag("curious", None), "curious")
        self.assertEqual(tts.resolve_audio_tag("happy", None), "laughs")

    def test_none_for_neutral_or_sincere_or_unknown(self):
        for emo, cm in [("neutral", None), ("neutral", "dry_ack"), ("neutral", "callback"),
                        ("sad", None), ("whatever", "nonexistent_mode")]:
            self.assertIsNone(tts.resolve_audio_tag(emo, cm), (emo, cm))


class ApplyTagsTest(unittest.TestCase):
    def test_prepends_and_pins_stability(self):
        text, vs = tts._apply_audio_tags("A bold plan.", "neutral", "smug_superiority", {"stability": 0.7})
        self.assertEqual(text, "[sarcastic] A bold plan.")
        self.assertEqual(vs["stability"], config.TTS_V3_TAG_STABILITY)

    def test_untagged_line_is_unchanged(self):
        text, vs = tts._apply_audio_tags("Just the facts.", "neutral", "dry_ack", {"stability": 0.7})
        self.assertEqual(text, "Just the facts.")
        self.assertEqual(vs["stability"], 0.7)   # not pinned when no tag

    def test_non_whitelisted_inline_tag_dropped(self):
        text, _ = tts._apply_audio_tags("[annoyed] Sure. [sarcastic] Fine.", "neutral", None, {})
        self.assertNotIn("[annoyed]", text)
        self.assertIn("[sarcastic]", text)

    def test_kill_switch_disables(self):
        with mock.patch.object(config, "TTS_V3_AUDIO_TAGS_ENABLED", False):
            text, vs = tts._apply_audio_tags("A bold plan.", "neutral", "smug_superiority", {"stability": 0.7})
        self.assertEqual(text, "A bold plan.")
        self.assertEqual(vs["stability"], 0.7)

    def test_inactive_on_non_v3_model(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_multilingual_v2"):
            text, _ = tts._apply_audio_tags("A bold plan.", "neutral", "smug_superiority", {})
        self.assertEqual(text, "A bold plan.")   # v2 would speak the brackets — never tag it


class StripTagsTest(unittest.TestCase):
    def test_strips_all_tags(self):
        self.assertEqual(
            tts.strip_audio_tags("[sarcastic] Two hundred credits. [laughs] Bold."),
            "Two hundred credits. Bold.",
        )

    def test_no_tags_unchanged(self):
        self.assertEqual(tts.strip_audio_tags("Just a normal line."), "Just a normal line.")


if __name__ == "__main__":
    unittest.main()
