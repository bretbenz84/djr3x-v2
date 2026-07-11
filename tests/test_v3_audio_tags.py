"""Eleven v3 audio-tag injection (audio/tts.py) — deterministic affect -> tag, transcript-clean.

Tags shape delivery at synthesis but must NEVER reach the transcript/log. The mapping is:
comedy_mode (sarcasm/mischief) wins, else the reply emotion, else nothing (neutral/sincere).
On eleven_v3 stability is pinned globally to one Natural preset (0.5) so Rex's voice is
consistent line to line; Natural still lets tags land (only HIGH/Robust stability mutes them).
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
    def test_prepends_tag_and_leaves_stability_alone(self):
        # Stability is owned by _pin_v3_stability now, not the tag layer.
        text, vs = tts._apply_audio_tags("A bold plan.", "neutral", "smug_superiority", {"stability": 0.7})
        self.assertEqual(text, "[sarcastic] A bold plan.")
        self.assertEqual(vs["stability"], 0.7)

    def test_untagged_line_is_unchanged(self):
        text, vs = tts._apply_audio_tags("Just the facts.", "neutral", "dry_ack", {"stability": 0.7})
        self.assertEqual(text, "Just the facts.")
        self.assertEqual(vs["stability"], 0.7)

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


class InlineTagSanitizeTest(unittest.TestCase):
    """Inline (mid-sentence) tags: whitelisted survive on v3, everything is stripped when
    tags can't land (non-v3 model / kill switch), and suppress_leading only skips the
    affect-mapped prepend — never the sanitize."""

    def test_suppress_leading_keeps_inline_tag_but_never_prepends(self):
        # 2nd+ chunk of a streamed reply: a mapped stance must NOT prepend, but an
        # authored/LLM inline tag mid-chunk still lands.
        text, _ = tts._apply_audio_tags(
            "Fine. [excited] Next time we triumph!", "neutral", "smug_superiority", {},
            suppress_leading=True,
        )
        self.assertEqual(text, "Fine. [excited] Next time we triumph!")
        # And with no inline tag at all, suppression yields the bare text.
        text, _ = tts._apply_audio_tags(
            "Fine. Next time.", "neutral", "smug_superiority", {}, suppress_leading=True,
        )
        self.assertEqual(text, "Fine. Next time.")

    def test_kill_switch_strips_inline_tags(self):
        with mock.patch.object(config, "TTS_V3_AUDIO_TAGS_ENABLED", False):
            text, _ = tts._apply_audio_tags("[excited] Onward to glory!", "neutral", None, {})
        self.assertEqual(text, "Onward to glory!")

    def test_non_v3_model_strips_inline_tags(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_multilingual_v2"):
            text, _ = tts._apply_audio_tags(
                "Got it. [excited] Better luck next time!", "neutral", None, {}
            )
        self.assertEqual(text, "Got it. Better luck next time!")

    def test_inline_tag_cap_keeps_earliest(self):
        with mock.patch.object(config, "TTS_V3_INLINE_TAG_CAP", 2):
            text, _ = tts._apply_audio_tags(
                "[laughs] One. [sighs] Two. [whispers] Three.", "neutral", None, {}
            )
        self.assertEqual(text, "[laughs] One. [sighs] Two. Three.")

    def test_inline_tag_suppresses_the_leading_prepend(self):
        # An inline tag means the line already carries its delivery — no mapped prepend.
        text, _ = tts._apply_audio_tags(
            "Sure. [excited] We ride!", "neutral", "smug_superiority", {}
        )
        self.assertEqual(text, "Sure. [excited] We ride!")


class LlmInlineTagRuleTest(unittest.TestCase):
    """The lean-brain prompt rule exists only when tags can actually land, and offers
    only whitelisted tags."""

    def test_rule_active_and_palette_whitelisted(self):
        with mock.patch.object(config, "TTS_V3_LLM_INLINE_TAGS_ENABLED", True):
            rule = tts.llm_inline_tag_rule()
        self.assertIn("[excited]", rule)
        self.assertIn("[sighs]", rule)
        self.assertNotIn("[snorts]", rule)   # vocalization-only tags are not offered

    def test_rule_empty_when_flag_off(self):
        with mock.patch.object(config, "TTS_V3_LLM_INLINE_TAGS_ENABLED", False):
            self.assertEqual(tts.llm_inline_tag_rule(), "")

    def test_rule_empty_on_non_v3_or_kill_switch(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_multilingual_v2"):
            self.assertEqual(tts.llm_inline_tag_rule(), "")
        with mock.patch.object(config, "TTS_V3_AUDIO_TAGS_ENABLED", False):
            self.assertEqual(tts.llm_inline_tag_rule(), "")


class StabilityPinTest(unittest.TestCase):
    """v3 pins stability to one preset so Rex doesn't sound like a different voice each sentence."""

    def test_v3_forces_fixed_stability_over_emotion_value(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_v3"), \
             mock.patch.object(config, "TTS_V3_STABILITY", 0.5):
            # An excited line would otherwise carry stability 0.30 from the per-style table.
            vs = tts._resolve_voice_settings("excited", None)
            self.assertEqual(vs["stability"], 0.5)

    def test_v3_forces_fixed_stability_on_explicit_override(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_v3"), \
             mock.patch.object(config, "TTS_V3_STABILITY", 0.5):
            vs = tts._resolve_voice_settings("neutral", {"stability": 0.66, "style": 0.2})
            self.assertEqual(vs["stability"], 0.5)
            self.assertEqual(vs["style"], 0.2)   # only stability is overridden

    def test_pin_is_noop_on_non_v3_model(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_multilingual_v2"), \
             mock.patch.object(config, "TTS_V3_STABILITY", 0.5):
            self.assertEqual(tts._pin_v3_stability({"stability": 0.3})["stability"], 0.3)

    def test_pin_disabled_when_stability_none(self):
        with mock.patch.object(config, "TTS_MODEL_ID", "eleven_v3"), \
             mock.patch.object(config, "TTS_V3_STABILITY", None):
            self.assertEqual(tts._pin_v3_stability({"stability": 0.3})["stability"], 0.3)


class _ResetElClientMixin:
    """Tests that mock elevenlabs.ElevenLabs must clear the module-level client cache
    (tts._el_client) before AND after, or whichever fake ran first sticks for the rest
    of the module and later tests capture nothing (observed: 2 order-dependent fails)."""

    def setUp(self):
        tts._el_client = None

    def tearDown(self):
        tts._el_client = None


class SeedTest(_ResetElClientMixin, unittest.TestCase):
    """A fixed v3 seed keeps the voice consistent across separate per-sentence API calls."""

    def test_seed_only_on_v3(self):
        with mock.patch.object(config, "TTS_V3_SEED", 42):
            self.assertEqual(tts._v3_seed("eleven_v3"), 42)
            self.assertIsNone(tts._v3_seed("eleven_multilingual_v2"))

    def test_seed_none_disables(self):
        with mock.patch.object(config, "TTS_V3_SEED", None):
            self.assertIsNone(tts._v3_seed("eleven_v3"))

    def test_seed_folded_into_cache_key(self):
        with mock.patch.object(config, "TTS_V3_SEED", 42):
            a = tts._cache_path("hi", "vid", "eleven_v3", {"stability": 0.5})
        with mock.patch.object(config, "TTS_V3_SEED", 99):
            b = tts._cache_path("hi", "vid", "eleven_v3", {"stability": 0.5})
        self.assertNotEqual(a, b)   # different seed -> different cache entry

    def test_seed_sent_to_api_for_v3(self):
        captured = {}

        class _FakeTTS:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return [b"x"]

        class _FakeClient:
            def __init__(self, *a, **k):
                self.text_to_speech = _FakeTTS()

        with mock.patch.object(config, "TTS_V3_SEED", 42), \
             mock.patch("elevenlabs.ElevenLabs", _FakeClient):
            tts._fetch_from_api("hello", "vid", "eleven_v3", {"stability": 0.5})
        self.assertEqual(captured.get("seed"), 42)

    def test_no_seed_for_non_v3(self):
        captured = {}

        class _FakeTTS:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return [b"x"]

        class _FakeClient:
            def __init__(self, *a, **k):
                self.text_to_speech = _FakeTTS()

        with mock.patch.object(config, "TTS_V3_SEED", 42), \
             mock.patch("elevenlabs.ElevenLabs", _FakeClient):
            tts._fetch_from_api("hello", "vid", "eleven_multilingual_v2", {"stability": 0.5})
        self.assertNotIn("seed", captured)


class StitchTest(_ResetElClientMixin, unittest.TestCase):
    """Request stitching (previous_text). eleven_v3 REJECTS previous_text (400 unsupported_model),
    so it must NEVER be sent on v3; stitching stays available for models that support it (v2/turbo)."""

    def test_v3_never_stitches(self):
        # The live API returns 400 for previous_text on eleven_v3 — the helper must return "".
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True), \
             mock.patch.object(config, "TTS_V3_STITCH_MAX_CHARS", 400):
            self.assertEqual(tts._stitch_previous_text("Prior line.", "eleven_v3"), "")

    def test_supported_model_stitches_when_enabled(self):
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True), \
             mock.patch.object(config, "TTS_V3_STITCH_MAX_CHARS", 400):
            self.assertEqual(tts._stitch_previous_text("Prior line.", "eleven_multilingual_v2"), "Prior line.")
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", False):
            self.assertEqual(tts._stitch_previous_text("Prior line.", "eleven_multilingual_v2"), "")

    def test_empty_previous_text_is_no_stitch(self):
        self.assertEqual(tts._stitch_previous_text("", "eleven_multilingual_v2"), "")
        self.assertEqual(tts._stitch_previous_text(None, "eleven_multilingual_v2"), "")

    def test_capped_to_last_chars(self):
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True), \
             mock.patch.object(config, "TTS_V3_STITCH_MAX_CHARS", 10):
            self.assertEqual(tts._stitch_previous_text("abcdefghijklmnop", "eleven_multilingual_v2"), "ghijklmnop")

    def test_v3_previous_text_never_folded_into_cache_key(self):
        # On v3 the stitch is inert, so previous_text must NOT change the key (else phantom misses).
        a = tts._cache_path("hi", "vid", "eleven_v3", {"stability": 0.5}, previous_text="")
        b = tts._cache_path("hi", "vid", "eleven_v3", {"stability": 0.5}, previous_text="Before.")
        self.assertEqual(a, b)

    def test_supported_model_previous_text_folded_into_cache_key(self):
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True):
            first = tts._cache_path("hi", "vid", "eleven_multilingual_v2", {"stability": 0.5}, previous_text="")
            stitched = tts._cache_path("hi", "vid", "eleven_multilingual_v2", {"stability": 0.5}, previous_text="Before.")
        self.assertNotEqual(first, stitched)

    def test_v3_never_sends_previous_text_to_api(self):
        captured = {}

        class _FakeTTS:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return [b"x"]

        class _FakeClient:
            def __init__(self, *a, **k):
                self.text_to_speech = _FakeTTS()

        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True), \
             mock.patch("elevenlabs.ElevenLabs", _FakeClient):
            tts._fetch_from_api("second sentence", "vid", "eleven_v3", {"stability": 0.5},
                                previous_text="First sentence.")
        self.assertNotIn("previous_text", captured)   # would 400 on the live v3 API

    def test_supported_model_sends_previous_text_to_api(self):
        captured = {}

        class _FakeTTS:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return [b"x"]

        class _FakeClient:
            def __init__(self, *a, **k):
                self.text_to_speech = _FakeTTS()

        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True), \
             mock.patch("elevenlabs.ElevenLabs", _FakeClient):
            tts._fetch_from_api("second sentence", "vid", "eleven_multilingual_v2", {"stability": 0.5},
                                previous_text="First sentence.")
        self.assertEqual(captured.get("previous_text"), "First sentence.")


class StripTagsTest(unittest.TestCase):
    def test_strips_all_tags(self):
        self.assertEqual(
            tts.strip_audio_tags("[sarcastic] Two hundred credits. [laughs] Bold."),
            "Two hundred credits. Bold.",
        )

    def test_no_tags_unchanged(self):
        self.assertEqual(tts.strip_audio_tags("Just a normal line."), "Just a normal line.")

    def test_strips_mid_sentence_tag(self):
        self.assertEqual(
            tts.strip_audio_tags("Got it. [excited] I'm sure we'll have better luck next time!"),
            "Got it. I'm sure we'll have better luck next time!",
        )

    def test_tts_reexports_shared_helper(self):
        # One canonical implementation: audio.tts and utils.conv_log must share it.
        from utils.audio_tags import strip_audio_tags as shared
        self.assertIs(tts.strip_audio_tags, shared)

    def test_conv_log_seam_strips(self):
        from utils import conv_log
        self.assertEqual(
            conv_log._strip_audio_tags("Sure. [whispers] Between us, the toaster lies."),
            "Sure. Between us, the toaster lies.",
        )


if __name__ == "__main__":
    unittest.main()
