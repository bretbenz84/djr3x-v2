"""Sound-event awareness (2026-08-08): local YAMNet classifier for non-speech hearing.

Covers the three layers:
  * audio/sound_events.py — family mapping, thresholds, per-family cooldown,
    priority ordering, fail-safe disable.
  * audio/scene.py — classifier events publish into world_state.audio_scene
    (last_sound_event + seq bump for reactable families, laughter corroboration,
    heuristic chain preserved as fallback).
  * intelligence/consciousness.py — the notable-family spoken reaction branch
    (flavored prompts, seq-aware re-fire, shared cooldown) alongside the
    untouched startle path.

Plus config-consistency pins: every configured AudioSet class name must exist in
the shipped class map, every reaction prompt must name a real family, and the
priority tuple must cover every family.
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from audio import sound_events

_MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models" / "yamnet"
_SR = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))


class _FakeSession:
    """Stands in for the ONNX session: returns a fixed [frames, n_classes] score
    matrix so detector logic is tested without model files."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=np.float32)

    def run(self, _outputs, _feeds):
        return [self._scores]


class _DetectorHarness(unittest.TestCase):
    """Injects a fake session + class index into the module and restores after."""

    CLASSES = {"Bark": 0, "Doorbell": 1, "Screaming": 2, "Shatter": 3, "Laughter": 4}

    def setUp(self):
        self._saved = (
            sound_events._session,
            sound_events._input_name,
            dict(sound_events._class_index),
            sound_events._load_failed,
        )
        sound_events._input_name = "waveform"
        sound_events._class_index = dict(self.CLASSES)
        sound_events._load_failed = False
        sound_events.reset_cooldowns()
        self._families = mock.patch.object(
            config, "SOUND_EVENT_FAMILY_CLASSES",
            {
                "dog_bark": ("Bark",),
                "doorbell": ("Doorbell",),
                "scream": ("Screaming",),
                "glass_break": ("Shatter", "Missing Class Name"),
                "laughter": ("Laughter",),
            },
        )
        self._families.start()
        self._thresholds = mock.patch.object(
            config, "SOUND_EVENT_FAMILY_THRESHOLDS", {"scream": 0.6}
        )
        self._thresholds.start()
        self._default = mock.patch.object(config, "SOUND_EVENT_DEFAULT_THRESHOLD", 0.4)
        self._default.start()
        self._priority = mock.patch.object(
            config, "SOUND_EVENT_PRIORITY",
            ("scream", "glass_break", "doorbell", "dog_bark", "laughter"),
        )
        self._priority.start()

    def tearDown(self):
        (
            sound_events._session,
            sound_events._input_name,
            _idx,
            sound_events._load_failed,
        ) = self._saved
        sound_events._class_index = _idx
        sound_events.reset_cooldowns()
        for p in (self._families, self._thresholds, self._default, self._priority):
            p.stop()

    def _set_scores(self, per_class: dict, frames: int = 2):
        row = [0.0] * len(self.CLASSES)
        for name, score in per_class.items():
            row[self.CLASSES[name]] = score
        # Event lives on ONE frame only — per-class max must find it.
        quiet = [0.0] * len(self.CLASSES)
        sound_events._session = _FakeSession([row] + [quiet] * (frames - 1))

    def _window(self, secs: float = 2.0):
        return np.zeros(int(_SR * secs), dtype=np.float32)


class FamilyDetectionTest(_DetectorHarness):
    def test_family_fires_above_threshold_with_top_class(self):
        self._set_scores({"Bark": 0.7})
        events = sound_events.classify_events(self._window(), now=100.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["family"], "dog_bark")
        self.assertEqual(events[0]["top_class"], "Bark")
        self.assertAlmostEqual(events[0]["score"], 0.7, places=3)

    def test_below_threshold_is_silent(self):
        self._set_scores({"Bark": 0.39})
        self.assertEqual(sound_events.classify_events(self._window(), now=100.0), [])

    def test_per_family_threshold_override_wins(self):
        # scream floor is 0.6 here: 0.5 (above default) must NOT fire.
        self._set_scores({"Screaming": 0.5})
        self.assertEqual(sound_events.classify_events(self._window(), now=100.0), [])
        self._set_scores({"Screaming": 0.65})
        events = sound_events.classify_events(self._window(), now=200.0)
        self.assertEqual([e["family"] for e in events], ["scream"])

    def test_cooldown_one_event_per_family_per_window(self):
        self._set_scores({"Doorbell": 0.9})
        with mock.patch.object(config, "SOUND_EVENT_FAMILY_COOLDOWN_SECS", 30.0):
            first = sound_events.classify_events(self._window(), now=100.0)
            during = sound_events.classify_events(self._window(), now=110.0)
            after = sound_events.classify_events(self._window(), now=131.0)
        self.assertEqual([e["family"] for e in first], ["doorbell"])
        self.assertEqual(during, [])
        self.assertEqual([e["family"] for e in after], ["doorbell"])

    def test_priority_orders_concurrent_events(self):
        self._set_scores({"Bark": 0.9, "Shatter": 0.9, "Doorbell": 0.9})
        events = sound_events.classify_events(self._window(), now=100.0)
        self.assertEqual(
            [e["family"] for e in events], ["glass_break", "doorbell", "dog_bark"]
        )

    def test_unknown_class_names_are_skipped_not_fatal(self):
        # glass_break lists "Missing Class Name"; only Shatter resolves.
        self._set_scores({"Shatter": 0.8})
        events = sound_events.classify_events(self._window(), now=100.0)
        self.assertEqual(events[0]["top_class"], "Shatter")

    def test_disabled_master_switch_returns_nothing(self):
        self._set_scores({"Bark": 0.9})
        with mock.patch.object(config, "SOUND_AWARENESS_ENABLED", False):
            self.assertEqual(sound_events.classify_events(self._window(), now=100.0), [])

    def test_short_window_is_ignored(self):
        self._set_scores({"Bark": 0.9})
        short = np.zeros(int(_SR * 0.5), dtype=np.float32)
        self.assertEqual(sound_events.classify_events(short, now=100.0), [])

    def test_inference_error_is_swallowed(self):
        broken = mock.Mock()
        broken.run.side_effect = RuntimeError("onnx exploded")
        sound_events._session = broken
        self.assertEqual(sound_events.classify_events(self._window(), now=100.0), [])


class ScenePublicationTest(unittest.TestCase):
    """audio/scene._analyze_cycle publishes classifier events into world_state."""

    def setUp(self):
        from world_state import world_state
        self._old_scene = dict(world_state.get("audio_scene"))

    def tearDown(self):
        from world_state import world_state
        world_state.update("audio_scene", self._old_scene)

    def _run_cycle(self, events):
        from audio import scene
        from world_state import world_state
        silence = np.zeros(int(_SR * 2.0), dtype=np.float32)
        with mock.patch.object(scene.sound_events, "classify_events", return_value=events):
            scene._analyze_cycle(silence)
        return world_state.get("audio_scene")

    def test_reactable_event_sets_last_sound_event_and_bumps_seq(self):
        before = int(self._old_scene.get("last_sound_event_seq") or 0)
        state = self._run_cycle([{"family": "doorbell", "score": 0.8, "top_class": "Doorbell"}])
        self.assertEqual(state["last_sound_event"], "doorbell")
        self.assertEqual(int(state["last_sound_event_seq"]), before + 1)
        self.assertEqual(state["sound_events"][0]["family"], "doorbell")

    def test_repeat_family_bumps_seq_again(self):
        s1 = self._run_cycle([{"family": "dog_bark", "score": 0.8, "top_class": "Bark"}])
        s2 = self._run_cycle([{"family": "dog_bark", "score": 0.7, "top_class": "Bark"}])
        self.assertEqual(
            int(s2["last_sound_event_seq"]), int(s1["last_sound_event_seq"]) + 1
        )

    def test_classifier_laughter_corroborates_heuristic_without_seq_bump(self):
        before = int(self._old_scene.get("last_sound_event_seq") or 0)
        state = self._run_cycle([{"family": "laughter", "score": 0.9, "top_class": "Laughter"}])
        self.assertTrue(state["laughter_detected"])
        self.assertEqual(state["last_sound_event"], "laughter")
        self.assertEqual(int(state["last_sound_event_seq"] or 0), before)

    def test_classifier_applause_corroborates_heuristic_without_seq_bump(self):
        before = int(self._old_scene.get("last_sound_event_seq") or 0)
        state = self._run_cycle([{"family": "applause", "score": 0.9, "top_class": "Clapping"}])
        self.assertTrue(state["applause_detected"])
        self.assertEqual(state["last_sound_event"], "applause")
        self.assertEqual(int(state["last_sound_event_seq"] or 0), before)

    def test_no_events_leaves_heuristic_chain_untouched(self):
        state = self._run_cycle([])
        self.assertEqual(
            state.get("last_sound_event"), self._old_scene.get("last_sound_event")
        )


class NotableReactionTest(unittest.TestCase):
    """The consciousness branch: flavored spoken reactions for classifier families."""

    def _profile(self):
        from awareness.situation import SituationProfile
        return SituationProfile(
            conversation_active=False,
            user_mid_sentence=False,
            rapid_exchange=False,
            child_present=False,
            apparent_departure=False,
            likely_still_present=False,
            social_mode="one_on_one",
            suppress_proactive=False,
            suppress_system_comments=False,
            force_family_safe=False,
            being_discussed=False,
            discussion_sentiment="neutral",
            interaction_busy=False,
        )

    def _speak_mock(self, prev_scene, curr_scene, **flags):
        from intelligence import consciousness
        old_snapshot = consciousness._last_snapshot
        old_notable = consciousness._last_notable_sound_reaction_at
        base = {"crowd": {"count": 1, "count_label": "alone"}, "animals": [], "time": {}}
        prev = dict(base, audio_scene=prev_scene)
        curr = dict(base, audio_scene=curr_scene)
        try:
            consciousness._last_snapshot = prev
            consciousness._last_notable_sound_reaction_at = flags.get("last_at", 0.0)
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak", return_value=True) as speak,
                mock.patch("sequences.animations.play_body_beat"),
                mock.patch("config.WORLD_SOUND_EVENT_REACTIONS_ENABLED", False),
                mock.patch("config.SOUND_AWARENESS_REACTIONS_ENABLED", flags.get("enabled", True)),
                mock.patch("config.SOUND_EVENT_REACTION_COOLDOWN_SECS", flags.get("cooldown", 0.0)),
            ):
                consciousness._step_proactive_reactions(curr, self._profile())
            return speak
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._last_notable_sound_reaction_at = old_notable

    def test_doorbell_fires_flavored_curious_reaction(self):
        speak = self._speak_mock(
            {}, {"last_sound_event": "doorbell", "last_sound_event_seq": 1}
        )
        speak.assert_called_once()
        prompt, emotion = speak.call_args.args[0], speak.call_args.args[1]
        self.assertIn("doorbell", prompt.lower())
        self.assertEqual(emotion, "curious")
        self.assertEqual(speak.call_args.kwargs["label"], "sound event: doorbell")

    def test_alarm_uses_concerned_emotion(self):
        speak = self._speak_mock(
            {}, {"last_sound_event": "alarm", "last_sound_event_seq": 1}
        )
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[1], "concerned")

    def test_same_family_refires_on_seq_bump_only(self):
        prev = {"last_sound_event": "dog_bark", "last_sound_event_seq": 3}
        same = self._speak_mock(prev, dict(prev))
        same.assert_not_called()
        bumped = self._speak_mock(
            prev, {"last_sound_event": "dog_bark", "last_sound_event_seq": 4}
        )
        bumped.assert_called_once()

    def test_notable_cooldown_suppresses(self):
        import time as _time
        speak = self._speak_mock(
            {}, {"last_sound_event": "doorbell", "last_sound_event_seq": 1},
            cooldown=3600.0, last_at=_time.monotonic(),
        )
        speak.assert_not_called()

    def test_kill_switch_suppresses(self):
        speak = self._speak_mock(
            {}, {"last_sound_event": "doorbell", "last_sound_event_seq": 1},
            enabled=False,
        )
        speak.assert_not_called()

    def test_glass_break_rides_startle_path(self):
        from intelligence import consciousness
        old_startle = consciousness._last_startle_sound_reaction_at
        try:
            consciousness._last_startle_sound_reaction_at = 0.0
            with mock.patch("config.STARTLE_SOUND_EVENT_REACTION_COOLDOWN_SECS", 0.0):
                speak = self._speak_mock(
                    {}, {"last_sound_event": "glass_break", "last_sound_event_seq": 1}
                )
        finally:
            consciousness._last_startle_sound_reaction_at = old_startle
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[1], "surprised")
        self.assertIn("startle sound", speak.call_args.kwargs["label"])


class ConfigConsistencyTest(unittest.TestCase):
    def test_reaction_prompts_name_real_families(self):
        families = set(config.SOUND_EVENT_FAMILY_CLASSES)
        for fam in config.SOUND_EVENT_REACTION_PROMPTS:
            self.assertIn(fam, families)

    def test_priority_covers_every_family(self):
        self.assertEqual(set(config.SOUND_EVENT_PRIORITY), set(config.SOUND_EVENT_FAMILY_CLASSES))

    def test_startle_families_are_not_also_notable_prompts(self):
        overlap = set(config.STARTLE_SOUND_EVENTS) & set(config.SOUND_EVENT_REACTION_PROMPTS)
        self.assertEqual(overlap, set())

    @unittest.skipUnless(
        (_MODEL_DIR / "yamnet_class_map.csv").exists(), "class map not downloaded"
    )
    def test_every_configured_class_exists_in_class_map(self):
        import csv
        with open(_MODEL_DIR / "yamnet_class_map.csv", newline="") as fh:
            names = {row["display_name"] for row in csv.DictReader(fh)}
        for family, class_names in config.SOUND_EVENT_FAMILY_CLASSES.items():
            for name in class_names:
                self.assertIn(name, names, f"{family}: {name!r} not an AudioSet class")


class RealModelSmokeTest(unittest.TestCase):
    @unittest.skipUnless((_MODEL_DIR / "yamnet.onnx").exists(), "model not downloaded")
    def test_silence_produces_no_events(self):
        saved = (sound_events._session, sound_events._load_failed)
        sound_events._session = None
        sound_events._load_failed = False
        sound_events.reset_cooldowns()
        try:
            self.assertTrue(sound_events.available())
            silence = np.zeros(int(_SR * 2.0), dtype=np.float32)
            self.assertEqual(sound_events.classify_events(silence, now=100.0), [])
        finally:
            sound_events._session, sound_events._load_failed = saved
            sound_events.reset_cooldowns()


if __name__ == "__main__":
    unittest.main()
