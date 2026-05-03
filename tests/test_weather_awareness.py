import unittest
from unittest import mock


class WeatherAwarenessTest(unittest.TestCase):
    def test_world_state_has_weather_slot(self):
        from world_state import world_state

        weather = world_state.get("weather")

        self.assertEqual(weather["condition"], "unknown")
        self.assertFalse(weather["available"])
        self.assertIn("mood_bias", weather)

    def test_refresh_weather_writes_weather_mood_to_world_state(self):
        from awareness import chronoception

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "current_condition": [
                        {
                            "temp_F": "58",
                            "FeelsLikeF": "56",
                            "humidity": "78",
                            "windspeedMiles": "11",
                            "weatherCode": "296",
                            "weatherDesc": [{"value": "Light rain"}],
                        }
                    ]
                }

        old_weather = chronoception.world_state.get("weather")
        with chronoception._weather_lock:
            old_cache = chronoception._weather_cache
            old_fetched_at = chronoception._weather_fetched_at
            chronoception._weather_cache = None
            chronoception._weather_fetched_at = 0.0

        try:
            with (
                mock.patch.object(chronoception, "_REQUESTS_OK", True),
                mock.patch.object(chronoception, "_requests", create=True) as requests_mod,
                mock.patch.object(chronoception.config, "WEATHER_LOCATION", "Sacramento, California"),
            ):
                requests_mod.get.return_value = FakeResponse()
                weather = chronoception.refresh_weather(force=True)

            self.assertTrue(weather["available"])
            self.assertEqual(weather["condition"], "rain")
            self.assertEqual(weather["temp_f"], 58)
            self.assertEqual(weather["feels_like_f"], 56)
            self.assertEqual(weather["mood_bias"], "rainy")
            self.assertEqual(
                chronoception.world_state.get("weather")["description"],
                "Light rain",
            )
        finally:
            chronoception.world_state.update("weather", old_weather)
            with chronoception._weather_lock:
                chronoception._weather_cache = old_cache
                chronoception._weather_fetched_at = old_fetched_at

    def test_llm_world_summary_includes_weather_and_mood_rule(self):
        from intelligence import llm

        weather = {
            "location": "Sacramento, California",
            "condition": "rain",
            "temp_f": 58,
            "feels_like_f": 56,
            "description": "Light rain",
            "available": True,
            "mood_bias": "rainy",
            "tone_hint": "rainy weather can make Rex a little drier and more atmospheric.",
        }
        ws = {
            "environment": {},
            "crowd": {"count": 0, "count_label": "alone"},
            "people": [],
            "audio_scene": {},
            "self_state": {},
            "time": {},
            "weather": weather,
            "animals": [],
        }

        summary = llm._summarize_world_state(ws)
        rule = llm._weather_tone_rule(weather)

        self.assertIn("Weather in Sacramento, California: 58°F, Light rain", summary)
        self.assertIn("Weather mood: rainy", summary)
        self.assertIn("rainy weather", rule)
        self.assertIn("Mood bias: rainy", rule)

    def test_weather_change_uses_named_governor_purpose(self):
        from intelligence import consciousness

        old_snapshot = consciousness._last_snapshot
        old_last = consciousness._last_weather_reaction_at
        old_ack = set(consciousness._acknowledged_weather_signatures)
        try:
            consciousness._last_snapshot = {
                "crowd": {"count": 0, "count_label": "alone"},
                "people": [],
                "animals": [],
                "audio_scene": {},
                "time": {},
                "weather": {
                    "available": True,
                    "condition": "clear",
                    "temp_f": 70,
                    "description": "Clear",
                    "location": "Sacramento, California",
                },
            }
            consciousness._last_weather_reaction_at = 0.0
            consciousness._acknowledged_weather_signatures.clear()
            snapshot = {
                "crowd": {"count": 0, "count_label": "alone"},
                "people": [],
                "animals": [],
                "audio_scene": {},
                "time": {},
                "weather": {
                    "available": True,
                    "condition": "rain",
                    "temp_f": 58,
                    "description": "Light rain",
                    "location": "Sacramento, California",
                },
            }

            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak", return_value=True) as generate,
            ):
                profile = type("Profile", (), {"suppress_proactive": False, "rapid_exchange": False})()
                consciousness._step_proactive_reactions(snapshot, profile)

            generate.assert_called_once()
            self.assertEqual(generate.call_args.kwargs["purpose"], "weather.proactive_comment")
            self.assertEqual(generate.call_args.kwargs["label"], "weather proactive comment")
            self.assertEqual(
                generate.call_args.kwargs["metadata"]["weather_signature"],
                "rain:mild",
            )
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._last_weather_reaction_at = old_last
            consciousness._acknowledged_weather_signatures.clear()
            consciousness._acknowledged_weather_signatures.update(old_ack)


if __name__ == "__main__":
    unittest.main()
