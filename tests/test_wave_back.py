"""
Wave back: when a visible person waves (pose pipeline classifies gesture="waving" onto
world_state.people), Rex returns the wave + one short warm line — debounced per person and
globally so a single wave fires one reaction, and gated like other proactive beats.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import consciousness as c


class _Profile:
    def __init__(self, suppress_proactive=False, user_mid_sentence=False):
        self.suppress_proactive = suppress_proactive
        self.user_mid_sentence = user_mid_sentence


def _waving(person_db_id=1, face_id="Bret", visible=True):
    return {
        "person_db_id": person_db_id, "face_id": face_id,
        "face_visible": visible, "gesture": "waving",
    }


class WaveBackHelpersTest(unittest.TestCase):
    def test_line_uses_name_when_known(self):
        with mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]):
            self.assertEqual(c._wave_back_line("Bret"), "Hey, Bret!")

    def test_line_falls_back_without_name(self):
        with mock.patch.object(c.config, "WAVE_BACK_LINES_NO_NAME", ["Hey there!"]):
            self.assertEqual(c._wave_back_line(""), "Hey there!")

    def test_person_key_prefers_db_id(self):
        self.assertEqual(c._wave_person_key({"person_db_id": 7}), "db:7")
        self.assertEqual(c._wave_person_key({"id": "person_2"}), "slot:person_2")
        self.assertEqual(c._wave_person_key({}), "unknown")


class StepWaveReactionTest(unittest.TestCase):
    def setUp(self):
        c._wave_reacted_keys.clear()
        c._wave_escalation.clear()
        c._last_wave_reaction_at = 0.0
        c._pending_wave_back = None

    def tearDown(self):
        c._wave_reacted_keys.clear()
        c._wave_escalation.clear()
        c._last_wave_reaction_at = 0.0
        c._pending_wave_back = None

    def _run(self, snapshot, *, profile=None, can_speak=True):
        captured = {}
        with mock.patch.object(c, "_can_proactive_speak", return_value=can_speak), \
             mock.patch.object(c, "_first_name", return_value="Bret"), \
             mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: captured.update(line=line, kw=k) or True), \
             mock.patch("sequences.animations.wave_back_gesture") as wave:
            c._step_wave_reaction(snapshot, profile or _Profile())
            captured["waved"] = wave.called
        return captured

    def test_waves_back_at_a_waver(self):
        out = self._run({"people": [_waving()]})
        self.assertTrue(out["waved"])
        self.assertIn("Bret", out["line"])
        self.assertEqual(out["kw"].get("purpose"), "wave_back")
        # reactive=True so the hello breaks through awaiting-reply / active-conversation
        # gates; governed=False so it bypasses the proactive-priority tournament (where a
        # priority-20 wave_back was always out-ranked + dropped). Both are required for a
        # wave during a conversation to actually be acknowledged.
        self.assertTrue(out["kw"].get("reactive"))
        self.assertIs(out["kw"].get("governed"), False)

    def test_neutral_gesture_does_nothing(self):
        p = _waving(); p["gesture"] = "neutral"
        out = self._run({"people": [p]})
        self.assertFalse(out["waved"])
        self.assertNotIn("line", out)

    def test_not_visible_does_nothing(self):
        out = self._run({"people": [_waving(visible=False)]})
        self.assertFalse(out["waved"])

    def test_disabled_flag(self):
        with mock.patch.object(c.config, "WAVE_BACK_ENABLED", False):
            out = self._run({"people": [_waving()]})
        self.assertFalse(out["waved"])

    def test_user_mid_sentence_blocks(self):
        # Don't wave back over someone who is mid-sentence; the wave stays latched for when
        # they pause (the next tick can still fire it).
        out = self._run({"people": [_waving()]}, profile=_Profile(user_mid_sentence=True))
        self.assertFalse(out["waved"])
        self.assertIsNotNone(c._pending_wave_back)  # latched, not lost

    def test_wave_latches_while_busy_then_fires_when_free(self):
        # A wave seen while Rex can't speak is latched and answered on a later tick.
        with mock.patch.object(c, "_first_name", return_value="Bret"), \
             mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: True), \
             mock.patch("sequences.animations.wave_back_gesture") as wave:
            with mock.patch.object(c, "_can_proactive_speak", return_value=False):
                c._step_wave_reaction({"people": [_waving()]}, _Profile())
            self.assertFalse(wave.called)                 # busy → didn't fire yet
            self.assertIsNotNone(c._pending_wave_back)    # but latched
            with mock.patch.object(c, "_can_proactive_speak", return_value=True):
                c._step_wave_reaction({"people": []}, _Profile())  # wave gone, Rex now free
            self.assertTrue(wave.called)                  # latched wave fired
            self.assertIsNone(c._pending_wave_back)       # latch cleared

    def test_give_space_gate_blocks(self):
        # _can_proactive_speak False (e.g. sober window / DJ / active game) → no wave-back.
        out = self._run({"people": [_waving()]}, can_speak=False)
        self.assertFalse(out["waved"])

    def test_per_person_cooldown_debounces(self):
        self.assertTrue(self._run({"people": [_waving()]})["waved"])  # fires
        # Same person still waving next tick, but within the per-person cooldown → no re-fire
        # (also reset the global gap so we isolate the per-person debounce).
        c._last_wave_reaction_at = 0.0
        self.assertFalse(self._run({"people": [_waving()]})["waved"])

    def test_failed_speech_does_not_burn_debounce(self):
        # If the speech can't be queued this tick (returns False), the per-person debounce
        # must NOT be marked — else a wave Rex couldn't voice locks out the next 25s. The
        # next tick must still be able to acknowledge the wave.
        calls = {"n": 0}

        def fake_speak(_line, **_k):
            calls["n"] += 1
            return calls["n"] >= 2  # 1st attempt fails to queue, 2nd succeeds

        with mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_first_name", return_value="Bret"), \
             mock.patch.object(c.config, "WAVE_BACK_LINES", ["Hey, {name}!"]), \
             mock.patch.object(c, "_speak_async", side_effect=fake_speak), \
             mock.patch("sequences.animations.wave_back_gesture"):
            c._step_wave_reaction({"people": [_waving()]}, _Profile())
            self.assertEqual(c._wave_reacted_keys, {})  # failed speech → debounce untouched
            c._last_wave_reaction_at = 0.0  # isolate per-person debounce from global gap
            c._step_wave_reaction({"people": [_waving()]}, _Profile())
            self.assertIn("db:1", c._wave_reacted_keys)  # retry succeeded → now marked
        self.assertEqual(calls["n"], 2)

    def test_unknown_waver_gets_no_name_line(self):
        captured = {}
        with mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda line, **k: captured.update(line=line)), \
             mock.patch.object(c.config, "WAVE_BACK_LINES_NO_NAME", ["Hey there!"]), \
             mock.patch("sequences.animations.wave_back_gesture"):
            c._step_wave_reaction({"people": [_waving(person_db_id=None, face_id=None)]}, _Profile())
        self.assertEqual(captured.get("line"), "Hey there!")


class WaveEscalationTest(unittest.TestCase):
    """Repeat-wave comedy bit: greet → silent wave → joke → give-up → ignore."""

    def setUp(self):
        c._wave_reacted_keys.clear()
        c._wave_escalation.clear()
        c._last_wave_reaction_at = 0.0
        c._pending_wave_back = None
        self.addCleanup(c._wave_escalation.clear)
        self.addCleanup(c._wave_reacted_keys.clear)

    def test_response_plan_per_level(self):
        plan = c._wave_response_plan
        # (line, should_speak, should_gesture)
        l, s, g = plan(1, "Bret"); self.assertTrue(s and g and l)        # greet + wave
        l, s, g = plan(2, "Bret"); self.assertEqual((bool(l), s, g), (False, False, True))   # silent wave
        l, s, g = plan(3, "Bret"); self.assertTrue(s and g and l)        # joke + wave
        l, s, g = plan(4, "Bret"); self.assertEqual((bool(l), s, g), (True, True, False))    # give-up, no wave
        l, s, g = plan(5, "Bret"); self.assertEqual((bool(l), s, g), (False, False, False))  # ignore

    def _fire(self, prev_level):
        """Drive one wave with a given pre-existing escalation level; return ('speak'|'gesture')."""
        import time
        events = []
        c._wave_reacted_keys.clear()       # so phase A latches a fresh wave
        c._last_wave_reaction_at = 0.0
        c._pending_wave_back = None
        if prev_level > 0:
            c._wave_escalation["db:1"] = (time.monotonic(), prev_level)
        else:
            c._wave_escalation.pop("db:1", None)
        with mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_first_name", return_value="Bret"), \
             mock.patch.object(c, "_speak_async",
                               side_effect=lambda *a, **k: events.append("speak") or True), \
             mock.patch("sequences.animations.wave_back_gesture",
                        side_effect=lambda *a, **k: events.append("gesture") or True):
            c._step_wave_reaction({"people": [_waving()]}, _Profile())
        return events

    def test_escalation_sequence(self):
        self.assertEqual(self._fire(0), ["speak", "gesture"])   # 1st: greet + wave
        self.assertEqual(self._fire(1), ["gesture"])            # 2nd: silent wave
        self.assertEqual(self._fire(2), ["speak", "gesture"])   # 3rd: joke + wave
        self.assertEqual(self._fire(3), ["speak"])              # 4th: give-up, no wave
        self.assertEqual(self._fire(4), [])                     # 5th: ignored
        self.assertEqual(self._fire(7), [])                     # still ignored


class WaveSpeedMirrorTest(unittest.TestCase):
    """Map the user's measured wave speed to Rex's wave-back half-period."""

    def test_mapping_is_monotonic_and_clamped(self):
        f = c._mirrored_half_period
        slow_hp = c.config.WAVE_BACK_WRIST_HALF_PERIOD_SLOW_SECS
        fast_hp = c.config.WAVE_BACK_WRIST_HALF_PERIOD_FAST_SECS
        self.assertIsNone(f(None))                       # no measurement → default
        self.assertAlmostEqual(f(0.0), slow_hp, places=2)    # very slow → slow (clamped)
        self.assertAlmostEqual(f(99.0), fast_hp, places=2)   # very fast → fast (clamped)
        self.assertLess(f(1.0), f(0.3))                  # faster wave → shorter half-period

    def test_disabled_returns_none(self):
        with mock.patch.object(c.config, "WAVE_SPEED_MIRROR_ENABLED", False):
            self.assertIsNone(c._mirrored_half_period(1.0))


if __name__ == "__main__":
    unittest.main()
