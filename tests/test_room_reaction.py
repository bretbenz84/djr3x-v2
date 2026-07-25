"""Land-the-laugh / take-a-bow + object-grounded curiosity.

_step_room_reaction reacts to the ROOM landing Rex's material (applause -> bow,
laughter -> dry follow-through), gated on a recent-Rex-utterance window + cooldown +
a low per-session cap. _visual_curiosity_objects_line feeds the local detector's
confirmed objects into the visual-curiosity prompt.
"""

import time
import unittest
from unittest import mock

from intelligence import consciousness as c


class _Profile:
    def __init__(self, user_mid_sentence=False):
        self.user_mid_sentence = user_mid_sentence


class StepRoomReactionTest(unittest.TestCase):
    def setUp(self):
        c._room_reacted["count"] = 0.0
        c._room_reacted["last_at"] = 0.0
        self.addCleanup(self._reset)

    def _reset(self):
        c._room_reacted["count"] = 0.0
        c._room_reacted["last_at"] = 0.0

    # since_spoke default sits comfortably past ROOM_REACTION_MIN_AFTER_REX_SECS —
    # below it, the reaction is (correctly) suppressed as Rex's own TTS tail.
    def _run(self, audio, *, can_speak=True, profile=None, since_spoke=3.0):
        captured = {}
        with mock.patch.object(c, "_can_proactive_speak", return_value=can_speak), \
             mock.patch("audio.speech_queue.seconds_since_last_speech", return_value=since_spoke), \
             mock.patch.object(
                 c, "_speak_async",
                 side_effect=lambda line, **k: captured.update(line=line, kw=k) or True), \
             mock.patch("sequences.animations.play_body_beat") as beat:
            c._step_room_reaction({"audio_scene": audio}, profile or _Profile())
            captured["beat"] = beat.call_args[0][0] if beat.called else None
        return captured

    def test_applause_takes_a_bow(self):
        out = self._run({"applause_detected": True})
        self.assertIn("line", out)
        self.assertEqual(out["kw"].get("purpose"), "room_reaction")
        self.assertTrue(out["kw"].get("reactive"))
        self.assertIs(out["kw"].get("governed"), False)
        self.assertEqual(out["beat"], "proud_dj_pose")
        self.assertEqual(c._room_reacted["count"], 1)

    def test_laughter_is_a_dry_line_with_no_beat(self):
        out = self._run({"laughter_detected": True})
        self.assertIn("line", out)
        self.assertIsNone(out["beat"])

    def test_applause_wins_over_laughter_in_the_same_cycle(self):
        with mock.patch.object(c.config, "ROOM_APPLAUSE_REACTION_LINES", ["BOW"]), \
             mock.patch.object(c.config, "ROOM_LAUGHTER_REACTION_LINES", ["dry"]):
            out = self._run({"applause_detected": True, "laughter_detected": True})
        self.assertEqual(out["line"], "BOW")

    def test_no_signal_does_nothing(self):
        self.assertNotIn("line", self._run({}))

    def test_ignores_ambient_when_rex_did_not_speak_recently(self):
        # Rex's last line finished long ago → a laugh now is ambient, not at him.
        self.assertNotIn("line", self._run({"applause_detected": True}, since_spoke=999.0))

    def test_ignores_his_own_tail_right_after_he_stops(self):
        # Field 2026-07-24 19:58:31: Rex took a bow at a silent, SEATED room. The first
        # analysis window after his TTS unmutes still holds his decaying tail + room
        # echo, and that read as applause. Real applause starts later than his reverb.
        self.assertNotIn("line", self._run({"applause_detected": True}, since_spoke=0.2))
        self.assertNotIn("line", self._run({"laughter_detected": True}, since_spoke=0.9))
        # Past the guard it still works.
        self.assertIn("line", self._run({"applause_detected": True}, since_spoke=2.0))

    def test_applause_lines_never_claim_what_the_person_is_doing(self):
        # Rex cannot see posture reliably; asserting it reads as a malfunction when he
        # is wrong ("No need to stand. ...Oh, you're already standing." to a seated
        # owner). Same rule as the persona's "never invent physical details".
        banned = ("standing", "stand up", "on your feet", "sitting", "seated",
                  "stretching")
        for line in (list(c.config.ROOM_APPLAUSE_REACTION_LINES)
                     + list(c.config.ROOM_LAUGHTER_REACTION_LINES)):
            low = line.lower()
            for word in banned:
                self.assertNotIn(word, low, f"{line!r} asserts a posture Rex can't see")

    def test_session_cap(self):
        c._room_reacted["count"] = float(c.config.ROOM_REACTION_SESSION_CAP)
        self.assertNotIn("line", self._run({"applause_detected": True}))

    def test_global_cooldown_dedups_a_burst(self):
        c._room_reacted["last_at"] = time.monotonic()  # just fired
        self.assertNotIn("line", self._run({"applause_detected": True}))

    def test_blocked_speech_gate_does_not_consume_the_cap(self):
        out = self._run({"applause_detected": True}, can_speak=False)
        self.assertNotIn("line", out)
        self.assertEqual(c._room_reacted["count"], 0.0)

    def test_user_mid_sentence_blocks(self):
        out = self._run({"applause_detected": True}, profile=_Profile(user_mid_sentence=True))
        self.assertNotIn("line", out)

    def test_disabled_flag(self):
        with mock.patch.object(c.config, "ROOM_REACTION_ENABLED", False):
            self.assertNotIn("line", self._run({"applause_detected": True}))


class VisualCuriosityObjectsLineTest(unittest.TestCase):
    def _objs(self, objs):
        with mock.patch.object(c.world_state, "get", return_value=objs):
            return c._visual_curiosity_objects_line()

    def test_disabled_returns_empty(self):
        with mock.patch.object(c.config, "VISUAL_CURIOSITY_USE_OBJECTS", False):
            self.assertEqual(self._objs([{"label": "chair", "confidence": 0.9}]), "")

    def test_empty_objects_returns_empty(self):
        self.assertEqual(self._objs([]), "")

    def test_formats_label_and_position_sorted_by_confidence(self):
        line = self._objs([
            {"label": "cup", "position": "foreground left", "confidence": 0.6},
            {"label": "guitar", "position": "center", "confidence": 0.95},
        ])
        self.assertIn("guitar (center)", line)
        self.assertIn("cup (foreground left)", line)
        self.assertLess(line.index("guitar"), line.index("cup"))  # confidence-desc order

    def test_low_confidence_is_filtered_out(self):
        with mock.patch.object(c.config, "VISUAL_CURIOSITY_OBJECTS_MIN_CONFIDENCE", 0.5):
            self.assertEqual(self._objs([{"label": "blur", "confidence": 0.2}]), "")

    def test_capped_to_max(self):
        objs = [{"label": f"obj{i}", "confidence": 0.9} for i in range(20)]
        with mock.patch.object(c.config, "VISUAL_CURIOSITY_OBJECTS_MAX", 3):
            line = self._objs(objs)
        self.assertEqual(line.count(","), 2)  # 3 items → 2 separators


class SecondsSinceLastSpeechTest(unittest.TestCase):
    def test_infinite_before_any_speech(self):
        from audio import speech_queue

        with mock.patch.object(speech_queue._queue, "_last_speech_end_at", 0.0), \
             mock.patch.object(speech_queue._queue, "is_speaking", return_value=False):
            self.assertEqual(speech_queue.seconds_since_last_speech(), float("inf"))

    def test_zero_while_speaking(self):
        from audio import speech_queue

        with mock.patch.object(speech_queue._queue, "is_speaking", return_value=True):
            self.assertEqual(speech_queue.seconds_since_last_speech(), 0.0)

    def test_elapsed_after_speech(self):
        from audio import speech_queue

        with mock.patch.object(speech_queue._queue, "is_speaking", return_value=False), \
             mock.patch.object(speech_queue._queue, "_last_speech_end_at", time.monotonic() - 3.0):
            self.assertGreaterEqual(speech_queue.seconds_since_last_speech(), 2.5)


if __name__ == "__main__":
    unittest.main()
