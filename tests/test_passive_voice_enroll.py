"""Passive voiceprint growth + the 2026-08-26 impersonation-capture fixes.

Owner spec: new people never read lines — when the solo visible person speaks
and the voice matches nobody's prints well, the turn silently becomes their
voiceprint. And the clone-ref capture must record the RECITATION, not whatever
padded utterance happened to be in flight ("impersonate me" was literally
Bret's stored clone ref).
"""

import unittest
from unittest import mock

import numpy as np

import config
from intelligence import interaction as I


def _voiced(secs: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * secs), dtype=np.float32) / sr
    return (0.1 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def _padded(voiced_secs: float, total_secs: float, sr: int = 16000) -> np.ndarray:
    body = _voiced(voiced_secs, sr)
    pad = np.zeros(int(sr * max(0.0, total_secs - voiced_secs)), dtype=np.float32)
    return np.concatenate([pad[: len(pad) // 2], body, pad[len(pad) // 2:]])


class VoicedDurationTest(unittest.TestCase):
    def test_padding_does_not_count(self):
        # The "impersonate me" shape: ~1.5s of speech in a 5.2s buffer.
        secs = I._voiced_duration_secs(_padded(1.5, 5.2))
        self.assertLess(secs, 2.0)
        self.assertGreater(secs, 1.0)

    def test_full_speech_counts(self):
        secs = I._voiced_duration_secs(_voiced(4.0))
        self.assertGreater(secs, 3.5)

    def test_silence_is_zero(self):
        self.assertEqual(I._voiced_duration_secs(np.zeros(80000, dtype=np.float32)), 0.0)


class PassiveEnrollTest(unittest.TestCase):
    def setUp(self):
        I._passive_enroll_last_at.clear()
        I._passive_enroll_session_counts.clear()
        I._recent_attributed_speaker_times.clear()
        I._pending_voice_sample_capture = None
        I._pending_impersonation_capture = None
        self.enroll = mock.patch.object(I, "_safe_enroll_voice", return_value=True)
        self.enrolled = self.enroll.start()
        self.addCleanup(self.enroll.stop)

    def _call(self, *, text="tell me about the band in the cantina",
              audio=None, person_id=7, raw_best_id=1, score=0.65,
              solo=(7, "PJ Thomas"), unknown_visible=False,
              other_visible=False, prints=0):
        with mock.patch.object(I, "_single_visible_person_identity", return_value=solo), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=unknown_visible), \
             mock.patch.object(I, "_other_known_visible_recently", return_value=other_visible), \
             mock.patch.object(I.speaker_id, "comparable_print_count", return_value=prints):
            I._maybe_passive_voice_enroll(
                text, audio if audio is not None else _voiced(3.0),
                person_id, raw_best_id, score,
            )

    def test_voiceless_solo_person_enrolls_despite_cross_match(self):
        # PJ's speech lands 0.55-0.80 on Bret's centroid — expected for a
        # voiceless twin; the solo face is the identity anchor.
        self._call(score=0.70, prints=0)
        self.enrolled.assert_called_once()
        self.assertEqual(self.enrolled.call_args.args[0], 7)

    def test_confident_foreign_match_blocks_even_voiceless(self):
        self._call(score=0.80, prints=0)   # >= 0.75 confident bar
        self.enrolled.assert_not_called()

    def test_thin_prints_grow_when_the_voice_matches_nobody_well(self):
        self._call(score=0.45, prints=1)
        self.enrolled.assert_called_once()

    def test_thin_prints_blocked_by_moderate_foreign_match(self):
        self._call(score=0.65, prints=1)   # >= LOW_BAR once prints exist
        self.enrolled.assert_not_called()

    def test_redundant_self_match_adds_nothing(self):
        self._call(raw_best_id=7, score=0.85, prints=2)
        self.enrolled.assert_not_called()

    def test_self_match_in_the_growth_band_enrolls(self):
        self._call(raw_best_id=7, score=0.62, prints=2)
        self.enrolled.assert_called_once()

    def test_someone_else_heard_recently_blocks(self):
        # Bret spoke 10s ago — classic off-camera-speaker hazard (game night).
        I._recent_attributed_speaker_times[1] = I.time.monotonic() - 10.0
        self._call(score=0.55)
        self.enrolled.assert_not_called()

    def test_other_known_face_recently_visible_blocks(self):
        self._call(other_visible=True)
        self.enrolled.assert_not_called()

    def test_unknown_face_visible_blocks(self):
        self._call(unknown_visible=True)
        self.enrolled.assert_not_called()

    def test_nobody_solo_visible_blocks(self):
        self._call(solo=(None, None))
        self.enrolled.assert_not_called()

    def test_short_turns_never_enroll(self):
        self._call(audio=_voiced(1.0))
        self.enrolled.assert_not_called()
        self._call(text="okay sure")
        self.enrolled.assert_not_called()

    def test_padded_short_speech_never_enrolls(self):
        self._call(audio=_padded(1.0, 6.0))
        self.enrolled.assert_not_called()

    def test_print_target_stops_growth(self):
        self._call(prints=4)
        self.enrolled.assert_not_called()

    def test_session_cap_and_spacing(self):
        self._call(score=0.40)
        self.enrolled.assert_called_once()
        # Immediately again: blocked by spacing.
        self._call(score=0.40)
        self.enrolled.assert_called_once()
        # Past spacing but at the session cap: blocked.
        I._passive_enroll_last_at[7] = I.time.monotonic() - 10_000.0
        I._passive_enroll_session_counts[7] = 3
        self._call(score=0.40)
        self.enrolled.assert_called_once()

    def test_pending_capture_flow_owns_the_audio(self):
        I._pending_voice_sample_capture = {"person_id": 7, "asked_at": 1.0}
        self._call()
        self.enrolled.assert_not_called()

    def test_kill_switch(self):
        with mock.patch.object(config, "PASSIVE_VOICE_ENROLL_ENABLED", False, create=True):
            self._call()
        self.enrolled.assert_not_called()


class ImpersonateMeRetargetTest(unittest.TestCase):
    """"Impersonate me" must follow the visible FACE, not the voice guess —
    PJ asks, the voice cross-match says Bret, Bret's ref used to perform."""

    def _run(self, *, target="me", person_id=1, face=(7, "PJ Thomas"),
             unknown_visible=False):
        from features import impersonation
        from intelligence import action_router
        decision = action_router.ActionDecision(
            action="performance.impersonate", confidence=1.0,
            args={"target": target}, reason="test",
        )
        seen = {}

        def fake_resolve(tgt, pid, name):
            seen["pid"], seen["name"] = pid, name
            return impersonation.Resolution("refuse", line="nope")

        with mock.patch.object(impersonation, "resolve_target", side_effect=fake_resolve), \
             mock.patch.object(I, "_single_visible_person_identity", return_value=face), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=unknown_visible), \
             mock.patch.object(I, "_speak_blocking"):
            I._handle_router_impersonation(decision, "impersonate me", person_id, "Bret Benziger", target)
        return seen

    def test_me_prefers_the_solo_visible_face(self):
        seen = self._run()
        self.assertEqual(seen["pid"], 7)
        self.assertEqual(seen["name"], "PJ Thomas")

    def test_named_targets_are_untouched(self):
        seen = self._run(target="Obama")
        self.assertEqual(seen["pid"], 1)

    def test_no_retarget_when_an_unknown_face_is_also_visible(self):
        seen = self._run(unknown_visible=True)
        self.assertEqual(seen["pid"], 1)

    def test_no_retarget_when_the_face_agrees(self):
        seen = self._run(person_id=7)
        self.assertEqual(seen["pid"], 7)


class CaptureRecitationGuardsTest(unittest.TestCase):
    """The clone-ref capture slot must record the recitation, nothing else."""

    def setUp(self):
        I._pending_impersonation_capture = None

    def tearDown(self):
        I._pending_impersonation_capture = None

    def _slot(self, **over):
        import time
        ctx = {
            "person_id": 7, "name": "PJ", "is_self": False,
            "expected_text": "Mary had a little lamb, its fleece was white as snow.",
            "asked_at": time.monotonic(),
        }
        ctx.update(over)
        I._pending_impersonation_capture = ctx
        return ctx

    def test_repeated_request_is_never_the_take(self):
        # THE bug: "impersonate me" said again became Bret's stored clone ref.
        self._slot()
        r = I._handle_impersonation_capture(
            "do an impersonation of me", _voiced(8.0), 7, 7, 0.9
        )
        line, spoken = r
        self.assertFalse(spoken)
        self.assertIn("not the request", line)
        self.assertIsNotNone(I._pending_impersonation_capture)

    def test_padded_buffer_does_not_pass_the_length_check(self):
        # 1.5s of speech riding in an 8s buffer used to sail past the 4s check.
        self._slot()
        r = I._handle_impersonation_capture(
            "Mary had a little lamb its fleece was white as snow",
            _padded(1.5, 8.0), 7, 7, 0.9,
        )
        line, spoken = r
        self.assertFalse(spoken)
        self.assertIn("blink", line)
        self.assertIsNotNone(I._pending_impersonation_capture)

    def test_off_script_take_gets_one_nudge_back_to_the_line(self):
        from features import impersonation
        self._slot()
        with mock.patch.object(I, "_known_person_visible_recently", return_value=False), \
             mock.patch.object(impersonation, "save_person_capture") as save:
            r = I._handle_impersonation_capture(
                "so anyway I went to the store yesterday and bought some milk",
                _voiced(8.0), 7, 7, 0.9,
            )
        line, spoken = r
        self.assertFalse(spoken)
        self.assertIn("the line itself", line)
        save.assert_not_called()
        # Second off-script (but substantial) take is accepted — never loop.
        ctx = I._pending_impersonation_capture
        self.assertTrue(ctx.get("recite_retry_done"))


class RefTrimTest(unittest.TestCase):
    def test_trim_strips_pads_and_keeps_speech(self):
        from features import impersonation
        arr = _padded(3.0, 9.0)
        trimmed = impersonation._trim_silence(arr, 16000)
        self.assertLess(len(trimmed), len(arr) * 0.5)
        self.assertGreater(len(trimmed), 16000 * 2.9)

    def test_all_silence_returned_unchanged(self):
        from features import impersonation
        arr = np.zeros(16000, dtype=np.float32)
        self.assertEqual(len(impersonation._trim_silence(arr, 16000)), len(arr))


if __name__ == "__main__":
    unittest.main()
