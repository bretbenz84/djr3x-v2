"""
Field 2026-08-06 (session 13-20-57): Rex played a triumphant chirp two seconds after
finishing a line, the chirp's audio was captured and transcribed as "Naturally." — the
first word of the line he had just spoken — and he answered his own echo as a stranger:

    13:21:41 REX   Naturally. The outage got a vote of no confidence, and I'm enjoying…
    13:21:50 [sfx] ▶ Droid_Proudtriumphant (proud)
    13:21:53 HEARD unknown_voice_1: Naturally.
    13:21:54 REX   Naturally, because mystery voices always have impeccable timing—who are you?

Two independent holes, covered here:

1. Effects fire in the window where Rex is WAITING for an answer. Same root as the
   13-04-31 session, where a servo whir one second after his question swallowed
   "This is the workshop room" whole.
2. The own-echo rejector skipped anything under OWN_ECHO_MIN_WORDS (3), so a one-word
   echo of his own opening walked straight through — and, because rejection happens
   before speaker resolution, it also reached the anonymous-speaker path and grew a
   persisted cross-session voice signature for Rex's own residual (id=15, 14 turns).
"""

import unittest
from unittest import mock

import config
from audio import sound_effects as sfx, speech_queue
from intelligence import interaction as I


class ReplyWindowSfxTests(unittest.TestCase):
    """No effects while he is waiting for someone to answer him."""

    def _window(self, family, *, drained=True, since=1.0):
        with mock.patch.object(speech_queue, "is_drained", return_value=drained), \
                mock.patch.object(speech_queue, "seconds_since_last_speech",
                                  return_value=since):
            return sfx._in_reply_window(family)

    def test_servo_whir_held_right_after_he_stops(self):
        # 13:07:10 — this is the clip that ate "This is the workshop room".
        self.assertTrue(self._window("servo", since=1.0))

    def test_speech_chirp_held_right_after_he_stops(self):
        # 13:21:50 — the triumphant chirp that came back as "Naturally."
        self.assertTrue(self._window("speech", since=2.0))

    def test_motion_whir_is_exempt(self):
        """Motor feedback for a move the person just asked for is expected, not
        decoration — dropping it re-opens the 2026-07-24 complaint."""
        self.assertFalse(self._window("motion", since=1.0))

    def test_effects_riding_his_own_speech_are_untouched(self):
        """is_drained() is False between the sentences of a reply and while anything
        is queued, so the synthesis-gap emotion chirp still fires."""
        self.assertFalse(self._window("speech", drained=False, since=0.1))

    def test_window_expires(self):
        self.assertFalse(self._window("speech", since=99.0))

    def test_disabled_by_zero(self):
        with mock.patch.object(config, "SOUND_EFFECTS_REPLY_WINDOW_SECS", 0.0):
            self.assertFalse(self._window("servo", since=0.1))

    def test_lookup_failure_fails_open(self):
        """An effect must never be silenced by a bookkeeping error."""
        with mock.patch.object(speech_queue, "is_drained",
                               side_effect=RuntimeError("boom")):
            self.assertFalse(sfx._in_reply_window("servo"))

    def test_play_consults_the_window(self):
        with mock.patch.object(sfx, "_in_reply_window", return_value=True) as gate, \
                mock.patch.object(sfx, "_enabled", return_value=True):
            self.assertFalse(sfx.play("proud"))
        gate.assert_called_once_with("speech")

    def test_forced_effects_still_play(self):
        """force=True already bypasses cooldowns and enables; keep it a real override."""
        with mock.patch.object(sfx, "_in_reply_window", return_value=True) as gate, \
                mock.patch.object(sfx, "_enabled", return_value=True), \
                mock.patch.object(sfx, "_stems_for", return_value=[]):
            sfx.play("proud", force=True)
        gate.assert_not_called()


class ShortOwnEchoTests(unittest.TestCase):
    """A one-word echo of his own opening is still his own voice."""

    def setUp(self):
        I._note_rex_spoke(
            "Naturally. The outage got a vote of no confidence, and I'm "
            "enjoying the silence it left behind."
        )

    def test_the_field_case_is_rejected(self):
        self.assertTrue(I._looks_like_own_echo("Naturally."))

    def test_backchannels_keep_the_min_words_protection(self):
        """These carry no signal about WHO said them, so they stay the human's."""
        for word in ("Yeah.", "Okay", "No.", "Sure", "Right", "Hmm", "What"):
            with self.subTest(word):
                self.assertFalse(I._looks_like_own_echo(word))

    def test_only_a_verbatim_opening_counts(self):
        # "the outage" appears in the line but is not how it OPENS.
        self.assertFalse(I._looks_like_own_echo("The outage"))
        self.assertFalse(I._looks_like_own_echo("Bananas"))

    def test_expires_with_the_capture_seam(self):
        seam = float(getattr(config, "OWN_ECHO_SEAM_SECS", 8.0))
        with mock.patch.object(I.time, "monotonic",
                               return_value=I.time.monotonic() + seam + 5.0):
            self.assertFalse(I._looks_like_own_echo("Naturally."))

    def test_kill_switch(self):
        with mock.patch.object(config, "OWN_ECHO_SHORT_PREFIX_ENABLED", False,
                               create=True):
            self.assertFalse(I._looks_like_own_echo("Naturally."))

    def test_long_echoes_are_unaffected(self):
        self.assertTrue(I._looks_like_own_echo(
            "Naturally. The outage got a vote of no confidence"))
        self.assertFalse(I._looks_like_own_echo(
            "what do you think about the weather today"))

    def test_rejection_precedes_speaker_resolution(self):
        """Why this also stops the voice-signature poisoning: the echo return is
        upstream of person resolution and the anonymous-speaker slot, so a rejected
        echo never reaches voice_signatures at all."""
        import inspect
        src = inspect.getsource(I._handle_speech_segment)
        self.assertLess(src.index("_looks_like_own_echo(text)"),
                        src.index("_resolve_anonymous_speaker_slot"))


if __name__ == "__main__":
    unittest.main()
