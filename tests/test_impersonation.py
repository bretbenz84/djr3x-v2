"""
Tests for the impersonation feature: target resolution, reference-clip discovery
(incl. loose famous-name matching), live-capture persistence, the boundary-excluding
script prompt, the spoken performance's voice_ref threading, and the router evidence
gate. The local TTS engine and the LLM are mocked — no model loads, no audio.
"""

import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import config
from audio import local_tts
from features import impersonation


def _write_ref(dir_path: Path, stem: str):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{stem}.wav").write_bytes(b"RIFFstub")
    (dir_path / f"{stem}.txt").write_text("a reference transcript", encoding="utf-8")


class PureHelpersTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(impersonation.slugify("Jimmy Carter"), "jimmy-carter")
        self.assertEqual(impersonation.slugify("  President  Carter! "), "president-carter")

    def test_is_self(self):
        self.assertTrue(impersonation._is_self("me"))
        self.assertTrue(impersonation._is_self("MySelf"))
        self.assertTrue(impersonation._is_self(""))
        self.assertFalse(impersonation._is_self("Bret"))

    def test_cancel_detection(self):
        self.assertTrue(impersonation.sounds_like_cancel("never mind"))
        self.assertTrue(impersonation.sounds_like_cancel("actually, no"))
        self.assertFalse(impersonation.sounds_like_cancel("The cantina is open."))

    def test_capture_lines_nonempty(self):
        self.assertTrue(impersonation.capture_line())
        self.assertTrue(impersonation.intro_line())


class ReferenceDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._p = mock.patch.object(config, "VOICES_DIR", self._tmp.name)
        self._p.start()
        self.voices = Path(self._tmp.name)

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def test_famous_exact_and_loose_match(self):
        _write_ref(self.voices / "famous", "jimmy-carter")
        self.assertIsNotNone(impersonation.find_famous_ref("Jimmy Carter"))   # exact
        loose = impersonation.find_famous_ref("Carter")                        # surname only
        self.assertIsNotNone(loose)
        self.assertEqual(loose.label, "famous:jimmy-carter")
        pres = impersonation.find_famous_ref("President Carter")               # stopword + surname
        self.assertIsNotNone(pres)
        self.assertIsNone(impersonation.find_famous_ref("Ronald Reagan"))      # not present

    def test_famous_missing_txt_is_unusable(self):
        (self.voices / "famous").mkdir(parents=True)
        (self.voices / "famous" / "solo.wav").write_bytes(b"stub")   # no .txt
        self.assertIsNone(impersonation.find_famous_ref("solo"))

    def test_person_ref_roundtrip(self):
        _write_ref(self.voices / "people", "7")
        ref = impersonation.person_ref(7)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.label, "person:7")

    def test_save_person_capture_writes_files(self):
        audio = np.linspace(-0.5, 0.5, 16000 * 5, dtype=np.float32)  # 5s
        ref = impersonation.save_person_capture(42, audio, "This is what I actually said.")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.label, "person:42")
        self.assertEqual(ref.ref_text, "This is what I actually said.")
        people = self.voices / "people"
        self.assertTrue((people / "42.wav").exists())
        self.assertTrue((people / "42.txt").exists())
        self.assertTrue((people / "42.json").exists())

    def test_save_person_capture_rejects_empty_transcript(self):
        audio = np.zeros(16000, dtype=np.float32)
        self.assertIsNone(impersonation.save_person_capture(1, audio, "   "))


class ResolveTargetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.voices = Path(self._tmp.name)
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(config, "VOICES_DIR", self._tmp.name))
        self._stack.enter_context(mock.patch.object(config, "IMPERSONATION_ENABLED", True))
        # Engine "installed".
        self._stack.enter_context(mock.patch.object(local_tts, "is_available", return_value=True))

    def tearDown(self):
        self._stack.close()
        self._tmp.cleanup()

    def test_disabled_refuses(self):
        with mock.patch.object(config, "IMPERSONATION_ENABLED", False):
            r = impersonation.resolve_target("me", 1, "Bret")
        self.assertEqual(r.kind, "refuse")

    def test_engine_unavailable_refuses(self):
        with mock.patch.object(local_tts, "is_available", return_value=False):
            r = impersonation.resolve_target("me", 1, "Bret")
        self.assertEqual(r.kind, "refuse")

    def test_self_unknown_speaker_refuses(self):
        r = impersonation.resolve_target("me", None, None)
        self.assertEqual(r.kind, "refuse")

    def test_self_known_no_ref_opens_capture(self):
        r = impersonation.resolve_target("myself", 1, "Bret")
        self.assertEqual(r.kind, "capture")
        self.assertEqual(r.person_id, 1)
        self.assertTrue(r.is_self)
        self.assertTrue(r.line)

    def test_self_known_with_ref_performs(self):
        _write_ref(self.voices / "people", "1")
        r = impersonation.resolve_target("me", 1, "Bret")
        self.assertEqual(r.kind, "perform")
        self.assertEqual(r.ref.label, "person:1")

    def test_named_known_person_beats_famous(self):
        _write_ref(self.voices / "famous", "bret")     # a famous "bret" clip also exists
        _write_ref(self.voices / "people", "9")        # and Bret is an enrolled person
        with mock.patch("memory.people.find_person_by_name", return_value={"id": 9, "name": "Bret"}):
            r = impersonation.resolve_target("Bret", 1, "Someone")
        self.assertEqual(r.kind, "perform")
        self.assertEqual(r.ref.label, "person:9")      # known person, not famous:bret

    def test_named_unknown_falls_to_famous(self):
        _write_ref(self.voices / "famous", "patrick-stewart")
        with mock.patch("memory.people.find_person_by_name", return_value=None):
            r = impersonation.resolve_target("Patrick Stewart", 1, "Bret")
        self.assertEqual(r.kind, "perform")
        self.assertEqual(r.ref.label, "famous:patrick-stewart")

    def test_named_unknown_no_clip_refuses(self):
        with mock.patch("memory.people.find_person_by_name", return_value=None):
            r = impersonation.resolve_target("Some Rando", 1, "Bret")
        self.assertEqual(r.kind, "refuse")


class ScriptPromptTest(unittest.TestCase):
    def test_prompt_includes_material_and_hard_boundaries(self):
        prompt = impersonation._script_prompt(
            "Bret",
            material=["loves obscure synths", "always late"],
            do_not=["do not mention the layoff", "grief: lost his dog"],
            is_self=True,
            famous=False,
        )
        self.assertIn("loves obscure synths", prompt)
        self.assertIn("NEVER reference", prompt)
        self.assertIn("the layoff", prompt)

    def test_famous_prompt_flags_no_cheap_shots(self):
        prompt = impersonation._script_prompt("Jimmy Carter", [], [], is_self=False, famous=True)
        self.assertIn("public figure", prompt.lower())


class PerformThreadingTest(unittest.TestCase):
    def test_perform_threads_voice_ref_and_frames_in_rex_voice(self):
        ref = local_tts.VoiceRef("/x.wav", "ref", "person:3")
        calls = []

        class _Done:
            def wait(self, timeout=None):
                return True

        def fake_enqueue(text, emotion, **kw):
            calls.append({"text": text, "voice_ref": kw.get("voice_ref"), "log_text": kw.get("log_text")})
            return _Done()

        with mock.patch("audio.speech_queue.enqueue", side_effect=fake_enqueue), \
             mock.patch.object(impersonation, "build_parody_script", return_value="I am Bret and I am always late."), \
             mock.patch("memory.episodes.record_episode") as rec, \
             mock.patch.object(config, "IMPERSONATION_OUTRO_ENABLED", True):
            result = impersonation.perform(ref, "Bret", 3, is_self=False)

        self.assertEqual(result, "I am Bret and I am always late.")
        # intro (Rex voice, no ref), parody (clone ref, not logged by queue), outro (Rex voice).
        self.assertGreaterEqual(len(calls), 3)
        parody_calls = [c for c in calls if c["voice_ref"] is ref]
        self.assertEqual(len(parody_calls), 1)
        self.assertFalse(parody_calls[0]["log_text"])   # caller logs it once
        self.assertIsNone(calls[0]["voice_ref"])        # intro is Rex's own voice
        rec.assert_called_once()

    def test_perform_script_miss_covers_in_rex_voice(self):
        ref = local_tts.VoiceRef("/x.wav", "ref", "person:3")

        class _Done:
            def wait(self, timeout=None):
                return True

        with mock.patch("audio.speech_queue.enqueue", return_value=_Done()) as enq, \
             mock.patch.object(impersonation, "build_parody_script", return_value=None), \
             mock.patch("memory.episodes.record_episode") as rec:
            result = impersonation.perform(ref, "Bret", 3)

        self.assertIn("fuse", result.lower())
        rec.assert_not_called()   # no episode logged for a failed bit
        # No enqueue used the clone voice_ref (synthesis never happened).
        for call in enq.call_args_list:
            self.assertIsNone(call.kwargs.get("voice_ref"))


class RouterGateTest(unittest.TestCase):
    def test_impersonate_not_blocked_by_evidence_gate(self):
        from intelligence import action_router as ar
        decision = ar.ActionDecision(
            action="performance.impersonate", args={"target": "speaker"},
            confidence=0.9, requires_confirmation=False,
        )
        # The evidence gate should NOT block it (no explicit-performance evidence needed).
        reason = ar.missing_required_evidence_reason("do an impression of me", decision)
        self.assertIsNone(reason)


class CaptureConsumerTest(unittest.TestCase):
    """The interaction.py pending-slot consumer _handle_impersonation_capture."""

    @classmethod
    def setUpClass(cls):
        from intelligence import interaction
        cls.itn = interaction

    def setUp(self):
        self.itn._pending_impersonation_capture = None

    def tearDown(self):
        self.itn._pending_impersonation_capture = None

    def _good_audio(self):
        return np.zeros(int(16000 * 5), dtype=np.float32)   # 5s ≥ min

    def test_no_slot_falls_through(self):
        self.assertIsNone(
            self.itn._handle_impersonation_capture("hi", self._good_audio(), 1, 1, 0.9)
        )

    def test_stale_slot_is_cleared(self):
        self.itn._pending_impersonation_capture = {"person_id": 1, "name": "Bret", "asked_at": 0.0}
        with mock.patch.object(config, "IMPERSONATION_CAPTURE_TIMEOUT_SECS", 1.0):
            r = self.itn._handle_impersonation_capture("line", self._good_audio(), 1, 1, 0.9)
        self.assertIsNone(r)
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_cancel_backs_out(self):
        import time
        self.itn._pending_impersonation_capture = {
            "person_id": 1, "name": "Bret", "asked_at": time.monotonic()
        }
        r = self.itn._handle_impersonation_capture("never mind", self._good_audio(), 1, 1, 0.9)
        self.assertIsNotNone(r)
        _line, spoken = r
        self.assertFalse(spoken)
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_wrong_confident_speaker_keeps_slot(self):
        import time
        self.itn._pending_impersonation_capture = {
            "person_id": 5, "name": "Bret", "asked_at": time.monotonic()
        }
        r = self.itn._handle_impersonation_capture(
            "the cantina is open and the music is loud", self._good_audio(),
            person_id=9, raw_best_id=9, speaker_score=0.9,
        )
        self.assertIsNone(r)
        self.assertIsNotNone(self.itn._pending_impersonation_capture)

    def test_short_clip_reasks_and_keeps_slot(self):
        import time
        self.itn._pending_impersonation_capture = {
            "person_id": 1, "name": "Bret", "asked_at": time.monotonic()
        }
        short = np.zeros(int(16000 * 1.0), dtype=np.float32)
        r = self.itn._handle_impersonation_capture("cantina", short, 1, 1, 0.9)
        _line, spoken = r
        self.assertFalse(spoken)
        self.assertIsNotNone(self.itn._pending_impersonation_capture)

    def test_good_clip_saves_and_performs(self):
        import time
        from features import impersonation
        self.itn._pending_impersonation_capture = {
            "person_id": 1, "name": "Bret", "is_self": True, "asked_at": time.monotonic()
        }
        fake_ref = local_tts.VoiceRef("/x.wav", "t", "person:1")
        with mock.patch.object(impersonation, "save_person_capture", return_value=fake_ref) as save, \
             mock.patch.object(impersonation, "perform", return_value="I'm Bret and I'm late.") as perf:
            r = self.itn._handle_impersonation_capture(
                "the cantina is open and the music is loud", self._good_audio(), 1, 1, 0.9
            )
        _line, spoken = r
        self.assertTrue(spoken)
        self.assertEqual(_line, "I'm Bret and I'm late.")
        save.assert_called_once()
        perf.assert_called_once()
        self.assertIsNone(self.itn._pending_impersonation_capture)


if __name__ == "__main__":
    unittest.main()
