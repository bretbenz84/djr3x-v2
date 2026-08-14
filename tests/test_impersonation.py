"""
Tests for the impersonation feature: target resolution, reference-clip discovery
(incl. loose famous-name matching), live-capture persistence, the boundary-excluding
script prompt, the spoken performance's voice_ref threading, and the router evidence
gate. The local TTS engine and the LLM are mocked — no model loads, no audio.
"""

import json
import threading
import time
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


class _ReadyTake:
    """Stand-in for local_tts.Take whose first sentence is already rendered."""

    def __init__(self):
        self.first_ready = threading.Event()
        self.first_ready.set()
        self.failed = False
        self.closed = False

    def close(self):
        self.closed = True


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

    def test_capture_prompt_frames_the_ask(self):
        # Field 2026-07-23: Rex spoke the bare phrase with no instruction and the
        # guest had no idea she was supposed to repeat it.
        p = impersonation.capture_prompt("Exudica Marbles", "An apple a day.")
        self.assertIn("repeat after me", p.lower())
        self.assertIn("An apple a day.", p)
        self.assertIn("Exudica", p)
        # No name still frames correctly.
        p2 = impersonation.capture_prompt(None, "Mary had a little lamb.")
        self.assertIn("repeat after me", p2.lower())
        self.assertIn("Mary had a little lamb.", p2)
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

    def test_save_anonymous_capture_uses_session_slot(self):
        audio = np.linspace(-0.5, 0.5, 16000 * 5, dtype=np.float32)
        ref = impersonation.save_person_capture(None, audio, "A stranger's line.")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.label, "person:anon")
        people = self.voices / "people"
        self.assertTrue((people / "anon-latest.wav").exists())
        self.assertTrue((people / "anon-latest.txt").exists())


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

    def test_self_unknown_speaker_gets_anonymous_capture(self):
        # A guest Rex doesn't know still gets the bit: voice cloning needs only
        # the captured clip (live-requested 2026-07-19 after Rex refused a guest).
        r = impersonation.resolve_target("me", None, None)
        self.assertEqual(r.kind, "capture")
        self.assertIsNone(r.person_id)
        self.assertTrue(r.is_self)
        self.assertTrue(r.line)

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

    def test_stranger_prompt_forbids_invented_facts_and_famous_framing(self):
        prompt = impersonation._script_prompt(
            "my mystery guest", [], [], is_self=True, famous=False, stranger=True
        )
        self.assertIn("nothing about them", prompt.lower())
        self.assertNotIn("public figure", prompt.lower())

    def test_build_parody_stranger_mode_not_famous(self):
        seen = {}
        def fake_prompt(name, material, do_not, *, is_self, famous, stranger=False,
                        avoid=None, angle=None):
            seen.update(famous=famous, stranger=stranger, angle=angle)
            return "p"
        with mock.patch.object(impersonation, "_script_prompt", side_effect=fake_prompt), \
             mock.patch("intelligence.llm._client") as client:
            client.chat.completions.create.return_value.choices = [
                mock.Mock(message=mock.Mock(content="a script"))
            ]
            impersonation.build_parody_script("guest", None, is_self=True, stranger=True)
        self.assertTrue(seen["stranger"])
        self.assertFalse(seen["famous"])
        self.assertIsNone(seen["angle"])   # angles are a famous-mode device only

    def test_famous_prompt_collides_their_world_with_the_droid(self):
        prompt = impersonation._script_prompt(
            "Richard Nixon", [], [], is_self=False, famous=True
        )
        low = prompt.lower()
        self.assertIn("droid", low)
        # the bit is ABOUT being impersonated, not just a voice match
        self.assertIn("borrowed his voice", low)
        # and it must not invert -- takes drifted into the president calling
        # HIMSELF a droid until the direction was spelled out
        self.assertIn("never calls himself a droid", low)

    def test_famous_prompt_carries_the_drawn_angle(self):
        prompt = impersonation._script_prompt(
            "Harry Truman", [], [], is_self=False, famous=True,
            angle="have them lodge a dignified complaint",
        )
        self.assertIn("Angle for THIS take: have them lodge a dignified complaint", prompt)

    def test_build_parody_draws_an_angle_for_famous_people(self):
        angles = set()

        def fake_prompt(name, material, do_not, *, is_self, famous, stranger=False,
                        avoid=None, angle=None):
            angles.add(angle)
            return "p"

        with mock.patch.object(impersonation, "_script_prompt", side_effect=fake_prompt), \
             mock.patch.object(impersonation, "_recent_scripts", return_value=[]), \
             mock.patch("intelligence.llm._client") as client:
            client.chat.completions.create.return_value.choices = [
                mock.Mock(message=mock.Mock(content="a script"))
            ]
            for _ in range(40):
                impersonation.build_parody_script("Nixon", None)
        self.assertTrue(angles <= set(impersonation._FAMOUS_ANGLES))
        # 40 draws off 8 angles landing on one lane would mean it isn't varying
        self.assertGreater(len(angles), 1)


class RecentScriptsTest(unittest.TestCase):
    """The avoid-list is what stops a second "do Nixon again" repeating the bit."""

    def _rows(self):
        return [
            {"person_id": None, "detail": json.dumps(
                {"subject": "Nixon", "voice": "famous:richard-nixon", "script": "bit one"})},
            {"person_id": None, "detail": json.dumps(
                {"subject": "President Nixon", "voice": "famous:richard-nixon",
                 "script": "bit two"})},
            {"person_id": None, "detail": json.dumps(
                {"subject": "Reagan", "voice": "famous:ronald-reagan", "script": "other guy"})},
        ]

    def test_voice_key_matches_across_different_spoken_names(self):
        with mock.patch("memory.episodes.recent_episodes", return_value=self._rows()):
            got = impersonation._recent_scripts(
                "Richard Nixon", None, voice_key="famous:richard-nixon"
            )
        # both prior Nixon takes come back even though neither was asked for by
        # the name used this time -- that miss is what made the bit repeat
        self.assertEqual(got, ["bit one", "bit two"])

    def test_voice_key_does_not_pull_in_another_president(self):
        with mock.patch("memory.episodes.recent_episodes", return_value=self._rows()):
            got = impersonation._recent_scripts(
                "Reagan", None, voice_key="famous:ronald-reagan"
            )
        self.assertEqual(got, ["other guy"])

    def test_falls_back_to_subject_name_for_episodes_written_before_the_key(self):
        rows = [{"person_id": None,
                 "detail": json.dumps({"subject": "Nixon", "script": "legacy bit"})}]
        with mock.patch("memory.episodes.recent_episodes", return_value=rows):
            got = impersonation._recent_scripts(
                "Nixon", None, voice_key="famous:richard-nixon"
            )
        self.assertEqual(got, ["legacy bit"])


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

        # Take "already rendered" — unit tests must never touch the real model.
        with mock.patch("audio.speech_queue.enqueue", side_effect=fake_enqueue), \
             mock.patch.object(impersonation, "build_parody_script", return_value="I am Bret and I am always late."), \
             mock.patch.object(impersonation.local_tts, "start_take", return_value=_ReadyTake()), \
             mock.patch("memory.episodes.record_episode") as rec, \
             mock.patch.object(config, "IMPERSONATION_OUTRO_ENABLED", True):
            result = impersonation.perform(ref, "Bret", 3, is_self=False)

        self.assertEqual(result, "I am Bret and I am always late.")
        # intro (Rex voice, no ref), parody (clone ref, not logged by queue), outro (Rex voice).
        self.assertGreaterEqual(len(calls), 3)
        parody_calls = [c for c in calls if c["voice_ref"] is ref]
        self.assertEqual(len(parody_calls), 1)
        # the queue must not log the parody -- perform() logs the SCRIPT itself,
        # since speech_text may be a spoken_form() rewrite of it
        self.assertFalse(parody_calls[0]["log_text"])
        self.assertIsNone(calls[0]["voice_ref"])        # intro is Rex's own voice
        rec.assert_called_once()

    def test_parody_is_logged_between_the_intro_and_the_bow(self):
        """The GUI showed the punchline AFTER the bow: perform() spoke three lines
        but returned one, and the caller's write of it landed last."""
        ref = local_tts.VoiceRef("/x.wav", "ref", "famous:richard-nixon")
        script = "I am not a crook, and that droid is no better."
        written = []

        class _Done:
            def wait(self, timeout=None):
                return True

        def fake_enqueue(text, emotion, **kw):
            if kw.get("log_text"):
                written.append(text)
            return _Done()

        with mock.patch("audio.speech_queue.enqueue", side_effect=fake_enqueue), \
             mock.patch.object(impersonation, "build_parody_script", return_value=script), \
             mock.patch.object(impersonation.local_tts, "start_take", return_value=_ReadyTake()), \
             mock.patch("memory.episodes.record_episode"), \
             mock.patch("utils.conv_log.log_rex", side_effect=written.append) as log_rex, \
             mock.patch("utils.conv_log.claim_rex_line") as claim, \
             mock.patch.object(config, "IMPERSONATION_OUTRO_ENABLED", True):
            result = impersonation.perform(ref, "Nixon", None)

        self.assertEqual(result, script)
        log_rex.assert_called_once_with(script)
        self.assertEqual(len(written), 3)
        self.assertEqual(written[1], script)         # intro, PARODY, bow
        self.assertNotEqual(written[2], script)
        # and the caller's own write of the return value is claimed away, so the
        # parody isn't repeated under the bow
        claim.assert_called_once_with(script)

    def test_perform_script_miss_covers_in_rex_voice(self):
        ref = local_tts.VoiceRef("/x.wav", "ref", "person:3")

        class _Done:
            def wait(self, timeout=None):
                return True

        with mock.patch("audio.speech_queue.enqueue", return_value=_Done()) as enq, \
             mock.patch.object(impersonation, "build_parody_script", return_value=None), \
             mock.patch.object(impersonation.local_tts, "start_take", return_value=_ReadyTake()), \
             mock.patch("memory.episodes.record_episode") as rec:
            result = impersonation.perform(ref, "Bret", 3)

        self.assertIn("fuse", result.lower())
        rec.assert_not_called()   # no episode logged for a failed bit
        # No enqueue used the clone voice_ref (synthesis never happened).
        for call in enq.call_args_list:
            self.assertIsNone(call.kwargs.get("voice_ref"))


class ExplicitClassifierTest(unittest.TestCase):
    """Deterministic impersonation routing — must beat the dialogue-act answer-
    binding that swallowed 'impersonate me' on the dev mac (2026-07-19 log)."""

    def _classify(self, phrase):
        from intelligence import action_router as ar
        return ar.classify_explicit_performance(phrase)

    def test_dev_mac_phrases_route(self):
        for phrase in ("I'd like you to impersonate me", "impersonate me"):
            d = self._classify(phrase)
            self.assertIsNotNone(d, phrase)
            self.assertEqual(d.action, "performance.impersonate")
            self.assertEqual(d.args.get("target"), "speaker")
            self.assertGreaterEqual(d.confidence, 0.85)

    def test_named_and_possessive_targets(self):
        cases = {
            "do an impersonation of Jimmy Carter": "Jimmy Carter",
            "give us your impression of Patrick Stewart": "Patrick Stewart",
            "can you imitate Jimmy Carter": "Jimmy Carter",
            "copy Bret's voice": "Bret",
            "do my voice": "speaker",
            "mimic me please": "speaker",
        }
        for phrase, target in cases.items():
            d = self._classify(phrase)
            self.assertIsNotNone(d, phrase)
            self.assertEqual(d.args.get("target"), target, phrase)

    def test_negations_and_non_requests_do_not_fire(self):
        for phrase in (
            "don't impersonate me",
            "stop imitating him",
            "never impersonate my mother",
            "that was a good impression",
            "what's your first impression of the room",
        ):
            d = self._classify(phrase)
            self.assertTrue(
                d is None or d.action != "performance.impersonate", phrase
            )

    def test_other_performance_patterns_unaffected(self):
        from intelligence import action_router as ar
        d = ar.classify_explicit_performance("do your dj thing")
        self.assertIsNotNone(d)
        self.assertEqual(d.action, "performance.dj_bit")


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

    def test_impersonate_in_execute_allowlist(self):
        # The dev-mac refusal (2026-07-19 21:05 log): the classifier fired at 0.95
        # but ACTION_ROUTER_EXECUTE_ACTIONS didn't list the action, so the gate
        # returned not_in_execute_allowlist and Rex improvised a refusal.
        self.assertIn(
            "performance.impersonate",
            getattr(config, "ACTION_ROUTER_EXECUTE_ACTIONS", set()),
        )

    def test_full_execution_gate_passes_for_dev_mac_phrase(self):
        # End-to-end: the exact phrase → deterministic decision → the REAL
        # interaction.py execution gate must return no block reason.
        from intelligence import action_router as ar
        from intelligence import interaction as itn
        d = ar.classify_explicit_performance("impersonate me")
        self.assertIsNotNone(d)
        self.assertIsNone(itn._router_execution_block_reason(d, text="impersonate me"))


class PreDialogueGateTakeoverTest(unittest.TestCase):
    """The dev-mac 21:09 failure: Rex asked 'what's up?', the user said
    'impersonate me', the dialogue act bound it as an answer (skip_action_router)
    and the whole fast lane was bypassed. The fix mirrors _explicit_motion_takeover:
    an explicit impersonation request runs BEFORE the dialogue gate."""

    @classmethod
    def setUpClass(cls):
        from intelligence import interaction
        cls.itn = interaction

    def setUp(self):
        self.itn._pending_impersonation_capture = None

    def tearDown(self):
        self.itn._pending_impersonation_capture = None

    def test_offline_takeover_fires_and_opens_capture_slot(self):
        # End-to-end through the REAL resolve_target: known speaker, no stored
        # ref → the takeover must speak the capture prompt and open the slot.
        #
        # OFFLINE lane since 2026-08-13: performance.impersonate migrated to the
        # live tool router, so online this pre-dialogue-gate takeover stands down
        # (the reply call sees the utterance whether or not the dialogue act bound
        # the turn as an answer). With the link down there is no tool surface, so
        # this takeover is still what keeps "impersonate me" out of an answer frame.
        from audio import local_tts
        with mock.patch("intelligence.connectivity.is_offline", return_value=True), \
             mock.patch.object(self.itn, "_speak_blocking") as speak, \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch("features.impersonation.person_ref", return_value=None):
            result = self.itn._explicit_impersonation_takeover(
                "impersonate me", person_id=1, person_name="Bret",
            )
        self.assertIsNotNone(result)
        speak.assert_called_once()
        slot = self.itn._pending_impersonation_capture
        self.assertIsNotNone(slot)
        self.assertEqual(slot["person_id"], 1)

    def test_online_takeover_stands_down_for_the_tool_router(self):
        """Online the reply call owns it — this lane must not open a slot."""
        from audio import local_tts
        with mock.patch.object(self.itn, "_speak_blocking") as speak, \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch("features.impersonation.person_ref", return_value=None):
            result = self.itn._explicit_impersonation_takeover(
                "impersonate me", person_id=1, person_name="Bret",
            )
        self.assertIsNone(result)
        speak.assert_not_called()
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_takeover_ignores_non_requests(self):
        self.assertIsNone(
            self.itn._explicit_impersonation_takeover(
                "that was a good impression", person_id=1, person_name="Bret",
            )
        )
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_takeover_runs_before_the_dialogue_gate(self):
        # Structural regression guard for the exact bug: the impersonation
        # takeover call must appear BEFORE the skip_action_router-gated fast-lane
        # call inside interaction.py, so an answer_to_rex binding can't swallow it.
        import inspect
        src = Path(inspect.getsourcefile(self.itn)).read_text()
        takeover_pos = src.find("fast_takeover_response = _explicit_impersonation_takeover(")
        gated_pos = src.find(
            "if fast_takeover_response is None and not dialogue_decision.skip_action_router:"
        )
        self.assertGreater(takeover_pos, 0)
        self.assertGreater(gated_pos, 0)
        self.assertLess(
            takeover_pos, gated_pos,
            "impersonation takeover must run before the dialogue-gated fast lane",
        )


class ResponseWaitSettleTest(unittest.TestCase):
    """The 21:20 barge-in: the takeover epilogue cleared the response wait right
    after 'Repeat after me…', so the smile-reaction (camera saw the user grin at
    the joke-shaped capture line) was free to speak ONE second later. With the
    capture slot open, the epilogue must ARM the wait for the capture window."""

    @classmethod
    def setUpClass(cls):
        from intelligence import interaction
        cls.itn = interaction

    def setUp(self):
        self.itn._pending_impersonation_capture = None

    def tearDown(self):
        self.itn._pending_impersonation_capture = None

    def test_arms_wait_while_capture_pending(self):
        self.itn._pending_impersonation_capture = {"person_id": 1, "asked_at": 0.0}
        with mock.patch.object(self.itn.consciousness, "begin_response_wait") as arm, \
             mock.patch.object(self.itn.consciousness, "clear_response_wait") as clear, \
             mock.patch.object(config, "IMPERSONATION_CAPTURE_TIMEOUT_SECS", 45.0):
            self.itn._settle_response_wait_after_action()
        arm.assert_called_once_with(45.0)
        clear.assert_not_called()

    def test_clears_wait_when_no_capture_pending(self):
        with mock.patch.object(self.itn.consciousness, "begin_response_wait") as arm, \
             mock.patch.object(self.itn.consciousness, "clear_response_wait") as clear:
            self.itn._settle_response_wait_after_action()
        clear.assert_called_once()
        arm.assert_not_called()

    def test_smile_reaction_blocked_while_waiting(self):
        # End-to-end on the real consciousness gate: an armed response wait must
        # suppress the smile reaction (the actual 21:20 barger).
        from intelligence import consciousness
        consciousness.begin_response_wait(45.0)
        try:
            self.assertFalse(consciousness._can_smile_reaction_speak())
        finally:
            consciousness.clear_response_wait()


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

    def test_anonymous_slot_rejects_known_speaker(self):
        # A known person speaking in the gap must NOT be captured as the guest.
        import time
        self.itn._pending_impersonation_capture = {
            "person_id": None, "name": "", "is_self": True, "asked_at": time.monotonic()
        }
        r = self.itn._handle_impersonation_capture(
            "the cantina is open and the music is loud", self._good_audio(),
            person_id=1, raw_best_id=1, speaker_score=0.9,
        )
        self.assertIsNone(r)
        self.assertIsNotNone(self.itn._pending_impersonation_capture)

    def test_anonymous_slot_accepts_unknown_speaker_and_performs(self):
        import time
        from features import impersonation
        self.itn._pending_impersonation_capture = {
            "person_id": None, "name": "", "is_self": True, "asked_at": time.monotonic()
        }
        fake_ref = local_tts.VoiceRef("/x.wav", "t", "person:anon")
        with mock.patch.object(impersonation, "save_person_capture", return_value=fake_ref) as save, \
             mock.patch.object(impersonation, "perform", return_value="I'm mysterious.") as perf:
            r = self.itn._handle_impersonation_capture(
                "the cantina is open and the music is loud", self._good_audio(),
                person_id=None, raw_best_id=None, speaker_score=0.0,
            )
        _line, spoken = r
        self.assertTrue(spoken)
        save.assert_called_once()
        self.assertIsNone(save.call_args.args[0])           # anonymous save
        perf.assert_called_once()
        self.assertEqual(perf.call_args.args[1], "my mystery guest")
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_recitation_match_overrides_misattribution(self):
        # Field 2026-07-23: the guest recited the phrase but her voice was pinned
        # on a junk twin (different person_id) and the strict gate skipped it — the
        # slot silently expired. A transcript matching the requested phrase IS the
        # recitation, whoever the voice system says is talking.
        import time
        from features import impersonation
        phrase = "An apple a day keeps the doctor away, and a penny saved is a penny earned."
        self.itn._pending_impersonation_capture = {
            "person_id": 3, "name": "Exudica", "is_self": True,
            "expected_text": phrase, "asked_at": time.monotonic(),
        }
        fake_ref = local_tts.VoiceRef("/x.wav", "t", "person:3")
        with mock.patch.object(impersonation, "save_person_capture", return_value=fake_ref) as save, \
             mock.patch.object(impersonation, "perform", return_value="Uncanny.") as perf:
            r = self.itn._handle_impersonation_capture(
                "An apple a day keeps the doctor away and a penny saved is a penny earned",
                self._good_audio(),
                person_id=2, raw_best_id=2, speaker_score=0.87,   # wrong person!
            )
        self.assertIsNotNone(r)
        save.assert_called_once()
        perf.assert_called_once()
        self.assertIsNone(self.itn._pending_impersonation_capture)

    def test_non_recitation_from_wrong_speaker_still_skipped(self):
        import time
        phrase = "An apple a day keeps the doctor away, and a penny saved is a penny earned."
        self.itn._pending_impersonation_capture = {
            "person_id": 3, "name": "Exudica", "is_self": True,
            "expected_text": phrase, "asked_at": time.monotonic(),
        }
        r = self.itn._handle_impersonation_capture(
            "hey can you turn the music up a bit", self._good_audio(),
            person_id=2, raw_best_id=2, speaker_score=0.87,
        )
        self.assertIsNone(r)
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


class WhoSlotTest(unittest.TestCase):
    """"Impersonate." with nobody named — Rex asks who, then performs on the answer.

    Before this, a cut-off request classified as nothing, fell through to the
    LLM, and got answered with a refusal to impersonate anyone (field
    2026-08-04)."""

    @classmethod
    def setUpClass(cls):
        from intelligence import interaction
        cls.itn = interaction

    def setUp(self):
        self.itn._pending_impersonation_target = None
        self.itn._pending_impersonation_capture = None

    def tearDown(self):
        self.itn._pending_impersonation_target = None
        self.itn._pending_impersonation_capture = None

    def _bare_decision(self):
        from intelligence import action_router as ar
        return ar.classify_explicit_impersonation("Impersonate.")

    def test_bare_request_asks_who_and_arms_the_slot(self):
        d = self._bare_decision()
        self.assertEqual(d.args["target"], "")
        with mock.patch.object(self.itn, "_speak_blocking") as say:
            line = self.itn._handle_router_impersonation(d, "Impersonate.", 1, "Bret", "")
        self.assertIsNotNone(line)
        # the ask is drawn at random from IMPERSONATION_WHO_LINES, so assert it
        # IS one of them rather than pinning wording ("Name your victim" has no
        # "who" in it and made this flaky)
        self.assertIn(line, config.IMPERSONATION_WHO_LINES)
        say.assert_called_once()
        self.assertIsNotNone(self.itn._pending_impersonation_target)

    def test_answer_is_treated_as_the_target(self):
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": time.monotonic(),
        }
        seen = {}

        def fake_handler(decision, text, pid, pname, target):
            seen["target"] = target
            return "a parody"

        with mock.patch.object(self.itn, "_handle_router_impersonation",
                               side_effect=fake_handler):
            r = self.itn._handle_impersonation_target_prompt("Obama")
        self.assertEqual(r, ("a parody", True))
        self.assertEqual(seen["target"], "Obama")
        self.assertIsNone(self.itn._pending_impersonation_target)

    def test_answer_may_repeat_the_verb(self):
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": time.monotonic(),
        }
        seen = {}
        with mock.patch.object(
            self.itn, "_handle_router_impersonation",
            side_effect=lambda d, t, p, n, target: seen.update(target=target) or "x",
        ):
            self.itn._handle_impersonation_target_prompt("impersonate Richard Nixon")
        self.assertEqual(seen["target"], "Richard Nixon")

    def test_no_slot_falls_through(self):
        self.assertIsNone(self.itn._handle_impersonation_target_prompt("Obama"))

    def test_stale_slot_is_cleared_and_ignored(self):
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": 0.0,
        }
        with mock.patch.object(config, "IMPERSONATION_WHO_TIMEOUT_SECS", 1.0, create=True):
            self.assertIsNone(self.itn._handle_impersonation_target_prompt("Obama"))
        self.assertIsNone(self.itn._pending_impersonation_target)

    def test_cancel_closes_the_slot(self):
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": time.monotonic(),
        }
        with mock.patch.object(self.itn, "_speak_blocking"):
            r = self.itn._handle_impersonation_target_prompt("never mind")
        self.assertIsNotNone(r)
        self.assertIsNone(self.itn._pending_impersonation_target)

    def test_answer_naming_an_enrolled_person_opens_their_capture_slot(self):
        """Answering with a KNOWN person (not "me", not famous) routes through
        resolve_target: they perform if a clip is saved, otherwise Rex opens the
        live-capture slot keyed to THEIR person_id, not the asker's."""
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": time.monotonic(),
        }
        person = {"id": 3, "name": "Exudica Marbles"}
        with mock.patch("memory.people.find_person_by_name", return_value=person), \
             mock.patch.object(impersonation, "person_ref", return_value=None), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(self.itn, "_speak_blocking") as say:
            r = self.itn._handle_impersonation_target_prompt("Exudica")

        self.assertIsNotNone(r)
        slot = self.itn._pending_impersonation_capture
        self.assertIsNotNone(slot)
        self.assertEqual(slot["person_id"], 3)          # hers, not the asker's
        self.assertEqual(slot["name"], "Exudica Marbles")
        self.assertFalse(slot["is_self"])
        self.assertIn("repeat after me", say.call_args.args[0].lower())

    def test_answer_naming_an_enrolled_person_with_a_clip_performs(self):
        self.itn._pending_impersonation_target = {
            "person_id": 1, "person_name": "Bret", "asked_at": time.monotonic(),
        }
        person = {"id": 3, "name": "Exudica Marbles"}
        ref = local_tts.VoiceRef("/x.wav", "ref", "person:3")
        with mock.patch("memory.people.find_person_by_name", return_value=person), \
             mock.patch.object(impersonation, "person_ref", return_value=ref), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(impersonation, "perform", return_value="a parody") as perf:
            r = self.itn._handle_impersonation_target_prompt("Exudica")

        self.assertEqual(r, ("a parody", True))
        self.assertIsNone(self.itn._pending_impersonation_capture)
        self.assertEqual(perf.call_args.args[0].label, "person:3")


class LineCyclingTest(unittest.TestCase):
    """Intro and bow walk the whole list before repeating, and never land on the
    same line twice running. With a random pick over three intros, "loading the
    impression module" opened nearly every bit (field 2026-08-04)."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        d = Path(self._tmp.name)
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(mock.patch.object(
            config, "IMPERSONATION_INTRO_STATE_PATH", str(d / "intro.json"), create=True))
        self._stack.enter_context(mock.patch.object(
            config, "IMPERSONATION_OUTRO_STATE_PATH", str(d / "outro.json"), create=True))

    def test_intro_covers_every_line_before_repeating(self):
        n = len(config.IMPERSONATION_INTRO_LINES)
        self.assertGreaterEqual(n, 10)
        seq = [impersonation.intro_line() for _ in range(n)]
        self.assertEqual(len(set(seq)), n)

    def test_bow_covers_every_line_before_repeating(self):
        n = len(config.IMPERSONATION_OUTRO_LINES)
        self.assertGreaterEqual(n, 10)
        seq = [impersonation.outro_line() for _ in range(n)]
        self.assertEqual(len(set(seq)), n)

    def test_no_back_to_back_repeat_across_the_cycle_boundary(self):
        n = len(config.IMPERSONATION_INTRO_LINES)
        seq = [impersonation.intro_line() for _ in range(n * 3)]
        dupes = [a for a, b in zip(seq, seq[1:]) if a == b]
        self.assertEqual(dupes, [], f"repeated back-to-back: {dupes}")

    def test_intro_and_bow_cycle_independently(self):
        # shared state would let one starve the other's rotation
        impersonation.intro_line()
        bows = {impersonation.outro_line() for _ in range(
            len(config.IMPERSONATION_OUTRO_LINES))}
        self.assertEqual(len(bows), len(config.IMPERSONATION_OUTRO_LINES))

    def test_unwritable_state_still_returns_a_line(self):
        with mock.patch.object(config, "IMPERSONATION_INTRO_STATE_PATH",
                               "/nonexistent-dir/x.json", create=True):
            self.assertIn(impersonation.intro_line(), config.IMPERSONATION_INTRO_LINES)

    def test_bow_disabled_returns_none(self):
        with mock.patch.object(config, "IMPERSONATION_OUTRO_ENABLED", False):
            self.assertIsNone(impersonation.outro_line())
