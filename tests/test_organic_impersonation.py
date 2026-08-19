"""
Tests for the unprompted-impression feature (features/organic_impersonation.py):
famous-name detection against the voice roster, the per-turn claim and its gates,
the self-mock judge, and the player's wait-for-the-moment ordering. The local TTS
engine, the LLM, and the speech queue are mocked — no model loads, no audio.
"""

import os
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import config
from audio import local_tts
from features import impersonation
from features import organic_impersonation as organic


def _write_ref(dir_path: Path, stem: str):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{stem}.wav").write_bytes(b"RIFFstub")
    (dir_path / f"{stem}.txt").write_text("a reference transcript", encoding="utf-8")


class _FakeTake:
    def __init__(self, ready=True, failed=False):
        self.first_ready = threading.Event()
        if ready:
            self.first_ready.set()
        self.failed = failed
        self.closed = False

    @property
    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True


class _Frame:
    def __init__(self, allow_roast="normal"):
        self.allow_roast = allow_roast


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        famous = self.root / "famous"
        for stem in ("jimmy-carter", "gerald-ford", "franklin-roosevelt", "barack-obama"):
            _write_ref(famous, stem)
        os.symlink(famous / "franklin-roosevelt.wav", famous / "fdr.wav")
        os.symlink(famous / "franklin-roosevelt.txt", famous / "fdr.txt")
        _write_ref(self.root / "people", "7")
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(impersonation, "_voices_dir", return_value=self.root))
        self._stack.enter_context(mock.patch.object(local_tts, "is_available", return_value=True))
        self._stack.enter_context(mock.patch.object(config, "IMPERSONATION_ENABLED", True, create=True))
        self._stack.enter_context(mock.patch.object(config, "IMPERSONATION_ORGANIC_ENABLED", True, create=True))
        self._stack.enter_context(mock.patch.object(config, "LOCAL_TTS_MODE", False, create=True))
        organic.reset_state()
        organic.invalidate_roster()

    def tearDown(self):
        organic.reset_state()
        organic.invalidate_roster()
        self._stack.close()
        self._tmp.cleanup()


class DetectFamousMentionTest(_Base):
    def test_full_name_possessive(self):
        hit = organic.detect_famous_mention("I'm going to Jimmy Carter's hometown in September.")
        self.assertIsNotNone(hit)
        name, ref = hit
        self.assertEqual(name, "Jimmy Carter")
        self.assertEqual(ref.label, "famous:jimmy-carter")

    def test_bare_surname_is_not_enough(self):
        self.assertIsNone(organic.detect_famous_mention("I drive a Ford, it's fine."))
        self.assertIsNone(organic.detect_famous_mention("Carter is coming over later."))

    def test_title_plus_surname(self):
        hit = organic.detect_famous_mention("President Ford pardoned him, right?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "Gerald Ford")

    def test_alias_slug_whole_word(self):
        hit = organic.detect_famous_mention("FDR had the fireside chats.")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "FDR")
        self.assertIsNone(organic.detect_famous_mention("the fdrx thing"))

    def test_voice_key_sees_through_alias(self):
        _, ref = organic.detect_famous_mention("FDR had the fireside chats.")
        self.assertEqual(organic._voice_key(ref), "famous:franklin-roosevelt")

    def test_no_mention(self):
        self.assertIsNone(organic.detect_famous_mention("What's the weather like tomorrow?"))


class MaybeClaimTest(_Base):
    def _claim(self, text, person_id=None, roast="normal", **kw):
        with mock.patch.object(organic, "_launch") as launch:
            out = organic.maybe_claim(text, person_id, frame=_Frame(roast), **kw)
        return out, launch

    def test_famous_mention_claims_and_directs(self):
        out, launch = self._claim("I'm going to Jimmy Carter's hometown next month.")
        self.assertIsNotNone(out)
        self.assertIn("Jimmy Carter", out)
        self.assertIn("Do NOT", out)
        launch.assert_called_once()
        prep = launch.call_args[0][0]
        self.assertEqual(prep.kind, "famous")
        self.assertEqual(prep.subject_name, "Jimmy Carter")
        self.assertEqual(prep.trigger, "mention:famous:jimmy-carter")

    def test_explicit_request_is_left_to_the_explicit_flow(self):
        out, launch = self._claim("Impersonate Jimmy Carter for me.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_disabled(self):
        with mock.patch.object(config, "IMPERSONATION_ORGANIC_ENABLED", False, create=True):
            out, launch = self._claim("Jimmy Carter was a peanut farmer.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_heavy_turn_no_bit(self):
        out, launch = self._claim("Jimmy Carter died the same week as my grandfather.", roast="none")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_global_cooldown(self):
        organic._last_fire_at = time.monotonic()
        out, launch = self._claim("Jimmy Carter was a peanut farmer.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_same_voice_cooldown(self):
        organic._voice_last_fire["famous:jimmy-carter"] = time.monotonic()
        out, launch = self._claim("Jimmy Carter was a peanut farmer.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_one_in_flight_at_a_time(self):
        organic._pending = organic._Prep(
            kind="famous", ref=impersonation.find_famous_ref("barack obama"),
            subject_name="Barack Obama", person_id=None, utterance="x", trigger="t",
        )
        out, launch = self._claim("Jimmy Carter was a peanut farmer.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_offline_no_claim(self):
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            out, launch = self._claim("Jimmy Carter was a peanut farmer.")
        self.assertIsNone(out)
        launch.assert_not_called()

    def test_self_mock_needs_captured_voice_and_roast(self):
        with mock.patch.object(config, "IMPERSONATION_SELF_MOCK_CONSIDER_PROB", 1.0, create=True), \
             mock.patch.object(organic, "_person_name", return_value="Bret"):
            # person 7 has a ref; person 8 does not
            out, launch = self._claim("I am definitely the best driver in this whole state.", person_id=8)
            launch.assert_not_called()
            out, launch = self._claim("I am definitely the best driver in this whole state.", person_id=7, roast="light")
            launch.assert_not_called()
            out, launch = self._claim("I am definitely the best driver in this whole state.", person_id=7)
        self.assertIsNone(out)  # no directive for a self-mock — the reply must not brace
        launch.assert_called_once()
        prep = launch.call_args[0][0]
        self.assertEqual(prep.kind, "self")
        self.assertEqual(prep.subject_name, "Bret")
        self.assertEqual(prep.person_id, 7)

    def test_self_mock_short_utterance_skipped(self):
        with mock.patch.object(config, "IMPERSONATION_SELF_MOCK_CONSIDER_PROB", 1.0, create=True):
            out, launch = self._claim("okay sure", person_id=7)
        launch.assert_not_called()


class SelfMockScriptTest(_Base):
    def _prep(self):
        return organic._Prep(
            kind="self", ref=impersonation.person_ref(7), subject_name="Bret",
            person_id=7, utterance="I could totally beat a bear in a fight.",
            trigger="self_mock:judged",
        )

    def _with_llm(self, reply):
        resp = mock.Mock()
        resp.choices = [mock.Mock(message=mock.Mock(content=reply))]
        client = mock.Mock()
        client.chat.completions.create.return_value = resp
        return mock.patch("intelligence.llm._client", client)

    def test_none_means_no_bit(self):
        with self._with_llm("NONE"), \
             mock.patch.object(impersonation, "_gather_material", return_value=([], [])):
            self.assertIsNone(organic._self_mock_script(self._prep()))

    def test_line_is_returned_and_remembered(self):
        with self._with_llm('"I, a soft man made of snacks, will defeat the bear."'), \
             mock.patch.object(impersonation, "_gather_material", return_value=([], ["the divorce"])):
            line = organic._self_mock_script(self._prep())
        self.assertEqual(line, "I, a soft man made of snacks, will defeat the bear.")
        self.assertIn(line, organic._recent_self_mock_scripts)

    def test_prompt_carries_boundaries_and_utterance(self):
        p = organic._self_mock_prompt(self._prep(), ["the divorce"], ["old joke"])
        self.assertIn("beat a bear", p)
        self.assertIn("the divorce", p)
        self.assertIn("old joke", p)
        self.assertIn("NONE", p)


class PlayerTest(_Base):
    def _prep(self, kind="famous"):
        if kind == "famous":
            ref = impersonation.find_famous_ref("jimmy carter")
            return organic._Prep(kind="famous", ref=ref, subject_name="Jimmy Carter",
                                 person_id=None, utterance="u", trigger="mention:famous:jimmy-carter",
                                 voice_key="famous:jimmy-carter", max_wait_secs=2.0)
        return organic._Prep(kind="self", ref=impersonation.person_ref(7), subject_name="Bret",
                             person_id=7, utterance="u", trigger="self_mock:judged",
                             voice_key="person:7", max_wait_secs=2.0)

    def test_plays_bridge_take_outro_after_reply_when_floor_free(self):
        prep = self._prep()
        prep.script = "I'm Jimmy Carter and I built a house."
        prep.speech_text = prep.script
        prep.take = _FakeTake(ready=True)
        prep.prepared.set()
        organic._pending = prep
        spoken = []
        floor = {"free": False}

        def _enqueue(line, emotion, **kw):
            spoken.append((line, kw.get("voice_ref")))
            ev = threading.Event(); ev.set()
            return ev

        with mock.patch("audio.speech_queue.enqueue", side_effect=_enqueue), \
             mock.patch.object(organic, "_floor_is_free", side_effect=lambda: floor["free"]), \
             mock.patch.object(organic, "_bridge_line", return_value="Oh — hang on. Jimmy Carter, everybody:"), \
             mock.patch.object(organic, "_outro_line", return_value="Thank you, thank you."), \
             mock.patch("memory.episodes.record_episode") as rec, \
             mock.patch("memory.conversations.add_to_transcript"), \
             mock.patch("utils.conv_log.log_rex"):
            t = threading.Thread(target=organic._player, args=(prep,))
            t.start()
            time.sleep(0.4)
            self.assertEqual(spoken, [])            # reply not done yet
            organic.note_reply_done()
            time.sleep(0.4)
            self.assertEqual(spoken, [])            # floor not free yet
            floor["free"] = True
            t.join(timeout=3.0)
        self.assertEqual([s[0] for s in spoken],
                         ["Oh — hang on. Jimmy Carter, everybody:",
                          "I'm Jimmy Carter and I built a house.",
                          "Thank you, thank you."])
        self.assertIsNone(spoken[0][1])
        self.assertEqual(spoken[1][1].label, "famous:jimmy-carter")
        self.assertIsNone(spoken[2][1])
        self.assertEqual(organic._session_fires, 1)
        self.assertIn("famous:jimmy-carter", organic._voice_last_fire)
        detail = rec.call_args[1]["detail"]
        self.assertTrue(detail["organic"])
        self.assertEqual(detail["trigger"], "mention:famous:jimmy-carter")
        self.assertIsNone(organic._pending)

    def test_expires_silently_when_floor_never_frees(self):
        prep = self._prep()
        prep.script = "x"; prep.speech_text = "x"
        prep.take = _FakeTake(ready=True)
        prep.prepared.set(); prep.reply_done.set()
        organic._pending = prep
        with mock.patch("audio.speech_queue.enqueue") as enq, \
             mock.patch.object(organic, "_floor_is_free", return_value=False), \
             mock.patch.object(local_tts, "pop_take", return_value=None):
            organic._player(prep)
        enq.assert_not_called()
        self.assertTrue(prep.take.closed)
        self.assertEqual(organic._session_fires, 0)
        self.assertIsNone(organic._pending)

    def test_no_script_drops(self):
        prep = self._prep()
        prep.failed = True
        prep.prepared.set(); prep.reply_done.set()
        organic._pending = prep
        with mock.patch("audio.speech_queue.enqueue") as enq, \
             mock.patch.object(organic, "_floor_is_free", return_value=True):
            organic._player(prep)
        enq.assert_not_called()

    def test_cancel_releases_take(self):
        prep = self._prep()
        prep.script = "x"; prep.speech_text = "x"
        prep.take = _FakeTake(ready=True)
        organic._pending = prep
        with mock.patch.object(local_tts, "pop_take", return_value=None):
            organic.cancel("explicit_request")
        self.assertTrue(prep.cancelled)
        self.assertTrue(prep.take.closed)
        self.assertFalse(organic.has_pending())

    def test_explicit_perform_cancels_pending(self):
        prep = self._prep()
        organic._pending = prep
        with mock.patch.object(organic, "cancel") as cancel, \
             mock.patch("audio.speech_queue.enqueue", side_effect=RuntimeError("stop here")), \
             mock.patch.object(impersonation, "build_parody_script", return_value=None):
            try:
                impersonation.perform(prep.ref, "Jimmy Carter", None)
            except Exception:
                pass
        cancel.assert_called_once_with("explicit_request")

    def test_self_mock_marks_person_cooldown(self):
        prep = self._prep("self")
        prep.script = "I could beat a bear."; prep.speech_text = prep.script
        prep.take = _FakeTake(ready=True)
        prep.prepared.set(); prep.reply_done.set()
        organic._pending = prep
        with mock.patch("audio.speech_queue.enqueue", side_effect=lambda *a, **k: (lambda e: (e.set(), e)[1])(threading.Event())), \
             mock.patch.object(organic, "_floor_is_free", return_value=True), \
             mock.patch("memory.episodes.record_episode"), \
             mock.patch("memory.conversations.add_to_transcript"), \
             mock.patch("utils.conv_log.log_rex"):
            organic._player(prep)
        self.assertIn("person:7", organic._voice_last_fire)


class PrepTest(_Base):
    def test_prepare_famous_starts_take_with_context(self):
        prep = organic._Prep(kind="famous", ref=impersonation.find_famous_ref("jimmy carter"),
                             subject_name="Jimmy Carter", person_id=None,
                             utterance="I'm going to Plains, Georgia.",
                             trigger="mention:famous:jimmy-carter", voice_key="famous:jimmy-carter")
        fake = _FakeTake()
        with mock.patch.object(impersonation, "build_parody_script", return_value="Howdy, I'm Jimmy.") as bps, \
             mock.patch.object(local_tts, "start_take", return_value=fake) as st, \
             mock.patch("audio.tts.spoken_form", side_effect=lambda s: s):
            organic._prepare(prep)
        self.assertTrue(prep.prepared.is_set())
        self.assertIs(prep.take, fake)
        self.assertEqual(bps.call_args[1]["context"], "I'm going to Plains, Georgia.")
        self.assertEqual(bps.call_args[1]["max_words"],
                         int(getattr(config, "IMPERSONATION_ORGANIC_SCRIPT_MAX_WORDS", 30)))
        st.assert_called_once()

    def test_prepare_local_tts_mode_waits_for_reply(self):
        prep = organic._Prep(kind="famous", ref=impersonation.find_famous_ref("jimmy carter"),
                             subject_name="Jimmy Carter", person_id=None, utterance="u",
                             trigger="t", voice_key="famous:jimmy-carter", max_wait_secs=3.0)
        started = threading.Event()
        with mock.patch.object(config, "LOCAL_TTS_MODE", True, create=True), \
             mock.patch.object(impersonation, "build_parody_script", return_value="Howdy."), \
             mock.patch.object(local_tts, "start_take", side_effect=lambda *a, **k: (started.set(), _FakeTake())[1]), \
             mock.patch("audio.tts.spoken_form", side_effect=lambda s: s):
            t = threading.Thread(target=organic._prepare, args=(prep,)); t.start()
            time.sleep(0.3)
            self.assertFalse(started.is_set())
            prep.reply_done.set()
            t.join(timeout=3.0)
        self.assertTrue(started.is_set())


class ScriptCapTest(unittest.TestCase):
    def test_cap_override(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve."
        self.assertEqual(impersonation._cap_script_words(text, max_words=8),
                         "One two three four. Five six seven eight.")


if __name__ == "__main__":
    unittest.main()
