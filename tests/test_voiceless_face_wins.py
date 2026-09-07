"""
The voiceless-face rule + voice-sample request flow, from the 2026-08-23 21:07
session (logs/djr3x-2026-08-23-21-07-31.log).

PJ's face was enrolled (3 references) but his voice never was. His speech,
having no row of its own to match, landed on Bret's centroid at 0.79–0.94 —
CONFIDENT territory — so "voice over visible face" credited off-camera Bret
turn after turn while PJ's recognized face was on camera ("I know, Bret — …"
spoken straight at PJ). And because nothing ever enrolled his voice, the
failure was permanent.

Under test:
  - speaker_id.comparable_print_count: the voiceless-face signature (0 clips
    under the ACTIVE embedder; other-embedder rows don't count).
  - _voice_primary_face_decision "voiceless_face_wins": a cross-match — even a
    confident one — does not override the sole visible known face when that
    face's person has no voice print, unless the matched person was themselves
    on camera moments ago or the visual latch contradicts the face.
  - _maybe_request_voice_sample / _handle_voice_sample_capture: Rex asks the
    person for a line and enrolls only a verified repetition onto their row.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from audio import speaker_id
from intelligence import interaction as I
from memory import database as db


class _TempPeopleDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA

        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()


class ComparablePrintCountTest(_TempPeopleDb):
    def setUp(self):
        super().setUp()
        from memory import people

        self.people = people
        self.pj = people.enroll_person("PJ Thomas")
        self.bret = people.enroll_person("Bret Benziger")
        people.add_biometric(self.bret, "voice", np.zeros(192, dtype=np.float32))

    def test_face_only_person_is_voiceless(self):
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.pj), 0)

    def test_enrolled_person_counts_matching_dim(self):
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.bret), 1)

    def test_other_embedder_rows_do_not_count(self):
        # A stale 256-dim Resemblyzer row can never match a live ECAPA query.
        self.people.add_biometric(self.pj, "voice", np.zeros(256, dtype=np.float32))
        with mock.patch.object(speaker_id.voice_score, "active_backend", return_value="ecapa"):
            self.assertEqual(speaker_id.comparable_print_count(self.pj), 0)


class VoicelessFaceWinsDecisionTest(unittest.TestCase):
    """ws_pid=7 (PJ, visible, print-less); voice candidate = Bret (id 1)."""

    def _decide(self, **kw):
        base = dict(
            person_id=1,
            raw_best_id=1,
            speaker_score=0.851,          # PJ's field score on Bret's centroid
            ws_pid=7,
            single_visible=True,
            engaged_is_visible=False,
            unknown_visible=False,
            other_known_recently=False,
            ws_voiceless=True,
            raw_best_recently_visible=False,
        )
        base.update(kw)
        return I._voice_primary_face_decision(**base)

    def test_confident_cross_match_loses_to_voiceless_face(self):
        self.assertEqual(self._decide(), "voiceless_face_wins")

    def test_even_the_slam_dunk_score_loses(self):
        # "Come here." hit 0.938 — as high as genuine Bret ever scores.
        self.assertEqual(self._decide(speaker_score=0.938), "voiceless_face_wins")

    def test_sub_confident_cross_match_also_resolves_voiceless(self):
        self.assertEqual(self._decide(speaker_score=0.62), "voiceless_face_wins")

    def test_matched_person_just_left_frame_keeps_voice(self):
        # Bret stepped out of frame seconds ago — a real off-camera speaker.
        self.assertEqual(
            self._decide(raw_best_recently_visible=True), "voice_over_face"
        )

    def test_face_with_a_print_keeps_the_old_rule(self):
        self.assertEqual(self._decide(ws_voiceless=False), "voice_over_face")

    def test_visual_latch_on_someone_else_keeps_voice(self):
        self.assertEqual(
            self._decide(visual_speaker_pid=1), "voice_over_face"
        )

    def test_visual_latch_on_the_face_still_wins(self):
        self.assertEqual(
            self._decide(visual_speaker_pid=7), "voiceless_face_wins"
        )

    def test_multi_face_scene_does_not_use_the_rule(self):
        self.assertNotEqual(
            self._decide(single_visible=False), "voiceless_face_wins"
        )

    def test_disabled_flag_restores_old_behavior(self):
        with mock.patch.object(config, "VOICELESS_FACE_WINS_ENABLED", False, create=True):
            self.assertEqual(self._decide(), "voice_over_face")


class VoiceSampleRequestTest(unittest.TestCase):
    def setUp(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()

    def tearDown(self):
        I._pending_voice_sample_capture = None
        I._voice_sample_requested_pids.clear()

    def test_arms_once_per_person_per_session(self):
        I._maybe_request_voice_sample(7, "PJ Thomas")
        self.assertIsNotNone(I._pending_voice_sample_capture)
        self.assertIsNone(I._pending_voice_sample_capture["asked_at"])
        I._pending_voice_sample_capture = None
        I._maybe_request_voice_sample(7, "PJ Thomas")   # second time: no re-arm
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_does_not_arm_during_intro_capture(self):
        with mock.patch.object(I, "_pending_intro_voice_capture", {"introduced_id": 5}):
            I._maybe_request_voice_sample(7, "PJ Thomas")
        self.assertIsNone(I._pending_voice_sample_capture)


class VoiceSampleCaptureTest(_TempPeopleDb):
    """The 16:12 pizza capture must not train PJ, through the real storage path."""
    phrase = "Hey Rex, it's PJ — remember my voice, not just my face."

    def setUp(self):
        super().setUp()
        from contextlib import ExitStack
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.face = {'person_db_id': 7, 'face_id': 'PJ', 'face_visible': True}
        self.ctx = dict(person_id=7, name='PJ Thomas', armed_at=89., asked_at=90.,
                        expected_text=self.phrase)
        self.capture = dict(started_at=92., ended_at=95., visual=[
            {'monotonic_at': 92.+i*.25, 'person_db_id': None, 'confidence': 0.,
             'faces': [dict(self.face)]} for i in range(13)])
        for name, value in {
            '_pending_voice_sample_capture': self.ctx, '_utterance_observations': self.capture,
            '_last_scan_windows': [], '_last_scan_ranked': [(1, 'Bret', .354, 1)],
            '_last_scan_secs': {'voiced': 3.}, '_last_confident_voice_at': {},
            '_session_person_ids': set(), '_pending_intro_voice_capture': None,
        }.items():
            self.stack.enter_context(mock.patch.object(I, name, value))
        self.stack.enter_context(mock.patch.object(I.time, 'monotonic', return_value=100.))
        self.stack.enter_context(mock.patch.object(I, '_turn_transcript_trusted', return_value=True))
        self.stack.enter_context(mock.patch.object(speaker_id.voice_score, '_active_backend', 'campplus'))
        for obj, method in ((I.consciousness, 'mark_engagement'), (I.consciousness, 'note_person_spoke'),
                            (I.topic_thread, 'note_user_turn'), (I.user_energy, 'note_user_turn')):
            self.stack.enter_context(mock.patch.object(obj, method))
        with sqlite3.connect(self._path) as conn:
            conn.executemany('insert into people(id,name) values (?,?)', [(1, 'Bret'), (7, 'PJ Thomas')])
        vec = np.zeros(192, dtype=np.float32); vec[0] = 1.
        self.embedding = self.stack.enter_context(mock.patch.object(speaker_id, 'get_embedding', return_value=vec))
        t = np.arange(48000, dtype=np.float32)/16000.
        self.audio = (.1*np.sin(2*np.pi*180*t)).astype(np.float32)

    def count(self):
        with sqlite3.connect(self._path) as conn:
            return conn.execute("select count(*) from biometrics where type like 'voice%'").fetchone()[0]

    def capture_reply(self, text=None, *, person_id=None, raw_id=1, score=.354):
        return I._handle_voice_sample_capture(
            self.phrase if text is None else text, self.audio, person_id, raw_id, score)

    def test_qualifying_requested_sentence_reaches_real_enrollment_storage(self):
        self.assertIn('Got it, PJ', self.capture_reply())
        self.assertEqual(self.count(), 1)
        self.assertIsNone(I._pending_voice_sample_capture)

    def test_field_replies_never_enroll_or_renew_request(self):
        for text in ['Yeah, yeah.', "Says it's a pretty good pizza.", 'My name is PJ.']:
            self.assertIsNone(self.capture_reply(text))
            self.assertEqual(self.ctx['asked_at'], 90.)
        self.assertEqual(self.count(), 0)
        self.embedding.assert_not_called()

    def test_punctuation_case_and_name_spacing_are_tolerated(self):
        self.assertTrue(self.capture_reply("hey rex its P. J. remember my voice not just my face"))
        self.assertEqual(self.count(), 1)

    def test_extra_conversation_or_wrong_name_rejected(self):
        for text in [self.phrase+' It is good pizza.', self.phrase.replace('PJ', 'Bret'), 'Repeat after me: '+self.phrase]:
            self.assertIsNone(self.capture_reply(text))
        self.assertEqual(self.count(), 0)

    def test_off_camera_known_voice_cannot_be_enrolled_as_target(self):
        I._last_scan_ranked = [(1, 'Bret', .851, 1)]
        self.assertIsNone(self.capture_reply(score=.851, person_id=1))
        self.assertEqual(self.count(), 0)
        self.assertFalse(I._safe_enroll_voice(7, self.audio, transcript_text=self.phrase,
                                            source='voice_sample_request'))

    def test_resolved_other_speaker_rejected(self):
        self.assertIsNone(self.capture_reply(person_id=1))
        self.assertEqual(self.count(), 0)

    def test_second_face_or_changed_face_during_capture_rejected(self):
        for change in [{'person_db_id': 1, 'face_id': 'Bret', 'face_visible': True},
                       {'person_db_id': None, 'face_visible': True}]:
            self.capture['visual'][5]['faces'] = [dict(self.face), change]
            self.assertIsNone(self.capture_reply())
        self.assertEqual(self.count(), 0)

    def test_no_interval_camera_or_incomplete_coverage_rejected(self):
        rows = self.capture['visual']
        for limited in [[], rows[8:], rows[:4]+rows[10:]]:
            self.capture['visual'] = limited
            self.assertIsNone(self.capture_reply())
        self.assertEqual(self.count(), 0)

    def test_mixed_audio_rejected(self):
        I._last_scan_windows = [{'change_suspected': True}]
        self.assertIsNone(self.capture_reply())
        I._last_scan_windows = [{'person_id': 1}]
        self.assertIsNone(self.capture_reply())
        self.assertEqual(self.count(), 0)

    def test_pre_ask_audio_and_unsaid_ask_rejected(self):
        self.capture['started_at'] = 89.
        self.assertIsNone(self.capture_reply())
        self.ctx['asked_at'] = None
        self.assertIsNone(self.capture_reply())
        self.assertEqual(self.count(), 0)

    def test_expired_or_missing_phrase_never_guesses(self):
        self.ctx.pop('expected_text')
        self.assertIsNone(self.capture_reply())
        self.ctx['asked_at'] = 1.
        self.assertIsNone(self.capture_reply())
        self.assertIsNone(I._pending_voice_sample_capture)
        self.assertEqual(self.count(), 0)

    def test_untrusted_transcript_rejected(self):
        with mock.patch.object(I, '_turn_transcript_trusted', return_value=False):
            self.assertIsNone(self.capture_reply())
        self.assertEqual(self.count(), 0)

    def test_quality_failure_reasks_only_after_phrase_and_identity_pass(self):
        self.audio = np.zeros(48000, dtype=np.float32)
        self.assertIn('Repeat after me', self.capture_reply())
        self.assertIsNone(self.ctx['asked_at'])  # retry must finish playing first
        self.assertEqual(self.count(), 0)

    def test_other_enrollment_paths_cannot_bypass_pending_sentence(self):
        for source in ['new_person', 'passive', 'campplus_first_voice:self_identification', 'identity_alias_refresh']:
            self.assertFalse(I._safe_enroll_voice(7, self.audio, transcript_text='My name is PJ.',
                                                 source=source, confirmed=True))
        self.assertFalse(I._safe_enroll_voice(1, self.audio, transcript_text=self.phrase,
                                             source='voice_sample_request'))
        self.assertFalse(I._safe_enroll_voice(7, self.audio, transcript_text='Yeah, yeah.',
                                             source='voice_sample_request'))
        self.assertEqual(self.count(), 0)

    def test_refusal_drops_request(self):
        self.assertIsNone(self.capture_reply('Not right now, Rex.'))
        self.assertIsNone(I._pending_voice_sample_capture)
        self.assertEqual(self.count(), 0)

    def test_templates_do_not_decline_their_own_reply(self):
        import re
        decline = re.compile(r"\b(no|nope|not now|not right now|later|wait|hold on|can'?t|cannot)\b")
        for template in config.VOICE_SAMPLE_LINE_TEMPLATES:
            self.assertIsNone(decline.search(template.format(name='PJ').lower()))

    def test_full_turn_handles_verified_sample_before_generic_name_introduction(self):
        # Exercise the real turn handoff and DB write; only I/O is mocked.
        old_people = I.world_state.get('people')
        self.addCleanup(I.world_state.update, 'people', old_people)
        I.world_state.update('people', [dict(self.face, id='pj-track', face_box=(860,400,200,200))])
        for name, value in {
            '_shutdown_requested': False, '_looks_like_own_echo': False,
            '_game_suppresses_conversation': False, '_audio_group_chatter_active': False,
            '_speak_blocking': None, '_maybe_auto_refresh_voice': None,
        }.items():
            self.stack.enter_context(mock.patch.object(I, name, return_value=value))
        self.stack.enter_context(mock.patch.object(I.consciousness, 'note_speaker_gaze_intent'))
        self.stack.enter_context(mock.patch.object(I.consciousness, 'consume_identity_prompt_request', return_value=False))
        self.stack.enter_context(mock.patch.object(I, '_last_speaker_turn', None))
        self.stack.enter_context(mock.patch.object(I, '_pending_offscreen_identify', None))
        self.stack.enter_context(mock.patch.object(I, '_register_rex_utterance'))
        self.stack.enter_context(mock.patch.object(I.conv_log, 'log_rex'))
        heard = self.stack.enter_context(mock.patch.object(I.conv_log, 'log_heard'))
        generic = self.stack.enter_context(mock.patch.object(I, '_handle_pending_name_merge_confirmation'))
        self.stack.enter_context(mock.patch.object(I.llm, 'get_response', return_value='Okay.'))
        I._handle_speech_segment(self.audio, transcribed_text=self.phrase,
            raw_best_id_override=1, raw_best_name_override='Bret', speaker_score_override=.354)
        self.assertEqual(self.count(), 1)
        generic.assert_not_called()
        heard.assert_called_once_with('PJ Thomas', self.phrase)


if __name__ == "__main__":
    unittest.main()
