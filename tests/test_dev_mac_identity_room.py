"""Regressions for the 2026-09-06 11:45:56 dev-Mac session. No live I/O."""
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from intelligence import attribution as A


def face(pid=1):
    return {'person_db_id': pid, 'face_id': 'Bret Benziger', 'face_visible': True}


def evidence(**changes):
    ev = A.UtteranceEvidence(
        raw_best_id=1, raw_best_name='Bret Benziger', raw_best_score=.234,
        margin=1.234, required_margin=.07, hard_threshold=.5,
        words=2, voiced_secs=.39, previous_speaker_pid=1,
        continuity_age_secs=12.5, allow_short_continuity=True,
        visible_known_ids=[1], visual_observations=[{'faces': [face()]} for _ in range(5)],
    )
    return replace(ev, **changes)


class ShortReplyTests(unittest.TestCase):
    def test_reported_short_replies_preserve_name_but_never_learn(self):
        for text, score, words, secs, age in (
            ('This is the bedroom.', .490, 4, 1.08, 67.8),
            ("No, it's me.", .414, 3, 1.35, 10.2),
            ("What's up.", .234, 2, .39, 12.5),
        ):
            with self.subTest(text=text):
                verdict = A.resolve_authoritative(evidence(text=text, raw_best_score=score,
                                                          words=words, voiced_secs=secs, continuity_age_secs=age))
                self.assertEqual(verdict.person_id, 1)
                self.assertEqual(verdict.status, 'known')
                self.assertFalse(verdict.as_dict()['learning_allowed'])

    def test_missing_or_contradictory_evidence_does_not_carry_name(self):
        for change in (
            dict(continuity_age_secs=91), dict(continuity_age_secs=None),
            dict(previous_speaker_pid=2), dict(raw_best_score=.10),
            dict(raw_best_id=2), dict(margin=.01), dict(visible_known_ids=[1, 2]),
            dict(bearing_contradiction=True), dict(bearing_selected_pid=2),
            dict(visual_observations=[]), dict(mixed_speakers=True),
            dict(visual_observations=[{'faces': [face(), {'face_visible': True}]}]*5),
            dict(visual_observations=[{'faces': [face()]}]*4 + [{'faces': []}]),
            dict(visual_observations=[{'faces': [dict(face(), face_missing=True)]}]*5),
            dict(voiced_secs=2.5, words=8), dict(allow_short_continuity=False),
        ):
            with self.subTest(change=change):
                self.assertIsNone(A.resolve_authoritative(evidence(**change)).person_id)

    def test_contextual_turn_cannot_extend_verified_anchor_or_learn(self):
        from intelligence import interaction as I
        with patch.object(I, '_last_speaker_turn', {'person_id': 1, 'verified_at': 100., 'at': 100.}), \
             patch.object(I, '_current_turn_speaker_evidence', {
                 'raw_best_id': 1, 'raw_best_score': .234,
                 'resolution': {'status': 'known', 'person_id': 1, 'learning_allowed': False}}), \
             patch.object(I.time, 'monotonic', return_value=120.):
            self.assertTrue(I._turn_speaker_uncertain())
            I._note_last_speaker_turn(1, 'Bret Benziger', None)
            self.assertEqual(I._last_speaker_turn['verified_at'], 100.)
            self.assertEqual(I._last_speaker_turn['at'], 120.)

    def test_production_resolver_receives_capture_and_verified_anchor(self):
        from intelligence import interaction as I
        with patch.object(I.speaker_id, 'active_backend', return_value='campplus'), \
             patch.object(I, '_utterance_observations', {'visual': evidence().visual_observations}), \
             patch.object(I, '_last_scan_secs', {'voiced': .39}), \
             patch.object(I, '_last_scan_windows', []), \
             patch.object(I, '_last_scan_ranked', [(1, 'Bret Benziger', .234, 1)]), \
             patch.object(I, '_current_turn_speaker_evidence', {}), \
             patch.object(I.time, 'monotonic', return_value=120.):
            result = I._resolve_turn_attribution(
                turn_id=7, text="What's up.", text_input=False,
                raw_best_id=1, raw_best_name='Bret Benziger', speaker_score=.234,
                speaker_margin=1.234, required_margin=.07, accept_tier=None,
                identity_resolution=None, person_id=None, person_name=None,
                off_camera_unknown=False, visible_known_ids=[1], bearing_match=None,
                engaged=None, previous_speaker={'person_id':1, 'at':107.5, 'verified_at':107.5})
            self.assertEqual(result.person_id, 1)
            self.assertFalse(result.learning_allowed)

    def test_intervening_unidentified_turn_clears_continuity(self):
        from intelligence import interaction as I
        with patch.object(I, '_last_speaker_turn', {'person_id': 1, 'verified_at': 100., 'at': 100.}):
            I._note_last_speaker_turn(None, None, None)
            self.assertIsNone(I._last_speaker_turn)

    def test_actual_voice_match_refreshes_verified_anchor(self):
        from intelligence import interaction as I
        with patch.object(I, '_last_speaker_turn', None), \
             patch.object(I, '_current_turn_speaker_evidence', {
                 'raw_best_id': 1, 'raw_best_score': .683,
                 'resolution': {'status': 'known', 'person_id': 1, 'learning_allowed': True}}), \
             patch.object(I.time, 'monotonic', return_value=120.):
            I._note_last_speaker_turn(1, 'Bret Benziger', None)
            self.assertEqual(I._last_speaker_turn['verified_at'], 120.)


class IdentityAnswerTests(unittest.TestCase):
    def test_answer_names_human_without_llm_pronoun_rewrite(self):
        from intelligence import interaction as I
        with patch.object(I, '_current_turn_speaker_evidence', {'resolution': {
                'status': 'known', 'person_id': 1, 'name': 'Bret Benziger', 'learning_allowed': True}}), \
             patch.object(I, '_speak_blocking') as speak, \
             patch.object(I.llm, 'get_response') as llm:
            result = I._handle_classified_intent('query_who_is_speaking', "Do you know who's speaking?", 1,
                                                raw_best_id=1, raw_best_name='Bret Benziger', raw_best_score=.564)
            self.assertEqual(result, "You're Bret Benziger.")
            speak.assert_called_once_with(result)
            llm.assert_not_called()

    def test_uncertain_resolution_wins_over_face_and_raw_argmax(self):
        from intelligence import interaction as I
        with patch.object(I, '_current_turn_speaker_evidence', {'resolution': {'status': 'ambiguous'}}), \
             patch.object(I, '_speak_blocking'):
            answer = I._handle_classified_intent('query_who_is_speaking', "Who's speaking?", None,
                                                raw_best_name='Bret', raw_best_score=.9, visible_known_name='Bret')
            self.assertNotIn('Bret', answer)
            self.assertIn('not certain', answer)


class DuplicatePrintTests(unittest.TestCase):
    def test_new_and_historical_duplicates_count_as_one_and_do_not_bias_centroid(self):
        from audio import speaker_id as S, voice_score
        from memory import people, database as db
        from setup_assets import DB_SCHEMA
        v = np.zeros(192, dtype=np.float32); v[0] = 1
        w = np.zeros(192, dtype=np.float32); w[1] = 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'people.db'
            with sqlite3.connect(path) as conn:
                conn.executescript(DB_SCHEMA)
                conn.execute("INSERT INTO people(id,name) VALUES(1,'Bret Benziger')")
            with patch.object(db, '_DB_FILE', path), patch.object(voice_score, '_active_backend', 'campplus'):
                first = people.add_biometric(1, 'voice', v)
                self.assertEqual(people.add_biometric(1, 'voice', v), first)
                # Old versions wrote duplicates; retain rows but ignore duplicate weight.
                db.execute("INSERT INTO biometrics(person_id,type,encoding) VALUES(1,'voice_campplus_zh_en_v1',?)", (v.tobytes(),))
                people.add_biometric(1, 'voice', w)
                self.assertEqual(S.comparable_print_count(1), 2)
                self.assertEqual(people.count_native_voice_prints(1), 2)
                self.assertEqual(people.count_biometrics(1, 'voice'), 2)
                rank = S.rank_embedding(v)[0]
                self.assertEqual(rank[3], 2)
                self.assertAlmostEqual(rank[2], 1/np.sqrt(2), places=5)


class RoomTests(unittest.TestCase):
    def test_reported_name_survives_pending_and_failed_visual_learning(self):
        from intelligence import place_questions as pq
        service = unittest.mock.Mock()
        service.belief_context.return_value = {'belief': None, 'enrolling': None}
        with patch.object(pq, '_service', return_value=service), \
             patch.object(pq, '_enabled', return_value=True), \
             patch.object(pq, '_last_capture', {'name': 'bedroom'}), \
             patch.object(pq, '_last_capture_at', 100.), \
             patch.object(pq.time, 'monotonic', return_value=160.):
            clause = pq.belief_clause()
            self.assertIn('bedroom', clause)
            self.assertIn('do not ask', clause)
            self.assertIn('not proof', clause)
            for known in (True, False):
                line = pq.ack_line({'name': 'bedroom', 'known': known})
                self.assertNotIn('recognize', line)
                self.assertNotIn('agree', line)
                self.assertNotIn("know it next time", line)

    def test_blocked_capture_explains_reason_and_recovers_after_clear_view(self):
        from perception.place_recognition import PlaceRecognizer
        now, occ, events = [100.], [.8], []
        with PlaceRecognizerContext(now, occ, events) as rec:
            rec.enroll('bedroom')
            for _ in range(4):
                rec.observe(np.array([1., 0.], dtype=np.float32)); now[0] += 3.1
            blocked = [p for name,p in events if name == 'enrollment_blocked']
            self.assertEqual(len(blocked), 1)
            self.assertEqual(blocked[0]['reason'], 'person_occlusion')
            occ[0] = 0
            for _ in range(8):
                rec.observe(np.array([1., 0.], dtype=np.float32)); now[0] += 3.1
            self.assertTrue(any(name == 'place_enrolled' for name,p in events))

    def test_failed_capture_contains_actual_rejection_counts(self):
        now, occ, events = [100.], [.8], []
        with PlaceRecognizerContext(now, occ, events) as rec:
            rec.enroll('bedroom')
            rec.observe(np.array([1., 0.], dtype=np.float32))
            now[0] += 61
            rec.tick()
            failed = [p for name,p in events if name == 'enrollment_failed'][0]
            self.assertEqual(failed['collected'], 0)
            self.assertEqual(failed['skipped'], {'person_occlusion': 1})


class PlaceRecognizerContext:
    def __init__(self, now, occ, events):
        from perception.place_recognition import PlaceRecognizer
        self.rec = PlaceRecognizer(embed_fn=lambda f:f, get_heading=lambda:None,
                                   get_person_occlusion=lambda:occ[0], clock=lambda:now[0],
                                   emit_event=lambda n,p:events.append((n,p)), db_path=':memory:', model_tag='test')
    def __enter__(self): return self.rec
    def __exit__(self, *args): self.rec.close()
