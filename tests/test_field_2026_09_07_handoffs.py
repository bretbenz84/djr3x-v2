"""Regression for the 02:23 run: missing expression detections and competing identity/motion owners."""
import unittest
from unittest.mock import patch
from intelligence import interaction as I, consciousness as C, motion_agency as MA
from vision import active_speaker as AS, face_expression as FE
import config


class HandoffTests(unittest.TestCase):
    def test_no_landmarks_still_records_faces_for_capture_continuity(self):
        AS.reset()
        self.addCleanup(AS.reset)
        face = {'id': 'track1', 'person_db_id': 1, 'face_id': 'Bret',
                'face_visible': True, 'face_box': (100, 100, 100, 100)}
        clock = [10.]
        with patch.object(FE, 'detect_expressions', return_value=[]), \
             patch.object(FE, 'merge_expressions_into_world_state'), \
             patch.object(I.world_state, 'get', side_effect=lambda key: [face] if key == 'people' else {}), \
             patch.object(I.world_state, 'mutate'), \
             patch.object(AS.time, 'time', return_value=500.), \
             patch.object(AS.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(config, 'ACTIVE_SPEAKER_ENABLED', True):
            for tick in (10., 10.3, 10.6, 10.9):
                clock[0] = tick
                FE.process_frame(None)
        rows = AS.evidence_between(10., 11.)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['person_db_id'] is None for r in rows), 'no invented mouth-motion speaker')
        with patch.object(I, '_utterance_observations', {'visual': rows}), \
             patch.object(I, '_last_scan_secs', {'voiced': 1.71}), \
             patch.object(I, '_last_scan_windows', []), \
             patch.object(I, '_last_scan_ranked', [(1, 'Bret', .484, 1)]), \
             patch.object(I.speaker_id, 'active_backend', return_value='campplus'), \
             patch.object(I, '_current_turn_speaker_evidence', {}), \
             patch.object(I.time, 'monotonic', return_value=20.):
            verdict = I._resolve_turn_attribution(
                turn_id=1, text="Do you know who's speaking?", text_input=False,
                raw_best_id=1, raw_best_name='Bret', speaker_score=.484, speaker_margin=1.484,
                required_margin=.07, accept_tier='known_floor', identity_resolution=None,
                person_id=None, person_name=None, off_camera_unknown=True,
                visible_known_ids=[1], bearing_match=None, engaged={'person_id': 1},
                previous_speaker={'person_id': 1, 'at': 9., 'verified_at': 9.})
        self.assertEqual(verdict.person_id, 1)
        self.assertFalse(verdict.learning_allowed)

    def test_come_gaze_follows_same_track_when_recognition_changes(self):
        intent = {'person_id': 1, 'track_id': 'person_1', 'unknown_voice': False}
        for pid in (None, 1, None):
            candidate = {'person_id': pid, 'track_id': 'person_1', 'area': 100}
            self.assertTrue(C._candidate_matches_speaker_gaze(candidate, intent))
            self.assertIs(C._speaker_gaze_candidate([candidate], intent), candidate)
            with patch.object(C, '_face_tracking_lock', candidate):
                self.assertTrue(C._speaker_gaze_lock_matches_intent(intent))
        self.assertFalse(C._candidate_matches_speaker_gaze(
            {'person_id': 4, 'track_id': 'person_1'}, intent))
        self.assertFalse(C._candidate_matches_speaker_gaze(
            {'person_id': None, 'track_id': 'other'},
            {'person_id': None, 'track_id': 'guest', 'unknown_voice': True}))
        # An anonymous caller becoming recognized should satisfy the same gaze.
        self.assertTrue(C._candidate_matches_speaker_gaze(
            {'person_id': 1, 'track_id': 'guest'},
            {'person_id': None, 'track_id': 'guest', 'unknown_voice': True}))

    def test_come_command_defers_identity_even_when_legacy_parser_misses(self):
        with patch.object(I.command_parser, 'parse', return_value=None):
            self.assertTrue(I._turn_should_defer_identity_prompts('Come here.'))

    def test_recent_conversation_blocks_new_stranger_prompt(self):
        with patch.object(C, '_known_face_recently_locked', return_value=False), \
             patch.object(C, 'get_recent_engagement', return_value={'person_id': 1}), \
             patch.object(MA, 'requested_come_active', return_value=False), \
             patch.object(C, '_can_proactive_speak') as speak:
            C._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
        speak.assert_not_called()

    def test_over_here_cannot_turn_away_from_visible_person(self):
        with patch.object(MA, 'requested_come_active', return_value=False), \
             patch.object(I.world_state, 'get', return_value=[
                 {'face_visible': True, 'id': 'unidentified', 'face_box': (0, 0, 100, 100)}]):
            self.assertEqual(MA._wake_orientation_guard('over_here'), 'on_camera')

    def test_known_voice_command_reaches_motion_with_old_name_prompt_pending(self):
        import numpy as np
        from contextlib import ExitStack
        face = {'id': 'track1', 'person_db_id': 1, 'face_id': 'Bret',
                'face_visible': True, 'face_box': (400, 200, 100, 100)}
        old_people = I.world_state.get('people')
        self.addCleanup(I.world_state.update, 'people', old_people)
        I.world_state.update('people', [face])
        with ExitStack() as stack:
            logs = stack.enter_context(self.assertLogs(I._log, level='INFO'))
            for name, value in {
                '_shutdown_requested': False, '_looks_like_own_echo': False,
                '_game_suppresses_conversation': False, '_audio_group_chatter_active': False,
                '_handle_router_motion_action': 'On my way.',
                '_speak_blocking': None,
            }.items():
                stack.enter_context(patch.object(I, name, return_value=value))
            stack.enter_context(patch.object(I.motion_controller, 'available', return_value=True))
            gaze = stack.enter_context(patch.object(C, 'note_speaker_gaze_intent'))
            motion = stack.enter_context(patch.object(I, '_handle_router_motion_action', return_value='On my way.'))
            stack.enter_context(patch.object(I, '_identity_prompt_until', I.time.monotonic()+30.))
            stack.enter_context(patch.object(I, '_pending_offscreen_identify', None))
            stack.enter_context(patch.object(I, '_last_scan_secs', {'voiced': 1.2, 'buffer': 4.}))
            stack.enter_context(patch.object(I, '_last_scan_windows', []))
            stack.enter_context(patch.object(I, '_last_scan_ranked', [(1, 'Bret', .636, 1)]))
            stack.enter_context(patch.object(I, '_utterance_observations', {'visual': []}))
            stack.enter_context(patch.object(I, '_last_confident_voice_at', {}))
            stack.enter_context(patch.object(I, '_last_speaker_turn', None))
            stack.enter_context(patch.object(I.speaker_id, 'active_backend', return_value='campplus'))
            stack.enter_context(patch.object(I.people_memory, 'get_person', return_value={'id': 1, 'name': 'Bret'}))
            stack.enter_context(patch.object(I.conv_log, 'log_heard'))
            stack.enter_context(patch.object(I.conv_log, 'log_rex'))
            stack.enter_context(patch.object(C, 'consume_identity_prompt_request', return_value=False))
            I._handle_speech_segment(np.ones(64000, dtype=np.float32), transcribed_text='Come here.',
                                     raw_best_id_override=1, raw_best_name_override='Bret',
                                     speaker_score_override=.636)
            self.assertEqual(motion.call_count, 1, "\n".join(logs.output))
            self.assertEqual(I._identity_prompt_until, 0.)
            self.assertIsNotNone(I._last_speaker_turn['verified_at'])
            self.assertIn(1, I._last_confident_voice_at)
            gaze.assert_called_once()
            self.assertEqual(gaze.call_args.args[0], 1)
            self.assertFalse(gaze.call_args.kwargs['unknown_voice'])

    def test_cleared_prompt_cannot_reopen_from_late_playback_callback(self):
        import threading
        from state import State
        with patch.object(C, '_known_face_recently_locked', return_value=False), \
             patch.object(C, 'get_recent_engagement', return_value=None), \
             patch.object(MA, 'requested_come_active', return_value=False), \
             patch.object(C, '_can_proactive_speak', return_value=True), \
             patch.object(C, '_pending_identity_prompt', threading.Event()), \
             patch.object(C, '_identity_prompt_in_flight', threading.Event()), \
             patch.object(C, '_identity_prompt_generation', 0), \
             patch.object(C, '_identity_prompt_reply_until', 0.), \
             patch.object(C, '_last_identity_prompt_at', 0.), \
             patch.object(C, '_solo_unknown_since', 1.), \
             patch.object(C.time, 'monotonic', return_value=100.), \
             patch.object(C.state_module, 'get_state', return_value=State.ACTIVE), \
             patch.object(config, 'IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE', True), \
             patch.object(C, '_speak_async', return_value=True) as submit:
            C._maybe_prompt_unknown_identity(unknown_count=1, known_unique=[])
            submit.assert_called_once()
            callback = submit.call_args.kwargs['on_done']
            C.clear_pending_identity_prompts(reason='authoritative_known_speaker')
            self.assertFalse(submit.call_args.kwargs['still_valid']())
            callback()
            self.assertFalse(C._pending_identity_prompt.is_set())
            self.assertEqual(C._identity_prompt_reply_until, 0.)
