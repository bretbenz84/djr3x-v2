"""Conversational enrollment, contamination rejection, original audio and lifecycle."""
from contextlib import ExitStack
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import config
from intelligence.voice_learning import Learner, Policy, Sample, position_for, unit


def vector(index=0, angle=0):
    v = np.zeros(192, dtype=np.float32)
    v[index] = np.cos(angle)
    v[index+1] = np.sin(angle)
    return v


def face(pid, x=100):
    return dict(person_db_id=pid, face_visible=True, face_id=str(pid),
                track_id=f'track-{pid}', face_box=[x, 10, 80, 80])


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.hold = Mock(side_effect=lambda duration: self.now)
        self.release = Mock()
        self.l = Learner(clock=lambda: self.now, hold=self.hold, release=self.release)
        self.seq = 0

    def sample(self, **kwargs):
        self.seq += 1
        self.now += 5
        s = Sample(np.ones(64000, dtype=np.float32)*(.02 + self.seq/1000), vector(), 16000,
                   3., 'I went fishing at the lake yesterday.', self.now-4, self.now-1,
                   faces=[[face(1), face(2, 500)]], metadata={})
        return replace(s, **kwargs)

    def seed(self, **kwargs):
        self.assertTrue(self.l.seed(2, 'Jeff Benziger', 1,
                                   self.sample(text='Jeff Benziger', voiced=.6, **kwargs), explicit=True))

    def test_name_then_three_long_turns_without_direction_or_mouth(self):
        self.seed()
        self.assertFalse(self.l.ready())
        for n in range(3):
            self.assertTrue(self.l.observe(self.sample(), 1))
            self.assertEqual(self.l.ready(), n == 2)
        self.assertEqual(len(self.l.pending.samples), 3)

    def test_three_turns_must_also_have_enough_voiced_speech(self):
        self.seed()
        for _ in range(3): self.l.observe(self.sample(voiced=2.), 1)
        self.assertFalse(self.l.ready())
        self.l.observe(self.sample(voiced=2.), 1)
        self.assertTrue(self.l.ready())

    def test_foreign_voice_never_changes_anchor_or_samples(self):
        self.seed()
        original = self.l.pending.anchor.copy()
        self.assertFalse(self.l.observe(self.sample(embedding=vector(5), ranked=[(1,'Bret',.88,1)]), 1))
        np.testing.assert_array_equal(original, self.l.pending.anchor)
        self.assertFalse(self.l.pending.samples)

    def test_distinct_direction_and_voice_rejected_then_owner_can_continue(self):
        self.seed()
        self.assertTrue(self.l.observe(self.sample(bearing=-20.), 1))
        self.assertFalse(self.l.observe(self.sample(bearing=30., embedding=vector(5)), 1))
        self.assertFalse(self.l.observe(self.sample(bearing=30.), 1))
        self.assertTrue(self.l.observe(self.sample(bearing=-18.), 1))
        self.assertEqual(len(self.l.pending.samples), 2)

    def test_circular_bearings_cross_zero(self):
        self.seed()
        self.assertTrue(self.l.observe(self.sample(bearing=179.),1))
        self.assertTrue(self.l.observe(self.sample(bearing=-178.),1))

    def test_chain_cannot_walk_away_from_original_anchor(self):
        self.seed()
        self.assertTrue(self.l.observe(self.sample(embedding=vector(angle=.5)),1))
        self.assertTrue(self.l.observe(self.sample(embedding=vector(angle=1.0)),1))
        self.assertFalse(self.l.observe(self.sample(embedding=vector(angle=1.4)),1))
        self.assertEqual(self.l.last_reason, 'anchor_disagrees')

    def test_short_nonmatching_anchor_does_not_move_to_latest_clip(self):
        self.seed()
        for _ in range(3): self.assertFalse(self.l.observe(self.sample(embedding=vector(5)),1))
        self.assertFalse(self.l.ready())

    def test_overlapping_and_untrusted_captures_never_count(self):
        self.seed()
        for kwargs in ({'mixed':True}, {'trusted':False}, {'direction_conflict':True}, {'embedding':np.full(192,np.nan)}):
            self.assertFalse(self.l.observe(self.sample(**kwargs),1))
        self.assertFalse(self.l.pending.samples)

    def test_duplicate_audio_and_padding_cannot_complete_enrollment(self):
        self.seed()
        s = self.sample()
        self.assertTrue(self.l.observe(s,1))
        for _ in range(4): self.assertFalse(self.l.observe(s,1))
        for _ in range(4): self.l.observe(self.sample(voiced=.3, text='yeah'),1)
        self.assertFalse(self.l.ready())

    def test_known_face_confirmation_is_positional_and_temporary(self):
        self.assertTrue(self.l.request(2,'Jeff Benziger',1,[face(1),face(2,500)]))
        self.assertIn('on my right',self.l.question())
        self.assertNotIn('repeat',self.l.question())
        self.l.prepare_question()
        self.l.question_spoken(True)
        self.assertTrue(self.l.confirm(self.sample(text='Yes',voiced=.4)))
        self.assertFalse(self.l.ready())
        self.assertEqual(self.l.pending.person_id,2)

    def test_foreign_yes_unrelated_sentence_or_unspoken_question_does_not_confirm(self):
        self.l.request(2,'Jeff Benziger',1,[face(2)])
        self.assertFalse(self.l.confirm(self.sample(text='yes')))
        self.l.prepare_question(); self.l.question_spoken(True)
        self.assertFalse(self.l.confirm(self.sample(text='yes', ranked=[(1,'Bret',.8,1)])))
        self.assertFalse(self.l.confirm(self.sample(text="Says it's a pretty good pizza.")))
        self.assertFalse(self.l.pending.confirmed)

    def test_no_cancels_and_interrupted_question_cannot_capture(self):
        self.l.request(2,'Jeff',1,[face(2)])
        self.l.prepare_question(); self.l.question_spoken(True)
        self.assertFalse(self.l.confirm(self.sample(text='No, that is Bret.')))
        self.assertIsNone(self.l.pending)

    def test_natural_confirmation_variants_do_not_require_a_script(self):
        for answer in ("Yep, that's me.", "That's right.", "Yeah, I'm Jeff.", "Yes, it is, Rex."):
            with self.subTest(answer=answer):
                self.l.reset(); self.l.request(2,'Jeff Benziger',1,[face(2)])
                self.l.prepare_question(); self.l.question_spoken(True)
                self.assertTrue(self.l.confirm(self.sample(text=answer,faces=[[face(2)]],voiced=.5)))
        self.l.reset(); self.l.request(2,'Jeff',1,[face(2)])
        self.l.prepare_question(); self.l.question_spoken(False)
        self.assertIsNone(self.l.pending)

    def test_changed_face_track_and_exchanged_positions_pause_collection(self):
        self.l.request(2,'Jeff',1,[face(1),face(2,500)])
        self.l.prepare_question(); self.l.question_spoken(True)
        self.assertTrue(self.l.confirm(self.sample(text='yes')))
        self.assertFalse(self.l.observe(self.sample(faces=[[face(2,20),face(1,500)]]),1))
        self.assertFalse(self.l.observe(self.sample(faces=[[face(1),dict(face(2,500),track_id='different')]]),1))

    def test_expiry_releases_even_without_new_speech_and_session_clears(self):
        self.seed(); self.now+=61; self.l.tick(1)
        self.release.assert_called()
        self.assertEqual(self.l.last_reason,'paused')
        self.l.tick(2)
        self.assertIsNone(self.l.pending)

    def test_only_one_person_collects_at_a_time(self):
        self.seed()
        self.assertFalse(self.l.seed(1,'Bret',1,self.sample(),explicit=True))
        self.assertEqual(self.l.pending.person_id,2)

    def test_settling_frames_excluded(self):
        self.hold.side_effect=lambda duration:self.now+10
        self.seed()
        self.assertFalse(self.l.observe(self.sample(),1))
        self.assertEqual(self.l.last_reason,'servo_settling')

    def test_three_face_positions(self):
        faces=[face(1),face(2,500),face(3,900)]
        self.assertEqual([position_for(i,faces) for i in (1,2,3)],['on my left','in the middle','on my right'])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction as I
        from intelligence.voice_learning_runtime import Runtime
        from memory import database as db
        from audio import speaker_id, voice_score
        from setup_assets import DB_SCHEMA
        self.I=I
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.tmp=self.stack.enter_context(tempfile.TemporaryDirectory())
        self.path=Path(self.tmp)/'people.db'
        with sqlite3.connect(self.path) as c:
            c.executescript(DB_SCHEMA)
            c.executemany('INSERT INTO people(id,name) VALUES (?,?)',[(1,'Bret Benziger'),(2,'Jeff Benziger')])
            c.execute("INSERT INTO biometrics(person_id,type,encoding) VALUES (1,'voice_campplus_zh_en_v1',?)",(vector(5).tobytes(),))
        self.stack.enter_context(patch.object(db,'_DB_FILE',self.path))
        self.stack.enter_context(patch.object(speaker_id,'_active_backend','campplus'))
        self.stack.enter_context(patch.object(voice_score,'_active_backend','campplus'))
        self.now=10000.
        self.stack.enter_context(patch('time.monotonic',side_effect=lambda:self.now))
        from hardware import servos
        self.stack.enter_context(patch.object(servos,'hold_voice_enrollment_mic',return_value=0.))
        self.stack.enter_context(patch.object(servos,'release_voice_enrollment_hold'))
        self.stack.enter_context(patch.object(servos,'manual_override_enabled',return_value=False))
        for name,result in [('_shutdown_requested',False),('_game_suppresses_conversation',False),('_turn_transcript_trusted',True)]:
            self.stack.enter_context(patch.object(I,name,return_value=result))
        self.stack.enter_context(patch.object(I.echo_cancel,'last_playback_ended_at',return_value=0.))
        from intelligence import motion_controller,voice_learning
        self.stack.enter_context(patch.object(motion_controller,'is_moving',return_value=False))
        self.stack.enter_context(patch.object(voice_learning,'_last_mic_motion_at',0.))
        self.stack.enter_context(patch.object(I,'_last_voice_bearing',None))
        self.stack.enter_context(patch.object(I,'_last_scan_windows',[]))
        self.stack.enter_context(patch.object(I,'_last_scan_ranked',[]))
        self.stack.enter_context(patch.object(I,'_utterance_observations',{}))
        self.faces=[face(1),face(2,500)]
        original_get = I.world_state.get
        self.stack.enter_context(patch.object(I.world_state,'get',side_effect=lambda key: self.faces if key=='people' else original_get(key)))
        self.embedding=vector()
        self.stack.enter_context(patch.object(speaker_id,'get_embedding',side_effect=lambda a:self.embedding))
        self.voiced=3.
        self.stack.enter_context(patch.object(speaker_id,'voiced_secs',side_effect=lambda a:self.voiced))
        self.r=Runtime(I); self.r.learner.clock=lambda:self.now
        self.stack.enter_context(patch.object(I,'_voice_learning_runtime',self.r))
        self.stack.enter_context(patch.object(I,'_voice_learner',self.r.learner))
        self.addCleanup(self.r.learner.reset)
        self.seq=0

    def prepare(self, mouth=999):
        self.seq+=1; self.now+=5
        audio=np.ones(64000,dtype=np.float32)*(.02+self.seq/1000)
        self.I._utterance_observations=dict(started_at=self.now-4,ended_at=self.now-1,
            visual=[dict(monotonic_at=t,faces=self.faces,person_db_id=mouth,confidence=1.)
                    for t in (self.now-4,self.now-3,self.now-2,self.now-1)])
        return audio

    def turn(self,text='I went fishing at the lake yesterday.', *, mouth=999):
        return self.r.process(self.prepare(mouth),text)

    def confirm(self):
        question=self.turn('Hey Rex, how are you?')
        self.assertIn('Jeff',question)
        self.r.learner.prepare_question(); self.r.learner.question_spoken(True)
        self.voiced=.4
        self.assertIn('Thanks, Jeff',self.turn('Yes'))
        self.voiced=3.

    def count(self,pid):
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT count(*) FROM biometrics WHERE person_id=? AND type='voice_campplus_zh_en_v1'",(pid,)).fetchone()[0]

    def test_two_person_no_doa_confirmation_to_actual_sqlite_and_recognition(self):
        from memory import voice_recordings
        from audio import speaker_id
        self.confirm()
        self.assertEqual(self.count(2),0)
        # Bret speaking midway cannot enter Jeff's chain.
        self.I._last_scan_ranked=[(1,'Bret Benziger',.88,1)]
        self.embedding=vector(5); self.turn('I am Bret, just checking in.')
        self.assertEqual(self.count(2),0)
        self.embedding=vector(); self.I._last_scan_ranked=[]
        for _ in range(3): self.turn()
        self.assertEqual(self.count(2),1)  # identical model outputs deduplicated
        self.assertEqual(self.count(1),1)
        clips=voice_recordings.list_recordings(2)
        self.assertEqual(len(clips),3)
        rate,audio=voice_recordings.decode(clips[0])
        self.assertEqual(rate,16000); self.assertEqual(audio.dtype,np.float32)
        self.assertEqual(speaker_id.rank_embedding(vector())[0][0],2)
        self.assertTrue(self.r.saved)
        self.assertIsNone(self.r.learner.pending)

    def test_mouth_identity_has_no_effect_on_enrollment(self):
        self.confirm()
        for mouth in (1,None,999): self.turn(mouth=mouth)
        self.assertTrue(self.r.saved)

    def test_storage_failure_is_atomic_and_not_a_success(self):
        self.confirm(); self.turn(); self.turn()
        # SQLite failure on recording insert must roll back the biometric too.
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TRIGGER fail_recording BEFORE INSERT ON voice_recordings BEGIN SELECT RAISE(ABORT,'disk full'); END")
        self.turn()
        self.assertEqual(self.count(2),0)
        self.assertFalse(self.r.saved)
        self.assertIsNone(self.r.learner.pending)

    def test_original_audio_survives_and_forgetting_deletes_clips(self):
        from memory import voice_recordings, admin
        self.confirm()
        for _ in range(3): self.turn()
        rows=voice_recordings.list_recordings(2)
        self.assertEqual(json.loads(rows[0]['metadata'])['source'],'conversational_enrollment')
        rate,a=voice_recordings.decode(rows[0])
        np.testing.assert_array_equal(a,np.ones(64000,dtype=np.float32)*.023)
        self.assertTrue(admin.clear_biometrics(2,'voice'))
        self.assertEqual(voice_recordings.list_recordings(2),[])
        self.assertEqual(self.count(1),1)

    def test_game_cancels_without_saving(self):
        self.confirm(); self.turn()
        self.I._game_suppresses_conversation.return_value=True
        self.turn()
        self.assertIsNone(self.r.learner.pending)
        self.assertEqual(self.count(2),0)

    def test_no_interval_camera_evidence_does_not_guess_in_a_group(self):
        self.turn()
        self.I._utterance_observations['visual']=[]
        self.assertIsNone(self.r._sample(np.ones(64000,dtype=np.float32),'yes'))

    def test_name_answer_only_starts_proposal_never_a_print(self):
        self.turn('My name is Jeff Benziger.')
        self.assertTrue(self.r.learner.pending.confirmed)
        self.assertEqual(self.count(2),0)

    def test_typed_text_and_preplayback_audio_never_seed(self):
        self.assertIsNone(self.r.process(None,'My name is Jeff Benziger.'))
        self.I.echo_cancel.last_playback_ended_at.return_value=self.now+20
        self.turn('My name is Jeff Benziger.')
        self.assertIsNone(self.r.learner.pending)

    def test_actual_speech_pipeline_confirmation_collection_and_storage(self):
        from audio import speaker_id
        I=self.I
        for name, result in {
            '_looks_like_own_echo':False, '_looks_like_third_party_crosstalk':False,
            '_speak_blocking':True, '_should_reprompt_low_trust':False,
            '_split_audio_speakers':None, '_audio_group_chatter_active':False,
            '_post_response':None,
        }.items(): self.stack.enter_context(patch.object(I,name,return_value=result))
        self.stack.enter_context(patch.object(speaker_id,'window_evidence',return_value=[]))
        self.stack.enter_context(patch.object(I.consciousness,'note_speaker_gaze_intent'))
        self.stack.enter_context(patch.object(I.consciousness,'consume_identity_prompt_request',return_value=False))
        self.stack.enter_context(patch.object(I,'_register_rex_utterance'))
        self.stack.enter_context(patch.object(I.conv_log,'log_rex'))
        self.stack.enter_context(patch.object(I.conv_log,'log_heard'))
        self.stack.enter_context(patch.object(I.llm,'get_response',return_value='Sounds good.'))
        self.stack.enter_context(patch.object(I.llm,'classify_surprise',return_value=False))
        self.stack.enter_context(patch.object(I.llm,'classify_self_emotion',return_value='neutral'))
        self.stack.enter_context(patch.object(I.empathy,'classify_affect',return_value=None))
        self.stack.enter_context(patch.object(I,'_reply_token_stream',side_effect=lambda *a,**k:iter(['That sounds like a good afternoon.'])))
        import threading
        done=threading.Event(); done.set()
        self.stack.enter_context(patch.object(I.speech_queue,'enqueue',return_value=done))
        self.stack.enter_context(patch.object(I.speech_queue,'enqueue_audio_file',return_value=done))
        from memory import conversations
        self.stack.enter_context(patch.object(conversations,'_log_turn'))
        # Name question and yes go through real ASR-result + speaker scoring and
        # real speech handler; no typed-GUI shortcut or direct learner calls.
        I._handle_speech_segment(self.prepare(),eager_transcript='Hey Rex, how are you?')
        self.assertIsNotNone(self.r.learner.pending.asked_at)
        self.voiced=.4
        I._handle_speech_segment(self.prepare(),eager_transcript='Yes')
        self.assertTrue(self.r.learner.pending.confirmed)
        self.assertEqual(self.count(2),0)
        self.voiced=3.
        for n in range(3):
            I._handle_speech_segment(self.prepare(),eager_transcript='I went fishing at the lake yesterday.')
            if n<2:
                self.assertEqual(self.count(2),0)
                res=I._current_turn_speaker_evidence.get('resolution') or {}
                self.assertEqual(res.get('person_id'),2)
                self.assertFalse(res.get('learning_allowed'))
        self.assertTrue(self.r.saved)
        self.assertEqual(self.count(2),1)

    def test_reembedding_uses_original_audio_and_preserves_previous_model(self):
        from tools.voice_recordings import reembed, apply_embeddings
        from memory.voice_recordings import list_recordings
        self.confirm()
        for _ in range(3): self.turn()
        records=list_recordings(2)
        embed=Mock(return_value=np.ones(256,dtype=np.float32)/16)
        results=reembed(records,embed,16000)
        self.assertEqual(len(results),3)
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM biometrics WHERE person_id=2 AND type='voice'").fetchone()[0],0)
        apply_embeddings(2,'voice',results)
        self.assertEqual(self.count(2),1)
        self.assertEqual([r['wav'] for r in records],[r['wav'] for r in list_recordings(2)])

    def test_person_merge_and_delete_include_original_recordings(self):
        from memory import people,voice_recordings
        self.confirm()
        for _ in range(3): self.turn()
        with patch.object(people,'_purge_episodes_for_person'):
            self.assertTrue(people.merge_person(1,2))
            self.assertEqual(len(voice_recordings.list_recordings(1)),3)
            self.assertEqual(voice_recordings.list_recordings(2),[])
            people.delete_person(1)
            self.assertEqual(voice_recordings.list_recordings(1),[])

    def test_new_person_name_handler_starts_chain_before_face_is_named(self):
        from memory import people
        I=self.I
        with sqlite3.connect(self.path) as c:
            c.execute('DELETE FROM people WHERE id=2')
        self.faces[:]=[face(1),dict(face(None,500),track_id='new-face')]
        self.voiced=.6
        self.turn('My name is Jeff Benziger.')
        self.assertIsNone(self.r.learner.pending)
        pid,_=people.find_or_create_person('Jeff Benziger')
        self.assertTrue(I._begin_conversational_voice_learning(pid,self.r.last_sample.audio,
                                                              source='new_person',confirmed=True))
        self.assertEqual(self.count(pid),0)
        self.faces[1].update(person_db_id=pid,face_id='Jeff Benziger')
        self.voiced=3.
        for _ in range(3): self.turn()
        self.assertTrue(self.r.saved)
        self.assertEqual(self.count(pid),1)

    def test_recording_and_print_caps_preserve_existing_voice_data(self):
        from memory import voice_recordings
        self.confirm()
        for n in range(3):
            self.embedding=vector(angle=.03*n)
            self.turn()
        original=self.count(2)
        samples=[]
        for n in range(14):
            self.embedding=vector(angle=.2+.01*n)
            self.prepare()
            samples.append(self.r._sample(np.ones(64000,dtype=np.float32)*(.1+n/100),
                                          f'Today I enjoyed a different conversation number {n}'))
        self.assertTrue(voice_recordings.save_batch(2,samples,'voice_campplus_zh_en_v1'))
        self.assertEqual(len(voice_recordings.list_recordings(2)),10)
        self.assertEqual(self.count(2),10)
        self.assertEqual(self.count(1),1)

    def test_face_history_is_independent_of_mouth_flags(self):
        from vision import face_presence
        from intelligence import consciousness
        frame=np.zeros((500,1000,3),dtype=np.uint8)
        with patch.object(consciousness,'_stop_event') as stop, \
             patch('vision.camera.get_frame',return_value=frame), \
             patch.object(consciousness,'_live_face_tracking_people',return_value=self.faces), \
             patch.object(consciousness,'_step_face_tracking'), \
             patch.object(config,'ACTIVE_SPEAKER_ENABLED',False):
            stop.is_set.side_effect=[False,True]
            consciousness._face_tracking_loop()
        rows=face_presence.between(self.now,self.now)
        self.assertTrue(rows)
        self.assertEqual(rows[-1]['faces'][1]['track_id'],'track-2')

    def test_mic_motion_invalidates_inflight_audio_and_pending_chain(self):
        from intelligence import voice_learning
        self.confirm()
        self.prepare()
        voice_learning.mic_moved('base_motion')
        self.assertIsNone(self.r.learner.pending)
        self.assertIsNone(self.r._sample(np.ones(64000,dtype=np.float32),'This was captured before movement'))

    def test_explicit_wave_cancels_but_story_about_driving_does_not(self):
        self.confirm()
        self.turn('I like to drive around the lake.')
        self.assertIsNotNone(self.r.learner.pending)
        self.turn('Please wave')
        self.assertIsNone(self.r.learner.pending)

    def test_declined_identity_never_reuses_its_provisional_samples(self):
        self.confirm(); self.turn()
        self.turn("I'm not Jeff.")
        self.assertIsNone(self.r.learner.pending)
        self.assertEqual(self.count(2),0)

    def test_brets_confident_turn_does_not_trigger_a_question_to_jeff_or_take_growth_ownership(self):
        self.I._last_scan_ranked=[(1,'Bret Benziger',.85,2)]
        self.embedding=vector(5)
        self.assertIsNone(self.turn('Hey Rex, I have a question for you.'))
        self.assertIsNone(self.r.learner.pending)

    def test_naming_anonymous_slot_cannot_create_a_second_named_voice(self):
        I=self.I
        slot=Mock(label='unknown_voice_1',signature_id=7)
        with patch.object(I,'_anonymous_speaker_slots',[slot]), \
             patch.object(I.voice_signatures,'attach_person') as attach:
            I._retire_anonymous_speaker_slot('unknown_voice_1',person_id=2,person_name='Jeff Benziger')
            attach.assert_not_called()
            self.assertEqual(I._anonymous_speaker_slots,[])

    def test_existing_named_signature_cannot_learn_an_unarchived_voice(self):
        from memory import voice_signatures as S, database as db
        S.reset_table_cache(); self.addCleanup(S.reset_table_cache)
        with patch.object(S,'_writes_suppressed',return_value=False):
            sid=S.record(vector(),label='old named voice')
            S.attach_person(sid,2)
            before=db.fetchone('SELECT embedding FROM voice_signatures_campplus WHERE id=?',(sid,))['embedding']
            S.bump(sid,vector(5))
            after=db.fetchone('SELECT embedding FROM voice_signatures_campplus WHERE id=?',(sid,))['embedding']
            self.assertEqual(before,after)


class ServoHoldTests(unittest.TestCase):
    def test_wire_commands_hold_only_hero_arm_at_safe_midpoint(self):
        from hardware import servos as S
        S.release_voice_enrollment_hold(); self.addCleanup(S.release_voice_enrollment_hold)
        with patch.object(S,'SERVOS_ENABLED',True),patch.object(S,'_program_servo_updates_blocked',return_value=False), \
             patch.object(S,'_send_command_locked',return_value=True) as wire,patch.object(S,'_record_servo_positions'):
            S.hold_voice_enrollment_mic(60)
            cfg=config.SERVO_CHANNELS['heroarm']; mid=(cfg['min']+cfg['max'])//2
            wire.reset_mock()
            S.set_servo(cfg['ch'],cfg['max'])
            self.assertEqual(wire.call_args.args[0],S._encode(S._CMD_SET_TARGET,cfg['ch'],mid))
            S._send_set_target(cfg['ch'],cfg['min'])
            self.assertEqual(wire.call_args.args[0],S._encode(S._CMD_SET_TARGET,cfg['ch'],mid))
            S.set_servos({cfg['ch']:cfg['min'],0:6000})
            self.assertEqual(S._commanded_positions[cfg['ch']],mid)
            S.release_voice_enrollment_hold()
            S.set_servo(cfg['ch'],cfg['max'])
            self.assertEqual(wire.call_args.args[0],S._encode(S._CMD_SET_TARGET,cfg['ch'],cfg['max']))

    def test_dev_mac_hold_never_touches_hardware(self):
        from hardware import servos as S
        with patch.object(S,'SERVOS_ENABLED',False),patch.object(S,'set_servo') as set_target:
            S.hold_voice_enrollment_mic(60)
            set_target.assert_not_called()


if __name__=='__main__': unittest.main()
