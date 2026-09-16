"""Field voice scores: continuity is conversational only; no new identity proof."""
import unittest
from dataclasses import replace
from unittest.mock import patch
from intelligence import attribution as A


def evidence(**kw):
    ev=A.UtteranceEvidence(started_at=10.,ended_at=12.,words=6,voiced_secs=1.86,
        raw_best_id=1,raw_best_name='Bret',raw_best_score=.462,margin=.022,
        required_margin=.07,hard_threshold=.5,known_floor=.45,
        scoreboard=[(1,'Bret',.462,1),(3,"T'Joy",.441,2)],
        visible_known_ids=[1],engaged_pid=1,allow_short_continuity=True,
        face_observations=[{'monotonic_at':t,'faces':[{'person_db_id':1,'face_visible':True}]}
                           for t in (10.,10.5,11.,11.5,12.)])
    return replace(ev,**kw)

class FieldIdentityTests(unittest.TestCase):
    def test_weak_tied_reply_keeps_conversation_but_cannot_learn(self):
        r=A.resolve_authoritative(evidence())
        self.assertEqual(r.person_id,1)
        self.assertFalse(r.as_dict()['learning_allowed'])

    def test_false_tjoy_winner_cannot_override_continuous_bret(self):
        r=A.resolve_authoritative(evidence(raw_best_id=3,raw_best_name="T'Joy",raw_best_score=.55,
            margin=.143,scoreboard=[(3,"T'Joy",.55,2),(1,'Bret',.407,1)]))
        self.assertEqual(r.person_id,1)
        self.assertFalse(r.learning_allowed)

    def test_visible_conflict_without_interval_evidence_abstains_not_tjoy(self):
        r=A.resolve_authoritative(evidence(raw_best_id=3,raw_best_name="T'Joy",raw_best_score=.55,
            margin=.143,face_observations=[]))
        self.assertIsNone(r.person_id)

    def test_first_turn_tie_can_use_continuous_face_without_learning(self):
        r=A.resolve_authoritative(evidence(engaged_pid=None,raw_best_id=7,raw_best_name='PJ',
            raw_best_score=.561,words=18,margin=.039,
            scoreboard=[(7,'PJ',.561,1),(1,'Bret',.521,1)]))
        self.assertEqual(r.person_id,1);self.assertFalse(r.learning_allowed)

    def test_other_speaker_and_insufficient_evidence_never_inherit_bret(self):
        rows=evidence().face_observations
        for change in ({'mixed_speakers':True},{'bearing_contradiction':True},
            {'bearing_selected_pid':2},{'face_observations':rows[:1]},
            {'visual_observations':[{'person_db_id':2}]},
            {'face_observations':[rows[0]]*5}, {'face_observations':[]},
            {'engaged_pid':None, 'raw_best_score':.3, 'scoreboard':[(1,'Bret',.3,1)]}, {'visible_known_ids':[1,2]},
            {'face_observations':[dict(r,faces=[{'person_db_id':1,'face_visible':True},
                {'person_db_id':None,'face_visible':True}]) for r in rows]}):
            with self.subTest(change=change):
                r=A.resolve_authoritative(evidence(**change))
                self.assertIsNone(r.person_id)

    def test_confident_other_enrolled_voice_is_not_overridden(self):
        r=A.resolve_authoritative(evidence(raw_best_id=3,raw_best_name="T'Joy",raw_best_score=.85,
                                          margin=.4))
        self.assertEqual(r.person_id,3)

    def test_full_identity_complaint_is_not_an_incomplete_turn(self):
        from intelligence import turn_completion as T
        self.assertIsNone(T.classify("You don't know who I am."))
        self.assertIsNotNone(T.classify('I am going to'))

    def test_generic_what_and_name_are_not_identity_check_requests(self):
        from intelligence import tool_router as T
        for text in ('What?', 'Bret Hendricks.', "I don't get it."):
            self.assertFalse(T.invites_identity_check(text))
        for text in ('Who am I?', "You don't know who I am.", "What's my name?"):
            self.assertTrue(T.invites_identity_check(text))

class ActualSpeechPipelineTests(unittest.TestCase):
    def setUp(self):
        from tests import test_voice_learning as V
        V.RuntimeTests.setUp(self)
        self._mock_speech_pipeline=lambda:V.RuntimeTests._mock_speech_pipeline(self)
        self.prepare=lambda **kw:V.RuntimeTests.prepare(self,**kw)
        self.count=lambda pid:V.RuntimeTests.count(self,pid)

    def test_two_visible_unnamed_people_still_allow_addressee_judgment(self):
        from memory import conversations
        I=self.I
        with patch.object(conversations,'get_session_transcript',return_value=[]), \
             patch.object(I,'_utterance_observations',{'faces':[{'faces':[
                 {'face_visible':True},{'face_visible':True}]}]}):
            hint=I._assess_turn_addressee('Are you watching both movies?',person_id=None,
                text_input=False,recent_engagement={'person_id':1})
        self.assertTrue(hint.offer_stay_quiet)

    def test_weak_reply_after_anonymous_labels_answers_without_identity_interruption(self):
        import numpy as np
        from tests.test_voice_learning import face,vector
        from intelligence import dialogue_act,conversation_state
        from memory import conversations
        self._mock_speech_pipeline()
        I=self.I
        dialogue_act.clear();conversations.clear_transcript();conversation_state.clear()
        self.addCleanup(dialogue_act.clear);self.addCleanup(conversations.clear_transcript)
        self.addCleanup(conversation_state.clear)
        self.stack.enter_context(patch.object(I.consciousness,'get_recent_engagement',
            return_value={'person_id':1,'name':'Bret Benziger'}))
        for name,value in (('_last_speaker_turn',None),('_pending_offscreen_identify',None),
                           ('_last_confident_voice_at',{})):
            self.stack.enter_context(patch.object(I,name,value))
        self.faces=[face(1)];self.embedding=vector(5,angle=np.arccos(.30));self.voiced=1.86
        for speaker in ('Bret Benziger','unknown_voice_1','Unknown','unknown_voice_4'):
            conversations.add_to_transcript(speaker,'Earlier conversation')
        audio=self.prepare(mouth=None)
        I._utterance_observations['faces']=I._utterance_observations.pop('visual')
        I._handle_speech_segment(audio,eager_transcript="You haven't been up that long.")
        r=I._current_turn_speaker_evidence['resolution']
        self.assertEqual(r.get('person_id'),1,r)
        self.assertFalse(r['learning_allowed'])
        self.assertFalse(I._current_turn_addressee.offer_stay_quiet)
        self.assertIsNone(I._pending_offscreen_identify)
        I._reply_token_stream.assert_called()
        self.assertEqual(self.count(1),1)
        self.assertIsNone(self.r.learner.pending)


class DelayedGreetingTests(unittest.TestCase):
    def test_existing_conversation_consumes_deferred_startup_without_greeting(self):
        from contextlib import ExitStack
        from collections import deque
        from unittest.mock import Mock
        from intelligence import consciousness as C
        with ExitStack() as stack:
            for name in ('_last_seen','_pending_departure_keys','_first_missing_at',
                         '_confirmed_absent_at','_first_sight_seen_at','_visit_started_at',
                         '_reported_departure_at'):
                stack.enter_context(patch.object(C,name,{}))
            for name in ('_visible_people','_greeted_this_session'):
                stack.enter_context(patch.object(C,name,set()))
            stack.enter_context(patch.object(C,'_group_turn_speaker_times',{1:deque([100.])}))
            stack.enter_context(patch.object(C.time,'monotonic',return_value=600.))
            stack.enter_context(patch.object(C,'_presence_tracking_map',return_value={1:('Bret Benziger',1)}))
            stack.enter_context(patch.object(C,'_camera_pose',return_value=None))
            fire=stack.enter_context(patch.object(C,'_should_fire_presence',return_value=True))
            speak=stack.enter_context(patch.object(C,'_generate_and_speak_presence'))
            C._step_presence_tracking({'people':[]},Mock())
            fire.assert_not_called()
            speak.assert_not_called()
            self.assertEqual(C._last_seen[1],600.)
            self.assertIn(1,C._visible_people)
            self.assertNotIn(1,C._greeted_this_session)
