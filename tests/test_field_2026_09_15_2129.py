"""Field: continuous visible caller, noisy voice/DOA, and 'keep going'."""
import unittest
from unittest.mock import patch
from intelligence import attribution, motion_agency as A
from tests.test_solo_conversation_2026_09_15 import evidence


class LocationTests(unittest.TestCase):
    def setUp(self):
        self.person={'id':'person_1','person_db_id':1,'face_visible':True,
                     'face_box':(900,400,100,100)}
        self.ev=evidence(raw_best_id=3,raw_best_name="T'Joy",raw_best_score=.576,
            margin=.064,scoreboard=[(3,"T'Joy",.576,2),(1,'Bret',.512,1)],
            words=2,previous_speaker_pid=None,bearing_contradiction=True).as_dict()

    def test_visible_partner_is_destination_without_certifying_voice(self):
        with patch.object(A,'_come_bearing_deg',return_value=21.):
            got=A._visible_come_requester({'people':[self.person]},None,self.ev,75.)
        self.assertIs(got,self.person)
        verdict=attribution.resolve_authoritative(evidence(**{
            k:v for k,v in self.ev.items() if k in attribution.UtteranceEvidence.__dataclass_fields__}))
        self.assertIsNone(verdict.person_id)
        self.assertFalse(verdict.learning_allowed)

    def test_another_person_or_decisive_voice_does_not_inherit_partner(self):
        cases=[{'mixed_speakers':True}, {'engaged_pid':None},
               {'raw_best_score':.85}, {'face_observations':[]},
               {'visual_latch_pid':2}, {'bearing_selected_pid':2}]
        with patch.object(A,'_come_bearing_deg',return_value=21.):
            for changes in cases:
                with self.subTest(changes=changes):
                    self.assertIsNone(A._visible_come_requester({'people':[self.person]},None,
                        dict(self.ev,**changes),75.))
            other=dict(self.person,id='person_2',person_db_id=2)
            self.assertIsNone(A._visible_come_requester({'people':[self.person,other]},None,self.ev,75.))


class CloserTests(unittest.TestCase):
    def setUp(self):
        from tests.test_come_arrival import MacTransportTests
        MacTransportTests.setUp(self)
        self.stack.enter_context(patch.object(A,'no_drive_room',return_value=None))
        self.stack.enter_context(patch.object(A,'cancel_requested_come'))
        self.stack.enter_context(patch('intelligence.consciousness.note_speaker_gaze_intent'))
        self.mc._last_come_detail={'owner':'mac','at':100.,'person_id':1,
            'track_id':'track','can_continue':True,'reason':'front stand-off reached; caller range uncertain'}

    def test_keep_going_uses_same_target_and_bounded_slow_approach(self):
        from intelligence import interaction as I
        with patch.object(I,'_speak_blocking'),patch.object(I,'_router_audit_note_fast_local_action'):
            reply=I._explicit_motion_takeover('Keep going.',person_id=None)
        self.assertEqual(reply,'Okay, a little closer.')
        plan=self.mc._host_approach['plan']
        self.assertEqual(self.mc._host_approach['track_id'],'track')
        self.assertEqual(plan.speed,.08)
        self.assertEqual(plan.max_travel,.20)
        self.assertEqual(plan.stop_at,.75)

    def test_logged_stop_clearance_allows_only_slow_closer_step(self):
        from tests.test_come_arrival import telemetry
        row=telemetry(100.,tof_mm={'fl':953,'fr':842,'fl_radial':1041,'fr_radial':-1})
        with patch.object(self.mc.motion,'telemetry',return_value=row):
            self.assertEqual(self.mc.continue_come(),'Okay, a little closer.')
            self.mc._heartbeat_tick()
        commands=[r for r in self.sent if r.get('lin',0)>0]
        self.assertTrue(commands)
        self.assertLessEqual(max(r['lin'] for r in commands),.08)

    def test_lost_target_does_not_move_or_ask_for_name(self):
        self.scene['people']=[]
        reply=self.mc.continue_come()
        self.assertIn('lost sight',reply)
        self.assertNotIn('speaking',reply)
        self.assertEqual(self.sent,[])

    def test_close_sensor_still_holds(self):
        with patch.object(self.mc.motion,'telemetry',return_value={'rx_monotonic':100.,
                'tof_mm':{'fl':400,'fr':400,'fl_radial':400,'fr_radial':-1}}):
            self.assertIn('limit',self.mc.continue_come())
        self.assertEqual(self.sent,[])

    def test_expired_or_stopped_approach_is_not_restarted(self):
        self.clock[0]=146.
        self.assertIsNone(self.mc.continue_come())
        self.clock[0]=100.
        self.mc.stop()
        count=len(self.sent)
        self.assertIsNone(self.mc.continue_come())
        self.assertEqual(len(self.sent),count)
