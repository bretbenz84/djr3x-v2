"""2026-09-06: a settled identity prompt must not enroll the next camping plan."""
import time
import unittest
from unittest import mock
import numpy as np
from intelligence import interaction as I
from memory.name_validation import normalize_person_name

class CampingIdentityRegression(unittest.TestCase):
    def test_name_reply_settles_prompt_before_ack_playback(self):
        with mock.patch.object(I, '_identity_prompt_until', time.monotonic()+20), \
                mock.patch.object(I, '_resolve_name_update_target', return_value=(1,'Bret Benziger')), \
                mock.patch.object(I.people_memory, 'find_person_by_name', return_value={'id':1,'name':'Bret Benziger'}), \
                mock.patch.object(I, '_speak_blocking') as speak, \
                mock.patch.object(I.consciousness, 'clear_pending_identity_prompts'):
            def heard_ack(*args, **kwargs):
                self.assertEqual(I._identity_prompt_until, 0)
                return True
            speak.side_effect = heard_ack
            result = I._handle_name_update_request('My name is Bret Benziger.',1,'Bret Benziger')
            self.assertIn('Already got you',result)
            self.assertIsNone(I._extract_introduced_name("I'm going camping tomorrow. I'm bringing my dog too.", allow_bare_name=I._identity_prompt_until>time.monotonic()))

    def test_camping_is_never_a_name_even_with_stale_prompt(self):
        for text in ("I'm going camping tomorrow. I'm bringing my dog too.",
                     "I am headed home.", "I'm bringing my dog too.", "I'm feeling tired."):
            for prompted in (False, True):
                with self.subTest(text=text,prompted=prompted):
                    self.assertIsNone(I._extract_introduced_name(text,allow_bare_name=prompted))

    def test_storage_rejects_activity_phrase(self):
        self.assertIsNone(normalize_person_name('Going Camping Tomorrow'))
        self.assertIsNone(normalize_person_name('Bringing My Dog'))

    def test_real_introductions_and_prompted_names_still_work(self):
        for text, expected in [('My name is Bret Benziger.','Bret Benziger'),("I'm Jeremy Thomas.",'Jeremy Thomas'),('Bret Benziger.','Bret Benziger'),('J T','JT')]:
            self.assertEqual(I._extract_introduced_name(text,allow_bare_name=True),expected)

    def test_bogus_name_cannot_reach_database_or_biometrics(self):
        with mock.patch.object(I, '_turn_transcript_trusted', return_value=True), \
                mock.patch.object(I.people_memory,'find_or_create_person') as create, \
                mock.patch.object(I,'_safe_enroll_voice') as enroll:
            self.assertIsNone(I._enroll_new_person('Going Camping Tomorrow',np.zeros(20)))
        create.assert_not_called()
        enroll.assert_not_called()

    def test_mixed_capture_cannot_enroll_even_a_valid_name(self):
        with mock.patch.object(I,'_turn_transcript_trusted',return_value=True), \
                mock.patch.object(I,'_last_scan_windows',[{'change_suspected':True}]), \
                mock.patch.object(I.people_memory,'find_or_create_person') as create:
            self.assertIsNone(I._enroll_new_person('Jeremy Thomas',np.zeros(20)))
        create.assert_not_called()
