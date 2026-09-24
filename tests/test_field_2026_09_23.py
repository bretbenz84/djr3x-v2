"""20:53 run: speaker abstention must survive prompt building; dotted initials."""
import unittest
from unittest import mock
from intelligence import introductions, social_scene, social_frame


class FieldTests(unittest.TestCase):
    def test_dotted_initials_survive_introduction_parser(self):
        for text in ("I'd like you to meet J.C.", 'This is J. C.', 'Meet my friend J.C.'):
            parsed = introductions.detect(text, has_unknown_face=True)
            self.assertTrue(parsed.is_introduction)
            self.assertEqual(parsed.name, 'JC')

    def test_owner_actual_spoken_initials_survive_introduction_parser(self):
        parsed = introductions.detect("I'd like you to meet JT", has_unknown_face=True)
        self.assertTrue(parsed.is_introduction)
        self.assertEqual(parsed.name, 'JT')

    def test_uncertain_speaker_is_not_named_for_visible_person(self):
        snap = {'people': [{'person_db_id': 4, 'face_id': 'Jeremy Thomas',
                            'face_visible': True, 'id': 'person_1'}]}
        with mock.patch.object(social_scene, '_pronouns_for_person', return_value=''):
            cast = social_scene.conversation_cast_context(snap, speaker_uncertain=True)
        self.assertIn('unidentified speaker', cast.addressee)
        self.assertIsNone(cast.current_speaker)
        self.assertNotIn('[current speaker]', cast.directive)
        with mock.patch.object(social_frame.world_state, 'snapshot', return_value=snap):
            self.assertIn('unidentified speaker', social_frame._addressee(None, speaker_uncertain=True))

    def test_known_speaker_keeps_named_cast(self):
        snap = {'people': [{'person_db_id': 4, 'face_id': 'Jeremy Thomas',
                            'face_visible': True, 'id': 'person_1'}]}
        with mock.patch.object(social_scene, '_pronouns_for_person', return_value=''):
            cast = social_scene.conversation_cast_context(snap, current_person_id=4)
        self.assertIn('Jeremy', cast.addressee)

    def test_async_motion_notice_cannot_duplicate_streamed_reply(self):
        from intelligence import interaction as I
        reply = "Yeah? That's a relief."
        with (mock.patch.object(I, '_prelogged_response', []),
              mock.patch.object(I.conv_memory, 'add_to_transcript') as transcript,
              mock.patch.object(I.conv_log, 'log_rex') as log,
              mock.patch.object(I.conv_log, 'finish_rex_stream')):
            I._record_streamed_response(reply)
            I.conv_log.log_rex("I can't confirm enough clearance.")
            # The turn's owner uses this marker, not last-line text deduplication.
            self.assertEqual(I._consume_prelogged_response(), reply)
            self.assertIsNone(I._consume_prelogged_response())
            transcript.assert_called_once_with('Rex', reply)
            self.assertEqual(log.call_args_list, [mock.call(reply, to_gui=False),
                                                  mock.call("I can't confirm enough clearance.")])
