import time
import unittest
from unittest import mock
from intelligence import command_parser, pride
from sequences import throttle_arm as arm
from tests import test_throttle_runtime as runtime


class CommandTests(unittest.TestCase):
    def test_phrases(self):
        for text, pose in [('put your arm down', 'down'), ('Rex, please lower your arm.', 'down'),
                           ('hold out your hand', 'offer'), ('Could you hold your hand out please?', 'offer'),
                           ('give me a high-five', 'high_five'), ('relax your arm', 'rest')]:
            match = command_parser.parse(text)
            self.assertEqual((match.command_key, match.args), ('throttle_pose', {'pose': pose}))
        for text in ('stand down pride mode', 'Please turn off pride mode.'):
            self.assertEqual(command_parser.parse(text).command_key, 'pride_off')

    def test_additional_arm_phrasings(self):
        examples = {
            'high_five': ('raise your arm', 'lift your arm', 'put your arm up',
                          'hold your hand up', 'high five me', 'give me five'),
            'offer': ('outstretch your arm', 'stretch your arm out',
                      'hold your arm straight out', 'reach out your hand', 'offer me your hand'),
            'down': ('lower your hand', 'bring your arm down', 'put your hand down'),
            'rest': ('pull your arm back', 'bring your hand back', 'return your arm to neutral'),
        }
        for pose, phrases in examples.items():
            for phrase in phrases:
                for text in (phrase, f'Rex, could you please {phrase}?'):
                    with self.subTest(text=text):
                        match = command_parser.parse(text)
                        self.assertEqual((match.command_key, match.args), ('throttle_pose', {'pose': pose}))
                for text in (f"don't {phrase}", f'he said to {phrase}', f'why did you {phrase}'):
                    self.assertIsNone(command_parser._parse_arm_or_pride(text))

    def test_negation_and_narration_do_not_claim_commands(self):
        for text in ("don't put your arm down", 'he asked me to hold out your hand',
                     'what happens when I say give me a high five', 'do not end pride mode'):
            self.assertIsNone(command_parser._parse_arm_or_pride(text))

    def test_dispatch_and_pride_exit(self):
        from intelligence import interaction as I
        with (mock.patch.object(arm, 'request_pose', return_value=True) as request,
              mock.patch.object(I, '_speak_blocking'),
              mock.patch('hardware.leds_head.send_command') as head,
              mock.patch('hardware.leds_chest.send_command') as chest):
            response = I._execute_command(command_parser.parse('give me a high five'), None, None, 'give me a high five')
            self.assertEqual(response, 'High five!')
            request.assert_called_once_with('high_five')
            request.return_value = False
            response = I._execute_command(command_parser.parse('hold out your hand'), None, None, 'hold out your hand')
            self.assertIn("can't", response)
            with mock.patch.object(pride, '_active_until', time.monotonic() + 600):
                I._execute_command(command_parser.parse('stand down pride mode'), None, None, 'stand down pride mode')
                self.assertFalse(pride.is_active())
                head.assert_called_once_with('PRIDE:0')
                chest.assert_called_once_with('PRIDE:0')


class PoseWorkerTests(runtime.RuntimeTest):
    def test_named_pose_holds_against_speech_and_base_takes_priority(self):
        self.assertTrue(arm.start())
        self.assertTrue(arm.request_pose('offer'))
        controller = arm._controller
        deadline = time.monotonic() + 3
        while controller.pose_until is None and time.monotonic() < deadline:
            controller.stop_event.wait(.02)
        self.assertIsNotNone(controller.pose_until)
        self.assertEqual(self.port.pose, arm.INTRODUCTION)
        arm.speech_start({'affect': 'excited', 'intensity': 1})
        controller.stop_event.wait(.2)
        self.assertEqual(self.port.pose, arm.INTRODUCTION)
        self.assertTrue(arm.prepare_base_motion(timeout=3))
        self.assertIsNone(controller.pose_request)
        self.assertFalse(arm.request_pose('high_five'))
        self.assertEqual(self.port.pose, arm.PARK)

    def test_no_worker_or_manual_override_refuses(self):
        self.assertFalse(arm.request_pose('down'))
        self.assertTrue(arm.start())
        with mock.patch.object(arm.servos, '_program_servo_updates_blocked', return_value=True):
            self.assertFalse(arm.request_pose('down'))
        self.assertFalse(arm.request_pose('unknown'))
