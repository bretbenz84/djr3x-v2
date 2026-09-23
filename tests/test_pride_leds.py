"""Pride colour synchronization without opening either serial port."""
import unittest
from unittest import mock
from hardware import leds_head as head, leds_chest as chest


class PrideLEDTests(unittest.TestCase):
    def setUp(self):
        for patch in (mock.patch.object(head, '_pride_sent', False),
                      mock.patch.object(head, '_speaking', False),
                      mock.patch.object(head, '_eyes_should_be_on', True),
                      mock.patch.object(head, '_serial_online', return_value=True),
                      mock.patch.object(head, '_serial_online_locked', return_value=True),
                      mock.patch.object(chest, '_awake', True),
                      mock.patch.object(chest, '_pride_sent', False),
                      mock.patch.object(chest, 'connected', return_value=True),
                      mock.patch('intelligence.pride.is_active', return_value=True)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_head_starts_rainbow_before_speech_and_refreshes_during_speech(self):
        with mock.patch.object(head, 'send_command') as send:
            head.speak('sad')
            self.assertEqual(send.call_args_list, [mock.call('PRIDE:1'), mock.call('SPEAK:sad')])
            send.reset_mock()
            head._heartbeat_tick()
            send.assert_called_once_with('PRIDE:1')

    def test_both_expire_and_stop_refreshing(self):
        for board, sync in ((head, head._sync_pride_mouth), (chest, chest._sync_pride)):
            with mock.patch.object(board, 'send_command') as send:
                sync()
                with mock.patch('intelligence.pride.is_active', return_value=False):
                    sync()
                    sync()
                self.assertEqual(send.call_args_list, [mock.call('PRIDE:1'), mock.call('PRIDE:0')])

    def test_sleeping_boards_do_not_receive_rainbow_refresh(self):
        head._eyes_should_be_on = False
        chest._awake = False
        with mock.patch.object(head, 'send_command') as h, mock.patch.object(chest, 'send_command') as c:
            head._heartbeat_tick()
            chest._sync_pride()
            h.assert_not_called()
            c.assert_not_called()

    def test_chest_sync_precedes_mode_command_and_off_blocks_refresh(self):
        fake = mock.Mock(is_open=True)
        with (mock.patch.object(chest, '_ser', fake),
              mock.patch.object(chest, 'CHEST_LEDS_ENABLED', True),
              mock.patch.object(chest, '_mirror_gui_chest_led_state')):
            chest.speak('happy')
            self.assertEqual(fake.write.call_args_list, [mock.call(b'PRIDE:1\n'), mock.call(b'SPEAK:happy\n')])
            chest.off()
            fake.write.reset_mock()
            chest._sync_pride()
            fake.write.assert_not_called()
