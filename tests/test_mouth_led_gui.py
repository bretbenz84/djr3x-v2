import unittest
from unittest import mock
import numpy as np
from gui.mouth_leds import MouthAnimation
from gui.state_bridge import GUIDashboardBridge

class MouthLEDTests(unittest.TestCase):
    def test_command_sequence_and_offline_mirror(self):
        from hardware import leds_head
        b=GUIDashboardBridge()
        with mock.patch('gui.state_bridge.gui_bridge',b),mock.patch.object(leds_head,'HEAD_LEDS_ENABLED',False):
            leds_head.speak('happy');leds_head.speak_level(190)
            s=b.get_snapshot()['head_led_state'];self.assertEqual(s['mouth_level'],190);self.assertEqual(s['mouth_emotion'],'happy')
            leds_head.send_command('SPEAK_STOP');self.assertEqual(b.get_snapshot()['head_led_state']['mouth_mode'],'active')
            leds_head.send_command('OFF');self.assertEqual(b.get_snapshot()['head_led_state']['mouth_mode'],'off')
    def test_audio_opens_bars_and_keeps_center_symmetry(self):
        a=MouthAnimation();s=dict(mouth_mode='speak',mouth_level=0,mouth_emotion='happy',mouth_updated_at=10)
        quiet=a.render(s,10)
        self.assertTrue(np.all(quiet[:32]==0));self.assertTrue(np.any(quiet[32:48]>0))
        s['mouth_level']=255
        for i in range(1,30):loud=a.render(s,10+i/30)
        self.assertGreater(np.count_nonzero(loud),np.count_nonzero(quiet))
        np.testing.assert_allclose(loud.reshape(10,8,3),loud.reshape(10,8,3)[::-1]);self.assertTrue(np.all(loud[:,0]==0))
    def test_idle_off_timeout_and_fade(self):
        a=MouthAnimation();s=dict(mouth_mode='active',mouth_emotion='angry')
        glow=a.render(s,10);self.assertTrue(np.all(glow[:,0]>0));self.assertTrue(np.all(glow[:,1:]==0))
        fade=dict(s,mouth_mode='fadeoff',mouth_updated_at=10)
        a.render(fade,10);np.testing.assert_allclose(a.render(fade,12),glow*.5)
        self.assertTrue(np.all(a.render(fade,14)==0));self.assertTrue(np.all(a.render({'mouth_mode':'off'},15)==0))
        stale=a.render(dict(s,mouth_mode='speak',mouth_level=255,mouth_updated_at=1),20)
        self.assertLessEqual(stale.max(),.1)
    def test_next_utterance_starts_at_zero(self):
        b=GUIDashboardBridge();b.update_mouth_led_command('SPEAK:happy');b.update_mouth_led_command('SPEAK_LEVEL:255');b.update_mouth_led_command('SPEAK:curious')
        self.assertEqual(b.get_snapshot()['head_led_state']['mouth_level'],0)

if __name__=='__main__':unittest.main()
