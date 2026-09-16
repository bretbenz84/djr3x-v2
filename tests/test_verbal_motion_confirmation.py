"""The 21:13 field refusal was a -15 degree RIGHT turn. No live I/O."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from intelligence import motion_controller as M

FIELD = {'fl':195,'fr':235,'rl':4000,'rr':2241,'lf':-1,'lb':-1,
         'rf':823,'rb':1169,'fl_radial':4000,'fr_radial':-1}


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.now=100.
        self.tof=dict(FIELD)
        self.stack.enter_context(patch.object(M,'_pending_verbal_turn',None))
        self.stack.enter_context(patch.object(M.time,'monotonic',side_effect=lambda:self.now))
        self.allowed=self.stack.enter_context(patch.object(M,'_autonomous_allowed',return_value=None))
        self.stack.enter_context(patch.object(M,'_user_commanded_fx',return_value=True))
        self.tele=self.stack.enter_context(patch.object(M.motion,'telemetry',side_effect=lambda:{
            'rx_monotonic':self.now,'tof_mm':self.tof}))
        self.send=self.stack.enter_context(patch.object(M.motion,'send',return_value=42))
        for name in ('_calibrated_compass_yaw','_relative_turn_yaw','_invalidate_turn_verification',
                     '_cancel_arc','_note_issued','_fx_drive_loop_start','_suppressed'):
            self.stack.enter_context(patch.object(M,name))

    def refuse_right(self):
        self.assertIsNone(M.turn(-15,allow_escape=False))
        self.send.assert_not_called()
        self.assertIsNotNone(M._pending_verbal_turn)

    def test_confirmation_repeats_exact_right_turn_slowly_once(self):
        self.refuse_right()
        self.assertIn('right slowly',M.confirm_refused_motion())
        self.send.assert_called_once_with({'cmd':'turn','deg':-15.,'rate':15.})
        self.assertIsNone(M.confirm_refused_motion())

    def test_confirmation_does_not_remove_independent_obstacle(self):
        self.refuse_right()
        self.tof['fl_radial']=195
        self.assertIn('another motion check',M.confirm_refused_motion())
        self.send.assert_not_called()
        self.assertIsNone(M._pending_verbal_turn)

    def test_missing_independent_reading_cannot_erase_matrix_obstacle(self):
        self.refuse_right(); self.tof['fl_radial']=-1
        self.assertIn('another motion check',M.confirm_refused_motion())
        self.send.assert_not_called()

    def test_expired_confirmation_never_moves(self):
        self.refuse_right(); self.now+=46
        self.assertIsNone(M.confirm_refused_motion())
        self.send.assert_not_called()

    def test_user_stop_clears_confirmation_but_sequence_cleanup_keeps_it(self):
        self.refuse_right()
        with patch.object(M.motion,'connected',return_value=False), \
             patch.object(M,'_cancel_swing_escape'), patch.object(M,'_fx_drive_loop_stop_all'):
            M.stop(preserve_refused_turn=True)
            self.assertIsNotNone(M._pending_verbal_turn)
            M.stop()
            self.assertIsNone(M._pending_verbal_turn)

    def test_charging_or_stale_sensor_still_prevents_retry(self):
        for reason in ('charging',None):
            with self.subTest(reason=reason):
                self.allowed.return_value=None; self.refuse_right()
                self.allowed.return_value=reason
                with patch.object(M.motion,'telemetry',return_value={'rx_monotonic':0.,'tof_mm':FIELD}):
                    self.assertIn('another motion check',M.confirm_refused_motion())
                self.send.assert_not_called()

    def test_spoken_yes_you_can_takes_local_path_even_with_unknown_voice(self):
        from intelligence import interaction as I
        self.refuse_right()
        with patch.object(I,'_speak_blocking') as speak, \
             patch.object(I,'_clear_motion_continuation'), \
             patch.object(I,'_router_audit_note_fast_local_action'):
            line=I._explicit_motion_takeover('Yes, you can.',person_id=None)
        self.assertIn('right slowly',line)
        speak.assert_called_once()
        self.send.assert_called_once_with({'cmd':'turn','deg':-15.,'rate':15.})
