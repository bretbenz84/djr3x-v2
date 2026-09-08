"""The answer window, not a fixed clip or an incidental VAD edge, owns music."""
import contextlib
import threading
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import config
from audio import speech_queue as sq
from features import games
from intelligence import interaction as I

THEME = 'assets/audio/jeopardy/jeopardy-theme.mp3'


class MusicClockTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(games, '_active_game', 'jeopardy'))
        self.enterContext(mock.patch.object(games, '_game_state', {
            'phase': 'awaiting_answer', 'awaiting_prompt_delivery': True,
            'current_clue': {'answer': 'Paris', 'clue': 'The capital of France', 'category': 'EUROPE', 'value': 200},
            'players': [{'name': 'A', 'score': 0}, {'name': 'B', 'score': 0}],
            'current_player_idx': 0, 'board': {'remaining': 3},
            'pending_after_response_clip': 'theme',
        }))
        self.enterContext(mock.patch.object(config, 'JEOPARDY_PLAY_THINKING_THEME', True))
        self.enterContext(mock.patch.object(games.threading, 'Timer'))
        self.audio = self.enterContext(mock.patch.object(sq, 'enqueue_audio_file'))
        self.enterContext(mock.patch.object(sq, 'enqueue'))
        self.enterContext(mock.patch.object(games, '_body_beat'))
        self.enterContext(mock.patch.object(games, '_jeopardy_queue_clip'))
        self.enterContext(mock.patch.object(games, '_jeopardy_schedule_post_timeout_rebound'))
        self.enterContext(mock.patch.object(games, '_quick_call', return_value='no'))

    def arm(self):
        games.on_response_spoken()
        return self.audio.call_args.kwargs['loop_stop']

    def test_delivered_clue_starts_one_clock_owned_bed(self):
        stop = self.arm()
        self.assertFalse(stop.is_set())
        self.assertIsNone(games.consume_pending_audio_after_response())
        games.on_response_spoken()
        self.audio.assert_called_once()
        self.assertNotIn('answer_music_stop', games.snapshot())

    def test_speech_grace_keeps_same_bed_running(self):
        stop = self.arm()
        with mock.patch.object(games, '_jeopardy_answer_in_flight', return_value=True):
            games._jeopardy_timeout_fired(games._game_state['answer_timer_token'])
        self.assertFalse(stop.is_set())
        self.audio.assert_called_once()

    def test_ignored_room_noise_does_not_stop_music_or_timer(self):
        stop = self.arm()
        token = games._game_state['answer_timer_token']
        self.assertEqual(games.handle_input('Shh.'), '')
        self.assertFalse(stop.is_set())
        self.assertEqual(games._game_state['answer_timer_token'], token)

    def test_judge_ignored_chatter_preserves_continuous_music(self):
        stop = self.arm()
        with mock.patch.object(games, '_quick_call', return_value='none'):
            self.assertEqual(games.handle_input('Come here Toby'), '')
        self.assertFalse(stop.is_set())
        games.on_response_spoken()
        self.audio.assert_called_once()

    def test_software_barge_resume_preserves_clock_deadline(self):
        stop = self.arm()
        token = games._game_state['answer_timer_token']
        deadline = games._game_state['answer_timer_deadline']
        stop.set()  # player capture stopped the output on a no-hardware-AEC Mac
        games.resume_answer_music_after_capture()
        self.assertEqual(self.audio.call_count, 2)
        self.assertFalse(self.audio.call_args.kwargs['loop_stop'].is_set())
        self.assertEqual(games._game_state['answer_timer_token'], token)
        self.assertEqual(games._game_state['answer_timer_deadline'], deadline)

    def test_submitted_answer_stops_music_and_scores_active_player(self):
        stop = self.arm()
        line = games.handle_input('What is Paris?', person_id=999)
        self.assertTrue(stop.is_set())
        self.assertIn('$200 to A', line)

    def test_timeout_stops_music_before_chime_then_rebound_starts_fresh(self):
        stop = self.arm()
        def chime(key, **kwargs):
            self.assertEqual(key, 'timesup')
            self.assertTrue(stop.is_set())
        with mock.patch.object(games, '_jeopardy_answer_in_flight', return_value=False), \
             mock.patch.object(games, '_jeopardy_queue_clip', side_effect=chime):
            games._jeopardy_timeout_fired(games._game_state['answer_timer_token'])
        fresh = self.arm()
        self.assertIsNot(fresh, stop)
        self.assertFalse(fresh.is_set())

    def test_stale_timeout_cannot_stop_new_answer_music(self):
        old = self.arm()
        token = games._game_state['answer_timer_token']
        games._jeopardy_cancel_timeout()
        games._game_state['awaiting_prompt_delivery'] = True
        new = self.arm()
        games._jeopardy_timeout_fired(token)
        self.assertTrue(old.is_set())
        self.assertFalse(new.is_set())

    def test_repeat_restarts_music_only_after_repeat_is_delivered(self):
        old = self.arm()
        games.handle_input('Repeat the clue')
        self.assertTrue(old.is_set())
        self.audio.assert_called_once()
        self.assertFalse(self.arm().is_set())

    def test_clear_game_cancels_bed(self):
        stop = self.arm()
        games._clear_game()
        self.assertTrue(stop.is_set())

    def test_daily_double_clock_stops_music_without_rebound(self):
        games._game_state['current_clue']['daily_double'] = True
        stop = self.arm()
        with mock.patch.object(games, '_jeopardy_answer_in_flight', return_value=False):
            games._jeopardy_timeout_fired(games._game_state['answer_timer_token'])
        self.assertTrue(stop.is_set())
        self.assertEqual(games._game_state['phase'], 'selecting')

    def test_switching_game_cancels_old_music_and_timer(self):
        stop = self.arm()
        timer = games._game_state['answer_timer']
        with mock.patch.object(games, 'can_play', return_value=(True, None)), \
             mock.patch.dict(games._GAME_HANDLERS['trivia'], start=mock.Mock(return_value='Trivia')):
            games.start_game('trivia')
        self.assertTrue(stop.is_set())
        timer.cancel.assert_called_once()

    def test_stop_confirmation_silences_bed(self):
        stop = self.arm()
        games.request_stop_confirmation()
        self.assertTrue(stop.is_set())

    def test_disabled_music_keeps_answer_clock(self):
        with mock.patch.object(config, 'JEOPARDY_PLAY_THINKING_THEME', False):
            games.on_response_spoken()
        self.audio.assert_not_called()
        self.assertIn('answer_timer_token', games._game_state)

    def test_only_active_bed_with_hardware_aec_survives_capture(self):
        stop = self.arm()
        with mock.patch.object(I.hardware_aec, 'is_active', return_value=True):
            self.assertTrue(I._keep_game_music_during_capture(THEME))
            self.assertFalse(I._keep_game_music_during_capture(THEME.replace('theme', 'timesup')))
            stop.set()
            self.assertFalse(I._keep_game_music_during_capture(THEME))
        stop.clear()
        with mock.patch.object(I.hardware_aec, 'is_active', return_value=False):
            self.assertFalse(I._keep_game_music_during_capture(THEME))


class LoopPlaybackTest(unittest.TestCase):
    def setUp(self):
        self.queue = object.__new__(sq._SpeechQueue)
        self.enterContext(mock.patch('soundfile.read', return_value=(np.ones(3100, dtype=np.float32)*0.1, 100)))
        self.enterContext(mock.patch.object(config, 'JEOPARDY_AUDIO_OUTPUT_SAMPLE_RATE', 100))
        self.enterContext(mock.patch.object(config, 'JEOPARDY_THEME_MAX_SECS', 12.0))
        self.play = self.enterContext(mock.patch('sounddevice.play'))
        self.enterContext(mock.patch('sounddevice.wait'))
        self.stream = SimpleNamespace(active=True, stop=mock.Mock())
        self.enterContext(mock.patch('sounddevice.get_stream', return_value=self.stream))
        self.enterContext(mock.patch('audio.output_gate.hold', return_value=contextlib.nullcontext(True)))
        self.enterContext(mock.patch('audio.echo_cancel.set_playing'))
        self.enterContext(mock.patch('audio.echo_cancel.was_canceled', return_value=False))
        self.enterContext(mock.patch('audio.delivery.allowed', return_value=True))

    def test_one_stream_loops_full_file_until_owner_ends_window(self):
        stop = threading.Event()
        polls = []
        def clock_tick(seconds):
            polls.append(seconds)
            if len(polls) == 1000:  # 25 seconds: beyond the old 12-second clip cap
                stop.set()
            return stop.is_set()
        with mock.patch.object(stop, 'wait', side_effect=clock_tick):
            self.queue._play_file(THEME, loop_stop=stop)
        self.play.assert_called_once()
        self.assertTrue(self.play.call_args.kwargs['loop'])
        self.assertEqual(len(self.play.call_args.args[0]), 3100)
        self.assertEqual(len(polls), 1000)
        self.stream.stop.assert_called_once()

    def test_standalone_file_retains_duration_cap(self):
        self.queue._play_file(THEME)
        self.assertEqual(len(self.play.call_args.args[0]), 1200)
        self.assertNotIn('loop', self.play.call_args.kwargs)

    def test_already_closed_window_never_starts_stale_music(self):
        stop = threading.Event(); stop.set()
        self.queue._play_file(THEME, loop_stop=stop)
        self.play.assert_not_called()

    def test_preempted_playback_is_not_restarted(self):
        self.stream.active = False
        self.queue._play_file(THEME, loop_stop=threading.Event())
        self.play.assert_called_once()
        self.stream.stop.assert_not_called()

    def test_normal_spoken_control_cannot_wait_behind_indefinite_bed(self):
        q = self.queue
        q._lock = threading.Lock(); q._not_empty = threading.Condition(q._lock)
        q._heap = []; q._seq = 0; q._speaking = True; q._current_priority = 1
        q._startup_chime_queued = True
        q._current_loop_stop = threading.Event()
        with mock.patch.object(sq, '_state_suppresses_output', return_value=False), \
             mock.patch.object(sq, '_audio_output_suppressed', return_value=False):
            q.enqueue('Stop the game?', priority=1)
        self.assertTrue(q._current_loop_stop.is_set())
        self.assertEqual(q._heap[0].text, 'Stop the game?')

    def test_spoken_control_drops_bed_that_has_not_started_yet(self):
        q = self.queue
        q._lock = threading.Lock(); q._not_empty = threading.Condition(q._lock)
        q._heap = []; q._seq = 0; q._speaking = False
        q._startup_chime_queued = True
        stop = threading.Event()
        with mock.patch.object(sq, '_state_suppresses_output', return_value=False), \
             mock.patch.object(sq, '_audio_output_suppressed', return_value=False):
            pending = q.enqueue_audio_file(THEME, priority=1, loop_stop=stop)
            q.enqueue('Stop the game?', priority=1)
        self.assertTrue(stop.is_set())
        self.assertTrue(pending.is_set())
        self.assertEqual(len(q._heap), 1)


if __name__ == '__main__':
    unittest.main()
