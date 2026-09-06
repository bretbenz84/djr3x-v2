"""Playback truth with fake sinks; never opens audio, hardware, or a model."""
import contextlib
import sys
import types
import unittest
from unittest import mock
import numpy as np
from audio import delivery, speech_queue as SQ
from tests.test_speech_generations import _bare_queue


class DeliveryContractTest(unittest.TestCase):
    def _queue(self, synth):
        item = SQ._Item(1, 1, 'A complete sentence.', 'neutral', None,
                        SQ.DoneEvent(), generation=SQ.generation())
        with (
            mock.patch.object(SQ, '_state_suppresses_output', return_value=False),
            mock.patch('audio.tts.speak', side_effect=synth),
            mock.patch('audio.sound_effects.play_for_speech'),
        ):
            _bare_queue()._process_item(item)
        return item.done

    def test_silent_backend_return_does_not_claim_sentence(self):
        ev = self._queue(lambda *a, **kw: None)
        self.assertFalse(ev.played)
        self.assertEqual(ev.dropped_reason, 'not_started')

    def test_partial_sentence_does_not_claim_ending(self):
        def synth(*args, **kwargs):
            delivery.started()
            delivery.finish(canceled=True)
        ev = self._queue(synth)
        self.assertTrue(ev.started)
        self.assertFalse(ev.played)
        self.assertEqual(ev.dropped_reason, 'interrupted')

    def test_synthesis_becomes_stale_before_buffered_sink(self):
        from audio import tts
        fake_sd = types.SimpleNamespace(play=mock.Mock(), wait=mock.Mock())
        def synth(*args, **kwargs):
            SQ.invalidate_pending('human during synthesis')
            with (
                mock.patch.dict(sys.modules, sounddevice=fake_sd),
                mock.patch.object(tts.output_gate, 'hold',
                                  return_value=contextlib.nullcontext(True)),
            ):
                tts._play(np.zeros(10), 100, 'neutral')
        ev = self._queue(synth)
        fake_sd.play.assert_not_called()
        self.assertFalse(ev.played)
        self.assertEqual(ev.dropped_reason, 'stale_generation')

    def test_nested_delivery_restores_outer_record(self):
        outer, inner = delivery.Delivery(), delivery.Delivery()
        with delivery.track(outer):
            delivery.started()
            with delivery.track(inner):
                delivery.started()
                delivery.finish(canceled=True)
            delivery.finish()
        self.assertTrue(outer.completed)
        self.assertFalse(inner.completed)

class BufferedSinkTest(unittest.TestCase):
    def test_sink_completion_error_and_interruption(self):
        from audio import tts
        for outcome in ('complete', 'error', 'interrupted'):
            with self.subTest(outcome=outcome), contextlib.ExitStack() as stack:
                record = delivery.Delivery()
                fake_sd = types.SimpleNamespace(play=mock.Mock(), wait=mock.Mock())
                if outcome == 'error':
                    fake_sd.wait.side_effect = RuntimeError('device failed')
                stack.enter_context(mock.patch.dict(sys.modules, sounddevice=fake_sd))
                stack.enter_context(mock.patch.object(tts.output_gate, 'hold',
                    return_value=contextlib.nullcontext(True)))
                for name in ('leds_head', 'leds_chest', 'servos', 'animations', 'emotion_orchestrator'):
                    stack.enter_context(mock.patch.object(tts, name))
                stack.enter_context(mock.patch.object(tts, '_drive_leds'))
                stack.enter_context(mock.patch.object(tts.echo_cancel, 'set_playing'))
                stack.enter_context(mock.patch.object(tts.echo_cancel, 'was_canceled',
                    return_value=outcome == 'interrupted'))
                with delivery.track(record):
                    tts._play(np.zeros(1), 100, 'neutral')
                self.assertTrue(record.started)
                self.assertEqual(record.completed, outcome == 'complete')

class StreamedSinkTest(unittest.TestCase):
    def test_interrupted_stream_does_not_claim_complete_sentence(self):
        from audio import tts
        from pathlib import Path
        record = delivery.Delivery()
        canceled = [False]
        closed = []
        def chunks():
            try:
                yield b'\x01\x00' * 100
                yield b'\x01\x00' * 100
            finally:
                closed.append(True)
        stream = mock.Mock()
        stream.write.side_effect = lambda *a: canceled.__setitem__(0, True)
        fake_sd = types.SimpleNamespace(OutputStream=mock.Mock(return_value=stream))
        with (
            mock.patch.dict(sys.modules, sounddevice=fake_sd),
            mock.patch.object(tts, '_get_el_client', return_value=types.SimpleNamespace(
                text_to_speech=types.SimpleNamespace(stream=lambda **kw: chunks()))),
            mock.patch.object(tts.output_gate, 'hold', return_value=contextlib.nullcontext(True)),
            mock.patch.object(tts.sd_guard, 'device_control', side_effect=lambda **kw: contextlib.nullcontext()),
            mock.patch.object(tts, '_begin_speech', return_value=(None, 'neutral')),
            mock.patch.object(tts, '_end_speech'),
            mock.patch.object(tts, '_MouthPacer'),
            mock.patch.object(tts, '_stream_output_latency', return_value=0),
            mock.patch.object(tts.echo_cancel, 'was_canceled', side_effect=lambda: canceled[0]),
            delivery.track(record),
        ):
            self.assertTrue(tts._speak_streaming('hello', 'hello', 'fake', 'fake', None,
                None, 'neutral', Path('/tmp/never-written-replay.mp3'), log_text=False))
        self.assertTrue(record.started)
        self.assertFalse(record.completed)
        stream.abort.assert_called_once()
        self.assertEqual(closed, [True])
