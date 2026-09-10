"""Breeze routing, complete impressions, streaming Rex speech and cancellation.

Run with tools/run_lean_checks.py: no Metal, audio device or network required.
"""
import contextlib
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import config
from audio import local_tts, tts

REF = local_tts.VoiceRef('/nonexistent/voice.wav', 'exact reference text', 'rex')


class BreezeTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(config, 'LOCAL_TTS_BACKEND', 'breeze'))
        self.enterContext(mock.patch('utils.conv_log.log_rex'))
        self.enterContext(mock.patch('utils.conv_log.claim_rex_line'))
        self.addCleanup(local_tts.discard_takes)

    def test_engine_selection_and_separate_cache(self):
        self.assertEqual(local_tts.model_id(), 'mlx-community/Breeze-TTS-2-mlx-8bit')
        self.assertEqual(local_tts._model_dir().name, '8bit')
        breeze = local_tts.cache_identity()
        with mock.patch.object(config, 'LOCAL_TTS_BACKEND', 'qwen'):
            self.assertIn('qwen_tts', str(local_tts._model_dir()))
            self.assertNotEqual(breeze, local_tts.cache_identity())
            self.assertFalse(local_tts.streams_clones())
        with mock.patch.object(config, 'LOCAL_TTS_BACKEND', 'bogus'):
            self.assertIn('Unknown LOCAL_TTS_BACKEND', local_tts.unavailable_reason())

    def test_missing_codec_or_tokenizer_cannot_report_available(self):
        with TemporaryDirectory() as temp, \
             mock.patch.object(local_tts, '_model_dir', return_value=Path(temp)), \
             mock.patch('importlib.metadata.version', return_value='0.5.1'):
            root = Path(temp)
            (root / 'model.safetensors').write_bytes(b'weight')
            self.assertIn('vocoder', local_tts.unavailable_reason())
            (root / 'audio_tokenizer').mkdir()
            (root / 'audio_tokenizer/model.safetensors').write_bytes(b'codec')
            self.assertIn('Breeze asset missing', local_tts.unavailable_reason())
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'audio_tokenizer/config.json'):
                (root / name).write_text('{}')
            self.assertIsNone(local_tts.unavailable_reason())
            with mock.patch('importlib.metadata.version', return_value='0.4.5'):
                self.assertIn('0.5.1', local_tts.unavailable_reason())

    def test_generation_uses_stream_and_bounded_tokens_without_splitting_voice(self):
        calls = []
        def generate(**kw):
            calls.append(kw)
            yield SimpleNamespace(audio=np.ones(240, np.float32))
        text = 'This is a long voice impression. ' * 8
        with mock.patch.object(local_tts, '_ensure_model', return_value=SimpleNamespace(generate=generate)):
            chunks = list(local_tts.generate_stream(text, REF))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]['stream'])
        self.assertEqual(calls[0]['ref_text'], REF.ref_text)
        self.assertEqual(calls[0]['ref_audio'], REF.wav_path)
        self.assertEqual(calls[0]['streaming_interval'], 0.25)
        self.assertEqual(calls[0]['cfg_scale'], 1.0)
        self.assertEqual(calls[0]['repetition_penalty'], 1.2)
        self.assertLessEqual(calls[0]['max_tokens'], config.BREEZE_TTS_MAX_TOKENS)

    def test_first_ready_is_a_chunk_not_a_complete_clip(self):
        finish = threading.Event()
        closed = threading.Event()
        def generate(*args):
            try:
                yield np.ones(6000, np.float32)
                finish.wait(2)
                yield np.ones(6000, np.float32)
            finally:
                closed.set()
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate):
            take = local_tts.Take('A whole impression.', REF)
            try:
                self.assertTrue(take.first_ready.wait(1))
                self.assertFalse(closed.is_set())
                self.assertEqual(next(take.stream()).size, 6000)
            finally:
                finish.set()
                take.close()
                take._thread.join(2)
            self.assertTrue(closed.is_set())
            self.assertFalse(take._thread.is_alive())

    def test_cancel_full_queue_releases_generation_lock(self):
        closed = threading.Event()
        def generate(*args):
            with local_tts._generate_lock:
                try:
                    while True:
                        yield np.ones(100, np.float32)
                finally:
                    closed.set()
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate), \
             mock.patch.object(config, 'BREEZE_TTS_QUEUE_CHUNKS', 1):
            take = local_tts.start_take('Old line', REF)
            self.assertTrue(take.first_ready.wait(1))
            fresh = local_tts.start_take('New line', REF)
            try:
                self.assertTrue(closed.wait(1))
                self.assertTrue(fresh.first_ready.wait(1))
                self.assertTrue(take.is_closed)
            finally:
                fresh.close()
                fresh._thread.join(2)
                take._thread.join(2)
            self.assertFalse(fresh._thread.is_alive())

    def test_failed_stream_unblocks_waiters(self):
        def generate(*args):
            raise RuntimeError('bad model')
            yield
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate):
            take = local_tts.Take('line', REF)
            self.assertTrue(take.first_ready.wait(1))
            self.assertTrue(take.failed)
            self.assertEqual(list(take.stream()), [])

    def test_stalled_producer_times_out_and_closes(self):
        release = threading.Event()
        def generate(*args):
            yield np.ones(100, np.float32)
            release.wait(2)
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate), \
             mock.patch.object(config, 'BREEZE_TTS_MAX_CHUNK_WAIT_SECS', 0.04):
            take = local_tts.Take('line', REF)
            try:
                with self.assertRaisesRegex(TimeoutError, 'stalled'):
                    list(take.stream())
                self.assertTrue(take.is_closed)
            finally:
                release.set()
                take.close()
                take._thread.join(2)

    def test_partial_producer_failure_reaches_player(self):
        def generate(*args):
            yield np.ones(100, np.float32)
            raise RuntimeError('bad model')
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate):
            take = local_tts.Take('line', REF)
            with self.assertRaisesRegex(RuntimeError, 'mid-stream'):
                list(take.stream())
            self.assertTrue(take.is_closed)

    def test_clone_readiness_waits_for_every_chunk_and_generator_cleanup(self):
        for label in ('famous:jimmy-carter', 'person:7'):
            with self.subTest(label=label):
                release = threading.Event()
                first_chunk = threading.Event()
                closed = threading.Event()
                chunks = [np.full(6000, i / 100, np.float32) for i in range(20)]
                def generate(*args):
                    try:
                        yield chunks[0]
                        first_chunk.set()
                        release.wait(2)
                        yield from chunks[1:]
                    finally:
                        closed.set()
                with mock.patch.object(local_tts, 'generate_stream', side_effect=generate), \
                     mock.patch.object(config, 'BREEZE_TTS_QUEUE_CHUNKS', 1):
                    take = local_tts.Take('The complete impression.', REF._replace(label=label))
                    try:
                        self.assertTrue(first_chunk.wait(1))
                        self.assertFalse(take.first_ready.is_set())
                        release.set()
                        self.assertTrue(take.first_ready.wait(1))
                        self.assertTrue(closed.is_set())
                        audio = list(take.stream())
                        self.assertEqual(len(audio), 1)
                        np.testing.assert_array_equal(audio[0], np.concatenate(chunks))
                    finally:
                        release.set()
                        take.close()
                        take._thread.join(2)

    def test_partial_clone_failure_discards_unfinished_audio(self):
        def generate(*args):
            yield np.ones(6000, np.float32)
            raise RuntimeError('generation failed before the punchline')
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate):
            take = local_tts.Take('An unfinished impression.', REF._replace(label='person:7'))
            self.assertTrue(take.first_ready.wait(1))
            self.assertTrue(take.failed)
            self.assertEqual(list(take.stream()), [])
            take._thread.join(1)

    def test_cancel_buffering_clone_discards_audio_and_releases_engine(self):
        release = threading.Event()
        first_chunk = threading.Event()
        def generate(*args):
            with local_tts._generate_lock:
                yield np.ones(6000, np.float32)
                first_chunk.set()
                release.wait(2)
                yield np.ones(6000, np.float32)
        with mock.patch.object(local_tts, 'generate_stream', side_effect=generate):
            take = local_tts.Take('Canceled impression.', REF._replace(label='person:7'))
            try:
                self.assertTrue(first_chunk.wait(1))
                take.close()
                release.set()
                take._thread.join(2)
                self.assertFalse(take._thread.is_alive())
                self.assertEqual(list(take.stream()), [])
                self.assertTrue(local_tts._generate_lock.acquire(timeout=1))
                local_tts._generate_lock.release()
            finally:
                release.set()
                take.close()
                take._thread.join(2)

    def test_rex_buffers_preroll_then_streams_remaining_audio(self):
        from audio import echo_cancel
        for label in ('rex',):
            with self.subTest(label=label), contextlib.ExitStack() as stack:
                first_written = threading.Event()
                closed = threading.Event()
                output_settings = []
                generated = []
                def generate(*args):
                    try:
                        for index in range(6):  # 6 x 0.25 s = 1.5 s preroll
                            self.assertFalse(first_written.is_set())
                            generated.append(index)
                            yield np.ones(6000, np.float32) * .1
                        if not first_written.wait(2):
                            raise AssertionError('Player waited for whole clip')
                        yield np.ones(6000, np.float32) * .1
                    finally:
                        closed.set()
                class Stream:
                    latency = 0.01
                    def __init__(self, **kw):
                        output_settings.append(kw)
                    def start(self): pass
                    def write(stream, audio):
                        self.assertEqual(len(generated), 6)
                        first_written.set()
                    def stop(self): pass
                    def abort(self): pass
                    def close(self): pass
                stack.enter_context(mock.patch.object(local_tts, 'generate_stream', side_effect=generate))
                stack.enter_context(mock.patch('sounddevice.OutputStream', Stream))
                stack.enter_context(mock.patch.object(config, 'LOCAL_TTS_CACHE_ENABLED', False))
                stack.enter_context(mock.patch.object(echo_cancel, 'was_canceled', return_value=False))
                stack.enter_context(mock.patch.object(tts.delivery, 'allowed', return_value=True))
                stack.enter_context(mock.patch.object(tts.delivery, 'started'))
                stack.enter_context(mock.patch.object(tts.delivery, 'finish'))
                stack.enter_context(mock.patch.object(tts.output_gate, 'hold', side_effect=lambda *a, **k: contextlib.nullcontext(True)))
                stack.enter_context(mock.patch.object(tts, '_begin_speech', return_value=(None, 'neutral')))
                stack.enter_context(mock.patch.object(tts, '_end_speech'))
                stack.enter_context(mock.patch.object(tts, '_MouthPacer'))
                with mock.patch.object(local_tts, 'synthesize') as full, \
                     mock.patch.object(local_tts, '_synthesize_unit') as unit:
                    self.assertTrue(tts._speak_local('Hello there.', REF._replace(label=label), 'neutral', log_text=False))
                    full.assert_not_called()
                    unit.assert_not_called()
                self.assertTrue(first_written.is_set())
                self.assertTrue(closed.wait(1))
                self.assertEqual(output_settings[0]['latency'], 0.35)
                self.assertEqual(output_settings[0]['blocksize'], 4096)

    def test_repeated_underflows_preserve_every_sample_including_the_ending(self):
        from audio import echo_cancel
        for label in ('rex', 'famous:jimmy-carter', 'person:7'):
            with self.subTest(label=label), contextlib.ExitStack() as stack:
                first_written = threading.Event()
                closed = threading.Event()
                output_settings = []
                generated = []
                writes = []
                aborted = threading.Event()
                def generate(*args):
                    try:
                        for index in range(6):  # 6 x 0.25 s = 1.5 s preroll
                            self.assertFalse(first_written.is_set())
                            generated.append(index)
                            yield np.ones(6000, np.float32) * .1
                        if label == 'rex' and not first_written.wait(2):
                            raise AssertionError('Player waited for whole clip')
                        yield np.ones(6000, np.float32) * .2  # distinctive ending
                    finally:
                        closed.set()
                class Stream:
                    latency = 0.01
                    def __init__(self, **kw):
                        output_settings.append(kw)
                    def start(self): pass
                    def write(stream, audio):
                        self.assertEqual(len(generated), 6)
                        if label != 'rex':
                            self.assertTrue(closed.is_set(), 'Clone played before generation finished')
                        first_written.set()
                        writes.append(audio.copy())
                        return True
                    def stop(self): pass
                    def abort(self): aborted.set()
                    def close(self): pass
                stack.enter_context(mock.patch.object(local_tts, 'generate_stream', side_effect=generate))
                stack.enter_context(mock.patch('sounddevice.OutputStream', Stream))
                stack.enter_context(mock.patch.object(config, 'LOCAL_TTS_CACHE_ENABLED', False))
                stack.enter_context(mock.patch.object(echo_cancel, 'was_canceled', return_value=False))
                stack.enter_context(mock.patch.object(tts.delivery, 'allowed', return_value=True))
                stack.enter_context(mock.patch.object(tts.delivery, 'started'))
                finish = stack.enter_context(mock.patch.object(tts.delivery, 'finish'))
                stack.enter_context(mock.patch.object(tts.output_gate, 'hold', side_effect=lambda *a, **k: contextlib.nullcontext(True)))
                stack.enter_context(mock.patch.object(tts, '_begin_speech', return_value=(None, 'neutral')))
                stack.enter_context(mock.patch.object(tts, '_end_speech'))
                stack.enter_context(mock.patch.object(tts, '_MouthPacer'))
                with mock.patch.object(local_tts, 'synthesize') as full, \
                     mock.patch.object(local_tts, '_synthesize_unit') as unit:
                    self.assertTrue(tts._speak_local('Hello there.', REF._replace(label=label), 'neutral', log_text=False))
                    full.assert_not_called()
                    unit.assert_not_called()
                self.assertFalse(aborted.is_set())
                spoken = np.concatenate(writes)
                # End padding may follow, but no generated sample was dropped.
                self.assertEqual(np.count_nonzero(spoken == np.float32(.1)), 36000)
                self.assertEqual(np.count_nonzero(spoken == np.float32(.2)), 6000)
                finish.assert_called_once_with(canceled=False)
                self.assertTrue(first_written.is_set())
                self.assertTrue(closed.wait(1))
                self.assertEqual(output_settings[0]['latency'], 0.35)
                self.assertEqual(output_settings[0]['blocksize'], 4096)

    def test_offline_and_api_breaker_dispatch_to_selected_breeze(self):
        from intelligence import connectivity
        for offline in (True, False):
            with self.subTest(offline=offline), \
                 mock.patch.object(config, 'NO_AUDIO_MODE', False), \
                 mock.patch.object(config, 'AUDIO_OUTPUT_SUPPRESSED', False), \
                 mock.patch.object(config, 'LOCAL_TTS_MODE', False), \
                 mock.patch.object(local_tts, 'is_available', return_value=True), \
                 mock.patch.object(local_tts, 'rex_voice_ref', return_value=REF), \
                 mock.patch.object(connectivity, 'is_offline', return_value=offline), \
                 mock.patch.object(tts, '_api_circuit_open', return_value=not offline), \
                 mock.patch.object(tts, '_speak_local', return_value=True) as speak, \
                 mock.patch.object(tts, '_fetch_from_api') as api:
                tts.speak('My local voice is ready.', 'neutral')
                speak.assert_called_once()
                self.assertEqual(speak.call_args.args[1], REF)
                api.assert_not_called()

    def test_breeze_impression_waits_for_intro_even_when_online(self):
        from features import impersonation
        events = []
        ready = threading.Event()
        ready.set()
        take = mock.Mock(first_ready=ready, failed=False)
        def enqueue(*args, **kwargs):
            events.append('clone' if kwargs.get('voice_ref') else 'rex')
            done = mock.Mock()
            done.wait.return_value = True
            return done
        def start(*args, **kwargs):
            events.append('start')
            return take
        with mock.patch.object(impersonation, 'build_parody_script', return_value='Welcome to my peanut farm.'), \
             mock.patch.object(local_tts, 'start_take', side_effect=start), \
             mock.patch.object(local_tts, 'pop_take'), \
             mock.patch.object(tts, '_use_local_backend', return_value=False), \
             mock.patch('audio.speech_queue.enqueue', side_effect=enqueue), \
             mock.patch('features.organic_impersonation.cancel'), \
             mock.patch('memory.episodes.record_episode', create=True):
            impersonation.perform(REF._replace(label='famous:jimmy-carter'), 'Jimmy Carter', None)
        self.assertLess(events.index('rex'), events.index('start'))
        self.assertLess(events.index('start'), events.index('clone'))

    def test_setup_downloads_only_selected_8bit_snapshot_and_rejects_incomplete(self):
        import setup_assets as setup
        with TemporaryDirectory() as temp, \
             mock.patch.object(setup, 'LOCAL_TTS_BACKEND', 'breeze'), \
             mock.patch.object(setup, 'download_qwen_tts_model') as qwen, \
             mock.patch('huggingface_hub.snapshot_download') as download:
            made, skipped, failures = setup.download_local_tts_model(Path(temp))
            self.assertTrue(failures)  # downloader returned without creating codec
            self.assertFalse(made)
            qwen.assert_not_called()
            self.assertEqual(download.call_args.kwargs['repo_id'], 'mlx-community/Breeze-TTS-2-mlx-8bit')
            root = Path(download.call_args.kwargs['local_dir'])
            for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'audio_tokenizer/model.safetensors', 'audio_tokenizer/config.json'):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'asset')
            download.reset_mock()
            self.assertTrue(setup.download_local_tts_model(Path(temp))[1])
            download.assert_not_called()
        with mock.patch.object(setup, 'LOCAL_TTS_BACKEND', 'qwen'), \
             mock.patch.object(setup, 'download_qwen_tts_model', return_value=([], ['qwen'], [])) as qwen:
            self.assertEqual(setup.download_local_tts_model(Path('/tmp'))[1], ['qwen'])
            qwen.assert_called_once()
