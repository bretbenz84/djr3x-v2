"""Speech leveling without network, playback, or robot I/O."""
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from audio.speech_level import SpeechLeveler, level_audio


def speech(sr=22050, seconds=2.2, amplitude=0.06):
    t = np.arange(round(sr * seconds)) / sr
    envelope = (0.35 + 0.65 * np.sin(2 * np.pi * 2.3 * t) ** 2)
    envelope[:round(sr * .1)] = 0
    envelope[-round(sr * .15):] = 0
    return (amplitude * envelope * (np.sin(2*np.pi*210*t) + .3*np.sin(2*np.pi*1900*t))).astype(np.float32)


class LevelerTests(unittest.TestCase):
    def test_quiet_and_loud_takes_reach_the_same_level(self):
        source = speech(amplitude=.1)
        quiet, loud = level_audio(source * .25, 22050), level_audio(source * 6.25, 22050)
        delta = 20 * np.log10(np.linalg.norm(quiet) / np.linalg.norm(loud))
        self.assertLess(abs(delta), .2)  # 28 dB input difference

    def test_network_chunk_sizes_do_not_change_sound_or_length(self):
        for sr in (16000, 22050, 44100, 48000):
            with self.subTest(sr=sr):
                source = speech(sr, seconds=.877)
                expected = level_audio(source, sr)
                processor = SpeechLeveler(sr)
                chunks = [processor.process(source[i:i+137]) for i in range(0, len(source), 137)]
                chunks.append(processor.process(np.empty(0), final=True))
                np.testing.assert_array_equal(np.concatenate(chunks), expected)
                self.assertEqual(len(expected), len(source))

    def test_empty_short_and_silent_inputs_flush_without_added_samples(self):
        for n in (0, 1, 220, 22050):
            out = level_audio(np.zeros(n), 22050)
            np.testing.assert_array_equal(out, np.zeros(n))
            self.assertEqual(out.dtype, np.float32)
        tiny = speech(seconds=.11)
        self.assertEqual(len(level_audio(tiny, 22050)), len(tiny))

    def test_preroll_is_bounded_and_emits_before_end_of_take(self):
        processor = SpeechLeveler(22050)
        data = speech()
        self.assertEqual(processor.process(data[:2205]).size, 0)
        self.assertGreater(processor.process(data[2205:8820]).size, 0)

    def test_pauses_do_not_raise_noise_or_change_gain(self):
        processor = SpeechLeveler(22050)
        processor.process(speech())
        before = processor._gain_db
        pause = processor.process(np.zeros(22050 * 4, dtype=np.float32))
        self.assertEqual(processor._gain_db, before)
        self.assertEqual(float(np.max(np.abs(pause[1000:]))), 0)

    def test_same_phrase_after_long_pause_does_not_swell(self):
        source = speech()
        gap = np.zeros(22050 * 4, dtype=np.float32)
        out = level_audio(np.concatenate((source, gap, source)), 22050)
        first, repeated = out[:len(source)], out[-len(source):]
        self.assertLess(abs(20 * np.log10(np.linalg.norm(repeated) / np.linalg.norm(first))), 1.5)

    def test_transients_are_limited_without_flattening_samples(self):
        source = speech(amplitude=.008)
        source[22050:22150] = np.linspace(-.9, .9, 100)
        out = level_audio(source, 22050)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(float(np.max(np.abs(out))), 10 ** (-3 / 20) + 1e-6)
        self.assertGreater(np.unique(out[22050:22150]).size, 90)

    def test_maximum_gain_and_nonfinite_inputs_are_bounded(self):
        source = speech(amplitude=.00001)
        out = level_audio(source, 22050, max_gain_db=12)
        self.assertLessEqual(np.linalg.norm(out) / np.linalg.norm(source), 10 ** (12 / 20) + 1e-4)
        bad = np.array([float('nan'), float('inf'), -float('inf')], dtype=np.float32)
        np.testing.assert_array_equal(level_audio(bad, 22050), np.zeros(3))


class PlaybackIntegrationTests(unittest.TestCase):
    def setUp(self):
        from audio import tts
        self.tts = tts
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.raw = (speech() * 32768).astype('<i2')
        self.samples = self.raw.astype(np.float32) / 32768
        self.enterContext(mock.patch.object(tts.echo_cancel, 'was_canceled', return_value=False))
        self.enterContext(mock.patch.object(tts.delivery, 'allowed', return_value=True))
        for key, value in {
            'NO_AUDIO_MODE': False, 'AUDIO_OUTPUT_SUPPRESSED': False,
            'TTS_LOUDNESS_ENABLED': True, 'TTS_LOUDNESS_TARGET_LUFS': -20.0,
            'TTS_LOUDNESS_MAX_GAIN_DB': 26.0, 'TTS_LOUDNESS_PEAK_DBFS': -3.0,
            'TTS_LOUDNESS_PREROLL_MS': 300.0, 'TTS_LOUDNESS_WHISPER_OFFSET_DB': -4.0,
            'TTS_STREAM_END_PAD_MS': 0.0, 'TTS_STREAM_PCM_FORMAT': 'pcm_22050',
            'TTS_TRIM_TRAILING_SILENCE_ENABLED': False,
        }.items():
            self.enterContext(mock.patch.object(tts.config, key, value))

    def test_odd_byte_chunks_preserve_raw_cache_and_match_cached_leveling(self):
        data = self.raw.tobytes()
        originals = []
        out = list(self.tts._pcm_playback_chunks(
            iter(data[i:i+731] for i in range(0, len(data), 731)),
            self.tts._speech_leveler(22050, 'hello'), originals,
        ))
        np.testing.assert_array_equal(np.concatenate(originals), self.samples)
        np.testing.assert_array_equal(np.concatenate(out), level_audio(self.samples, 22050))

    def test_cancel_during_preroll_does_not_flush_buffered_speech(self):
        canceled = False
        def chunks():
            nonlocal canceled
            yield self.raw[:1000].tobytes()
            canceled = True
            yield self.raw[1000:2000].tobytes()
        with mock.patch.object(self.tts.echo_cancel, 'was_canceled', side_effect=lambda: canceled):
            self.assertEqual(list(self.tts._pcm_playback_chunks(chunks(), SpeechLeveler(22050), [])), [])

    def test_disabled_processor_passes_original_samples(self):
        with mock.patch.object(self.tts.config, 'TTS_LOUDNESS_ENABLED', False):
            processor = self.tts._speech_leveler(22050, 'hi')
        self.assertIsNone(processor)
        out = list(self.tts._pcm_playback_chunks(iter([self.raw.tobytes()]), processor, []))
        np.testing.assert_array_equal(np.concatenate(out), self.samples)

    def test_explicit_whisper_keeps_a_lower_target(self):
        ordinary = self.tts._speech_leveler(22050, 'hello').process(self.samples, final=True)
        whisper = self.tts._speech_leveler(22050, '[whispers] hello').process(self.samples, final=True)
        self.assertAlmostEqual(20*np.log10(np.linalg.norm(whisper)/np.linalg.norm(ordinary)), -4, delta=.1)

    def test_cached_wav_and_mp3_decode_are_leveled_without_rewriting_cache(self):
        tts = self.tts
        for suffix in ('.wav', '.mp3'):
            with self.subTest(suffix=suffix):
                path = self.root / ('take' + suffix)
                path.write_bytes(b'original cache data')
                with (
                    mock.patch.object(tts, '_cache_path', return_value=path),
                    mock.patch.object(tts, '_read_audio', return_value=(self.samples.copy(), 22050)),
                    mock.patch.object(tts, '_use_local_backend', return_value=False),
                    mock.patch('intelligence.gaze_engine.note_about_to_speak'),
                    mock.patch.object(tts, '_play') as play,
                    mock.patch.object(tts, '_get_el_client') as api,
                ):
                    tts.speak('Hello.', log_text=False)
                np.testing.assert_array_equal(play.call_args.args[0], level_audio(self.samples, 22050))
                api.assert_not_called()
                self.assertEqual(path.read_bytes(), b'original cache data')

    def test_stream_speaker_and_mouth_receive_same_leveled_samples_cache_stays_raw(self):
        tts = self.tts
        stream, pacer = mock.Mock(), mock.Mock()
        stream.latency = 0.0
        data = self.raw.tobytes()
        client = mock.Mock()
        client.text_to_speech.stream.return_value = iter(data[i:i+731] for i in range(0, len(data), 731))
        with (
            mock.patch('sounddevice.OutputStream', return_value=stream),
            mock.patch.object(tts, '_get_el_client', return_value=client),
            mock.patch.object(tts.output_gate, 'hold', return_value=contextlib.nullcontext(True)),
            mock.patch.object(tts.sd_guard, 'device_control', side_effect=lambda **kw: contextlib.nullcontext()),
            mock.patch.object(tts, '_begin_speech', return_value=(None, 'neutral')),
            mock.patch.object(tts, '_end_speech'),
            mock.patch.object(tts, '_MouthPacer', return_value=pacer),
        ):
            tts._speak_streaming('hello', 'hello', 'fake', 'fake', None, None, 'neutral', self.root/'stream.mp3', log_text=False)
        played = np.concatenate([c.args[0] for c in stream.write.call_args_list])
        mouth = np.concatenate([c.args[0] for c in pacer.push.call_args_list])
        np.testing.assert_array_equal(played, level_audio(self.samples, 22050))
        np.testing.assert_array_equal(mouth, played)
        cached, sr = sf.read(self.root/'stream.wav', dtype='float32')
        np.testing.assert_array_equal(cached, self.samples)
        self.assertEqual(sr, 22050)

    def test_failure_during_preroll_closes_request_without_opening_speaker(self):
        tts = self.tts
        closed = []
        def chunks():
            try:
                yield self.raw[:1000].tobytes()
                raise OSError('connection dropped during pre-roll')
            finally:
                closed.append(True)
        client = mock.Mock()
        client.text_to_speech.stream.return_value = chunks()
        with (
            mock.patch('sounddevice.OutputStream') as device,
            mock.patch.object(tts, '_get_el_client', return_value=client),
            mock.patch.object(tts, '_note_api_failure') as failed,
        ):
            handled = tts._speak_streaming('hi', 'hi', 'v', 'm', None, None, 'neutral', self.root/'fail.mp3', log_text=False)
        self.assertFalse(handled)
        self.assertEqual(closed, [True])
        failed.assert_called_once()
        device.assert_not_called()
        self.assertFalse((self.root/'fail.wav').exists())


if __name__ == '__main__':
    unittest.main()
