"""CAM++ migration, attribution and enrollment without recording/network access."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import config
from audio import speaker_id, voice_score
from memory import database as db, people, voice_signatures as signatures


def unit(index=0):
    result = np.zeros(192, dtype=np.float32)
    result[index] = 1
    return result


class StorageTests(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'people.db'
        with sqlite3.connect(self.path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.executemany('INSERT INTO people (id, name) VALUES (?, ?)', [(1, 'Bret'), (2, 'PJ')])
            # An EXACT same-dimension legacy match must never name CAM++ audio.
            conn.execute("INSERT INTO biometrics(person_id,type,encoding) VALUES (2,'voice',?)", (unit().tobytes(),))
        p = patch.object(db, '_DB_FILE', self.path)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(voice_score, '_active_backend', 'campplus')
        p.start()
        self.addCleanup(p.stop)
        signatures.reset_table_cache()
        self.addCleanup(signatures.reset_table_cache)

    def test_legacy_192_print_cannot_match_or_prevent_enrollment(self):
        self.assertEqual(speaker_id.rank_embedding(unit()), [])
        self.assertIsNone(people.find_by_voice(unit()))
        self.assertFalse(people.has_voice_biometric(2))
        self.assertEqual(speaker_id.comparable_print_count(2), 0)
        people.add_biometric(1, 'voice', unit())
        self.assertEqual(speaker_id.rank_embedding(unit())[0][0], 1)
        self.assertEqual(people.find_by_voice(unit())['id'], 1)
        self.assertEqual(people.count_native_voice_prints(1), 1)
        self.assertEqual(people.count_biometrics(1, 'voice'), 1)
        with patch.object(voice_score, '_active_backend', 'ecapa'):
            self.assertEqual(speaker_id.rank_embedding(unit())[0][0], 2)
            self.assertEqual(people.count_native_voice_prints(1), 0)

    def test_signatures_do_not_cross_model_boundary(self):
        with patch.object(voice_score, '_active_backend', 'ecapa'):
            signatures.record(unit(), label='legacy')
        self.assertIsNone(signatures.match(unit()))
        sid = signatures.record(unit(), label='cam')
        self.assertIsNotNone(sid)
        signatures.bump(sid, unit())
        signatures.attach_person(sid, 1)
        match = signatures.match(unit())
        self.assertEqual((match['person_id'], match['turns']), (1, 2))
        with patch.object(voice_score, '_active_backend', 'ecapa'):
            self.assertIsNone(signatures.match(unit())['person_id'])

    def test_admin_clear_preserves_rollback_prints(self):
        from memory import admin
        people.add_biometric(2, 'voice', unit(1))
        self.assertTrue(admin.clear_biometrics(2, 'voice'))
        self.assertEqual(people.count_native_voice_prints(2), 0)
        with patch.object(voice_score, '_active_backend', 'ecapa'):
            self.assertEqual(people.count_native_voice_prints(2), 1)






    def test_failed_storage_is_not_reported_as_enrollment(self):
        with patch.object(speaker_id, 'get_embedding', return_value=unit()), \
             patch.object(people, 'add_biometric', return_value=None):
            self.assertFalse(speaker_id.enroll_voice(1, np.zeros(32000)))

    def test_unknown_window_difference_is_diagnostic_not_proof_of_two_speakers(self):
        audio = np.ones(int(config.AUDIO_SAMPLE_RATE * 4), dtype=np.float32) * .1
        with patch.object(speaker_id, 'get_embedding', side_effect=[unit(0), unit(1)]), \
             patch.object(speaker_id, 'voiced_secs', return_value=1.5):
            windows = speaker_id.window_evidence(audio)
        self.assertEqual(len(windows), 2)
        self.assertTrue(windows[1]['acoustic_change_suspected'])
        self.assertFalse(windows[1]['change_suspected'])
        self.assertIsNone(windows[0]['person_id'])










class EncoderTests(unittest.TestCase):
    def test_failure_does_not_fallback_or_mix_spaces(self):
        with patch.object(config, 'VOICE_EMBEDDER', 'campplus'), \
             patch.object(config, 'CAMPPLUS_MODEL_PATH', '/nonexistent/campplus.onnx'), \
             patch.object(speaker_id, '_encoder', None), \
             patch.object(speaker_id, '_UNAVAILABLE', False), \
             patch.object(speaker_id, '_load_ecapa') as ecapa, \
             patch.object(speaker_id, '_load_resemblyzer') as legacy:
            self.assertIsNone(speaker_id._get_encoder())
            ecapa.assert_not_called()
            legacy.assert_not_called()

    def test_real_cpu_model_contract_and_audio_preprocessing(self):
        from audio.campplus import Encoder
        path = Path(__file__).resolve().parents[1] / config.CAMPPLUS_MODEL_PATH
        if not path.exists():
            self.skipTest('Run tools/download_campplus.py for the real-model integration check')
        encoder = Encoder(path)
        wave = (.1 * np.sin(2*np.pi*173*np.arange(32000)/16000)).astype(np.float32)
        emb = encoder.embed(wave, 16000)
        self.assertEqual(emb.shape, (192,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1, places=5)
        self.assertTrue(np.isfinite(encoder.embed(wave[::2], 8000)).all())
        self.assertTrue(np.isfinite(encoder.embed(wave[:8000], 16000)).all())
        for invalid in (np.zeros(32000), np.full(32000, np.nan), wave[:1000]):
            self.assertIsNone(encoder.embed(invalid, 16000))
        self.assertIsNone(encoder.embed(wave, 0))
        # Only inference/shape checks; synthetic tones cannot establish accuracy.

    def test_cam_scores_have_no_ecapa_offset(self):
        with patch.object(voice_score, '_active_backend', 'campplus'):
            self.assertEqual(voice_score.map_similarity(.4), .4)
            self.assertEqual(voice_score.match_threshold(), config.CAMPPLUS_MATCH_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
