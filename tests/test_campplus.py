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
from intelligence.voice_bootstrap import target


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

    def test_production_first_turn_enrolls_and_attributes_then_recognizes(self):
        from intelligence import interaction as I
        audio = np.ones(int(config.AUDIO_SAMPLE_RATE * 2), dtype=np.float32) * .1
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(speaker_id, 'get_embedding', return_value=unit()), \
             patch.object(speaker_id, 'window_evidence', return_value=[]), \
             patch.object(speaker_id, 'voiced_secs', return_value=2), \
             patch.object(I, '_utterance_observations', {'visual': []}), \
             patch.object(I, '_last_confident_voice_at', {}):
            result = I._process_audio(audio, pretranscribed="My name is Bret.")
            self.assertEqual(result[1:3], (1, 'Bret'))
            self.assertEqual(people.count_native_voice_prints(1), 1)
            result = I._process_audio(audio, pretranscribed='What are we doing tomorrow?')
            self.assertEqual(result[1], 1)
            self.assertEqual(people.count_native_voice_prints(1), 1)

    def test_bootstrap_rejects_untrusted_short_mixed_or_disabled(self):
        from intelligence import interaction as I
        from audio.transcription import Transcript
        audio = np.ones(int(config.AUDIO_SAMPLE_RATE * 2), dtype=np.float32)
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(I, '_utterance_observations', {'visual': []}), \
             patch.object(I, '_last_scan_secs', {'voiced': 2}), \
             patch.object(I, '_last_scan_windows', []), \
             patch.object(I, '_safe_enroll_voice') as enroll:
            self.assertFalse(I._maybe_bootstrap_campplus(audio, Transcript('My name is Bret.', confident=False)))
            with patch.dict(I._last_scan_secs, voiced=.2):
                self.assertFalse(I._maybe_bootstrap_campplus(audio, 'My name is Bret.'))
            with patch.object(I, '_last_scan_windows', [{'change_suspected': True}]):
                self.assertFalse(I._maybe_bootstrap_campplus(audio, 'My name is Bret.'))
            with patch.object(config, 'CAMPPLUS_AUTO_ENROLL_ENABLED', False):
                self.assertFalse(I._maybe_bootstrap_campplus(audio, 'My name is Bret.'))
            self.assertFalse(I._maybe_bootstrap_campplus(None, 'My name is Bret.'))
            enroll.assert_not_called()

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

    def test_cam_growth_cannot_use_merely_visible_face(self):
        from intelligence import interaction as I
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(I, '_utterance_observations', {'visual': [{'faces': [{'person_db_id': 1}]}]}), \
             patch.object(I, '_last_scan_windows', []), \
             patch.object(I, '_safe_enroll_voice') as enroll:
            I._maybe_auto_refresh_voice(1, .99, np.ones(48000), face_confirmed=True, visual_speaker_pid=1)
            I._maybe_passive_voice_enroll('A clear full sentence.', np.ones(48000), 1, 1, .99)
            enroll.assert_not_called()

    def test_legacy_face_voice_agreement_seeds_without_mouth_motion(self):
        from intelligence import interaction as I
        from audio import voice_migration
        # Existing ECAPA Bret profile; the already-created PJ legacy profile
        # points in another direction. Neither is a CAM++ profile.
        db.execute("INSERT INTO biometrics(person_id,type,encoding) VALUES (1,'voice',?)", (unit(1).tobytes(),))
        face = {'person_db_id': 1, 'face_id': 'Bret', 'face_visible': True}
        audio = np.ones(int(config.AUDIO_SAMPLE_RATE*4), dtype=np.float32) * .1
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(speaker_id, 'get_embedding', return_value=unit(2)), \
             patch.object(speaker_id, 'voiced_secs', return_value=3.42), \
             patch.object(voice_migration, '_embedding', return_value=unit(1)) as legacy, \
             patch.object(I, '_utterance_observations', {'visual': []}), \
             patch.object(I.world_state, 'get', return_value=[face]), \
             patch.object(I, '_last_confident_voice_at', {}):
            result = I._process_audio(audio, pretranscribed="It's good to see you too, Rex. How are you doing today?")
            self.assertEqual(result[1:3], (1, 'Bret'))
            self.assertEqual(people.count_native_voice_prints(1), 1)
            legacy.assert_called_once()
            result = I._process_audio(audio, pretranscribed="What's on your plate for the day?")
            self.assertEqual(result[1], 1)
            legacy.assert_called_once()  # CAM++ alone after the first print
            self.assertEqual(voice_score.active_backend(), 'campplus')
            self.assertEqual(db.fetchone("SELECT count(*) AS n FROM biometrics WHERE type='voice'")['n'], 2)

    def test_legacy_foreign_voice_or_close_scores_cannot_seed_visible_owner(self):
        from audio import voice_migration
        db.execute("INSERT INTO biometrics(person_id,type,encoding) VALUES (1,'voice',?)", (unit(1).tobytes(),))
        for query in (unit(0), (unit(0)+unit(1))/np.sqrt(2)):
            with patch.object(voice_migration, '_embedding', return_value=query):
                proof = voice_migration.verify(np.ones(32000), 1)
            self.assertFalse(proof['accepted'])
        self.assertEqual(people.count_native_voice_prints(1), 0)

    def test_legacy_match_cannot_override_conflicting_explicit_identity(self):
        from intelligence import interaction as I
        from audio import voice_migration
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(I, '_utterance_observations', {'visual': [{'person_db_id': 2, 'confidence': .8}]*3}), \
             patch.object(I, '_last_scan_windows', []), \
             patch.object(I, '_last_scan_secs', {'voiced': 3}), \
             patch.object(I.world_state, 'get', return_value=[{'person_db_id': 2, 'face_id': 'PJ', 'face_visible': True}]), \
             patch.object(voice_migration, 'verify') as legacy:
            self.assertFalse(I._maybe_bootstrap_campplus(np.ones(48000), 'My name is Bret.'))
            legacy.assert_not_called()

    def test_full_existing_name_reply_enrolls_without_mouth_or_legacy_model(self):
        from intelligence import interaction as I
        from audio import voice_migration
        db.execute("UPDATE people SET name='Bret Benziger' WHERE id=1")
        face = {'person_db_id': 1, 'face_id': 'Bret Benziger', 'face_visible': True}
        audio = np.ones(int(config.AUDIO_SAMPLE_RATE*2.5), dtype=np.float32) * .1
        with patch.object(speaker_id, '_active_backend', 'campplus'), \
             patch.object(speaker_id, 'get_embedding', return_value=unit(1)), \
             patch.object(speaker_id, 'voiced_secs', return_value=1.5), \
             patch.object(I, '_utterance_observations', {'visual': []}), \
             patch.object(I.world_state, 'get', return_value=[face]), \
             patch.object(I, '_last_confident_voice_at', {}), \
             patch.object(voice_migration, 'verify') as legacy:
            result = I._process_audio(audio, pretranscribed='Bret Benziger.')
            self.assertEqual(result[1:3], (1, 'Bret Benziger'))
            self.assertEqual(people.count_native_voice_prints(1), 1)
            legacy.assert_not_called()

    def test_mouth_free_growth_requires_real_cam_voice_and_face_agreement(self):
        from intelligence import interaction as I
        people.add_biometric(1, 'voice', unit(1))
        face = {'person_db_id': 1, 'face_id': 'Bret', 'face_visible': True}
        with patch.object(I.world_state, 'get', return_value=[face]), \
             patch.object(I, '_turn_transcript_trusted', return_value=True), \
             patch.object(I, '_last_scan_windows', []):
            self.assertTrue(I._campplus_growth_supported(1, 1, .85))
            self.assertFalse(I._campplus_growth_supported(1, 2, .99))
            self.assertFalse(I._campplus_growth_supported(1, 1, .4))
            with patch.object(I, '_last_scan_windows', [{'person_id': 2}]):
                self.assertFalse(I._campplus_growth_supported(1, 1, .85))


class EvidenceTests(unittest.TestCase):
    def test_visible_face_without_active_speaker_is_insufficient(self):
        self.assertIsNone(target(observations=[{'faces': [{'person_db_id': 1}]}]*10, windows=[]))

    def test_sustained_active_speaker_bootstraps(self):
        self.assertEqual(target(observations=[{'person_db_id': 1, 'confidence': .8}]*3, windows=[]), 1)

    def test_different_active_speakers_refuse_whole_clip(self):
        rows = [{'person_db_id': p, 'confidence': .8} for p in (1, 1, 1, 2)]
        self.assertIsNone(target(observations=rows, windows=[], explicit_person_id=1))

    def test_unknown_speaker_change_blocks_first_print(self):
        self.assertIsNone(target(observations=[], windows=[{'change_suspected': True}], explicit_person_id=1))

    def test_conflicting_face_blocks_self_claim(self):
        self.assertIsNone(target(observations=[{'person_db_id': 2}], windows=[], explicit_person_id=1))

    def test_only_current_named_face_can_corroborate_legacy_voice(self):
        from intelligence.voice_bootstrap import visible_identity
        known = {'person_db_id': 1, 'face_id': 'Bret', 'face_visible': True}
        self.assertEqual(visible_identity([known]), 1)
        self.assertIsNone(visible_identity([dict(known, face_missing=True)]))
        self.assertIsNone(visible_identity([{'person_db_id': 1, 'voice_id': 'Bret'}]))
        self.assertIsNone(visible_identity([known, {'face_visible': True}]))


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
