"""Jeffrey Davis field regression: reuse enrollment audio and retain short takes."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import config
from features import impersonation as P
from memory import database as db, voice_recordings as V


def speech(secs, rate=16000, pitch=180):
    return (.1 * np.sin(2*np.pi*pitch*np.arange(int(secs*rate))/rate)).astype(np.float32)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.temp)
        self.enterContext(patch.object(db, '_DB_FILE', self.path/'people.db'))
        self.enterContext(patch.object(config, 'VOICES_DIR', str(self.path/'voices')))
        self.enterContext(patch.object(config, 'TTS_CACHE_DIR', str(self.path/'cache')))
        self.enterContext(patch.object(config, 'AUDIO_SAMPLE_RATE', 16000))
        self.enterContext(patch.object(config, 'IMPERSONATION_CAPTURE_MIN_VOICED_SECS', 6.))
        self.enterContext(patch.object(P, 'is_enabled', return_value=True))
        with sqlite3.connect(self.path/'people.db') as conn:
            conn.executescript(DB_SCHEMA)
            conn.executemany('INSERT INTO people(id,name) VALUES (?,?)',
                             [(1, 'Bret Benziger'), (5, 'Jeffrey Davis')])

    def archive(self, rate=16000, verified=True):
        for secs, text in ((2.46, 'Do you know anything about Jimmy Carter?'),
                           (2.31, "How do you know where I'm at?"),
                           (4.11, "Uh, here. That's personal.")):
            data = io.BytesIO(); wavfile.write(data, rate, speech(secs, rate))
            blob = data.getvalue()
            meta = dict(source='conversational_enrollment', anchor_and_cluster_verified=verified,
                        transcript=text, voiced_secs=secs, sample_rate=rate)
            db.execute('INSERT INTO voice_recordings(person_id,digest,wav,metadata) VALUES(?,?,?,?)',
                       (5, hashlib.sha256(blob).hexdigest(), blob, json.dumps(meta)))

    def test_jeffrey_has_enough_original_audio_without_a_new_sample(self):
        self.archive()
        result = P.resolve_target('Jeffrey Davis', 1, 'Bret Benziger')
        self.assertEqual(result.kind, 'perform')
        self.assertEqual(result.person_id, 5)
        self.assertIn('Jimmy Carter', result.ref.ref_text)
        rate, audio = wavfile.read(result.ref.wav_path)
        self.assertEqual(rate, 16000)
        self.assertGreater(len(audio)/rate, 8.)
        self.assertEqual(len(V.list_recordings(5)), 3)
        self.assertEqual(db.fetchone('SELECT count(*) AS n FROM biometrics')['n'], 0)

    def test_actual_request_uses_archive_instead_of_opening_capture(self):
        from intelligence import interaction as I, action_router
        self.archive()
        with patch.object(I, '_pending_impersonation_capture', None), \
             patch.object(I, '_speak_blocking') as speak, \
             patch.object(P, 'perform', return_value='A Jeffrey impression.') as perform:
            text = 'Impersonate Jeffrey Davis.'
            result = I._handle_router_impersonation(action_router.classify_explicit_impersonation(text),
                text, 1, 'Bret Benziger', 'Jeffrey Davis')
            self.assertIsNone(I._pending_impersonation_capture)
        self.assertEqual(result, 'A Jeffrey impression.')
        self.assertEqual(perform.call_args.args[2], 5)
        speak.assert_not_called()

    def test_asr_name_variant_resolves_without_renaming(self):
        self.archive()
        for name in ('Jaffrey', 'Jeffery'):
            result = P.resolve_target(name, None, None)
            self.assertEqual(result.kind, 'perform')
            self.assertEqual(result.person_id, 5)
        self.assertEqual(db.fetchone('SELECT name FROM people WHERE id=5')['name'], 'Jeffrey Davis')

    def test_nickname_request_uses_the_known_persons_enrollment_audio(self):
        from intelligence import interaction as I, action_router
        self.archive()
        db.execute("UPDATE people SET name='Jeffery Benziger', nickname='Jeff' WHERE id=5")
        with patch.object(I, '_pending_impersonation_capture', None), \
             patch.object(I, '_speak_blocking') as speak, \
             patch.object(P, 'perform', return_value='A Jeff impression.') as perform:
            text = 'Impersonate Jeff.'
            result = I._handle_router_impersonation(action_router.classify_explicit_impersonation(text),
                text, 1, 'Bret Benziger', 'Jeff')
            self.assertIsNone(I._pending_impersonation_capture)
        self.assertEqual(result, 'A Jeff impression.')
        self.assertEqual(perform.call_args.args[0].label, 'person:5')
        self.assertEqual(perform.call_args.args[1:3], ('Jeffery Benziger', 5))
        speak.assert_not_called()

    def test_nickname_lookup_normalizes_case_and_spacing(self):
        self.archive()
        db.execute("UPDATE people SET name='Jeffery Benziger', nickname='Jeff' WHERE id=5")
        result = P.resolve_target('  jEfF  ', None, None)
        self.assertEqual((result.kind, result.person_id), ('perform', 5))

    def test_nickname_without_recordings_captures_for_the_existing_person(self):
        db.execute("UPDATE people SET name='Jeffery Benziger', nickname='Jeff' WHERE id=5")
        result = P.resolve_target('Jeff', None, None)
        self.assertEqual((result.kind, result.person_id, result.name),
                         ('capture', 5, 'Jeffery Benziger'))

    def test_shared_nickname_does_not_choose_someones_voice(self):
        self.archive()
        db.execute("UPDATE people SET nickname='Jeff' WHERE id=5")
        db.execute("INSERT INTO people(id,name,nickname) VALUES(6,'Geoffrey Smith','Jeff')")
        self.assertEqual(P.resolve_target('Jeff', None, None).kind, 'refuse')

    def test_close_names_are_not_guessed(self):
        self.archive()
        db.execute("INSERT INTO people(id,name) VALUES(6,'Jeffrey Smith')")
        self.assertEqual(P.resolve_target('Jaffrey', None, None).kind, 'refuse')

    def test_unknown_or_unverified_audio_cannot_be_used_as_someones_voice(self):
        self.archive(verified=False)
        self.assertIsNone(P.person_ref(5))
        self.assertIsNone(P.person_ref(1))

    def test_short_archive_waits_for_more_audio(self):
        self.archive()
        db.execute('DELETE FROM voice_recordings WHERE id != 1')
        self.assertIsNone(P.person_ref(5))

    def test_different_source_rate_keeps_whole_clips_and_transcripts(self):
        self.archive(rate=24000)
        result = P.person_ref(5)
        rate, audio = wavfile.read(result.wav_path)
        self.assertEqual(rate, 16000)
        self.assertGreater(len(audio)/rate, 8.)
        self.assertTrue(result.ref_text.endswith("That's personal."))

    def test_clearing_source_audio_invalidates_derived_reference(self):
        self.archive()
        ref = P.person_ref(5)
        V.clear(5)
        self.assertFalse(Path(ref.wav_path).exists())
        self.assertIsNone(P.person_ref(5))

    def test_existing_manual_reference_is_preserved(self):
        self.archive()
        ref = P.save_person_capture(5, speech(7), 'The manual reference stays mine.')
        self.assertEqual(P.person_ref(5), ref)

    def test_removing_a_bad_print_removes_its_derived_audio(self):
        from memory import people
        self.archive()
        db.execute("INSERT INTO biometrics(id,person_id,type,encoding) VALUES(42,5,'voice_campplus_zh_en_v1',?)",
                   (np.zeros(192, dtype=np.float32).tobytes(),))
        db.execute('UPDATE voice_recordings SET biometric_id=42 WHERE id=1')
        ref = P.person_ref(5)
        people.delete_biometric(42)
        self.assertFalse(Path(ref.wav_path).exists())
        self.assertEqual(len(V.list_recordings(5)), 2)

    def prepare_capture(self):
        from intelligence import interaction as I
        self.enterContext(patch.object(I, '_pending_impersonation_capture', {
            'person_id': 5, 'name': 'Jeffrey Davis', 'asked_at': time.monotonic(),
        }))
        self.enterContext(patch.object(I, '_last_scan_windows', []))
        self.enterContext(patch.object(I, '_turn_transcript_trusted', return_value=True))
        self.enterContext(patch.object(I, '_voiced_duration_secs', side_effect=lambda audio: len(audio)/16000))
        self.perform = self.enterContext(patch.object(P, 'perform', return_value='A Jeffrey impression.'))
        return I

    def test_partial_capture_survives_cancel_and_resumes_without_repeating(self):
        from intelligence import action_router
        I = self.prepare_capture()
        line, spoken = I._handle_impersonation_capture('I enjoyed exploring the museum today.',
                                                      speech(2.5), 5, 5, .8)
        self.assertFalse(spoken)
        self.assertIsNone(P.person_ref(5))
        self.assertAlmostEqual(P.capture_progress(5)['voiced_secs'], 2.5)
        I._handle_impersonation_capture('Never mind.', None, 5, 5, .8)
        self.assertIsNone(I._pending_impersonation_capture)
        self.assertIsNotNone(P.capture_progress(5))
        with patch.object(I, '_speak_blocking', return_value=True):
            result = I._handle_router_impersonation(
                action_router.classify_explicit_impersonation('Impersonate Jeffrey.'),
                'Impersonate Jeffrey.', 1, 'Bret', 'Jeffrey')
        self.assertIn('kept the sample', result)
        line, spoken = I._handle_impersonation_capture('We also spent some time walking through the park.',
                                                      speech(3.6, pitch=210), 5, 5, .8)
        self.assertTrue(spoken)
        self.assertEqual(line, 'A Jeffrey impression.')
        self.perform.assert_called_once()
        self.assertIsNone(P.capture_progress(5))
        self.assertIn('museum', P.person_ref(5).ref_text)
        self.assertIn('park', P.person_ref(5).ref_text)

    def test_complaint_bystander_and_duplicate_are_not_training_material(self):
        I = self.prepare_capture()
        audio = speech(2.5)
        I._handle_impersonation_capture('I enjoyed exploring the museum today.', audio, 5, 5, .8)
        for text, pid, raw, score in (('I said that already.', 5, 5, .8),
                                     ('I enjoyed exploring the museum today.', 5, 5, .8),
                                     ('My father is talking to you.', 1, 1, .9)):
            I._handle_impersonation_capture(text, audio, pid, raw, score)
        self.assertAlmostEqual(P.capture_progress(5)['voiced_secs'], 2.5)
        self.perform.assert_not_called()
        I._handle_impersonation_capture("He's not gonna be able to do this.", audio, 1, 1, .9)
        self.assertIsNone(I._pending_impersonation_capture)

    def test_expiry_keeps_saved_partial(self):
        I = self.prepare_capture()
        I._handle_impersonation_capture('I enjoyed the park today.', speech(2.5), 5, 5, .8)
        I._pending_impersonation_capture['asked_at'] = 0.
        result = I._handle_impersonation_capture('I said that already.', speech(3), 5, 5, .8)
        self.assertIn('kept that part', result[0])
        self.assertIsNotNone(P.capture_progress(5))

    def test_story_with_dont_is_not_mistaken_for_cancel(self):
        self.assertFalse(P.sounds_like_cancel("I don't usually visit museums but this one was fun."))

    def test_untrusted_or_mixed_capture_never_saves_a_reference(self):
        I = self.prepare_capture()
        with patch.object(I, '_turn_transcript_trusted', return_value=False):
            I._handle_impersonation_capture('It was a great day at the museum.', speech(8), 5, 5, .8)
        with patch.object(I, '_last_scan_windows', [{'person_id': 1}, {'person_id': 5}]):
            I._handle_impersonation_capture('It was a great day at the museum.', speech(8), 5, 5, .8)
        self.assertIsNone(P.person_ref(5))
        self.assertIsNone(P.capture_progress(5))
        self.perform.assert_not_called()

    def test_laughter_with_a_long_buffer_is_not_a_reference(self):
        I = self.prepare_capture()
        I._handle_impersonation_capture('Haha.', speech(8), 5, 5, .8)
        self.assertIsNone(P.person_ref(5))
        self.perform.assert_not_called()
