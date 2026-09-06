"""Read-only legacy voice verification for first CAM++ enrollment.

Re-embed captured audio with the old model; never reinterpret old vectors as
CAM++ and never change the active backend. A separately recognized face must
agree with the legacy winner. Normal recognition remains entirely CAM++.
"""
import logging

import numpy as np
import config
from memory import database as db

_log = logging.getLogger(__name__)
_encoders = {}


def _embedding(audio, backend):
    from audio import speaker_id
    if backend not in _encoders:
        _encoders[backend] = (speaker_id._load_ecapa() if backend == 'ecapa'
                              else speaker_id._load_resemblyzer())
    encoder = _encoders[backend]
    if encoder is None:
        return None
    if backend == 'ecapa':
        return speaker_id._embed_ecapa(encoder, audio)
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(audio, source_sr=config.AUDIO_SAMPLE_RATE)
    return encoder.embed_utterance(wav)


def verify(audio, visible_person_id):
    """Return diagnostic evidence, including accepted=False when unverified."""
    result = {'accepted': False, 'reason': 'no_legacy_profile'}
    rows = db.fetchall("SELECT person_id, encoding FROM biometrics WHERE type='voice'")
    for backend, dim, threshold in (
        ('ecapa', 192, float(getattr(config, 'CAMPPLUS_MIGRATION_ECAPA_MIN_COSINE', .45))),
        ('resemblyzer', 256, float(getattr(config, 'CAMPPLUS_MIGRATION_RESEMBLYZER_MIN_COSINE', .75))),
    ):
        native = [r for r in rows if len(r['encoding']) == dim*4]
        if not any(r['person_id'] == visible_person_id for r in native):
            continue
        try:
            query = _embedding(audio, backend)
            if query is None:
                result = {'accepted': False, 'reason': 'legacy_model_unavailable', 'backend': backend}
                continue
            query = np.asarray(query, dtype=np.float32).reshape(-1)
            if query.size != dim or not np.isfinite(query).all() or np.linalg.norm(query) < 1e-10:
                continue
            query = query / np.linalg.norm(query)
            by_person = {}
            for row in native:
                vec = np.frombuffer(row['encoding'], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if not np.isfinite(vec).all() or norm < 1e-10:
                    continue
                by_person.setdefault(row['person_id'], []).append(vec/norm)
            scored = []
            for pid, vecs in by_person.items():
                centroid = np.mean(vecs, axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-10)
                scored.append((pid, float(np.dot(query, centroid))))
            scored.sort(key=lambda row: row[1], reverse=True)
            if not scored:
                continue
            pid, score = scored[0]
            margin = score - (scored[1][1] if len(scored)>1 else -1.)
            accepted = (pid == visible_person_id and score >= threshold
                        and margin >= float(getattr(config, 'CAMPPLUS_MIGRATION_MIN_MARGIN', .10)))
            result = {'accepted': accepted, 'reason': 'legacy_voice_and_face_agree' if accepted else 'legacy_voice_not_confirmed',
                      'backend': backend, 'person_id': pid, 'cosine': round(score, 4), 'margin': round(margin, 4)}
            # A contradictory legacy profile is not an invitation to shop another model.
            return result
        except Exception as exc:
            _log.warning('CAM++ migration verification unavailable (%s): %s', backend, exc)
            result = {'accepted': False, 'reason': 'legacy_verification_error', 'backend': backend}
    return result
