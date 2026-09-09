"""Accepted original audio, stored atomically with its voiceprints in people.db.

WAV blobs preserve float32 capture samples at their original rate, before model
resampling/trimming. Keeping the clips in the ignored database makes backup,
person merges and forgetting transactional; the CLI can export ordinary WAVs.
"""
import hashlib
import io
import json

import numpy as np
from scipy.io import wavfile

from memory import database as db

SCHEMA = """CREATE TABLE IF NOT EXISTS voice_recordings (
    id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id),
    digest TEXT NOT NULL, wav BLOB NOT NULL, metadata TEXT NOT NULL,
    biometric_id INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(person_id, digest)
)"""


def save_batch(person_id, samples, biometric_type, *, limit=10, print_limit=10):
    """All samples + embeddings commit together, or none do. No old print deletion."""
    if not samples:
        return False
    rows = []
    for s in samples:
        audio = np.asarray(s.audio, dtype=np.float32)
        embedding = np.asarray(s.embedding, dtype=np.float32).reshape(-1)
        if (audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()
                or not embedding.size or not np.isfinite(embedding).all() or s.rate <= 0):
            raise ValueError('invalid voice recording')
        out = io.BytesIO()
        wavfile.write(out, s.rate, audio)
        wav = out.getvalue()
        meta = dict(s.metadata, source='conversational_enrollment', transcript=s.text,
                    sample_rate=s.rate, voiced_secs=s.voiced, buffer_secs=len(audio)/s.rate,
                    bearing_deg=s.bearing, embedding_model=biometric_type,
                    anchor_and_cluster_verified=True)
        rows.append((hashlib.sha256(wav).hexdigest(), wav, json.dumps(meta, sort_keys=True), embedding.tobytes()))
    with db.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(SCHEMA)
        if not conn.execute('SELECT 1 FROM people WHERE id=?', (int(person_id),)).fetchone():
            return False
        count = conn.execute('SELECT count(DISTINCT encoding) FROM biometrics WHERE person_id=? AND type=?',
                             (person_id, biometric_type)).fetchone()[0]
        for digest, wav, meta, blob in rows:
            if conn.execute('SELECT 1 FROM voice_recordings WHERE person_id=? AND digest=?',
                            (person_id, digest)).fetchone():
                continue
            existing = conn.execute('SELECT id FROM biometrics WHERE person_id=? AND type=? AND encoding=?',
                                    (person_id, biometric_type, blob)).fetchone()
            bio = existing[0] if existing else None
            if bio is None and count < print_limit:
                bio = conn.execute(
                    'INSERT INTO biometrics(person_id,type,encoding,created_at) VALUES (?,?,?,CURRENT_TIMESTAMP)',
                    (person_id, biometric_type, blob)).lastrowid
                count += 1
            conn.execute('INSERT INTO voice_recordings(person_id,digest,wav,metadata,biometric_id) VALUES (?,?,?,?,?)',
                         (person_id, digest, wav, meta, bio))
        # Prefer long, useful speech; preserve a bounded varied collection, not
        # repeated captures of a single phrase. Print history itself stays intact.
        prune(conn, person_id, limit)
    return True


def prune(conn, person_id, limit=10):
    records = conn.execute('SELECT id,metadata FROM voice_recordings WHERE person_id=?', (person_id,)).fetchall()
    records = sorted(records, key=lambda r: (float(json.loads(r[1]).get('voiced_secs', 0)), r[0]), reverse=True)
    for row in records[max(1, int(limit)):]:
        conn.execute('DELETE FROM voice_recordings WHERE id=?', (row[0],))


def list_recordings(person_id):
    return [dict(r) for r in db.fetchall('SELECT * FROM voice_recordings WHERE person_id=? ORDER BY id', (person_id,))]


def decode(record):
    return wavfile.read(io.BytesIO(record['wav']))


def clear(person_id):
    db.execute('DELETE FROM voice_recordings WHERE person_id=?', (int(person_id),))


def invalidate_pending(person_id=None):
    # Don't import the conversation engine in a data-only/CLI process.
    import sys
    interaction = sys.modules.get('intelligence.interaction')
    learner = getattr(interaction, '_voice_learner', None)
    if learner and learner.pending and (person_id is None or learner.pending.person_id == person_id):
        learner.cancel('voice_data_deleted_or_merged')
