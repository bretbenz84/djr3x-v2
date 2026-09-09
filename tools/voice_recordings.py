#!/usr/bin/env python3
"""Inspect/export accepted voice clips or re-embed them with another model.

Examples:
  venv/bin/python tools/voice_recordings.py 'Jeff Benziger'
  venv/bin/python tools/voice_recordings.py 'Jeff Benziger' --export
  venv/bin/python tools/voice_recordings.py 'Jeff Benziger' --reembed campplus
  venv/bin/python tools/voice_recordings.py 'Jeff Benziger' --reembed campplus --apply

Re-embedding previews by default. --apply appends model-namespaced prints in one
transaction; it never removes a previous model or changes the active runtime.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reembed(records, embed, input_rate):
    """Compute every new vector before any storage; original WAVs are untouched."""
    import numpy as np
    from scipy.signal import resample_poly
    from memory.voice_recordings import decode
    results = []
    for row in records:
        rate, audio = decode(row)
        if rate != input_rate:
            divisor = math.gcd(rate, input_rate)
            audio = resample_poly(audio, input_rate//divisor, rate//divisor).astype(np.float32)
        vector = embed(audio)
        if vector is None or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
            raise RuntimeError(f'Embedding failed for recording {row["id"]}; nothing saved')
        results.append((row['id'], np.asarray(vector, dtype=np.float32).tobytes()))
    return results


def apply_embeddings(pid, model, results):
    from memory import database as db
    with db.connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        for recording_id, blob in results:
            row = conn.execute('SELECT metadata FROM voice_recordings WHERE id=? AND person_id=?',
                               (recording_id, pid)).fetchone()
            if row is None:
                raise RuntimeError('Recording removed or reassigned during re-embedding; nothing saved')
            existing = conn.execute('SELECT id FROM biometrics WHERE person_id=? AND type=? AND encoding=?',
                                    (pid, model, blob)).fetchone()
            bid = existing[0] if existing else conn.execute(
                'INSERT INTO biometrics(person_id,type,encoding,created_at) VALUES (?,?,?,CURRENT_TIMESTAMP)',
                (pid, model, blob)).lastrowid
            metadata = json.loads(row[0])
            metadata.setdefault('reembedded_models', {})[model] = bid
            conn.execute('UPDATE voice_recordings SET metadata=? WHERE id=?',
                         (json.dumps(metadata, sort_keys=True), recording_id))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('person')
    p.add_argument('--export', action='store_true', help='Export WAVs/metadata below ignored assets/memory/voice_exports/')
    p.add_argument('--reembed', choices=['campplus','ecapa','resemblyzer'])
    p.add_argument('--apply', action='store_true', help='Save the previewed re-embedding; preserves existing prints')
    args = p.parse_args()
    if args.apply and not args.reembed:
        p.error('--apply requires --reembed')
    if args.reembed:
        os.environ['VOICE_EMBEDDER'] = args.reembed
    from memory import people, voice_recordings
    person = people.find_person_by_name(args.person)
    if not person:
        p.error('No unique person matched that name')
    records = voice_recordings.list_recordings(person['id'])
    print(f'{person["name"]}: {len(records)} accepted original recordings')
    for row in records:
        meta = json.loads(row['metadata'])
        print(f'  {row["id"]}: {meta["voiced_secs"]:.2f}s speech, {meta["sample_rate"]} Hz, {row["created_at"]}')
    if args.export:
        dest = ROOT/'assets'/'memory'/'voice_exports'/str(person['id'])
        dest.mkdir(parents=True, exist_ok=True)
        for row in records:
            (dest/f'{row["id"]}.wav').write_bytes(row['wav'])
            (dest/f'{row["id"]}.json').write_text(json.dumps(json.loads(row['metadata']), indent=2)+'\n')
        print(f'Exported to {dest}')
    if args.reembed and records:
        import config
        from audio import speaker_id, voice_score
        results = reembed(records, speaker_id.get_embedding, int(config.AUDIO_SAMPLE_RATE))
        if speaker_id.active_backend() != args.reembed:
            raise RuntimeError('Requested model unavailable; refusing fallback-model writes')
        if args.apply:
            apply_embeddings(person['id'], voice_score.biometric_type(), results)
        print(f'{"Saved" if args.apply else "Previewed"} {len(results)} embeddings using {args.reembed}.')


if __name__ == '__main__':
    main()
