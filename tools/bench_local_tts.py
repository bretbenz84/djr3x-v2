#!/usr/bin/env python3
"""Measure local TTS to files, with networking blocked and no playback/hardware.

    venv/bin/python tools/bench_local_tts.py --backend breeze --out-dir /tmp/rex-breeze

Each voice gets two calls, showing cold-reference and warm-reference behavior.
Reports first-chunk latency and additional head start needed for uninterrupted
playback from the observed arrivals. It does not predict device/robot latency.
"""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('breeze', 'qwen'), default='breeze')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    def offline(*a, **kw):
        raise RuntimeError('Network forbidden in local TTS benchmark')
    socket.socket.connect = offline
    import config
    config.LOCAL_TTS_BACKEND = args.backend
    from audio import local_tts
    import numpy as np
    import soundfile as sf
    reason = local_tts.unavailable_reason(require_rex_ref=True)
    if reason:
        raise RuntimeError(reason)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not local_tts.preload():
        raise RuntimeError('Local TTS load failed')
    print(f'Load + first-chunk warmup: {time.perf_counter() - start:.2f}s', flush=True)
    famous = ROOT / 'assets/voices/famous/jimmy-carter'
    refs = [local_tts.rex_voice_ref(), local_tts.voice_ref_from_files(
        famous.with_suffix('.wav'), famous.with_suffix('.txt'), 'famous:jimmy-carter')]
    results = []
    for ref in refs:
        if ref is None:
            continue
        for call in range(2):
            text = 'Welcome to the cantina, everybody. Hold on tight and enjoy the music.'
            start = time.perf_counter()
            chunks, arrivals, durations = [], [], []
            for chunk in local_tts.generate_stream(text, ref):
                arrivals.append(time.perf_counter() - start)
                chunks.append(chunk)
                durations.append(chunk.size / local_tts.sample_rate())
            if not chunks:
                raise RuntimeError(f'No audio: {ref.label}')
            before = 0.0
            headstart = 0.0
            for arrival, duration in zip(arrivals, durations):
                headstart = max(headstart, arrival - arrivals[0] - before)
                before += duration
            wall = time.perf_counter() - start
            sf.write(args.out_dir / f'{args.backend}-{ref.label.replace(":", "-")}-{call}.wav',
                     np.concatenate(chunks), local_tts.sample_rate())
            row = dict(backend=args.backend, voice=ref.label, call=call,
                       first_chunk_s=round(arrivals[0], 3), headstart_s=round(headstart, 3),
                       to_speech_s=round(arrivals[0] + headstart, 3),
                       audio_s=round(before, 3), wall_s=round(wall, 3),
                       realtime_factor=round(wall / before, 3), chunks=len(chunks))
            results.append(row)
            print(json.dumps(row), flush=True)
    (args.out_dir / f'{args.backend}-results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
