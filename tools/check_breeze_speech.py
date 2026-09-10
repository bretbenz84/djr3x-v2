#!/usr/bin/env python3
"""Reproduce the 09-09 startup failures with real streamed synthesis and local ASR.

No playback, microphone, serial or remote API calls. Requires Metal and locally
installed Breeze + Qwen ASR weights. Output files are diagnostics, never a cache.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LINES = (
    "Ready. And I've been ready for four seconds, so technically everyone else is the slow one now.",
    "Sit tight, still booting. I was a fresh install once. That was forty years and several regrettable firmware updates ago.",
)


def words(text):
    text = text.lower().replace('40', 'forty').replace('4', 'four')
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    def offline(*a, **kw):
        raise RuntimeError('Network forbidden in Breeze regression check')
    socket.socket.connect = offline
    import config
    config.LOCAL_TTS_BACKEND = 'breeze'
    from audio import local_tts
    from audio.transcription import _qwen_transcribe
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not local_tts.preload():
        raise RuntimeError('Breeze not available')
    rex = local_tts.rex_voice_ref()
    famous = ROOT / 'assets/voices/famous/jimmy-carter'
    clone = local_tts.voice_ref_from_files(famous.with_suffix('.wav'), famous.with_suffix('.txt'), 'famous:jimmy-carter')
    if rex is None or clone is None:
        raise RuntimeError('Reference pair missing')
    generate = local_tts._model.generate
    results = []
    for ref in (rex, clone):
        for seed in range(args.rounds):
            for index, text in enumerate(LINES):
                # Fixed seeds make failures reproducible; production remains stochastic.
                def seeded(**kwargs):
                    return generate(**dict(kwargs, seed=seed))
                local_tts._model.generate = seeded
                t0 = time.monotonic()
                take = local_tts.Take(text, ref)
                chunks, arrivals = [], []
                for chunk in take.stream():
                    chunks.append(chunk.copy())
                    arrivals.append(time.monotonic() - t0)
                wall = time.monotonic() - t0
                if not chunks:
                    raise RuntimeError('No audio produced')
                audio = np.concatenate(chunks)
                name = f'{ref.label.replace(":", "-")}-{seed}-{index}.wav'
                sf.write(args.out_dir / name, audio, local_tts.sample_rate())
                decoded, _ = _qwen_transcribe(resample_poly(audio, 2, 3), use_context=False)
                expected, actual = Counter(words(text)), Counter(words(decoded))
                coverage = sum((expected & actual).values()) / sum(expected.values())
                row = dict(file=name, expected=text, decoded=decoded, coverage=round(coverage, 3),
                           first_chunk_s=round(arrivals[0], 3), wall_s=round(wall, 3),
                           audio_s=round(len(audio) / local_tts.sample_rate(), 3),
                           repetition_penalty=config.BREEZE_TTS_REPETITION_PENALTY)
                results.append(row)
                print(json.dumps(row), flush=True)
                (args.out_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    # ASR is a content smoke check, not a listening-quality or hardware test.
    raise SystemExit(0 if all(r['coverage'] >= .95 for r in results) else 1)


if __name__ == '__main__':
    main()
