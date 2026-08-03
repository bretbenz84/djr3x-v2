"""A/B: Qwen3-ASR decode latency vs context-prompt size.

Uses cached TTS wavs (real speech, Rex's voice) resampled to 16k mono, decodes
each under different context conditions, and reports wall time + transcript so
accuracy regressions are visible alongside the timing.
"""
import glob
import statistics
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/bbenziger/djr3x-v2")
import config
from audio import transcription

REPS = 4

def load_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        # linear resample is fine for a latency benchmark
        n = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n),
                         np.arange(len(data)), data).astype(np.float32)
    return data

# Pick 3 cached clips of sensible conversational length (1.5-4s)
clips = []
for p in sorted(glob.glob("/Users/bbenziger/djr3x-v2/assets/audio/tts_cache/*.wav")):
    a = load_16k(p)
    dur = len(a) / 16000
    if 1.5 <= dur <= 4.0:
        clips.append((p.split("/")[-1][:8], dur, a))
    if len(clips) == 3:
        break
assert clips, "no suitable cached wavs"
print("clips:", [(n, f"{d:.1f}s") for n, d, _ in clips])

# Realistic live Rex lines for the context (from today's field log)
transcription.note_rex_line(
    "A kite, huh—either you're very confident or very wrong; who's speaking?")
transcription.note_rex_line(
    "That colorful thing over there—kite, art project, or just the workshop's "
    "way of showing off?")

CONDITIONS = [
    # (label, bias_enabled, rex_lines, max_chars)
    ("no_context",          False, 0, 600),
    ("vocab_only",          True,  0, 600),
    ("vocab+1line",         True,  1, 600),
    ("vocab+2lines (live)", True,  2, 600),
    ("live capped 300",     True,  2, 300),
]

# Warm the model once
_ = transcription._qwen_transcribe(clips[0][2])

for label, enabled, n_lines, max_chars in CONDITIONS:
    config.QWEN_ASR_CONTEXT_BIAS_ENABLED = enabled
    config.QWEN_ASR_CONTEXT_REX_LINES = n_lines
    config.QWEN_ASR_CONTEXT_MAX_CHARS = max_chars
    prompt = transcription._asr_context_prompt()
    plen = len(prompt) if prompt else 0
    times = []
    texts = []
    for _, _, audio in clips:
        for _ in range(REPS):
            t0 = time.monotonic()
            text, _lp = transcription._qwen_transcribe(audio)
            times.append(time.monotonic() - t0)
        texts.append(text.strip()[:60])
    print(f"{label:22} prompt={plen:3d}ch  mean={statistics.mean(times):.3f}s  "
          f"median={statistics.median(times):.3f}s  "
          f"min={min(times):.3f}s max={max(times):.3f}s")
    for t in texts:
        print(f"    -> {t!r}")
