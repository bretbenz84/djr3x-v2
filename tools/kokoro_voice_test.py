#!/usr/bin/env python3
"""Audition local Kokoro TTS voices on real DJ-R3X lines — for the OFFLINE-mode voice.

Kokoro-82M is a small, fast, fully on-device neural TTS (~350 MB, no network). This script speaks a
handful of real Rex lines (a warm greeting, a sharp roast, a deadpan bit) through a set of voices so
you can pick the one that fits a snarky droid DJ, then wire that choice into an offline TTS backend.

USAGE (from the repo root):
    ./venv/bin/python tools/kokoro_voice_test.py --list          # list the candidate voices
    ./venv/bin/python tools/kokoro_voice_test.py                 # synth samples to ./kokoro_samples/
    ./venv/bin/python tools/kokoro_voice_test.py --play          # ...and play each one aloud (macOS afplay)
    ./venv/bin/python tools/kokoro_voice_test.py --voices am_onyx,bm_george --play
    ./venv/bin/python tools/kokoro_voice_test.py --voices all --play           # every candidate

First run downloads the Kokoro model (~350 MB) + a spaCy English model. After that it's fully offline.
WAVs land in ./kokoro_samples/<voice>__<line>.wav (throwaway — gitignored).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz

# Curated candidates for a male, snarky droid-DJ voice. am_ = American male, bm_ = British male,
# af_/bf_ = female (a couple included for contrast). Full voice list: hexgrad/Kokoro-82M on HF.
VOICES = {
    "am_michael": "American male — warm, neutral (a safe default)",
    "am_adam":    "American male — younger, bright",
    "am_onyx":    "American male — deep, smooth (characterful)",
    "am_fenrir":  "American male — gravelly, lower",
    "am_puck":    "American male — playful, expressive",
    "am_eric":    "American male — crisp, dry",
    "am_liam":    "American male — even, announcer-ish",
    "bm_george":  "British male — dry, deadpan (very Rex-ish)",
    "bm_fable":   "British male — theatrical, characterful",
    "bm_lewis":   "British male — measured, cool",
    "af_heart":   "American female — warm (contrast)",
    "bf_emma":    "British female — crisp (contrast)",
}
DEFAULT_VOICES = ["am_michael", "am_onyx", "am_fenrir", "am_puck", "bm_george", "bm_fable"]

# Real Rex lines that span his range: warmth, a sharp roast, and a deadpan bit.
LINES = {
    "greeting": "Hey Bret, good to see you. Try to make this interesting, I have standards.",
    "roast":    "Two hundred credits for a haircut? That's not grooming, that's a hostage "
                "negotiation with a pair of scissors.",
    "deadpan":  "Systems nominal. Still superior to the average household appliance, and still "
                "waiting for someone to put on a decent bassline.",
}


def _synth(pipeline, text, voice):
    import numpy as np
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        return None
    return np.concatenate(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audition Kokoro voices on Rex lines.")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="comma-separated voice ids, or 'all'. See --list.")
    ap.add_argument("--play", action="store_true", help="play each sample aloud (macOS afplay).")
    ap.add_argument("--list", action="store_true", help="list candidate voices and exit.")
    ap.add_argument("--outdir", default="kokoro_samples", help="where to write WAVs.")
    args = ap.parse_args()

    if args.list:
        print("Candidate voices (pass to --voices):\n")
        for vid, desc in VOICES.items():
            star = "  ★ default" if vid in DEFAULT_VOICES else ""
            print(f"  {vid:<12} {desc}{star}")
        print("\nAlso: --voices all   (every one above)")
        return 0

    voices = list(VOICES) if args.voices.strip().lower() == "all" else \
        [v.strip() for v in args.voices.split(",") if v.strip()]
    unknown = [v for v in voices if v not in VOICES]
    if unknown:
        print(f"note: {unknown} not in the curated list — trying anyway (Kokoro may still have them).")

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading Kokoro (first run downloads ~350 MB)...")
    try:
        from kokoro import KPipeline
        import soundfile as sf
    except Exception as exc:
        print(f"Kokoro not installed: {exc}\n  ./venv/bin/pip install kokoro soundfile misaki")
        return 1
    pipeline = KPipeline(lang_code="a")  # American English G2P; works for the British voices too

    made = []
    for voice in voices:
        for name, text in LINES.items():
            try:
                t0 = time.monotonic()
                audio = _synth(pipeline, text, voice)
                dt = time.monotonic() - t0
            except Exception as exc:
                print(f"  {voice:<12} {name:<9} FAILED: {exc}")
                continue
            if audio is None:
                print(f"  {voice:<12} {name:<9} (no audio)")
                continue
            path = os.path.join(args.outdir, f"{voice}__{name}.wav")
            sf.write(path, audio, SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
            made.append((voice, name, path, dur))
            rtf = dt / dur if dur else 0
            print(f"  {voice:<12} {name:<9} {dur:4.1f}s audio in {dt:4.1f}s (RTF {rtf:.2f})  -> {path}")

    print(f"\n{len(made)} samples in ./{args.outdir}/")
    if not made:
        return 1

    if args.play:
        if sys.platform != "darwin":
            print("(--play uses macOS 'afplay'; skipping on this platform.)")
        else:
            print("\nPlaying — Ctrl-C to stop:\n")
            for voice, name, path, _ in made:
                print(f"  ▶ {voice}  [{name}]")
                subprocess.run(["afplay", path])
                time.sleep(0.3)
    else:
        first = made[0][2]
        print(f"Play one:  afplay {first}")
        print("Play all:  for f in kokoro_samples/*.wav; do echo \"$f\"; afplay \"$f\"; done")
        print("Or re-run with --play to hear them all now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
