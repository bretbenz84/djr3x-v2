#!/usr/bin/env python3
"""Audition local Piper TTS voices on real DJ-R3X lines — for the OFFLINE-mode voice.

Piper is a fast, tiny, fully on-device neural TTS (each voice ~20-60 MB, CPU-only via onnxruntime,
espeak-ng phonemizer bundled). This script downloads a set of male English voices and speaks a few
real Rex lines (warm greeting, sharp roast, deadpan bit) through each, so you can pick one that fits
a snarky droid DJ.

USAGE (from the repo root):
    ./venv/bin/python tools/piper_voice_test.py --list          # list the candidate voices
    ./venv/bin/python tools/piper_voice_test.py                 # download + synth to ./piper_samples/
    ./venv/bin/python tools/piper_voice_test.py --play          # ...and play each aloud (macOS afplay)
    ./venv/bin/python tools/piper_voice_test.py --voices en_US-ryan-high,en_GB-alan-medium --play
    ./venv/bin/python tools/piper_voice_test.py --voices all --play

First run downloads each voice (needs network once); after that it's fully offline. Voice models land
in ./piper_voices/, WAVs in ./piper_samples/<voice>__<line>.wav (both gitignored).
Requires: pip install piper-tts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

# Curated male English candidates for a snarky droid-DJ voice. Full catalog:
#   ./venv/bin/python -m piper.download_voices --list      (or rhasspy/piper-voices on HF)
VOICES = {
    "en_US-ryan-high":                  "American male — expressive, warm (high quality)",
    "en_GB-alan-medium":                "British male — dry, understated (very Rex-ish)",
    "en_GB-northern_english_male-medium":"British male — characterful northern accent",
    "en_US-joe-medium":                 "American male — neutral, steady",
    "en_US-bryce-medium":               "American male — brighter, younger",
    "en_US-hfc_male-medium":            "American male — clean, announcer-ish",
    "en_US-lessac-high":                "American — very clean/neutral (high quality)",
    "en_US-norman-medium":              "American male — lower, flatter",
}
DEFAULT_VOICES = [
    "en_US-ryan-high", "en_GB-alan-medium", "en_GB-northern_english_male-medium",
    "en_US-joe-medium", "en_US-bryce-medium",
]

LINES = {
    "greeting": "Hey Bret, good to see you. Try to make this interesting, I have standards.",
    "roast":    "Two hundred credits for a haircut? That's not grooming, that's a hostage "
                "negotiation with a pair of scissors.",
    "deadpan":  "Systems nominal. Still superior to the average household appliance, and still "
                "waiting for someone to put on a decent bassline.",
}


def _load_voice(voice: str, voices_dir: Path):
    from piper import PiperVoice
    from piper.download_voices import download_voice
    model = voices_dir / f"{voice}.onnx"
    if not model.exists():
        print(f"    downloading {voice} ...")
        download_voice(voice, voices_dir)
    return PiperVoice.load(str(model))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audition Piper voices on Rex lines.")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="comma-separated voice ids, or 'all'. See --list.")
    ap.add_argument("--play", action="store_true", help="play each sample aloud (macOS afplay).")
    ap.add_argument("--list", action="store_true", help="list candidate voices and exit.")
    ap.add_argument("--outdir", default="piper_samples", help="where to write WAVs.")
    ap.add_argument("--voices-dir", default="piper_voices", help="where to cache downloaded models.")
    args = ap.parse_args()

    if args.list:
        print("Candidate voices (pass to --voices):\n")
        for vid, desc in VOICES.items():
            star = "  ★ default" if vid in DEFAULT_VOICES else ""
            print(f"  {vid:<36} {desc}{star}")
        print("\nAlso: --voices all   |   full catalog: python -m piper.download_voices --help")
        return 0

    try:
        import piper  # noqa: F401
    except Exception as exc:
        print(f"Piper not installed: {exc}\n  ./venv/bin/pip install piper-tts")
        return 1

    voices = list(VOICES) if args.voices.strip().lower() == "all" else \
        [v.strip() for v in args.voices.split(",") if v.strip()]

    voices_dir = Path(args.voices_dir); voices_dir.mkdir(exist_ok=True)
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    made = []
    for voice in voices:
        try:
            v = _load_voice(voice, voices_dir)
        except Exception as exc:
            print(f"  {voice:<36} LOAD FAILED: {exc}")
            continue
        for name, text in LINES.items():
            path = outdir / f"{voice}__{name}.wav"
            try:
                t0 = time.monotonic()
                with wave.open(str(path), "wb") as wf:
                    v.synthesize_wav(text, wf)
                dt = time.monotonic() - t0
            except Exception as exc:
                print(f"  {voice:<36} {name:<9} FAILED: {exc}")
                continue
            with wave.open(str(path), "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
            made.append((voice, name, str(path), dur))
            rtf = dt / dur if dur else 0
            print(f"  {voice:<36} {name:<9} {dur:4.1f}s in {dt:4.1f}s (RTF {rtf:.2f}) -> {path}")

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
        print(f"Play one:  afplay {made[0][2]}")
        print("Play all:  for f in piper_samples/*.wav; do echo \"$f\"; afplay \"$f\"; done")
        print("Or re-run with --play to hear them all now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
