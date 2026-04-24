#!/usr/bin/env python3
"""
tools/test_voice_id.py — Voice-print identification diagnostic.

Records a short utterance from the mic, computes a Resemblyzer embedding,
and prints a ranked scoreboard against every enrolled voice in the DB.

Usage:
    python tools/test_voice_id.py                   # 5-second record + scan
    python tools/test_voice_id.py --secs 8          # longer recording
    python tools/test_voice_id.py --repeat 3        # 3 back-to-back samples
    python tools/test_voice_id.py --enroll "Name"   # re-enroll or add voice

No Rex stack required. Uses the project's config + DB + audio device.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import numpy as np
import sounddevice as sd

import config
from memory import database as db
from memory import people as people_mod
from audio import speaker_id
from utils.config_loader import AUDIO_DEVICE_INDEX


def _record(seconds: float) -> np.ndarray:
    print(f"  Recording {seconds:.1f}s... speak now.")
    frames = int(seconds * config.AUDIO_SAMPLE_RATE)
    audio = sd.rec(
        frames,
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype="float32",
        device=AUDIO_DEVICE_INDEX,
    )
    sd.wait()
    print("  Done.")
    if audio.ndim > 1:
        audio = audio[:, 0]
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"  RMS={rms:.4f}  peak={float(np.max(np.abs(audio))):.3f}")
    if rms < 0.005:
        print("  WARNING: very quiet — check mic / speak up / move closer.")
    return audio


def _scan_once(audio: np.ndarray) -> None:
    embedding = speaker_id.get_embedding(audio)
    if embedding is None:
        print("  ERROR: could not compute embedding (Resemblyzer unavailable?).")
        return

    rows = db.fetchall("SELECT person_id, encoding FROM biometrics WHERE type = 'voice'")
    if not rows:
        print("  No voice prints enrolled. Use --enroll <Name> to create one.")
        return

    query = embedding.astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-10)

    results = []
    for row in rows:
        stored = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if stored.shape != query.shape:
            continue
        stored = stored / (np.linalg.norm(stored) + 1e-10)
        sim = float(np.dot(stored, query))
        person = people_mod.get_person(row["person_id"])
        name = (person.get("name") if person else None) or "?"
        results.append((sim, row["person_id"], name))

    results.sort(reverse=True)

    threshold = config.SPEAKER_ID_SIMILARITY_THRESHOLD
    print(f"\n  Ranking (threshold={threshold:.2f}):")
    print(f"  {'score':>7}  {'id':>4}  verdict  name")
    print(f"  {'-----':>7}  {'--':>4}  -------  ----")
    for sim, pid, name in results:
        if sim >= 0.80:
            verdict = "HIGH   "
        elif sim >= threshold:
            verdict = "LOW-CONF"
        else:
            verdict = "REJECT "
        print(f"  {sim:>7.3f}  {pid:>4}  {verdict}  {name}")
    top = results[0]
    if top[0] >= threshold:
        print(f"\n  → would be identified as {top[2]} (person_id={top[1]}, score={top[0]:.3f})")
    else:
        print(f"\n  → would NOT be identified (best score {top[0]:.3f} < threshold {threshold:.2f})")


def _enroll(name: str, seconds: float) -> None:
    print(f"Enrolling voice for {name!r}.")
    audio = _record(seconds)
    pid_existing = None
    for row in db.fetchall("SELECT id, name FROM people WHERE LOWER(name) = LOWER(?)", (name,)):
        pid_existing = row["id"]
        break
    if pid_existing is not None:
        print(f"  Found existing person_id={pid_existing}, adding voice biometric.")
        pid = pid_existing
    else:
        pid = people_mod.enroll_person(name)
        print(f"  Created new person_id={pid}.")
    ok = speaker_id.enroll_voice(pid, audio)
    print(f"  Voice enrollment {'OK' if ok else 'FAILED'}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=5.0, help="Seconds to record (default 5)")
    ap.add_argument("--repeat", type=int, default=1, help="Number of back-to-back samples")
    ap.add_argument("--enroll", type=str, default=None, help="Enroll a new voice under this name instead of testing")
    args = ap.parse_args()

    if AUDIO_DEVICE_INDEX is None:
        print("ERROR: AUDIO_DEVICE_INDEX not set in .env")
        return 1

    print(f"Audio device index: {AUDIO_DEVICE_INDEX}")
    print(f"Sample rate: {config.AUDIO_SAMPLE_RATE} Hz\n")

    if args.enroll:
        _enroll(args.enroll, args.secs)
        return 0

    for i in range(args.repeat):
        print(f"── Sample {i + 1} of {args.repeat} ──")
        # brief countdown so the user can prepare
        for c in (3, 2, 1):
            print(f"  {c}...")
            time.sleep(0.6)
        audio = _record(args.secs)
        _scan_once(audio)
        if i + 1 < args.repeat:
            print()
            time.sleep(0.5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
