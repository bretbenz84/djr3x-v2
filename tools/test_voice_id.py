#!/usr/bin/env python3
"""
tools/test_voice_id.py — Voice-print identification diagnostic.

Records a short utterance from the mic, computes a Resemblyzer embedding,
and prints a ranked scoreboard against every enrolled voice in the DB.

Usage:
    python tools/test_voice_id.py                        # 5-second record + scan
    python tools/test_voice_id.py --secs 8               # longer recording
    python tools/test_voice_id.py --repeat 3             # 3 back-to-back samples
    python tools/test_voice_id.py --enroll "Name"        # add a voice biometric
    python tools/test_voice_id.py --enroll "Name" --replace
                                                         # replace ALL prior voice rows with a fresh one
    python tools/test_voice_id.py --trim "Name"          # keep only newest voice row for Name
    python tools/test_voice_id.py --trim-all             # keep newest voice row per person, DB-wide

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
from utils.config_loader import AUDIO_DEVICE_INDEX, AUDIO_SELECTION_DESCRIPTION


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
    # Use the SAME production matcher (per-person centroid, one entry per person).
    results = speaker_id.rank_speakers(audio)  # [(person_id, name, sim)] sorted desc
    if not results:
        print("  No voice prints enrolled / embedding failed. Use --enroll <Name>.")
        return

    threshold = config.SPEAKER_ID_SIMILARITY_THRESHOLD
    margin = float(getattr(config, "SPEAKER_ID_KNOWN_MARGIN", 0.0) or 0.0)
    print(f"\n  Ranking (centroid; threshold={threshold:.2f}, margin={margin:.2f}):")
    print(f"  {'score':>7}  {'id':>4}  verdict   name")
    print(f"  {'-----':>7}  {'--':>4}  -------   ----")
    for pid, name, sim in results:
        if sim >= 0.80:
            verdict = "HIGH    "
        elif sim >= threshold:
            verdict = "LOW-CONF"
        else:
            verdict = "REJECT  "
        print(f"  {sim:>7.3f}  {pid:>4}  {verdict}  {name}")

    best_pid, best_name, best_sim = results[0]
    second = results[1][2] if len(results) > 1 else -1.0
    if best_sim >= threshold and (best_sim - second) >= margin:
        print(f"\n  → would be identified as {best_name} "
              f"(person_id={best_pid}, score={best_sim:.3f}, margin={best_sim - second:.3f})")
    elif best_sim >= threshold:
        print(f"\n  → AMBIGUOUS: best {best_name}={best_sim:.3f} but only "
              f"{best_sim - second:.3f} over the next person (< {margin:.2f} margin)")
    else:
        print(f"\n  → would NOT be identified (best {best_sim:.3f} < threshold {threshold:.2f})")


def _find_person_id(name: str) -> int | None:
    row = db.fetchone(
        "SELECT id FROM people WHERE LOWER(name) = LOWER(?)", (name,)
    )
    return row["id"] if row else None


def _trim_voices_for(person_id: int, label: str) -> int:
    """Keep only the newest voice biometric for person_id. Returns rows deleted."""
    rows = db.fetchall(
        "SELECT id, created_at FROM biometrics WHERE person_id = ? AND type = 'voice' "
        "ORDER BY created_at DESC",
        (person_id,),
    )
    if len(rows) <= 1:
        print(f"  {label}: {len(rows)} voice row(s); nothing to trim.")
        return 0
    keep = rows[0]
    drop_ids = [r["id"] for r in rows[1:]]
    for rid in drop_ids:
        db.execute("DELETE FROM biometrics WHERE id = ?", (rid,))
    print(f"  {label}: kept biometric id={keep['id']} ({keep['created_at']}), "
          f"dropped {len(drop_ids)} older row(s): {drop_ids}")
    return len(drop_ids)


def _enroll(name: str, seconds: float, replace: bool = False) -> None:
    print(f"Enrolling voice for {name!r}{' (replace mode)' if replace else ''}.")
    audio = _record(seconds)
    pid_existing = _find_person_id(name)
    if pid_existing is not None:
        print(f"  Found existing person_id={pid_existing}, adding voice biometric.")
        pid = pid_existing
    else:
        pid = people_mod.enroll_person(name)
        print(f"  Created new person_id={pid}.")
    ok = speaker_id.enroll_voice(pid, audio)
    print(f"  Voice enrollment {'OK' if ok else 'FAILED'}.")
    if ok and replace:
        _trim_voices_for(pid, name)


def _trim_named(name: str) -> None:
    pid = _find_person_id(name)
    if pid is None:
        print(f"  No person named {name!r} found.")
        return
    _trim_voices_for(pid, f"{name} (person_id={pid})")


def _trim_all() -> None:
    rows = db.fetchall("SELECT id, name FROM people ORDER BY id")
    total_dropped = 0
    for row in rows:
        total_dropped += _trim_voices_for(row["id"], f"{row['name']} (person_id={row['id']})")
    print(f"\nTotal voice biometrics removed: {total_dropped}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=5.0, help="Seconds to record (default 5)")
    ap.add_argument("--repeat", type=int, default=1, help="Number of back-to-back samples")
    ap.add_argument("--enroll", type=str, default=None, help="Enroll a new voice under this name instead of testing")
    ap.add_argument("--replace", action="store_true", help="With --enroll: after adding the new row, delete older voice rows for this person")
    ap.add_argument("--trim", type=str, default=None, help="Keep only the newest voice biometric for NAME")
    ap.add_argument("--trim-all", action="store_true", help="Keep only the newest voice biometric per person, DB-wide")
    args = ap.parse_args()

    # Trim modes don't need the mic.
    if args.trim:
        print(f"Trimming voice biometrics for {args.trim!r}...")
        _trim_named(args.trim)
        return 0
    if args.trim_all:
        print("Trimming older voice biometrics across all people...")
        _trim_all()
        return 0

    if AUDIO_DEVICE_INDEX is None:
        print("ERROR: AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env")
        return 1

    print(f"Audio device: {AUDIO_SELECTION_DESCRIPTION}")
    print(f"Sample rate: {config.AUDIO_SAMPLE_RATE} Hz\n")

    if args.enroll:
        _enroll(args.enroll, args.secs, replace=args.replace)
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
