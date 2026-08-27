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
    """Record from the configured device, discovering a channel count PortAudio
    will actually OPEN (mirrors audio/stream.py: query_devices can report more
    channels than Pa_OpenStream accepts on macOS, so trying-and-catching the real
    open is the only reliable probe — config asks for 2 for the ReSpeaker AEC
    channel while e.g. the MacBook mic opens 1-ch only)."""
    print(f"  Recording {seconds:.1f}s... speak now.")
    frames = int(seconds * config.AUDIO_SAMPLE_RATE)

    requested = int(getattr(config, "AUDIO_INPUT_CHANNELS", config.AUDIO_CHANNELS) or 1)
    candidates = [requested]
    try:
        max_in = int(sd.query_devices(AUDIO_DEVICE_INDEX).get("max_input_channels") or 0)
        if max_in and max_in not in candidates:
            candidates.append(max_in)
    except Exception:
        pass
    if 1 not in candidates:
        candidates.append(1)

    audio = None
    last_exc = None
    # Second pass: the system default input. macOS device INDICES float between
    # sessions (live failure 2026-07-06: AUDIO_DEVICE_INDEX=1 pointed at the
    # SPEAKERS — zero input channels — while the mic had moved to index 0), so a
    # stale configured index must not brick enrollment. Prefer AUDIO_DEVICE_NAME
    # in .env for a stable selection.
    for device in (AUDIO_DEVICE_INDEX, None):
        for ch in candidates:
            try:
                audio = sd.rec(
                    frames,
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=ch,
                    dtype="float32",
                    device=device,
                )
            except Exception as exc:
                last_exc = exc
                continue
            if device is None:
                try:
                    name = sd.query_devices(sd.default.device[0])["name"]
                except Exception:
                    name = "default"
                print(f"  (configured device {AUDIO_DEVICE_INDEX} unusable — "
                      f"recording from system default input: {name})")
            elif ch != requested:
                print(f"  (device rejected {requested}-ch; recording {ch}-ch)")
            break
        if audio is not None:
            break
    if audio is None:
        sys.exit(f"could not open any input device "
                 f"(tried device {AUDIO_DEVICE_INDEX} and default, "
                 f"channels {candidates}): {last_exc}")
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
    results = speaker_id.rank_speakers(audio)  # [(person_id, name, sim, n_prints)] sorted desc
    if not results:
        print("  No voice prints enrolled / embedding failed. Use --enroll <Name>.")
        return

    threshold = config.SPEAKER_ID_SIMILARITY_THRESHOLD
    # Scoreboard-specific bar (thin-challenger relief), same as the live path.
    margin = speaker_id.required_ambiguity_margin(results)
    print(f"\n  Ranking (centroid; threshold={threshold:.2f}, required margin={margin:.3f}):")
    print(f"  {'score':>7}  {'id':>4}  {'prints':>6}  verdict   name")
    print(f"  {'-----':>7}  {'--':>4}  {'------':>6}  -------   ----")
    for pid, name, sim, n_prints in results:
        if sim >= 0.80:
            verdict = "HIGH    "
        elif sim >= threshold:
            verdict = "LOW-CONF"
        else:
            verdict = "REJECT  "
        print(f"  {sim:>7.3f}  {pid:>4}  {n_prints:>6}  {verdict}  {name}")

    best_pid, best_name, best_sim, _n = results[0]
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
    """Resolve a spoken/typed name to a person row the way the live stack does.

    Exact name first, then aliases and unique first tokens via
    people.find_person_by_name — "PJ" is an alias of the row named "PJ Thomas",
    and an exact-only lookup silently SPLIT such a person in two here: --enroll
    "PJ" created a second row and put the voice there while the face stayed on
    the original (the face tool hit the same trap, fixed f201e94)."""
    row = db.fetchone(
        "SELECT id FROM people WHERE LOWER(name) = LOWER(?)", (name,)
    )
    if row:
        return row["id"]
    try:
        resolved = people_mod.find_person_by_name(name)
    except Exception:
        resolved = None
    if resolved:
        print(f"  resolved {name!r} → {resolved['name']!r} (person_id={resolved['id']})")
        return resolved["id"]
    return None


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
    # Resolve the person BEFORE recording: you should know whose row you are
    # about to write to while you can still Ctrl-C, and a typo must not mint a
    # phantom person row that then competes with the real one in every scan.
    pid = _find_person_id(name)
    if pid is None:
        rows = db.fetchall("SELECT id, name FROM people ORDER BY id")
        print(f"\n  No person matches {name!r}. People on file:")
        for row in rows:
            print(f"    {row['id']:>3}  {row['name']}")
        answer = input(f"\n  Create a NEW person named {name!r}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  Aborted — nothing written.")
            return
        pid = people_mod.enroll_person(name)
        print(f"  Created new person_id={pid}.")
    else:
        existing = db.fetchone(
            "SELECT COUNT(*) AS n FROM biometrics WHERE person_id = ? AND type='voice'",
            (pid,),
        )["n"]
        print(f"  Target: person_id={pid} ({existing} voice row(s) on file).")

    audio = _record(seconds)
    ok = speaker_id.enroll_voice(pid, audio)
    print(f"  Voice enrollment {'OK' if ok else 'FAILED'}.")
    if ok and replace:
        _trim_voices_for(pid, name)
    if ok:
        # Immediate feedback: score the clip you just enrolled against every
        # print, so a cross-matching twin shows up now rather than mid-party.
        _scan_once(audio)


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


# Calibration rounds: sample the SAME voice across the conditions that actually vary
# in the field (distance, level, head angle, utterance length) — the self-score band
# this produces is the ground truth for setting thresholds.
_CALIBRATION_ROUNDS = [
    "Speak NORMALLY at your usual distance from the robot.",
    "Normal again — different sentence, keep talking the whole window.",
    "From ACROSS THE ROOM (a few meters back).",
    "Speak SOFTLY — almost a murmur, normal distance.",
    "Speak LOUDLY / energetically.",
    "Turn your HEAD AWAY from the mic while speaking.",
    "SHORT command style: a couple of quick one-liners with pauses.",
    "Normal one last time.",
]


def _calibrate(name: str, seconds: float) -> None:
    """Record N varied-condition rounds and report the self-score distribution for
    `name`, with verdicts under the CURRENT thresholds and data-driven suggestions."""
    pid = _find_person_id(name)
    if pid is None:
        print(f"  No person named {name!r} in the DB.")
        return
    n_rows = db.fetchone(
        "SELECT COUNT(*) AS n FROM biometrics WHERE person_id = ? AND type='voice'", (pid,)
    )["n"]
    print(f"CALIBRATION for {name} (person_id={pid}, {n_rows} enrolled sample(s))")
    print(f"{len(_CALIBRATION_ROUNDS)} rounds × {seconds:.0f}s. Talk for the WHOLE window each time —")
    print("read a sentence, describe your day, anything. Ready?\n")
    threshold = config.SPEAKER_ID_SIMILARITY_THRESHOLD
    confident = float(getattr(config, "SPEAKER_ID_CONFIDENT_THRESHOLD", 0.70))

    scores: list[float] = []
    results_table: list[tuple[str, float, str]] = []
    for i, condition in enumerate(_CALIBRATION_ROUNDS, 1):
        print(f"── Round {i}/{len(_CALIBRATION_ROUNDS)}: {condition}")
        for c in (3, 2, 1):
            print(f"  {c}...")
            time.sleep(0.8)
        audio = _record(seconds)
        ranked = speaker_id.rank_speakers(audio)
        me = next((sim for p, _nm, sim, _n in ranked if p == pid), None)
        if me is None:
            print("  (embedding failed / no score — skipping round)\n")
            results_table.append((condition, float("nan"), "no-score"))
            continue
        verdict = ("CONFIDENT" if me >= confident
                   else "accepted " if me >= threshold else "REJECTED ")
        print(f"  → your score this round: {me:.3f}  [{verdict.strip()}]\n")
        scores.append(me)
        results_table.append((condition, me, verdict.strip()))
        time.sleep(0.4)

    if not scores:
        print("No usable rounds — check the mic and rerun.")
        return

    arr = np.array(scores, dtype=np.float64)
    print("\n════════ CALIBRATION SUMMARY ════════")
    print(f"  {'score':>7}  verdict     condition")
    for condition, sc, verdict in results_table:
        sc_txt = f"{sc:.3f}" if sc == sc else "  —  "
        print(f"  {sc_txt:>7}  {verdict:<10}  {condition}")
    print(f"\n  rounds={len(scores)}  min={arr.min():.3f}  median={np.median(arr):.3f}  "
          f"mean={arr.mean():.3f}  max={arr.max():.3f}  spread={arr.max()-arr.min():.3f}")
    print(f"  current thresholds: accept={threshold:.2f}  confident={confident:.2f}")

    # Data-driven readout — worst case is what matters (every round is genuinely you).
    lo = float(arr.min())
    print("\n  READOUT:")
    print("  • NB: these are CLEAN continuous windows. Live turns (short VAD segments")
    print("    through echo-cancel, room noise) score ~0.05-0.12 LOWER — measured")
    print("    2026-07-05: Bret calibrated 0.742-0.830 but logged 0.625-0.715 in real")
    print("    conversation. Judge thresholds against the live band, not this one.")
    if lo >= confident:
        print(f"  • Even your worst condition ({lo:.3f}) clears the confident bar — the")
        print(f"    challenge gate should essentially never fire on you as configured.")
    else:
        below = int((arr < confident).sum())
        print(f"  • {below}/{len(scores)} of your OWN rounds score below the {confident:.2f} 'confident'")
        print(f"    bar (worst {lo:.3f}). This is the overlap zone: an unenrolled voice")
        print(f"    (JT) lands on your print at ~0.60-0.67 — INSIDE your own band. No")
        print(f"    absolute threshold can split 'weak you' from 'strong someone-else'.")
        print(f"    That's why the fix is a second print (enroll JT) + the visual")
        print(f"    challenge gate, not threshold tuning.")
    if int((arr < threshold).sum()) > 0:
        print(f"  • {int((arr < threshold).sum())} round(s) fell below the ACCEPT floor {threshold:.2f} — in the field")
        print(f"    those turns would go unattributed. If they were the quiet/far rounds,")
        print(f"    that's the far-field gain issue, not identity.")
    if len(scores) >= 4 and n_rows >= 3 and float(np.median(arr)) < 0.80:
        # Only when the band is actually LOW — a healthy print (median >= 0.80,
        # e.g. Bret's fresh 6-row print at 0.868) should not be poked.
        print(f"  • This band is LOW for clean windows — consider a clean re-enroll:")
        print(f"    ./venv/bin/python tools/test_voice_id.py --enroll \"{name}\" --replace")
        print(f"    then 2-3 more --enroll \"{name}\" runs (varied distance) to rebuild the centroid.")
    elif float(np.median(arr)) >= 0.80:
        print(f"  • Print looks healthy (median {np.median(arr):.3f}) — leave it alone.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=5.0, help="Seconds to record (default 5)")
    ap.add_argument("--repeat", type=int, default=1, help="Number of back-to-back samples")
    ap.add_argument("--enroll", type=str, default=None, help="Enroll a new voice under this name instead of testing")
    ap.add_argument("--replace", action="store_true", help="With --enroll: after adding the new row, delete older voice rows for this person")
    ap.add_argument("--trim", type=str, default=None, help="Keep only the newest voice biometric for NAME")
    ap.add_argument("--trim-all", action="store_true", help="Keep only the newest voice biometric per person, DB-wide")
    ap.add_argument("--calibrate", type=str, default=None, metavar="NAME",
                    help="Guided multi-condition session measuring NAME's self-score distribution")
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

    if args.calibrate:
        secs = args.secs if args.secs != 5.0 else 6.0   # calibration wants longer windows
        _calibrate(args.calibrate, secs)
        return 0

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
