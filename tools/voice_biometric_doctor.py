#!/usr/bin/env python3
"""Diagnose (and optionally repair) the stored voice biometrics in people.db.

Wrong-speaker attribution ("R3X thought JT was Bret") usually traces to voiceprints that are thin
(too few samples) or that overlap each other — often a leftover from the old bug where auto-refresh
saved a segment to the wrong person. A code fix does NOT clean data already written, so the bad
samples sit in the DB until purged. This tool surfaces that and lets you wipe a person's voiceprint
for a clean re-enroll.

    ./venv/bin/python -m tools.voice_biometric_doctor                 # full report
    ./venv/bin/python -m tools.voice_biometric_doctor --purge-voice 2 # wipe JT's voiceprints (re-enroll after)

Runs against config.DB_PATH by default (--db to point elsewhere). Read-only unless --purge-voice.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

# A pair of DIFFERENT people should sit well below this; at/above it they collide in recognition.
OVERLAP_WARN = 0.45
# Fewer than this many voice rows = a fragile voiceprint that won't separate reliably.
THIN_WARN = 4


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-10)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose/repair voice biometrics in people.db")
    ap.add_argument("--db", default=None, help="path to people.db (default: config.DB_PATH)")
    ap.add_argument("--purge-voice", type=int, metavar="PERSON_ID",
                    help="DELETE all voiceprints for this person so they can re-enroll cleanly")
    args = ap.parse_args()

    import config
    if args.db:
        config.DB_PATH = args.db
    from memory import people as P, admin
    from memory import database as memdb

    rows = memdb.fetchall("SELECT id, person_id, encoding, created_at FROM biometrics WHERE type='voice' ORDER BY person_id, id")
    names = {r["id"]: r["name"] for r in memdb.fetchall("SELECT id, name FROM people")}

    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["person_id"]].append((r["id"], _norm(P._from_blob(bytes(r["encoding"])).astype(np.float32)), r["created_at"]))

    if args.purge_voice is not None:
        pid = args.purge_voice
        n = len(by_pid.get(pid, []))
        ok = admin.clear_biometrics(pid, "voice")
        print(f"[voice-doctor] purged {n} voiceprint(s) for person {pid} ({names.get(pid,'?')}): {ok}")
        print("[voice-doctor] re-enroll them by having ONLY that person speak a few clear sentences.")
        return 0 if ok else 1

    print(f"[voice-doctor] db={config.DB_PATH}  people={ {k:v for k,v in names.items()} }")
    print(f"[voice-doctor] {len(rows)} voice rows across {len(by_pid)} people\n")

    cent = {pid: _norm(np.mean(np.stack([e for _, e, _ in items]), axis=0)) for pid, items in by_pid.items()}

    for pid, items in by_pid.items():
        ts = sorted(str(t)[:19] for _, _, t in items)
        flag = "  ⚠ THIN" if len(items) < THIN_WARN else ""
        window = f"{ts[0]} … {ts[-1]}" if ts else "-"
        print(f"person {pid} ({names.get(pid,'?')}): {len(items)} samples  [{window}]{flag}")

    print("\ncentroid overlap (a HIGH value between two different people = they will collide):")
    pids = list(cent)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            s = float(np.dot(cent[pids[i]], cent[pids[j]]))
            warn = "  ⚠ OVERLAP — voiceprints too similar" if s >= OVERLAP_WARN else ""
            print(f"  {names.get(pids[i])} <-> {names.get(pids[j])}: {s:+.3f}{warn}")

    print("\nper-sample leave-one-out (does a sample look more like SOMEONE ELSE?):")
    print(f"  {'row':>4} {'owner':>14} {'created':>19}  simOwn  bestOther            verdict")
    for pid, items in by_pid.items():
        others = [p for p in by_pid if p != pid]
        for rid, emb, ts in items:
            loo = [it for it in items if it[0] != rid]
            s_own = float(np.dot(emb, _norm(np.mean(np.stack([e for _, e, _ in loo]), axis=0)))) if loo else float("nan")
            best_o, best_os = None, -1.0
            for op in others:
                so = float(np.dot(emb, cent[op]))
                if so > best_os:
                    best_os, best_o = so, op
            bad = not np.isnan(s_own) and best_os > s_own
            verdict = f"<-- looks like {names.get(best_o)}" if bad else "ok"
            print(f"  {rid:>4} {names.get(pid,'?')[:14]:>14} {str(ts)[:19]:>19}  {s_own:+.3f}  {names.get(best_o,'?')[:12]:>12}={best_os:+.3f}   {verdict}")

    print("\n[voice-doctor] Fix a colliding/thin voiceprint:  --purge-voice <person_id>  then re-enroll that person alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
