#!/usr/bin/env python3
"""
Offline A/B for the lean brain — no robot required.

Replays a real conversation log through intelligence/lean_brain and prints, for each of YOUR
turns: what OLD Rex actually said (from the log) vs what LEAN Rex would say now — plus the
latency of each lean call (time-to-first-token and total). Each lean reply is generated given
the SAME real conversation history up to that point, so it's a fair turn-by-turn comparison.

Usage:
    ./venv/bin/python tools/lean_replay.py logs/conversation-2026-06-30-01-10-16.log
    ./venv/bin/python tools/lean_replay.py logs/conversation-*.log --person "Bret Benziger"

Needs the OpenAI key the app already uses (it makes one real lean call per user turn).
"""

import argparse
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Matches:  2026-06-30 01:10:56 | HEARD | Bret Benziger: life's good
#           2026-06-30 01:10:57 | REX   | Systems nominal; ...
_LINE = re.compile(r"^\s*\d[\d\-]*\s+[\d:]+\s*\|\s*(REX|HEARD)\s*\|\s*(.*)$")


def parse_log(path: str) -> list[dict]:
    turns: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            m = _LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            kind, text = m.group(1), m.group(2).strip()
            if not text:
                continue
            if kind == "HEARD":
                name, sep, said = text.partition(":")
                turns.append({"speaker": name.strip(), "text": (said if sep else text).strip(), "user": True})
            else:
                turns.append({"speaker": "Rex", "text": text, "user": False})
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a conversation log through the lean brain.")
    ap.add_argument("log", help="path to a logs/conversation-*.log file (globs allowed)")
    ap.add_argument("--person", default=None, help="speaker name to resolve to a person_id (default: first HEARD speaker)")
    ap.add_argument("--limit", type=int, default=0, help="only replay the first N user turns (0 = all)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.log)) or [args.log]

    from intelligence import lean_brain
    try:
        from memory import people
    except Exception:
        people = None

    all_ttft: list[float] = []
    all_total: list[float] = []

    for path in paths:
        turns = parse_log(path)
        speaker = args.person or next((t["speaker"] for t in turns if t["user"]), None)
        person_id = None
        if speaker and people is not None:
            try:
                p = people.find_person_by_name(speaker)
                person_id = int(p["id"]) if p and p.get("id") is not None else None
            except Exception:
                person_id = None

        print(f"\n{'='*88}\n# {path}\n# speaker={speaker!r}  person_id={person_id}  model={lean_brain._model()}\n{'='*88}")

        transcript: list[dict] = []
        done = 0
        for i, t in enumerate(turns):
            if t["user"]:
                # gather OLD Rex's actual reply (the consecutive REX lines that follow)
                old = []
                j = i + 1
                while j < len(turns) and not turns[j]["user"]:
                    old.append(turns[j]["text"])
                    j += 1
                res = lean_brain.respond(t["text"], person_id=person_id, transcript=transcript)
                print(f"\nYOU:       {t['text']}")
                print(f"OLD REX:   {' ⏎ '.join(old) if old else '(no reply / proactive)'}")
                print(f"LEAN REX:  {res['text']}")
                print(f"           [ttft {res['ttft_s']*1000:.0f}ms · total {res['total_s']*1000:.0f}ms]")
                all_ttft.append(res["ttft_s"])
                all_total.append(res["total_s"])
                done += 1
                if args.limit and done >= args.limit:
                    transcript.append(t)
                    break
            transcript.append(t)

    if all_ttft:
        print(f"\n{'='*88}")
        print(f"# LATENCY over {len(all_ttft)} turns — first token: median {statistics.median(all_ttft)*1000:.0f}ms, "
              f"max {max(all_ttft)*1000:.0f}ms  |  full reply: median {statistics.median(all_total)*1000:.0f}ms, "
              f"max {max(all_total)*1000:.0f}ms")
        print("# (live first-audio ≈ first-token + a short TTS start; the whole reply keeps streaming after Rex starts talking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
