#!/usr/bin/env python3
"""
tools/tool_router_report.py — aggregate the Phase 0 tool-router shadow logs.

Scans logs/djr3x-*.log for [tool_router_shadow] JSON lines and reports overall
and per-action agreement between the shipping router and the tool-choice
shadow, plus every disagreement (the review queue: label each side right/wrong
by hand — that labeled list is what cutover decisions are made from).

    ./venv/bin/python tools/tool_router_report.py            # all logs
    ./venv/bin/python tools/tool_router_report.py --days 7   # recent only
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LINE = re.compile(r"\[tool_router_shadow\] (\{.*\})\s*$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=None,
                    help="only logs modified within the last N days")
    args = ap.parse_args()

    cutoff = time.time() - args.days * 86400.0 if args.days else None
    records = []
    for path in sorted(_ROOT.glob("logs/djr3x-*.log")):
        if cutoff and path.stat().st_mtime < cutoff:
            continue
        for line in path.read_text(errors="replace").splitlines():
            m = _LINE.search(line)
            if not m:
                continue
            try:
                records.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue

    if not records:
        print("No [tool_router_shadow] lines found — is TOOL_ROUTER_SHADOW_ENABLED on?")
        return

    ok = [r for r in records if not r.get("error")]
    errs = len(records) - len(ok)
    agree = sum(1 for r in ok if r.get("agree"))
    lat = sorted(float(r.get("secs") or 0.0) for r in ok)
    print(f"{len(records)} shadow decisions ({errs} errored)")
    if ok:
        print(f"overall agreement: {agree}/{len(ok)} ({agree / len(ok) * 100:.1f}%)")
        p95 = lat[max(0, int(round(0.95 * len(lat))) - 1)]
        print(f"shadow latency: median {statistics.median(lat):.2f}s  p95 {p95:.2f}s "
              f"(off-turn — informational only)\n")

    per = defaultdict(lambda: [0, 0])          # shipped action → [agree, total]
    for r in ok:
        row = per[str(r.get("shipped"))]
        row[1] += 1
        row[0] += bool(r.get("agree"))
    print(f"{'shipped action':>28}  {'agree':>7}  {'n':>4}")
    for action, (a, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"{action:>28}  {a / n * 100:6.1f}%  {n:>4}")

    diffs = [r for r in ok if not r.get("agree")]
    if diffs:
        print(f"\nDISAGREEMENTS ({len(diffs)}) — the hand-review queue:")
        pair_counts = Counter((str(r.get('shipped')), str(r.get('tool'))) for r in diffs)
        for (shipped, tool), n in pair_counts.most_common():
            print(f"  {n:>3}x  shipped={shipped}  tool={tool}")
        print()
        for r in diffs:
            print(f"  shipped={r.get('shipped')}  tool={r.get('tool')}  "
                  f"args={r.get('args')}  text={r.get('utterance')!r}")


if __name__ == "__main__":
    main()
