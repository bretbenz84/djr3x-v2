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
_AUDIT = re.compile(r"\[character_loop\] (\{.*\})\s*$")

# The TRUE baseline is what the turn actually EXECUTED — production routes
# through three layers (action router, intent classifier, legacy commands), and
# the router's own decision says "reply" for turns the other layers served
# (first collection 2026-08-01: every game/weather/vision turn showed as a
# false disagreement until this join). Map executed paths → catalog keys.
_PATH_TO_ACTION = {
    "llm.stream": "conversation.reply",
    "turn_completion.hold": "conversation.reply",
    "repair.factual": "conversation.repair",
    "game.active_turn.early": "game.answer",
    "legacy_command.start_game": "game.start",
    "legacy_command.stop_game": "game.stop",
    "identity.offscreen_identify_reply": "identity.who_is_speaking",
    "intent_classifier.query_time": "time.query",
    "intent_classifier.query_date": "date.query",
    "intent_classifier.query_weather": "weather.query",
    "intent_classifier.query_capabilities": "status.capabilities",
    "intent_classifier.query_uptime": "status.uptime",
    "intent_classifier.query_what_do_you_see": "vision.describe_scene",
    "intent_classifier.query_who_is_speaking": "identity.who_is_speaking",
    "intent_classifier.query_memory": "memory.query",
    "intent_classifier.play_music": "music.play",
    "intent_classifier.query_music_options": "music.options",
}


def _executed_for(path: Path) -> "dict[str, list[str]]":
    """utterance → FIFO of final_executed_path values, from this log's audits."""
    out: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        m = _AUDIT.search(line)
        if not m:
            continue
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        heard = str(d.get("heard_text") or "")
        fep = str((d.get("execution") or {}).get("final_executed_path") or "")
        if heard and fep:
            path_key = fep if fep.startswith("fast_local_takeover.") else fep
            out[heard].append(path_key)
    return out


def _baseline_action(executed_path: str | None) -> str | None:
    """Catalog key for an executed path, or None when unmappable (hand review)."""
    if not executed_path:
        return None
    if executed_path.startswith("fast_local_takeover."):
        key = executed_path.split(".", 1)[1]
        return key if "." in key else None
    return _PATH_TO_ACTION.get(executed_path)


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
        executed = _executed_for(path)
        for line in path.read_text(errors="replace").splitlines():
            m = _LINE.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            fifo = executed.get(str(rec.get("utterance") or ""))
            rec["executed"] = fifo.pop(0) if fifo else None
            rec["baseline"] = _baseline_action(rec["executed"])
            records.append(rec)

    if not records:
        print("No [tool_router_shadow] lines found — is TOOL_ROUTER_SHADOW_ENABLED on?")
        return

    ok = [r for r in records if not r.get("error")]
    errs = len(records) - len(ok)
    judged = [r for r in ok if r.get("baseline")]
    agree = sum(1 for r in judged if r.get("tool") == r.get("baseline"))
    lat = sorted(float(r.get("secs") or 0.0) for r in ok)
    print(f"{len(records)} shadow decisions ({errs} errored, "
          f"{len(ok) - len(judged)} with unmappable/missing executed path)")
    if judged:
        print(f"agreement vs EXECUTED behavior: {agree}/{len(judged)} "
              f"({agree / len(judged) * 100:.1f}%)")
    if ok:
        p95 = lat[max(0, int(round(0.95 * len(lat))) - 1)]
        print(f"shadow latency: median {statistics.median(lat):.2f}s  p95 {p95:.2f}s "
              f"(off-turn — informational only)\n")

    per = defaultdict(lambda: [0, 0])          # baseline action → [agree, total]
    for r in judged:
        row = per[str(r.get("baseline"))]
        row[1] += 1
        row[0] += r.get("tool") == r.get("baseline")
    print(f"{'executed (baseline) action':>28}  {'agree':>7}  {'n':>4}")
    for action, (a, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"{action:>28}  {a / n * 100:6.1f}%  {n:>4}")

    diffs = [r for r in judged if r.get("tool") != r.get("baseline")]
    unmapped = [r for r in ok if not r.get("baseline")]
    if diffs:
        print(f"\nDISAGREEMENTS ({len(diffs)}) — the hand-review queue:")
        pair_counts = Counter((str(r.get('baseline')), str(r.get('tool'))) for r in diffs)
        for (base, tool), n in pair_counts.most_common():
            print(f"  {n:>3}x  executed={base}  tool={tool}")
        print()
        for r in diffs:
            print(f"  executed={r.get('baseline')} ({r.get('executed')})  "
                  f"tool={r.get('tool')}  args={r.get('args')}  text={r.get('utterance')!r}")
    if unmapped:
        print(f"\nUNMAPPED ({len(unmapped)}) — executed path has no catalog mapping yet:")
        for r in unmapped:
            print(f"  executed={r.get('executed')}  tool={r.get('tool')}  "
                  f"text={r.get('utterance')!r}")


if __name__ == "__main__":
    main()
