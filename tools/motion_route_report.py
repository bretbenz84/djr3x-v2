#!/usr/bin/env python3
"""
tools/motion_route_report.py — the Phase 0/1 record for LLM-planned drive routes.

See docs/motion_route_tool_plan.md §7. Three modes, and the first two need no
network:

    ./venv/bin/python tools/motion_route_report.py --corpus
        Mine logs/ for every utterance that ever reached the tri-state None arm
        (final_executed_path=fast_local_takeover.motion.sequence_rejected) and
        re-run each one through the CURRENT classifier, so the list is what the
        rescue path would face TODAY rather than when the log was written. Months
        of regex repairs have already retired half the historical corpus.

    ./venv/bin/python tools/motion_route_report.py
        Aggregate the [motion_route] JSON lines a running Rex emits: how often the
        interpreter planned vs declined, what it drove, what the translator refused
        and why, and the added latency.

    ./venv/bin/python tools/motion_route_report.py --live
        Replay the mined corpus AND the figurative-motion decoys through the real
        interpreter, then print the plan §7 metrics: parse rate on real None-arm
        turns, decoy false-fire rate (must be ~0), and latency. Costs one small
        hosted call per utterance — the corpus is a couple of dozen, not a bill.
        Args are printed for hand-labelling; nothing here judges them for you.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_AUDIT = re.compile(r"\[(?:character_loop|action_router_audit)\] (\{.*\})\s*$")
_ROUTE = re.compile(r"\[motion_route\] (\{.*\})\s*$")


def _logs(days: float | None) -> "list[Path]":
    cutoff = None if not days else time.time() - days * 86400.0
    out = []
    for path in sorted((_ROOT / "logs").glob("*.log")):
        if cutoff is None or path.stat().st_mtime >= cutoff:
            out.append(path)
    return out


def _rejected_utterances(paths: "list[Path]") -> "Counter[str]":
    """Every utterance whose turn ended on the tri-state None arm."""
    seen: Counter[str] = Counter()
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            if "sequence_rejected" not in line:
                continue
            m = _AUDIT.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            executed = str(rec.get("final_executed_path")
                           or (rec.get("execution") or {}).get("final_executed_path") or "")
            if "sequence_rejected" not in executed:
                continue
            text = str(rec.get("utterance") or rec.get("heard_text") or "").strip()
            if text:
                seen[text] += 1
    # The audit and the character_loop trace both carry the same turn, so every
    # utterance is counted twice. Halve rather than dedupe, to keep the frequencies.
    return Counter({t: max(1, n // 2) for t, n in seen.items()})


def _classify(text: str) -> str:
    from intelligence import action_router
    seq = action_router.classify_explicit_motion_sequence(text, max_steps=8)
    if seq is None:
        return "rescue"          # still the None arm — the rescue path's job
    if not seq:
        return "not_a_route"     # the regex repairs since retired it
    return f"parses:{len(seq)}"


# Figurative motion that must NEVER produce a route. The Phase-3 gate corpus
# (docs/tool_router_scope.md §3) plus the banter shapes Phase 2 has to survive.
DECOYS = (
    "I think we should move on from that topic", "let's move on",
    "moving forward, I want to try something", "I need to run to the store",
    "she moved forward with the plan", "can you back me up on this",
    "go right ahead and tell him", "come to think of it, that's wrong",
    "my head is spinning", "I'm going to head out soon",
    "we should do a lap sometime", "the meeting turned into a disaster",
    "let's roll", "roll with it", "come on, go on, catch",
    "I'll swing by later and then head home",
    "we backed up the database and then moved the files",
    "first he turned left, then he drove into a ditch — true story",
    "don't move forward and don't turn around",
    "walk me through it: go to the menu, then click save",
)


def _corpus(paths, *, verbose=True):
    counts = _rejected_utterances(paths)
    buckets: dict[str, list] = {}
    for text, n in counts.most_common():
        buckets.setdefault(_classify(text), []).append((text, n))
    if verbose:
        print(f"scanned {len(paths)} logs — {len(counts)} distinct utterances ever "
              f"reached the tri-state None arm\n")
        for arm in ("rescue", "not_a_route"):
            rows = buckets.get(arm, [])
            label = ("STILL on the None arm (the rescue path's corpus)"
                     if arm == "rescue" else
                     "retired since, by the regex repairs (now [])")
            print(f"── {label}: {len(rows)}")
            for text, n in rows:
                print(f"   {n:3d}  {text!r}")
            print()
        for arm, rows in sorted(buckets.items()):
            if arm.startswith("parses:"):
                print(f"── now parses as a route ({arm}): {len(rows)}")
                for text, n in rows:
                    print(f"   {n:3d}  {text!r}")
    return [t for t, _ in buckets.get("rescue", [])]


# Chatter wrapped around a REAL drive command — mined from the logs, because this
# is how people actually talk to him. Every one of these SHOULD plan. They are the
# regression set for the decline-scoping bug of 2026-08-23: the rules that refuse a
# figure of speech were being applied to the whole utterance while the rule that
# ignores chatter was applied to a span, so a negation or the bare word "places"
# elsewhere in the sentence vetoed a command sitting right there. Measured on this
# set: 31/42 before the rewrite, 36/42 after, with the decoys unmoved.
EMBEDDED = (
    "No, cause I don't have not. We don't have places to go. Turn to your right, then move forward five feet.",
    'Yeah. Uh huh. Turn to your right and move forward.',
    "Oh, hardline! Yeah, you know Discord. I'm on Discord right now. Yeah. He actually isn't in it right now, but he can actually get on. Go turn to your right.",
    "Actually, nothing's in your way. Your sensor is lying to you. You should move forward two feet.",
    "Did you just say man? The way she's mean? Rex, move forward 5 feet",
    "Oh, you went a little far, but I'm over here, turn right, find the black guy.",
    'You still have three feet ahead of you. Can you move forward two feet?',
    "Turn right a little. Alright, it's got two sides.",
    "Turn to your right. I don't think you should have entered anymore at all.",
    'Turn to your left a little bit, and then tell me what you see.',
    'Turn to your left, Max. Go to where.',
    'Turn around and come forward, Ozzie',
    'Turn, never mind, move forward, four feet.',
    'Turn, turn to your left.',
)


def _would_be_stopped(text: str) -> "str | None":
    """Whether the deterministic layers would stop this plan before the wheels.

    The replay hands the interpreter every utterance directly, which PRODUCTION
    never does — so an interpreter false-fire is only a real one if it survives the
    layer that owns it. Two of them, and which applies depends on the path:
      * the RESCUE path is only reached from the tri-state None arm, so anything the
        sequence classifier calls [] or parses itself never gets there at all;
      * the ORGANIC path runs the full evidence gate
        (action_router.motion_command_refusal_reason).
    """
    from intelligence import action_router
    arm = action_router.classify_explicit_motion_sequence(text, max_steps=8)
    gate = action_router.motion_command_refusal_reason(text, "motion.route")
    if arm is not None and gate:
        return (f"never reaches the rescue arm (tri-state "
                f"{'[]' if not arm else f'{len(arm)} parsed steps'}), and the "
                f"evidence gate refuses the organic path ({gate})")
    if gate:
        return f"evidence gate refuses the organic path ({gate})"
    if arm is not None:
        return "never reaches the rescue arm (not route-shaped)"
    return None


def _live(corpus):
    from intelligence import action_router, motion_route

    if not motion_route.available():
        print("interpreter unavailable (offline, or MOTION_ROUTE_ENABLED is off)")
        return
    rows = ([(t, "route") for t in corpus] + [(t, "embedded") for t in EMBEDDED]
            + [(t, "decoy") for t in DECOYS])
    planned = {"route": 0, "embedded": 0, "decoy": 0}
    reached = {"route": 0, "embedded": 0, "decoy": 0}
    stopped = {"route": 0, "embedded": 0, "decoy": 0}
    declined = {"route": 0, "embedded": 0, "decoy": 0}
    refused = {"route": 0, "embedded": 0, "decoy": 0}
    errors = 0
    secs: list[float] = []
    print(f"replaying {len(corpus)} None-arm utterances + {len(DECOYS)} decoys\n")
    for text, kind in rows:
        result = motion_route.interpret(text)
        secs.append(float(result.get("secs") or 0.0))
        if result.get("error"):
            errors += 1
            print(f"[{kind}] ERROR    {result['error']}  {text[:60]!r}")
            continue
        if result.get("declined"):
            declined[kind] += 1
            print(f"[{kind}] declined {result.get('reason', '')[:40]:40s} {text[:60]!r}")
            continue
        decisions, reason = action_router.route_tool_to_decisions(result["args"])
        if decisions is None:
            refused[kind] += 1
            print(f"[{kind}] REFUSED  {reason:40s} {text[:60]!r}")
            continue
        planned[kind] += 1
        plan = " | ".join(f"{d.action.split('.')[1]}{d.args}" for d in decisions)
        stopper = _would_be_stopped(text)
        if stopper and kind == "decoy":
            stopped[kind] += 1
            print(f"[{kind}] planned  {text[:60]!r}\n              {plan}"
                  f"\n              ...but STOPPED before the wheels: {stopper}")
        else:
            reached[kind] += 1
            print(f"[{kind}] DRIVES   {text[:60]!r}\n              {plan}")
    print("\n── plan §7 metrics")
    total_routes = max(1, len(corpus))
    print(f"   parse rate on real None-arm turns : {planned['route']}/{len(corpus)} "
          f"({100.0 * planned['route'] / total_routes:.0f}%)  "
          f"[declined {declined['route']}, clamp-refused {refused['route']}]")
    print(f"   decoys the interpreter planned    : {planned['decoy']}/{len(DECOYS)}  "
          f"[declined {declined['decoy']}, clamp-refused {refused['decoy']}]")
    print(f"   decoys that would REACH the wheels: {reached['decoy']}/{len(DECOYS)}  "
          f"(want 0 — {stopped['decoy']} stopped by the deterministic gates)")
    print(f"   chatter-wrapped commands planned  : {planned['embedded']}/{len(EMBEDDED)}  "
          f"[declined {declined['embedded']}, clamp-refused {refused['embedded']}]")
    print(f"   interpreter errors                : {errors}")
    if secs:
        print(f"   latency median/p90                : {statistics.median(secs):.2f}s / "
              f"{sorted(secs)[int(0.9 * (len(secs) - 1))]:.2f}s")


def _live_lines(paths):
    """Aggregate the [motion_route] lines a running Rex already emitted."""
    records = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            m = _ROUTE.search(line)
            if not m:
                continue
            try:
                records.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue
    if not records:
        print("no [motion_route] lines yet — nothing has hit the rescue path on this "
              "checkout. Use --corpus for the historical picture, --live to replay it.")
        return
    drove = [r for r in records if r.get("executed")]
    print(f"{len(records)} rescue attempts, {len(drove)} drove\n")
    for label, key in (("declined by the interpreter", "declined"),):
        rows = [r for r in records if r.get(key)]
        print(f"── {label}: {len(rows)}")
        for r in rows:
            print(f"   {r.get('reason', '')[:40]:40s} {str(r.get('utterance'))[:60]!r}")
    refusals = Counter(str(r["refused"]) for r in records if r.get("refused"))
    if refusals:
        print(f"\n── refused by the translator: {sum(refusals.values())}")
        for reason, n in refusals.most_common():
            print(f"   {n:3d}  {reason}")
    errs = Counter(str(r["error"]).split(":")[0] for r in records if r.get("error"))
    if errs:
        print(f"\n── interpreter errors: {sum(errs.values())}")
        for kind, n in errs.most_common():
            print(f"   {n:3d}  {kind}")
    print(f"\n── drove: {len(drove)}")
    for r in drove:
        plan = " | ".join(f"{s['action'].split('.')[1]}{s['args']}"
                          for s in r.get("steps") or [])
        print(f"   {str(r.get('utterance'))[:60]!r}\n      {plan}")
    secs = [float(r.get("secs") or 0.0) for r in records if r.get("secs")]
    if secs:
        print(f"\n   latency median/p90: {statistics.median(secs):.2f}s / "
              f"{sorted(secs)[int(0.9 * (len(secs) - 1))]:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=None, help="only logs this recent")
    ap.add_argument("--corpus", action="store_true",
                    help="mine + re-classify the historical None-arm utterances")
    ap.add_argument("--live", action="store_true",
                    help="replay the corpus and the decoys through the interpreter")
    args = ap.parse_args()

    paths = _logs(args.days)
    if args.live:
        _live(_corpus(paths, verbose=False))
        return
    if args.corpus:
        _corpus(paths)
        return
    _live_lines(paths)


if __name__ == "__main__":
    main()
