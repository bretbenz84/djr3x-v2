#!/usr/bin/env python3
"""LLM-in-the-loop conversation-quality eval for DJ-R3X.

Runs a corpus of real-ish scenarios through the ACTUAL reply-generation path
(deterministic governor stack → gpt-4o-mini stream → the same sentence/tail
assembly the speech queue uses) and scores each reply against failure CLASSES
(over-questioning, cantina bleed, invented prop, roasting a sincere share,
re-asking, trail-off). It reports per-class flag RATES across the corpus, so
generation regressions are measured — not patched one bad live line at a time.

    ⚠️  This makes REAL OpenAI calls (one generation + a couple of judge calls
        per reply). A full corpus at --samples 1 is ~20 cheap gpt-4o-mini calls.
        It is intentionally NOT part of `unittest discover` (which stays
        network-free); run it deliberately.

Usage:
    python evals/run_quality_eval.py                  # whole corpus, 1 sample each
    python evals/run_quality_eval.py --samples 3      # 3 samples/scenario (rates)
    python evals/run_quality_eval.py --only cantina   # scenarios whose name matches
    python evals/run_quality_eval.py --out report.json
    python evals/run_quality_eval.py --gate 0.0       # exit 1 if ANY class flags

Add a scenario by dropping an object into evals/quality_corpus.json (no Python).
See evals/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The generation path is chatty at INFO; keep eval output clean.
logging.disable(logging.WARNING)

from evals import checkers as C  # noqa: E402

_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_corpus.json")


# ─────────────────────────────────────────────────────────────────────────────
# Generation — the real spoken reply (deterministic stack + LLM + tail assembly)
# ─────────────────────────────────────────────────────────────────────────────

def _reset_session() -> None:
    from intelligence import (
        conversation_steering, user_energy, topic_thread, repair_moves,
        end_thread, comedy_modes, rex_pov,
    )
    from memory import conversations as conv
    for module in (conversation_steering, user_energy, topic_thread,
                   repair_moves, end_thread, rex_pov):
        try:
            module.clear()
        except Exception:
            pass
    try:
        comedy_modes.reset_recent_state()
    except Exception:
        pass
    try:
        conv.clear_transcript()
    except Exception:
        pass


def _seed_context(scenario: dict) -> None:
    """Replay the scenario's prior context so the prompt's 'Session so far' and
    the governors reflect it (lets re-asking / continuity be exercised)."""
    from memory import conversations as conv
    from intelligence import (
        comedy_modes, repair_moves, topic_thread, conversation_steering as cs,
    )
    person_id = scenario.get("person_id")
    for turn in scenario.get("prior_turns", []):
        conv.add_to_transcript(turn.get("speaker", "User"), turn.get("text", ""))
    rex_last = scenario.get("rex_last_line")
    if rex_last:
        conv.add_to_transcript("Rex", rex_last)
        for fn in (comedy_modes.note_spoken_line, repair_moves.note_assistant_turn,
                   topic_thread.note_assistant_turn):
            try:
                fn(rex_last)
            except Exception:
                pass
    for prior in scenario.get("prior_user_turns", []):
        try:
            cs.note_user_turn(person_id, prior)
        except Exception:
            pass


def generate_spoken(scenario: dict) -> str:
    """Generate the reply Rex would actually SAY, mirroring the live speech path
    (`interaction._stream_and_speak_sentences`) FAITHFULLY: gpt-4o-mini streams;
    each sentence is governed + comedy-polished PER SENTENCE
    (`_prepare_stream_sentence`, which strips disallowed questions and banned
    openers); a one-question cap holds across the whole reply; and a non-speakable
    end-of-stream tail is dropped (the ellipsis-trail-off fix). Deliberately does
    NOT re-govern the joined whole reply — that would truncate mid-sentence to
    max_words and over-count questions, artifacts the robot never produces."""
    from intelligence import (
        interaction as I, llm, conversation_agenda as ca, social_frame as sf,
        comedy_modes,
    )
    person_id = scenario.get("person_id")
    utterance = scenario["utterance"]
    answered = scenario.get("answered_question")

    plan = ca.build_turn_plan(utterance, person_id, answered_question=answered)
    frame = sf.build_frame(
        utterance, person_id, answered_question=answered,
        agenda_directive=plan.directive, turn_plan=plan,
    )
    try:
        mode = comedy_modes.select_mode(
            utterance, person_id, frame=frame, agenda_directive=plan.directive)
    except Exception:
        mode = None

    spoken: list[str] = []
    state = {"spoke_question": False}

    def _consume(sentence: str) -> None:
        try:
            prepared = I._prepare_stream_sentence(sentence, frame, mode)
        except Exception:
            prepared = (sentence or "").strip()
        if not prepared:
            return
        if sf.is_question_sentence(prepared):  # one-question cap across the reply
            if state["spoke_question"]:
                return
            state["spoke_question"] = True
        spoken.append(prepared)

    buffer = ""
    raw_chunks: list[str] = []
    for chunk in llm.stream_response(utterance, person_id, agenda_directive=plan.directive):
        raw_chunks.append(chunk)
        buffer += chunk
        ready, buffer = I._split_stream_sentences(buffer, 12)
        for sentence in ready:
            _consume(sentence)
    tail = buffer.strip()
    if tail and I._tail_is_speakable(tail):
        _consume(tail)

    # Safety net (mirrors _stream_and_speak_sentences): if every sentence was
    # governed away (e.g. an all-questions reply under a no-question frame), fall
    # back to whole-reply governance so the eval scores a real line, not "" —
    # trimming a trailing mid-sentence fragment first (the truncated-tail fix).
    if not spoken:
        raw_full = I._complete_sentence_prefix("".join(raw_chunks))
        if raw_full:
            try:
                fb = sf.govern_response(raw_full, frame).text or raw_full
            except Exception:
                fb = raw_full
            if fb:
                spoken.append(fb)

    return llm.clean_response_text(" ".join(spoken))


# ─────────────────────────────────────────────────────────────────────────────
# Run / aggregate / report
# ─────────────────────────────────────────────────────────────────────────────

def run(corpus: list, samples: int) -> list:
    results = []
    total = len(corpus) * samples
    i = 0
    for scenario in corpus:
        for sample in range(samples):
            i += 1
            print(f"  [{i}/{total}] {scenario.get('name')} (sample {sample + 1})...",
                  end="", flush=True)
            error = None
            try:
                reply = generate_spoken(scenario)
            except Exception as exc:  # noqa: BLE001
                reply, error = "", repr(exc)
            findings = []
            if reply:
                for checker in C.ALL:
                    try:
                        finding = checker(reply, scenario)
                    except Exception as exc:  # noqa: BLE001
                        finding = C.Finding(getattr(checker, "__name__", "?"),
                                            False, f"checker_error: {exc!r}"[:80])
                    if finding is not None:
                        findings.append(finding)
            flagged = [f for f in findings if f.flagged]
            print(f" {'OK' if not flagged and not error else 'FLAG' if flagged else 'ERR'}"
                  + (f" ({', '.join(f.cls for f in flagged)})" if flagged else ""))
            results.append({
                "scenario": scenario.get("name"),
                "sample": sample,
                "reply": reply,
                "error": error,
                "findings": [vars(f) for f in findings],
            })
    return results


def aggregate(results: list) -> dict:
    classes: dict = {}
    judge_errors = 0
    for row in results:
        for f in row["findings"]:
            stats = classes.setdefault(f["cls"], {"flagged": 0, "total": 0})
            stats["total"] += 1
            if f["flagged"]:
                stats["flagged"] += 1
            if "judge_error" in (f.get("detail") or ""):
                judge_errors += 1
    return {"classes": classes, "judge_errors": judge_errors}


def report(results: list, agg: dict) -> None:
    print("\n" + "=" * 64)
    print("CONVERSATION-QUALITY EVAL — failure-class rates")
    print("=" * 64)
    n = len(results)
    errors = [r for r in results if r["error"]]
    print(f"replies scored: {n - len(errors)}/{n}"
          + (f"   ({len(errors)} generation error(s))" if errors else ""))
    if agg["judge_errors"]:
        print(f"⚠️  {agg['judge_errors']} judge call(s) errored (counted as not-flagged)")
    print("-" * 64)
    print(f"{'failure class':<22}{'flagged':>10}{'rate':>10}")
    print("-" * 64)
    for cls in sorted(agg["classes"]):
        st = agg["classes"][cls]
        rate = st["flagged"] / st["total"] if st["total"] else 0.0
        bar = "█" * round(rate * 16)
        print(f"{cls:<22}{st['flagged']:>4}/{st['total']:<5}{rate:>8.0%}  {bar}")
    print("-" * 64)

    flagged_rows = [r for r in results if any(f["flagged"] for f in r["findings"])]
    if flagged_rows:
        print("\nFLAGGED REPLIES:")
        for r in flagged_rows:
            hits = ", ".join(f"{f['cls']}={f['detail']}".rstrip("=")
                             for f in r["findings"] if f["flagged"])
            print(f"\n  • [{r['scenario']}] {hits}")
            print(f"    “{r['reply']}”")
    if errors:
        print("\nGENERATION ERRORS:")
        for r in errors:
            print(f"  • [{r['scenario']}] {r['error'][:120]}")
    print()


_JUDGE_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_cases.json")


def check_judges(cases_path: str) -> int:
    """Validate the LLM judges against LABELED cases (ground truth) so the judges
    themselves are trustworthy — a judge is a measurement instrument; calibrate it
    against known-correct labels, don't tune it in the same pass you grade a fix
    with. Makes real judge calls. Returns the mismatch count (0 = fully agrees)."""
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"⚠️  Validating judges against {len(cases)} labeled case(s) (real calls)...\n")
    by_checker: dict = {}
    mismatches = []
    for i, case in enumerate(cases, 1):
        name = case["checker"]
        checker = getattr(C, name, None)
        if checker is None:
            print(f"  [{i}] unknown checker {name!r} — skipped")
            continue
        finding = checker(case["reply"], case.get("scenario", {}))
        got = bool(finding.flagged) if finding is not None else False
        want = bool(case["expect_flagged"])
        ok = got == want
        st = by_checker.setdefault(name, {"ok": 0, "total": 0})
        st["total"] += 1
        st["ok"] += int(ok)
        print(f"  {'OK  ' if ok else 'MISS'} [{name}] want={want} got={got} — {case.get('note', '')}")
        if not ok:
            mismatches.append(case)
    print("\n" + "=" * 60 + "\nJUDGE CALIBRATION\n" + "=" * 60)
    for name in sorted(by_checker):
        st = by_checker[name]
        print(f"  {name:<18} {st['ok']}/{st['total']} agree ({st['ok'] / st['total']:.0%})")
    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(es) — judge disagrees with the label:")
        for c in mismatches:
            print(f"  • [{c['checker']}] want={c['expect_flagged']} — {c.get('note', '')}\n    “{c['reply']}”")
    print()
    return len(mismatches)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=_CORPUS, help="scenarios JSON (default: evals/quality_corpus.json)")
    ap.add_argument("--samples", type=int, default=1, help="generations per scenario (default 1)")
    ap.add_argument("--only", default=None, help="run only scenarios whose name contains this substring")
    ap.add_argument("--out", default=None, help="also write the full results to this JSON file")
    ap.add_argument("--gate", type=float, default=None,
                    help="exit 1 if any failure class exceeds this rate (e.g. 0.0). For CI/regression use.")
    ap.add_argument("--check-judges", action="store_true",
                    help="validate the LLM judges against evals/judge_cases.json instead of running the corpus")
    args = ap.parse_args()

    if args.check_judges:
        return 1 if check_judges(_JUDGE_CASES) else 0

    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)
    if args.only:
        corpus = [s for s in corpus if args.only.lower() in (s.get("name") or "").lower()]
    if not corpus:
        print("no scenarios to run", file=sys.stderr)
        return 2

    print(f"⚠️  Real OpenAI calls: {len(corpus)} scenario(s) × {args.samples} sample(s) "
          f"(+ judges). Generating...\n")
    results = run(corpus, args.samples)
    agg = aggregate(results)
    report(results, agg)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"results": results, "aggregate": agg}, f, indent=2)
        print(f"wrote {args.out}")

    if args.gate is not None:
        worst = max((st["flagged"] / st["total"] for st in agg["classes"].values() if st["total"]),
                    default=0.0)
        if worst > args.gate:
            print(f"GATE FAILED: worst class rate {worst:.0%} > {args.gate:.0%}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
