#!/usr/bin/env python3
"""
tools/gpt5_ab_test.py — A/B the conversation model: gpt-4o-mini vs gpt-5.4-mini.

Runs a FIXED corpus of user turns through the REAL conversation pipeline once per model
(clean reset between), capturing Rex's reply + per-turn wall time, and writes a
side-by-side markdown + JSONL. Both variants see identical user turns and the same
accumulating conversation context — only the conversation model differs — so the diff
isolates the model. Reuses the conversation_text_harness machinery (full prompt
assembly, social_frame, the wired llm_compat path).

LIVE API — makes real calls to BOTH models (a few cents). NOT part of the unittest
suite. See docs/gpt-5_4_mini.md ("A/B testing method").

    venv/bin/python tools/gpt5_ab_test.py
    venv/bin/python tools/gpt5_ab_test.py --file my_corpus.txt --person "Bret Benziger"
    venv/bin/python tools/gpt5_ab_test.py --candidate-effort low   # try a touch of reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from intelligence import interaction, premise_memory, topic_thread  # noqa: E402
from memory import conversations as conv_memory  # noqa: E402
from tools import conversation_text_harness as H  # noqa: E402

# A corpus that exercises the failure modes we've been fixing — a casual one-off mention
# (pizza), a string of low-energy answers, a topic the user cares about, a music decline,
# and a remarkable claim. Mirrors the real logged sessions so the A/B is comparable to
# what Bret actually experienced. Override with --file (one turn per line).
DEFAULT_CORPUS = [
    "not much, just eating pizza",
    "now just enjoying it",
    "I've made a lot of good progress on my robot",
    "the motor control system",
    "autonomous motion",
    "yeah, I'm making him sassy",
    "I think so",
    "I like classical music",
    "no thank you",
    "I love pondering the universe and how it came to be",
    "just the incomprehensible size of the universe",
    "I like a good challenge",
]


def _reset_conversation() -> None:
    """Clear the per-session conversation state so each model starts from the same blank
    slate (the person's stored facts are NOT touched — both models get identical
    person context; only the live conversation resets)."""
    try:
        conv_memory.clear_transcript()
    except Exception:
        pass
    for mod in (topic_thread, premise_memory):
        try:
            mod.clear()
        except Exception:
            pass
    # Session-scoped interaction state added in earlier rounds (harmless if absent).
    for attr, value in (
        ("_idle_plans_asked", set()),
        ("_idle_banter_count", 0),
        ("_idle_banter_threshold", None),
    ):
        if hasattr(interaction, attr):
            try:
                setattr(interaction, attr, value)
            except Exception:
                pass


def _apply_variant(model: str, effort, pass_temp: bool, verbosity=None) -> None:
    config.LLM_CONVERSATION_MODEL = model
    config.LLM_REASONING_EFFORT = effort
    config.LLM_GPT5_PASS_TEMPERATURE = pass_temp
    config.LLM_VERBOSITY = verbosity


def _run_variant(label: str, corpus: list[str], person_id, name) -> list[dict]:
    _reset_conversation()
    # Warm the connection (uses the now-selected conversation model) so the first turn
    # doesn't eat cold-TLS latency in the comparison.
    try:
        interaction.llm.warmup()
    except Exception:
        pass
    results = []
    for i, text in enumerate(corpus, 1):
        t0 = time.monotonic()
        try:
            turn = H._run_turn(text, person_id=person_id, person_name=name, no_llm=False)
            reply = (turn.get("response") or "").strip()
            err = None
        except Exception as exc:  # noqa: BLE001
            reply, err = "", f"{type(exc).__name__}: {exc}"
        secs = time.monotonic() - t0
        print(f"  [{label}] turn {i:2d} ({secs:4.1f}s): {reply[:80]}")
        results.append({"user": text, "reply": reply, "secs": round(secs, 2), "error": err})
    return results


def _write_sidebyside(corpus, runs, out_dir: Path, meta: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"gpt5_ab_{stamp}.md"
    jsonl_path = out_dir / f"gpt5_ab_{stamp}.jsonl"

    labels = list(runs.keys())
    a, b = labels[0], labels[1]
    lines = [
        f"# Conversation A/B — {a}  vs  {b}",
        "",
        f"_Run {stamp}. Person: {meta['person']}. {len(corpus)} turns. Same turns + "
        "context for both; only the conversation model differs._",
        "",
    ]
    # Per-turn side by side.
    for i, text in enumerate(corpus):
        ra, rb = runs[a][i], runs[b][i]
        lines += [
            f"### Turn {i + 1} — User: {text}",
            "",
            f"- **{a}** _({ra['secs']}s)_: {ra['reply'] or '[no reply]'}"
            + (f"  ⚠️ {ra['error']}" if ra.get("error") else ""),
            f"- **{b}** _({rb['secs']}s)_: {rb['reply'] or '[no reply]'}"
            + (f"  ⚠️ {rb['error']}" if rb.get("error") else ""),
            "",
        ]
    # Latency summary.
    def _avg(label):
        xs = [r["secs"] for r in runs[label]]
        return sum(xs) / len(xs) if xs else 0.0
    lines += [
        "## Latency (wall-clock per turn, includes constant pipeline overhead)",
        "",
        f"- **{a}**: avg {_avg(a):.2f}s  (total {sum(r['secs'] for r in runs[a]):.1f}s)",
        f"- **{b}**: avg {_avg(b):.2f}s  (total {sum(r['secs'] for r in runs[b]):.1f}s)",
        "",
        "_Note: per-turn time includes the (constant) command/intent classifiers, so the "
        "model delta is smaller than the totals. Raw model TTFT is in the smoke-test "
        "results in docs/gpt-5_4_mini.md._",
        "",
        "## Variants",
        "",
        f"- **{a}**: {json.dumps(meta['variants'][a])}",
        f"- **{b}**: {json.dumps(meta['variants'][b])}",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, text in enumerate(corpus):
            f.write(json.dumps({"turn": i + 1, "user": text,
                                a: runs[a][i], b: runs[b][i]}) + "\n")
    return md_path, jsonl_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", help="Corpus file (one user turn per line; # comments ok).")
    ap.add_argument("--person", default="Bret Benziger", help="Person name for context.")
    ap.add_argument("--person-id", type=int, help="Existing person_id to use.")
    ap.add_argument("--candidate-model", default="gpt-5.4-mini")
    ap.add_argument("--candidate-effort", default="none",
                    help="reasoning_effort for the candidate (none/low/medium/...).")
    ap.add_argument("--candidate-verbosity", default=None,
                    help="verbosity for the candidate (low/medium/high). low = terser replies.")
    ap.add_argument("--out", default="logs/gpt5_ab", help="Output directory.")
    args = ap.parse_args()

    # Corpus.
    corpus = list(DEFAULT_CORPUS)
    if args.file:
        corpus = [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]

    # Person + world state (reuse the harness; create a test row if needed).
    args.create_person = True
    args.mode = "script"
    person_id, name = H._resolve_person(args)
    H._prime_world_state(person_id, name)

    cand_label = f"{args.candidate_model} (effort={args.candidate_effort}"
    cand_label += f", verbosity={args.candidate_verbosity})" if args.candidate_verbosity else ")"
    variants = {
        "gpt-4o-mini (baseline)":
            {"model": "gpt-4o-mini", "effort": None, "pass_temp": False, "verbosity": None},
        cand_label:
            {"model": args.candidate_model, "effort": (args.candidate_effort or None),
             "pass_temp": True, "verbosity": args.candidate_verbosity},
    }

    print(f"A/B over {len(corpus)} turns. Person={name!r} (id={person_id}).\n")
    runs: dict[str, list[dict]] = {}
    # Snapshot the live config so we can restore it (this is a diagnostic, not a flip).
    saved = (config.LLM_CONVERSATION_MODEL, config.LLM_REASONING_EFFORT,
             config.LLM_GPT5_PASS_TEMPERATURE, config.LLM_VERBOSITY)
    try:
        for label, v in variants.items():
            print(f"--- {label} ---")
            _apply_variant(v["model"], v["effort"], v["pass_temp"], v.get("verbosity"))
            runs[label] = _run_variant(label, corpus, person_id, name)
            print()
    finally:
        (config.LLM_CONVERSATION_MODEL, config.LLM_REASONING_EFFORT,
         config.LLM_GPT5_PASS_TEMPERATURE, config.LLM_VERBOSITY) = saved

    meta = {"person": name, "variants": variants,
            "stamp": datetime.now().strftime("%Y%m%d-%H%M%S")}
    md_path, jsonl_path = _write_sidebyside(corpus, runs, Path(args.out), meta)
    print(f"Wrote:\n  {md_path}\n  {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
