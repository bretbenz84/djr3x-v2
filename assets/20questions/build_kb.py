#!/usr/bin/env python3
"""Distill the allenai/twentyquestions dataset into a compact knowledge base for R3X.

The raw dataset (twentyquestions-all.jsonl, ~20 MB) is research data: 8,800+ subjects
and 22,000+ questions, but very sparse — a median of ~6 answered questions per subject.
That is too sparse to drive a pure belief-narrowing ("Akinator") engine, but it is an
excellent source for two things we DO use at runtime:

  1. A SPINE of proven, high-coverage yes/no discriminator questions (real phrasings,
     real yes-rates) for strong, varied openings.
  2. A SUBJECT VOCABULARY — the real things people actually pick in 20 Questions — used
     to ground/clean R3X's final guess.

This script emits `r3x_kb.json` (small, committed). The raw jsonl files are gitignored.

Run:  ./venv/bin/python assets/20questions/build_kb.py
Source: https://github.com/allenai/twentyquestions
"""
from __future__ import annotations

import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "twentyquestions-all.jsonl")
OUT = os.path.join(HERE, "r3x_kb.json")

# Spine concepts in roughly the order a good 20-questions player narrows the field.
# For each concept we offer candidate phrasings; the builder picks the highest-coverage
# phrasing that actually appears in the dataset (so the wording + stats are real).
# `parent` lets the runtime skip questions made redundant by an earlier answer, e.g. once
# "is it alive?" is NO, the animal/person/plant branches are pointless.
SPINE_CONCEPTS = [
    ("alive",      None,      ["is it alive?", "is it a living thing?", "is it living?"]),
    # `person` is intentionally parentless (not gated behind alive=yes): famous DEAD people and
    # fictional characters answer "is it alive?" -> no, and we must still discover they're a
    # person rather than chasing them down the inanimate-object branch.
    ("person",     None,      ["is it a person?", "is it human?", "is it a human?"]),
    ("animal",     "alive",   ["is it an animal?"]),
    ("plant",      "alive",   ["is it a plant?"]),
    ("manmade",    "not_alive", ["is it man made?", "is it manmade?", "is it artificial?"]),
    ("place",      None,      ["is it a place?", "is it a location?"]),
    ("edible",     None,      ["is it edible?", "can you eat it?"]),
    ("bigger",     None,      ["is it bigger than a breadbox?", "is it bigger than a microwave?",
                               "is it large?", "is it big?"]),
    ("handheld",   None,      ["can you hold it in your hand?", "is it small enough to hold in your hand?",
                               "can you hold it?"]),
    ("indoors",    None,      ["is it found indoors?", "is it found in a house?",
                               "is it usually found indoors?", "do you find it inside?"]),
    ("metal",      "not_alive", ["is it made of metal?", "is it metal?"]),
    ("electronic", "not_alive", ["is it electronic?", "does it use electricity?", "is it electrical?"]),
    ("tool",       "not_alive", ["is it a tool?"]),
    ("wearable",   "not_alive", ["is it clothing?", "do you wear it?", "can you wear it?"]),
    ("vehicle",    "not_alive", ["is it a vehicle?", "can you ride in it?"]),
]


def norm_q(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip().lower())
    if not q.endswith("?"):
        q += "?"
    return q


def main() -> None:
    if not os.path.exists(SRC):
        raise SystemExit(f"missing {SRC} — download from allenai/twentyquestions")

    # subject -> {normalized_question: majority(bool)}; plus per-question coverage & yes count.
    subj_q: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    q_cov: collections.Counter = collections.Counter()
    q_yes: collections.Counter = collections.Counter()
    subjects: set[str] = set()

    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("high_quality") or r.get("is_bad"):
                continue
            subj = str(r.get("subject", "")).strip().lower()
            if not subj:
                continue
            q = norm_q(str(r.get("question", "")))
            maj = bool(r.get("majority"))
            subjects.add(subj)
            subj_q[subj][q] = maj
            q_cov[q] += 1
            if maj:
                q_yes[q] += 1

    # Build the spine: best real phrasing per concept.
    spine = []
    for concept, parent, candidates in SPINE_CONCEPTS:
        best = None
        for cand in candidates:
            nq = norm_q(cand)
            cov = q_cov.get(nq, 0)
            if best is None or cov > best[1]:
                best = (nq, cov)
        nq, cov = best
        if cov < 30:
            # Not enough real data for this phrasing; still include the canonical wording
            # (the LLM voices it) but mark coverage so the runtime can deprioritize.
            yes_rate = None
        else:
            yes_rate = round(q_yes.get(nq, 0) / cov, 3)
        spine.append({
            "concept": concept,
            "parent": parent,
            "question": nq,
            "coverage": cov,
            "yes_rate": yes_rate,
        })

    # Subject vocabulary for grounding the final guess. Keep clean, short, real subjects.
    vocab = sorted({
        s for s in subjects
        if 2 <= len(s) <= 30
        and re.fullmatch(r"[a-z][a-z '\-]*[a-z]", s)
        and len(s.split()) <= 3
    })

    kb = {
        "version": 1,
        "source": "allenai/twentyquestions (twentyquestions-all.jsonl)",
        "spine": spine,
        "subjects": vocab,
        "stats": {
            "raw_subjects": len(subjects),
            "vocab_subjects": len(vocab),
            "spine_questions": len(spine),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB)")
    print(f"  spine: {len(spine)} questions  |  vocab: {len(vocab)} subjects "
          f"(from {len(subjects)} raw)")
    for s in spine:
        yr = "n/a " if s["yes_rate"] is None else f"{s['yes_rate']:.2f}"
        print(f"    [{s['concept']:>10}] yes={yr} cov={s['coverage']:>4}  {s['question']}")


if __name__ == "__main__":
    main()
