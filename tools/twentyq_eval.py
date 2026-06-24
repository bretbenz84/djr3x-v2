#!/usr/bin/env python3
"""Offline self-play evaluator for R3X's 20 Questions guesser.

Plays full games against an LLM "player" oracle that answers truthfully about a given secret,
so we can measure how well the guesser converges without a human in the loop. The oracle is
deliberately fair (answers "is it a type of X" yes when the secret IS an X), but it is still
an imperfect literalist — a real human gives cleaner answers, so the win rate here is a LOWER
bound on real performance.

Usage:
    ./venv/bin/python tools/twentyq_eval.py "guitar" "bicycle" "Coney Island" ...
    ./venv/bin/python tools/twentyq_eval.py --verbose "pizza"

Prints one line per secret (WIN/LOSE + question count + final guess) and a SUMMARY with the
win count. With --verbose it also prints each turn and the live candidate shortlist.
"""
from __future__ import annotations

import os
import re
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import unittest.mock as mock  # noqa: E402

from features import games  # noqa: E402


def _oracle(secret: str, question: str) -> str:
    raw = games._quick_call(
        f'You are playing 20 Questions and you are thinking of "{secret}". Another player asks: '
        f'"{question}". Answer honestly about {secret} with ONLY one word: yes, no, or sometimes. '
        f'If the question asks "is it a type of X" and {secret} is an X (or is X itself), answer yes.',
        temperature=0, max_tokens=4,
    ).strip().lower()
    for w in ("sometimes", "yes", "no"):
        if w in raw:
            return w
    return "no"


def _fake_rex(ctx, pid=None):
    m = re.search(r'"([^"]*\?)"', ctx)
    return m.group(1) if m else ctx[:60]


def play(secret: str, verbose: bool = False, max_turns: int = 24) -> dict:
    games._active_game = "20_questions"
    games._game_state = {}
    games._20q_start(None)
    games._20q_handle("ready", None)
    last_guess = ""
    for _ in range(max_turns):
        st = games._game_state
        cands = st.get("candidates") or []
        if st.get("phase") == "guessing":
            guess = (st.get("pending_guess") or "").lower().strip()
            last_guess = st.get("pending_guess") or ""
            ans = "yes" if (guess in secret.lower() or secret.lower() in guess) else "no"
            if verbose:
                print(f"    Q{st.get('question_count')}: GUESS -> {last_guess!r} [{ans}]")
        else:
            ans = _oracle(secret, st.get("last_question", ""))
            if verbose:
                sl = ("  | " + ", ".join(cands[:5])) if cands else ""
                print(f"    Q{st.get('question_count'):>2}: {st.get('last_question',''):<42} [{ans}]{sl}")
        _, done = games._20q_handle(ans, None)
        if done:
            return {
                "secret": secret,
                "result": games._game_state.get("result", "?"),
                "questions": games._game_state.get("question_count", 0),
                "final_guess": games._game_state.get("final_guess") or last_guess,
            }
    return {"secret": secret, "result": "timeout", "questions": max_turns, "final_guess": last_guess}


def main(argv: list[str]) -> None:
    verbose = "--verbose" in argv
    secrets = [a for a in argv if not a.startswith("--")]
    if not secrets:
        print("usage: twentyq_eval.py [--verbose] <secret> [<secret> ...]")
        return
    rows = []
    with mock.patch.object(games, "_rex_respond", side_effect=_fake_rex), \
         mock.patch.object(games, "_body_beat", return_value=None):
        for s in secrets:
            if verbose:
                print(f"\n===== {s!r} =====")
            r = play(s, verbose=verbose)
            rows.append(r)
            print(f"  {r['result'].upper():8} q={r['questions']:>2}  {s!r}"
                  f"  (last guess: {r['final_guess']!r})")
    wins = sum(1 for r in rows if r["result"] == "win")
    print(f"\nSUMMARY: {wins}/{len(rows)} wins")


if __name__ == "__main__":
    main(sys.argv[1:])
