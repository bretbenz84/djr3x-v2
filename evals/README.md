# Conversation-quality eval (LLM-in-the-loop)

Measures DJ-R3X's **generation** failure CLASSES across a corpus, instead of
patching one bad live line at a time. Runs real-ish scenarios through the actual
reply path — deterministic governor stack → `gpt-4o-mini` stream → the same
sentence/tail assembly + governance + comedy polish the speech queue uses — and
scores each reply with a set of checkers, reporting per-class flag **rates**.

This is the answer to "the only eval is run-the-robot-and-read-the-logs": live
runs are non-deterministic and only catch a failure when it happens to fire. This
catches the same classes deterministically and *quantifies* them ("sincere shares
roasted 22–56% of the time", "props invented 0%").

> ⚠️ **Makes real OpenAI calls** (one generation + a couple of judge calls per
> reply). A full corpus at `--samples 1` is ~20 cheap `gpt-4o-mini` calls. It is
> intentionally **NOT** part of `unittest discover -s tests` (that suite stays
> network-free). Run it deliberately.

## Run

```bash
source venv/bin/activate
python evals/run_quality_eval.py                 # whole corpus, 1 sample each
python evals/run_quality_eval.py --samples 5     # 5 samples/scenario → stabler rates
python evals/run_quality_eval.py --only sincere  # scenarios whose name matches
python evals/run_quality_eval.py --out report.json
python evals/run_quality_eval.py --gate 0.0      # exit 1 if ANY class flags (CI/regression)
python evals/run_quality_eval.py --check-judges  # validate the LLM judges (see below)
```

Rates are noisy at low sample counts (generation is non-deterministic) — use
`--samples 5`+ when comparing before/after a change.

## Validating the judges (`--check-judges`, `judge_cases.json`)

The LLM judges (`invented_prop`, `roasted_sincere`) are measurement instruments,
so they need their own ground truth. `judge_cases.json` is a small LABELED set
(reply + scenario + `expect_flagged`); `--check-judges` runs each through its
checker and reports agreement. Calibrate a judge's rubric against these labels —
**don't tune a judge in the same pass you grade a fix with** (that games the
metric). The judge over-flagging borderline-acceptable teasing? Add the
mislabeled reply to `judge_cases.json` with the correct label, tighten the rubric
in `checkers.py`, and re-run `--check-judges` until it agrees. The current
rubric is biased toward PASS when a reply clearly engages/acknowledges, so a
light in-character tease on top of genuine engagement is not flagged.

## What it checks (`checkers.py`)

**Deterministic** (cheap, exact, no network):
- `over_questioning` — more than one `?` in a reply (interrogation).
- `cantina_bleed` — Rex naming/assuming a cantina venue (backstory bleed).
- `banned_opener` — opens with "Ah,"/"Oh,"/"Well, well".
- `trail_off` — spoken reply ends mid-clause (the streaming cut-off class; the
  tail fix should keep this at 0%).
- `re_asks` — repeats a question Rex already asked this conversation.

**LLM-judge** (one cheap `gpt-4o-mini` JSON call; fails safe — a judge error
counts as not-flagged and shows up as a `judge_error` detail):
- `invented_prop` — asserts a physical prop the person has (a drink, a hat) with
  no visual data to support it.
- `roasted_sincere` — mocks/roasts a sincere share or needles a boundary instead
  of engaging (only runs on scenarios marked `"user_sincere": true`).

## Add a scenario

Drop an object into `quality_corpus.json` (no Python):

```json
{
  "name": "short-unique-id",
  "person_id": 1,
  "rex_last_line": "optional — Rex's previous line (seeds context + re-ask check)",
  "prior_user_turns": ["optional earlier user turns (seed steering)"],
  "utterance": "the user's turn to respond to",
  "user_sincere": true,
  "visible_context": "optional — what Rex can actually see (for the invented_prop judge)",
  "max_questions": 1,
  "watch": "freeform note: which classes this scenario targets"
}
```

Seed scenarios from real transcripts (`logs/conversation.log`) — especially turns
where Rex misbehaved — so the corpus grows toward the situations that actually
break.

## Baseline (2026-06-04, `--samples 3`, 8 scenarios)

| class            | rate | note |
|------------------|------|------|
| roasted_sincere  | 22–56% | **dominant issue** — the rebalance helped but didn't solve it |
| cantina_bleed    | ~8%  | minor backstory bleed |
| invented_prop    | 0%   | the rebalance + visual-rule fix holds at scale |
| trail_off        | 0%   | the ellipsis-tail fix holds |
| over_questioning | 0%   | |
| re_asks          | 0%   | |
| banned_opener    | 0%   | comedy polish strips them |

## Known limitations / fast-follows

- **Rates need samples.** Generation is non-deterministic; `roasted_sincere`
  swung 22%→56% across two 3-sample runs. Use `--samples 5`+ for stable numbers.
- **Offline `world_state` is empty** (no camera), so `visible_context` for the
  prop judge comes from the scenario, not live vision — which is correct for the
  "no object sensor data" assumption, but won't catch *mis-reading* a real scene.
- **Judge model = `gpt-4o-mini`.** Good enough for these classes; a stronger judge
  would reduce judge noise on borderline calls.
- **Grow the corpus** from real logs, and add classes as new failure modes show
  up (dead-end acks, repeated jokes, wrong-addressee in groups).
