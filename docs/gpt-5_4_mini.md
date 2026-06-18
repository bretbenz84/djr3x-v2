# Migrating Rex's conversation to a GPT-5-class model (`gpt-5.4-mini`)

**Status:** ✅ **LIVE** as of 2026-06-17 — the **conversation path runs on `gpt-5.4-mini`**
(`effort=none`, `verbosity=low`, temperature forwarded), paired with a prompt-brevity
pass. The classifiers / routers / JSON-extraction / vision calls still run on
`gpt-4o-mini` (hybrid rollout). **Rollback = set `LLM_CONVERSATION_MODEL` back to
`LLM_MODEL`** — one line. Config block at the bottom.

Researched 2026-06-17 (model facts re-verified against OpenAI docs same day). Audited
surface area: **37 `chat.completions.create` call sites across 14 files**.

---

## TL;DR

- `gpt-5.4-mini` is a **hybrid model** — non-reasoning by default, with optional
  reasoning via `reasoning_effort`. Two hard API breakages vs `gpt-4o-mini`:
  1. `max_tokens` is rejected (when reasoning engaged) → must send `max_completion_tokens`.
  2. A non-default `temperature` is rejected (HTTP 400) when reasoning is engaged — but
     **accepted at `reasoning_effort="none"`** (confirmed by live smoke test 2026-06-17;
     see "Settle the temperature question"). The conversation hot path runs at `none`, so
     it can keep deterministic/explicit temperatures.
- A **compatibility shim** (`intelligence/llm_compat.py`) translates these in ONE place
  and is a **no-op for `gpt-4o-mini`**, so wiring call sites through it changed nothing.
- **Hybrid rollout**: flip only the **user-facing conversation** to `gpt-5.4-mini`; keep
  the deterministic classifiers / routers / JSON-extraction / vision calls on
  `gpt-4o-mini`. Shrinks the risk + test surface from 37 calls to ~6.
- The **real gate** is `tools/gpt5_smoke_test.py` (live API). The unittest suite mocks
  the LLM, so it will NOT catch real API breakage.
- **Rollback is one line**: set `LLM_CONVERSATION_MODEL` back to `LLM_MODEL`.

---

## What's already built (this scaffolding)

| Piece | File | What it does |
|---|---|---|
| Compatibility shim | `intelligence/llm_compat.py` | `prepare_chat_params()` (pure) + `create()`. Renames `max_tokens`→`max_completion_tokens`, injects `reasoning_effort`/`verbosity`, drops/keeps `temperature` for GPT-5 models; no-op for `gpt-4o-mini`. |
| Config (OFF) | `config.py` | `LLM_CONVERSATION_MODEL` (=`LLM_MODEL`), `LLM_REASONING_EFFORT` (None), `LLM_VERBOSITY` (None), `LLM_GPT5_PASS_TEMPERATURE` (False). |
| Wired conversation calls | `intelligence/llm.py` | 6 user-facing generators routed through the shim via `llm_compat.conversation_model()`: `warmup`, `stream_response` (main path), `scenery_change_remark`, `generate_curiosity_question`, `generate_onboarding_reaction`, `generate_expression_reaction`. |
| Mock tests | `tests/test_llm_compat.py` | 17 tests locking the param-translation contract (no network). |
| Live smoke test | `tools/gpt5_smoke_test.py` | One real API call per shape; settles the temperature question. Not in CI. |
| A/B runner | `tools/gpt5_ab_test.py` | Runs a fixed corpus through both models (real pipeline), writes a side-by-side. Not in CI. See "A/B results" below. |

The other ~31 call sites (action router, intent/empathy/address classifiers, memory
extraction, session summary, vision, games, trivia) **still call the client directly**
and stay on `gpt-4o-mini`. They are listed under "Full migration" below.

---

## The model: `gpt-5.4-mini` facts

(Verified via OpenAI docs, 2026-06-17.)

| Property | Value |
|---|---|
| Model ID | `gpt-5.4-mini` (snapshot `gpt-5.4-mini-2026-03-17`) |
| Type | **Hybrid** — OpenAI's model page tags it *"not a reasoning model, though it supports reasoning token support."* Runs **non-reasoning by default**; engages reasoning via `reasoning_effort`. (Artificial Analysis lists separate `-non-reasoning` and `-medium` variants.) |
| APIs | Chat Completions **and** Responses (we stay on Chat Completions) |
| Context / output | 400K context / 128K max output |
| Pricing | **$0.75/M input, $4.50/M output**, $0.075/M cached input (+10% on regional data-residency endpoints) |
| Knowledge cutoff | **August 31, 2025** |
| Capabilities | vision (input), streaming, JSON/structured outputs, tool calling |
| `reasoning_effort` | `none | low | medium | high | xhigh` — `none` keeps time-to-first-token low. **NOTE:** `minimal` (a GPT-5/5.1-era value) appears to be **dropped** for 5.4 — every current source omits it. Don't rely on it without confirming. |
| `verbosity` | `low | medium | high` |

**Cost vs `gpt-4o-mini`** (~$0.15/$0.60 per M): roughly **5× input / 7.5× output**,
*plus* reasoning tokens count as output. Low absolute cost for a personal robot, but not
nothing — keep `reasoning_effort` low on the hot path.

**Latency is the real concern** for a real-time voice robot. Reasoning raises
time-to-first-token. Mitigate with `reasoning_effort="none"` on the conversation path —
the **non-reasoning** mode benchmarks at ~**0.67s TTFT** and ~150 tok/s output on
OpenAI's endpoint, which is in the right ballpark for voice. (There's now a body of
"GPT-5.4-mini for voice AI" writeups recommending exactly this `none` approach.) Verify
on our own traffic with the existing `[ttfs]` logs.

---

## The two breakages and how the shim handles them

1. **`max_tokens` → `max_completion_tokens`.** The shim renames it automatically for any
   reasoning model. (Affects 36/37 call sites — every call passes `max_tokens`.)

2. **`temperature`.** ✅ **SETTLED by live smoke test (2026-06-17)** — see results below.
   Temperature support is **gated on `reasoning_effort`, not on the model**:
   - `reasoning_effort="none"` → `temperature=0` is **ACCEPTED** (non-reasoning mode behaves
     like a standard model). So on the conversation hot path (which runs at `none`) we can
     forward temperature: `LLM_GPT5_PASS_TEMPERATURE=True` is safe.
   - `reasoning_effort="medium"` (or any reasoning-engaged level) → `temperature=0` is
     **REJECTED, HTTP 400**: *"Unsupported value: 'temperature' does not support 0.0 with
     this model. Only the default (1) value is supported."*

   The shim's default is still to **drop** `temperature` for GPT-5 models, gated on
   `LLM_GPT5_PASS_TEMPERATURE`; flip that flag to forward it **only while everything wired
   runs at `effort="none"`**.

   - 16 of our calls use `temperature=0` for determinism (routers/classifiers/JSON).
     The hybrid rollout still leaves them on `gpt-4o-mini` for now. When migrated, they
     **must run at `effort="none"`** to keep `temperature=0` — any reasoning-engaged level
     will 400 on `temperature=0`.

Also worth a real test when you migrate those paths: GPT-5 is **stricter about JSON
schemas** (7 `response_format={"type":"json_object"}` calls) and vision detail handling
(4 `image_url` calls).

---

## The safe sequence to actually turn it on

Do these in order. Stop and evaluate between each.

### Step 1 — Settle the temperature question (live smoke test)
```bash
venv/bin/python tools/gpt5_smoke_test.py --model gpt-5.4-mini --effort none            # temp dropped
venv/bin/python tools/gpt5_smoke_test.py --model gpt-5.4-mini --effort none --pass-temp # temp forwarded
```
- Watch which `temperature=0` line PASSES. If it passes **with `--pass-temp`**, the model
  accepts temperature → you may later keep deterministic temps. If it 400s, leave temp
  dropped.
- Note the per-shape **seconds** (especially `streaming`) — that's your latency budget.

#### Results (run 2026-06-17, `gpt-5.4-mini`)

| Run | Outcome |
|---|---|
| `--effort none` (temp dropped) | **4/4 PASS** — plain, streaming, temperature=0, JSON |
| `--effort none --pass-temp` | **4/4 PASS** — `temperature=0` **FORWARDED and accepted** |
| `--effort none --pass-temp --vision` | **5/5 PASS** — vision (`image_url`) returns `'Blue'` |
| `--effort medium --pass-temp` | **temperature=0 → HTTP 400** (only default temp allowed when reasoning is engaged) |

- **Temperature question: answered** — accepted at `effort="none"`, rejected once reasoning
  is engaged. Set `LLM_GPT5_PASS_TEMPERATURE=True` *if* you keep the hot path at `none`.
- **Latency: great for voice** — every shape < 2s; `streaming` ~0.4–0.6s to first content.
- **JSON + vision both work** out of the box on the conversation-class shapes.
- ⚠️ **Gotcha: reasoning eats the output budget.** At `effort="medium"`, `plain`/`streaming`
  returned **empty strings** — the small `max_completion_tokens` (20) was consumed by hidden
  reasoning tokens before any visible text. If you ever raise `reasoning_effort` above
  `none`, **raise `max_completion_tokens`** or replies will truncate to nothing. This is a
  strong reason to keep the conversation path at `none`.

Then set in `config.py` (or `.env`):
- `LLM_REASONING_EFFORT = "none"` for the hot path (`"minimal"` appears dropped for 5.4 —
  see model facts; use `"low"` if you want a touch more reasoning).
- `LLM_GPT5_PASS_TEMPERATURE = True` **only if** the smoke test confirmed it.

### Step 2 — Flip ONLY the conversation model and A/B it
```python
# config.py
LLM_CONVERSATION_MODEL = "gpt-5.4-mini"
```
This flips the 6 wired generators (incl. the main streaming reply) to `gpt-5.4-mini`.
Everything else stays on `gpt-4o-mini`. Then A/B (see below). **Rollback = set it back to
`LLM_MODEL`.**

### Step 3 — (optional) widen the rollout
If the A/B is clearly better and latency is acceptable, route more call sites through the
shim and give them their own model config (e.g. `CONVERSATION_ARC_OPENAI_MODEL`,
`generate_session_summary`). Do the deterministic classifiers/routers LAST and only after
the smoke test proves temperature behavior — they're the riskiest.

---

## A/B testing method

You can run full conversation turns with **no hardware**:

- **Text harness** (fastest iteration):
  ```bash
  venv/bin/python tools/conversation_text_harness.py --person "Bret Benziger" \
    "not much, just eating pizza" "I've made progress on my robot" "I like classical music"
  ```
  Run the *same* turns once with `LLM_CONVERSATION_MODEL="gpt-4o-mini"` and once with
  `"gpt-5.4-mini"`, and diff the replies for quality (curiosity, repetition, persona).

- **Text-only GUI** for a live feel: `python main.py --gui --noaudio`.

- **Latency**: grep the run logs for `[ttfs]` and compare `audio_started_*`/
  `response_queued_*` between models. The reasoning model WILL be slower to first token;
  decide if it's acceptable, and tune `reasoning_effort` down if not.

- **Cost**: spot-check token usage (the reply path's `max_completion_tokens`) and
  remember reasoning tokens are billed as output.

**Build an A/B corpus**: a handful of canned conversations (`--file turns.txt`) that
exercise the failure modes we've fixed (idle nudges, onboarding, holiday/plans, casual
one-off topics, surprise reactions). Re-run them on each model and eyeball + diff. This
is also the natural place to add a small script that loops the corpus through both models
and writes a side-by-side.

> ⚠️ The unittest suite **mocks the LLM**. It will pass green on `gpt-5.4-mini` and prove
> nothing about real behavior. The smoke test + harness A/B are the real evaluation.

### A/B runner

`tools/gpt5_ab_test.py` runs a fixed corpus through the REAL pipeline once per model
(clean reset between), captures Rex's reply + per-turn wall time, and writes a
side-by-side `.md`/`.jsonl` to `logs/gpt5_ab/` (gitignored). Live API; not in CI.
```bash
venv/bin/python tools/gpt5_ab_test.py                          # baseline vs gpt-5.4-mini@none
venv/bin/python tools/gpt5_ab_test.py --candidate-verbosity low
venv/bin/python tools/gpt5_ab_test.py --file corpus.txt --candidate-effort low
```

### A/B results (run 2026-06-17, `gpt-5.4-mini` @ `effort=none`, `pass_temp=true`)

12-turn corpus (pizza one-off → robot progress → low-energy answers → music decline →
cosmology), two runs (default + `verbosity=low`). Verdict:

- **Quality: clear win for `gpt-5.4-mini`.** Sharper, more specific, more in-character —
  consistent across both runs. Representative:
  - *"Motor control is where the robot stops being a glorified paperweight and starts earning its keep."*
  - *"A robot without attitude is just a broom with ambitions."*
  - *"A good challenge is just a problem with better posture."*
  - vs gpt-4o-mini's more generic *"That's awesome! I bet it's feeling good to see your hard work come together."*
- **Length: a real tradeoff.** `gpt-5.4-mini` averaged **~30 words/reply vs gpt-4o-mini's ~22**,
  *even at `verbosity=low`* (which barely moved it). It reliably lands 2–3 sentences **+ a
  question** every turn. NOTE: gpt-4o-mini also over-runs the persona's "one short sentence"
  rule — **brevity is a shared PROMPT problem the model swap does not fix (and slightly
  worsens).** If "too long / too many questions" still grates after the flip, that's
  separate prompt-brevity work (tighten the brevity directive / `_max_tokens_for_agenda`),
  not a model issue.
- **Latency: fine for voice.** Full-reply wall-time was ~+2–3s (baseline avg 9.6s vs
  candidate 11.8s), but that's *complete-reply* time incl. constant pipeline overhead and
  longer replies. Real user-facing TTFS is ~0.5s (smoke test) because the path streams
  sentence-by-sentence. gpt-5.4-mini just *talks a little longer*. Verify on-device with
  `[ttfs]`.
- **Model-independent quirk seen:** the turn "no thank you" (after "I like classical
  music") routed both models to *"I heard the correction, but I need one clear fact to
  update."* — a pipeline routing issue, identical for both, unrelated to the model choice.

**Recommendation:** the flip is worth it for the persona/wit gain. Pair it with a brevity
pass if reply length matters to you. `verbosity="low"` helps only marginally here.

### Flipped + brevity pass (2026-06-17)

Acted on the recommendation. `LLM_CONVERSATION_MODEL="gpt-5.4-mini"`, `effort="none"`,
`verbosity="low"`, `pass_temp=True`, plus a sharpened `REX_CORE_PROMPT` brevity rule that
kills the **react + elaborate + question** three-part stack ("most turns END ON A
STATEMENT; pick ONE move — land a line OR ask one real question"). Re-A/B'd: gpt-5.4-mini
dropped from ~30 → ~27 words/reply and now mostly lands **one reaction + one question**
(~1–2 sentences) while keeping the wit.

Open observation (NOT changed): in the A/B harness, ~10/11 turns still end on a question
(gpt-4o-mini does too). That's the conversation **agenda** ("react, then ask one
follow-up") — on the **real robot** the `question_budget` (deliberately tight; do not
raise) throttles questions across turns, which the text harness doesn't replicate. So
live question frequency is lower than the harness shows. If it still reads as an interview
on-device, the lever is `conversation_agenda` / the question budget, not the model.

---

## Full migration (the other call sites, when/if you widen)

Route these through `llm_compat.create(client, ...)` the same way, giving each group its
own model config so you can flip them independently. **Test JSON + temperature on each.**

| Group | Files / calls | Notes |
|---|---|---|
| Action routing | `intelligence/action_router.py` (decide, warmup) | `temperature=0` — needs the temp answer first |
| Classifiers | `intent_classifier.py`, `empathy.py` (JSON), `awareness/address_mode.py`, `llm.py:classify_surprise/analyze_sentiment` | `temperature=0`, several JSON |
| Memory extraction | `llm.py` extract_facts/preferences/interests/events, `consolidate_session_memories` (JSON), `extract_name*` (JSON) | `temperature=0`, large `max_tokens`, JSON |
| Session summary / arc | `llm.py:generate_session_summary`, `_call_openai_summarizer` (`CONVERSATION_ARC_OPENAI_MODEL`) | background, can tolerate higher latency |
| Vision | `vision/scene.py`, `vision/face.py` (×3), `features/games.py` | `image_url` + `detail`; keep on `VISION_MODEL` until tested |
| Games / trivia | `features/games.py`, `features/trivia.py` | trivia.py has NO `max_tokens` today |
| Lazy/optional | `onboarding.py:_maybe_rephrase`, `tell_me_about.py` (JSON), `evals/checkers.py` | off-by-default / eval-only |

Each of these constructs its own `OpenAI(...)` client; pass that client to
`llm_compat.create(...)`. The shim doesn't own a client — it just translates params.

---

## Config reference

```python
# config.py — all default to current gpt-4o-mini behavior (migration OFF)
LLM_CONVERSATION_MODEL    = LLM_MODEL   # set "gpt-5.4-mini" to flip the conversation path
LLM_REASONING_EFFORT      = None        # "none" | "low" | "medium" | "high" | "xhigh"  ("minimal" appears dropped for 5.4)
LLM_VERBOSITY             = None        # "low" | "medium" | "high"
LLM_GPT5_PASS_TEMPERATURE = False       # smoke-confirmed: True is safe IF the routed path stays at effort="none"
```

`reasoning_effort`/`verbosity`/temperature-handling apply **only** to reasoning models;
they're ignored for `gpt-4o-mini`, so these are safe to leave set even while OFF.
