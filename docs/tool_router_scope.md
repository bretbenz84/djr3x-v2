# Tool-Calling Router — Scope & Migration Plan

Status: PHASE 2 (humor + performance) LIVE 2026-08-13. The routing audit
(commit c7ef872) measured the shipped split and it was not close: the regex
lanes took **416 of 440** non-conversation executions (94.5%) across 1,141
audited turns, while the tool router took **8**. Phase 1's reasoning for
deferring humor/character — "their fast lanes work" — did not survive contact
with that data: for these families the classifier WAS the whole decision, so
every phrasing outside the pattern list became prose ("give me a zinger", "be
mean to me for a second", "pull a face" all classified as None), and the
matches only stayed safe because guards were hand-written one field incident at
a time.

What Phase 2 changed:
- `humor.*` (tell_joke, roast, free_bit) and `performance.*` (dj_bit,
  body_beat, mood_pose, impersonate) are in `TOOL_ROUTER_LIVE_ACTIONS`.
- The classifiers are NOT deleted. Online they are a DETECTOR, not a claim:
  `action_router.tool_router_owns()` converts a match into conversation.reply,
  which skips the ~0.8s JSON-prose call AND lets the reply call pick the tool —
  so the migration costs zero extra round-trips.
- The evidence gate for these seven is now GUARDS-ONLY
  (`humor_performance_refusal_reason`). Re-running the classifier as "evidence"
  had capped the tool router at exactly the patterns it was migrated off — 25 of
  31 valid off-pattern requests were vetoed. The refusal/narration/description
  guards remain and now cover `humor.tell_joke` and `performance.impersonate`,
  neither of which had them before.
- `performance.body_beat` / `mood_pose` schemas carry ENUMS generated from
  `performance_plan`; previously body_beat took no parameters at all, so the
  model could not name the gesture and `plan_for_action` defaulted to a head
  tilt. `performance.impersonate`'s arg was renamed `who` -> `target` to match
  every other router.
- OFFLINE keeps the deterministic classifiers (§2.4) — the local reply model is
  called with no tool surface, so with the link down they are the only thing
  that still performs. `TOOL_ROUTER_LIVE_ENABLED` reverts everything instantly.

`character.preference_query` was RETIRED entirely on 2026-08-13 (commit
6267d38) rather than migrated: an opinion question is conversation. Its stored
tastes became lean-brain context (`rex_preferences.prompt_lines`).

PHASE 3 (motion) LIVE 2026-08-13, and it is NOT shaped like Phases 1–2b. Motion
keeps its regex as the PRIMARY claim — `motion.*` is deliberately absent from
`TOOL_ROUTER_OWNED_ACTIONS`, so a >=0.95 classifier match still executes
immediately at today's latency and the tool governs only what it missed. The
misses are real and numerous: `rotate ninety degrees`, `rotate 90 degrees`,
`back yourself up a bit`, `back it up`, `scoot a little closer`, `get closer`,
`drive up here`, `go straight`, `face me`, `swivel left`, `hang a left`,
`veer right`, `scootch to your right` and `why don't you scoot forward` all
classify as None today and become conversation.

The evidence gate is the one place motion deviates from the other families.
GUARDS-ONLY was measured and REJECTED here: built from the three shipped negative
guards it admitted 31 of 31 figurative-motion utterances, including `let's move
on`, `moving forward, I want to try something` and `I need to run to the store` —
none of which carries a negator, a leading "why", or a speech verb. Guards-only
works where refusal and narration are lexically marked; figurative motion shares
its verbs with the literal sense. So `motion_command_refusal_reason` keeps a
positive test, just a much LOOSER one than `classify_explicit_motion`: an
imperative drive verb aimed at Rex (lead-ins stripped, base forms only), the
existing negative guards re-run on the lead-stripped body, and a new figurative
guard. Measured: admits 46/47 real misses, refuses 47/48 decoys, and refuses none
of the commands the regex already claims.

`motion.stop` did NOT migrate and never will (2.2) — the deterministic escape
(`_errand_stop_demanded` + `motion_controller.is_moving()`, watched by the eager
endpointer) claims it before any LLM sees the turn. `motion.explore` did not
migrate either: `classify_explicit_exploration` is already the
imperative-addressed-to-Rex test a tool gate would have to be, and an accepted
invite seizes the floor for minutes. Also fixed in this stage: three silent
arg-name drifts in the motion tool schemas (`degrees`/`distance`+`unit`/arc's
lone `direction`) and one enum-VALUE drift — the move schema said `backward`
while the executor tests `== "back"`, so "back up" would have driven Rex FORWARD.

PHASE 4a LANDED 2026-08-13: the JSON-prose fallback router is RETIRED behind
`ACTION_ROUTER_LLM_FALLBACK_ENABLED` (default False). The field logs made the
case — across 1,340 audited turns the LLM branch produced TWO executions, both
`character.preference_query`, retired the same day (6267d38). Every other
`router_takeover.*` in the corpus came from decide()'s deterministic pre-LLM
ladder. It cost 0.74s median on 30.5% of routed turns (42% of pre-reply
latency), and its prompt was silently truncated at
`ACTION_ROUTER_MAX_CONTEXT_CHARS` — `weather.query` and `web.search` fell off the
end of the router's own catalog, so it had been running degraded for weeks.

Retired behind a flag rather than deleted, deliberately: `_SYSTEM_PROMPT`,
`_coerce_decision`, `_clearly_conversational` and `_ACTION_CUE_RE` all stay, so
the rollback is a config flip rather than a code revert.

Remaining (Phase 4b, once the flag has held off in the field): delete the retired
branch and its now-dead helpers. `_clearly_conversational` has exactly one
caller; nothing else in the repo consults it. `conversation.repair` is the only
action the fallback was ever the sole route for, and it is deliberately NOT
becoming a live tool — `repair_moves.detect` served all 48 logged repairs, and
the router lane bypasses the bare-restatement and correction-reroute guards.

Phase 1 record follows.

Status: PHASE 1 LIVE 2026-08-01 — the seven intent-backed actions (time/date/
weather/capabilities/uptime/describe-scene/music-options) ride the lean reply
call as native tools (`TOOL_ROUTER_LIVE_ENABLED`, instant kill switch): the
model answers in prose or calls a tool, a call raises ToolCallRequested out of
the stream (before any speech) and dispatches the existing
_handle_classified_intent executor, stamped as final_executed_path=
"tool_router.<action>". Deterministic layers still run first — this catches
only what fell through to conversation. Cutover evidence: 50-turn collection,
tool ~92% vs shipped ~80%, decoys 0/6 false positives, and verified shipped
misses ("Stop the game." fumbled by three layers, "kill the music" answered
with "Systems nominal"). Humor/character deferred (their fast lanes work);
motion unchanged. Phase 0 record follows:

PHASE 0 SHIPPED 2026-08-01 — `intelligence/tool_router.py` (shadow-only,
off by default: set `TOOL_ROUTER_SHADOW_ENABLED = True` in user_config.py to
collect), report via `tools/tool_router_report.py`, contracts pinned in
`tests/test_tool_router.py`. Live smoke test: 10/10 correct tool choices
including the banter traps and context-bound game answers. Phase-0 deviation
from §2.1: per-action schemas live in tool_router._TOOL_DEFS (coverage-enforced
by tests) rather than on ActionSpec — they merge into the spec at Phase 4. Owner decision driving this:
"when the LLM does all the talking it's great — it's the regex commands that are
fragile." Constraint: **no new models** (16 GB RAM) — this design reuses the
existing conversation model's native tool-calling. Zero additional memory or
model downloads; routing folds INTO the reply call, so it *removes* one LLM
call per routed turn rather than adding one.

## 1. What exists today (inventory)

### 1.1 The action catalog — 40 actions, 14 categories

Source of truth: `ACTION_SPECS` in `intelligence/action_router.py` (auto-derives
`ACTION_CATALOG` / `ACTION_CATEGORIES` / valid + executable sets).

| Category | Actions | Notes |
|---|---|---|
| conversation | reply *(non-exec)*, repair | the default sink |
| memory | query, forget_specific, recent_discard, forget_person *(non-exec)*, event.cancel | recent_discard now has the grounded-reminder branch (2026-08-01) |
| boundary | emotional.boundary | |
| identity | who_is_speaking, name_correction, introduce_person *(non-exec)* | |
| humor | tell_joke, roast, free_bit | |
| performance | dj_bit, body_beat, mood_pose, impersonate | |
| character | preference_query | |
| game | start, stop, answer | |
| music | play, stop, skip, options | |
| vision | describe_scene, snapshot *(non-exec)* | |
| world | time.query, date.query, weather.query | |
| status | capabilities, uptime | |
| motion | turn, move, arc, come, stop, explore | firmware-guarded |
| system | sleep | |

### 1.2 The current decision ladder (`action_router.decide()`, ~line 2378)

1. `classify_explicit_control` — regex (memory discard, name correction, snapshot…)
2. `classify_explicit_humor` / `classify_explicit_performance` /
   `classify_explicit_character_preference` — regex families
3. `classify_explicit_motion` + `classify_explicit_motion_sequence` +
   `classify_motion_continuation` + `classify_explicit_exploration` — regex with
   a `_NON_MOTION_ACTION_WORDS` stoplist
4. `classify_explicit_impersonation` — regex
5. "deterministic conversational skip" — if no cue words, return
   `conversation.reply` WITHOUT any LLM call
6. Fallback: an **extra LLM call** (`ACTION_ROUTER_MODEL`, JSON-in-prose, no
   native tools) with the catalog pasted into the prompt, parsed by
   `_strip_code_fence` + `json.loads`
7. `_apply_context_overrides` on every path (games/music context, etc.)

Layered ON TOP of that, in `interaction.py`:
- `dialogue_act` gate — `answer_to_rex` bindings set `skip_router=True`
  (has swallowed motion AND impersonation commands; wiring point 5)
- pre-dialogue-gate "takeovers" (`_explicit_motion_takeover` etc.) that bypass
  the gate for specific regex families
- `config.ACTION_ROUTER_EXECUTE_ACTIONS` allowlist — anything not listed logs
  `not_in_execute_allowlist` and Rex improvises a refusal (wiring point 4)
- dispatch in `_handle_router_takeover_action` (category does NOT auto-wire;
  wiring point 3), plus the legacy_command matcher as yet another parallel path
- reply-window epilogues (`clear_response_wait` vs `begin_response_wait`;
  wiring point 6)

### 1.3 Why this layer is the fragile one (field evidence)

- Every new action needs **six wiring points**, each failing OPEN into
  conversation with no error (memory: new-executable-action-checklist;
  performance.impersonate cost one live debug round per missed point).
- `answer_to_rex` swallowing commands (motion 2026-06-23, impersonation 07-19).
- "I don't remember saying what that was" → routed to memory discard →
  stock shrug (2026-08-01).
- Phrasing outside the regex silently becomes conversation ("come over here
  please" vs cue list). The stoplists and cue lists grow monotonically.
- The LLM fallback router is a JSON-in-prose call: no schema enforcement, no
  arg validation, one more network round-trip, and a separate model contract
  to keep in sync with the catalog.

### 1.4 What already exists that the migration reuses

- `start_shadow_decision()` + `log_decision(mode="shadow")` + config flag
  `ACTION_ROUTER_SHADOW_ENABLED` — shadow infrastructure is ALREADY BUILT.
- `[action_router_audit]` / `[character_loop]` JSON log lines — per-turn record
  of utterance, decision, allowlist result, executed path. This is the
  evaluation dataset: weeks of real labeled traffic in `logs/`.
- `tools/conversation_text_harness.py` + `evals/` — offline replay harness.
- `llm_compat.create(...)` — the single chokepoint for the conversation model
  call, where the `tools=[...]` parameter gets added.
- `ActionSpec` — becomes the single source for generated tool schemas.

## 2. Target design

### 2.1 One call, speech + tools

The main reply call (`llm.stream_response` → `llm_compat.create`) gains
`tools=[...]` generated from `ACTION_SPECS`. Each turn the model returns
prose, tool call(s), or both ("Rolling forward." + `motion_move(dist=0.3)`).

- `ActionSpec` grows a `params` field (JSON-schema fragment per action) and a
  one-line `when` hint; `tool_schemas()` derives the OpenAI tools array.
  ONE place to add an action — wiring points 1–3 collapse into the spec.
- Tool dispatch reuses the existing executors (`_execute_*` /
  `_handle_router_takeover_action` bodies) — the handlers are NOT rewritten,
  only reached through a schema-validated door instead of regex + JSON prose.
- The allowlist, confirmation prompts, charging lockout, no-base denial, and
  room no-drive rules stay exactly where they are (execution layer). Safety
  remains deterministic code; the model only *chooses*, never *bypasses*.

### 2.2 What stays deterministic (deliberately)

- Bare **"stop" / "halt"** during motion & the wake/shutdown words — latency
  and safety-critical; keep the regex fast lane.
- Mid-game answer capture (`game.answer`) while a game owns the turn.
- The dialogue-act reply binding for pure conversation stays, but it no longer
  gets to *veto* tool choice: the model sees the reply frame in context and
  decides — this removes wiring point 5's failure mode.
- Firmware reflexes, flinch, exploration ownership: untouched.

### 2.3 Token budget

40 tool schemas ≈ 2–3k tokens. Mitigations if it matters: collapse per-category
(one `motion` tool with an `op` enum → ~14 tools), and/or gate rarely-relevant
categories on context (games list only when a game could start). Decide from
shadow-phase measurements, not up front.

### 2.4 Local-model fallback

`qwen2.5:1.5b` (Ollama) currently backs some classifiers. Ollama supports tool
calling, but small-model tool accuracy is unproven here — the OFFLINE fallback
path keeps the deterministic classifiers indefinitely. Tool calling rides the
hosted conversation model only (same availability as replies today).

## 3. Migration plan

### Phase 0 — Shadow (no behavior change)
Add tool schemas + a parallel shadow call: for each live turn, log
`tool_choice` next to the shipped router decision (reuse the existing shadow
scaffolding + audit log). Also replay the harness transcripts. Collect ≥1 week
of live traffic.

**Metrics:** agreement rate vs shipped router; disagreements manually labeled
(who was right); per-action precision/recall; arg-extraction accuracy
(degrees/distances/game names); added latency to first token.

### Phase 1 — Cutover, low-risk categories
world, status, humor, character, vision.describe_scene, music.options.
Wrong routing here costs a joke, not a collision. Regex families for these
categories are demoted to fallback-only (used when the model call fails).

### Phase 2 — music, game, performance, memory, identity
The confirmation machinery (`requires_confirmation`) is exercised here.

### Phase 3 — motion (last), system.sleep
Motion keeps its regex fast lane as a *parallel* path during this phase: if
regex fires with ≥0.95 confidence, it executes immediately (today's latency);
otherwise the tool choice governs. Only after shadow data shows the model
matches the fast lane do we consider retiring it — and "stop" never retires.

### Phase 4 — cleanup
Delete demoted regex families + the JSON-prose fallback router + the
`_SYSTEM_PROMPT` rule list; ACTION_SPECS becomes the whole contract. The
six wiring points become: spec (with schema) + executor + allowlist entry.

## 4. Effort & risk

- Phase 0: the real work — schema generation from ActionSpec, the shadow call,
  a divergence report script. Roughly a weekend of sessions.
- Phases 1–3: mostly config flips + deleting demotions, gated on data.
- Biggest risks: (a) tool-call latency on the streaming reply path — measure in
  phase 0; mitigation is the parallel fast lane; (b) the model calling tools
  over-eagerly in banter ("I could roast you" ≠ roast command) — the shadow
  divergence log catches this before it ships; system-prompt guidance + the
  existing governor/confirmation gates bound the blast radius.

## 5. Explicit non-goals

- No new/local multimodal models (16 GB constraint; owner decision 2026-08-01).
- No change to perception, memory guards, speaker ID, TTS, or firmware safety.
- No agentic multi-step planning loop — one turn, one decision, same as today.
