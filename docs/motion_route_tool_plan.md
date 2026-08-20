# LLM Motion Routes — Implementation Plan

**Status: PLAN — not yet implemented.** Owner idea 2026-08-19: "rather than all
this fragile regex motion commands, an infinite way of saying things could be
interpreted and the LLM calls the motion... instead of a regex failure where he
says 'I can't parse that route' the LLM could plan out the moves. Or if someone
asked him to spin around, the LLM would command a 360."

This is a handoff document in the style of `exploration_mode_plan.md`: enough
detail for another session/engineer to build it against the existing codebase.
All referenced modules/functions/flags exist today unless marked NEW.

---

## 1. What already exists (read first — more than you'd think)

The 2026-08-13 tool-router migration (docs/tool_router_scope.md, Phase 3)
already put **single-verb motion on the LLM**:

- `motion.turn` / `motion.move` / `motion.arc` / `motion.come` are live tools on
  the reply call (`tool_router._TOOL_DEFS` ~line 236), with schemas whose arg
  names are pinned to the EXACT keys `interaction._handle_router_motion_action`
  reads (`deg`, `dist_m`, `ang_dir`, `lin_dir`, `direction`) — three silent
  arg-name drifts and one enum drift ("backward" vs "back", which would have
  driven him INTO the requester) were found and fixed during that migration.
  Every new schema must repeat that check.
- The regex fast lane **keeps the first claim**: `motion.*` is deliberately
  absent from `TOOL_ROUTER_OWNED_ACTIONS`, so a ≥0.95 `classify_explicit_motion`
  match executes at ~70 ms while the tool governs only what the regex missed
  ("scootch to your right", "hang a left", "get closer" — all real logged
  misses that now route).
- The evidence gate for tool-routed motion is
  `action_router.motion_command_refusal_reason` — a LOOSER positive test than
  the classifier (imperative drive verb aimed at Rex, negative + figurative
  guards). Guards-only was measured and rejected for motion: figurative motion
  ("let's move on", "I need to run to the store") shares its verbs with the
  literal sense. Measured: admits 46/47 real misses, refuses 47/48 decoys.
- `motion.stop` **never migrates** (scope doc §2.2): the deterministic escape
  (`_errand_stop_demanded` + `motion_controller.is_moving()`, watched by the
  eager endpointer) claims it before any LLM sees the turn. `motion.explore`
  also stays deterministic (an accepted invite seizes the floor for minutes).
- Execution-layer safety is shared and untouched by routing:
  `ACTION_ROUTER_EXECUTE_ACTIONS` allowlist, the charging lockout, the no-drive
  room decline, `motion_controller._autonomous_allowed()`, firmware ToF
  reflexes, carpet traction detection. The model chooses; it never bypasses.

**Multi-step routes are the gap.** Today a spoken route is parsed by
`action_router.classify_explicit_motion_sequence` (~line 2074) — regex, TRI-STATE:

- `[]` — not a sequence at all → normal single-command/conversation flow.
- 2+ `ActionDecision`s — parsed route → `motion_sequence.start(...)` executes
  the legs (it already stands realign down, keeps the drive sounds
  user-commanded, announces blocked legs).
- `None` — **route-shaped but unparseable** → Rex says "I couldn't safely parse
  that whole route" and nothing moves. This arm is the owner's exact complaint,
  and it is also this plan's cleanest entry point: the utterance has already
  been positively identified as an attempted route command — the only thing
  missing is an interpreter.

**Transcript trust exists**: `audio/transcription.py` returns
`Transcript(confident=...)` from `avg_logprob`/`no_speech_prob` per backend,
and low-confidence turns already carry a policy ("Rex will answer this but not
learn from it"). Motion gets the same treatment, stricter (§4.3).

---

## 2. Feature summary

Two additions, one new tool:

1. **`motion.route` (NEW tool)** — the model emits an ordered list of bounded
   motion steps; a validating translator turns them into `ActionDecision`s and
   hands them to the existing `motion_sequence` executor. Reached two ways:
   - **The rescue path**: when `classify_explicit_motion_sequence` returns
     `None` (route-shaped, regex couldn't parse it), instead of the canned
     denial, make ONE focused interpreter call — tiny prompt, `tools=[motion.route]`,
     `tool_choice` forced, no persona — and execute what it returns. Denial only
     if the interpreter also declines or its output fails validation.
   - **The organic path**: `motion.route` also joins the reply-call tool surface
     next to the single verbs, so "back up a little and then face the other way"
     said conversationally can route without ever looking like a regex sequence.
2. **Expressive single-verb vocabulary** — extend `motion.turn`'s schema so a
   full spin is expressible (today `direction: around` means 180 and the hint
   says so; "spin around" / "do a 360" has no encoding). Small schema + hint
   change, no new machinery.

Explicitly **v1-geometric**: routes compose `turn`/`move`/`arc` only. Target-
relative motion ("face the window", "go to the couch") needs vision grounding
and is out of scope (§9).

---

## 3. Why the rescue path is a separate call (the impersonation lesson)

Do NOT rely on the persona-loaded reply call for known-route utterances.
Field record, 2026-08-14 (scope doc, Phase 2 carve-out): eight explicit
impersonation requests — four got no tool call at all (the model performed in
prose; *prose wins*), one called the tool with a stale argument, while the
shadow router returned the right answer every time. "Routing was never the hard
part; a persona-loaded reply call at conversational temperature was."

A route command is an unambiguous imperative. The tri-state `None` arm has
already deterministically detected it. So the rescue path uses a **dedicated
interpreter call**: minimal system prompt (units, conventions, clamps — no
character), `tools=[motion.route]` only, tool choice required, temperature low.
This is immune to the prose-wins failure by construction, costs one bounded
round-trip (~0.5–1 s, comparable to the retired JSON router's 0.74 s median but
only on the rare unparsed-route turn, not 30% of traffic), and can speak a
canned instant ack ("Plotting that out...") while it thinks.

The organic path (reply-call tool surface) keeps covering phrasings that never
looked like a route; there the prose-wins risk is acceptable because the regex
never claimed the turn and the alternative today is plain conversation.

---

## 4. Target design

### 4.1 The tool schema (`tool_router._TOOL_DEFS` entry, NEW)

```
"motion.route": (
  "A MULTI-STEP drive request — two or more movements in one command
   ('go forward a bit, then turn around and come back'). Single movements use
   the single tools. Steps run in order; each is closed-loop and obstacle-
   guarded. Never for figures of speech, retold stories, or negated commands.",
  {"steps": {"type": "array", "maxItems": 6, "items": {
      "type": "object",
      "properties": {
        "op":      {"type": "string", "enum": ["turn", "move", "arc"]},
        "deg":     {num, "description": "turn only: degrees, + = left/CCW,
                    90 = quarter turn, 180 = about-face, 360 = full spin"},
        "dist_m":  {num, "description": "move only: metres, + = forward,
                    - = back; a 'bit'/'smidge' ≈ 0.2, a 'step' ≈ 0.3"},
        "ang_dir": {"enum": ["left", "right"], "description": "arc only"},
        "lin_dir": {"enum": ["forward", "back"], "description": "arc only"},
        "pace":    {"enum": ["slow", "normal"], "description": "optional"}},
      "required": ["op"]}}},
  ["steps"])
```

Notes that are load-bearing:
- **Arg keys mirror the single-verb executors exactly** (`deg`, `dist_m`,
  `ang_dir`, `lin_dir`) — the Phase-3 drift class ("backward" vs "back") is the
  named failure mode here; the translator and the schema must be built against
  `interaction._handle_router_motion_action`'s actual reads, and the
  tests must pin every key (the `tests/test_tool_router.py` coverage-enforcement
  pattern already exists for this).
- The single-verb schemas tell the model to OMIT distances so the executor
  re-reads them from the human's words. A route can't do that (which number
  belongs to which leg is the whole problem), so `motion.route` args carry the
  magnitudes — that is why the clamps in §4.2 are not optional.
- `pace` maps to the per-command `speed`/`rate` the wire already accepts
  (`move(dist, speed=)`, `turn(deg, rate=)`, and `come` since the 2026-08-19
  firmware) — "slow" ≈ 0.5× the default, matching the saunter conventions.

### 4.2 The translator (NEW, `action_router.route_tool_to_decisions(args)`)

Pure function: validated tool args → `list[ActionDecision]` or a refusal
reason. Enforced server-side (never trusted from the model):

| Clamp | Default (config, NEW) | Why |
| --- | --- | --- |
| steps per route | `MOTION_ROUTE_MAX_STEPS = 6` | matches `MOTION_SEQUENCE_MAX_STEPS`'s spirit |
| per-step distance | `MOTION_ROUTE_MAX_STEP_M = 1.5` | one leg never crosses a room blind |
| total route translation | `MOTION_ROUTE_MAX_TOTAL_M = 3.0` | same ballpark as the explore tether |
| per-step turn | ±360° | full spin is the ceiling, firmware clamps again |
| total rotation | `MOTION_ROUTE_MAX_TOTAL_DEG = 720.0` | no pirouette marathons |

Anything out of range → the whole route is refused (tri-state discipline:
"turn left then sing" must not half-execute; same rule here — a route with one
bad step is a refused route, not a truncated one).

Execution reuses `motion_sequence.start(...)` unchanged — it already owns
sequencing, per-leg `wait_done`, blocked-leg announcement, realign stand-down,
and commanded-FX volume (`note_user_motion` → full-volume whir, overlay mode).

### 4.3 Gates (in order, all existing unless marked NEW)

1. **Transcript trust (NEW policy)**: a tool-initiated route executes only when
   the turn's `Transcript.confident` is True. Low-confidence transcripts already
   don't write memory; a *fabricated* drive is strictly worse than a fabricated
   fact (ASR fabrications are a known, memory-documented failure class), and the
   regex lane's rigidity was an accidental guard the LLM lane gives up. The
   regex fast lane keeps its current behavior (unchanged bar).
2. **Evidence gate**: `motion_command_refusal_reason` on the raw utterance —
   already applied to tool-routed motion via
   `interaction._router_execution_block_reason` (~line 14073); routes go through
   the same door. The rescue path may skip it: the tri-state `None` arm already
   ran the sequence classifier's own negation/figurative checks.
3. **Execution layer** (unchanged): `ACTION_ROUTER_EXECUTE_ACTIONS` allowlist
   (add `motion.route`), charging lockout, no-drive room decline with the
   voice-liftable line, `_autonomous_allowed()`, firmware ToF zones, traction.
4. **Full-spin lateral check (NEW, recommended)**: the drive axle sits aft of
   the body-ring centre — the front sweeps a wide arc in place (this is why
   exploration bans 360° survey spins). Before any single step with |deg| ≥
   ~270°, check side clearance from the radial ring (reuse
   `motion_agency._wander_clearances()`; require both sides ≥ the idle-wander
   side-clear floor) and refuse in character when cramped ("Not enough elbow
   room for a full spin in here.").

### 4.4 What stays deterministic (unchanged, deliberately)

- **"stop"/"halt"** — never a tool, never waits on a model (scope doc §2.2).
- The regex single-command fast lane and the PARSEABLE-route lane — first
  claim, ~70 ms, zero model risk. This plan only rescues what they decline.
- The eager motion endpointer (`_eager_motion_transcript_matches`) — probes
  stay regex-only; an LLM call inside the endpointing hot path is a non-starter.
- `motion.explore`, wake/shutdown words, mid-game answer capture.
- **Offline mode**: the local reply model gets no tool surface today and small-
  model tool accuracy is unproven (scope doc §2.4) — offline, the `None` arm
  keeps the current spoken denial. The denial line only dies online.

---

## 5. Wiring points (the six, enumerated — each fails OPEN if missed)

Per the new-executable-action checklist:

1. `ActionSpec(key="motion.route", category="motion", executable=True, ...)` in
   `ACTION_SPECS`.
2. Tool schema in `tool_router._TOOL_DEFS` + membership in the live-action set
   (`TOOL_ROUTER_LIVE_ACTIONS`), shadow-first (§7).
3. Dispatch: a `motion.route` branch in the tool-call handling beside
   `_handle_router_motion_action`, calling the translator then
   `motion_sequence.start`.
4. `config.ACTION_ROUTER_EXECUTE_ACTIONS` allowlist entry (`"motion.route"`),
   plus the NEW config cluster (`MOTION_ROUTE_*`, `MOTION_ROUTE_ENABLED` master).
5. Dialogue-act interaction: the rescue path runs where the sequence takeover
   already runs (pre-dialogue-gate), so `answer_to_rex` cannot swallow it; the
   organic path rides the reply call, which sees the reply frame anyway.
6. Reply-window epilogue: route execution follows the motion-sequence pattern
   (spoken ack from the sequence machinery, `clear_response_wait`), not a
   fresh reply.

---

## 6. Latency & cost budget

- Organic path: zero added calls (rides the reply call; ~15 extra schema lines
  ≈ trivially small against the existing 40-tool surface).
- Rescue path: one focused call on the rare `None`-arm turn only. Target
  ≤ 1.2 s to first wheel motion; a canned instant ack line covers it (the
  exploration ack pattern — pre-cached TTS, no LLM latency).
- No new models (16 GB constraint, standing owner decision). The interpreter
  uses the existing hosted conversation model via `llm_compat.create`.

---

## 7. Phased plan

### Phase 0 — Shadow (no behavior change)
- Add the schema + translator. On every `None`-arm utterance, run the
  interpreter call in the background, LOG the route it would have driven
  (`[motion_route_shadow]`), and still speak today's denial. On reply-call
  turns, `motion.route` joins the shadow tool set only.
- Replay the harness: the audited logs contain every historical "couldn't
  safely parse that whole route" turn plus the figurative-motion decoy corpus
  from the Phase-3 gate work (47 decoys) — run both through the interpreter.
- **Metrics**: interpreter parse rate on real `None`-arm turns; decoy false-fire
  rate (must be ~0 — these turns never asked for motion); arg sanity (units,
  signs, step counts vs. the human's words, hand-labeled); added latency.

### Phase 1 — Rescue path live
`MOTION_ROUTE_ENABLED = True` flips the `None` arm from denial to interpreter.
Blast radius: turns that today produce a refusal and no motion. Wrong-but-safe
by construction (clamps + firmware). Keep the shadow log line for comparison.

### Phase 2 — Organic path live
Add `motion.route` to `TOOL_ROUTER_LIVE_ACTIONS` so the reply call can choose
it. Watch the same decoy metric on live traffic — this is where over-eager
banter routing would appear ("we should do a lap sometime" must stay prose).

### Phase 3 — Expressive turn vocabulary
Extend `motion.turn`'s hint/enum for full spins (+ the §4.3.4 lateral check).
Can ship with Phase 1; listed separately because it touches a live schema.

---

## 8. Test plan (per-module, never full discover)

`tests/test_motion_route_tool.py` (NEW) + additions to `tests/test_tool_router.py`:

1. **Translator**: every schema key maps to the executor's exact read; clamps
   (per-step, totals, step count); whole-route refusal on one bad step; pace
   mapping; empty/garbage args.
2. **Tri-state integration**: `[]` and 2+-decision arms unchanged; `None` arm
   calls the interpreter (mocked) online and speaks the denial offline/failure.
3. **Gates**: low-confidence transcript blocks the tool route but not the regex
   lane; no-drive room, charging, allowlist, no-base each refuse with the
   existing lines; full-spin refused when side clearance is tight (mock ToF).
4. **Prose-wins regression**: interpreter call asserts `tool_choice` forced and
   no persona prompt content.
5. **Decoy corpus**: the 47 figurative-motion decoys through the whole path —
   zero executions.
6. Schema-coverage enforcement in `test_tool_router.py` picks up the new tool
   automatically — verify args round-trip through the shared executor.

---

## 9. Non-goals (v1)

- **Target-relative motion** ("face the window", "go to the couch", "get out of
  the way"): needs a bearing source (vision/room-model grounding). The tool
  schema deliberately has no `target` field so the model can't pretend
  otherwise. Natural v2 once the room model can answer "bearing of X".
- No SLAM, no waypoint memory, no agentic replanning loop — one utterance, one
  bounded route, exactly like the regex sequence lane today.
- No change to `motion.stop`, exploration, wake words, or any firmware safety.
- No new/local models; offline keeps the deterministic classifiers.

---

## 10. Decisions made (don't relitigate)

- Regex fast lanes are KEPT, not replaced — they're the latency floor and the
  offline path. This plan is additive tier-2 (the Phase-3 precedent).
- Rescue path is a dedicated forced-tool call, not the persona reply call —
  the 2026-08-14 impersonation record is the controlling evidence.
- Route args carry magnitudes; safety comes from server-side clamps, not model
  restraint. A route with one invalid step is refused whole.
- Transcript-confidence gate applies to model-initiated motion only; the
  regex lane's bar is unchanged.
- Shadow-first with the decoy corpus before anything drives.

## 11. Open questions (fine to resolve during implementation)

- Should the rescue interpreter see one prior turn of context ("do that again
  but further")? Recommend NO for v1 — stale-context args are exactly what
  burned impersonation (`target='speaker'`).
- Whether `motion.come` may appear as a route step ("come here then turn
  around"). Recommend NO for v1: come seizes the requester-errand machinery;
  a route containing it should refuse and let the errand own the turn.
- Exact `pace` → speed/rate mapping constants (suggest 0.5× / 1.0× of the
  defaults, reusing the saunter conventions).
- Whether the rescue ack line is worth its own canned-lines config
  (`MOTION_ROUTE_ACK_LINES`) or reuses the sequence machinery's existing ack.
