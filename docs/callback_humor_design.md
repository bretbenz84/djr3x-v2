# Callback Humor — Design Proposal

Status: PROPOSAL (no code yet). Schema changes require sign-off before implementation.

## 1. Goal

Rex roasts reactively but never banks a detail and resurfaces it later for comedic
payoff. Target behavior: Bret mentions astrophotography and 3D-printed telescopes
early in a session; twenty minutes later, in a lull, Rex says *"Counting ceiling
panels again — you know, fewer than the stars you pretend to photograph."* That
requires four mechanics, none of which fully exist today:

1. **A durable callback-candidate store** — per-person "fun fact" premises that
   survive sessions, distinct from throwaway lines.
2. **A trigger/relevance layer** — fire only when the live topic connects to a
   banked premise, or in a conversational lull.
3. **Cooldown/freshness** — a used premise is spent for a while; per-session
   no-repeat; decaying reuse.
4. **Tone gating** — hard-subordinate to every existing sincerity/boundary/grief
   gate, with a sensitivity classification that hard-excludes protected material
   from the joke pool while leaving it rememberable for sincere continuity.

## 2. What exists today (survey — verified against code, June 2026)

The repo already has substantial adjacent machinery. The feature must slot into
it, not duplicate it. (Note: CONTEXT.md's "episodic memory is capture-only" is
stale — Phase 2 recall shipped: `memory/episodic_recall.py`,
`EPISODIC_RECALL_*` in config, two production consumers.)

**An existing one-callback-per-reply budget chain** — `llm._build_person_context`
(llm.py:469–547) runs a strictly ordered chain, each entrant gated by
`callback_hook_used`:

1. Unacknowledged emotional events claim the slot first (sincerity supremacy).
2. Stale-fact confirmation hook (`_pick_stale_fact`).
3. Nostalgia hook (`_pick_nostalgia_callback`, p=0.05, close friends only).
4. Profile next-question (`rel_db.get_next_question`).
5. Episodic shared-memory hook (`_pick_episodic_callback`, rex.db, p=0.25).

Plus a prompt rule: *"use at most one remembered fact, callback, inside joke,
stale-fact confirmation, or relationship follow-up in a single reply"*
(llm.py:839–843). **None of these is a humor callback on a banked personal fact**
— confirmation question, warm reminiscence, profile question, experiential
memory. The humor variant is the gap.

**A premise-starved 'callback' comedy mode** — `comedy_modes._MODES["callback"]`
(comedy_modes.py:111) exists but only echoes `_RECENT_PREMISES`, an in-memory
deque (maxlen 12) of *Rex's own recent joke premises*, wiped on restart. It
demotes to `dry_ack` when empty. It never sees user-revealed facts.

**Session-local callback permission** — the conversation arc directive
(topic_thread.py:584–589) already tells the model it "MAY call back to an open
thread when it fits naturally. Never force a callback." Free-text nudge; no
selection, timing, sensitivity, or cooldown.

**Friendship-pattern inside jokes** — `person_facts` rows with
category=`inside_joke` (friendship_patterns.py), injected as "Running/inside
jokes available for rare use" with prose-only restraint rules. Regex-captured,
no sensitivity classification, no usage tracking, no boundary retirement.

**The tone-gating stack the feature must sit below** (verified, in evaluation
order on a reply turn — interaction.py:7992–8015):

- `conversation_agenda.build_turn_plan` — sensitive-disclosure early-return
  whose directive **bans callbacks by name** ("no memory callback",
  conversation_agenda.py:547–560); boundary early-return.
- `social_frame.build_frame` → `_roast_level` (social_frame.py:698): tone
  repair → boundary regex → empathy caring modes → affect/sensitivity →
  persisted boundary rows → preference facts → flat-arc easing.
- `comedy_modes.select_mode`: `_SENSITIVE_PAT` or safety purposes → `straight`
  (`allow_callback=False`).
- Post-generation: `govern_response`/`govern_stream_sentence` delete roast
  sentences on suppressed turns; `_VULNERABLE_TOPIC_JOKE_PAT` strips
  health-adjacent jokes even at roast level "light".
- `empathy` cache (`peek`, TTL 300s; grief flow `force_mode` 600s),
  `classify_local_sensitivity` deterministic same-turn prepass,
  `repair_moves.recent_tone_repair` (180s no-roast), persisted
  `person_conversation_boundaries` (`is_blocked(pid, behavior, topic)`).

**Cooldown precedents**: `PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS=14`
(cross-process spend), `FOLLOWUP_MIN_GAP_EXCHANGES/COOLDOWN_SECS/SUPPRESS_WHEN_FLAT`
(anti-interrogation clamp), `person_facts.last_used_at` + overuse penalty
(30-day linear decay in `score_fact_for_prompt`), per-session in-memory sets.

**Proactive plumbing**: 1 Hz consciousness loop; every proactive line is a
`CandidateMove` arbitrated by the action governor (`ACTION_GOVERNOR_ENFORCE=True`);
`_step_visual_curiosity` (consciousness.py:5346) is the canonical
mid-conversation-lull step (quiet-window detection, cooldown armed at submit,
content spend in `on_spoke` only). New purposes need entries in
`action_governor._PURPOSE_PRIORITIES`, `conversation_agenda._PROACTIVE_RULES`,
and `_GRACE_SUPPRESSED_PROACTIVE_PURPOSES` (registries have drifted; all three
matter).

**Local LLM conventions**: qwen2.5:1.5b via `local_llm.generate` — labelled-line
output only (never JSON), factual judgments only (never affect), validate
against echo/repetition, fail closed, never on the TTFS critical path
(`CONVERSATION_TURN_CLASSIFIER_ENABLED=False` precedent). Background workers use
the arc pattern: coalesced daemon worker + lock + stale-discard cursor.

## 3. Design overview

Five pieces, two new modules:

```
memory/callbacks.py            storage: person_callback_material table (people.db)
intelligence/callback_engine.py the brain: banker, relevance stash, reactive
                                trigger + settle, lull candidate, gates, ledger
```

Data flow:

```
user turn ──► _post_response background thread
                ├─ BANKER (qwen, labelled lines + deterministic sensitivity wall)
                │    └─► person_callback_material upsert
                └─ RELEVANCE WORKER (qwen: live topics × active premises)
                     └─► in-memory stash {premise_id, score, cursor, ts}

next reply turn ──► _stream_llm_response
                      agenda → frame → comedy select_mode
                      └─ REACTIVE TRIGGER (deterministic gates + stash read)
                           ├─ claims the turn (one-callback budget)
                           ├─ comedy mode → 'callback' with banked premise
                           └─ settle: spend only if premise echoed in spoken text

conversation lull ──► consciousness._step_lull_callback (1 Hz step)
                        └─ CandidateMove(purpose="lull_callback", priority 58)
                             speak_fn: compose (OpenAI) → govern → speak
                             on_spoke: persist spend
```

## 4. Storage (the schema decision — needs your sign-off)

### Recommended: new table `person_callback_material` in people.db

rex.db is ruled out: it is Rex's first-person autobiography, is never touched by
forget flows, and isn't re-pointed on `merge_person` — personal roast material
there would violate the consent model (forget/delete must work).

Extending `person_facts` (category=`callback` + new columns) is workable but
fights four verified behaviors: (a) `get_prompt_worthy_facts` filters only
`skin_color`, so callback rows would compete for the 12 dossier fact slots and
get `mark_fact_used` stamped **every turn**, destroying any cooldown built on
`last_used_at`; (b) `memory_query` ("what do you remember about me") and
`person_summary` would recite them unless three exclude lists are extended;
(c) `add_fact` upserts on the global `(person_id, key)` namespace — a colliding
key silently overwrites a real fact; (d) `_normalize_source` coerces unknown
sources to `explicit` (rank 3), corrupting overwrite semantics. A dedicated
table costs explicit wiring but every wire is enumerable and testable.

```sql
CREATE TABLE IF NOT EXISTS person_callback_material (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    premise         TEXT,               -- one-line third-person premise:
                                        -- "does astrophotography and 3D-prints telescopes"
    category        TEXT,               -- passion|hobby|project|quirk|opinion|self_description|running_bit
    topic_slug      TEXT,               -- normalized topic for boundary matching + relevance
    sensitivity     TEXT DEFAULT 'guarded',  -- 'safe' | 'guarded' | 'excluded'  (§5)
    source          TEXT,               -- 'explicit' only in v1 (self-volunteered)
    source_quote    TEXT,               -- the user line it came from (debugging/eval)
    source_fact_id  INTEGER,            -- optional person_facts provenance, nullable
    volunteered_playfully INTEGER DEFAULT 0,  -- stance was playful/engaged at capture
    session_id      TEXT,               -- capture session (same-session lull boost)
    created_at      DATETIME,
    updated_at      DATETIME,
    last_used_at    DATETIME,           -- last time a callback FIRED on this premise
    use_count       INTEGER DEFAULT 0,
    retired_at      DATETIME,           -- boundary/forget retirement (mute, don't delete)
    retired_reason  TEXT,
    UNIQUE(person_id, topic_slug)
);
```

Wiring checklist (all verified as required):

| Touch point | Why |
|---|---|
| `setup_assets.py` DB_SCHEMA | tests build temp DBs from this script |
| `memory/database.py` `_MIGRATIONS` + `_EXPECTED_TABLES` | live-DB migration + `verify_schema` |
| `memory/people.py` `_PERSON_TABLES` | delete/merge/wipe must include the table |
| `memory/forgetting.py` `forget_specific_memory` | "forget that" must retire matching rows (per-table deletes are explicit, not discovered) |
| `memory/boundaries.py` `apply_detected_boundary` | a NEW boundary (any of roast/ask/mention — a "stop asking about X" makes an X joke tone-deaf too) retires matching premises at set time, mirroring event muting; "clear" does NOT auto-unretire (conservative; un-retire is manual/v2). Boundaries from PRIOR sessions don't need a reconcile pass: the selector re-checks `is_blocked(pid,'roast'/'mention',slug)` per premise at fire time, so the hard guarantee is read-side |

Upsert semantics: re-banking an existing `(person_id, topic_slug)` refreshes
`premise/updated_at` but **keeps** `use_count/last_used_at` (re-mentioning a
topic doesn't reset its cooldown) and never resurrects a retired row. Pool
capped at `CALLBACK_BANK_MAX_PER_PERSON` (default 12) active `safe` rows; on
overflow, evict lowest value (unused-longest, lowest score). Roast material
should be a small curated pool, not an archive.

### Alternative (if you prefer zero new tables)

`person_facts` + three new columns (`sensitivity`, `cb_last_used_at`,
`cb_use_count`) + category `callback` with key prefix `callback_<slug>`, plus
exclusion edits in `get_prompt_worthy_facts`, `memory_query._FACT_CATEGORY_EXCLUDES`,
and `person_summary`. More shared-table coupling, fewer moving parts. I
recommend the dedicated table.

## 5. Sensitivity taxonomy and hard-exclusion rules (explicit)

Three buckets. The classifier may only move material **down** (toward excluded),
never up. Uncertainty lands in `guarded`, never `safe`.

**`excluded` — protected. Never joke material, never surfaced by this engine in
any form.** Stored only as an audit/idempotence row (so re-extraction doesn't
re-litigate it); the selector hard-filters `sensitivity='safe'` at read time, so
even a mislabeled write needs two failures to fire. Categories (deterministic,
enforced by regex/structural checks at write time AND read time — the model
cannot override):

1. Health & medical — physical, mental, disability, medication, diagnoses.
2. Grief & loss — deaths, dying relatives, funerals.
3. Body & appearance — weight, build, height, skin, attractiveness, aging.
   (Directly addresses the "heavy-set build / droid-rave bouncer" incident:
   appearance observations are *never* bankable callback material. Live visual
   riffs remain `do_people_roast`'s domain with its own exclusion list.)
4. Sexual orientation, gender identity, romantic/sexual life.
5. Finances — income, debt, job loss, money trouble.
6. Religion and political identity. (Trivial cultural opinions — pineapple
   pizza, tabs-vs-spaces — are `safe`; identity-level convictions are not.)
7. Family conflict & relationship trouble — divorce, custody, estrangement.
8. Addiction/recovery, legal trouble, immigration status.
9. **All third-party material** — anything secondhand (`told_by` ≠ subject) or
   about someone other than the speaker. Structurally excluded: only
   first-party, self-volunteered statements enter the pool. (Kills the
   friend's-sexuality gossip case at the structural level, independent of
   content classification.)
10. Anything under an active `roast`/`mention` boundary (write-time check +
    read-time `is_blocked`).
11. Anything matching session forget-terms (`_extracted_memory_allowed`).

**`guarded` — remembered, never joked.** Personal-but-warm material the
classifier isn't certain is joke-safe: aspirations, family pride, nostalgia,
lightly-shared insecurities. Stored (gives the classifier's hedge a safe landing
spot, protecting the `excluded` wall from erosion), but this engine never fires
on it. Sincere continuity on such material stays with the existing systems
(emotional events, interests, stale-fact confirmation) — this engine is
humor-only by design.

**`safe` — the pool.** Self-volunteered passions, hobbies, projects; quirky
clean habits and admissions; strong opinions on trivial matters;
self-descriptions ("I'm a night owl"); achievements they brag about; bits the
person initiated or visibly enjoyed. `volunteered_playfully` records whether the
capture turn's stance was playful/engaged — premises volunteered flatly default
to `guarded` unless clearly identity-level passion.

Enforcement is defense-in-depth, mirroring repo patterns (prompt + deterministic
backstop):

- **Write time**: protected-category regex bank (built from the existing
  vocabularies: `empathy._LOCAL_*_PAT`, `comedy_modes._SENSITIVE_PAT`,
  `emotional_events._HEAVY_NEGATIVE_CATEGORIES`, plus body/orientation/finance
  terms) force-classifies to `excluded` regardless of model output; structural
  rules (secondhand, boundary, forget-terms) likewise.
- **Read time**: selector filters `sensitivity='safe' AND retired_at IS NULL`,
  then re-checks `boundaries.is_blocked(pid,'roast',slug)` and
  `is_blocked(pid,'mention',slug)` per candidate.
- **Compose time**: the directive/compose prompt states the premise is
  volunteered material and restates the no-protected-categories rule; the
  social-frame governor remains the last net (unchanged).
- **Test time**: a content test asserts no `excluded`-category vocabulary ever
  appears in a `safe` row in fixture flows (precedent: the no-cantina POV-seed
  test).

## 6. The banker (capture)

Runs in `_post_response`'s `_background` daemon thread alongside
`extract_facts/preferences/interests/events` (interaction.py:10209–10415) —
off the speech path, latency-free. Honors the same gates: skipped when
`suppress_memory_learning`, each candidate through `_extracted_memory_allowed`
(forget-terms), transcript window already forget-filtered.

Backend (`CALLBACK_BANK_BACKEND`): default **`local`** (qwen2.5:1.5b) — zero
cost per the project's cost posture; `openai` available as an opt-in spend
upgrade. The local call follows the proven shape: labelled-line output (never
JSON), tiny budget (≈120 tokens, 2.5s timeout, background so latency is free),
anti-echo/repetition validation, fail-closed (no bank on garbage). Finding
candidates is a factual task (allowed for the 1.5B); the sensitivity *wall* is
deterministic (§5), so the model only contributes recall within the allowed
region, never the safety boundary.

Prompt sketch (labelled lines, mirroring `turn_classifier._build_prompt`):

```
User said: "<turn text>"
Did the user reveal a durable, personal, light fact about THEMSELVES —
a passion, hobby, project, quirky habit, strong trivial opinion, or
self-description? Ignore: small talk, moods, plans, anything about another
person, anything about health, body, money, relationships, religion, or grief.
Output EXACTLY these lines, nothing else:
Found: yes | no
Premise: <one third-person line, or ->
Topic: <1-3 word slug, or ->
Category: passion | hobby | project | quirk | opinion | self_description
Playful: yes | no
```

Deterministic post-validation: `Found: yes` requires the premise to share ≥1
content word with the actual utterance (anti-hallucination); protected-regex
scan runs on premise + source quote; stance check
(`topic_thread.snapshot()`) sets `volunteered_playfully`. Only `explicit`
first-party material is ever written.

Test-runner suppression: the banker fires from the turn path's background
thread, so it gets the `_under_test_runner` guard with a `DJR3X_CALLBACK_TEST_OPT_IN`
escape (the arc/rex_pov idiom), plus the kill-switch.

No session-end consolidation pass in v1 (the consolidation prompt explicitly
discards "jokes without durable meaning" — fighting that standing instruction
is not worth it when per-turn capture covers the need).

## 7. Reactive trigger (topical relevance)

**Relevance worker** (background, free): refreshed from `_post_response` after
each turn — one qwen call scoring the live conversation (topic_thread label +
arc Topics/Open-threads + last user lines) against the person's ≤12 active
premises, labelled-line output ("P3: yes"), stashed under a lock as
`{premise_id, score, transcript_cursor, ts}` with stale-discard on session
clear (the arc-worker pattern). Deterministic fallback when Ollama is down:
content-word overlap between live topic keywords and `topic_slug`/premise text
(high precision, low recall). One-turn lag is deliberate — a callback that
lands one exchange into a topic reads as wit; echoing the current sentence
reads as parroting. Nothing runs on the TTFS path.

**Claim seam** — in `_stream_llm_response` between `comedy_modes.select_mode`
and the directive join (interaction.py:8005–8015):

```python
comedy_mode = comedy_modes.select_mode(...)
claim = callback_engine.maybe_claim_reactive(
    person_id, text, frame=frame, comedy_mode=comedy_mode, turn_plan=turn_plan)
if claim:
    comedy_mode = comedy_modes.with_banked_premise(comedy_mode, claim.premise)
```

`with_banked_premise` returns the existing `callback` ComedyMode with the
banked premise rendered into its directive (replacing the `_RECENT_PREMISES`
echo text) — the engine becomes the premise source for the already-existing,
premise-starved mode. The guardrail line in `build_directive` gets a carve-out
**only when a banked premise is supplied**: "…no private-fact jokes (the
supplied callback premise is material they volunteered about themselves — that
one is fair game, kept affectionate)". No standing-instruction conflict on
ordinary turns.

**Gates** (all deterministic, fail-closed, cheapest first), `maybe_claim_reactive`
returns None unless ALL pass:

1. `CALLBACK_HUMOR_ENABLED` and known `person_id`.
2. Ledger: session cap (`CALLBACK_MAX_PER_SESSION`), min-gap
   (`CALLBACK_MIN_GAP_EXCHANGES` transcript lines), wall-clock
   (`CALLBACK_COOLDOWN_SECS`) — one ledger shared with the lull path.
3. `comedy_mode.allow_callback` (inherits `straight` on sensitive turns) and
   `frame.allow_roast == 'normal'` (`'light'` allowed only if
   `CALLBACK_ALLOW_LIGHT_ROAST_FRAME`, with affectionate-only phrasing) and
   `frame.purpose` not in {closure, repair, identity, answer_ack, boundary}.
4. Live-turn safety: `empathy.classify_local_sensitivity(text) is None`;
   `social_frame._looks_like_boundary(text)` False.
5. Empathy state: `empathy.peek(person_id)` fresh-entry mode not in the caring
   set {listen, support, validate, ground, brief, kind_default, child_kind,
   course_correct, crisis, gentle_probe, acknowledge_then_yield}, affect not in
   {sad, withdrawn, angry, anxious}, `topic_sensitivity == 'none'`.
6. `repair_moves.recent_tone_repair()` False.
7. Sober-room rule: no heavy-sensitivity turn or emotional-event capture in the
   last `CALLBACK_SUPPRESS_AFTER_HEAVY_SECS` (default 1800) — a grief
   disclosure keeps the room joke-free for 30 minutes even after empathy's
   5-minute cache expires.
8. No unacknowledged surfaceable emotional event (same check the budget chain
   runs first — refactored into a small shared helper so both call one code
   path).
9. `topic_thread`: `emotional_weight != 'heavy'`, stance not avoidant/terse;
   `arc_reads_flat()` → skip (don't roast a flat room).
10. Crowd ≤ `CALLBACK_MAX_CROWD` (default 2) and the premise's person is the
    engaged speaker (personal material discretion; the existing chain lacks a
    crowd guard — this hook gets one from day one).
11. Per-premise: `sensitivity='safe'`, not retired, off cooldown
    (`last_used_at` + `CALLBACK_REUSE_COOLDOWN_DAYS`), not in the session
    used-set, `boundaries.is_blocked(pid,'roast'|'mention',slug)` both False,
    relevance score ≥ threshold and stash fresh.
12. Person calibration: `friendship_tier` in `CALLBACK_ELIGIBLE_TIERS`
    (default acquaintance+), `callback_style` preference ≠ "prefers callback
    restraint" (hard skip).
13. `random.random() < CALLBACK_FIRE_PROBABILITY` (default 0.6) — never
    metronomic.

**Budget coordination** (prevents double callbacks): the claim is turn-scoped
module state. `_build_person_context` consults
`callback_engine.turn_claim_active()` immediately after the emotional-event
check and sets `callback_hook_used=True`, so stale-fact/nostalgia/next-question/
episodic all skip. Same thread, strict ordering (trigger runs before
`assemble_system_prompt`), no race. Emotional events still outrank everything:
gate 8 means the engine never claims when one is pending.

**Settle (spend-at-speak, not spend-at-injection)**: the existing chain spends
at injection — a known landmine (hooks burn even when the model doesn't voice
them, the governor deletes the line, or TTS is barged). The reply path has no
`on_spoke`, so: after the spoken text is final, `_stream_llm_response` calls
`callback_engine.settle_turn(spoken_text)` — if the premise's content words
appear in what was actually said, persist the spend (`last_used_at=now`,
`use_count+=1`, session used-set, ledger); if not, release the claim (a
two-exchange soft backoff prevents immediate retry). The echo test is
stem-based and deliberately ignores a topic-word-only match: the claim only
existed because the live topic already connected to the premise, so an
ordinary on-topic reply repeats the topic word without the joke — spending
requires premise content beyond the topic (or two matches). Synonym-phrasing
can still under-count usage; the arc's "don't reuse jokes you've already used"
line and the min-gap bound the repetition risk.

## 8. Lull trigger (the marquee case)

New `_step_lull_callback` in consciousness.py, modeled on `_step_visual_curiosity`
(the verified template), dispatched from `_loop` after the emotional-checkin
step:

- Envelope: `profile.conversation_active`; quiet-for within
  [`CALLBACK_LULL_MIN_SILENCE_SECS`=12, `CALLBACK_LULL_ACTIVE_WINDOW_SECS`=60]
  (longer minimum than visual curiosity's 8s — let the lull breathe; the pause
  is part of the joke); ≥2 real user turns this session; engaged person known,
  visible, and crowd ≤ `CALLBACK_MAX_CROWD`; `not profile.suppress_proactive`;
  not waiting for a response.
- All §7 tone gates (4–12) re-checked; premise selection prefers
  `session_id == current` (the "earlier tonight you said…" boost,
  `CALLBACK_LULL_W_SAME_SESSION`) then recency/use-decay score.
- Governor: `CandidateMove(purpose="lull_callback", priority=58)` — above
  `visual_curiosity` (55; a personal callback beats commenting on the room),
  below `celebration_checkin` (64), `memory_followup` (65), and
  `emotional_checkin` (100), so every sincerity flow outranks it. Registered in
  all three registries: `action_governor._PURPOSE_PRIORITIES`,
  `conversation_agenda._PROACTIVE_RULES` (directive text + claim path), and
  `_GRACE_SUPPRESSED_PROACTIVE_PURPOSES`. Not added to
  `_ACTIVE_CONVERSATION_LOW_PRIORITY` (it is *meant* for active-conversation
  lulls). Metadata: `topic_key=f"callback:{pid}:{premise_id}"`,
  `target_person_id`, honest `family_safe`.
- Trigger cooldowns (global `CALLBACK_LULL_COOLDOWN_SECS`=600 + per-person
  900s) armed **at submit** (anti-resubmission; a governor loss spends the
  cooldown — accepted, matching visual curiosity). Content spend happens only
  in `on_spoke`.
- `speak_fn` (off-tick worker): re-check `_can_proactive_speak` + the cheap
  tone gates (state can shift between submit and win), compose via OpenAI with
  a **self-contained** prompt (premise + person name + affectionate-roast
  framing + the protected-category exclusion list, like `do_people_roast`'s
  in-prompt rules — the proactive compose path carries no person dossier, so
  the prompt must carry its own safety), then govern with a synthetic frame
  (`build_frame` + `govern_response`, the interest-idle-followup pattern) and
  speak. `on_spoke` → persist spend + shared ledger.

The qwen relevance judgment is *not* needed on this path (nothing is being
discussed — that's the point); selection is score-ranked from the stored pool.

## 9. Cooldown / freshness model (summary)

| Layer | Mechanism | Default |
|---|---|---|
| Per-premise, cross-session | `last_used_at` + `CALLBACK_REUSE_COOLDOWN_DAYS` | 7 days |
| Per-premise, decaying reuse | score × `0.5 ** (use_count / CALLBACK_USE_DECAY_HALFLIFE_USES)` | halflife 3 uses |
| Per-session no-repeat | in-memory used-set, **cleared in `_end_session`** (the `_interest_idle_followups_spoken` pattern — llm.py's never-cleared "session" sets are a known bug-shape to avoid) | — |
| Session volume | `CALLBACK_MAX_PER_SESSION` across both paths | 2 |
| Pacing | `CALLBACK_MIN_GAP_EXCHANGES` + `CALLBACK_COOLDOWN_SECS`, shared ledger | 8 lines / 240s |
| Lull pacing | global + per-person lull cooldowns, armed at submit | 600s / 900s |
| Pool hygiene | cap per person, evict lowest-value | 12 |

## 10. Config (mirrors the EPISODIC capture/recall flag-pair pattern)

```python
# ── Callback humor: bank durable fun facts per person; resurface them as
# timed callbacks. Capture and firing have SEPARATE kill switches (the
# EPISODIC_MEMORY/EPISODIC_RECALL pattern) so the pool can build silently.
CALLBACK_BANK_ENABLED  = _env_bool("CALLBACK_BANK_ENABLED", True)   # capture
CALLBACK_HUMOR_ENABLED = _env_bool("CALLBACK_HUMOR_ENABLED", True)  # firing (A/B via env)
CALLBACK_BANK_BACKEND = "local"          # "local" (free, default) | "openai" (opt-in spend)
CALLBACK_BANK_MAX_PER_PERSON = 12
CALLBACK_REUSE_COOLDOWN_DAYS = 7
CALLBACK_USE_DECAY_HALFLIFE_USES = 3
CALLBACK_MAX_PER_SESSION = 2
CALLBACK_MIN_GAP_EXCHANGES = 8
CALLBACK_COOLDOWN_SECS = 240.0
CALLBACK_FIRE_PROBABILITY = 0.6
CALLBACK_ALLOW_LIGHT_ROAST_FRAME = False
CALLBACK_ELIGIBLE_TIERS = ("acquaintance", "friend", "close_friend", "best_friend")
CALLBACK_MAX_CROWD = 2
CALLBACK_SUPPRESS_AFTER_HEAVY_SECS = 1800.0
CALLBACK_RELEVANCE_MIN_SCORE = 0.5
CALLBACK_RELEVANCE_MAX_STALE_EXCHANGES = 4
CALLBACK_LULL_ENABLED = _env_bool("CALLBACK_LULL_ENABLED", True)
CALLBACK_LULL_MIN_SILENCE_SECS = 12.0
CALLBACK_LULL_ACTIVE_WINDOW_SECS = 60.0
CALLBACK_LULL_COOLDOWN_SECS = 600.0
CALLBACK_LULL_PERSON_COOLDOWN_SECS = 900.0
CALLBACK_LULL_PRIORITY = 58
CALLBACK_LULL_W_SAME_SESSION = 0.3
```

## 11. Invariants honored (the "must never break" list)

- **One callback per reply** — joins the existing budget chain via the claim
  token; never a parallel slot. The comedy 'callback' mode and arc nudge are
  routed *through*, not stacked on.
- **Sincerity supremacy** — structurally below: agenda sensitive/boundary
  early-returns, comedy `straight`, `_roast_level`, empathy modes + grief
  `force_mode`, unacked-event budget claim, governor priorities, the
  post-generation governor, PLUS the new 30-minute sober-room rule. The
  mother-passed-away instinct is protected by five independent layers before
  this engine can even run.
- **Boundaries are consent** — write-time and read-time `is_blocked`, plus
  boundary→retire hook; "stop bringing that up" retires the premise (today it
  only nudges prose — this feature closes that hole for its own material).
- **Forget flows work** — table in `_PERSON_TABLES` + forgetting.py;
  `suppress_memory_learning` + forget-terms gate the banker.
- **No new critical-path latency** — banker/relevance are background; the
  reactive trigger is DB reads + regex + rolls; lull compose is off-tick.
- **No new cloud spend by default** — qwen for bank/relevance; OpenAI only
  composes lines it already composes (reply / proactive line). `openai`
  banking backend is explicit opt-in.
- **Test hygiene** — kill switches, `_under_test_runner` +
  `DJR3X_CALLBACK_TEST_OPT_IN`, temp-DB via `setup_assets.DB_SCHEMA`, all
  network seams mockable module-level functions.

## 12. Tests & evals

Unit (network-free): banker classification incl. every protected category and
the model-can-only-demote rule; structural exclusions (secondhand, boundary,
forget-terms); upsert/cooldown/decay/cap semantics; reactive gates one by one
(comedy straight, boundary text, empathy modes, sober-room, crowd, tier,
restraint preference); claim/budget handoff (no double callback with
nostalgia/episodic); settle echo-check; lull step envelope + governor
registration + spend-only-on-spoke; boundary→retire; forget→retire;
merge/delete include the table; no-`excluded`-vocab-in-safe-rows content test.

Evals: the harness's `generate_spoken` now includes the callback claim seam so
the eval path stays faithful to the live one (it no-ops unless a scenario seeds
premises + a relevance stash). Dedicated callback scenarios + a judge checker
are deferred: the hard-exclusion properties are binary gates and are pinned
deterministically in unit tests, which is stronger than LLM-judge sampling for
them. When live runs surface tone-quality questions (does the composed line
land affectionately?), add scenarios with a `banked_callback` seeding field and
a lenient-threshold judge class (`--samples 12+`, documented judge noise).

## 13. Implementation order

1. `memory/callbacks.py` + schema wiring + lifecycle tests (after schema
   sign-off).
2. Banker in `_post_response` + sensitivity wall + tests.
3. Reactive path: relevance worker + claim seam + comedy integration + budget
   handoff + settle + tests.
4. Lull path: consciousness step + governor registration + compose/govern +
   tests.
5. Eval-path fidelity (claim seam in `generate_spoken`); CONTEXT.md update
   (also fixing the stale "capture-only" episodic note). Dedicated eval
   scenarios deferred — see §12.

Each lands behind its flags; banking can run solo for a few sessions to build
pools before firing is enabled, or both can go live together.

## 14. Out of scope (v2 candidates)

- Surfacing `guarded` material as warm sincere references from this engine.
- `made_laugh` episodic feedback boosting premise scores (bit-landed evidence).
- Un-retiring premises when a boundary is explicitly cleared.
- Embedding-based relevance (qwen labelled-line is sufficient and free).
- Group-scoped premises ("inside joke with the room").
