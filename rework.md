# DJ-R3X Conversational Engine — Rework Notes & Roadmap

> **Purpose:** a self-contained handoff for improving DJ-R3X's *conversation engine*.
> Written 2026-06-02 after several rounds of fixes (see the "do not regress" entries
> at the bottom of `CONTEXT.md`, and `tests/test_conversation_revamp.py`). If you're a
> fresh context window: read this first, then `CONTEXT.md`. It captures the architecture
> map, the core diagnosis, and a prioritized improvement roadmap with the exact code
> areas each idea touches.

## Conventions (do not relearn the hard way)
- **Work on `main` only. No PRs / feature branches.** Commit + push to `main` is how code
  reaches the physical robot for testing.
- Tests: `source venv/bin/activate && venv/bin/python -m unittest discover -s tests`
  (no pytest). ~749 tests today; keep them green. Text-only manual test:
  `venv/bin/python main.py --gui --noaudio`.
- The repo has many interacting behavior layers and a strong "do not regress" culture
  (every behavior change gets a `CONTEXT.md` entry). Keep changes scoped; add a CONTEXT.md
  note for anything non-obvious.
- The local `assets/memory/people.db` is disposable dev/test data — reset freely.

---

## What's already been done (so you don't re-litigate)
Three rounds of conversation fixes landed (details in `CONTEXT.md`):
- **Round 1 (curious conversationalist):** interest "topic seeds" (one-word answers get
  room + curiosity), question-budget split (earned on-thread follow-ups bypass the budget),
  killed the dead-end "Fair enough." fallback, engage-first on sincere shares, narrowed the
  wrong-person repair, gated vague-memory cold opens, a per-interest follow-up *angle*
  rotation, topic-label hardening, comedy/frame de-contradiction.
- **Round 2 (turn-taking + routing + dialogue):** hold unfinished thoughts
  (`turn_completion`), stop the proactive pile-on (no-response quip + idle-banter
  coordination, floor-hold after a question), tightened the time-query regex, "blocked
  conversational turn stays conversational", pacing-complaint repair, ROAST-LEAN (not every
  turn), banned-opener stripping + opener-variety.
- **Round 3:** don't repeat Rex's own question (dedup vs last spoken line), filter Whisper
  subtitle/credit hallucinations (also un-blocked departure detection), drop incomplete
  streamed sentence tails, joke double-log fix (`log_text=False` for performance pieces),
  stale-steering fixes ("I'm building X" detection, shorter TTL), don't needle reassurances.
- **Latest feature — subject pivot:** when a topic stops landing (2 bare low-content
  replies in a row), steering flips `deepen`→`pivot` and Rex changes the channel.

These were all **incremental patches on a stateless, regex-governed engine.** The roadmap
below is about the next gear.

---

## Architecture map (where everything lives)

**Per-turn pipeline** — `intelligence/interaction.py` (~16.5k lines, the spine):
- Main turn loop ≈ lines 14600–15750 (speech → resolve speaker → `dialogue_act` gate →
  `action_router`/`intent_classifier`/`command_parser` → LLM stream → speech queue →
  memory extraction).
- Streaming: `_stream_and_speak_sentences`, `_split_stream_sentences`, `_tail_is_speakable`.
- Speech: `_speak_blocking` (has `log_text`), `_speak_proactive`, `_register_rex_utterance`
  (central "Rex said a line" chokepoint — notifies repair/end_thread/topic_thread/comedy/
  consciousness), `_proactive_line_recently_fired`, `_floor_held_until`.
- Proactive/no-response: `_maybe_idle_banter`, `_arm_no_response_recovery`,
  `_question_expects_response`.
- Answer capture: `_maybe_capture_pending_qa`, `_maybe_capture_topic_thread_answer`.
- Turn-policy gate: `_intent_execution_block_reason`, `_legacy_command_execution_block_reason`.
- Local takeovers (e.g. joke): `_handle_fast_local_takeover`, `_handle_router_performance_action`.
- Session reset: `_begin_user_turn` (resets idle/proactive/floor globals).

**The governor stack** (each appends to one big per-turn prompt — THIS is the tangle):
- `intelligence/conversation_agenda.py` — `build_turn_directive()` assembles the giant
  directive. Branch order: sensitive → offscreen-correction → health-resolved → reassurance
  → end-thread → unknown-group → **steering** → answered_question → user-question →
  person_id block (plan / `_next_useful_question` / compliment-ack / generic-react).
  `_PROACTIVE_RULES`, `claim_proactive_purpose`.
- `intelligence/social_frame.py` — the FINAL governor. `build_frame()` → `SocialFrame`
  (purpose, max_words, allow_question, allow_roast, allow_visual). `govern_response()`
  (strip disallowed questions/visual/roast, `_salvage_pure_question`, `_is_near_repeat`
  dedup, `_fallback`). `build_directive()` renders the "Final response shape contract".
  **`_purpose_from()` and `_explicit_followup_allowed()`/`_EXPLICIT_FOLLOWUP_PAT`/
  `_ASK_ALLOWED_PAT` regex-parse the agenda's own string output — this is the fragility.**
  `_roast_level()`.
- `intelligence/conversation_steering.py` — interest deepening + pivot. `note_user_turn`,
  `build_context` → `SteeringContext(mode=deepen|pivot)`, `_directive_for`,
  `_pivot_directive_for`, `detect_interest`/`_INTEREST_PATTERNS`, `_FOLLOWUP_ANGLES`,
  `_looks_disengaged`/`_GENERIC_REPLY_RE`, `_active` dict, `_TTL_SECS`, `_PIVOT_AFTER_MISSES`.
- `intelligence/topic_thread.py` — keyword topic tracking. **Produces garbage labels
  ("things / are", "watching / apple") via `_keywords`/`_classify_topic`/`_STOPWORDS`.**
  `TopicThread` dataclass, `note_user_turn`, `build_directive`.
- `intelligence/user_energy.py` — quiet/banter/depth/engagement classification (`_classify`).
- `intelligence/question_budget.py` — question rationing (`can_ask`, `build_directive`;
  config `QUESTION_BUDGET_*`).
- `intelligence/response_length.py` — per-turn length budget (`classify`); token budget map
  `_RESPONSE_LENGTH_TOKEN_BUDGET` lives in `llm.py` (`_max_tokens_for_agenda`).
- `intelligence/comedy_modes.py` — comedy stance + anti-repeat. `select_mode`,
  `build_directive`, `polish_response`/`polish_stream_sentence`, `strip_banned_opener`,
  `note_spoken_line`/`last_spoken_line` (full last line), `_RECENT_MODES/PREMISES/OPENERS`.
- `intelligence/empathy.py` — user affect classification (per-turn).
- `intelligence/end_thread.py` — closure grace. `intelligence/repair_moves.py` — corrections
  (`detect`, `_WRONG_PERSON_PAT`, `_INTERRUPT_PAT`, `_BARE_CONTENT_DENIAL_PAT`).
- `intelligence/dialogue_act.py` — cheap gate (answer_to_rex vs new_command vs general_chat).

**LLM / prompt:**
- `intelligence/llm.py` (~1.6k) — `assemble_system_prompt`, `stream_response`,
  `_max_tokens_for_agenda`, `extract_facts`/sentiment/etc. (the per-turn OpenAI calls).
- `config.py` (huge) — `REX_CORE_PROMPT`, `QUESTION_POOL`, `TIER_MAX_DEPTH`, all knobs.
- `intelligence/local_llm.py` — **Ollama sidecar `qwen2.5:1.5b`, currently barely used.**
- `intelligence/rex_preferences.py` — Rex's *static* tastes.
- `intelligence/friendship_patterns.py` — running jokes/bits (underused → callbacks).
- `intelligence/profile_questions.py` — `QUESTION_POOL` selection.

**Proactive / presence:** `intelligence/consciousness.py` (~9k) — `_step_presence_tracking`,
greeting priority chain, `_pick_due_celebration_checkin`/`_celebration_worth_leading_with`,
departure detection (`PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS`).

**Memory (underused for tone/arc):** `memory/people.py` (warmth/antagonism/trust scores,
familiarity, greeting counts), `facts.py`, `interests.py`, `emotional_events.py`,
`social.py` (relationship edges), `database.py`.

**Audio / turn-taking:** `audio/transcription.py` (`_is_hallucination`,
`_SUBTITLE_HALLUCINATION_RE`), `audio/speech_queue.py` (`enqueue(log_text=...)`),
`audio/vad.py`, `audio/barge_guard.py`, `audio/echo_cancel.py`, `audio/tts.py`.

**Tests:** `tests/test_conversation_revamp.py` (all recent fixes — eval seed),
`tests/fixtures/misroute_replays.json` (replay corpus seed),
`tests/test_audio_and_conversation_gating.py` (big conversation suite).

---

## Core diagnosis (the two structural problems)

Almost every bug we patched is a symptom of one of these:

1. **Governors communicate via fragile prose-matching.** The agenda emits a paragraph;
   `social_frame._purpose_from` / `_EXPLICIT_FOLLOWUP_PAT` then *regex-parse that paragraph*
   to re-derive what the agenda already decided (e.g. whether a question is allowed). Every
   fix has to phrase directives "just so" to match the next layer's regex. Brittle; changes
   break neighbors.
2. **No memory of the conversation's arc.** Each turn is re-derived from scratch.
   `topic_thread` is keyword-garbage. So continuity, callbacks, not-repeating-yourself, and
   not-re-roasting-the-same-target all have to be *faked deterministically* (the
   `_is_near_repeat` dedup, `strip_banned_opener`, the angle rotation) instead of emerging.

The engine has been pushed impressively far by careful incremental patching, but it's hit
the ceiling of "stateless turn + pile of regex governors." The next gear is **memory of the
conversation + a structured spine**, leaning on the LLM for fuzzy judgments.

---

## Roadmap (prioritized, with code areas)

> **WHERE THINGS STAND (2026-06-03) — see "Status & where to resume" at the bottom for the resume plan.**
> Done & tested (full suite green): **Bet 1** (arc-memory, full), **Bet 2** (TurnPlan), **relationship-tone**,
> **offline eval harness**, **cold-open ranker** — all **COMMITTED on `main`** (commits `d6b3afe`→`e3aa5d4`).
> **Bet 3** (turn classifier) was built, validated, and **SHELVED** (slower AND worse than the deterministic
> heuristics + ~1s on-path latency). NEW this session: **Rex persistent POV** (the north-star smaller win)
> — landed + tested, **UNCOMMITTED** until you commit+push (commit+push to `main` is how code reaches the
> robot). Each landed item carries an inline **STATUS** note below; deferred/next items + reminders are in
> **Status & where to resume**.

### ★ Bet 1 — Arc-memory (running summary + callbacks)  [highest felt impact]
> **STATUS — first cut landed + live-tuned (2026-06-03).** Folded into
> `intelligence/topic_thread.py` (not a parallel module): a running summary (topics /
> shared facts / mood / landed-vs-flopped / open threads) maintained by a coalesced
> background worker off the speech path, injected via `llm.assemble_system_prompt`
> (§6b, after "Session so far"). **Backend is configurable** (`config.CONVERSATION_ARC_BACKEND`):
> `gpt-4o-mini` by default (rich schema; the local `qwen2.5:1.5b` froze/looped and
> couldn't judge affect — validated both live), or `"local"` for the 3-field sidecar
> version. Off-path + cloud-already-required → no latency/dependency cost. Purely
> additive — no anti-repetition hacks deleted yet. Tests: `tests/test_conversation_arc.py`
> (note the test-runner safety gate — the suite has a live OpenAI key). See the
> "Conversation arc memory (Bet 1)" do-not-regress entry in `CONTEXT.md` for the full
> design, the tuning lessons, and knobs. **Still to do:** more live use on the robot,
> then the fast-follows — cross-session persistence (`memory/conversations.py`),
> `friendship_patterns` callback selector, and *then* start deleting the deterministic
> anti-repetition once the arc proves out.

A short **running conversation summary**, updated each turn (cheap `local_llm` call),
holding: topics covered, what **landed vs flopped**, the person's mood, open threads. Feed
it back into the prompt. Then: repetition dissolves (he can see he already asked/roasted
that), and **callbacks become possible** ("earlier you said the droid's eyes weren't
working — fixed?") — the single biggest "feels alive" lever. Persist per-person so it spans
sessions ("last time you were chasing the Whirlpool galaxy…").
- New module e.g. `intelligence/conversation_arc.py` (or fold into / replace
  `topic_thread.py`). Update from `interaction._register_rex_utterance` + the user-turn path.
- Inject via `llm.assemble_system_prompt` (or as an agenda line in
  `conversation_agenda.build_turn_directive`).
- Summarize with `intelligence/local_llm.py`. Persist via a new `memory/` table (or reuse
  `facts`/`emotional_events`).
- Callbacks tie into `intelligence/friendship_patterns.py`.
- Lets you delete much of the deterministic anti-repetition (comedy openers, angle rotation,
  `_is_near_repeat`) once the model can just *see* what it already said.

### ★ Bet 2 — Structured `TurnPlan` (kill the prose-matching)  [code-health multiplier]
> **STATUS — landed (2026-06-03), patterns kept as fallback.** `intelligence/turn_plan.py`
> `TurnPlan` carries the agenda's decisions (purpose + the allow_question signals);
> `conversation_agenda.build_turn_plan()` populates them and renders the directive
> (`build_turn_directive` is now a back-compat wrapper); `social_frame.build_frame(turn_plan=…)`
> reads the structured fields, so the LIVE agenda→social_frame handoff no longer
> regex-reparses prose. Equivalence guaranteed by construction + `tests/test_turn_plan.py`.
> The regex patterns (`_purpose_from`/`_EXPLICIT_FOLLOWUP_PAT`/`_ASK_ALLOWED_PAT`/
> `_HARD_NO_QUESTION_PAT`) are KEPT as the no-plan fallback; literally deleting them is a
> deferred follow-up (rewrite ~5 pinned string-based tests onto the plan API, then delete).
> See the "TurnPlan" do-not-regress entry in `CONTEXT.md`.

Replace string-append + regex-reparse with a typed object. Each governor populates fields
(`purpose`, `allow_question`, `roast_level`, `topic`, `mode`, `length`, `addressee`…); render
the prompt **once** from it.
- New `intelligence/turn_plan.py` dataclass.
- Refactor `conversation_agenda.build_turn_directive` to populate a `TurnPlan` (not just a
  string). Refactor `social_frame.build_frame`/`build_directive` to READ `TurnPlan` fields
  instead of `_purpose_from`/`_explicit_followup_allowed`/`_EXPLICIT_FOLLOWUP_PAT`/
  `_ASK_ALLOWED_PAT` (those become deletable). Medium effort, not a rewrite. Do this *while*
  wiring Bets 1 & 3 since you're touching the handoffs anyway.

### ★ Bet 3 — Local-LLM turn classifier (retire the regex zoo)  [enables 1 & 2]
> **STATUS — SHELVED (2026-06-03); do NOT rebuild on-path.** Built `intelligence/turn_classifier.py`
> + `tests/test_turn_classifier.py` and validated against the real `qwen2.5:1.5b`: it is **slower AND
> worse** than the existing deterministic heuristics on the fields that matter — `engagement` defaulted
> to "engaged" even for flat replies ("pretty much", "yeah whatever"), and it MISSED an explicit pivot
> request — while adding ~1s of ON-PATH latency per turn (it must finish before the reply is built, so
> the latency can't be hidden). Any on-path classifier, even the fastest cloud model (GPT-5.4-nano,
> ~0.57s TTFT → ~1s full), hurts time-to-first-speech, which the user prioritizes. The deterministic
> heuristics (`conversation_steering._looks_disengaged`, `user_energy._classify`) already read
> engagement/pivot well and instantly, and the **arc already supplies a clean topic**. The module is left
> INERT behind `config.CONVERSATION_TURN_CLASSIFIER_ENABLED` (default False) for a possible future
> OFF-path use (background/lagged like the arc). See its do-not-regress entry in `CONTEXT.md`.

One small structured `qwen2.5:1.5b` call per turn returning
`{topic, engagement, intent, sentiment, wants_pivot, addressee}`.
- New `intelligence/turn_classifier.py` on `intelligence/local_llm.py`.
- Replaces/augments: `topic_thread._classify_topic`/`_keywords`, most of
  `user_energy._classify`, `conversation_steering._looks_disengaged` + parts of
  `detect_interest`, `conversation_agenda._looks_like_reassurance`, some of `dialogue_act`,
  and feeds a clean LLM-written **topic label** to the arc-memory.
- Possibly net-faster: the logs show **5+ OpenAI round-trips per turn** (empathy, agenda,
  sentiment, extraction); several could collapse into one local call. Watch latency
  (`[latency]`/`[ttfs]` telemetry).

### Smaller, high-ROI
- **Rex leads with substance / persistent POV.** ✅ **DONE — first cut (2026-06-03):**
  new `intelligence/rex_pov.py` gives Rex ONE persistent *current preoccupation* (curated
  `config.REX_POV_SEEDS`, **hybrid** context-biased selection, held on a transcript clock so it
  CARRIES across turns, rotates on a material context change or max-hold). Surfaced via
  `llm.assemble_system_prompt` §6c (which — through `get_response` — covers normal replies AND every
  proactive/idle line) and the `interaction._maybe_idle_banter` "volunteer" attempt. Deterministic
  (no LLM call → no latency, per the shelved-classifier lesson), session-only, kill-switch
  `config.REX_POV_ENABLED`; seeds are venue-neutral (no "cantina", test-enforced). Tests
  `tests/test_rex_pov.py`; do-not-regress entry in `CONTEXT.md`. **Fast-follows:** cross-session
  persistence, feed `_do_private_thought`/`_do_aspiration` from the POV, an LLM-evolved POV.
  **Needs a live `--gui` pass to judge the felt change (the qualitative payoff).**
  *Original idea:* loop was react→roast→question; give Rex a small evolving POV so he volunteers real
  in-character content instead of just interviewing — the cheapest way to make him less exhausting.
- **Tone tracks the relationship, not per-turn dials.** ✅ **DONE (2026-06-03):**
  `llm._relationship_tone_rule` maps `warmth_score`/`antagonism_score`/`trust_score` into a
  persistent tone line in `assemble_system_prompt` (affectionate with warm friends, sharper
  with people who needle Rex, neutral otherwise). Additive + tone-only (the `_roast_level`
  care/affect gates are untouched — that's where "whether to roast" lives; this colors "how").
  Gated by `config.RELATIONSHIP_TONE_ENABLED`. See the do-not-regress entry in `CONTEXT.md`;
  tests `tests/test_relationship_tone.py`.
- **Better "what's worth bringing up" selector.** ✅ **DONE (2026-06-03):**
  `consciousness._pick_due_celebration_checkin` now ranks the gate-passing celebration
  candidates via `_celebration_lead_score` = invited (dominant) × recency × concreteness
  and leads with the BEST one (was: first-that-passes). Gate unchanged; kill switch
  `config.PRESENCE_CELEBRATION_RANK_ENABLED`. Tests `tests/test_celebration_ranker.py`;
  do-not-regress entry in `CONTEXT.md`. **Follow-up:** extend the same score across
  `facts.py` / `interests.py` (currently ranks emotional-event celebrations only).
- **Offline replay/eval harness.** ✅ **DONE (2026-06-03):** `tests/test_conversation_replay.py`
  + `tests/fixtures/conversation_replays.json` — replays scenarios through the DETERMINISTIC stack
  (`build_turn_plan → build_frame → govern_response`) and asserts STRUCTURAL properties (purpose,
  allow_question, governed includes/excludes, ≤N questions, governance notes). **No LLM, suite-safe,
  data-driven** (add a JSON scenario, no Python). Complements the existing ROUTING replay
  (`tests/fixtures/misroute_replays.json` via `test_dialogue_act.py`). do-not-regress entry in
  `CONTEXT.md`. **Follow-ups:** grow the corpus from real conversation logs; it's also the lever to
  de-risk the deferred TurnPlan pattern-deletion (assert structural outcomes unchanged across it).
- **Real barge-in via text-echo rejection** (assessed earlier, still the best software path
  to "he heard me while talking"): transcribe during playback on the un-attenuated rolling
  buffer, fuzzy-diff the transcript against the known in-flight TTS text (`audio/tts.py` /
  speech_queue knows what's playing), strip Rex's words, keep the residual = the user. Robust
  to clock drift/reverb (the things that killed the acoustic AEC in `audio/aec.py`, now
  disabled). Bias HARD toward keeping (asymmetric thresholds + speaker-ID on the residual) so
  a user echoing Rex isn't falsely dropped. Touch: `audio/barge_guard.py`,
  `audio/transcription.py`, the wake/VAD path. NOT a full fix for deep talk-over (acoustic
  masking → needs the ReSpeaker hardware-AEC firmware path noted in CONTEXT.md).

### Status & where to resume (as of 2026-06-03)
The original "arc → classifier → TurnPlan" ordering is now mostly executed (and one bet was
shelved). Concrete state:

**Landed & COMMITTED on `main`** (prior session; full suite green; commits `d6b3afe`→`e3aa5d4`):
**Bet 1** arc-memory (gpt-4o-mini running summary + cross-session persistence + act-on-signal) ·
**Bet 2** TurnPlan (live agenda→social_frame handoff de-brittled; regex kept as fallback) ·
**relationship-tone** · **offline eval harness** · **cold-open ranker**. **Bet 3** turn classifier =
**SHELVED** (see its STATUS). Each has an inline STATUS note + a do-not-regress entry in `CONTEXT.md`.

**This session (NEW — landed + tested, UNCOMMITTED; commit+push to `main` to reach the robot):**
**Rex persistent POV** (`intelligence/rex_pov.py`) — the north-star smaller win; first cut (see its
STATUS above + the `CONTEXT.md` do-not-regress entry). Still needs a live `--gui` pass to judge the
felt change.

**Deferred / good resume points** (roughly in value order):
1. **Delete the TurnPlan regex patterns** (Bet 2 follow-up) — needs live testing; de-riskable via the
   eval harness (expand its corpus over the affected branches, then assert outcomes unchanged). It
   requires removing the no-plan fallback + rewriting ~5 pinned string-based `social_frame` tests onto
   the plan API. See the TurnPlan do-not-regress entry.
2. **Extend the cold-open ranker across `facts`/`interests`** — currently ranks emotional-event
   celebrations only (small, unit-testable).
3. **Grow the eval corpus from real conversation logs** — turn recent transcripts into pinned scenarios.
4. **Rex POV fast-follows** — cross-session persistence (resume/evolve a preoccupation across visits),
   feed `_do_private_thought`/`_do_aspiration` from the POV, and (only if it proves worth the cost) an
   LLM-evolved POV. See the Rex persistent POV do-not-regress entry.

**Reminders for the next window:**
- `config.LOG_SYSTEM_PROMPT` is left **`True`** (verbose full-prompt logging) — flip it `False` once done
  inspecting prompts.
- Consider migrating off **`gpt-4o-mini`** → GPT-5.4 mini/nano before it sunsets (affects the main reply
  AND the arc backend; both via `config.*_MODEL`).
- Speaker-ID flicker / weak voiceprints in a NOISY room (seen live 2026-06-03: every voice scan landed
  0.35–0.69, all below the 0.75 threshold) is a known **hardware-AEC** limitation, NOT a code bug — don't
  chase it in software. The introduction flow + the arc both handled that noisy session correctly.

---

## The user's north star (design intent)
Keep the roast/snark personality, but Rex must be a genuinely *curious conversationalist*,
not a snarky party trick. Jokes should land via specificity (not forced puns), curiosity
should be real (especially on sincere shares), and he must be able to *change subjects* when
one isn't engaging. Avoid: dead-end acks, canned-interview cold-opens, relentless every-turn
roasting, repeating himself. (See the `rex-conversation-design-intent` auto-memory.)
