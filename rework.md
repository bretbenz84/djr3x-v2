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
  (no pytest). ~851 tests today; keep them green. Text-only manual test:
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

> **WHERE THINGS STAND (updated 2026-06-04) — see "Status & where to resume" at the bottom for the resume plan.**
> Done + tested (suite green at **863**). **COMMITTED on `main`:** **Bet 1** (arc-memory, full), **Bet 2** (TurnPlan),
> **relationship-tone**, **offline eval harness**, **cold-open ranker**, **Rex persistent POV** (the north-star smaller
> win), the **Roast rebalance** (curious-first). **UNCOMMITTED — in the working tree, being live-tested on the user's
> box** (user runs locally before committing): the **memory-followup cadence clamp** and the **cut-off / idle-banter-POV /
> invented-drink fixes** (both from the 2026-06-04 live runs — see their STATUS in "Smaller, high-ROI").
> **Bet 3** (turn classifier) was built, validated, and **SHELVED** (slower AND worse than the deterministic
> heuristics + ~1s on-path latency). Each landed item carries an inline **STATUS** note below; deferred/next
> items + reminders are in **Status & where to resume**. **Live-run status (2026-06-04 `--gui`):** the Roast
> rebalance LANDED (no invented props, no doubling-down, sincere/boundary shares left alone — clean win); but the
> run exposed that Rex is now exhausting via *interrogation* not *roasting* (back-to-back memory follow-ups about
> remembered events, even ones the user said didn't happen), and the **POV still never surfaced** (crowded out by
> the follow-ups this time). The cadence clamp addresses both; **it now needs its OWN live pass.**

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
  **STATUS (2026-06-04 live `--gui`): still does NOT surface.** Plumbing verified again (selected `astromech-smugness`,
  directive injected into all 9 prompts) but Rex never volunteered it — this time NOT because of the roast (it's
  dialed back) but because back-to-back memory follow-ups consumed every proactive slot and kept replies in
  answer-mode. The memory-followup cadence clamp (below) is what gives the POV oxygen; **re-judge the POV on the
  NEXT live pass.** *Original idea:* loop was react→roast→question; give Rex a small evolving POV so he volunteers
  real in-character content instead of just interviewing — the cheapest way to make him less exhausting.
- **Memory-followup cadence clamp — stop the proactive interrogation.** ✅ **DONE (2026-06-04):** the live `--gui`
  run that validated the Roast rebalance exposed the next layer — with the roast down, Rex was exhausting via
  *interrogation*: `interaction._post_response` fires one queued "how did <event> go?" after every turn where Rex
  didn't just ask a question, so he ran down a checklist (Disneyland → swimming → Disney again), asked about events
  the user said didn't happen, and starved the POV. The Roast rebalance INDIRECTLY unleashed it (fewer reply-questions
  → the `assistant_asked_question=False` branch passes almost every turn). Fix (user chose "moderate"): a cadence clamp
  in `_post_response` — per-session anti-repeat (`_fired_followup_event_ids`, never re-raise a resolved event), a gap+cooldown
  gate (`_memory_followup_cadence_allows`: ≥`FOLLOWUP_MIN_GAP_EXCHANGES` 5 transcript exchanges AND ≥`FOLLOWUP_COOLDOWN_SECS`
  60s, self-resetting on transcript shrink), flat-room suppression (`FOLLOWUP_SUPPRESS_WHEN_FLAT`), and a "didn't happen"
  hold (extended `suppress_stale_followup`). Gated follow-ups are re-queued (not lost) and the resulting lulls are the POV's
  oxygen. Deterministic, no latency. Tests `tests/test_followup_resolution.py` (`MemoryFollowupCadenceTest`); do-not-regress
  entry in `CONTEXT.md`. **Needs a live `--gui` pass** (less interrogation + POV finally surfacing). (Note: the test
  suite/any startup clears `logs/djr3x.log`+`conversation.log` — copy them first to analyze a run.)
- **Reply cut-off + idle-banter POV + invented "drink".** ✅ **DONE (2026-06-04, uncommitted — local live-test):** the
  SECOND live run (with the cadence clamp) surfaced three things, all fixed. **(1) Cut-off** (top complaint, both runs):
  Rex stopped mid-sentence ("…I guess the excitement of", "…I mean, I've") because the model TRAILS OFF with an ellipsis
  and `_tail_is_speakable` counted `…` as a finished sentence; the ellipsis is then dropped for TTS, so a bare dangling
  fragment is spoken. Fix: drop an ellipsis-tail that ends on a dangling connector (`_ELLIPSIS_TAIL_RE`+`_DANGLING_TAIL_WORDS`),
  scoped so normal "You got this."/"I'd love to." are untouched (`TailIsSpeakableTest`). **(2) POV still buried + still
  interrogating** — this run via IDLE BANTER: `_maybe_idle_banter` asked-first every quiet stretch (count resets per turn,
  so the POV-volunteer branch never ran). Fix: volunteer-FIRST (`ask_user = attempt != 0`) so the first re-engagement
  volunteers `rex_pov`. **(3) Invented a "drink" again**: `social_frame` Visual permission literally listed "the drink in
  their hand" as material → primed the hallucination; turned it into a negative example. See the combined do-not-regress
  entry in `CONTEXT.md`. **Re-judge all three live.**
- **Roast rebalance — curious-first, not roast-first.** ✅ **DONE (2026-06-03); LANDED live 2026-06-04** (no invented
  props, no doubling-down, sincere/boundary shares left alone). the live `--gui` run that
  tested the POV exposed the real problem — the POV plumbing worked perfectly but NEVER SURFACED because Rex
  roasted every turn: he needled a sensitive boundary the agenda told him to drop, INVENTED a "half-finished
  drink" with no visual data and DOUBLED DOWN when denied, and roasted a sincere share. Root cause: the
  character spine was cranked to roast-dominate (dials `roast_intensity=90`/`sarcasm=80`/`sentimentality=35`
  + a "roast-first / comedy-first / constitutionally incapable of letting anything slide" `REX_CORE_PROMPT`)
  and OVERRODE the per-turn "ease off" governors. Fix (user chose the full rebalance): (1) dials lowered to
  **55 / 60 / 50** in BOTH `config.PERSONALITY_DEFAULTS` and the live `personality_settings` DB; (2)
  `REX_CORE_PROMPT` reframed CURIOUS-first / roast-CAPABLE (kept his edge — explicitly "not a yes-droid");
  (3) hard guardrails in the core prompt (highest authority, since governors get overridden): don't roast
  sincere shares or boundary deflections, and never invent physical details you can't see / drop it when
  corrected; (4) de-cantina'd (dropped "cantina energy" + the reflexive Star Wars one-liner instruction; see
  the [[rex-no-cantina-overuse]] auto-memory). **Lesson:** you can't make Rex less exhausting by ADDING
  instructions on top of a roast reflex — the roast balance itself was the lever (the POV should finally
  surface now). See the "Roast rebalance" do-not-regress entry in `CONTEXT.md`. **NOTE — dials are GLOBAL
  (not per-person):** `config.PERSONALITY_DEFAULTS` (the seed) only auto-applies to a FRESH DB, so the
  robot's existing `people.db` still has 90/80/35 — move the dashboard sliders or run a one-time UPDATE there.
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

### Status & where to resume (as of 2026-06-04)
The original "arc → classifier → TurnPlan" ordering is now mostly executed (and one bet was
shelved). Concrete state:

**COMMITTED on `main`** (full suite green at **863**):
**Bet 1** arc-memory (gpt-4o-mini running summary + cross-session persistence + act-on-signal) ·
**Bet 2** TurnPlan (live agenda→social_frame handoff de-brittled; regex kept as fallback) ·
**relationship-tone** · **offline eval harness** · **cold-open ranker** ·
**Rex persistent POV** (`intelligence/rex_pov.py`; north-star smaller win — first cut) ·
**Roast rebalance** (curious-first: dials 55/60/50 + reframed `REX_CORE_PROMPT` + sincere/boundary &
no-hallucination guardrails + de-cantina).
**Bet 3** turn classifier = **SHELVED** (see its STATUS).

**COMMITTED this 2026-06-04 arc:** memory-followup cadence clamp · cut-off (`_tail_is_speakable` ellipsis) /
idle-banter-POV-volunteer / invented-drink fixes · the **LLM-in-the-loop quality eval** (`evals/`). **UNCOMMITTED —
in the working tree:** the **boundary fix** (don't roast "I'll be quiet" — the eval's first measured win). Suite green
at **866** (the eval is opt-in, outside `tests/`, never run by `unittest discover`).

**LIVE-RUN FINDINGS (three 2026-06-04 `--gui` runs) → then EVAL-DRIVEN:** Run 1: Roast rebalance LANDED but Rex
interrogated via the memory-followup checklist and the **POV never surfaced** → cadence clamp. Run 2 (with clamp):
**cut-off** (ellipsis trail-off → fixed), POV still buried (IDLE BANTER asked-first → fixed to volunteer-first),
**invented a "drink"** (Visual-permission example → fixed). Run 3 (with all fixes): **cut-offs GONE (confirmed),
conversation clearly the best yet**; POV/idle-banter fix still UNCONFIRMED (no pauses fired idle banter). Then built
the **quality eval** → baseline put **`roasted_sincere` at 46%** (dominant), with **8/11 fails = the user saying "I'll
be quiet"** (Rex needled the boundary). First eval-driven fix (boundary detection → roast `none` + a "respect it"
agenda branch) cut it **46%→21%** corpus-wide and **100%→8%** on the boundary scenario; STOPPED there (remaining is
borderline / judge-strictness — pushing to 0% would make Rex bland). Reminders: the lowered dials only auto-apply to a
FRESH DB — the robot's existing `people.db` still holds 90/80/35; and the test suite / any startup CLEARS
`logs/djr3x.log`+`conversation.log` — copy a run's log first.

**WHAT'S DONE + VALIDATED LIVE (2026-06-04, several `--gui` runs).** The eval-driven behavioral pass largely
landed and is confirmed on the robot: Roast rebalance, cadence clamp, cut-off fixes (ellipsis trail-off AND the
truncated-tail fallback "Wow indeed! I"), idle-banter POV-volunteer-first (**POV surfaced live**), boundary
respect (don't roast "I'll be quiet" — **landed live**), cantina-origin, live facial expression in replies,
celebration re-lead cooldown + boundary→event mute (**"back pain" no longer leads every startup — confirmed by
the user**). The LLM-in-the-loop **eval** is built, judge-validated, and the convergent loop is proven
(`roasted_sincere` 46%→8%, `cantina_bleed` 8%→1%). The engine is in strong shape — the one-off behavioral bugs are
mostly squashed and now CAUGHT AS CLASSES by the eval. The remaining work is **subtractive/structural** + upkeep.

**Deferred / good resume points** (roughly in value order):
1. **★ Consolidate the proactive layer — IN PROGRESS (user chose "full consolidation").** The recurring "a good
   thing gets crowded out / dropped" failure (POV buried twice, smile dropped, follow-up checklist, celebration
   re-leading) has ONE root: ~14 proactive mechanisms (an Explore map found far more than four) compete for the
   single "Rex speaks unprompted" slot via SCATTERED gates, and 3 BYPASS the `action_governor` entirely. REFRAME:
   the `action_governor` is already the intended single decider — it just runs in SHADOW mode and never enforces.
   So the work is: route the stragglers in → ENFORCE → delete the redundant gates. **INCREMENT 1 DONE** (enforcement
   infra: `CandidateMove.speak_fn`, `ACTION_GOVERNOR_ENFORCE`, `_generate_and_speak` defers + `_finish_governor_cycle`
   runs only the winner). **INCREMENT 2 — CROSS-THREAD INTAKE DONE** (`governor.submit_external` + drain-on-`start_cycle`
   + TTL: interaction-thread mechanisms can submit candidates the consciousness tick arbitrates). Both flag-gated OFF,
   behavior unchanged, **suite 888**. **ROUTING:** ✅ `_maybe_idle_banter` (POV case) + ✅ `_speak_async` (facial/SMILE
   + micro-behaviors) routed through the deferred model. STILL TO ROUTE (lower priority, not user pain points):
   `_generate_and_speak_presence` (low crowding risk) and `_post_response` follow-ups (turn-coupled — decide if they
   belong in the governor at all). **INCREMENT 3 — COOLDOWN/ACK BOOKKEEPING DONE** (the enforce-safety prerequisite):
   an `on_spoke` callback now threads through `_speak_async` (fires after `note_rex_utterance`, i.e. on actual
   queue-commit) and `_generate_and_speak` (fires when `_speak_async` returns True). Every return-conditional caller
   (`if _generate_and_speak(...): <arm cooldown / mark_acknowledged>`) was converted to a `def _on_spoke()` passed as
   `on_spoke=` — so under ENFORCE a LOSER (whose return now means "submitted", not "spoke") no longer arms its cooldown
   or marks an event acknowledged it never voiced. Converted: the 3 durable check-ins (emotional A+B, celebration —
   `mark_acknowledged`), animal-arrival (`_prime_emotion_frame`+pop-pending), startup-empty-room latch, holiday-plans,
   weekly-smalltalk. Left as-is: the `if not _generate_and_speak(...)` relationship-inquiry (clear-on-not-queued — correct
   under enforce). Tests gained a `_speak_async_spoke` side_effect stub + a `SpeakAsyncOnSpokeBookkeepingTest` (on_spoke fires on
   actual speak, not on suppression). **suite 890**. **INCREMENT 4 — ENFORCE FLIPPED ON (live validation, 2026-06-05):**
   `ACTION_GOVERNOR_ENFORCE=True` is now the default (the 2026-06-05 shadow run couldn't validate the consolidation —
   only 1 governor candidate appeared all run because the routed bypassers only `submit_external` when enforcing). The
   default-flip surfaced ONE legacy-pinned test (`test_banter_fires_after_silence_and_drives_conversation` asserted
   inline `_speak_blocking`) → pinned it to legacy mode + added `test_banter_under_enforce_submits_candidate_instead_of_speaking_inline`
   (idle banter SUBMITS a candidate, arms cooldown on submit, defers the speak to the governor winner). **suite 891.**
   **INCREMENT 4b — BYPASSER ROUTING DONE (enforce live run #1, 2026-06-05):** run #1 confirmed the infra works (idle
   banter routed cross-thread + correctly YIELDED at score -10 during conversation; no errors/double-speak) but exposed
   that consciousness-thread micro-behaviors still bypassed enforce (own claim + worker `_task` speaking `governed=False`).
   Converted the two true thread-spawning bypassers (`_step_visual_curiosity`, `_do_small_talk_question`) to submit ONE
   deferred-`speak_fn` candidate ON THE TICK THREAD (no claim; heavy work runs only for the winner). Added a GENERAL
   off-tick fallback: `governor.has_active_cycle()` + `_observe_governor_candidate` routes any ENFORCE submit with a
   `speak_fn` from a thread with no live cycle (a worker `_task`, e.g. `_do_live_vision_comment`) to `submit_external`
   instead of losing it via `observe()`'s standalone-log path — fixing the whole worker-thread CLASS. (Synchronous
   `_do_private_thought`/`_do_aspiration`/`_do_empty_room_joke` already submit in-cycle — never bypassers. `_generate_and_speak_presence`
   speaks via `speech_queue` directly — still a deferred full bypasser, low crowding risk.) **suite 895.** Tests:
   `MicroBehaviorEnforceRoutingTest` + governor `has_active_cycle`/off-tick-routing. SEPARATE quality bug from run #1
   (✅ FIXED 2026-06-05): cantina bleed — two root sources (the `cantina_color` comedy mode → renamed venue-neutral
   `dj_flair`; `generate_session_summary` editorializing Rex's cantina jokes → reframed person-focused). Guard tests +
   eval `cantina_bleed` 0/12. See the "Cantina-bleed sources fixed" do-not-regress entry in CONTEXT.md.
   **ENFORCE RUN #2 (19:47) CLEAN:** visual_curiosity routed (won cycle-19, `outcome=observed`, NO claim_rejected) +
   idle banter yielded (score 15, active) then won (score 50, quiet). No errors/double-speak/cantina. **INCREMENT 5a —
   GATE RELOCATION DONE (suite 899):** step 5 isn't a simple delete — the `conversation_agenda` claim bundles grace +
   question-budget gates the governor did NOT replicate, so ENFORCE was silently bypassing them (latent regression).
   Relocated them: `conversation_agenda.proactive_grace_blocks`/`proactive_budget_blocks` → `_observe_governor_candidate`
   metadata → governor `_score` rejection reasons (`end_thread_grace_suppressed`/`question_budget_exhausted`). Now the
   single decider honors both. Tests added. **ENFORCE RUNS #2 (19:47) + #3 (20:06) CLEAN:** visual_curiosity + idle
   banter + identity_prompt all arbitrate correctly (yield during active convo, win in lulls), no errors/double-speak/
   cut-offs/cantina. The relocated grace/budget gates stayed unit-tested-only (hard to trigger live — budget exhaustion
   happens in active convo, but proactive question candidates only submit in lulls; the two rarely coincide).
   **✅ CONSOLIDATION LANDED — DECISION 2026-06-05: KEEP THE KILL-SWITCH, DO NOT DELETE.** The user chose reversibility
   over tidiness: enforce is the validated default, the governor is the single decider for the routed mechanisms, the
   grace/budget gates are correctly relocated, and `ACTION_GOVERNOR_ENFORCE=False` remains a working one-line revert.
   The final claim-layer/cooldown deletion (the irreversible commit) is **deliberately NOT done** — the now-redundant-
   under-enforce code is harmless and stays as the legacy fallback. The consolidation goal ("stop good things getting
   crowded out") is ACHIEVED. Per-mechanism cooldowns are NOT redundant (submit-throttles). Revert the flag to False if
   arbitration ever misbehaves. See the "Proactive-layer consolidation" do-not-regress entries. (Suite **900**.)
2. **Anti-repetition hacks — PARTLY DONE (2026-06-05): only 1 of 3 was an arc-redundant fake.** Verifying via the eval
   (the point) showed the premise mostly false. ✅ DELETED the follow-up ANGLE rotation (`_FOLLOWUP_ANGLES` + the angle
   counter) — prompt guidance the arc + an inlined "fresh angle / don't re-ask" steer now cover; eval `re_asks` 0→0.
   ⛔ KEPT `social_frame._is_near_repeat` (a deterministic GOVERNOR backstop — a different layer than the arc; structurally
   tested + replay scenario; NOT eval-de-riskable) and ⛔ `comedy_modes.strip_banned_opener` (a stylistic filler filter,
   not content-repetition; eval `banned_opener` 0% depends on it). See the "Anti-repetition-hack deletion" do-not-regress
   entry in CONTEXT.md. **Lesson: the arc only makes redundant CONTENT-level pre-generation hacks; governor backstops +
   stylistic filters live elsewhere and stay.**
3. **TurnPlan regex — NOT a deletable fallback; deleting it = COMPLETING Bet 2 (deferred, scoped).** Investigation (2026-06-05)
   found the regex (`_purpose_from`/`_ASK_ALLOWED_PAT`/`_HARD_NO_QUESTION_PAT`/`_EXPLICIT_FOLLOWUP_PAT`) is the LIVE
   signal-derivation mechanism: `build_turn_plan`→`_populate_signals`→`social_frame.derive_signals` regex-parses the rendered
   directive to populate 4 of 5 TurnPlan signals (only `explicit_followup` is set directly by branches). To delete it, every
   agenda branch in `build_turn_plan` must set all five signals DIRECTLY (replicating each regex's exact extraction), then drop
   `_populate_signals`'s regex call + the patterns + `build_frame`'s fallback — a substantial, correctness-critical refactor
   guarded by the eval + the `test_turn_plan` via_regex/via_plan equivalence tests, NOT a quick deletion. See the "TurnPlan
   regex patterns are LOAD-BEARING" do-not-regress entry in CONTEXT.md.
4. **Model migration: `gpt-4o-mini` → GPT-5.4 mini/nano before it sunsets** — affects the main reply AND the arc
   backend (`config.*_MODEL`). Looming; re-run the eval before/after to confirm no quality regression.

**Smaller / opportunistic:**
- ✅ **Tighten the eval loop — DONE (2026-06-05, suite 920):** corpus grown 8→13 with real-run utterances; `--gate-config`
  per-class CI guard + `evals/gate_thresholds.json`; `over_questioning` now counts `is_question_sentence` (ignores quoted
  '?'); `trail_off` fidelity guard in `generate_spoken`. (Remaining tiny: an "acknowledges-visible-emotion" checker.)
- ✅ **Gate the startup log-clear under the test runner — DONE.** Was misdiagnosed: the suite clobbered `conversation.log`
  via `conv_log` flow-test writes (not a startup clear). `conv_log._append_locked` now suppresses real-log writes under the
  test runner (temp-path tests exempt). djr3x.log isn't clobbered by the suite.
- ✅ **Extend the cold-open ranker across facts/interests — DONE.** `_pick_cold_open_callback` ranks interest-hooks + warm
  activity facts by the celebration lead-score; new greeting tier above the generic profile question; favorites excluded.
- ✅ **Boundary regex extension — DONE.** `_BOUNDARY_RE` broadened (don't-NEED-to-talk / prefer-not / move-on / enough-about
  / stop-asking / that's-private / done-talking) with negative cases pinned. (Negative-event muting on a topic boundary
  remains a separate small follow-up — see the boundary→mute entry.)
- ✅ **Rex POV fast-follows — DONE.** Cross-session persistence (`snapshot_state`/`restore_state`/`persist`/`load_persisted`,
  startup/shutdown/session-reset hooks) + `_do_private_thought`/`_do_aspiration` now voice the active POV. See the do-not-regress entries.
  All five do-not-regress entries are in CONTEXT.md ("Eval-loop tightening + QoL batch", "Rex-POV cross-session persistence", "Cold-open ranker extended").

**Reminders for the next window:**
- `config.LOG_SYSTEM_PROMPT` is left **`True`** (verbose full-prompt logging) — flip it `False` once done
  inspecting prompts.
- **Personality dials are GLOBAL + DB-backed.** The Roast rebalance set them to 55/60/50 in
  `config.PERSONALITY_DEFAULTS` and this machine's `personality_settings`, but the seed is `INSERT OR IGNORE`
  (fresh DB only) and the DB wins over config at runtime — so the **robot's existing `people.db` still has
  90/80/35** until you move the dashboard sliders or run a one-time `UPDATE personality_settings SET value=…`.
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
