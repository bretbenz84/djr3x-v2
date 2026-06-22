# DJ-R3X v2 — Comprehensive Code Review (June 2026)

**Date:** 2026-06-21
**Reviewed at commit:** `88fdaa9` (main)
**Goal lens:** DJ-R3X is meant to be an engaging, FUN conversational robot that is **funny**, **roasts** the end-user, **remembers** things across conversations, and is **fun to use**. Every finding below is ranked by impact on those four pillars.

---

## 0. How to use this document (read first)

This is a handoff for a future context window. The review was done by fanning out 16 deep subsystem reviewers over the ~138K-line codebase, then adversarially verifying **every** concrete finding with independent agents. 104 findings survived verification; 5 were refuted (see §5 — don't re-chase them).

**Before acting on any finding:**
1. **Re-read the cited code** — line numbers drift as the tree changes. Treat `file:line` as a starting point, not gospel.
2. **The findings were verified as of `88fdaa9`.** If a finding looks already-fixed, it may have been addressed in a later commit; confirm before "fixing" again.
3. **Run the suite** before and after: `./venv/bin/python -m unittest discover -s tests` (use `./venv/bin/python`, not bare `python` — bare pyenv lacks numpy/openai). Baseline at review time was **1942 tests, 2 failures** (see §1).
4. **Add a regression test with each fix.** Most findings name the test to add.
5. **Workflow:** this project works directly on `main` (no PRs/branches); commit + push after each verified change.

**Suggested first batch** (small, high-leverage, goal-aligned): the 2 red tests + Tier-1 items #1–#7 below. Each is a focused change with an obvious regression test.

---

## 1. Currently-red tests (ground truth at review time)

`1942 tests, 2 failures`:

1. **Comedy regression (goal-relevant).** `test_collapses_overexplained_joke_tail` — `intelligence/comedy_modes.py:433` `_collapse_overexplained_joke`. The collapser only strips explainer clauses set off by a comma/dash (`..., see` / `... — because`). The most natural over-explained form — `"That was a great landing. Get it? Because I crashed."` (sentence-separated by a period) — slips through, so **Rex explains his own jokes**. The narrowing that protects legit lines like `"I can't see you."` broke the period-separated case.
   - *Fix direction:* also match a sentence break followed by a standalone explainer sentence, e.g. `\.\s+(?:Get it\?|Because\b).*$`, while keeping the comma/dash guard so `"I can't see you."` is never truncated. Re-run `tests/test_streaming_tts.py::PolishStreamSentenceTest`.
2. **Responsiveness tuning drift (low goal impact).** `test_live_tracked_edge_face_slews_neck_responsively` (`tests/test_face_tracking.py:836`) — neck slew `144` vs expected `≥150`. Either a face-tracking gain regressed or the threshold is stale; decide which is correct on-device.

---

## 2. Prioritized changes (most goal impact first)

Format: **rank. title** — `category` / severity / goal_impact · `location` · what & why · **Fix**.

### Tier 1 — Fix first (high goal impact: funny / games / remembers)

**1. 20 Questions ends on turn 1** — bug / high / high · `features/games.py:404-409` (`_20q_handle`)
`is_final_guess` is True if the text contains `"is it"` (among others), but `"Is it ...?"` is the canonical yes/no question. `"Is it alive?"` → final_guess=True → fuzzy-matches the secret (fails) → clears game state → reveals the answer → ends. The most natural first question instantly loses; the game is effectively unplayable. **Verified first-hand.**
**Fix:** treat input as a final guess only when it's a declarative naming attempt, not an interrogative. Detect `is_question` (ends with `?` or starts with is/are/does/do/can/was/will). Drop the bare `"is it"` trigger; require commit phrasing (`"my final answer is"`, `"i think it's"`, `"my guess is"`) or `questions_left<=0`. Test: `"Is it alive?"` returns `done=False` and increments `question_count`.

**2. Humor/game/performance commands swallowed for 120s after any Rex question** — bug / high / high · `intelligence/dialogue_act.py:326-345` + `intelligence/interaction.py:1349-1380` (`_dialogue_allows_action_breakout`), `:18461`, `:18497`
Any Rex turn containing `?` arms a 120s `answer_to_rex` frame (never consumed after a reply). Under it, explicit FUN commands (`"make me laugh"`, `"riff for us"`, `"start trivia"`, `"wave at them"`) are labeled `answer_to_rex` with `skip_action_router=True`, bypassing both the fast-local takeover and the LLM router; the breakout allowlist only covers sleep/shutdown/music/vision. Since Rex is *designed* to ask frequent questions, this routinely degrades funny/games/performance responsiveness. Coverage today is accidental (only phrases already in `_EXPLICIT_COMMAND_RE` survive).
**Fix:** broaden `_EXPLICIT_COMMAND_RE`/`_direct_control_kind` to recognize humor/performance/game/wave imperatives so `skip_action_router` stays False; and/or add `humor.*`, `performance.*`, `game.start`, `wave` to `_dialogue_allows_action_breakout` gated on the matching explicit classifier. Also consume/deactivate the frame after the first user turn.

**3. Episodic & nostalgia callback dedup sets never cleared per session → fire once per BOOT not per conversation** — bug / high / high · `intelligence/llm.py:344,348,352,393,436,463`; `intelligence/interaction.py:13223-13260` (`_end_session`)
`_episodic_callbacks_used_this_session`, `_nostalgia_used_this_session`, `_stale_facts_asked_this_session` are module-level sets never reset in `_end_session` (which clears `callback_engine`, `topic_thread`, etc.). On the long-running robot, conversations are ACTIVE↔IDLE transitions, so each shared-memory/nostalgia callback fires at most once per process uptime. With `EPISODIC_RECALL_PERSON_CALLBACK_PROBABILITY=0.25` this bites quickly, silently starving "I made you laugh / we played trivia" recall for returning visitors — directly defeating the remembers-across-conversations pillar. `callback_engine.clear_session()` even names this "known bug-shape".
**Fix:** add `llm.clear_session()` that clears all three sets and call it from both branches of `interaction._end_session` alongside `callback_engine.clear_session()` (guarded import). Update the misleading module comments. Repoint test helpers to call `llm.clear_session()`.

**4. Interest regex mints garbage interests from auxiliary "I do …" and persists them at confidence 0.95** — bug / high / high · `intelligence/conversation_steering.py:73` (pattern), `:441-468` (`_store_interest_fact`), `:232-244` (`note_user_turn`)
The 2nd interest pattern's `i\s+do\s+(?!not\b)` alternative matches emphatic auxiliaries: `"I do think the weather is nice"` → interest `"think the weather is nice"`. `_topic_is_substantive` only rejects pure function-word fragments, so it passes and is written as a `person_fact` AND typed interest at confidence 0.95 (source 'explicit'), steering this session and resurfacing next conversation. Poisons the "remembers" store with sentence fragments at the highest confidence tier. Fires on the live per-turn path, ungated. **Verified first-hand.**
**Fix:** extend the negative lookahead to exclude common following verbs (think/believe/need/have/want/know/remember/...) and/or reject topics whose first token is a common verb in `_clean_topic`/`_topic_is_substantive`. Tests: `detect_interest('I do think the weather is nice')` is None while `'I do yoga'`/`'I do woodworking'` still return the hobby.

**5. Scene analyzer runs during DJ music → Rex startles/laughs at his own playback** — bug / high / high · `audio/scene.py:65-85` (skip guard), `:114-122`; consumed at `intelligence/consciousness.py:4808-4839` and `intelligence/llm.py:243-247`
The scene loop's "skip while playing own audio" guard only checks `speech_queue.is_speaking()` and `output_gate.seconds_since_release()`. DJ/radio playback uses a raw `sd.OutputStream` and only calls `echo_cancel.set_playing(True)` — never holding the output_gate or speech-queue flag. So during DJ music the scene loop analyzes the mic full of Rex's own music: startle reactions (yelp/squeal) can fire and the prompt is told "music/laughter detected" near-every session. **Verified first-hand.**
**Fix:** add `from audio import echo_cancel` and short-circuit the scene loop when `echo_cancel.is_suppressed()` is True (True for both TTS and DJ, plus the post-playback tail) — the same idiom `wake_word.py`/`barge_guard.py` already use.

**6. Word Association crashes (IndexError) on any empty LLM result** — bug / medium / medium · `features/games.py:1587-1589` (`_wordassoc_handle`)
`next_word_raw = _quick_call(...).strip().split()[0]...` and `_quick_call` returns `''` on any OpenAI exception. `''.split()[0]` raises IndexError one line before the intended `'systems'` fallback. The exception is swallowed by the broad except in `interaction.py`; `_active_game`/`_game_state` are never cleared, so the game soft-bricks for the duration of a transient LLM outage.
**Fix:** `parts = _quick_call(...).strip().split(); next_word_raw = parts[0].strip(...).lower() if parts else ''; next_word = next_word_raw or 'systems'`. Optionally wrap the handler body in try/except that clears `_game_state` on error.

**7. Cross-session voice→person resolution not implemented — named, departed speaker returns as unknown_voice_N** — ineffective_feature / medium / high · `intelligence/interaction.py:2519-2549` (`_resolve_anonymous_speaker_slot`); `memory/voice_signatures.py:111-147` (match), `:199-219` (`attach_person`)
When a fresh unknown voice matches a persisted signature whose `person_id` was promoted via `attach_person`, `_resolve_anonymous_speaker_slot` reads `prior.get('person_id')` only into a log line and always returns an anonymous label. So a person named last session who returns (and isn't re-recognized by face/biometric) comes back as `unknown_voice_N` — and Rex says "familiar voice, never got a name" when the name is on file. Defeats the cross-session voice-memory payoff the write side built. **(Also a CONTEXT discrepancy — see §3.)**
**Fix:** in the new-slot branch, when `prior` matches and `prior.get('person_id')` is not None and clears a confidence floor (with margin guard), resolve the turn to that known person instead of minting `unknown_voice_N`. Widen the function return (or look up the name internally). Gate behind a default-ON flag; add a test.

**Web search (rank 12, high/high) — listed here for Tier-1 visibility:** with `WEB_SEARCH_MODEL=None` the hosted `web_search` tool runs on `gpt-5.4-mini`, whose tool support was never live-verified; if unsupported, every "look it up" silently degrades to stale-knowledge answers. See full entry below.

### Tier 2 — Meaningful goal impact

**8. GUI text input bypasses INTERACTION_PAUSED** — bug / low / low · `intelligence/interaction.py:19826-19861` (`submit_text`); `gui/memory_banks.py:211-220,234-237`
Opening Memory Banks sets `INTERACTION_PAUSED=True` and shows a "Rex won't listen, respond, or speak" banner. The audio loop honors it, but `submit_text()` (GUI/CLI text path) only guards SLEEP/SHUTDOWN and otherwise runs the full LLM reply + post-turn memory extraction. The Memory Banks window is non-modal, so typing while editing memory still generates a reply and can write memories.
**Fix:** in `submit_text()`, return False early when `config.INTERACTION_PAUSED` is set (mirror the SLEEP/SHUTDOWN guard). Update the GUI drop message. Add a paused-case test.

**9. Governor drops force_salient candidates (animal arrival starved)** — ineffective_feature / medium / medium · `intelligence/speech_engine.py:343-392,415-417`; `intelligence/action_governor.py:358,403,407`
Animal-arrival and scenery-change reactions speak via `force_salient=True` under ENFORCE, but the submitted `CandidateMove`'s metadata (`governor_speech_metadata`) computes `cooldown_active` and `can_proactive_speak` with NO salient awareness, and `_score` hard-rejects on them. The deferred `speak_fn` that honors salient never runs because the governor rejects it first. So a delightful "a dog walked in!" within the 12s pacing window is silently dropped.
**Fix:** stamp `metadata={'salient':bool(force_salient),'reactive':bool(reactive)}` into the candidate, and in `action_governor._score` skip the `cooldown_active`/`can_proactive_speak_false` rejections when salient/reactive is set (the deferred `speak_fn` still re-checks live-speech/music gates).

**10. people_roast and memory_musing default to floor priority 20** — ineffective_feature / low / medium · `intelligence/action_governor.py:74-105,304`; `intelligence/idle_behaviors.py:138-145,287-300`
`do_people_roast` and `do_memory_musing` call `_generate_and_speak` with no explicit priority, and neither purpose is in `_PURPOSE_PRIORITIES`, so both fall to the accidental floor (20). They still clear the min-score gate (not dead), but lose to any higher-priority candidate in the same governor cycle. `people_roast` directly serves the ROAST goal.
**Fix:** add explicit `_PURPOSE_PRIORITIES`: `people_roast ~46` (wins ties vs ambient idle chatter, below sincerity flows like emotional_checkin/memory_followup), `memory_musing ~24` (just above floor). Do not raise into the sincerity tier.

**11. gentle_probe (masked-distress) mode still allows roasting and visual jabs** — bug / medium / medium · `intelligence/social_frame.py:848` (`_roast_level`), `:779` (`_visual_allowed`); directive at `intelligence/empathy.py:745-750`
`_roast_level` and `_visual_allowed` treat only `{listen,support,validate,ground,brief}` as tender. `gentle_probe` (returned when someone says "I'm fine" but prosody/face signals strain) keeps affect neutral and sensitivity 'none', so it passes the backstops: `_roast_level(...,'gentle_probe','neutral','none')` returns `'normal'` and `_visual_allowed` returns True — even though the gentle_probe directive says "No personal roasts." `callback_engine` and `consciousness` already carry `gentle_probe` in their tender sets; social_frame's drifted.
**Fix:** add `gentle_probe` (and `kind_default`/`child_kind`/`course_correct`/`crisis`/`acknowledge_then_yield`) to the tender-mode sets in BOTH `_roast_level` and `_visual_allowed`; best to define a single shared `TENDER_MODES` constant referenced by all three modules. Test: `_roast_level(...,'gentle_probe','neutral','none')=='none'` and `_visual_allowed=False`.

**12. Web search runs on gpt-5.4-mini with unverified hosted web_search tool support** — ineffective_feature / high / high · `intelligence/web_search.py:83-88` (`_search_model`), `406-449` (`answer`); `config.py:4625` (`WEB_SEARCH_MODEL=None`), `170` (`LLM_CONVERSATION_MODEL`)
`_search_model()` falls through `WEB_SEARCH_MODEL=None` to the conversation model `gpt-5.4-mini`, whose Responses-API `web_search` tool support was never live-verified (the smoke test covers chat only; every web_search test mocks the call). `answer()` wraps the call in a broad except returning `ok=False`, and `_maybe_web_search_reply` just logs and falls through to a normal from-knowledge reply. If the tool is unsupported, every "look it up" silently degrades to stale, confidently-wrong knowledge.
**Fix:** default `_search_model()` to a known search-capable model (e.g. `gpt-4o-mini`) for the hosted tool, OR live-verify gpt-5.4-mini via a real `responses.create` web_search smoke test and document it. When `forced=True` and the call raises unsupported-tool, retry once with a fallback model; optionally surface an in-character "couldn't reach the net" line. (See also rank 26.)

**13. Multi-person gesture/wave mis-attribution — pose always writes slot 0** — bug / medium / medium · `vision/pose.py:547-597` (`_update_world_state`); `intelligence/consciousness.py:2582-2604` (`_step_wave_reaction`)
MediaPipe Pose is single-person and `_update_world_state` merges its gesture/engagement/keypoints purely positionally onto slot 0 (clearing slots ≥1), with no IoU/spatial alignment to the face slots (bound by identity, not detection index). `face_expression` already IoU-matches via `_match_expression_to_people`; pose does not. With 2+ people Rex waves back at / names the wrong person, or attributes "waving" to someone holding still.
**Fix:** derive a head bbox from the pose's own keypoints and IoU-match it to the best face slot before merging (mirror `_match_expression_to_people`). Cheap guardrail: only attribute gesture/engagement when exactly one face is visible; with 2+ and no confident match, suppress the wave attribution.

**14. Onboarding is_pivot misfires on enthusiastic answers** — bug / medium / medium · `intelligence/onboarding.py:416-448` (`is_pivot`/`_PIVOT_PAT`), called at `intelligence/interaction.py:9142-9146`
`is_pivot` returns True for a bare `"can you"` substring anywhere, OR any `?`-ending sentence containing `"you"`. So `"I'm a paramedic, can you believe it?"` and `"I do a lot of climbing, you know?"` are read as pivots: the burst closes with `_close_onboarding('pivot')` and the answer is NOT recorded (`record_answer` runs only after the gate). A new person's enthusiastic answer is dropped from memory and the first-impression burst ends prematurely. The sibling `tell_me_about` guard is more careful.
**Fix:** tighten `is_pivot` — drop the bare `?`+`you` heuristic (or restrict to "what/how about you?"), anchor command matches to the start of the turn, exempt filler tics ("you know", "can you believe it"). Record the answer before bailing on a true pivot. Add tests.

**15. "Postponed" treated as a hard cancellation — a rescheduled event is durably lost** — bug / low / medium · `memory/events.py:21-30` (`_CANCEL_PAT` includes 'postponed'), `:188-199` (`cancel_event`), `:233-274` (`cancel_matching_events`)
`_CANCEL_PAT` matches 'postponed', so `looks_like_cancellation` returns True and `cancel_event` sets `status='canceled', followed_up=TRUE`. A postponement is a reschedule — the plan still exists. There's no events.py path that updates `event_date`; the row drops out of `get_open_events`/`get_pending_followups` forever, so Rex stops anticipating and never asks "how did the camping trip go?".
**Fix:** split postpone/reschedule out of `_CANCEL_PAT` into a `looks_like_postponement` helper. In `_cancel_stale_event_memory`, branch on postponement: keep `status='planned'`/`followed_up=FALSE` and update `event_date`+`mentioned_at` if a new date is parseable; otherwise leave open. Update the test that asserts "they postponed it" is a cancellation.

**16. World-reaction triggers consume dedupe state before random.choice** — bug / medium / medium · `intelligence/consciousness.py:4843-4844,4864-4866,4918-4922,4941-4958`
In `_step_proactive_reactions`, the notable-date/time-of-day/weather triggers mark themselves acknowledged (and arm timestamps) at append time, but only ONE trigger is chosen by `random.choice` and spoken. If two world-changes co-occur in a tick, the un-chosen ones are permanently marked acknowledged this session (`notable_date` never cleared at all) and Rex permanently swallows commentary he never voiced. The startle path proves the correct pattern (defers timestamp until after selection).
**Fix:** defer the dedupe add and timestamp arming until after `random.choice`, keyed off the chosen trigger's metadata (stash `ack_date`/`ack_tod`/`weather_signature` in metadata at append time, apply post-selection like startle). Add `_acknowledged_dates.clear()` to `_reset_state`.

**17. Reactive callbacks require roast level exactly 'normal'** — ineffective_feature / low / medium · `intelligence/callback_engine.py:714-719`; `intelligence/social_frame.py:867-880` (`_roast_level`)
`maybe_claim_reactive` only allows roast in `{'normal'}` unless `CALLBACK_ALLOW_LIGHT_ROAST_FRAME` (default False). But `_roast_level` downgrades `'normal'→'light'` for micro/brief targets — and `'brief'` is the DEFAULT conversational target — and for arc-flat turns. So reactive callbacks are confined to turns that escape the brief/micro default into full `'normal'`, a narrow surface, making "remember a bit and weave it back in" feel near-silent. No safety gate depends on roast level.
**Fix:** default `CALLBACK_ALLOW_LIGHT_ROAST_FRAME=True` (the banked fun-fact is gentle, all safety gates are roast-level-independent). Or add `'light'` to allowed_roasts only when the frame target is micro/brief. Keep the `arc_reads_flat()` block.

**18. Rex-POV double-utterance guard never arms on the normal reply path** — ineffective_feature / medium / medium · `intelligence/rex_pov.py:297-318`; `intelligence/interaction.py:4180-4186` (only caller); `intelligence/llm.py:1086-1089`
`current_pov_directive()` is injected into every normal reply, and the cooldown that prevents re-volunteering the POV (`pov_recently_spoken`) only returns True after `note_pov_spoken()`. That's called from exactly one place — the idle-banter branch — which is itself dead (`ask_user` hardcoded True, `pov_text ''`). So `note_pov_spoken()` never fires in production and the guard is fully inert; the near-verbatim repeat it was built to stop can recur.
**Fix:** in `_register_rex_utterance`, fuzzy-match the finalized line against `rex_pov.active_pov_text()` and call `rex_pov.note_pov_spoken()` on a hit (covers all reply paths). Add a regression test. Fix the misleading docstrings.

**19. Surprise pre-beat is effectively dead on the live streaming reply path** — ineffective_feature / medium / medium · `intelligence/interaction.py:9644-9655, 10140-10149, 9982-9995`; `intelligence/llm.py:1283-1311` (`classify_surprise`)
`classify_surprise` is a gpt-4o-mini round-trip started when the reply begins. On the default streaming path the surprise pre-beat is inserted only if `surprise_result` is already resolved when the first sentence is ready, with no join. The first gpt-5.4-mini sentence at `reasoning_effort='none'` arrives fast, so the classifier usually loses the race and the "...didn't see that coming" beat rarely fires. The non-streaming path joins with a 0.3s timeout and works, but isn't the default.
**Fix:** on the streaming path, briefly join the surprise thread (timeout ≈0.2-0.25s) before emitting the first sentence. Better long-term: move surprise detection to the local qwen sidecar (like `classify_self_emotion`).

**20. Third-party/secondhand premise exclusion is only a soft LLM instruction** — ineffective_feature / medium / medium · `intelligence/callback_engine.py:457-494` (`_llm_candidate`), `497-557` (`bank_from_turn`)
The design doc states a HARD invariant: secondhand/third-party material never enters the callback pool. But `bank_from_turn` gets the raw user turn and the only third-party guard on the LLM banker path is a prompt instruction; post-checks only verify content-word overlap and transcript-echo prefixes. So "My brother is obsessed with rock climbing" can be banked and later fire — Rex roasting the user about their brother's hobby as if it were the user's.
**Fix:** add a deterministic third-party backstop in `bank_from_turn`: require first-person evidence near the premise and reject third-party possessives/subjects ("my brother/sister", "he/she/they ... loves/collects"). Test feeding "My brother is obsessed with rock climbing" with a yes-LLM mock asserts no row banked.

**21. Trivia hard/easy rounds silently serve wrong-difficulty questions** — ineffective_feature / medium / low · `features/trivia.py:240-248` (`get_question`); `features/games.py:548-557` (`_trivia_question_line`)
`get_question` falls back to ALL unasked questions when the difficulty pool is exhausted, without signaling the caller. Small categories have only 2-3 hard questions but rounds are 5, so off-difficulty questions are served while `_trivia_question_line` keeps announcing the fixed 'hard'/'easy' label — an honesty bug on an advertised option.
**Fix:** have `get_question` report the difficulty actually served. In games.py recompute `difficulty_label` per question; optionally a one-time "out of hard ones, switching to mixed" aside. Or pad difficulty-locked rounds / add more questions.

**22. Only 6 of 10 advertised trivia categories exist on disk; starter generator never fills the gap** — dead_feature / medium / medium · `features/trivia.py:42-55` (`_STARTER_CATEGORIES`), `131-144` (`_generate_starter_set`), `176-185` (`_load_bank`)
`_STARTER_CATEGORIES` lists 10 (incl. on-brand Space & Astronomy) at 20 questions each, but only 6 files exist (most with 8 questions) and `_generate_starter_set` runs only when the directory is completely empty. So 4 categories never generate and thin files never top up.
**Fix:** make starter generation per-category and count-aware: generate any missing category and top up files below a minimum (e.g. ≥2× round length), appending rather than skipping. Or commit complete ~20-question banks for all 10, prioritizing Space & Astronomy. Dedupe by question text.

**23. Banked reactive callbacks depend on a racing background relevance thread** — ineffective_feature / low / low · `intelligence/callback_engine.py:562-625` (`refresh_relevance`), `774-785` (stash freshness gate); `intelligence/interaction.py:12448-12454`
`refresh_relevance` runs on the post-response daemon thread; the next turn's `maybe_claim_reactive` only reads that stash and requires it fresh. The deterministic word-overlap fast path lives only inside the background judge, so a literal topic match still depends on the prior thread completing. On fast/barge-in turns the stash can be stale.
**Fix:** run the pure deterministic overlap pass synchronously inside `maybe_claim_reactive` against the active pool (regex/set-intersection), synthesizing a fresh verdict on a literal hit. Keep the qwen stash for semantic matches; optionally widen the stale-exchanges window.

**24. Idle banter never volunteers Rex's own take anymore (dead POV branch)** — dead_feature / low / low · `intelligence/interaction.py:4070-4072, 3923-3934, 3752-3767` (`_IDLE_BANTER_DIRECTIVES[1]`)
`ask_user` is hardcoded True and `pov_text ''`, so the POV-volunteer branch and `_IDLE_BANTER_DIRECTIVES[1]` are unreachable. The change is intentional (POV now surfaces via the reply-path prompt injection), but the docstring, CONTEXT, and a test still exercise/claim the dead path.
**Fix (cleanup):** fix the `_maybe_idle_banter` docstring, remove the unreachable `pov_text`/`pov_volunteered` plumbing, rewrite the `ask_user=False` test cases. Do NOT restore alternation.

**25. is_soft_disengage treats any one-word answer as lukewarm** — ineffective_feature / low / medium · `intelligence/onboarding.py:451-460` (`is_soft_disengage`), consumed at `interaction.py:9152-9168` and `onboarding.py:315,371`
`is_soft_disengage` returns True for any <2-word answer. Legitimate one-word answers ("Austin", "jazz", "paramedic") accrue `soft_streak` toward early wind-down, force `allow_depth` off (skipping the LLM depth probe), and get a content-blind canned ack. A cooperative terse answerer can have onboarding wound down at MIN, never reaching the more memorable interest questions.
**Fix:** only count <2 words as soft when the token is empty, matches `_DUNNO_PAT`, or is pure filler — so "Austin"/"jazz"/"paramedic" stay non-soft while "idk"/"nothing" remain soft. Single change fixes all three call sites; add a test.

**26. WEB_SEARCH_MAX_OUTPUT_TOKENS=600 shared between reasoning and spoken answer** — improvement / low / medium · `intelligence/web_search.py:400,406-409`; `config.py:4629-4632`
`answer()` sets `max_output_tokens=600` and attaches reasoning effort 'low' on the GPT-5 model, where that budget is shared between reasoning and visible output. If 'low' reasoning consumes a meaningful share, the 2-4 sentence answer can be truncated.
**Fix:** raise `WEB_SEARCH_MAX_OUTPUT_TOKENS` (e.g. 1000-1200), or set `WEB_SEARCH_REASONING_EFFORT='none'` for the search call.

**27. Gaze engine's SPEAKING eye-contact rhythm and multi-person include-sweep never fire** — ineffective_feature / low / low · `intelligence/consciousness.py:10057-10085,10069-10077`; `intelligence/gaze_engine.py:371,464-474`
The live adapter folds `speech_active` into the suppression that blocks driving aversions, so the engine idles during SPEAKING and the documented "50% on-target while speaking" duty cycle produces discarded decisions. The include-sweep additionally requires `listener_bearings`, which the live `GazeInputs` never populates (and `center_on` is never consumed). Both inert. (The PREP_TURN "look away to think" beat does work.)
**Fix:** decide intent. If wanted: populate `listener_bearings`/`active_speaker_id` from `world_state.people`, allow gaze to compose with speech motion (bounded amplitude) during SPEAKING, consume `decision.center_on`. If not: drop the SPEAKING duty-cycle and include-sweep claims from the docstring/CONTEXT.

**28. Onboarding self-reveal (reciprocity beat) fires at most ~once** — ineffective_feature / low / medium · `intelligence/interaction.py:9095,9183-9193`; `config.py:3411` (`ONBOARDING_REVEAL_EVERY=3`), `3394` (MAX=5)
`since_reveal` is seeded 1 at the opener and the reveal only fires at `>=3` after the wind-down checks, so the documented "self-reveal woven in ~every N questions (reciprocity, not an intake form)" lands exactly once in a full 5-question burst and zero times on the common 3-4 question bursts.
**Fix:** set `ONBOARDING_REVEAL_EVERY=2`. Do NOT drop the opener increment (that delays the first reveal and makes short bursts worse).

**29. Startle-animal reaction ("Yah! New lifeform") unreachable in continuous-detection mode** — ineffective_feature / low / low · `intelligence/consciousness.py:1863-1890`; `config.py:3759,3787`; `vision/scene.py:1122-1133`
In continuous mode `world_state.animals` comes only from the local MediaPipe detector, whose species set (bird/cat/dog/horse) has zero overlap with `STARTLE_ANIMAL_SPECIES` (snake/spider/wasp). The OpenAI lifeform paths are gated off when local detection is enabled. So the startle bit only fires via the narrow user-initiated directed-attention path.
**Fix:** if wanted, periodically run `detect_lifeforms` even with local detection on (low-frequency, people-present gated). Otherwise document that startle only triggers via directed-attention and delete the unused `detect_animals` helper.

**30. do_bored_environment_snark / do_live_vision_comment burn their cooldown even when speech is blocked** — bug / low / medium · `intelligence/idle_behaviors.py:303-343,455-506`; `config.py:4110,4125-4126`
Both set their cooldown timestamp (300s / 240s) at the TOP, before the worker's `_can_proactive_speak()` gate (and before checking the scene description). If the gate fails or the scene returns nothing, the full cooldown is consumed on a no-op, so these heavily-weighted "riff on the room" behaviors fire less than intended.
**Fix:** move the cooldown-timestamp assignment into the worker, after `_can_proactive_speak()` passes and after a usable line is produced. Keep the read-side check as a cheap pre-filter only.

### Tier 3 — Lower impact / cleanup / latent safety

**31. Smile/presence reactions bypass the action governor** — context_discrepancy / low / low · `intelligence/consciousness.py:2457-2495,2716-2717`; `intelligence/speech_engine.py:614-744`
`_speak_smile_reaction` enqueues directly (no `CandidateMove`) at loop step 10g before `_finish_governor_cycle`, so a low-stakes smile ack can pre-empt a same-tick priority-100 emotional check-in. `generate_and_speak_presence` also self-arbitrates. Violates CONTEXT's "single decider" invariant (narrow concrete harm).
**Fix:** route smile/wave/presence through the governor as deferred candidates, OR move `_step_smile_reaction`/`_step_wave_reaction` below `_finish_governor_cycle` and amend CONTEXT to document these as intentionally governor-exempt.

**32. Stale-event reroute / event.cancel can't fire** — ineffective_feature / low / low · `intelligence/action_router.py:1607-1618`; `intelligence/interaction.py:1086-1091`; `memory/events.py _CANCEL_PAT`
`event.cancel` is absent from `ACTION_ROUTER_EXECUTE_ACTIONS`, so it's blocked as 'not_in_execute_allowlist' at any confidence. A primary path handles common cancellations, but gap phrasings ("no longer happening", "is off/over", "ended", "scrapped", "already passed") survive and Rex keeps re-asking.
**Fix (lowest-risk):** broaden `memory/events.py _CANCEL_PAT` to cover the gap phrasings (keep `event.cancel` off the router allowlist). Alternatively allowlist `event.cancel` AND floor the reroute at `>=0.85`.

**33. Voiceprint auto-refresh has an undocumented visual-speaker guard** — context_discrepancy / low / low · `intelligence/interaction.py:7520-7538` (`_maybe_auto_refresh_voice`); `config.py:4024-4026`
CONTEXT says refresh is gated only on `raw_best_id==person_id`, but the code adds a second default-on guard requiring the visual active-speaker latch to confirm the same person. Retried per-turn, so not permanently skipped, but can be delayed under occlusion.
**Fix:** document the `AUTO_VOICE_REFRESH_REQUIRE_VISUAL_SPEAKER` guard in CONTEXT. Optionally relax it for the unambiguous single-visible case.

**34. settle_turn over-spends single-content-word callback premises** — bug / low / low · `intelligence/callback_engine.py:838-878` (specifically 861-865)
When a premise's only content word equals its topic (e.g. "loves astrophotography" → `{astrophotography}`), `settle_turn` falls back to `voiced = bool(matched)`, so ANY mention of the topic word marks the premise used + records a fire — even when the directive made Rex skip the callback. Burns a session slot and stamps a 7-day cooldown on a joke never told.
**Fix:** when `non_topic_premise` is empty, don't treat a bare topic echo as fired — require ≥2 topic-word hits or a structural cue, else release for retry. Best: bank premises that always carry a content word beyond the topic. Add a regression test.

**35. Legacy commands not in the action map unconditionally blocked during answer_to_rex** — bug / medium / medium · `intelligence/interaction.py:1242-1259` (`_legacy_command_blocked_by_dialogue`)
Any `command_key` not in `_LEGACY_COMMAND_ACTION_MAP` (wave_to, volume_up/down, set/query_personality, memory_correct_fact, memory_remember_fact) yields `decision=None`, so the breakout can never fire and the gate falls through to `return True`. Under an `answer_to_rex` frame these explicit commands are dropped to conversation; goal-relevant casualties: `wave_to` (fun gag) and `memory_remember_fact` ("remember I'm vegetarian" right after a Rex question is silently not stored). *(Closely related to rank 2.)*
**Fix:** when the command is not a contextual-reply candidate (decision None / key unmapped), return False (route normally) rather than default-blocking; or add the missing keys with breakout rules. Regression: "remember I'm vegetarian" / "wave at them" under an active reply-expecting frame.

**36. memory.people.find_by_voice is dead and lacks the margin guard** — dead_feature / low / low · `memory/people.py:183-210`
`find_by_voice` has zero callers (live voice path is `audio/speaker_id.py` with a margin/ambiguity guard) and accepts the single best match with no margin check, plus a stale '0.75' docstring. A future caller could re-introduce close-voice misattribution.
**Fix:** delete `find_by_voice` (no callers, no tests). If a helper is wanted, route through `audio.speaker_id`. Fix the stale 0.75 docstring to 0.50.

**37. Forget-by-target matches against fact source/category fields** — bug / low / low · `memory/forgetting.py:130-185` (`_delete_matching` field lists), `:73-91`
`_delete_matching` searches structural columns including 'source'/'category'. 'explicit' is the default source for preferences/interests, so "forget all explicit memories" / "forget anything secondhand" would wipe whole stores via substring matching. Unusual phrasing, but a latent data-loss path.
**Fix:** drop structural columns ('source','category','domain','preference_type') from the searchable field tuples in `_delete_matching`/`forget_memory_detail`/`fact_or_event_matches`; or denylist the source/category vocabulary in `target_terms`.

**38. Large block of legacy animation functions + speech_start/_speaking_loop are dead** — dead_feature / low / low · `sequences/animations.py:1021-1099, 1106-1158, 1312-1339, 1560-1586`
16 legacy top-level animation functions (nod/headshake/excited_burst/roast_pose/etc.) and the entire `animations.speech_start`/`_speaking_loop` path have no live callers — the expressive path is `play_body_beat` + `servos.speech_reactive_move`. The dead speech path would even fight servos for the head channels through the single serial lock if invoked.
**Fix:** delete the dead legacy functions and the speech_start/speech_stop/speech_level/_speaking_loop block (keep arm_wave, arm_idle, camera_pose, speech_activity_start/stop). Verify no orphaned constants remain; run the suite.

---

## 3. CONTEXT.md corrections (the doc is wrong here)

Confirmed claim-vs-code mismatches. Apply these to `CONTEXT.md` (and the cited docstrings).

| # | CONTEXT claim (location) | Reality (code) | Fix |
|---|---|---|---|
| C1 | L565/723: minor-holiday proactivity **OFF** by default | `config.py:5193` ships `HOLIDAY_PLANS_INCLUDE_MINOR = True`; the getattr-False fallbacks (`awareness/holidays.py:154`, `consciousness.py:7926`) are never reached | State minor-holiday plans are ON by default; set False to restrict to major. |
| C2 | L684/164: Ollama "when configured" sidecar | `LOCAL_LLM_ENABLED=True` + `OLLAMA_PRELOAD_REQUIRED=True`; `main.py:1123-1132` `sys.exit(1)`s if unreachable | State Ollama is a required boot dependency by default that fatally aborts boot; name the kill switch (`OLLAMA_PRELOAD_REQUIRED=False`/`LOCAL_LLM_ENABLED=False`). |
| C3 | L356-358: a named-then-departed voice resolves "straight to them" next session via `attach_person` | `interaction.py:2519-2549` reads `person_id` only into a log; returns anonymous label (finding #7) | State cross-session voice→name resolution is not yet wired; a returning named speaker re-enters as an anonymous slot unless re-recognized by face/biometric. |
| C4 | L501-503: onboarding "does NOT skip VIPs/creator" | `onboarding.eligible()` (`onboarding.py:89-96`) skips them unless `ONBOARDING_INCLUDE_VIPS=True` (default False, `config.py:3404`) | State onboarding skips creator/VIPs by default; include only via `ONBOARDING_INCLUDE_VIPS=True` (a fresh-DB testing flag). |
| C5 | L803: wave-back section (5 stale facts) | function is `animations.wave_back_gesture(half_period=...)` not `wake_word_ack_wave`; cooldowns **6s/4s** not 25s/8s (`config.py:2279-2285`); `POSE_ANALYSIS_INTERVAL_SECS=0.2` in `vision.pose`'s own loop not 2s; there's an escalation ladder (greet/silent/joke/giveup/ignore) + user-wave-speed mirroring + detect-latch/fire-when-free design | Rewrite L803 with the real function, 6s/4s cooldowns, 0.2s interval, the detect-latch + escalation ladder + speed-mirroring. Keep the note that the *detection* heuristic is single-frame (motion only feeds speed mirroring). |
| C6 | L743: wake-over-music `_threshold(dj_playing=True)` **raises** the bar | `audio/wake_word.py:220-222` **lowers** it (`max(floor, base - delta)`) so a music-masked wake still fires | Change "raises the bar" → "lowers the bar (drops the threshold)". |
| C7 | L767 + `topic_thread.py:9-16` docstring: conversation arc uses a "cheap local-LLM (Ollama)" | `CONVERSATION_ARC_BACKEND` defaults to `'openai'` (gpt-4o-mini, `config.py:240`); local runs only when set to `'local'` | Fix the topic_thread docstring to say default backend is OpenAI gpt-4o-mini (override via `CONVERSATION_ARC_BACKEND='local'`). |
| C8 | L494 + `onboarding.py:4`: `QUESTION_BUDGET_MAX_QUESTIONS=3`/90s | `config.py:3357` sets **5**; `question_budget.py:230` getattr fallback is a third value (2) | Update to 5/90s; align the getattr fallback to 5. |
| C9 | Gaze "look **UP** to think" for complex replies; SPEAKING ~0.50 on-target rhythm + include-sweep "live" | aversions are **down-only** (`min(0.0,pitch)`); think pose is look-down-and-aside; SPEAKING duty cycle + include-sweep suppressed/never-populated live (`gaze_engine.py`, `consciousness.py:10057-10085`) | Correct docstrings to "down-and-aside"; either wire or remove the SPEAKING-rhythm/include-sweep claims. |
| C10 | `active_speaker.py:17-21` docstring: "`update()` is a no-op until built up" | all layers implemented, live (`ACTIVE_SPEAKER_ENABLED` default True), consumed by voice attribution | Rewrite as live; drop "no-op" language. |
| C11 | README L22: web-search trigger phrases "editable in your user config" | live gitignored `user_config.py` predates web search and lacks the entire WEB SEARCH section; `setup_macos.sh:415-417` never re-copies an existing file | Re-sync `user_config.py` from `user_config.example.py` (or document that users must re-copy new template sections). Runtime is unaffected (config.py defaults stand). |
| C12 | `empathy.py:8` lists `acknowledge_then_yield` as a fused mode | never returned by `select_mode`/`force_mode`, absent from `_MODE_DIRECTIVES` (`empathy.py:1167,1199`) | Wire `force_mode` to emit it (+ a `_MODE_DIRECTIVES` entry) or remove from the docstring/aux maps. |
| C13 | `rex_db.py:47` kind list: `emotional`, `said`, `other` | real kinds: person_seen, made_laugh, animal, scene, conversation_summary, person_enrolled, game_played, visit_departure, boundary, celebrity, **emotional_checkin**, birthday_wish, milestone, celebration, reunion (`said`/`other` never written) | Update the comment to the real set; mirror in `setup_assets.py:868`; optionally point to `config.py EPISODIC_RECALL_KIND_WEIGHTS` as canonical. |
| C14 | L319/322: "margin-guarded accept tiers (hard 0.50, known floor 0.45, session-sticky 0.60)" | session-sticky tier intentionally drops the margin guard (requires match to recently-engaged person instead); only hard + known-floor are margin-guarded (`interaction.py:16508-16518`) | Reword: session-sticky relies on recent-engagement continuity in place of the margin guard. |
| C15 | L336-338/733: voiceprint refresh "gated on raw_best_id == person_id" only | second default-on guard `AUTO_VOICE_REFRESH_REQUIRE_VISUAL_SPEAKER` also requires the visual active-speaker latch (finding #33) | Add the visual-speaker guard to the description; note refresh is retried per-turn. |
| C16 | `onboarding.py:195` header "LLM depth follow-up (local qwen sidecar...)" | `generate_followup` runs on the main OpenAI conversation model via `llm.generate_curiosity_question`; no qwen in this chain | Change header to "(main OpenAI conversation model via llm.generate_curiosity_question; templated fallback)". |
| C17 | L549 / `config.py:3409`: onboarding step TTL "a stale flow self-expires" (implying inactivity) | `onboarding_flow_active` measures from `created_at` (wall-clock since armed), not `asked_at` (sliding), so a long active burst can hard-expire mid-exchange (`interaction.py:8978-8982`) | After fixing code to slide on `asked_at`, keep inactivity wording; until then note TTL is wall-clock-since-armed. |
| C18 | `docs/supervisor.md:14,40` + README L127: supervisor "launches main.py / the full controller" | `rex_supervisor.py:362-370` hardcodes `main.py --gui` on every voice launch (headless fallback only if PySide6 missing) | Document the always-`--gui` launch + headless fallback, or gate `--gui` behind `REX_SUPERVISOR_GUI`. |
| C19 | `memory_banks.py:212-214`: INTERACTION_PAUSED is "the TRUE pause: no responses, no wasted LLM calls" | `submit_text` (text path) doesn't check it, so typed messages during a pause still reply + write memories (finding #8) | After fixing the code, the docstring becomes accurate; until then it overstates coverage for typed input. |
| C20 | Memory model implies emotional events consistently carry recency for check-in/greeting gates | session-end consolidation re-extractor inserts emotional events with `recency='unknown'` (the LLM consolidation schema doesn't request recency), so a consolidation-only path leaves a recent loss check-in-inert (`interaction.py:12750-12760`, `emotional_events.py:184`, `llm.py:2367-2368`) | Make consolidation resolve/pass recency, or document that consolidation-path emotional events default to 'unknown' (check-in-inert) by design. |
| C21 | L811 callback paragraph (accurate, one clarification) | the legacy comedy "callback" mode (echo a recent bit, `comedy_modes.py:111-115`) is a SEPARATE feature from the banked-callback engine; a reader could conflate them | Optionally add a half-sentence distinguishing them. No accuracy change required. |

---

## 4. Subsystem health snapshot (one line each)

- **interaction-identity** ✅ mature; voice-primary hierarchy correct & tested. Gap: cross-session voice→name half-wired (#7); a couple of doc/threshold discrepancies; minor wake/quiet turn-gate mapping bug.
- **interaction-flows** ✅ well-guarded. Gaps: dead idle-banter hot-take branch (#24); onboarding pivot over-fires (#14); stale onboarding CONTEXT claims.
- **consciousness-proactive** ✅ mature, well-gated. Arbitration leaks: governor drops force_salient (#9), people_roast/memory_musing at floor (#10), dedupe-before-choice (#16), smile/presence bypass governor (#31).
- **routing** ✅ well-structured, evidence policy consistent. Biggest issue: answer_to_rex frame swallows fun commands (#2, #35); roast food-guard article bypass; dead intent-classifier LLM fallback.
- **llm-prompt** ✅ mature, hybrid model rollout correct. Risks: web search on unverified gpt-5.4-mini (#12); dead surprise pre-beat (#19); Ollama hard-dependency doc gap.
- **persona-shaping** ✅ roasting NOT over-suppressed (default 'normal', engage-first still permits a roast). Edges: interest-poisoning regex (#4); visual-suppression ordering on sensitive turns; POV guard never arms (#18).
- **social-layers** ✅ caring/roast machinery fires. Main issue: `gentle_probe` omitted from tender sets so masked distress can be roasted (#11); plus doc discrepancies.
- **callbacks-premise** ✅ well-engineered, 39 tests pass, intentionally conservative. Weaknesses: soft third-party wall (#20); background-thread freshness misses (#23); 'normal'-only roast gate (#17); settle over-spend (#34).
- **memory-core** ✅ writes/reads/migrations solid; ephemeral-filter, birthday-window, visit-milestone, greeting-count ordering all correct. Edges: postpone-as-cancel (#15); over-broad forget (#37); dead find_by_voice (#36).
- **memory-episodic** ✅ recall genuinely enabled & wired; unified retrieval consumed; semantic off & degrades safely. One real defect: per-boot dedup sets (#3). Secondary: shutdown-summary writes recall-dead row; non-scene episodes grow unbounded.
- **audio** ✅ mature; software-AEC correctly off; expressive TTS wired; watchdog/debounce real. Main issue: scene analyzer during DJ music (#5); wake-over-music doc inverted (C6).
- **vision** ✅ mature; brow-furrow baseline, phantom-face guard, pose Tasks-API migration correct. Issues: CONTEXT L803 badly stale (C5); 2 dead public functions; startle-animal unreachable (#29); multi-person attribution (#13).
- **features-games** ⚠️ orchestration solid but **two of five games broken** (#1 20Q, #6 Word Assoc); trivia content thin + difficulty mislabel (#21, #22).
- **hardware-motion** ✅ defensive, graceful no-op without hardware. Rot: dead legacy animation block + speech_start path (#38); gaze SPEAKING/include-sweep dead (#27); CONTEXT wave-back stale (C5); gaze "look up" doc (C9).
- **startup-gui-state** ✅ mature; **no-audio guarantees genuinely hold** (verified: no mic/TTS/ElevenLabs/playback leak); config re-derive tail, GUI-first startup, single-instance flock all correct. Gaps: supervisor `--gui` doc (C18); pause-bypass for text (#8).
- **config-flags-awareness** ✅ override mechanism + re-derive tail work. Issues: holiday-default discrepancy (C1); stale user_config.py (C11); a couple dead helpers; vestigial `STARTUP_BOOT_TTS_LINE[0]` re-derive.

---

## 5. Checked and CLEARED — do not re-chase these (refuted findings)

Adversarial verification refuted 5 plausible-looking findings. They are **correct as-is**:

1. **Comedy on `answer_ack` turns is fine.** `comedy_modes.select_mode` short-circuits to `"straight"` for `purpose in {"closure","repair","identity","answer_ack"}` (`comedy_modes.py:135-141`), so self-absorbed bits are unreachable there. Adding `answer_ack` to the interest-turn condition would be a *regression*.
2. **Rex-POV `["any"]`-tagged seeds rotate evenly.** The anti-repeat machinery (`_used_ids`, cycle restart only when all seeds used) gives every seed identical long-run frequency; they're merely ordered later within a cycle in steady context. No CONTEXT discrepancy.
3. **20Q `"pass"/"skip"` do NOT end the game.** Neither word contains a final-guess substring; both fall through as normal questions. A working graceful exit exists (`stop`/`quit`/`end` → `_20q_stop`). (The real 20Q bug is #1, the over-broad `"is it"` guess detector.)
4. **`leds_head` "Uno" docstring is CORRECT.** It matches the flash FQBN (`arduino:avr:uno` at `setup_macos.sh:1672`, `diag_head.ino:20`). The misleading artifact is the `head_nano` *directory name*, not the docstring. Do NOT change the docstring to "Nano".
5. **Weather fog→rain mapping is complete.** All WWO fog/mist codes (143/248/260) are captured at `chronoception.py:42-43`; none leak into the rain range. (Minor unrelated nit: frozen/mixed-precip codes are inconsistently bucketed between rain/snow — low-impact banter flavor only.)

---

## 6. Review provenance

- **Scope:** all non-test modules across `audio/ awareness/ features/ gui/ hardware/ intelligence/ memory/ vision/ sequences/ evals/ tools/ utils/ firmware/` + top-level (`main.py`, `state.py`, `world_state.py`, `config.py`).
- **Process:** 16 parallel deep subsystem reviewers → adversarial per-finding verification (independent agents, default-skeptical) → cross-cutting synthesis. 109 raw findings, 104 confirmed, 5 refuted.
- **First-hand re-confirmed by the lead reviewer:** findings #1 (20Q), #4 (interest regex), #5 (DJ-scene), and both red tests (§1).
- **Not exhaustively re-run:** on-device/hardware behaviors (servo motion, ReSpeaker AEC, camera FPS) were reviewed at the code level only — validate hardware-sensitive fixes on the robot.
