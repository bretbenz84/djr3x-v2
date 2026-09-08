# Jeopardy live reliability review — 2026-09-07

Review baseline: `9e93a83`. Production behavior changed in this review only for the separately requested answer-music fix. The other changes below are proposals.

**Recommendation: resolve the scoring/delivery problems and add host recovery before another friends-and-family session.** These are reproducible software failures, not things the group should have to diagnose. Passing tests for individual past incidents did not exercise all the interactions among capture, timers, speech delivery and game state.

## Evidence and scope

Read the annotated 20:58 conversation, its detailed runtime log, game engine, answer/selection/wager parsers, interaction routing, speech queue, GUI and clue bank. Ran additional adversarial probes with network, audio and serial access blocked, MLX disabled and temporary databases. Inputs below are synthetic reproductions unless explicitly labeled as logged evidence. No robot movement, microphone capture, hosted model calls or live group test was performed.

Standing behavior stays intact: any voice can answer the open player's turn; first names and stored aliases resolve players; game play does not require voice enrollment; routine category/value readback stays off in GUI mode. Do not solve turn ownership by putting voice ID back in charge.

## Fix implemented: answer music follows the clock

The 20:58 runtime log repeatedly records `interrupted game audio for player speech`, followed by answer-clock deferrals. At 21:04:34 the theme was interrupted; the clock continued deferring and reached its hard ceiling at 21:04:49. Independently, the theme was capped at 12 seconds while the answer clock could allow another 10 seconds for in-flight speech.

The regular, Daily Double and rebound answer windows now own a looping playback item. It starts after the spoken clue is reported delivered, survives clock deferrals and ignored chatter, and ends before answer feedback, a repeat request or the time-up chime. Clearing/switching games or requesting stop cancels it. An expired queued bed cannot start later, and a spoken control cannot wait behind an indefinite loop.

On the ReSpeaker hardware-AEC path, incidental VAD detections no longer cut this bed off. Without hardware echo cancellation, playback still pauses for capture to preserve intelligibility; if the turn remains open afterward, music resumes without extending its deadline. This fallback cannot provide uninterrupted music while using software playback suppression. The separate Final Jeopardy flow remains a proposal below; it currently has no answer-expiry clock to synchronize with its 30-second theme.

Validation: `tests/test_jeopardy_music_clock.py` exercises clock ownership, simulated playback past the old cap, chatter, deferrals, correct answers, repeats, rebounds, Daily Doubles, stale timers/items, interruption recovery and game stopping/switching. Playback is mocked; these checks establish software behavior, not acoustic quality in the room.

## 1. A late answer can still score for the wrong player — highest priority

**Reproduced:** set an open rebound for B with `timeout_rebound.from_idx = A` and `awaiting_prompt_delivery = True`, then submit “What is Paris?” for a Paris clue. The engine gives **$200 to B**, even though its documented grace policy says A still owns that answer until the announcement finishes.

`_jeopardy_handle_answer` calls `_jeopardy_cancel_timeout`, which removes `awaiting_prompt_delivery`, before checking that same flag to apply grace. The grace branch therefore never executes on this path.

There is also a broader timing risk: `handle_input` receives text/audio/person ID, but no game/clue/attempt identity captured at speech onset. Timer callbacks mutate state on another thread, while the answer handler does its work outside the game lock. An answer that begins on A's turn and finishes transcription after B's announcement can be applied to B. Two captures of one repeated answer can cross a turn boundary too.

**Proposed fix:** repair the broken grace check, then bind each admitted capture to a game, clue and attempt at onset. Serialize score/turn transitions and reject stale or duplicate capture IDs. An in-flight answer retains its original turn; elapsed time must not silently transfer it. This needs a small game event coordinator, not another conversation-brain rewrite.

**Acceptance:** delayed ASR, repeated callbacks, a timeout during grading and back-to-back answers never score the wrong attempt or score twice. Unknown and incorrectly identified voices still work.

## 2. Speech failure can strand an unheard clue — highest priority

**Reproduced through the interaction handler:** selecting a square with `_speak_blocking` returning false consumes the square, sets `phase=awaiting_answer`, leaves `awaiting_prompt_delivery=True`, and creates no answer timer. The GUI can show a clue nobody heard while the voice flow silently waits.

**Also reproduced:** a canceled/dropped timeout announcement, represented by a completed `DoneEvent` with `played=False`, still arms the rebound clock. `_jeopardy_schedule_post_timeout_rebound` checks that the event is set, not that the announcement played.

**Proposed fix:** make clue delivery an explicit state with confirmed delivery results. On failure, retain and retry the same square; show a visible retry control. Only the matching, successfully delivered announcement can open a turn. A canceled announcement must not start a clock. Define the no-audio GUI case explicitly: visible text can satisfy delivery when that mode is selected.

**Acceptance:** inject failed synthesis, canceled playback, a busy output gate and stale completion callbacks. No square is lost, no invisible clock runs, and retry delivers the original clue once.

## 3. Lenient matching can confidently award a wrong answer

**Reproduced with the local matcher, before any model call:**

| Spoken/transcribed answer | Expected | Current result |
|---|---|---|
| “not Paris, London” | Paris | Correct |
| “Paris or London” | Paris | Correct |
| “Paris, no, London” | Paris | Correct |
| “pepper” | salt and pepper | Correct |
| “registration” | license and registration | Correct |
| “Queen” | Dancing Queen | Correct |

Token/partial matching ignores the structure of corrections and alternative guesses. The surname shortcut is also applied to multiword answers that are not people's names. The new pronunciation matching addresses rejected homophones, but these older acceptance shortcuts still bypass the judge.

**Proposed fix:** extract a single final answer while preserving negation and self-corrections. “Paris—no, London” means London; unresolved alternatives should prompt “Which one?” Require all parts of multipart answers on every matching path. Restrict surname matching to person answers. Keep exact homophones and spelling variants fast; send structurally ambiguous matches to clarification instead of automatic credit.

**Acceptance:** an adversarial corpus covers both false rejections and false acceptance: homophones, numbers/years, related names, missing components, titles, alternatives, negation and self-correction.

## 4. Ambiguous category names silently choose a square

**Reproduced:** with WORLD HISTORY and AMERICAN HISTORY both available, “history for 200” silently consumes WORLD HISTORY. The fuzzy selector chooses the first best score without requiring a margin over the runner-up. Bare dollar amounts can also inherit the last category, including one selected on a previous player's turn.

**Proposed fix:** require a unique category match or a clear margin. Ask only the useful clarification: “World or American history?” Keep the pending dollar amount. Use explicit “same category” or current selection context for reuse. Highlight ambiguous choices in the GUI; do not restore repetitive readbacks for clear selections.

**Acceptance:** reorder the board and the same ambiguous input still asks for clarification. An unavailable or uncertain square never consumes a different one.

## 5. Final Jeopardy has a different, less protected answer flow

**Reproduced:** with the judge mocked to return its supported `none` verdict, “Can you hear me?” is locked as an incorrect Final answer and the queue advances. Regular clues handle that verdict as a non-answer; Final does not.

**Verified in code/snapshot:** Final clue and answer queue are absent from the GUI snapshot. The panel renders clues only for `awaiting_answer`; `final_answer` falls through to “CHOOSE A CATEGORY AND VALUE.” The active answerer follows `final_queue`, but the displayed player index can remain from an earlier phase. Daily Double wagering has a similarly misleading phase prompt. Final and wagering do not have the regular answer timeout, so silence can stall them indefinitely.

**Proposed fix:** share answer validation across regular, Daily Double and Final attempts. Publish explicit phase data for the clue, active answerer, wager bounds and clock. Add bounded waiting with host skip/pause controls. Give Final an explicit timed thinking phase whose music and display share its clock, then collect answers. For fairness, optionally collect Final answers privately in the GUI; the current sequential spoken method lets later players hear earlier answers.

**Acceptance:** complete an entire game through Final using GUI snapshots and simulated delivery, including silence, non-answers, repeats, zero-score players and failed playback. No phase displays the wrong task or player.

## 6. A spoken wager correction can radically change the bet

**Reproduced with bounds $5–$2,000:**

| Input | Parsed wager |
|---|---:|
| “five hundred no two hundred” | $5 |
| “not all in, just five hundred” | $2,000 |
| “-500” | $500 |

**Proposed fix:** preserve negation; resolve a clear correction to its final amount; reject multiple unresolved amounts. Repeat the actual interpreted wager and allow correction before revealing the clue. Confirm an ambiguous all-in request. Never silently turn a negated “all in” into the maximum.

**Acceptance:** numeric/spoken amounts, self-corrections, negatives, minimum/maximum boundaries and ambiguous all-in phrases preserve the intended stake or ask for clarification.

## 7. Names and nicknames can create duplicate seats

**Reproduced with mocked name resolution:** “Jeremy, JT” resolves both entries to person 4, but `_jeopardy_prepare_players` creates two separate Jeremy players with separate scores. Raw-string deduplication is insufficient after aliases resolve. Two different people sharing a first name can also become hard to distinguish in the GUI.

**Proposed fix:** deduplicate known players by resolved person ID; require distinct display names for different people. Confirm an editable roster once before dealing the board. Keep genuinely unknown players as session players until an intentional identity action, rather than persisting every garbled roster name as a new person.

**Acceptance:** first name + nickname + full name for one person yields one seat. Ambiguous first names ask which person, while genuinely new guests can play immediately.

## 8. Game words can still collide with robot controls

The recent category-start-command fix protects some start-game commands. `_game_escape_command` still recognizes other global controls during game input. Categories or answers involving “stop,” “sleep” or similar control-shaped words are candidate collisions; this is a code-path risk, not a claimed new field incident.

**Proposed fix:** reserve explicit addressed phrases for non-safety robot controls while a game owns the turn, and give actual category/answer text first claim on ordinary words. Preserve immediate physical stop behavior. Test routing against real category and answer strings, not just handpicked command examples.

## 9. Some clue-bank entries depend on missing media

**Verified examples that pass `_valid_clue`:**

- DUNGEONEERING, 2001-05-08: “The monster emerging from a dungeon here was seen in this 1931 movie.” The clue references unseen material.
- YELLOWSTONE NATIONAL PARK, 2004-10-19: the entry includes a stage direction describing all four Clue Crew members listening to a howl and asks players to identify the animal from its call. Rex has no accompanying media to play.

A broad keyword filter would overcorrect: “If you listen to CDs…” is a perfectly playable text clue. Historical “now/current” facts are another audit target; the source air date exists but is not included in the judge's clue context.

**Proposed fix:** validate and flag the bank offline for required media, spoken-only ambiguity and date-sensitive facts. Exclude or adapt unusable clues before dealing. Give the host a no-penalty replacement for a defective clue. Preserve the source date when judging historical questions.

## 10. The host has no practical way to repair a bad ruling

The inspected GUI paints controls/instructions but has no click/key handlers for undoing a ruling. There is no score transaction history, ruling appeal, general game pause or saved board to resume after a restart. Repeating the previous answer can merely trigger “already scored.” That turns one recoverable transcription mistake into an argument with the robot.

**Proposed fix:** add host controls for **undo last ruling, accept this answer, retry clue, skip defective clue, pause/resume and correct roster**. Keep a score/turn transaction journal so undo restores rebounds and wagers too. Save after committed transitions so restarting Rex does not destroy the session. Replayed captures must not score again after recovery.

The judge also lacks a game-specific request timeout; its connectivity wrapper only short-circuits known outages. Add a short judging deadline, visible “checking answer” state, retained original transcript/audio, and a recoverable service-error result. A failed model call must not force repeated guessing or disguise itself as a wrong answer. This is about flow reliability, not API cost.

## Proposed implementation order and release gate

1. **Protect the round:** repair turn ownership/grace, serialize transitions, bind capture IDs, and handle successful/failed clue delivery explicitly.
2. **Make mistakes recoverable:** add the host repair controls and score journal; apply stricter correction/multipart matching and unambiguous category/wager parsing.
3. **Finish every phase:** bring Final/Daily Double GUI and clocks onto the same contract, resolve duplicate seats, audit control collisions and filter unplayable clues.
4. **Rehearse offline:** run complete sessions with prerecorded answers and adversarial event scheduling. Inject delayed/duplicated ASR, speech during the last clue syllable, pauses mid-answer, laughter/chatter, two overlapping voices, disconnects, canceled playback, wrong voice labels and a restart after scoring. Check scores, turn ownership, GUI state and audio lifecycle after every transition.
5. **Solo hardware rehearsal:** one person plays all seats through both rounds, Daily Double and Final, exercises each repair control, and verifies capture/music/chime timing at the usual distance and speaker volume. Only then schedule another social game.

Do not use a large passing unit-test count as the release gate. The gate is a complete playable session under injected failures: every admitted answer belongs to exactly one attempt; every score can be explained and repaired; every phase has a visible next action; every clock follows actual delivery; and no service failure forces the host to restart the whole game.
