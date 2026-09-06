# Lean Brain restructuring plan

Code review: 2026-09-04. Status (2026-09-05): incremental implementation in
progress. Initial slices exist across phases 0–5; this is not a claim that all
phase acceptance criteria have passed. See the phase notes and the continuation
below. Owner's direction: behavior over measurement.

### Implementation continuation (2026-09-05)

- Phase 0: `utils/runtime_report.py` adds allowlisted effective configuration,
  actual imported model handles, ownership, peak RSS and CPU time to turn traces.
  Peak RSS is explicitly not current RSS or system memory pressure.
  `tools/production_replay.py` exercises `submit_text()` with injected model and
  speech transports, temporary databases and blocked hardware/network access.
  `evals/run_quality_eval.py` now uses the production `_reply_token_stream` seam.
- Phases 1/2: one context-local retrieval deadline covers Lean's prompt build;
  only one query embedding may be awaited. Candidate misses use keyword scoring
  immediately and queue bounded, expiring prewarm work; fact/interest writes also
  prepare candidates. Foreground ASR/local TTS prevent admission of new optional
  embedding work. Existing GPU/HTTP requests cannot be preempted. Cache epochs
  prevent late prewarm commits after reset; configured model changes clear vectors.
  Conversation summaries now reject stale session epochs and transcript revisions,
  including speaker corrections. Worker retirement is atomic with the dirty flag.
- Phase 2B: uncertain transcript entries are unlearnable immediately, durable
  conversation logging omits guessed identity, summaries retain the uncertainty,
  and Lean replies do not retrieve the guessed person's personal context.
- Phase 4: firmware completion of a safety-shortened turn is a partial goal;
  no compass measurement is invented when verification is unavailable.
- Phase 5: queue delivery requires a signal from the buffered/streamed/local audio
  sink. Interrupted or failed sentences are excluded from delivered text, including
  no-audio/whole-reply and fallback branches. Generation validity is checked after
  synthesis and at the sinks; interrupted provider streams are closed.
  This is sentence-level truth, not word-level alignment of partial audio.
- Owner behavior decision: **finish the pending reply, then handle new speech**
  when someone speaks while Rex is thinking. `GAP_MERGE_ENABLED=False` is the
  default; explicit playback barge-in remains. `turn_coordinator.pending` retains
  up to four later completed captures from the recovery scan in order, with a
  60-second/session expiry and original capture times. Queued speech cannot answer
  dialogue frames created after its capture; Lean receives that ordering note.

Still outstanding: fully concurrent capture/response ownership (the pending-input
adapter still uses the existing recovery scan); authoritative utterance-aligned
attribution and mixed-speaker capture evaluation; complete per-target pending
exchange ownership; minimal agenda preparation; unified action narration and goal
verification; token/pressure/cost measurements and live acceptance scenarios.
The heading-alternative flag remains OFF. No live robot validation was performed
for this continuation. Do not remove recovery branches or enable alternative
physical turns on the strength of offline fixture tests.

Sources: README.md, CONTEXT.md, CLAUDE.md, and the implementation seams cited
below. CLAUDE.md adds important test-isolation and real-hardware constraints;
its historical failure inventory must be rechecked rather than assumed current.

## Objective and constraints

Make Rex follow conversations, choose appropriate proactive behavior, and adapt to
action failures without making ordinary replies slower or requiring new hardware.
The target is the existing Apple Silicon M2 with 16 GB unified memory and the
existing hosted conversation voice. Preserve Rex's personality and physical expression.

Hard design constraints:

- Keep exact, context-checked command fast paths. Regex misses go to the existing
  LLM tool caller. Do not add a universal local or hosted classification/planning call.
- Ordinary conversation uses one primary generation call; context preparation must
  not require a preceding model call. Existing exceptional search/vision/offline
  paths are measured separately rather than hidden inside this claim.
- Preserve first-sentence TTS splitting and streamed audio. No mandatory JSON plan,
  reasoning explanation, or metadata preamble before the first spoken sentence.
- No larger local model, additional always-resident model, or hardware purchase.
- Replace existing background work where possible. Do not add a continuous paid
  “thinking loop.” Track actual requests and tokens, including unsuccessful work.
- Python/firmware remain authoritative for identity/write permissions, physical
  limits, sensing validity, stop, manual ownership, and execution truth.
- Implement in small reversible slices. A replacement takes ownership from its
  predecessor; the result must not be another permanent parallel brain.

## Findings that determine the plan

These are source findings, not fresh timing measurements. Config defaults are not
proof of effective robot settings or loaded models. The current local logs do not
establish today's hardware latency.

| Finding | Evidence and implication |
| --- | --- |
| The separate JSON-prose action-router model fallback is already disabled by default. | `config.py:6167`, `action_router.decide()` in `intelligence/action_router.py`. Preserve the deterministic routes and main-call tool fallback; do not budget a fictional always-on router call as a saving. |
| General local turn classification is off. The normal intent path is deterministic. | `CONVERSATION_TURN_CLASSIFIER_ENABLED=False`; `interaction.py` calls `intent_classifier.classify_deterministic()`. The local model is a collection of helpers, not a universal routing stage. |
| A conversation arc already runs in the background, using OpenAI by default. | `intelligence/topic_thread.py` refreshes a coalesced summary; `llm.assemble_system_prompt()` injects it. Lean `_system_prompt()` does not directly include that arc. Reuse useful state before commissioning another summarizer. |
| Lean's primary history is eight transcript entries by default. | `lean_brain._messages()` and `LEAN_BRAIN_TRANSCRIPT_TURNS`. A short window needs explicit continuity; increasing the transcript alone increases prompt cost and still does not establish action truth. |
| Memory ranking can perform synchronous local inference. | `memory/retrieval.py` calls `memory.semantic.relevance()`, which calls `_topic_vector()` and `_embed_candidate()`. Cache misses request embeddings inline, potentially for multiple candidates. A per-request timeout is not a total context budget. |
| Lean still enters legacy agenda/comedy preparation. | `interaction._stream_llm_response()` calls `conversation_agenda.build_turn_plan()` before choosing Lean. `_plan_branch()` can invoke `plan_intent.classify()` and its local specificity check. Some outputs still support callbacks/delivery; trace consumers before removing work. |
| Optional expression work can wait before speech. | `_stream_and_speak_sentences()` joins the surprise worker for up to `SURPRISE_STREAM_JOIN_SECS` (default 0.25 s). Streaming self-emotion classification is already asynchronous; do not misidentify it as a mandatory first-word wait. |
| Proactive choice is preselected in Python. | `_maybe_lean_impulse()` constructs a priority ladder and `lean_brain.consider_initiating()` selects an instruction via another ladder. The model sees recent conversation, but usually not competing eligible opportunities. |
| Main-call tools are a dispatch seam, not a general result/replan loop. | `lean_brain.stream_reply()` raises `ToolCallRequested`; `_execute_tool_routed_action()` dispatches existing handlers that often speak themselves. Extra calls are ignored after the first, and prose can win over tools. Do not silently turn this into unrestricted agency. |
| Motion results are not yet a uniform goal/outcome contract. | `motion_controller.last_refusal()` is a short-lived global record; `_suppressed()` may queue speech. `motion_swing.check_turn()` can shorten a turn as well as refuse it. A completed shortened command may not achieve the original heading. |
| Existing safety has limits that planning cannot wish away. | `firmware/djr3x_motion/safety.cpp`, `motion_controller._tof_sensing_fault()`, and `motion_swing.py`: host sensing gates matter, radial coverage is sparse, and cliff sensing is unavailable. Passing a current swing calculation is not proof of every point in a long sweep being observed. |
| Evaluation has drifted from the main reply entry point. | `evals/run_quality_eval.py` reconstructs generation and calls `llm.stream_response()`, which can use Lean's directive path; production replies use `_reply_token_stream()`/`lean_brain.stream_reply()`. Share the production seam instead of maintaining another implementation. |
| Group attribution already uses DoA, but evidence is not uniformly utterance-bound. | `_note_voice_bearing()` keeps `_last_voice_bearing` when a new read returns None; `_recent_voice_bearing()` permits the last result for 12 s by default. `_voice_bearing_face_match()` compares it with current faces/neck pose. This creates a stale-evidence risk, not proof of the cause of any particular field mistake. |
| Current speaker signals favor one winner per captured segment. | `audio/speaker_id.py` embeds a buffer; `flex_doa.bearing_between()` chooses a dominant bearing cluster; `vision/active_speaker.py` maintains a recent winner latch. Multiple sequential speakers in one capture, short interjections, and overlapping speech need explicit treatment. |

Owner hardware context: the robot now has a four-microphone directional array and
three base radar modules shielded with grounded foil. That is reported hardware
configuration, not a verified calibration or a guarantee that reflections/ghosts
are gone. Existing radar transport describes its output as bearing hints rather
than identity evidence. Do not assign a person's name from a radar target alone.

## Target ownership

Keep a single process and existing service boundaries initially. Use typed records
and a small in-memory state owner; no message broker, new database, or framework.

1. **Observation producers** publish what was heard/seen and what changed, with
   timestamps, source, identity confidence, and relevant body pose. They do not
   independently decide what conversational line to say.
2. **Conversation state** maintains current participants, turn history, outstanding
   questions, active subject, corrections, commitments, and recent action outcomes.
   Deterministic facts update immediately. Model summaries are advisory and versioned.
3. **Lean Brain** receives a compact snapshot. The ordinary reply call chooses prose
   or an allowed tool. The existing proactive call chooses silence or one opportunity
   and writes its line in that same call.
4. **Action executors** validate requests, execute, and return correlated results.
   They do not claim completion at dispatch or retry outside the current goal.
5. **Turn/output coordination** owns cancellation and which generation may speak.
   Speech-start/end feedback updates what was actually delivered.

Suggested new seams are `intelligence/conversation_state.py`,
`intelligence/brain_context.py`, and an `ActionResult` record beside the existing
action contracts. These are proposed names, not an instruction to scatter new
abstractions across the repository. Extract a turn coordinator only after the
state and cancellation contracts are proven.

### Context contract

Internally typed data can be rendered as concise text; JSON in the prompt is not
necessary. Each snapshot has session ID, turn ID, revision, and creation time.

| Field | Contents and authority |
| --- | --- |
| Participants | Current speaker, visible people, recently present people, identity uncertainty; visibility is not presence. |
| Conversation | Recent verbatim messages plus the arc's covered-through turn ID; current topic is an advisory interpretation, not a permanent fact. |
| Pending exchanges | Question text, intended addressee, actual speech-start/delivery state, answer/expiry state. |
| Corrections and boundaries | Recent explicit corrections and active limits; these outrank stale summaries and recalled assertions. |
| Current activity | Game/intro/search/movement and who owns it; user request and interruption status. |
| Action outcome | Requested goal, attempted command, accepted/running/completed/blocked/aborted/timed-out status, measured progress, reason, freshness. |
| Relevant memory | Bounded selected records with provenance and dates; absent retrieval means unknown, not “never told me.” |
| Perception changes | Significant recent changes, their evidence, age, and whether Rex's own motion explains them. |

Do not dump raw telemetry or all database rows into every reply. Motion detail is
included when acting or discussing motion. Hard boundaries and pending interactions
take priority over decorative context. Token budgets must count persona, tools,
history, and new state together. Replace duplicated blocks before expanding totals.

Commit state updates through one owner. Maintain existing DBs as durable sources;
initial adapters read existing frames, ledgers, and stores. Migrate pending-state
ownership one domain at a time and remove the old writer for that domain.

## Delivery phases

### 0. Establish the actual baseline and production replay seam

- Add a sanitized effective-config report and loaded-model/resource report without
  importing startup or exposing secrets. Record feature ownership, not just flags.
- Extend existing latency/character-loop traces with a shared turn ID: last human
  voiced sample, endpoint decision, ASR start/end, attribution/routing, context
  build, each model request, first token, first speakable sentence, TTS request,
  first PCM, audible playback start, and cancellation. Separate queue delay from
  synthesis and model time. Distinguish a chirp/ack from a meaningful answer.
- Count sidecar/embedding/cloud requests by purpose, prompt/output tokens, cache
  hits, fallbacks, and cancelled work. Log process memory, memory pressure/swap
  deltas, and inference overlap on the target machine.
- Reuse the production reply entry point with injected input, transport, and audio
  sinks. Keep `submit_text()` as the text entry; do not invent an alternate brain.
- Baselines: ordinary chat, contextual answers, direct commands, personal recall,
  first-turn cold cache, mixed-party conversation, and speech during generation.
  Search, vision, motion planning, and offline fallback get separate distributions.

Acceptance: reproducible stage attribution and prompt capture sufficient to explain
what the model knew on a failed turn. Existing behavior is unchanged. Historical
timings in comments are hypotheses, not the baseline.

Status 2026-09-04 — first slice shipped: `utils/turn_trace.py` stage stamps and
per-turn model-call counts in `[character_loop]` (`stages` / `calls` / `context` /
`cancel_reason`), transcript entries carry `turn_id` + `ts`. See CONTEXT.md
"Latency And Telemetry". Still owed in this phase: token counts (needs
`stream_options` on streamed calls), the effective-config / loaded-model report,
process memory and pressure sampling, and the injected-sink replay harness on
`submit_text`.

### 1. Remove avoidable waiting before adding richer behavior

- Change optional surprise delivery to consume a result only if already available;
  do not join before first audio. Preserve expression through existing tags,
  inexpensive defaults, and timely background results.
- Introduce a Lean-specific minimal preparation path. Trace agenda outputs used by
  callback claims, delivery, and safety; retain those necessary consumers while
  removing unused classic directives and plan-specific local classification from
  ordinary Lean replies. Do not delete the classic fallback in this phase.
- Precompute candidate embeddings on memory changes or idle work, with keys tied
  to record content and embedding-model version. Deletions/corrections invalidate
  cached items immediately. No candidate-by-candidate inference during prompt build.
- Apply one total retrieval budget. A query embedding may run once, concurrently
  with other necessary preparation; use it only if ready by the context deadline.
  Cached/keyword fallback must be immediate. Direct recall questions need richer
  deterministic lookup and quality tests so speed does not create permanent amnesia.
- Start with a simple shared permit for optional local inference: coalesce pending
  work, expire stale jobs, and avoid launching it while ASR or local TTS needs the
  machine. Do not claim a Python thread can preempt an already-running GPU request.

Status 2026-09-04 — shipped: surprise join → 0 s, plan-intent local confirm off
under Lean, one inline embedding budget per retrieval with background prewarm
(`memory/semantic.turn_budget`). Not done: the Lean-specific minimal agenda path
(frame/comedy mode are still built; they feed TTS voice settings and the audio tag).

Acceptance: no new model calls on ordinary replies; before/after p50 and p95 show
no regression and identify actual savings. No degradation of recall, safety gates,
first-sentence voice, or interruption recovery. Do not blindly shorten endpoint
silence: the history records a tradeoff with cutting people off.

### 2. Give Lean consistent context using existing work

- Adapt current world state, dialogue frames, decision ledger, and conversation
  records into the context contract. All snapshot reads are local and bounded.
- Connect the existing arc to Lean as a compact advisory summary, with an explicit
  covered-through turn ID. Keep recent exact messages after that point. Current
  corrections and pending exchanges override old arc content.
- Harden background commits with session epoch and input revision checks; transcript
  length alone cannot safely distinguish every reset/race. Clear state by identity
  and session boundaries rather than leaving pending callbacks to mutate new state.
- Consolidate duplicate person/boundary/recent-topic sections. Record context size
  and compare model request latency as well as Python assembly time.
- Reuse and, if necessary, reduce the existing arc refresh schedule. Do not add a
  second summarizer or assume the small local model can replace the hosted summary
  at equal quality. Evaluate that only as a separate, measured cost experiment.

Acceptance: Rex can follow a reference beyond eight messages, acknowledge a recent
correction, know which question remains unanswered, and distinguish “I looked away”
from “they left,” without waiting for a fresh summary or adding a serial call.

Status 2026-09-04 — shipped: `intelligence/brain_context.py` +
`intelligence/conversation_state.py` (arc with covered-through turn_id, widened
verbatim window, corrections, body-action outcomes, pending questions per target,
presence notes) in both Lean calls. See CONTEXT.md "Conversation Voice". Owner's
call: skip further measurement; behavior first.

### 2B. Make group attribution a first-class contract

Priority: alongside phase 2, before expanding proactive choice. More context is
harmful if it confidently credits one person's words to another. Fix provenance
before encouraging the model to infer more from it.

Separate three decisions: where speech came from, who spoke, and whom they were
addressing. These may have different confidence levels. The person Rex looked at,
the expected game player, and the last speaker are not proof of current identity.

- Create an utterance evidence record with utterance/session ID and start/end times:
  transcript, voice candidate scores/margins and voiced duration, DoA samples and
  ambiguity, time-aligned face/lip-motion observations, neck/base pose, and optional
  radar occupancy. Normalize clocks explicitly (DoA uses monotonic time while the
  visual-speaker latch uses wall time). Preserve raw score semantics; do not label
  cosine similarity a calibrated identity probability.
- Eliminate previous-turn bearing substitution: missing DoA for this utterance is
  missing evidence. Keep historical bearing for gaze/search only as explicitly
  historical data. Match faces and pose from the utterance interval, not simply
  whatever is visible after ASR finishes. Test padding across speaker transitions.
- Extract one attribution resolver from interaction.py's override ladder. Begin
  with existing thresholds and replay comparisons. Return candidate identities,
  evidence conflicts, and known/unknown/ambiguous status. Do not multiply correlated
  face/DoA/radar observations as if they were independent votes.
- Distinguish a stable anonymous session participant from a known person. Preserve
  uncertainty in the transcript sent to Lean. Do not expose guessed names as settled
  user-role labels; Lean must not invent identity from conversational plausibility.
- Treat sequential speaker changes within one capture and truly overlapping speech
  separately. Existing dominant-cluster/whole-buffer scoring does not establish
  word-level diarization. First detect suspect mixed segments cheaply and prevent
  person-specific learning from them. Split sequential segments only where timing
  evidence supports a boundary, reusing captured audio; measure any added ASR cost.
  Do not promise to recover two simultaneous transcripts from the current pipeline.
- Reuse the existing visual mouth-motion data and DoA samples; no new resident model
  and no LLM speaker classifier before every reply. Voice ID already runs alongside
  ASR in the parallel transcription/identification helper; preserve that overlap. Expire evidence rather
  than extending the endpoint wait to force an identity.
- Use radar only as a spatial corroboration/search hint, with age and track quality.
  A body near a bearing does not prove who spoke. Track IDs may change or merge;
  foil shielding alone does not justify relaxing attribution thresholds. Verify
  geometry, mount transforms, seams, reflections, and dropout behavior on real data.
- Separate reply permission from learning permission. Rex can answer without using
  a name when identity is uncertain. Do not update voiceprints, durable personal
  facts, relationships, game credit, or sensitive actions from that uncertainty.
  Where attribution is necessary, ask a short clarification; avoid doing so on
  every harmless interjection. Session-only unresolved evidence may be retained
  briefly for explicit correction; do not silently rewrite durable history.
- Track pending questions per target. Another person answering does not automatically
  answer for that target. Include “speaker uncertain” and addressee uncertainty in
  Lean context. The LLM can interpret conversational intent, not certify biometrics.
- Record delivered reply ownership and interruption order. Group conversation cannot
  assume human/Rex/human alternation; incoming turns must retain their actual order
  even if a response to an earlier person is still generating.

Baseline/evaluation: use labeled fixture sequences with two and three people,
irregular order, rapid A-B-A switches, one-word answers, similar voices, off-camera
speakers, head/base movement, people changing seats, Rex speaking, and overlap.
Measure wrong-person assignments separately from abstentions, per-person confusion,
switch delay, wrong-addressee replies, and incorrectly attributed memory writes.
Report attribution quality versus latency; do not improve accuracy by refusing
every group turn. Add tests around `test_voice_primary_identity`,
`test_voice_bearing_match`, `test_active_speaker`, `test_lean_multi_party`, and
`test_game_roster_identity` in separate processes per CLAUDE.md.

First concrete slice: utterance IDs on bearing/visual evidence and no reuse of the
previous person's DoA when a current read fails. (Shipped 2026-09-04: `_note_voice_bearing`
clears the stored bearing on a failed read and stamps `utterance_t0`.)

Status 2026-09-05 (later) — the THIRD decision (whom they addressed) shipped:
`intelligence/addressee.py` hint + optional `conversation.stay_quiet` live tool on the
reply call + the dialogue-act targeted-frame fix. See CONTEXT.md "Whom was that said to?".

Status 2026-09-05 — shipped in shadow: `intelligence/attribution.py`
(`UtteranceEvidence`, `resolve()` → known/unknown/ambiguous + conflicts), wired at the
end of the ladder; ambiguous turns reach Lean as no-name instructions, mark the
transcript `uncertain`, stand passive voiceprint growth down, and suppress per-turn
memory learning. NOT done: making the resolver authoritative (replacing the ladder),
mixed-segment detection / sequential splitting, per-target pending-question
bookkeeping beyond the dialogue-act frames — those need live labeled captures. Follow with the unified resolver
and learning gates, then measured segmentation improvements. Live labeled capture
requires the owner's explicit go; this planning pass neither records nor moves Rex.

### 3. Replace the proactive priority ladder with bounded model choice

- Keep eligibility, consent, cooldowns, game/motion ownership, and quiet-window
  checks in Python. Convert cue builders into side-effect-free candidate readers.
- Offer at most a small bounded set (initially three) of eligible, diverse cues plus
  current context. Candidate fields: stable ID, evidence, target, expiration,
  last-offered state, and why it might matter now. Python filters eligibility;
  the model chooses relevance among the survivors or chooses silence.
- Reuse the existing `consider_initiating()` call to return a compact selection and
  line together. Proactive output may be structured because it is generated during
  a lull; do not impose that format on normal replies or add a separate chooser call.
- Preserve room for an unprompted playful observation grounded in current context.
  The objective is appropriate spontaneity, not permanently staying on one topic.
- Revalidate the chosen cue, target, boundaries, and conversation revision before
  playback. New human speech invalidates the proposal. Update “offered/asked” only
  when output actually starts, and track interrupted delivery separately.
- Keep existing frequency limits, including a cooldown on PASS. No per-tick LLM
  polling, speculative retry storm, or extra proactive calls per quiet period.

Acceptance: tests/replays cover unanswered questions, serious conversation, a person
thinking, new speaker arrival, stale news/event cues, interrupted generation, and
ordinary playful silence. The new chooser replaces both existing winning-cue ladders
and their duplicated bookkeeping; it does not run alongside them.

Status 2026-09-04 — shipped: `_collect_lean_cue_candidates` (eligibility in Python,
same gates) + `lean_brain` menu (`CHOICE:` reply, `last_choice_kind()`), spend of the
chosen cue only, conversation-revision + target revalidation before playback. The
classic consciousness proactive path is unchanged (it is suppressed under Lean
except for perception reactors).

### 4. Introduce action outcomes, then bounded heading alternatives

- Add correlated `ActionResult` records keyed by request/goal ID. Wrap existing
  executors initially; distinguish accepted, running, completed, partially achieved,
  blocked, aborted, and timed out. Include requested and actual angles, sequence ID,
  sensing/pose revision, and reason. Never treat `None` or thread creation as success.
- Move routine action narration to one owner after adapters are proven. Preserve
  urgent safety alerts; prevent the controller and planner from speaking conflicting
  acknowledgments/refusals. Stop/manual takeover invalidate pending plans immediately.
- First support only a bounded orientation goal. Separate desired final heading
  from an explicitly required direction/path. Equivalent-angle generation is cheap
  geometry; Python should enumerate and evaluate those candidates, not ask an LLM
  to calculate collision clearance.
- Give the model the goal, blocked result, and candidate evidence when interpretation
  or choice is needed. Permit at most one additional replanning call for that blocked
  action, with a time budget and deterministic failure response. An unambiguous
  validated alternative may eventually be selected without another call.
- Validate the entire proposed sweep using actual asymmetric body geometry and
  fresh sensing. An alternative beyond existing angle/time limits is declined,
  not silently clamped or split to bypass them. Unknown coverage is not proof of
  clearance. Revalidate during execution and verify final heading afterward.
- No automatic reverse/forward escape maneuver, arm repositioning, exploration,
  or general navigation in this phase. Those require separate capabilities and
  physical validation. A specific “turn left” instruction must not silently become
  a long rightward turn when direction itself is part of the request.

Acceptance: simulated/mocked cases for asymmetric arms, both routes blocked, sensor
loss/staleness, partial turn, heading mismatch, stop, manual takeover, and stale
results. Physical alternative-turn behavior stays disabled until explicitly
authorized floor tests verify the real robot. No firmware bypass or weakened guard.

Status 2026-09-05 — shipped: `intelligence/action_result.py` records issued /
refused / done / compass-verified (partial) outcomes; `turn(allow_reverse=True)`
heading alternatives behind `MOTION_HEADING_ALTERNATIVES_ENABLED` (default OFF).
Not done: a single narration owner (motion_controller still speaks user-commanded
refusals, the reply path consults `last_refusal`), the optional one-call LLM
replan, and the decision about the existing forward swing-escape (left as is,
now recorded).
### 5. Separate continuous listening from cancellable reply work

This is the largest risk and is deliberately isolated from the context improvement.
Introduce generation IDs earlier for stale output; do not wait for this phase to
prevent a cancelled proactive line from speaking.

- Extract turn coordination from `_handle_speech_segment()` and the active loop in
  small steps. Keep existing capture, VAD, attribution, AEC, and gap-recovery rules.
- Input processing produces turn events while one response job generates/plays.
  New speech can extend an unfinished turn or cancel obsolete response work.
- Carry generation ID through token stream, TTS preparation, queue items, and state
  callbacks. Reject obsolete items before synthesis and playback; close streams
  where supported. Cancelling locally does not guarantee provider billing stops.
- Preserve what Rex actually spoke separately from drafted text and make memory
  extraction use trusted human input and delivered output. No phantom questions
  from cancelled drafts and no duplicate writes on a recovered turn.
- Bound queues and coalesce updates. Do not implement speculative LLM generation
  on every ASR partial; that spends money and competes for resources.

Acceptance: human speech during thinking and playback, partial TTS cancellation,
late callbacks, own-echo rejection, two speakers, shutdown, and no-audio mode pass
the same production-path replay suite. Retire gap-recovery branches only when the
replacement demonstrates equivalent capture behavior.

Status 2026-09-04 — first slice shipped: speech generations on proactive items
(dropped at enqueue or pop once a human turn / barge-in began), `DoneEvent`
played/dropped truth, delivered-text return on cut-short streamed replies. NOT
done, deliberately: the turn-coordinator extraction and concurrent input/response
processing — those need live sessions to validate and stay as designed here.

## Budget and release gates

Numeric values below are proposed engineering targets, not measured promises.

- First release gate: ordinary-chat p50 and p95 do not regress against a same-device
  baseline with comparable warmed/cold runs and mixed background load. Report sample
  sizes and variability; do not judge from a handful of favorable turns.
- Working responsiveness target: ordinary warm chat p50 at or below 2.5 seconds and
  p95 at or below 4 seconds from last human voiced sample to meaningful audio. This
  is an intermediate target; if external service time prevents it, report the stage
  evidence rather than conceal it with filler. No guarantee of meeting it yet.
- Initial context-assembly target: p95 below 50 ms excluding any explicitly budgeted
  query-embedding opportunity; all retrieval has a single hard elapsed budget and
  can fall back. Tune targets from phase 0, not arbitrary per-call timeout sums.
- Ordinary turns: one main generation, zero new serial classifier/planner calls.
  Existing utility calls must be inventoried and reduced or justified separately.
- Proactivity: no more generation opportunities per quiet period than the present
  configuration; one call chooses and writes. Failed/stale calls count in the budget.
- Motion: at most one exceptional replan call per blocked goal in the first version.
- Cost: compare hosted tokens/requests and TTS characters per matched conversation
  workload, plus cost per active hour including idle speech. Use actual account
  pricing when measuring; no assumed price estimates in this plan.
- Memory: no additional resident models; bounded caches/queues; no sustained new
  swap growth attributable to the change under the target workload. Measure RSS
  and system pressure rather than assuming model file size is usable RAM.

Each phase has a short-lived rollback flag or adapter, a documented owner, and an
exit condition for removing superseded code. Do not multiply unrelated feature flags.
Mechanical file splitting follows ownership changes; moving 30,000 lines into ten
files without changing the contracts is not the restructuring objective.

## Verification and implementation order

Follow CLAUDE.md: run each relevant module in its own process, for example
`venv/bin/python -m unittest tests.test_conversation_arc`. Never use one-process
`unittest discover` for this repository. Extend tests at the production seams:
semantic recall, conversation arc, Lean agency, gap speech,
tool routing, speech queue, motion swing, and motion sequence. Repair the quality
eval to use the production reply seam before treating its scores as release evidence.
Add multi-turn fixtures for one trip with several destinations, explicit corrections,
pending questions, self-motion visibility changes, and blocked actions.

Use `tests/_lean_impulse_state.py`'s reset helper for existing impulse tests. The
documented cross-module global-state leaks reinforce the need for a session-owned
state object and explicit reset/cancellation semantics. Use relative event dates or
an injected clock; historical hardcoded dates have already caused fixture failures.
Check suspected regressions against the actual pre-change revision in a controlled
environment that retains required untracked assets/config. Do not blindly stash or
overwrite user work. CLAUDE.md's August failure list is a baseline lead, not permission
to label a present failure pre-existing without verification.

Run headless replay with fake hardware and recorded fixtures first. Real cloud evals
must be budgeted and identified; live recording/audio/motion needs the owner's
explicit go. CLAUDE.md warns that some tests reach the real Maestro and local audio;
audit/mask those dependencies before running them, rather than trusting “unit test”
to mean hardware-free. Its servo-park procedure applies after an authorized session
that touched servos; do not run it as part of this read-only planning work. This
document authorizes neither a robot run nor implementation.

Recommended first implementation batch: phase 0 instrumentation/replay, then phase 1
first-audio waits and embedding preparation, then phase 2 context integration.
That delivers measurable speed and continuity improvements before changing proactive
selection or physical planning. Phase 2B is part of the early context work, with
utterance-bound evidence prioritized over additional proactive features. Phases 3
and 4 can ship independently after phases 2 and 2B;
phase 5 remains a separate, carefully tested concurrency change.
