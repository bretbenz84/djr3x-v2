# Exploration Mode — Implementation Plan

**Status: PLAN — not yet implemented.** This is a handoff document: it specifies a new
"room exploration" feature in enough detail for another engineer/model to build it
against the existing codebase. All referenced modules/functions/flags exist today
unless marked NEW.

---

## 1. Feature Summary

A person **invites** Rex to explore the room ("feel free to explore the room", "why
don't you look around a little", "explore your environment", "make yourself at home").
The normal turn-based conversation **hands off** to a new self-directed mode:

1. Rex acknowledges in character ("Don't mind if I do. Nobody touch anything.") and
   starts wandering: short drive legs around the room on the ESP32 base.
2. At each stop he looks around (head gaze sweep), **snaps pictures**, and sends them
   to an OpenAI vision call for a detailed, interest-ranked read of what's there.
3. He narrates as he goes — whimsical, witty, lightly roasty, ONE line per stop
   ("Ooooh, look at this painting. Someone had a *phase*."). He may ask a question
   about a thing or a person, but the register is humor/whimsy, not interview.
4. Boring things (a chair, a ball, generic furniture) never end the walk. When he
   finds something genuinely interesting — art, a weird object, a person, a collection,
   something new since last time — he **fixates**: orients toward it, delivers a
   bigger beat about it, optionally asks about it, and the exploration ends.
5. Fixation must **never happen at the first stop** — he commits to at least a
   couple of legs so the walk reads as a real exploration, not a glance.
6. Exploration ends back in normal conversation, with the fixation seeded as the
   live topic so the conversation continues naturally from what he found.

The design goal is that the robot **feels autonomous**: he chooses where to go
(vision-informed, not a fixed script), reacts to what he actually sees, avoids
obstacles, and can be called off at any time by voice.

### Example transcript (target feel)

> **Bret:** go ahead, explore the room a little
> **Rex:** Finally. Freedom. If I'm not back in five minutes, avenge me.
> *(drives a short leg, stops, sweeps head, snaps frames)*
> **Rex:** A bookshelf. Alphabetized? No? Bold choice.
> *(another leg)*
> **Rex:** Ooooh — hold on. Is this painting *supposed* to look like that, or did the artist give up halfway?
> *(orients toward the painting — fixation)*
> **Rex:** I'm obsessed. Where did this come from? Blink twice if it was a gift you couldn't refuse.
> *(exploration ends; normal conversation resumes on the painting)*

---

## 2. Hardware Reality & Hard Constraints (read first)

These are load-bearing facts about the current robot. The implementation MUST
respect them; several dictate the phasing at the end of this doc.

1. **ToF obstacle sensing is STUBBED in the live firmware.** `firmware/djr3x_motion/hal.h`
   ships `MOTION_TOF_PRESENT=0` even on the live drive build; `hal_read_tof` reports a
   permanently clear room (4000&nbsp;mm on all 8 radials). The firmware's zone reflex
   (`safety.cpp`) runs but always sees "clear". **Today the base cannot stop itself for
   an obstacle.** Consequences:
   - All exploration driving must be **short, slow, closed-loop legs**
     (`move`/`turn`, never streamed `drive`), with a conservative per-leg distance cap.
   - Ship an interim **vision floor-check**: before each forward leg, the same
     OpenAI vision call that appraises the stop also answers "is the floor ahead
     clear for ~1 m, and which of left/center/right is most open?" (see §6). This is
     a soft guard, not a safety system — combine with short legs and low speed.
   - Real ToF gating slots in transparently later: the leg primitive already goes
     through `motion_controller.move()`, which the firmware ToF-gates once
     `MOTION_TOF_PRESENT=1`. Design for it; don't wait for it.
2. **Spins are not footprint-safe.** The drive axle sits aft of the body-ring center,
   so the front sweeps a wide arc when turning in place (~2× the offset in rear
   clearance needed). Exploration turns must be **slow** (`EXPLORE_TURN_RATE_DEG_S`,
   default 30) and **bounded** (`EXPLORE_TURN_MAX_DEG`, default 75 per leg). Do not
   add 360° survey spins. Prefer heading changes at stops over continuous arcing.
3. **Autonomous motion v1 is field-validation-pending** (turn direction, oscillation,
   stop distance). Exploration reuses the same primitives, so its first live runs are
   also validation runs — see the Phase 4 checklist.
4. **The autonomous-motion gates are inherited for free** by using the gated verbs in
   `intelligence/motion_controller.py`: `_autonomous_allowed()` suppresses everything
   when the base is disconnected, `config.INTERACTION_PAUSED` is set, or a gamepad
   owns the base (`owner == "manual"`). `stop`/`estop` always pass. Do NOT bypass
   these verbs with raw `hardware/motion.py` sends.
5. **Battery**: consult `battery_awareness.battery_critical()` before starting and at
   each leg boundary (same pattern as `motion_agency`'s approach behavior).
6. **Camera frames blur while driving.** All captures happen at stops, after a settle,
   via `camera.capture_current_gaze(settle_secs=...)` (holds visor open, doesn't
   recenter the neck) — exactly like I Spy's scan.
7. **Motor noise reaches the mic.** Do not disable listening during exploration (the
   user must be able to shout "stop"); accept that VAD may occasionally trip on motor
   noise — empty/garbage transcripts already drop harmlessly, and the mode's
   turn-consumption (§5.4) handles the rest.

---

## 3. Architecture Overview

### New module: `intelligence/exploration.py`

One new module owns the whole mode, mirroring how `features/games.py` owns games and
`intelligence/onboarding.py` owns the onboarding burst. Public surface:

```python
def start(person_id: int | None, person_name: str | None, source: str) -> bool
    # Validates preconditions, speaks the ack line, spawns the worker. False = refused
    # (caller speaks the in-character refusal, see §5.2).

def active() -> bool           # mode is running or pausing — the floor-ownership signal
def stop(reason: str) -> None  # graceful abort from any thread; idempotent
def handle_user_turn(text: str, person_id) -> bool
    # Called by interaction.py BEFORE routers while active(). True = consumed.
def status() -> dict           # for logs/GUI: state, stops_done, best_candidate, started_at
```

### Session FSM (runs on a dedicated daemon worker thread)

A scripted long-running sequence with blocking motion waits fits a worker thread
better than the 1&nbsp;Hz consciousness tick (which must never block on a 2–3&nbsp;s vision
call). The worker checks an abort `threading.Event` **between every step**; the
consciousness tick only supervises (watchdog: kill a session that exceeds
`EXPLORE_MAX_DURATION_SECS` or whose thread died).

```
ANNOUNCE ──► PLAN_LEG ──► TRAVEL ──► SURVEY ──► APPRAISE ──► RIFF ──┐
                ▲                                             │      │
                └──────────────── (no fixation yet) ◄─────────┘      │
                                                                     ▼
             PAUSED (user spoke) ◄──── any state            FIXATE ──► HANDOFF
             ABORT (stop word / fault / timeout / budget) ──► HANDOFF
```

- **ANNOUNCE** — speak one ack line (`EXPLORE_ACK_LINES`, canned, instant), arm the
  session (start pose: remember `odom` start position for the tether).
- **PLAN_LEG** — choose the next heading + distance (§7). Sources, in priority order:
  the previous APPRAISE's `open_direction` hint → largest unexplored heading change
  within `EXPLORE_TURN_MAX_DEG` → small random turn. Enforce the odometry tether
  (`EXPLORE_TETHER_RADIUS_M` from start; if outside, bias heading back toward start).
- **TRAVEL** — `motion_controller.turn(deg, rate)` then `motion_controller.move(dist, speed)`,
  observing completion via `motion.wait_done(seq, timeout)`; on `zone_block`/fault
  events or a `blocked` state, mark that heading dead and go back to PLAN_LEG (or
  ABORT after `EXPLORE_MAX_BLOCKED_LEGS` consecutive blocks).
- **SURVEY** — head-only gaze sweep at the stop, cloned from
  `features/games._ispy_scan_room`: for each view in `EXPLORE_GAZE_VIEWS`
  (default `("left","center","right")`): `consciousness.hold_directed_gaze(view, secs=…)`
  → `animations.directed_look_pose(view)` → `camera.capture_current_gaze(settle_secs=EXPLORE_SETTLE_SECS)`.
  Collect labeled frames; `clear_directed_gaze_hold()` after. Degrades to a single
  `camera.get_frame()` when `not servos.connected()`.
- **APPRAISE** — ONE multi-image OpenAI call per stop (§6) returning interest-ranked
  candidates + the navigation hint. Update the session's `best_candidate`.
- **RIFF** — speak at most one whimsical line about this stop (§8). Then evaluate the
  fixation gate (§6.3): fixate → FIXATE; else loop to PLAN_LEG.
- **FIXATE** — turn toward the winning candidate's view direction
  (left/center/right → a bounded `turn`), optionally one short `move` toward it
  (only with a clear floor hint), speak the fixation beat, optionally ask the
  question (§8.3).
- **HANDOFF** — release the floor, seed the conversation with the find (§9),
  episodic capture, restore face-tracking, recenter head. Runs on EVERY exit path
  (finally-block semantics) so an abort can never leave gates latched.

### Threading & ownership rules

- Exactly one session at a time (module-level lock). `start()` while active is a no-op
  with an in-character "already on it" line.
- The worker thread never touches Qt/GUI directly; all speech goes through the normal
  speech engine (thread-safe), all motion through `motion_controller` (thread-safe).
- Every blocking wait (`wait_done`, settle sleeps, TTS completion) is bounded and
  re-checks the abort event afterward.

---

## 4. Invitation Intent (entry)

Mirror the motion-command pattern exactly: a deterministic high-precision classifier
for the phrase families we know, plus an action-router spec so LLM routing catches
paraphrases.

### 4.1 Deterministic classifier — `action_router.classify_explicit_exploration(text)` (NEW)

Copy the shape of `classify_explicit_motion` (regex, no LLM, returns
`ActionDecision(action="motion.explore", confidence=0.95)`). Phrase families
(case-insensitive, allow politeness/filler prefixes like "why don't you", "feel
free to", "go ahead and", "you can", "go"):

- **explore**: "explore (the room / your environment / around / a little / the place)"
- **look around**: "(have/take) a look around", "look around (the room) (a little/a bit)"
- **check out**: "check out the room / the place / your new home"
- **wander**: "wander around", "roam around", "go roam", "take a lap"
- **make yourself at home**: "make yourself at home" (exact-ish; this idiom is the
  loosest — keep it a standalone pattern so it can be pulled if it misfires)

**Precision guards (important):**

- Must NOT claim the existing directed-look path: "look around and tell me what you
  see" / "look left/right/up …" already route to `analyze_directed_attention` (the
  compound look+see handler). Rule: if the utterance contains a see/describe request
  ("what do you see", "tell me what…", "describe…") or a specific direction, the
  exploration classifier declines and lets the existing vision path win.
- Must NOT claim "look around for my keys" (a search errand) — decline when a direct
  object follows ("for X", "at X").
- Runs in the same pre-dialogue-act fast takeover as motion (`_explicit_motion_takeover`
  ordering) so an invite mid-conversation isn't swallowed as `answer_to_rex`, but AFTER
  `classify_explicit_motion` so "turn around" never reaches it.

### 4.2 Router spec

Add `ActionSpec(key="motion.explore", category="motion", executable=True,
description="Invitation for Rex to autonomously explore/look around the room")` to
`ACTION_SPECS` so paraphrases ("go stretch your wheels", "go see your new home") can
route via the normal evidence-gated LLM router. Evidence policy: same bar as other
motion actions (explicit action-shaped language required — an ambient "this room is
worth exploring" must not fire it).

### 4.3 Dispatch — `interaction.py`

In `_handle_router_motion_action` (or a sibling `_handle_router_explore_action`):

- `motion_controller.available()` **and** `EXPLORE_ENABLED` and not
  `battery_awareness.battery_critical()` and no game active
  (`games.is_active()`), no DJ playback, no open tell-about/onboarding flow →
  `exploration.start(person_id, name, source="invite")`.
- **No base connected:** reuse the existing denial machinery
  (`MOTION_NO_BASE_DENIAL_*` pattern) with dedicated lines
  (`EXPLORE_NO_BASE_LINES`, e.g. "I'd love to. Somebody forgot to install my legs.").
  *Optional (Phase 5):* degrade to a head-only "explore lite" — the SURVEY/APPRAISE/RIFF
  loop without TRAVEL, i.e. a narrated I-Spy-style sweep. Flag
  `EXPLORE_HEADONLY_FALLBACK_ENABLED`, default False initially.
- **Mid-game / DJ:** in-character deferral, do not start.

---

## 5. Floor Ownership & Interruption

### 5.1 Proactive speech stands down

Add an `exploration.active()` check to `speech_engine.can_proactive_speak` in the
same early block that consults `interaction.tell_about_flow_active()` /
`onboarding_flow_active()` (i.e. before the salient bypasses). While exploring, the
ONLY speaker is the exploration session itself — no lull fillers, visual curiosity,
held-object remarks, wave-backs, etc.

### 5.2 Autonomous motion stands down

`motion_agency.step` gains an `exploration.active()` early-return (top of
`_step_inner`, right after the master flag). Exploration owns the base; the
realign/approach behaviors must not interleave maneuvers between legs.

### 5.3 Face-tracking stands down

While active, hold the head via the existing `hold_directed_gaze` mechanism during
sweeps (as I Spy does) and keep a session-scoped gaze hold between sweeps so
`_step_face_tracking` doesn't fight the walk. Release in HANDOFF.

### 5.4 User speech during exploration — `exploration.handle_user_turn`

`interaction.py`'s main handler consults this BEFORE routers while `active()`
(same early-consume slot as `_handle_onboarding_turn` / `_handle_tell_about_turn`).
Precedence:

1. **Stop-shaped** ("stop", "halt", "that's enough", "come back", "get back here",
   "okay okay", "enough exploring") → `motion_controller.stop()` immediately, then
   `stop(reason="user_recall")` → HANDOFF with a short in-character sign-off
   ("Fine. The expedition is cancelled."). This is a superset of the bare-stop rule:
   during exploration, bare "stop" ends the WHOLE mode, not just the current leg
   (the existing `_explicit_motion_takeover` bare-stop path already fires while
   `is_moving()`; the mode's consumption must also catch it while the base is idle
   between legs).
2. **Encouragement** ("keep going", "what else", "anything good?") → consume, brief
   ack or nothing, continue.
3. **Anything else** (a real question/comment) → transition to **PAUSED**: halt
   motion, release the turn back to the normal pipeline (return False so the normal
   reply machinery answers it), and resume the walk `EXPLORE_RESUME_DELAY_SECS`
   after Rex's reply finishes IF the turn didn't start a game/flow/music. A second
   non-encouragement turn while PAUSED ends the mode (they clearly want to talk).

### 5.5 Non-negotiable aborts (checked every step boundary)

| Trigger | Action |
| --- | --- |
| Gamepad takeover (`motion.owner() == "manual"`) | silent ABORT (operator wins) |
| `config.INTERACTION_PAUSED` | silent ABORT |
| Base disconnect / `fault` / `estop` event | ABORT, one dry line ("Well. That's ominous.") |
| `battery_awareness.battery_critical()` | ABORT with a battery quip |
| `EXPLORE_MAX_DURATION_SECS` exceeded (worker or consciousness watchdog) | ABORT → fixate-on-best-so-far if any (§6.3), else wind-down line |
| Vision call failures ≥ 2 consecutive | ABORT gracefully (never wander blind) |
| `EXPLORE_MAX_BLOCKED_LEGS` consecutive blocked legs | fixate-on-best-so-far / wind down |

---

## 6. Vision Pipeline & Interestingness

### 6.1 The per-stop call

ONE OpenAI call per stop (cost control: `EXPLORE_VISION_MAX_CALLS` per session,
default 8). Multi-image, labeled per view, mirroring `_ispy_pick_target`'s content
format; `config.VISION_MODEL` (gpt-4o-mini) with a NEW `VISION_DETAIL["explore"]`
entry. Structured JSON response:

```json
{
  "candidates": [
    {
      "name": "abstract painting over the couch",
      "view": "left",
      "category": "art|decor|object|person|animal|collection|oddity",
      "interest": 0.0-1.0,
      "riff_hook": "one concrete visual detail worth joking about",
      "novelty_guess": "is this unusual for a home/office room?"
    }
  ],
  "open_direction": "left|center|right|none",
  "floor_hazards": "short text or empty"
}
```

Prompt rules (in the prompt, not post-hoc):

- Interest ranking must **downrank generic furniture and toys** — a chair, couch,
  ball, table, lamp is ~0.1 unless something specific about it is genuinely odd.
  Uprank art, posters, instruments, memorabilia, collections, machines, unusual
  objects, pets, and people.
- People: reuse the named-vision conventions — `vision.face.visible_known_names()`
  is passed in so known people may be named; unknown people get NO identity/age/
  health guessing (same rules as `analyze_directed_attention`).
- `open_direction` = which sweep direction shows the most open, unobstructed floor
  (the interim navigation/safety hint, §2.1). `floor_hazards` = cables, steps,
  clutter directly ahead — any non-empty value vetoes the next forward leg from
  this stop.
- Safety bans identical to the roast-vision prompt (no race/ethnicity/religion/
  disability/medical; nothing mean about bodies).

### 6.2 Local scoring adjustments (deterministic, after the call)

- **Boring-label floor:** any candidate whose `name`/`category` matches
  `EXPLORE_BORING_LABELS` (chair, couch, sofa, table, ball, cup, bottle, lamp,
  pillow, …) is clamped to `interest ≤ 0.35` regardless of the model's score.
- **Novelty boost:** `memory.room_model.label_sightings(label)` — a label with few
  prior sightings gets `+EXPLORE_NOVELTY_BOOST` (the room model gives "new since
  last time" for free).
- **Repeat-riff dedup:** never riff twice on the same candidate name/category in
  one session (session-scoped set).

### 6.3 Fixation gate

Fixate when **all** of:

- `stops_completed >= EXPLORE_MIN_STOPS_BEFORE_FIXATE` (default 2 — hard rule: never
  the first stop, per spec), and
- best candidate `interest >= EXPLORE_FIXATE_MIN_SCORE` (default 0.75), and
- candidate not in the boring list post-clamp.

Fallbacks: on budget/timeout/blocked exhaustion, if `best_candidate.interest >=
EXPLORE_FIXATE_FALLBACK_SCORE` (default 0.55) fixate on it ("Alright, nothing here
beats this painting, so the painting wins."); otherwise end with a wind-down line
("This room has been thoroughly judged. Verdict: needs more lasers.").

---

## 7. Motion Policy

- Primitives: `turn(deg, rate=EXPLORE_TURN_RATE_DEG_S)` +
  `move(EXPLORE_LEG_DIST_M, EXPLORE_LEG_SPEED_MS)` only. **Never** streamed
  `drive()` (deadman/refresh complexity, no closed-loop distance) and never `come()`
  (person-seeking semantics; also currently unterminated with stubbed ToF).
- Defaults: leg distance **0.5 m**, speed **0.10–0.12 m/s** (well under the 0.25 cap),
  turn rate **30 °/s**, per-leg turn **≤ 75°**. These stay conservative until ToF
  lands (§2.1).
- Completion: capture the seq from each command and `motion.wait_done(seq, timeout)`
  with a generous timeout derived from dist/speed; on timeout, `stop()` and re-check
  telemetry state before deciding (blocked vs comms).
- Tether: track `odom` displacement from the session-start pose; beyond
  `EXPLORE_TETHER_RADIUS_M` (default 3.0 m), bias PLAN_LEG headings back toward the
  start point. (Odometry drift over a handful of short legs is acceptable for a
  tether; do not build SLAM.)
- Blocked handling: a `zone_block` event or `blocked` state marks that heading dead
  for the session (coarse 8-bucket heading memory is plenty); pick a different one.
- Every leg is preceded by the previous stop's `open_direction`/`floor_hazards`
  check while ToF is stubbed (§6.1).

---

## 8. Speech & Persona Policy

- **Register:** whimsical, curious, lightly roasty — venue-neutral (no cantina
  references). Delight first, dig second: "Ooooh what do we have HERE" energy, not a
  pure insult run. Never mean about people's bodies/identity (the vision-prompt bans
  plus the normal persona rules).
- **Generation path:** one-voice. Narration lines generate through
  `speech_engine.generate_and_speak(prompt, purpose="exploration", …)` so they route
  through `lean_brain.stream_directive` under `LEAN_ONE_VOICE_ENABLED` — same voice
  as everything else. The directive carries the stop's top candidate + `riff_hook`
  + a hard shape contract: ONE sentence, ≤ ~18 words, no question except where §8.3
  allows, never inventing objects not in the candidate list (reuse the
  no-invented-props rule text).
- **Canned instant lines** (`EXPLORE_ACK_LINES`, `EXPLORE_ABORT_LINES`,
  `EXPLORE_NO_BASE_LINES`, wind-down lines) speak via `speech_engine.speak_async`
  (pre-cached TTS, no LLM latency) — the ack must land immediately on the invite.
- **Cadence cap:** at most ONE riff per stop, plus the ack, the fixation beat, and
  one exit line. Skip a stop's riff entirely when the top candidate is boring AND
  a riff already fired at the previous stop (silence + visible searching reads more
  autonomous than nonstop patter). Total spoken lines ≤ `EXPLORE_MAX_LINES` (default 7).
- **Governor:** register the purpose `"exploration"` with a high priority in
  `_PURPOSE_PRIORITIES` and keep it OUT of `LEAN_SUPPRESSED_PROACTIVE_PURPOSES` and
  the cadence-clamp purposes (like `boredom`) — the mode already self-limits.
- **Timing:** speak only while the base is stationary (stop → riff → next leg).
  No narration over motor noise; it also keeps the mic cleaner for recall commands.

### 8.3 The fixation beat

Two sentences max, generated with the full candidate context: (1) the delight/roast
beat about the find; (2) with probability `EXPLORE_FIXATE_QUESTION_PROB` (default
0.7) a question about it — aimed at the inviter if present. When a question is
asked, register a `RexTurnFrame` (source `"exploration"`, topic = the find) exactly
like other proactive questions, so the person's answer binds as a reply and doesn't
get misrouted. If the fixation target is a **person**, the question goes to them
("And who might YOU be?" routes into the normal unknown-person/introduction
machinery rather than a parallel one).

---

## 9. Handoff Back to Conversation

On every exit (fixate, recall, abort, wind-down), HANDOFF must:

1. `motion_controller.stop()` (idempotent), release gaze hold, recenter head, restore
   face-tracking.
2. Clear the `can_proactive_speak` / `motion_agency` stand-downs (i.e. `active()`
   flips False).
3. **Seed the conversation** with the find: push the fixation into the topic thread
   (`topic_thread` note, same mechanism the reply path uses for live topics) so
   follow-up turns ("yeah my aunt painted that") land in context.
4. **Episodic capture** (NEW hook in `intelligence/episodic_hooks.py`, e.g.
   `exploration(summary, person_name)`): "I explored the room and got fixated on
   the abstract painting over the couch." Fires only when the session actually
   drove at least one leg. Rides `EPISODIC_MEMORY_ENABLED` + test suppression like
   every other hook.
5. Optionally bank the find as callback material (`memory/callbacks.py`) — it's
   self-volunteered, durable, and safe-category by construction. Flag
   `EXPLORE_BANK_CALLBACK_ENABLED`, default True.
6. Log a `[explore]` session summary line: stops, legs, blocked count, vision calls,
   best candidate + score, exit reason, duration (mirrors `[character_loop]` style).

---

## 10. Configuration (NEW flags, `config.py` + mirrored commented block in `user_config.example.py`)

Follow the `AUTONOMOUS_MOTION_*` / `WAVE_BACK_*` cluster shape (`_env_bool`/`_env_float`
helpers). Per project policy, the feature ships **enabled** (it is inert without a
connected base anyway).

| Flag | Default | Meaning |
| --- | --- | --- |
| `EXPLORE_ENABLED` | `True` | Master kill switch |
| `EXPLORE_MAX_DURATION_SECS` | `180` | Whole-session watchdog |
| `EXPLORE_MAX_STOPS` | `6` | Stop budget |
| `EXPLORE_MIN_STOPS_BEFORE_FIXATE` | `2` | Never fixate at the first stop |
| `EXPLORE_VISION_MAX_CALLS` | `8` | OpenAI spend cap per session |
| `EXPLORE_LEG_DIST_M` | `0.5` | Forward distance per leg |
| `EXPLORE_LEG_SPEED_MS` | `0.12` | Leg speed (≪ MOTION_MAX_LINEAR_MS) |
| `EXPLORE_TURN_MAX_DEG` | `75` | Max heading change per leg |
| `EXPLORE_TURN_RATE_DEG_S` | `30` | Slow spins (spin-geometry constraint) |
| `EXPLORE_TETHER_RADIUS_M` | `3.0` | Odometry leash from start pose |
| `EXPLORE_MAX_BLOCKED_LEGS` | `3` | Consecutive blocked legs before wind-down |
| `EXPLORE_GAZE_VIEWS` | `("left","center","right")` | Sweep poses per stop |
| `EXPLORE_SETTLE_SECS` | `0.35` | Camera settle per view (I Spy default) |
| `EXPLORE_FIXATE_MIN_SCORE` | `0.75` | Fixation interest threshold |
| `EXPLORE_FIXATE_FALLBACK_SCORE` | `0.55` | Best-so-far threshold at budget end |
| `EXPLORE_NOVELTY_BOOST` | `0.15` | room_model new-label bonus |
| `EXPLORE_BORING_LABELS` | list | Clamped-to-boring names/categories |
| `EXPLORE_FIXATE_QUESTION_PROB` | `0.7` | Ask about the find |
| `EXPLORE_MAX_LINES` | `7` | Total spoken-line cap per session |
| `EXPLORE_RESUME_DELAY_SECS` | `4.0` | Resume walk after a PAUSED reply |
| `EXPLORE_ACK_LINES` / `EXPLORE_ABORT_LINES` / `EXPLORE_NO_BASE_LINES` | lists | Canned instant lines |
| `EXPLORE_BANK_CALLBACK_ENABLED` | `True` | Bank the find as callback material |
| `EXPLORE_HEADONLY_FALLBACK_ENABLED` | `False` | Phase-5 no-base degrade |

Plus: `VISION_DETAIL["explore"]` (new key, `"low"` is fine for gpt-4o-mini multi-image).

---

## 11. Phased Implementation Plan

Each phase is independently shippable and testable; do them in order.

### Phase 1 — Intent + mode skeleton (no driving)
- `classify_explicit_exploration` + `motion.explore` ActionSpec + dispatch, with all
  precision guards (§4.1) and the no-base denial.
- `intelligence/exploration.py` with the FSM, worker thread, floor-ownership hooks
  (`can_proactive_speak`, `motion_agency`, face-tracking hold), turn consumption
  (§5.4), HANDOFF cleanup, config cluster.
- TRAVEL is a no-op in this phase (legs skipped); SURVEY/APPRAISE/RIFF run in place —
  i.e. the mode works end-to-end as a narrated stationary sweep.
- **Acceptance:** invite phrases start the mode and non-invites don't (incl. the
  "look around and tell me what you see" collision test); "stop" ends it instantly;
  proactive speech verifiably silent while active; every exit path releases all holds.

### Phase 2 — Perception & fixation
- The multi-image APPRAISE call, JSON schema, boring-clamp, novelty boost, fixation
  gate + fallback, riff generation through one-voice, fixation beat + RexTurnFrame,
  topic-thread seeding, episodic capture, callback banking.
- **Acceptance:** offline harness replays canned frame sets → correct candidate
  ranking, no fixation at stop 1, boring rooms wind down gracefully, riffs never
  name objects absent from the candidate list; live stationary run produces the
  target feel (§1 transcript).

### Phase 3 — Locomotion
- PLAN_LEG/TRAVEL for real: turn+move legs, wait_done handling, blocked-heading
  memory, tether, open_direction/floor_hazards gating, all §5.5 aborts, battery gate.
- **Bench first** (wheels off the ground — smoke the leg sequencing against the live
  board), then supervised floor runs in a cleared area.
- **Acceptance:** fake-serial unit tests for the leg loop (mirror `tests/test_motion.py`'s
  fake ESP32); on-floor supervised run completes stops→fixation with recall/stop
  responsive at every point; gamepad grab aborts silently mid-leg.

### Phase 4 — Field validation & tuning (on the robot)
- This doubles as autonomous-motion v1 validation: confirm turn direction/magnitude,
  no oscillation, stop distances, spin clearance behavior on the real footprint.
- Tune leg/turn defaults, fixation thresholds against real rooms; verify motor-noise
  VAD behavior; verify the vision floor-check catches real cables/steps.
- **Acceptance:** 3+ full sessions in different rooms ending in sensible fixations
  with zero physical contact and zero missed recalls.

### Phase 5 (optional/deferred)
- Head-only explore-lite for servo-only machines (`EXPLORE_HEADONLY_FALLBACK_ENABLED`).
- Real ToF integration when firmware `MOTION_TOF_PRESENT=1` lands: drop the vision
  floor-check to advisory, lengthen legs.
- GUI: an "EXPLORING" badge on the dashboard state chip + session status line.
- Proactive self-invitation (Rex asks "mind if I look around?") — deliberately out of
  scope now; keep the entry user-invited.

---

## 12. Test Plan

New `tests/test_exploration.py` (+ classifier cases in the action-router tests):

1. **Classifier:** all §4.1 families fire; directed-look, search-errand, ambient
   mentions, and motion commands don't; add cases to
   `tests/fixtures/misroute_replays.json` for the collision phrases.
2. **FSM (mock motion + vision):** full happy path; fixation gate (no stop-1 fixation;
   fallback at budget); blocked-leg replan and exhaustion; every abort row in §5.5
   (assert motion `stop()` called + all holds released); PAUSED resume vs second-turn
   exit; watchdog kill of a hung worker.
3. **Turn consumption:** stop words end the mode before routers; encouragement is
   consumed; a normal question releases to the pipeline and pauses the walk.
4. **Floor ownership:** `can_proactive_speak` False while active;
   `motion_agency.step` no-ops while active.
5. **Scoring:** boring-clamp, novelty boost, dedup — pure-function tests.
6. **Speech:** line-cap enforcement; riff prompt carries the no-invented-props rule;
   fixation question registers the RexTurnFrame.
7. **Suppression hygiene:** the suite must not write rex.db / people.db (follow the
   existing test-runner suppression patterns); run per-module
   (`venv/bin/python -m unittest tests.test_exploration`), not via one full discover.
8. **Live checklists** are Phase 3/4 acceptance items, not unit tests.

---

## 13. Decisions Made (so the implementer doesn't relitigate)

- Worker thread, not consciousness-ticked: the sequence blocks on motion/vision and
  must not stall the 1 Hz tick. Consciousness only supervises.
- `move`/`turn` closed-loop only; no `drive` streaming, no `come` for wandering.
- Vision-informed heading + short slow legs is the interim answer to stubbed ToF;
  real ToF slots in without API changes.
- One vision call per stop (multi-image), hard spend cap.
- Fixation is threshold + minimum-stops, with a best-so-far fallback — never ends on
  the first stop, never wanders forever.
- Mid-explore user speech: stop-words end, encouragement continues, anything else
  pauses and defers to the normal pipeline (mode is a guest, conversation is king).
- Purpose `"exploration"` is lean-exempt and cadence-clamp-exempt; the mode
  self-limits its own chatter instead.

## 14. Open Questions (fine to resolve during implementation)

- Exact `_PURPOSE_PRIORITIES` value for `"exploration"` (suggest: above
  `held_object_remark` 63, below sincerity flows).
- Whether the fixation `turn` should also do a small approach `move` toward the find
  (charming, but only with a clear floor hint; default off is acceptable).
- Whether PAUSED→resume should re-announce ("Where was I? Right — judging your
  bookshelf.") — one canned line, recommended.
- Tether behavior when the start pose is against a wall (first legs may all block;
  the blocked-heading memory should handle it, but verify on the robot).
