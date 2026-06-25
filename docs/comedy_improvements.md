# DJ-R3X v2 — "More Fun" Improvement Plan (scoped + code-verified)

> **Purpose:** the build plan for making Rex funnier (more of a roaster/showman), more
> genuinely curious about his surroundings, and capable of bigger bits. Every idea is
> grounded in machinery that **already exists** — `file:function` hooks are named so this
> can be picked up cold.
>
> **This revision (2026-06-25):** the original brainstorm was put through a 9-cluster
> multi-agent **code-verification pass** (every load-bearing claim checked at `file:line`)
> and then an **owner scoping interview**. Corrections are inline and marked **⚠**. Cut /
> deferred clusters are called out so nobody builds something that was scoped out.
>
> **Effort tags:** `S` = a weekend · `M` = ~a week · `L` = multi-week · `XL` = a project.
> Where verification disagreed with the original tag, the corrected tag is shown as
> `M (was L)`.

---

## Decisions locked (2026-06-25 owner interview)

| Area | Decision | Cascade |
| --- | --- | --- |
| **Smart home (§7)** | **CUT entirely** | Drops HA, Alexa, timers, the announce + scheduler primitives, show-cue, Movie Time, and the "act" half of notice→offer→act. **Rescued:** "inject open plans into the live reply" is pure conversation, not home control — **moved to §8**. |
| **Storytelling (§4)** | **Explicit-ask only** | No idle/self-initiated tall tales. Kills the over-eager failure mode. Simplifies E-2. |
| **Sharp roast tier (§1)** | **Build it**, gated on **warmth alone**, accept `roasted_sincere` drift | Loosest gate → keep the harsh-word governor regex as the cruelty backstop. |
| **Roast Battle (E-3)** | **CUT** | Off-brand for a warm-curious Rex; the rest of crowdwork stays. |
| **Curiosity (§2)** | **ALL-IN** on `room_model` | Full object permanence + change detection + object-grounded curiosity + docent (docent is ask-only). |
| **Room privacy posture** | **Rich** | Open vocabulary **minus screens/devices**, persist indefinitely, coarse buckets. **+ new item:** GUI object bounding boxes + labels. |
| **Games (§5)** | **DEFER** to a someday note | Parks DJ-announce, stingers, Name That Tune, Final Jeopardy, leaderboard, live-trivia, word-assoc clock, registry refactor. |
| **Host/MC mode (§3)** | **Operator-triggered only** ("host this") | Never self-promotes. Light crowdwork (arrivals, call-outs, compare-the-room, gestures, land-the-laugh) stays. |
| **SFX / clips (§6)** | **Skip for now** | Defers rimshot/sad-trombone/airhorn, composite show-moment, reveal stingers, puppeteer-board expansion. Keep the no-clip wins. |
| **Feed callback bank (§8)** | **About-present-people only, hard-gated** | New deterministic wall (present + warm). Do **not** bank Rex's own bits. |
| **Beat pack (§6)** | **Fully LLM-addressable** | + a frequency cooldown so self-directed bits don't become constant mugging. |
| **Running gags (§1)** | **Silent higher-probability** | Gag recurs more; Rex never numbers it aloud (no miscount risk). |
| **Signature roast handle (§1)** | **Thin layer over the callback bank** | Reuse the per-person volunteered-premise pool + its safety wall; no parallel facts pipeline. |
| **Named arrivals (§3)** | **Arrivals + warm departures** | Clock people on entry AND send off a known departer warmly. |
| **First build batch** | **Comedy felt immediately** | Delivery profiles + smug-after-a-roast mood + comedic personas. |

---

## Verification corrections (load-bearing — these were wrong/imprecise in the brainstorm)

- **⚠ Sharp roast tier is lower-risk than framed.** `llm._TIER_ROAST_STYLE` (`llm.py:348`)
  *already* steers best-friend roasts hot ("the full arsenal, zero mercy"). Only the
  deterministic governor (`social_frame._roast_level`, `social_frame.py:849`) caps everyone
  at `normal`. The feature is "stop the governor clamping what the prompt already
  encourages," **not** "invent escalation." Two cited config keys
  (`ANTAGONISM_TIER_CAPS_LIFT_WARMTH`, `TIER_ROAST_STYLE`) **do not exist** — must be created.
- **⚠ Room-energy landmine:** `social_mode=='performance'` **also fires for an empty room**
  (`awareness/social.py:70-71`, count==0). A "hot room" boost keyed on it fires to nobody.
  Derive energy from headcount + smiles + laughter, never from `social_mode` alone.
- **⚠ Tall-tale & saga are `M`, not `L`.** Performance actions speak via
  `performance_output.execute_plan` → `_speak_blocking`, which **skips the `max_sentences`
  trim entirely** (the same trick `web_search` uses). The length cap lives in *three* places
  (`config.py:386` core prompt, `comedy_modes.build_directive:218`, `social_frame`
  `max_sentences` trim ~`:619`) but the performance path sidesteps all three for free. Only
  the action's own `prompt_contract` caps it.
- **⚠ Leaderboard prereq is narrower than stated.** The doc said 20q returns `''` — **false**;
  20q's outcome survives (`games.py:1930`). Only **i_spy and word_assoc** self-clear before
  capture and are contentless. (Moot now — games deferred — but recorded for accuracy.)
- **⚠ "Flip `SCENERY_CHANGE_REMARK_ENABLED`" is a red herring.** That flag governs a
  boot-to-boot *caption* remark (`episodic_hooks.py:280`), unrelated to a live object diff.
  Change detection must read `world_state.objects` vs `room_model` per idle tick.
- **⚠ E-1 Reunion already half-exists.** `consciousness._pick_absence_phase` (`:3894`,
  `LONG_ABSENCE_THRESHOLD_DAYS=60`) is live and swaps the greeting *text* today. Only the
  *staged* sequence + one-turn greeting lockout are missing → "upgrade," not "build."
- **⚠ `directed_look_pose(target=name)` does not turn toward a person.** `target` is
  diagnostic-only (`animations.py:1304`); the head moves by `direction` (left/right). A
  named turn needs a bbox→direction resolution step that **does not exist** — shared by
  named-arrivals and call-outs; build it once.
- **⚠ Games never apply the comedy content-guardrail.** `comedy_modes.build_directive`'s
  "no body/health/identity/private-fact" ban is **not** on the game path (`features/games.py`
  builds its own prompts). (Moot now — Roast Battle cut, games deferred — but a hard rule if
  any game work resumes.)
- **⚠ `smug` is already an alias of `proud`** (`body_mood.py:97`); `body_mood` is driven by
  ~6 sites today (not "almost nothing"), just none on a roast outcome. The feature is one new
  call site, not a new mood.
- **⚠ Asset gap (now deferred):** only `Air Horn.mp3` exists — no ding/buzzer/drumroll/
  rimshot. A Final-Jeopardy thinking-music file sits on disk **unwired**. All clip-dependent
  bits were scoped out this round.

---

## Scope at a glance

**IN (build):**
- §1 Comedy & roasting — all five items (sharp tier, running gags, personas, tease-obsession,
  signature handle) + the delivery-profile "do-first."
- §2 Curiosity — **all-in**: COCO unlock, `room_model` (Rich), object-grounded curiosity,
  change detection, POV seeds, react-to-you, crowd curiosity, **+ GUI object bboxes**. Docent
  bit is in but **ask-only**.
- §3 Crowdwork — land-the-laugh, room-energy read, named arrivals + departures, name-and-point
  call-outs, compare-the-room, gesture-reaction layer, **host mode (operator-triggered only)**.
- §4 Storytelling — tall-tale, multi-episode saga (`person_story`), docent — **all ask-only**.
- §6 Physical — **no-clip wins only**: comedic beat pack (LLM-addressable), thinking eyes,
  smug-after-a-roast mood.
- §8 Personalization — open commitments, honorifics (celebration-sourced for now), gossip
  discharge, feed-callback-bank (about-present-people only), **+ rescued "inject open plans."**

**DEFERRED (someday — keep the notes, don't build yet):**
- §5 Games (entire cluster), the clip-dependent §6 bits (rimshot/show-moment/reveal-stingers/
  puppeteer expansion), beat-synced dance mode (needs a beat signal), E-4 Cantina Open
  (diminished), E-5 Long Con (diminished).

**CUT (removed from scope):**
- §7 Smart home (whole cluster), §8 per-person smart-home anticipation, Roast Battle (E-3),
  Name That Tune Showdown (E-6), Movie Night (E-8), roll-up/mock-retreat base bits
  (base not show-ready).

---

## The three things to do first (revised)

1. **Comedic delivery profiles** *(batch 1).* A roast and a condolence currently use
   **identical** ElevenLabs voice settings — ⚠ verified: delivery shaping flows *only* through
   `empathy.get_delivery_overrides` (`empathy.py:1214`), keyed on empathy mode; a default-mode
   roast gets the plain voice. Mirror `empathy._MODE_VOICE_SETTINGS` (`empathy.py:1163`) with a
   `comedy_mode → voice+timing` map and inject it at the **existing** delivery seams
   (`interaction.py:9931` non-stream, `:10193` stream — both verified symmetric).
   **Default to a small set first** (deadpan + smug) to limit TTS cache regen — each
   `{stability,style}` combo is a fresh cache key (`tts.py` `_settings_cache_token`); add
   mischief/dj_hype later. **Precedence:** empathy outranks comedy on a non-default empathy
   turn; comedy layers only on neutral-empathy turns. **Effort: M.**

2. **Free the COCO object detector → `world_state.objects`.** ⚠ verified: `animal_detector.py`
   loads the full 80-class MediaPipe ObjectDetector (`efficientdet_lite0.tflite`) and
   `_best_animal_category` (`:149`) discards every non-animal detection — the data is computed
   and thrown away. Generalize behind the **Rich** posture (open vocab **minus** screens/
   devices — drop `laptop`/`tv`/`cell phone` to honor the existing no-screens rule), add an
   `objects` key to `world_state._DEFAULTS` (`world_state.py:4`, currently absent — `update()`
   raises on unknown keys), and publish via a `detect_objects_local` sibling mirroring
   `vision/scene.py:detect_animals_local` (`:491`). Reuse the confirm-streak + confidence
   floors (`efficientdet_lite0` is weak; indoor flicker is real). This is the substrate the
   whole §2 stands on. **Effort: M.**

3. **Land-the-laugh / take-a-bow.** ⚠ verified: `audio/scene.py:_analyze_cycle` computes
   `laughter_detected` *and* `applause_detected` every cycle; `laughter` is read as passive
   prompt flavor in one place (`llm.py:261`), and **`applause` has no reader at all**. Add a
   `_step_room_reaction`: laughter shortly after a Rex line → a dry follow-through; applause →
   `proud_dj_pose` bow. Copy the wave-back detect→latch→fire-when-free template
   (`consciousness._step_wave_reaction:2580`). **Gate on a recent-Rex-utterance window** so he
   doesn't react to ambient noise/music, and a low per-session cap so "see, that one's free"
   doesn't read as needy. **Effort: M.**

---

## 1. Comedy & roasting

### Warmth-earned "sharp" roast tier — **BUILD** *(gate: warmth alone)*
⚠ The prompt already steers close friends hot; only the governor caps everyone at `normal`.
Add a `sharp` return to `_roast_level` (`social_frame.py:849`) gated on **effective warmth ≥
threshold AND no active downgrade** (tender/sad/boundary/flat/child/micro). Per the owner call,
`sharp` is earned by **warmth alone** — it does *not* require reciprocal banter. `sharp` loosens
the slim single-beat cap to allow a real two-beat roast and stops `_is_sharp_roast_sentence`
(`:1123`) from stripping the pointed line.
**Risk (accepted):** warmth-alone is the loosest gate and the most likely to lift the noisy
`roasted_sincere` eval (`evals/checkers.py:202`). Mitigation: **keep the harsh-word governor
regex as the cruelty backstop** (it must still catch genuinely cruel output), and define
`sharp`-vs-`normal` concretely so the model knows the line. Create the two missing config keys.
**Hooks:** `social_frame.py:_roast_level`/`roast_rule`/`_slim_roast_rule`/`govern_response`/
`govern_stream_sentence`; new config keys; `BANTER_WARMTH_THRESHOLD` (`config.py:2981`, exists).
**Effort: M** *(touches the whole roast pipeline + an eval pass).*

### Running gags that escalate (resurrect dead `running_bit`) — **BUILD** *(silent escalation)*
⚠ verified dead scaffolding: `running_bit` is in `callbacks.py:CATEGORIES` but never produced
(`bank()` coerces out-of-set categories to `quirk`, `:141`) or read for selection. ⚠ The real
work is **carving an escalating gag OUT of the reuse-suppression pipeline**: `freshness_factor`
(`callbacks.py:336`) halves weight per fire, a 7-day `off_cooldown`, and the per-session
`_used_premise_ids` set (`callback_engine.py:151`) all suppress repeats. Promote a premise at
~3 lands (`use_count` is tracked), exempt it from those three gates, and **raise its fire
probability silently** — per the owner call, **no audible numbering** (Rex never says "third
time tonight"; avoids miscount). Add a hard fire cap + cross-session cooldown + auto-retire so a
beloved gag doesn't outstay its welcome; honor boundary/forget → retire.
**Hooks:** `callbacks.py` CATEGORIES/`mark_used`/`freshness_factor`; `callback_engine.py`
`settle_turn`/`build_callback_directive`/`build_lull_prompt`. **Effort: M.**

### Comedic personas (smug_superiority / appliance_conspiracy / dramatic_narrator) — **BUILD** *(batch 1)*
⚠ verified lowest-risk/highest-leverage: `select_mode` rotates 8 thin one-beat stances with
anti-repeat (`_choose_without_stutter:381`) and premise rotation; new `ComedyMode` entries are
genuinely drop-in (add to `_MODES`, `_SLIM_STANCE`, a `select_mode` pool branch, `_premise_for`
tags). **Watch the interest-turn guard** (`comedy_modes.py:150`): `dramatic_narrator` and
`appliance_conspiracy` are self-absorbed bits — add them to the same engage-first exclusion or
they'll talk over a sincere share. `appliance_conspiracy` is a recurring *frame*; keep it
context-free character (Rex is venue-neutral — no guaranteed toaster). Pairs with delivery
profiles (each persona gets one). **Effort: S.**

### Tease-the-obsession lane — **BUILD**
⚠ verified: `conversation_steering._directive_for` (`:430`) caps humor to "light roasts only
about the hobby or Rex's ignorance, not the person's competence." But `_directive_for` takes
only the topic — **no person/warmth signal** — so "relax it *for warm relationships*" needs
warmth threaded from its caller (`steer_directive`). Coupled: `social_frame._slim_roast_rule`
(`:503`) *also* forces ENGAGE-FIRST on `interest` turns, so relaxing only the steering line
isn't enough — the governor's interest rule must also allow a depth-tease. Spell out the line:
"you have a SPREADSHEET for nebulae" (intensity, allowed) vs "your spreadsheet is amateur hour"
(competence, never). **Effort: S** *(slightly more than one line: thread warmth + coordinate the
two gates).*

### Per-person signature roast handle — **BUILD** *(thin layer over the callback bank)*
⚠ verified heavy overlap with the callback engine, which **already** banks per-person
volunteered safe premises (passion/hobby/project) with a protected-category wall
(`callback_engine.protected_category_hit:419`). Per the owner call, build this as a **thin layer
over `callbacks.active_pool`**, not a parallel `facts`-ranked pipeline: pick one durable premise
as the recurring angle, rotate among the top 2-3 so it never fixates, and dedupe with the
callback engine. Expose it via `llm._relationship_tone_rule` (`:885`, warm/sparring) — note that
function doesn't read facts/callbacks today, so the chosen handle gets threaded in.
**Effort: M (was S)** — the seam isn't wired; it's a small selector + rotation state, not a
one-liner.

---

## 2. Curiosity & a persistent mental model of the room — **ALL-IN (Rich posture)**

> Root cause of "curiosity feels generic": **Rex has no memory of the room.** The owner chose
> to build the full persistent substrate. Privacy posture = **Rich**: open vocabulary minus
> screens/devices, persist indefinitely, coarse buckets.

### Room baseline + object permanence (`memory/room_model.py` + rex.db table) — **BUILD (L)**
⚠ verified: no `room_model.py`, no objects table; scene memory today is fuzzy free-text captions
(`episodic_hooks.py:213`) — the "clustered into one vibe" framing was loose, but per-object
structured memory genuinely does not exist. Build a new `rex.db` table (label, location bucket,
first_seen, last_seen, sighting_count) + migration in `setup_assets.py`, fed by the
`world_state.objects` stream, with a compact "remembered room" summary injected into the
curiosity prompt. **Loosen** the `visual_curiosity` proactive rule (`conversation_agenda.py:125`
literally says "visible scene only. Do not also bring up memory") so a remembered-room reference
is allowed. **Rich posture details:** coarse buckets (left/center/right + fg/bg), single room,
persist (no auto-forget), but **never log screens/text** (the COCO stream already drops devices).
Inherit episodic test-suppression. **Hard-depends on the COCO unlock.** **Effort: L.**

### NEW — GUI object bounding boxes + labels — **BUILD** *(owner request)*
Overlay each detected object's bbox + label on the dashboard camera preview, mirroring the
existing pose-wireframe + face-box overlay. Feed from `world_state.objects` (the COCO stream).
Gate it like the pose wireframe (`GUI_POSE_WIREFRAME_ENABLED` → a new `GUI_OBJECT_BOXES_ENABLED`).
**Hooks:** `gui/dashboard.py` camera-preview overlay; `world_state.objects`. **Effort: S** *(once
the COCO stream exists).*

### "Wait — that's new": live change detection — **BUILD (needs COCO + room_model first)**
⚠ Don't flip `SCENERY_CHANGE_REMARK_ENABLED` (that's the boot-to-boot caption path). Diff
`world_state.objects` vs `room_model` each idle tick; on a confirm-streak-crossed new/departed
object, fire `do_noticed_change` through the governor. Dispatch slots into
`consciousness._idle_micro_behavior_choices` (`:5318`) → a new `idle_behaviors.do_noticed_change`.
**Per-object cooldown + confirm-streak** so he doesn't cry wolf; one genuine double-take per
visit. **Effort: M** *(after the prereqs).*

### Object-grounded curiosity questions — **BUILD (needs room_model first)**
⚠ Today's `_step_visual_curiosity` (`:6133`) already spans clothing/accessories/activity, not
just objects (the "name an object and ask" paraphrase undersold it) — but it's stateless.
Rewrite the prompt to prioritize the most novel/longest-present/most-personal object from
`room_model`, crossed with known interests (`conversation_steering.build_context`). Preserve the
elaborate gating chain (cooldowns, silence window, min-turns). **Effort: S** *(once room_model
exists).*

### Room-and-person-aware POV seeds — **BUILD** *(person/interest slice ships now)*
⚠ verified: `REX_POV_SEEDS` are 100% internal; `rex_pov._context_signature` (`:157`) emits only
{people|quiet|flat}. The **person/interest/unknown-face tags ship today** (no detection
dependency); only the **object-class tag** wants the COCO stream. Keep the hallucinated-prop
guardrail intact (`rex_pov.py:67` bans claiming to SEE a specific prop) — seeds stay "a category
of thing," never the actual guitar in frame. **Effort: S.**

### React to YOU (clothing / pet / held-object callbacks) — **BUILD**
⚠ verified: `describe_scene_detailed` (`scene.py:893`) exposes `people[].visible_clothing/
accessories/activity` — but `do_appearance_riff` is **fact-based, not vision-diff-based**, so the
diff machinery is net-new. Build a lightweight per-person visual cache (last clothing/accessories/
held object), cache on the **existing periodic scene scan** (don't diff every turn — it's a GPT
vision call), with a confidence gate + per-person cooldown ("don't 'new jacket?' the same
jacket"). Entrance/idle trigger. Signals: clothing/accessories/held-object/pet. **Effort: M.**
*(Independent of room_model — it's per-person, not room memory.)*

### Let curiosity work IN a crowd — **BUILD**
⚠ verified: `_step_visual_curiosity` does a bare `return` when `crowd_count > 2` (`:6074`).
Replace with a crowd-aware branch fed by `social_scene.conversation_cast_context`. Per the
recommended default, **address the group** ("what's the story with the three of you?") rather
than single out a guest; tighter cooldown than 1:1; keep an upper ceiling so he doesn't
interrogate a packed room. Note the path assumes a single `_engaged_person_id` — the crowd
branch relaxes that, slightly more than a one-line swap. **Effort: S.**

---

## 3. Crowdwork & reading the room

### Room-energy read → governor boost (hot/warm/cold) — **BUILD** *(shadow-log first)*
⚠ **Do not key on `social_mode=='performance'`** — it also fires for an empty room
(`social.py:70`). Add a `room_energy()` helper deriving hot/warm/cold from **headcount ≥ N AND
(smiles OR laughter OR chatter)** — `_person_is_smiling` (`consciousness.py:2112`),
`laughter_detected`, `group_chatter_detected` all exist. Surface on `SituationProfile`; apply a
bonus in `action_governor._score` (`:311`, already receives the profile). **Ship in shadow-log
mode first** before wiring to scoring so he can't go manic in a loud room. **Effort: M.**

### Energetic, named arrivals & departures — **BUILD** *(arrivals + warm departures)*
⚠ verified: identity is **already in hand and discarded** — `known_now` (`consciousness.py:4821`)
is computed at the trigger site and used only as a suppression gate, never to name anyone. Swap
the generic count-based line for a named one. Per the owner call, **also react to a known person
leaving** with a warm rib. ⚠ The physical turn-toward-door needs the **bbox→direction primitive**
(`directed_look_pose(target)` is diagnostic-only) — build it once; shared with call-outs. Tone
scales with relationship (rib at warm/sparring, plain welcome otherwise). **Require a stable
`person_db_id` for a couple ticks** before naming (identity flicker on entry is the worst
failure). Don't double-fire the intentional JT-volleyball bit. **Effort: M.**

### Name-and-point individual call-outs — **BUILD (L)**
⚠ verified: pieces mostly exist (`active_speaker.current_speaker:391`, `roast_pose:1574`) but
`conversation_cast_context` is a *prompt-context builder*, **not** a "quiet one / loud one"
selector — that selection policy is net-new, as is the bbox→direction turn. New
`_step_crowd_callout`, **reuse `do_people_roast`'s allow/boundary gates** (`_person_roast_allowed`/
`_cues`) so a stranger is never singled out. Highest embarrassment cost if identity/selection is
wrong → high confidence floor on identity AND active-speaker. **Effort: L.**

### Compare-and-rank the room — **BUILD**
⚠ verified clean extension: `do_people_roast`/`group_turn_invite` are single-target; `pair_label`
(`social_scene.py:413`, relationship-aware) + `visible_group_label` + `_person_roast_cues/
_allowed` exist. New `do_group_compare` gathers 2+ non-engaged allowed people, pulls per-person
cues, asks one comparison line. **Keep both sides flattering** (affectionate-by-construction);
all-warm/known; per-pair cooldown. Lowest safety risk of the call-out family. **Effort: M.**

### React to raised hands & crossed arms (generalize wave-back) — **BUILD**
⚠ verified: waving is the **only** gesture consumed proactively; `pointing`/`leaning_in`/
`crossed_arms`/`raising_hand` are all classified (`pose.py:_classify_gesture:251`) and ignored.
`world_state.people[].gesture` carries them; `_step_wave_reaction` is the copy template. The
hidden cost is **per-gesture debounce/probability** (single-frame gestures are noisy) — each
needs a multi-tick latch + low fire rate + per-person cooldown so it's a delight not a tic.
Restrict to the less ambiguous reads first (`raising_hand`); `crossed_arms` can misread neutral
posture as hostility. **`raising_hand` pairs with host mode** — build the gesture-latch layer
once, both consume it. **Effort: M.**

### MC / host mode — **BUILD (operator-triggered only)**
⚠ verified XL capstone: no `host_mode` flag, no MC banks, no showman governor purpose;
`COMEDY_LINE_BANKS` lacks `mc_hype`/`show_of_hands`. Per the owner call, **operator-triggered
only** — Rex MCs when told "host this," **never self-promotes** (this removes the dependency on
the autonomous room-energy detector being perfect, and sidesteps the "stays manic after energy
drops" failure mode). Still wants the gesture layer (`raising_hand` for "show of hands") + new
content banks + a new governor purpose. **Hard-exit** the instant anyone speaks directly to him;
graceful self-deprecating out when nobody plays along. **Effort: L** *(down from XL — the
operator trigger removes the autonomous-energy + self-promotion machinery).*

---

## 4. Storytelling (the "bigger bits") — **EXPLICIT-ASK ONLY**

> Owner call: long bits fire **only on an explicit ask** ("tell me a story"), never
> self-initiated. This kills the over-eager failure mode and removes the idle-fired variants.

### Tall-tale mode — **BUILD (M, was L)** *(ask-only)*
⚠ verified much cheaper than tagged: `execute_plan`/`_speak_blocking` already bypasses the
`max_sentences` trim (like `web_search`), so "lifting the cap" is just authoring a new
`JOKE_ANGLES`-style `prompt_contract` + `ActionSpec` + the **explicit-ask gate** (no idle
candidate). The cap lives in three places but the performance path sidesteps all three. Delivered
with the `dramatic_narrator` persona + pause beats; yields instantly to barge-in (`_speak_blocking`
already returns False if cut short). Personalize from a callback detail, inside the same content
ban. **Effort: M.**

### "Remember the whole saga" — multi-episode told stories — **BUILD (M, was L)** *(ask-only)*
⚠ verified cheaper: `person_story()` can be ~30 lines reusing `person_episodes`' fetch/rank/dedupe
(`episodic_recall.py:265`); `do_memory_musing` (`idle_behaviors.py:120`) is the delivery template.
The real cost is the same length-cap lift (idle musing is prompt-capped to one line) + an ordering
rule (episodes have **no thread/linkage column** — `person_id` + topic overlap is the only signal).
Per the ask-only decision, fire on an explicit "tell me about the time we…", not idle. Require ≥3
verified rows. **Effort: M.**

### "State of the room" docent bit — **BUILD (after room_model)** *(ask-only)*
⚠ verified: the temporal joke ("undisturbed for 72 hours") is **impossible without `room_model`**
(frame-only `describe_scene_detailed` can't age objects). Build it on top of the §2 object ledger;
reuse the `WEB_SEARCH_PERSONA_ADDENDUM` cap-bypass pattern (shared with tall-tale/saga). Trigger:
directed ask only. **Effort: L** *(gated on §2 room_model).*

---

## 5. Games / game-show host — **DEFERRED (someday)**

> Owner call: **skip games this round.** Kept as a someday note. If resumed, the cheapest
> high-leverage first step is the **unified `GAME_REGISTRY`** (⚠ verified: the list is triplicated
> across `games.py` dicts, `command_parser._KNOWN_GAME_NAMES:726`, and
> `action_router._GAME_START_REQUEST_RE:389`) and the **i_spy/word_assoc state-clear fix**
> (⚠ they self-clear before `_extract_game_outcome` runs — but 20q is fine, contrary to the
> brainstorm). Then DJ-announces-tracks (`S`, `DJ_START_AFTER_TTS_DELAY_SECS` exists), reveal
> stingers (needs clip assets), Name That Tune (`M`, needs a time-bounded snippet variant), Final
> Jeopardy wagers (`M`, the thinking-music asset already sits on disk unwired). **Any resumed game
> work must inject the comedy content-guardrail into the game prompt** — ⚠ `build_directive` is not
> on the game path, so games inherit zero safety by default.

---

## 6. Physical showmanship — **NO-CLIP WINS ONLY**

> Owner call: **skip the SFX/clip-dependent bits this round** (rimshot/sad-trombone/airhorn on
> Rex's own punchline, composite "show moment", reveal stingers, fuller puppeteer board) — all
> need clip assets that don't exist yet, and auto-SFX is the overuse that kills a joke. Build the
> wins that need no clips.

### Comedic-timing beat pack (double-take, eye-roll, mic-drop, etc.) — **BUILD** *(fully LLM-addressable)*
⚠ verified: 15 existing beats are emotion reactions; none is a comedy bit. Registration is
**3-paths-for-free** (LLM via `performance_plan.BODY_BEAT_NAMES`, event via `_EVENT_BODY_BEATS`,
gamepad) — each new beat = 1 runner func + 3 small map edits (the "for free" really means three
tiny coordinated edits, incl. `_BODY_BEAT_CHANNELS` so `play_body_beat` can snapshot/restore). Per
the owner call, **fully LLM-addressable now** (Rex can self-direct a mic-drop) — **add a frequency
cooldown** so it doesn't become constant mugging.
**⚠ Hardware reality — the eyes do not physically move.** They're fixed LED clusters (color/emotion
only, via `leds_head.set_eye_color`/`set_eye_emotion`), so **no beat may rely on eye motion** — the
reads come from the head/neck, the **visor**, and the heroarm. Concretely: the **eye-roll is the
visor as a brow-lift** — raise/open the visor like a cocked eyebrow (within the lens-clear floor
clamp, ≥6400, so it never blinds the camera) paired with a slow head/neck arc; it is NOT an eye
movement. The rest follow the same substitution: double-take = look away → head snap-back + visor
pop; spit-take = sharp head recoil + visor snap; mic-drop = heroarm forward-then-drop + dismissive
head turn. ⚠ `headtilt`/`elbow` are mechanically tiny; lean the reads on neck+visor+heroarm;
bench-test ~8 beats on real servos. Pairs with the batch-1 delivery profiles. **Effort: M.**

### "Thinking eyes" while the LLM works — **BUILD**
⚠ verified all hooks exist: `set_eye_color` (`leds_head.py:422`), the eye heartbeat, and the
VAD-onset hook (`interaction._begin_user_turn:329`). A slow amber/cyan breathe from turn-start
until `speech_start`, plus a triple-blink-to-gold on first token. **Critical discipline (already
codified):** the head-LED link drops bytes during speech (`leds_head.py:501`) — keep the update
rate low and confine the animation to the **silent transcription→LLM gap** before `_speaking` is
set. Optionally only fire when the wait exceeds a threshold. **Effort: S.**

### Smug-after-a-good-roast sustained mood — **BUILD** *(batch 1)*
⚠ verified: `smug` is already an alias of `proud` (`body_mood.py:97`); the machine + the
`made_laugh → giddy` precedent (`consciousness.py:2762`) exist — this is **one new call site**
`set_mood('smug', source='roast_landed')`, plus an optional sulk-after-flop. Modest intensity so
face-tracking stays primary (`set_mood` already guards a weaker mood stomping a stronger one).
Decide "landed" = fire on `humor.roast` completion vs wait for an audience laugh. **Effort: S.**

### Deferred §6 notes
- **Rimshot/sad-trombone/airhorn on his own punchline**, **composite show-moment**, **reveal
  stingers**, **fuller puppeteer board** — all need the clip pack first (⚠ only `Air Horn.mp3`
  exists); ⚠ the post-punchline pause only cleanly exists on the `quick_punchline` joke path, and a
  clip fired during live TTS is dropped by the output gate.
- **Beat-synced DJ dance mode** — ⚠ `arm_rhythm_tick` is a complete function with **zero callers**;
  needs a beat signal the playback loop doesn't emit (librosa is installed; live beat-tracking adds
  latency/CPU). The chest-per-beat flash is also a firmware gap. `L`, later.
- **Roll-up / mock-retreat base bits** — **CUT:** the base is not show-ready (slow caps, ToF
  avoidance still a stub, calibration unfinished, topple risk). Revisit at motion Phase 1.

---

## 7. Usefulness & smart home — **CUT**

> Owner call: **cut the entire smart-home cluster.** ⚠ verified 100% greenfield anyway — no timer,
> reminder, light control, device transport, `announce`, or scheduled-callback primitive exists,
> and Alexa cannot be driven directly. Home Assistant, Alexa-via-switch, timers/reminders, the
> announce primitive, show-cue, Movie Time, and the "act" half of notice→offer→act are all out of
> scope.
>
> **One item rescued and moved to §8:** "inject open plans into the live reply" is pure
> conversation (events → `_build_person_context`), not home control.

---

## 8. Personalization & running gags from memory

### Inject open plans into the LIVE reply — **BUILD** *(rescued from §7 — cheapest real win)*
⚠ verified exactly: `_build_person_context` (`llm.py:572`) only reads `emotional_events`; the
calendar readers `get_open_events`/`get_upcoming_events` (`memory/events.py:249/264`) are never
called there, so mid-conversation Rex doesn't know you have a thing tomorrow. Add a short "Open
plans they mentioned: X (on DATE)" block with a restraint rule, gated against the
`_anticipated_events` throttle (`consciousness.py:181`) to avoid double-mentioning with the
proactive path. Inject only the next 1-2 within N days; prefer dated over undated (undated nags).
~15-30 lines. **Effort: S.**

### Open commitments ("you SWORE you'd fix that sensor") — **BUILD**
⚠ verified: the `status` column + cancel/reschedule machinery exist (`events.py:160/215/229`); a
`status='promised'` value is schema-compatible but the **commitment regex + injection are net-new**.
The hard 80% is the linguistic gate — a **tight first-person future-intent regex** that excludes
"I should really…" and is cleared by cancel/postpone phrases. Recommended default: **one dry needle
on return as background context** (not an authored accusatory cold-open); clears on explicit
done/never-mind; reuse the anticipation throttle so it doesn't double-mention with `get_pending_
followups`. **Effort: M.**

### Durable honorifics from positive milestones — **BUILD** *(celebration-sourced for now)*
⚠ verified: `inside_joke` category exists and is **HIGH_IMPORTANCE / slow-decay** (`facts.py:52`) —
so naively writing the honorific as an `inside_joke` fact fights the "decays / dethrone-on-loss"
intent; **use an explicit expiry**, not the category default. Since games are deferred, source
honorifics from **celebrations/milestones** (`emotional_events.get_due_celebrations`/
`is_celebration_event`) for now — game-win honorifics wait for §5. One honorific per person at a
time; dethrone on a contradicting outcome (being wrong can itself be a joke). Tie expiry to the
celebration's decay window. **Effort: M.**

### Gossip discharge: the knowing look when the subject walks in — **BUILD**
⚠ verified catch: `pre_met_note` is only written when the subject is a **brand-new** person row
(`interaction.py:8765`), so a returning/known briefed subject won't have one — **trigger off any
`told_by`-tagged fact on the recognized person, not `pre_met_note`.** The suppression wall is real
(`facts.format_fact_for_prompt:633` — "NEVER repeat or hint at" unkind gossip) — **the comedy is the
meta framing only** ("Oh. So YOU'RE {name}. I've heard… things"), never the gossip text.
**Hard-gate `crowd_count==2`** (just the two relevant people; reuse the emotional-events
crowd-discretion pattern); **unkind gossip never surfaces, even meta**; one-time per subject.
**Effort: M.**

### Feed the callback bank its best material — **BUILD (about-present-people only, hard-gated)**
⚠ verified double-walled today: the bank sees **only the speaker's own first-person turns**
(`bank_from_turn:525`), hard-drops third-party material (`_looks_like_thirdparty_premise:497`), and
never banks Rex's own lines (recall filters non-Rex speakers, `:625`). Per the owner call:
**bank only warm/safe things said ABOUT a person who is physically present**, behind a **new
deterministic wall (present + warm)** — and **do NOT bank Rex's own bits** (the "remember when I was
funny?" flavor was rejected as tiresome). This deliberately relaxes the speaker-only invariant, so
the new path needs its own hard gate, separate from the existing one. Feeds the running-gag engine
(§1). **Effort: M.**

### Per-person smart-home anticipation — **CUT** *(depended on §7).*

---

## The big swings — revised after scoping

> E-tickets are fusions of the above. After the scope cuts, three survive intact-ish, two are
> diminished, three are cut.

### E-1 — "The Reunion" — **KEEP (reframed as an upgrade)**
⚠ The absence detector already exists and ships a real long-absence greeting today
(`_pick_absence_phase:3894`, 60-day threshold). This is **upgrade the existing greeting into a
staged moment**, not build from scratch: eyes-amber → beat of silence → `directed_look_pose` →
`excited_burst` → mood-slam, with a one-turn greeting **lockout**, paying off into a `person_story()`
monologue (§4, ask-or-reunion) + the open-commitments needle. **Risk:** a face-recognition false
positive would fire a loud withheld-greeting performance at the wrong person — the lockout amplifies
that, so gate on identity stability. **New glue:** the staged sequence + lockout through the governor.

### E-2 — "Story Time" — **KEEP (diminished: LED staging, ask-only)**
With §7 cut, there's no `show_cue`/`home.scene` — stage with **chest-LED dim + duck his own music**
(`dj.set_volume`/`volume_down` exist). `dramatic_narrator` persona + the tall-tale cap-bypass,
ask-only. ⚠ `dramatic_narrator` and `show_cue` don't exist yet — both net-new. The false-climax
prompt scaffold is the one real authoring challenge.

### E-3 — "Roast Battle" — **CUT** *(off-brand; the guardrail isn't even on the game path).*

### E-4 — "Cantina Open" — **DIMINISHED (defer)**
Loses `home.scene` (cut) and the leaderboard + auto-DJ-announce (games deferred). What survives is a
once-per-window first-arrival ritual + dance mode (needs the beat signal, deferred) + an episodic
callback + the open-plans bulletin. Keep as a **later** note; revisit if games/dance resume.

### E-5 — "The Long Con" — **DIMINISHED**
Loses the cross-session scheduled-callback payoff and `show_cue` (both §7-adjacent, cut). Survives as
the **`running_bit` escalation (silent, §1) + the `appliance_conspiracy` persona** — an evolving
multi-session frame without the staged scheduled reveal. ⚠ `appliance_conspiracy`/`toaster` and a
cross-session scheduled callback are both vaporware today.

### E-6 — "Name That Tune: Showdown" — **CUT** *(games deferred).*

### E-7 — "The Welcome Committee" — **KEEP (diminished: no home offer)**
⚠ The consecutive-tick confirmation primitive exists (`consciousness.py:294`,
`FACE_UNKNOWN_CONFIRM_FRAMES`) — reuse it for an arrival identity-stability gate. Chain: named
call-out (+ warm departure) → episodic brief → open-plan needle. The **signature handle** is now a
thin layer over callbacks (§1, buildable); the **per-person home offer is cut**. ⚠ There is **no
door-direction sensing** in the build — degrade to a plain face-arrival (no directional snap).
**New glue:** the identity-stability arrival gate + chaining call→brief into one sequence.

### E-8 — "Movie Night" — **CUT** *(depended entirely on Home Assistant).*

---

## Revised build order (maximizes fun-per-week, post-scope)

1. **Batch 1 — Comedy felt immediately:** delivery profiles (start deadpan + smug) → smug-after-a-
   roast mood → comedic personas. All ready, no prereqs; the change a human feels instantly.
2. **The COCO unlock → `world_state.objects`** (+ GUI bounding boxes) — the substrate for all of §2.
3. **Reactions in parallel:** land-the-laugh / take-a-bow · the gesture-reaction layer · named
   arrivals + departures (build the bbox→direction primitive here, shared with call-outs).
4. **Cheap conversation wins:** inject open plans into the live reply · open commitments.
5. **room_model (L)** → then object-grounded curiosity, change detection, the docent bit (ask-only).
6. **The roast lane:** sharp tier (warmth-gated) · running-gag escalation · tease-the-obsession ·
   signature handle (thin over callbacks).
7. **Storytelling (ask-only):** tall-tale → `person_story` saga → E-1 Reunion staging.
8. **Crowdwork depth:** room-energy read (shadow-log) · compare-the-room · name-and-point call-outs ·
   host mode (operator-triggered) · E-7 Welcome Committee.

## Top items (ranked, post-scope)

1. **Comedic delivery profiles** — the single change felt immediately; every joke under-lands today
   because a roast and a condolence sound identical. **M.**
2. **Free the COCO detector → object inventory (+ GUI boxes)** — the zero-cost data substrate that
   kills "generic curiosity" at the root and feeds 5+ §2 ideas. **M.**
3. **Land-the-laugh / take-a-bow** — Rex is currently deaf to laughter *and* applause; the signal's
   already computed. **M.**
4. **Running gags that escalate** — resurrects dead `running_bit`; turns callbacks into bits that
   recur (silently) instead of decaying. **M.**
5. **Comedic-timing beat pack** — 8 LLM-addressable physical bits a room reads instantly. **M.**
6. **Energetic named arrivals + departures** — identity is already in hand and discarded; clock people
   by name. **M.**
7. **Warmth-earned sharp roast tier** — gives the roaster energy a place to live; warmth-gated. **M.**
8. **Inject open plans into the live reply** — the cheapest real "more useful" win, fully ready. **S.**
9. **Smug-after-a-roast mood + thinking eyes** — two `S` personality wins from existing machinery.
10. **room_model + object permanence** — the L investment that makes object-grounded curiosity,
    change-detection, and the docent bit possible. **L.**

## Open tuning knobs (recommended defaults — adjust in build)

- **Delivery profiles:** start with **deadpan + smug** only (limit TTS cache regen); add
  mischief/dj_hype later; keep `similarity_boost` high. Empathy outranks comedy on emotional turns.
- **Sharp tier:** warmth-alone gate is loosest → keep the **harsh-word governor backstop** and define
  `sharp`-vs-`normal` concretely; watch `roasted_sincere`.
- **Beat pack:** LLM-addressable + a **frequency cooldown**; bench-test the 8 beats on real servos.
- **Named arrivals:** require ~N stable identity ticks before naming; rib at warm/sparring tier, plain
  welcome otherwise.
- **Crowd curiosity:** address the **group** by default (don't single out a guest); ceiling on big
  crowds.
- **React-to-you:** cache on the periodic scan (not every turn); confidence gate + per-person cooldown.
- **Gossip discharge:** `crowd_count==2` hard gate; meta-only; unkind gossip never surfaces.
- **Open commitments:** one dry needle as background context, not an accusatory cold-open.

## Hard problems / open questions (updated)

- **Raising the roast ceiling is the exact failure mode the 80/90→55 rebalance fixed.** With the
  **warmth-alone** gate (loosest), this is the highest-risk change in scope: the harsh-word governor
  regex must still catch genuinely cruel output, and `roasted_sincere` will likely tick up (accepted).
- **Longer bits fight the rest of the system** — but ask-only + the `_speak_blocking` cap-bypass keeps
  it bounded. Tall-tale/saga must yield instantly to barge-in (the path already does) and never
  self-initiate.
- **Detector noise vs crying wolf.** COCO/`room_model`/change-detection/clothing-gesture callbacks all
  ride single-frame vision that flickers indoors. Every one needs a confirm-streak + per-object/
  per-person cooldown. The **Rich** posture persists indefinitely, so a stale object needs a sanity
  check, and **screens/devices are dropped at the detector** to honor the no-screens rule.
- **TTS cache thrash from voice profiles.** Each delivery profile is a fresh cache key → first-use
  regen. Start with two profiles.
- **Memory-driven personalization risks creepiness.** Room model, gossip discharge, and the
  about-present-people callback path must inherit the existing carve-outs — in-room-volunteered facts
  only, never log screens/text, gossip never recited, crowd-discretion on sensitive material.
- **`directed_look_pose` doesn't turn toward a named person** — build the bbox→direction primitive
  once; named arrivals and call-outs both need it.
- **(Deferred) games outcome capture** — i_spy/word_assoc clear state before `_extract_game_outcome`;
  fix that *before* any leaderboard if games resume (20q is fine, contrary to the original brainstorm).
