# DJ-R3X v2 — "More Fun" Improvement Brainstorm

> **Purpose:** a handoff/idea-catalog for making Rex funnier (more jokes, more of a
> roaster/showman), more genuinely curious about his surroundings, better at crowdwork,
> more useful than a party trick (incl. smart-home), and capable of bigger bits like
> storytelling. Every idea below is grounded in machinery that **already exists** in the
> codebase — file:function hooks are named so this can be picked up cold in another window.
>
> Generated 2026-06-25 from a code-grounded multi-agent pass over comedy / curiosity /
> proactive / vision / games / memory / hardware / smart-home subsystems.
>
> **Effort tags:** `S` = a weekend · `M` = ~a week · `L` = multi-week · `XL` = a project.

## The core insight

Rex already has the entire nervous system of a showman — an action-governor choke point, a
comedy-mode selector, a callback engine, an episodic diary, a body-beat catalog, a
soundboard, a DJ, and a clean action-router spec system. **The problem isn't missing
organs; it's that the organs are tuned conservative, siloed, and rarely talk to each
other.** Almost everything below is *wiring two things that already exist together* or
*lifting a deliberate cap*.

Two findings are almost embarrassing in their upside:
- Rex **sees every object in your room and throws the data away** (the COCO detector — see §2).
- Rex **can detect laughter and applause and does nothing with it** (the audio scene flags — see §3).

---

## The three things to do first (zero setup, transformational)

1. **Make him react to laughter & applause.** `audio/scene.py` *already computes*
   `laughter_detected` and `applause_detected` every cycle — read only as passive prompt
   flavor. **Nothing reacts.** Add a `_step_room_reaction`: laughter shortly after a Rex
   line → a follow-through (*"see, that one's free"*); applause → take a bow
   (`proud_dj_pose`). A comedian who can't tell if the room laughed isn't a comedian. This
   is also the live "judge" signal that makes Roast Battle / Name That Tune feel alive.
   **Hooks:** `audio/scene.py:_analyze_cycle`; `speech_engine.generate_and_speak`;
   `action_governor._PURPOSE_PRIORITIES`; `sequences/animations.py:play_body_beat('proud_dj_pose')`;
   wave-back latch pattern. **Effort: M** *(gate on a recent-Rex-utterance window so he
   doesn't smugly react to ambient noise/music).*

2. **Free the object detector.** `vision/animal_detector.py` runs a full MediaPipe **COCO
   ObjectDetector — 80 object classes (mug, laptop, guitar, plant, bottle, chair, tv…)**
   locally every ~2s at **zero OpenAI cost**, then `_best_animal_category` *discards every
   detection* whose label isn't one of 10 animal types. Generalize it behind a new
   `LOCAL_OBJECT_DETECTION_SPECIES` allowlist and publish to a new `world_state.objects`
   field, mirroring `detect_animals_local`. Position + bbox + confidence are already
   computed. This is the data substrate that fixes "curiosity feels generic" at the root.
   **Hooks:** `vision/animal_detector.py:_best_animal_category`/`_records_from_detections`;
   `vision/scene.py:detect_animals_local` pattern; `world_state.py`; config
   `LOCAL_ANIMAL_DETECTION_SPECIES` (new sibling). **Effort: M** *(reuse the existing
   confidence floors + confirm-streak so indoor flicker doesn't churn).*

3. **Comedic delivery profiles.** A savage roast and a sincere condolence are spoken with
   the *identical* ElevenLabs voice settings — delivery shaping is owned entirely by the
   empathy/surprise path. Add a `comedy_mode → voice+timing` map mirroring
   `empathy._MODE_VOICE_SETTINGS`: **deadpan** (high stability, low style, ~250ms
   post-beat button), **smug** (mid), **mischief** (lower stability), **dj_hype**
   (low stability/high style). Pair each with a pre-punchline pause and a matching body
   beat. A beat of silence after a deadpan line is the difference between a joke and a
   sentence; every existing joke under-lands today for lack of this.
   **Hooks:** `intelligence/interaction.py` delivery seam (non-stream + stream);
   `intelligence/empathy.py:_MODE_VOICE_SETTINGS` pattern;
   `intelligence/comedy_modes.py:ComedyMode`; `performance_plan.py` body beats.
   **Effort: M** *(each profile changes the TTS cache key → first-use regen; keep to a few
   profiles and keep `similarity_boost` high).*

---

## 1. Comedy & roasting

### Warmth-earned "sharp" roast tier
Today the roast ceiling caps at `normal` — a best friend in a mutual-roast relationship
gets the *same maximum heat* as a stranger. The downgrade machinery is rich; the upgrade
machinery is absent. Add a `sharp` tier that `_roast_level` only returns when effective
warmth ≥ `ANTAGONISM_TIER_CAPS_LIFT_WARMTH` AND banter history exists AND none of the
existing downgrades (tender/sad/boundary/flat/child/micro) apply. `sharp` loosens the slim
directive's single-beat cap to allow a real two-beat roast. Strangers and sincere moments
stay exactly as protected — the heat is *earned*.
**Hooks:** `intelligence/social_frame.py:_roast_level`, `roast_rule`/`_slim_roast_rule`,
`govern_response` filters; config `BANTER_WARMTH_THRESHOLD` /
`ANTAGONISM_TIER_CAPS_LIFT_WARMTH` / `TIER_ROAST_STYLE`.
**Effort: M** *(must keep the `roasted_sincere` quality eval green — this is the exact
failure mode the 80/90→55 rebalance fixed; highest-risk comedy change).*

### Running gags that escalate (resurrect the dead `running_bit` category)
Callbacks fire as *decaying* one-offs — the more a premise fires, the more
`freshness_factor` suppresses it, which is backwards for a running joke. The `running_bit`
category is defined in `memory/callbacks.py:CATEGORIES` but **never written or read** —
pure dead scaffolding. When a banked premise lands ~3 times (`use_count` is already
tracked), promote it to `running_bit`, exempt it from freshness decay, and branch
`build_callback_directive` to reference the history — *"third time the 3D printer's come up
tonight."* The room gets in on it; the laugh grows each repetition.
**Hooks:** `memory/callbacks.py:CATEGORIES`/`mark_used`/`use_count`/`freshness_factor`;
`intelligence/callback_engine.py:settle_turn`/`build_callback_directive`.
**Effort: M** *(hard fire cap + cross-session cooldown so a beloved gag doesn't go stale).*

### Comedic personas (smug / appliance-conspiracy / dramatic-narrator)
`select_mode` picks from thin one-beat stances with no recurring *character* bits. Add 2-3
new `ComedyMode` stances with distinct directives: **smug_superiority** (clinical
condescension about organics — his core voice), **appliance_conspiracy** (a one-sided
rivalry with the toaster as a recurring frame), **dramatic_narrator** (narrate a mundane
moment like an epic). Each pairs with a delivery profile. Repeatable, recognizable bits —
*"oh, he's doing the appliance thing again"* — instead of a fresh improvised quip each time.
**Hooks:** `intelligence/comedy_modes.py:_MODES`/`_SLIM_STANCE`/`select_mode` (reuses
premise-rotation + anti-repeat). **Effort: S**

### Tease-the-obsession lane
The funniest, most loving roast is about *how much someone cares about their thing* — and
that exact material is fenced off. `conversation_steering._directive_for` caps humor to
"light roasts only about Rex's ignorance, not the person's competence." Relax it so, for
warm relationships, the affectionate roast can target the *depth* of the obsession —
*"you have a SPREADSHEET for nebulae, of course you do"* — while never mocking competence.
**Hooks:** `intelligence/conversation_steering.py:_directive_for`;
`social_frame._slim_roast_rule` for the 'interest' purpose. **Effort: S**

### Per-person signature roast handle
A good roaster has *one perfect recurring angle* per person, not random facts. From a
person's high-confidence, explicit, safe facts (`facts.py` already ranks these), pick ONE
durable trait as their personalized roast handle and expose it in `_relationship_tone_rule`
for warm/sparring tiers. Rotate among the top 2-3 seeds so it never fixates.
**Hooks:** `memory/facts.py:get_prompt_worthy_facts`/`score_fact_for_prompt`;
`intelligence/llm.py:_relationship_tone_rule`; reuse `callback_engine`'s
protected-category wall to vet the seed. **Effort: S**

---

## 2. Curiosity & a persistent mental model of the room

> The owner's complaint — "curiosity feels weak/generic" — has a single root cause:
> **Rex has no memory of the room.** Scenes are stored as fuzzy free-text captions,
> clustered into one "vibe," and pruned. There's nothing for him to be *specifically*
> curious about. The COCO unlock (top of doc) is the substrate everything here stands on.

### Room baseline + object permanence ("that mug's been there three days")
Add `memory/room_model.py` backed by a new `rex.db` table (label, location bucket,
first_seen, last_seen, sighting_count), fed by the `world_state.objects` stream. Feed a
compact "remembered room" summary into the curiosity prompt via the `conversation_agenda`
directive seam so Rex references longevity and novelty: *"that plant's been dying in the
corner since Tuesday."* Object permanence is the Imagineering move — it proves he actually
*lives* in the space.
**Hooks:** `memory/rex_db.py` + `setup_assets.py` schema; new `memory/room_model.py`;
`consciousness.py:_step_visual_curiosity` prompt;
`conversation_agenda._PROACTIVE_RULES['visual_curiosity']` — **loosen its current
no-memory clause**. **Effort: L** *(coarse location buckets only — left/center/right +
fg/bg; inherit episodic test-suppression; never log screens/text — safety rules already
forbid this).*

### "Wait — that's new": live change detection
The only change-detector that exists (`episodic_hooks` scenery remark) is boot-to-boot,
fuzzy-caption-based, and **disabled by default**. Run a cheap diff of `world_state.objects`
vs the room_model each idle tick; when a new/departed object crosses a confirm-streak, fire
a `do_noticed_change` behavior through the governor — a genuine double-take: *"Hold on. Is
that a new monitor? You've been holding out on me."* The unprompted double-take is the most
human form of curiosity.
**Hooks:** `world_state.objects` vs `memory/room_model.py`; `intelligence/idle_behaviors.py`
(new `do_noticed_change`); `consciousness.py:_idle_micro_behavior_choices`; reuse
`episodic_hooks._maybe_queue_scenery_remark`/`take_scenery_remark`; flip
`SCENERY_CHANGE_REMARK_ENABLED` on. **Effort: M** *(confirm-streak + per-object cooldown so
he doesn't cry wolf on flicker).*

### Object-grounded curiosity questions that are actually good
Replace the stateless "name a visible object and ask about it" in `_step_visual_curiosity`
with a prompt that prioritizes the most novel/longest-present/most-personal object from the
room_model, crossed with the person's known interests. *"You've got three guitars but I've
never heard you play — which one's the favorite?"* instead of *"what's that?"*
**Hooks:** `consciousness.py:_step_visual_curiosity`; `memory/room_model.py`;
`conversation_steering.build_context` for interest overlap. **Effort: S** (once room_model
exists).

### Room-and-person-aware POV seeds
`REX_POV_SEEDS` are 100% Rex-internal navel-gazing (venue- and person-neutral by design) —
which is exactly why "curiosity feels generic." Add seeds tagged to fire when specific
context is present (a detected object class, the active steering interest, an unknown face),
and extend `rex_pov._context_signature` to surface those tags.
**Hooks:** `config.py:REX_POV_SEEDS`; `intelligence/rex_pov.py:_context_signature`/`_choose`;
`world_state` snapshot + steering active topic as new tags. **Effort: S** *(must stay "a
category of thing," never a hallucinated prop — the existing directive already bans claiming
to see a specific object).*

### React to YOU (clothing / pet / object-you're-holding callbacks)
On a turn where `describe_scene_detailed` sees a person, diff their clothing/accessories/
held objects against the last sighting (lightweight per-person visual memory). Fire short
reactive beats: *"New jacket? Bold." / "You brought the dog!" / "Is that... a sandwich?
Don't mind me."* Noticing what someone walked in *with* is the warmest crowdwork.
**Hooks:** `vision/scene.py:describe_scene_detailed` people[]/notable_details;
`idle_behaviors.py:do_appearance_riff` pattern; small per-person visual cache.
**Effort: M** *(require a confident read + per-person cooldown; single-frame clothing
detection is unreliable — don't "new jacket?" the same jacket).*

### Let curiosity work IN a crowd
`_step_visual_curiosity` **hard-disables itself when crowd_count > 2** — Rex's curiosity
literally switches off exactly when there's a room full of people to be curious *about*.
Replace the hard return with a crowd-aware branch: ask a curious question about the *group*
or a specific visible person/object. *"What's the story with the three of you?" / "Okay,
who brought the guitar?"* This fixes the owner's complaint at the exact line that causes it.
**Hooks:** `consciousness.py:_step_visual_curiosity` crowd gate;
`social_scene.conversation_cast_context`; `VISUAL_CURIOSITY_MAX_CROWD_COUNT`. **Effort: S**

---

## 3. Crowdwork & reading the room

### Land-the-laugh / take-a-bow
**The single biggest untapped crowdwork signal** — see "do first" #1 above. `audio/scene.py`
already sets `laughter_detected`/`applause_detected`; nothing reacts. **Effort: M**

### Room-energy read → governor boost (hot/warm/cold)
A `performance` social_mode tier already exists and is computed every tick — but *almost
nothing branches on it.* Add a `room_energy()` helper fusing group size +
`laughter_detected` + `group_chatter_detected` + visible smiles into hot/warm/cold, surface
it on `SituationProfile`, and in `action_governor._score` give crowd/showman purposes a
bonus when energy is hot while damping sincere 1:1 purposes. "It just knows" magic — felt,
not announced.
**Hooks:** `awareness/situation.py:SituationProfile`; `awareness/social.py`;
`audio/scene.py`; `_person_is_smiling` in consciousness; `action_governor._score`.
**Effort: M** *(ship in shadow-log mode first before wiring into scoring so he can't go
manic in a loud room).*

### Energetic, named arrivals & departures ("look who it is")
Crowd-change reactions speak generic count-based lines ("the crowd shifted from pair to
alone") and don't *name* who arrived even when identity is known. Upgrade so that when an
arriving/departing slot resolves to a known person, Rex greets/ribs them BY NAME with
energy — *"JT! The volleyball menace returns!"* — and turns toward the door via
`directed_look_pose`. Keep the generic line only as the unknown-person fallback. Walking in
and having the droid clock you by name is the most memorable, repeatable show moment there
is.
**Hooks:** `consciousness.py:_step_proactive_reactions` new-person block/`_crowd_change_settled`;
`social_scene.from_snapshot`; vision face names; `directed_look_pose`. **Effort: M**
*(require a stable `person_db_id` for a couple ticks before naming — identity flicker on
arrival).*

### Name-and-point individual call-outs
The defining crowdwork move, and Rex has every piece but never assembles them. New
`_step_crowd_callout`: when social_mode is small_group/crowd and warm, pick a target via
`social_scene` referent candidates + active-speaker (the quiet one, the loud one, the newest
arrival) and fire a ribbing call-out *while physically turning to them* via
`directed_look_pose(target=name)`.
**Hooks:** `social_scene.conversation_cast_context`; `vision/active_speaker.py:current_speaker`;
`vision/pose.py` gestures; `sequences/animations.py:directed_look_pose`/`roast_pose`.
**Effort: L** *(gate hard on friendship tier + family_safe + warmth; reuse `do_people_roast`'s
allow/boundary gates so a stranger never gets singled out).*

### Compare-and-rank the room
`do_people_roast` and `group_turn_invite` are strictly single-target. Extend into a
`do_group_compare` path: when 2+ non-engaged people are visible, build a comparison from
`social_scene.pair_label` + per-person cues. *"Bret looks like he runs the meeting; JT looks
like he survives it."* Comparisons implicate everyone present, so everyone leans in.
**Hooks:** `idle_behaviors.py:do_people_roast`; `social_scene.pair_label`/`visible_group_label`;
`_person_roast_cues`/`_person_roast_allowed`. **Effort: M** *(affectionate only, no protected
attributes, cap to known/warm people).*

### React to raised hands & crossed arms (reuse the wave-back pattern)
The wave-back bit is the best-built repeatable physical bit in the codebase
(detect→latch→escalate→mirror→debounce). But waving is the *only* gesture consumed
proactively — `pointing`, `leaning_in`, `crossed_arms`, `raising_hand` are all classified by
`vision/pose.py` and then ignored. Generalize `_step_wave_reaction` into a per-person
gesture layer: *"You've got a question, I can feel it." / "Tough crowd — arms crossed, I see
you."*
**Hooks:** `vision/pose.py:_classify_gesture` → `world_state.people[].gesture`;
`consciousness.py:_step_wave_reaction` template; `directed_look_pose`/`nod`. **Effort: M**
*(latch over a couple ticks — gestures are single-frame and noisy; keep probability low so
it's a delight not a tic).*

### MC / host mode when the room fills up
There is no behavior that turns up showman energy when the room is full and lively. Add a
`host_mode` flag derived from social_mode='crowd'/'performance' + warm energy that (a)
raises showman governor priorities, (b) swaps in MC line banks, (c) periodically runs the
room — *"alright, show of hands, who actually likes Mondays?"* — and watches for raised
hands to react.
**Hooks:** `SituationProfile.social_mode`; `action_governor._PURPOSE_PRIORITIES`;
`COMEDY_LINE_BANKS` (+ new mc_hype/show_of_hands banks); `vision/pose.py:raising_hand`;
`excited_burst`/`arm_wave`. **Effort: XL** *(easy to become obnoxious — strong cooldowns,
hard exit when energy goes cold, shadow-log first, yields instantly to a direct turn).*

---

## 4. Storytelling (the "bigger bits")

### Tall-tale mode
The owner explicitly wants bigger bits — and they're *structurally impossible* today because
`build_directive`/`build_slim_directive` tell the model "one joke shape only; no stacked
punchlines; if the answer's funny enough, stop" on essentially every turn. Add a
`humor.tall_tale` action that **sanctions a 3-5 sentence absurd Star-Tours/flight-record
story**, lifting the single-beat cap *only* for this gated action (explicit ask or an idle
"room is dead" candidate). Delivered with the `dramatic_narrator` voice profile + pause
beats, pulling a detail from the live person's callbacks so the tale is personalized.
**Hooks:** `performance_plan.py:JOKE_ANGLES`/`plan_for_action` +
`interaction._handle_router_performance_action`; the single-beat cap in
`comedy_modes.build_directive`; `action_router` action set. **Effort: L** *(must bypass the
HARD LENGTH LIMIT cleanly only for this action, and yield instantly to a barge-in so a long
bit can't trap the user — model it on how `web_search` already buys multi-sentence room).*

### "Remember the whole saga" — multi-episode told stories
The diary captures richly (*"I made Bret smile about his fantasy team," "I played trivia with
you"*) but recall is single-shot — `person_episodes` returns isolated one-liners and
`_pick_episodic_callback` weaves at most ONE. Nothing stitches related episodes into a
*story*. Add `episodic_recall.person_story()`: cluster 3-5 related episodes about one person
into an ordered mini-narrative + a `do_tell_a_story` idle behavior.
**Hooks:** new `episodic_recall.person_story` beside `person_episodes`;
`idle_behaviors.py:do_memory_musing` as template; `action_governor._PRIORITIES`.
**Effort: L** *(require ≥3 verified rows, hard word limit, idle-only so it never blocks a
live reply).*

### "State of the room" docent bit
When asked "what's going on?" or during a crowd lull, have Rex narrate the room as an
over-the-top museum-docent / nature-documentary bit, grounded entirely in the room_model and
live objects: *"To my left, a noble coffee mug, undisturbed for 72 hours — a monument to
procrastination..."*
**Hooks:** `memory/room_model.py` (object ages/novelty); `describe_scene_detailed` for live
color; a directed-attention trigger phrase; reuse the `WEB_SEARCH_PERSONA_ADDENDUM` pattern
to buy multi-sentence room. **Effort: L** *(prerequisite: room_model from §2; bound it,
opt-in/triggered not idle-fired).*

---

## 5. Games / game-show host

### DJ Rex announces every track he drops
Music starts cold like a jukebox today — `dj.play()` *knows* the title/artist but never
speaks them, and the hype path (`performance.dj_bit`) is deliberately *disjoint* from
playback. Fix: on `dj.play()`, speak a one-line in-character intro using the known TrackInfo
— *"Pulling this one out of the crate... 'Mr. Blue Sky.' Try not to embarrass yourselves."*
— routed through `performance_plan`, with music starting ~0.25s later
(`DJ_START_AFTER_TTS_DELAY_SECS` already exists).
**Hooks:** `features/dj.py:play`/`now_playing`; `performance_plan.py:plan_for_action`;
`DJ_START_AFTER_TTS_DELAY_SECS`. **Effort: S** *(skip on barge-in / rapid re-requests).*

### Host-grade reveal stingers across all games
Jeopardy has audio stingers; nothing else does. Give every game the game-show audio
language: drumroll before a 20-Questions lock-in, "ding" on a correct trivia answer, buzzer
on a whiff, airhorn on a win — all via the existing `soundboard.play()` /
`_jeopardy_queue_clip` pattern. Add a `drumroll`/`anticipation` body beat and map
`game.reveal` to it.
**Hooks:** `audio/soundboard.py:play`; `_jeopardy_queue_clip`; `_body_beat`;
`performance_plan.py` game.* map. **Effort: S** *(sequence stinger-then-line — the output
gate drops a clip over live TTS; need the actual clip assets).*

### Name That Tune
The most-requested party-music game, and Rex *already has every piece* — music index, fuzzy
matcher (`dj.handle_request`'s WRatio), audio pipeline — just never assembled. Pick a random
local MP3, play a 5-10s snippet, take guesses with fuzzy title/artist matching, scored
across rounds with buzzer/ding stingers and escalating roasts on whiffs.
**Hooks:** `features/games.py:_GAME_HANDLERS`; `features/dj.py:handle_request`/`_playback_loop`/
`now_playing`; `audio/soundboard.py:play`. **Effort: M** *(needs a time-bounded snippet
variant — don't reuse the full-song loop; gate on library size >N).*

### Final Jeopardy with real wagers
Jeopardy is genuinely strong (real clue board, multi-player scoring, daily doubles, timeout
timer, voiceprint auto-enroll) but has **no Final round** and wagering is faked
(auto-double). Add round 3: reveal the category, each player wagers up to their score, one
final clue, the real jeopardy-theme.mp3 (already in `_JEOPARDY_CLIPS`) during thinking time,
then the dramatic reveal and a roast for anyone who bet it all and bombed.
**Hooks:** `features/games.py:_jeopardy_complete_round_or_finish`, `_JEOPARDY_CLIPS` theme;
`features/jeopardy.py:build_board`/`is_correct`. **Effort: M** *(clamp + confirm far-field
wager parsing; handle zero/negative scores).*

### Persistent crowd leaderboard + rivalries
The **only** persistence today is a prose episode ("scored 4 out of 5"). No scores table, no
per-person tally, no reigning champion. Add a `game_scores` table +
`record_game_result(person_id, game, won, points)` called wherever games finish, and surface
standings as a `callback_engine` premise so it leaks into normal conversation — *"Bret,
you've lost to me at trivia four times running."*
**Hooks:** `memory/episodes.py:record_game_played`; `memory/people.py`;
`features/games.py:_extract_game_outcome`; `callback_engine`. **Effort: L**
*(⚠ several games clear state before the outcome hook runs — `_extract_game_outcome` returns
'' for i_spy/20q/word_assoc; fix those first or the leaderboard data is empty).*

### Roast Battle (head-to-head, audience-judged)
A new game leaning *directly* into the roaster/showman identity: two named players trade
Rex-curated setups, Rex MCs each round's premise, tosses in his own roast as the undefeated
third contestant, scores crowd reaction (reuse laughter signals from §3, else manual "who
won?"), and crowns a winner with an airhorn. Pure verbal — no camera/music needed.
**Hooks:** `features/games.py:_GAME_HANDLERS`; `comedy_modes.line_for`/`COMEDY_LINE_BANKS`;
`audio/soundboard.py`; `intelligence/user_energy.py`. **Effort: L** *(the standing "no
body/health/identity/private-fact" guardrail from `comedy_modes.build_directive` MUST carry
into the game prompts or roasts turn mean).*

### Live trivia about the people in the room
Generic trivia is replaceable by any app; trivia *about the actual humans present* is
something only Rex-with-memory can do. A trivia variant that generates questions from
episodic/person memory — *"Who here has a dog named Biscuit?"* — reusing the round/scoring
shell, with Rex roasting whoever forgets a fact about their own friend.
**Hooks:** `features/games.py:_trivia_*` shell; `intelligence/memory_query.py`;
`memory/episodes.py`; `memory/people.py`. **Effort: L** *(only use in-room-volunteered
facts, same carve-out as banked callbacks; graceful fallback to standard trivia when memory
is sparse).*

### Word association with a speed clock
The party version lives or dies on tempo, and today there's none. Add a soft per-turn timer
(reuse the jeopardy answer-timeout pattern) so a slow answer is a "break," and have Rex speed
up and get cockier as the chain grows. Track longest chain as a persistent high score.
**Hooks:** `features/games.py:_wordassoc_handle` + `_jeopardy_arm_timeout` timer; leaderboard
table. **Effort: M** *(it's already two LLM calls/turn — validation must be fast or the timer
false-triggers).*

### Unified GAMES registry (the enabler)
Not user-facing fun, but it's what makes *every other game idea here* a 1-file change instead
of a 5-file scavenger hunt. The game list is duplicated across `games.py` dicts,
`command_parser._KNOWN_GAME_NAMES`/CANONICAL, and `action_router` regexes with no shared
source. Introduce one declarative `GAME_REGISTRY` that all layers derive from.
**Hooks:** `features/games.py:_GAME_ALIASES`/`_GAME_DISPLAY_NAMES`/`_GAME_HANDLERS`;
`command_parser._KNOWN_GAME_NAMES`/CANONICAL; `action_router._GAME_START_REQUEST_RE`.
**Effort: M** *(refactor on the hot routing path — keep test_command_parser/
test_action_router green).*

---

## 6. Physical showmanship (servos / LEDs / sound / motion)

### Comedic-timing beat pack (double-take, eye-roll, spit-take, lean-in, shrug, mic-drop, slow-clap, facepalm)
**Highest physical-ROI addition.** The 15 existing body beats are emotion *reactions*, not
comedy *bits* — there's no double-take, eye-roll, or mic-drop. Add ~8 new `_beat_*` runners:
double-take = look away, snap back, visor pop; eye-roll = visor dip + slow neck arc;
spit-take = sharp recoil + visor snap; mic-drop = heroarm forward then drop + dismissive
turn. The instant each is registered it becomes LLM-addressable (`performance.body_beat`),
event-addressable, AND gamepad-addressable — three call paths for free.
**Hooks:** `sequences/animations.py:_BODY_BEAT_RUNNERS`/`_BODY_BEAT_CHANNELS`/`_BODY_BEAT_ALIASES`;
`performance_plan.BODY_BEAT_NAMES`; action_router `performance.body_beat`. **Effort: M**
*(headtilt and elbow are mechanically tiny — lean reads on neck+visor+heroarm; bench-test on
real servos).*

### Rimshot / sad-trombone / airhorn synced to Rex's own punchline
The soundboard is mature, output-gated, mic-suppressed — and triggered **only by the human
gamepad.** Rex never punctuates his own joke. Ship 3-4 SFX clips (rimshot, sad_trombone,
airhorn, drumroll) and fire them from `performance_output.execute_plan` in the
*post-punchline pause that already exists.* Map joke-land → rimshot, flop → sad-trombone,
win → airhorn.
**Hooks:** `audio/soundboard.py:play`; `performance_output.py:execute_plan` (the
`JOKE_SETUP_PUNCHLINE_PAUSE_MS`/`post_beat_ms` window); `performance_plan._EVENT_BODY_BEATS`.
**Effort: S** *(must fire strictly in the silent pause — the output gate drops a clip over
live TTS; cooldown so overuse doesn't kill the joke).*

### Composite "show moment" verb (beat + SFX + LED + mood in one call)
Imagineering moments are multi-sensory, but every layer is invoked separately today
(`excited_burst` does servo+LED but no sound, no sustained mood). Add
`animations.show_moment('win'/'roast'/'flop'/'reveal')` that fires a body beat + soundboard
sting + chest/eye LED flash + `body_mood.set_mood` together. A win becomes
`tiny_victory_dance` + airhorn + gold eye-sparkle + chest confetti + 30s of smug posture —
one call, one repeatable real celebration instead of a twitch.
**Hooks:** `sequences/animations.py:excited_burst` pattern + `play_body_beat`;
`soundboard.play`; `leds_chest.compliment_flash` + `leds_head.set_eye_color`;
`body_mood.set_mood`; fired from `_EVENT_BODY_BEATS`. **Effort: M** *(sequence carefully —
LED+mood safe always, SFX in the pause, beat owns the arm via the arm-gesture latch the
wave-back already uses).*

### "Thinking eyes" while the LLM works
LLM latency is currently dead air. Pulse/color-cycle the eye LEDs during
transcription→LLM→TTS — a slow amber/cyan breathe says "I'm thinking," a quick
triple-blink-to-gold says "got it." Disney animatronics *sell* the wheels-turning look;
curiosity reads as genuine when his eyes visibly work a problem.
**Hooks:** `hardware/leds_head.py:set_eye_color`/`ensure_eyes_on` + the eye heartbeat; tied
to the VAD-onset hook that starts `start_listening_motion` (`interaction._begin_user_turn`).
**Effort: S** *(must yield to the keep-alive heartbeat and `_speaking` flags; low update
rate — the head-LED serial link drops bytes during speech).*

### Smug-after-a-good-roast sustained mood
Comedy is about the *attitude between* the jokes. `body_mood` decays over ~45s but almost
nothing drives it. Call `set_mood('smug', source='roast_landed')` when a roast lands
(mirroring the existing `made_laugh → giddy` hook) so Rex visibly basks — chin up, visor
wide, swagger in his rest posture — then "sulk" after a flop.
**Hooks:** `intelligence/body_mood.py:set_mood`; fire from the same humor/roast outcome path
that calls `_play_event_body_beat`. **Effort: S** *(modest intensities so face-tracking
stays primary; `set_mood` already guards a weaker mood stomping a stronger one).*

### Beat-synced DJ dance mode (revive the dead `arm_rhythm_tick`)
Rex is a DJ who **doesn't dance** — `arm_rhythm_tick(beat_phase)` is written to dip the
elbow on a downbeat and is **never called anywhere.** Wake it up from the music playback loop
and extend into a full dance: elbow dip + heroarm pump on the downbeat, head bob + neck sway
on off-beats, chest LED flash per beat.
**Hooks:** `sequences/animations.py:arm_rhythm_tick` (uncalled) + the music playback beat
loop; `leds_chest.py` beat-flash. **Effort: L** *(needs a real beat signal from the music
subsystem — librosa is available but live beat tracking adds latency/CPU; servo cadence
within Maestro limits; yield instantly to speech).*

### Roll-up + mock-retreat base bits
Physical proximity is showmanship a stationary droid can't do. Add two cap-respecting
`motion.*` actions: `roll_up` (`come` toward the addressed person, then a lean-in beat) for
crowdwork, and `mock_retreat` (small `move_back` + `offended_recoil` + *"well, EXCUSE me"*)
when insulted. Both degrade to a pure body beat when `motion_controller.available()` is False.
**Hooks:** `motion_controller.py:come`/`move_back`/`arc_move`; registered as `motion.*` in
action_router; gated by `available()` + `_autonomous_allowed`. **Effort: L** *(MOST
safety-sensitive idea here — base is intentionally slow, ToF avoidance is still a STUB, left
motor/calibration unfinished; require clear-floor/operator opt-in, respect MOTION_MAX_*
caps, never run while INTERACTION_PAUSED).*

### Full puppeteer soundboard (every gamepad button → clip + beat + LED)
Gives a human operator a live comedy controller — the way the best animatronic shows are
actually run. Only 7 gamepad buttons are mapped; the dispatcher already supports
clip+animation, just add an `led` leg and a few comedy SFX. Rimshot on demand, eye-roll on
demand, victory dance on demand.
**Hooks:** `config.py:MOTION_GAMEPAD_BUTTON_ACTIONS`;
`motion_controller._dispatch_button_action` (extend with led leg); `soundboard` +
`play_body_beat`. **Effort: S** *(mostly config; needs the comedy SFX clips to exist first).*

---

## 7. Usefulness & smart home

> **Honest framing:** there is *zero* smart-home plumbing today — no timer, no reminder, no
> light control, no device transport, no scheduled-callback primitive. But the *patterns* to
> build them cleanly all exist (`fetch_weather`'s requests+cache+fail-soft template, the
> action-spec system, the deterministic classifier template, the web-search slow-branch).
> **And Alexa is NOT directly driveable** — Amazon deprecated most third-party
> skill-to-device control and there's no public local API. The only tractable paths are
> indirect (see below).

### Home Assistant as the one hub (`home.*` family)
The right architecture. Add `awareness/home.py` as a thin Home Assistant REST client built
*exactly* like `fetch_weather`: config URL + token in apikeys, TTL-cached entity list,
fail-soft. Add specs `home.lights_on/off/dim`, `home.scene`, `home.device_toggle` with a
`classify_explicit_home()` deterministic classifier (copy `classify_explicit_motion`), an
evidence entry, and one dispatch branch. HA natively bridges Hue/Kasa/etc. **AND fronts the
user's Alexa devices** — sidestepping Alexa's lack of a control API. *Rex says "Lights. Now."
and the room obeys — the single biggest "he's real, not a toy" moment.*
**Hooks:** `awareness/chronoception.py:fetch_weather` (pattern); `action_router.py:ACTION_SPECS`
+ `classify_explicit_motion` template + `missing_required_evidence_reason`;
`interaction._handle_router_takeover_action`; `config.ACTION_ROUTER_EXECUTE_ACTIONS`;
`apikeys.py`. **Effort: L** *(owner must stand up HA + a token; config device map maps spoken
phrases to entities; fail-soft "I can't reach the house right now" reusing the
weather-offline idiom).*

### Alexa via virtual-switch / routine trigger (the *tractable* Alexa path)
Don't try to drive Alexa directly (you can't). Expose Rex-controlled virtual switches (HA
template switches or IFTTT webhooks) that Alexa Routines watch as *triggers*; flipping one
via the `home.*` transport makes Alexa run "Goodnight" / "Movie" / announce. *"Rex, tell the
house goodnight"* commands the Alexa ecosystem the owner already has, without fighting
Amazon's closed surface.
**Hooks:** reuses the HA REST client OR a tiny requests-based IFTTT webhook (web_search.py's
dedicated-client + fail-soft pattern); action specs; `config.ALEXA_TRIGGER_SWITCHES`
mapping; `apikeys.py`. **Effort: M** *(Routine round-trip latency — cover with a "Telling the
house..." stall line like web_search; setup burden on the owner, ship a documented recipe).*

### Timers & reminders (the missing scheduled-callback primitive)
**The single biggest usefulness gap** — there is no way to "say or do X at time T" anywhere.
Add `features/scheduler.py`: a thread-backed `{due_at, kind, payload, person_id}` store
persisted to memory/, specs `timer.set`/`timer.cancel`/`reminder.set`, and a deterministic
classifier parsing durations (reuse `_motion_dist_to_m`'s unit-parse style). On fire, push an
in-character announcement — *"Hey JT, your ten minutes are up, the pizza's calling."*
Genuinely useful AND a character moment: Rex doesn't beep like a microwave, he *roasts you
about your forgotten pizza.*
**Hooks:** `chronoception.py` time source + `start_periodic_update` thread pattern;
`action_router.ACTION_SPECS` + motion unit-parse template; the speech path;
`conversation_agenda` proactive directives for phrasing. **Effort: L** *(firing
mid-conversation needs the announce primitive below; persist across restarts so a timer
survives a supervisor bounce).*

### Intercom / announcement beat ("broadcast this now")
Rex can speak, but there's no "say this line now regardless of conversation state"
primitive — which timers, reminders, and proactive offers all need. Add `announce(text, *,
cue=False)` distinct from a conversational reply: speaks immediately (respecting the output
gate + paused flag), optionally wrapped in a show-cue sting.
**Hooks:** `interaction.py` speech path + `INTERACTION_PAUSED` handling; `soundboard.py` for
the sting; `show_cue.py`; `action_router.ACTION_SPECS`. **Effort: S** *(must queue not stomp
an in-progress turn unless it's a real alarm; keep short).*

### Show-cue primitive (dim + sting + line)
The Imagineering glue. The parts exist independently — soundboard, head/chest LEDs, music
ducking — but nothing *composes* them around an action. Add `features/show_cue.py` that
sequences `soundboard.play(sting)` + LED recolor + `dj.volume_down` (duck) + the spoken line
+ restore. Used by storytelling, "movie time," and big bits; drives real lights via `home.*`
when present, falls back to Rex's own LEDs when not.
**Hooks:** `soundboard.play`; `leds_head.set_eye_color`/`set_eye_emotion`;
`leds_chest.compliment_flash`; `dj.set_volume`/`volume_down`/`stop`; the output-gate
discipline soundboard already respects. **Effort: M**

### "Movie Time" scene (flagship useful+showman combo)
One `home.scene` action with `scene='movie'` that fires the show-cue (sting + "Enjoy the
show" + LEDs to dim warm), tells HA to run the movie scene (lights down, optionally TV on),
and ducks Rex's own music — plus a "lights up / show's over" counterpart. One phrase
transforms the room; *the exact demo that makes people go "wait, he runs the lights?"*
**Hooks:** builds on Home Assistant transport + show_cue; classifier + dispatch + config
scene map. **Effort: M** *(graceful degrade: dim his OWN LEDs + duck music + "I can set the
mood even without the house wired up" so the bit plays pre-HA).*

### Curiosity-as-usefulness: notice → offer → act
The bridge that makes Rex a roommate, not a toy — and it *unifies* §2 and §7. Tie the
existing scene/time awareness to the new `home.*` control so curiosity produces *action
offers*: when time-of-day rolls to night or the scene reads dark, a proactive directive
offers *"Getting dark in here — want me to bring the lights up?"* Accept → `home.lights`.
**Hooks:** `conversation_agenda.proactive_purpose_directive`/`with_proactive_directive`;
`TIME_OF_DAY_REACTIONS`/`WEATHER_PROACTIVE_REACTIONS` banks; awareness scene analysis;
`home.*` family. **Effort: M** *(reuse proactive cooldown/once-per-transition guards so it
doesn't nag; never auto-change lights without a yes).*

### Inject open plans into the LIVE reply
`events.py` readers are called *only* in proactive paths — `_build_person_context` omits
events entirely, so mid-conversation Rex *doesn't know you have a thing tomorrow.* Add a short
"Open plans they mentioned: X (on DATE)" block to `_build_person_context` with a restraint
rule. Cheapest "more useful" win.
**Hooks:** `memory/events.py:get_upcoming_events`/`get_open_events`;
`intelligence/llm.py:_build_person_context`. **Effort: S** *(reuse the `_anticipated_events`
throttle so it doesn't double-mention with the proactive path).*

---

## 8. Personalization & running gags from memory

### Open commitments ("last time you SWORE you'd fix that sensor")
Accountability ribbing is the most relatable thing a friend does. When someone says they'll
do X, file it as a `person_event` with new `status='promised'` (the status column +
cancel/reschedule machinery already exist). On return, inject the open promise into
`_build_person_context` as a dry needle; on confirmation, `mark_followed_up` records the
outcome as future roast fuel.
**Hooks:** `memory/events.py:add_event`/`get_open_events` (new status='promised') + a
commitment regex beside cancel/postpone; new hook in `llm.py:_build_person_context` chain.
**Effort: M** *(tight first-person future-intent regex — "I should really..." is NOT a
commitment; let cancel/postpone phrases clear it).*

### Durable honorifics from positive milestones ("the reigning trivia champ")
Positive milestones get one soft acknowledgment then decay. When a celebration event or a
game-win episode is recorded, mint a short honorific fact (`category='inside_joke'`, already
defined) and surface it as a greeting flourish, decaying slowly; dethrone it on a
contradicting outcome.
**Hooks:** `emotional_events.py:get_due_celebrations`/`is_celebration_event`;
`episodes.py:record_game_played`; honorific writer via `facts.py:add_fact`
category='inside_joke'; surfaced in `episodic_hooks.record_greeting_event` or
`_relationship_tone_rule`. **Effort: M** *(tie expiry to the celebration's decay window; a
stale "champion" after a loss reads as out of touch — let a contradicting outcome dethrone
it, and being wrong can itself be a joke).*

### Gossip discharge: the knowing look when the subject walks in
The `tell-me-about` flow theatrically *collects* gossip but has no payoff —
`format_fact_for_prompt` explicitly says "don't recite it back," so the dossier just sits.
The funniest part of knowing a secret is *visibly knowing it without telling.* When a
briefed-about subject is later recognized, fire a one-time coy beat: *"Oh. OH. So YOU'RE
{name}. I've heard... things."* — never reciting the gossip; the comedy is the suppression.
And when teller + subject are both present: *"{teller} told me ALL about you, by the way."*
**Hooks:** `tell_me_about.py:classify_detail` (told_by/fact_kind); the `pre_met_note` fact in
`interaction._create_tell_about_subject`; a recognition-time hook in greeting selection; the
kindness gate in `facts.py:format_fact_for_prompt`. **Effort: M** *(hard-gate on
crowd_count==2 — just the two relevant people; NEVER surface gossip text, only the meta
framing; reuse the emotional-events crowd-discretion pattern).*

### Per-person smart-home anticipation memory
The cantina-bartender fantasy: Rex knows your usual order. Capture a per-person device-routine
preference (*"Bret dims the lights when he sits down"*) as a `facts.py` preference tagged for
proactive offer. On recognition, Rex offers in character — *"Want the lights down like usual,
or are we feeling brave today?"* — handing the remembered preference to the `home.*` action
layer.
**Hooks:** `memory/preferences.py` + `facts.py:add_fact(category='preference')`; recognition
path in consciousness; feeds the `home.*` capability (§7). **Effort: L** *(depends on §7's
device layer existing; keep it OFFER-then-confirm, never auto-act; gate behind explicit
per-person opt-in stored as a preference).*

### Feed the callback bank its best material
The callback engine is starved: it banks *only* the speaker's own first-person flat premises,
hard-drops third-party material, and never banks Rex's *own* funny moments. Extend
`bank_from_turn`'s capture seam to (carefully, behind the same sensitivity wall) ingest Rex's
landed bits and warm things said *about* a present person, so the running-gag engine (§1) has
more to work with.
**Hooks:** `callback_engine.py:bank_from_turn` + `_SAFE_WHEN_FLAT_CATEGORIES`;
`memory/callbacks.py`. **Effort: M** *(third-party material is a sensitivity minefield — only
present people, only warm/safe, same deterministic wall).*

---

## The big swings — signature, staged, repeatable "E-tickets"

These fuse the above into experiences a guest leaves *talking about*. Each names the
subsystems it fuses and the new "glue" required.

### E-1 — "The Reunion" (Rex stages your homecoming)
Face recognized after a long gap. Rex withholds the normal greeting — eyes pulse amber, a
beat of silence — then *"Wait. WAIT. Is that—"*, turns to you (`directed_look_pose`), arms up
(`excited_burst`), mood slams to giddy. Payoff: a `person_story()` monologue stitching 2-3
real episodes from before you left, with the accountability needle inside the warmth —
*"...the trivia night, the dog that wandered in, the sensor you swore you'd fix — still
broken, by the way."* Button: mood settles to warm, *"...okay. Don't do that again."*
**Fuses:** episodic diary + `person_story()` · long-absence timestamp · thinking eyes + chest
LEDs + body_mood arc · `directed_look_pose`/`excited_burst` · open-commitments roast ·
callback engine. **New glue:** an *absence detector* (gap since `last_seen`) firing a
`do_reunion` show-sequence through the governor, locking out the normal greeting for one turn.

### E-2 — "Story Time" (fully staged tall tale)
*"Alright. Settle in."* Show-cue fires: music ducks, lights dim warm (`home.scene`) or chest
LEDs dim amber as fallback, a low synth sting, eyes to a slow storyteller breathe. Then the
`dramatic_narrator` voice, single-beat cap lifted, a 4-5 sentence absurd flight-record yarn
personalized by a real callback detail, structured setup → escalation → planted false climax
→ real punchline with a rimshot kicker. Lights restore, a bow (`proud_dj_pose`), mood to smug.
**Fuses:** tall-tale action (length-cap bypass) · `dramatic_narrator` profile + pause beats ·
show_cue + `home.scene` · callback personalization · soundboard punctuation · body beats.
**New glue:** the show_cue composer wrapping the gated action; a "false climax" prompt
scaffold so the structure is reliable.

### E-3 — "Roast Battle: Cantina Edition" (audience-judged game show)
Airhorn, host_mode on, MC bank. Rex names both players (face memory) and turns between them.
Each round he feeds a curated setup (drawn from their *signature roast handle*), they fire, he
MCs and inserts his own roast as the undefeated third contestant. He **reads the room's
laughter/applause as the judge** — *"ooh, the room HEARD that"* — falling back to "who won?"
The crown: airhorn + `show_moment('win')` + a minted honorific.
**Fuses:** comedy_modes + sharp roast tier · signature handles + person facts ·
laughter/applause sensing as live judge · MC banks + host_mode · soundboard stingers +
`show_moment` · honorific minting + leaderboard. **New glue:** a roast-battle game format
(games are all quiz-shaped today); laughter→score mapping; the standing roast guardrails
carried *into* the game prompt.

### E-4 — "Cantina Open" (the daily ritual)
First person detected after a long quiet stretch in the morning/evening window → lights up to
warm (`home.scene`), a signature track auto-announced and dropped (*"Doors are open."*),
beat-synced dance mode (the revived `arm_rhythm_tick`), chest LEDs pulse on the beat. Over the
intro, a "state of the room" bulletin — who's expected (open plans), what happened last
session (one episodic callback), the standing champ. *"Bret's still got the trivia belt. For
now."* Close: *"...alright. We're live."*
**Fuses:** time-of-day + first-arrival trigger · `home.scene` · DJ auto-announce + dance mode
+ beat LEDs · events/open-plans · episodic callback · leaderboard. **New glue:** a
once-per-window ritual scheduler; the dance-mode beat signal; sequencing the bulletin over the
music bed.

### E-5 — "The Long Con" (an evolving multi-session bit)
A premise lands and gets promoted to `running_bit` (the resurrected dead category) — say, the
appliance-conspiracy frame. Each return it grows via the running_bit history branch —
*"Update on the toaster situation." / "The toaster has recruited the microwave."* Exempt from
freshness decay; it gets *funnier*, not staler. A planned payoff fires through the
scheduled-callback primitive — *"It happened. The toaster blinked at me. I have proof."* —
staged with a show_cue. Then it graduates to a fond callback and a fresh long-con seeds.
**Fuses:** running_bit + callback history branch · appliance_conspiracy persona ·
scheduled-callback primitive · show_cue · cross-session episodic memory. **New glue:** a
long-arc state machine (seed → escalate → payoff → retire); a scheduled-callback that can fire
on a future *session*, not just a future clock-time.

### E-6 — "Name That Tune: Showdown" (the flagship party game, fully produced)
host_mode + MC bank, drumroll. Rex names the players, reads the leaderboard. 5-10s snippet
drops, crowd shouts, fuzzy match, Rex roasts wrong guesses *in the delivery profile*
(deadpan whiff, smug correct), ding/buzzer stingers. `room_energy()` makes him cockier and
faster as the room heats up. Final wager round (Jeopardy mechanic ported) with the theme during
thinking time, dramatic reveal, airhorn. Leaderboard persisted, honorific minted, rivalry
premise banked.
**Fuses:** music index + fuzzy matcher + snippet playback · face memory + leaderboard +
honorifics · delivery profiles + comedy banks · soundboard stingers · `room_energy()` boost ·
ported wager mechanic. **New glue:** time-bounded snippet variant; leaderboard table (+ the §5
game-outcome state-clear fix as prerequisite).

### E-7 — "The Welcome Committee" (named, staged arrivals)
Door-direction arrival resolves to a stable known identity. Rex snaps toward the door
(`directed_look_pose`), energy up — *"JT! The volleyball menace returns!"* — with their
signature handle, a one-line episodic callback + any open plan (*"Did you fix that sensor or
are we still pretending?"*), and if a per-person preference exists, the host offer (*"Lights
down like usual, or feeling brave?"* → `home.*`). Strangers get the warm generic fallback.
**Fuses:** door-direction arrival + face identity · `directed_look_pose` + energy beat ·
signature handle + episodic callback + open plans · per-person smart-home anticipation ·
stranger fallback. **New glue:** identity-stability gate on arrival; chaining call → brief →
host-offer into one arrival sequence.

### E-8 — "Movie Night, Presented by Rex" (the useful+showman flagship)
*"Movie time? Say no more."* Show_cue: chest LEDs dim, sting, *"Enjoy the show."*
`home.scene='movie'` — lights down, TV on via HA; his own music ducks/stops. A one-line
in-character intro riffing on the crowd. Rex goes visibly quiet (eyes to low ambient idle),
respecting the room. Encore: *"Show's over"* — lights up, *"Reviews?"* — invites post-movie
banter, banks reactions. Degrades gracefully: no HA → dims his own LEDs, ducks his music.
**Fuses:** `home.scene` (HA transport) · show_cue · DJ volume control · curiosity-aware intro ·
graceful pre-HA fallback · post-show callback banking. **New glue:** HA REST transport + movie
scene map; announce/quiet-mode handoff; encore counterpart.

---

## Top 10 highest-impact (ranked)

1. **Comedic delivery profiles** (§do-first/§1) — the single change a human feels
   *immediately*; every existing joke under-lands because a roast and a condolence sound
   identical. **M**
2. **Free the COCO detector → live object inventory** (§do-first/§2) — highest-leverage
   *unlock*; zero-cost data substrate that makes 6+ other ideas possible and kills "generic
   curiosity" at the root. **M**
3. **Land-the-laugh / take-a-bow** (§do-first/§3) — Rex is currently *deaf to laughter and
   applause*; reacting is the defining showman move and the signal's already computed. **M**
4. **Running gags that escalate** (§1) — resurrects dead `running_bit` scaffolding; turns
   recall callbacks into bits that get funnier each time. **M**
5. **Home Assistant `home.*` hub** (§7) — the biggest "he's real, not a toy" leap; the
   correct architecture that also solves Alexa. **L**
6. **Comedic-timing beat pack** (§6) — 8 new physical bits, each instantly addressable via 3
   call paths; eye-rolls and mic-drops a room reads instantly. **M**
7. **Storytelling / tall-tale mode** (§4) — the "bigger bit" explicitly wanted, *structurally
   impossible* today; lifts the single-beat cap safely for one gated action. **L**
8. **Energetic named arrivals** (§3) — the most memorable repeatable moment: walk in, get
   clocked by name with energy. **M**
9. **Timers & reminders + announce primitive** (§7) — the biggest *usefulness* gap closed,
   delivered as a roast not a beep. **L**
10. **Warmth-earned sharp roast tier** (§1) — gives the roaster/showman energy a place to
    live, gated so strangers/sincere moments stay protected. **M**

## Quick wins (a weekend each)

- **Rimshot / sad-trombone / airhorn on Rex's own punchline** (§6) — reuses the working
  output-gated soundboard; fire in the existing post-punchline pause. **S**
- **DJ Rex announces every track** (§5) — TrackInfo already known; one polished line. **S**
- **Host-grade reveal stingers across all games** (§5) — reuse jeopardy's pipeline. **S**
- **Thinking eyes during LLM latency** (§6) — turns dead air into "wheels turning." **S**
- **Smug-after-a-good-roast mood** (§6) — one `set_mood` call from the existing roast outcome
  path; instant personality arc. **S**
- **Inject open plans into the live reply** (§7) — events readers exist; just call them. **S**
- **Let curiosity work in a crowd** (§2) — delete one hard `crowd_count>2` return, add a group
  branch. **S**
- **Comedic personas** (§1) — 2-3 new ComedyMode stances flow through existing assembly. **S**
- **Per-person signature roast handle** (§1) — pick one ranked fact, expose it in the tone
  rule. **S**
- **Tease-the-obsession lane** (§1) — relax one clause in `conversation_steering._directive_for`.
  **S**

## Hard problems / open questions

- **Smart-home is genuinely greenfield.** No timer, reminder, light control, device
  transport, or scheduled-callback primitive exists — every §7 idea is net-new (though the
  *patterns* are proven). Sequence: build `scheduler.py` + `announce()` first (prerequisites
  for timers/reminders/proactive offers), then the HA transport, then the show-cue glue.
- **Alexa cannot be driven directly.** Amazon deprecated most third-party skill-to-device
  control; no public local API. The only tractable path is *indirect*: Home Assistant
  fronting the Alexa devices, or Rex flipping virtual switches / IFTTT webhooks that Alexa
  Routines watch as triggers. This pushes real setup burden onto the owner — ship a documented
  recipe, and every home action must fail-soft so an unreachable house never makes Rex go
  silent.
- **Raising the roast ceiling is the exact failure mode the 80/90→55 rebalance fixed.** The
  `sharp` tier must be gated hard on warmth + tier + *no active downgrade*, the harsh-word
  governor regex must still catch genuinely cruel output, and the `roasted_sincere` quality
  eval has to stay green. Highest-risk comedy change.
- **Longer bits fight the rest of the system.** The HARD LENGTH LIMIT, `govern_response`
  trimming, and the single-beat cap all actively suppress multi-sentence output. Tall-tale and
  saga modes must bypass the length governor *cleanly and only for the sanctioned action*
  (model it on web_search), and must yield instantly to barge-in.
- **The base is not show-ready.** ToF avoidance is still a STUB, the left motor + calibration
  are unfinished, and the tall base can topple — so all motion "bits" are capped slow/gentle
  and must degrade to a pure body beat when `available()` is False. Roll-up/mock-retreat are
  the most safety-sensitive ideas; require clear-floor/operator opt-in.
- **Detector noise vs. crying wolf.** Object permanence, change detection, and clothing/gesture
  callbacks all ride single-frame vision that flickers indoors (a lamp already mis-reads as a
  "bird"). Every one needs a confirm-streak + per-object/per-person cooldown. Start the object
  allowlist tight.
- **TTS cache thrash from voice profiles.** Each comedic delivery profile changes the
  ElevenLabs cache key → first-use regen (cost/latency). Limit to a few profiles, keep
  `similarity_boost` high.
- **Output-gate serialization.** SFX/stings can't overlap live TTS — a fired clip during a
  line is simply *dropped*. Everything in §6 must place sound strictly in the post-line pause
  windows, not during speech.
- **Episodic outcome capture is broken for half the games.** i_spy / 20q / word_association
  clear their own state before `_extract_game_outcome` runs, so the leaderboard would record
  contentless results. Fix the state-clear ordering *before* building leaderboards.
- **Memory-driven personalization risks creepiness.** Room model, gossip discharge, and
  per-person preferences must respect the existing carve-outs: in-room-volunteered facts only,
  never log screens/text, gossip never recited (only the meta "I've heard about you"),
  crowd-discretion on sensitive material. The privacy posture is already strong — new ideas
  must inherit it, not route around it.

## Suggested build order (maximizes fun-per-week)

Laughter/applause reaction → room model + room_energy (in parallel) → scheduled-action
primitive → smart-home transport (gated on the HA decision). **With the first three alone —
all zero-setup, all M effort — five of the eight E-tickets become buildable.**

## Open scoping question for the owner

How much setup are you willing to do? The difference between "Rex controls the lights" being
an **L** and an **XL** is almost entirely whether Home Assistant is standing up. If you'll run
HA, §7 becomes tractable fast; if not, the realistic ceiling is read-only awareness + offers
Rex can't yet fulfill.
