# Dead-code removal plan (2026-09-23)

Survey baseline: **HEAD `c00eed5`**. All `file:line` references below are at that commit
and WILL drift as stages land — re-derive every span with `ast` (`end_lineno`) at execution
time; never delete by the numbers in this file alone.

How this was produced: six subsystem survey agents (reply brain, proactive, routing, audio,
vision/motion, periphery) each traced flag-gated and backend-gated code against the
EFFECTIVE config (`config.py` defaults; `user_config.py` overrides nothing; `.env` only sets
devices/ports/servo limits; launchd sets no flags). Then 12 verification agents: one
adversarial "try to refute" skeptic per area, a second side-effect-lens verifier on the
three `interaction.py`-heavy areas, a live-bug verifier, an AST transitive-closure pass, and
a sweep of areas the surveys did not cover. Everything below survived that pass; spans are
the verifiers' corrected ones.

Live modes treated as LIVE (code behind them is not dead): `--noaudio`, `--local-tts`,
`--gui`, `--noservos`, `--jeopardy`, and OFFLINE mode (`connectivity.is_offline()` — the
deterministic command lanes are the ONLY router when offline, so they stay).

## Honest sizing

The tree is ~187k prod lines + ~97k test lines. Provably dead or config-dead code is
roughly **10–12k prod lines** (+ several thousand test lines and ~600–800 config lines),
plus 3–5k lines of stale tools. That is ~6% of prod. The bulk of the size
(`interaction.py` 32k, `consciousness.py` 15k) is LIVE code; shrinking it further is a
refactor (splitting modules, collapsing duplicated gates), not a deletion, and is out of
scope here.

## Part A — Bugs found during the survey (fix separately; NOT part of the deletion)

These change behavior, so each needs owner sign-off. Ordered by impact.

| # | Bug | Evidence | Fix |
|---|---|---|---|
| A1 | **Voice ID is off on this Mac.** `VOICE_EMBEDDER` defaults `campplus` but `assets/models/campplus/` is absent; `speaker_id._get_encoder` (65-78) disables embeddings with no fallback. All 2026-09-18 runs log `CAM++ unavailable`. 14/15 turns that day had `suppress_memory_learning=true`. people.db has only 5 ECAPA prints and 0 CAM++ prints, so even after download everyone starts unknown. | logs/djr3x-2026-09-18-*.log | `tools/download_campplus.py` + re-enroll, or `VOICE_EMBEDDER=ecapa` in `.env`. Add a visible startup alert when the embedder is unavailable. |
| A2 | **Callback humor has never fired live** (since `85dcf2d`, 2026-06-14). `callback_engine.py` 248 and 315 (`_crowd_ok`) call `world_state.snapshot()` on the MODULE → AttributeError swallowed → `_crowd_ok()` always False. Kills reactive callbacks, lull callbacks, and the Lean `_lean_callback_lull_cue`. Tests mock `_crowd_ok`. `person_callback_material` has 0 rows. `refresh_relevance` still runs a local-LLM judge every turn with no consumer. | no live claim in 104 logs | 2-line fix `from world_state import world_state` — but this switches on a feature that has never run live; needs a live test. Alternative: retire it (~1,450 prod + ~1,010 test lines). **Owner decision D2.** |
| A3 | One-voice lines never see the transcript. `llm.py:1565` `from intelligence import conv_memory` (module doesn't exist; since `fde1553`, 2026-07-02) → `_one_voice_transcript()` always None. Affects greetings, proactive lines, acks, onboarding questions, game lines. Test mocks it. | | `from memory import conversations as conv_memory` |
| A4 | Classic reply fallback can't fire in voice mode. `_reply_token_stream` returns the `lean_brain.stream_reply` GENERATOR inside `try`; errors surface on first `next()` outside it; the streaming except (≈17374-17386) just logs and the turn is **silent**. Only `--noaudio` really falls back. | | Prime the generator inside the `try` (re-raise `ToolCallRequested`), else `yield from llm.stream_response(..., classic=True)`. |
| A5 | Celebration check-in (Trigger A2, `consciousness.py` 11988-12027) submits a Lean-suppressed line then `return`s, so while any celebration is due the sustained-negative-mood check-in (Trigger B) never runs for that person; A2 also re-submits every 10 s. | | Delete A2 (Stage 4) — that IS the fix. Behavior-neutral alternative: `if get_due_celebrations(...): return`. |
| A6 | 4 tests ERROR on HEAD, not on the CLAUDE.md known-failure list: `test_audio_and_conversation_gating` :9609, :9757, :9760 (helper :9696), :10286 — they patch `IDENTITY_VOICE_ENROLL_MIN_*` / `_voice_enrollment_sample_allowed`, removed in `e7a0f54`. After dropping the patches the offscreen tests still expect an enrollment the current handler never does. | | Delete the two config patches + `test_voice_enrollment_requires_longer_sample`; change offscreen tests to assert no enrollment. |
| A7 | Under Lean, the cancellation-empathy directive (≈16389-16401) reaches NO model at all (the slim render overwrites it even for classic) and the news-follow-up grounding (≈16509-16524) reaches only classic. The Lean ordinary `TurnPlan.directive` prose also never reaches a model. | | Route both into `lean_turn_directive` (behavior change). |
| A8 | Environment drift on this Mac: `mlx-audio` 0.4.5 vs pinned 0.5.1 and `assets/models/breeze_tts` missing → `--local-tts`, offline mode and the ElevenLabs-breaker fallback have no working voice. `cmudict` missing → Jeopardy homophone rescue is off. `pyusb`/`libusb-package` missing (Flex tools). | | `pip install -r requirements.txt` then `setup_assets.py`. |
| A9 | Minor: `do_live_vision_comment` pays a gpt-4o-mini `describe_scene()` then submits an always-rejected line (goes away in Stage 4). Weekly small talk / small-talk question remove a pending follow-up before their doomed submit (transient; goes away in Stage 4). `generate_and_speak` winners are recorded in the decision ledger with `purpose=None`. Fuzzy non-stop escape keys mid-game ("skip this song now") release the game claim then get blocked as `legacy_fuzzy_disabled`. `tools/mic_check.py:108` reads `config.AUDIO_DEVICE_NAME` (only in `utils/config_loader`). | | small, individual |

Resolved during verification (NOT bugs at HEAD): "a log-only presence line can win the
governor tick" — true before `e4df544` (2026-08-20); since then a presence candidate's own
`accepted` stamp puts it on `topic_repeat_cooldown` (all September log lines show
`selected=False`). "Mid-game 'quit the game now' is dropped" — refuted; it arms the stop
confirmation.

## Part B — Owner decisions (block specific stages)

**Decided 2026-09-23 (owner):**
- D1 — YES: delete all non-Lean-brain behavior (Stage 4 goes ahead).
- D2 — FIX callback humor (separate commit; needs a live session since it has never run).
- D3 — KEEP the governor exactly as is ("an idea we pursued but never finished"). Stage 5 is
  CANCELLED; do not edit `action_governor.py`, `presence_cadence.py` or the governor plumbing in
  `speech_engine.py` (stale table rows that name deleted generators stay).
- D6 — NO fallbacks for voice or face ID: remove dlib, Resemblyzer AND ECAPA (plus the
  voice-signature→person resolution that only works under ECAPA). InsightFace and CAM++ become
  the only engines and must fail loudly when unavailable. ALSO remove (owner, same day) the
  MediaPipe EfficientDet object-detector fallback (RF-DETR only) and ElevenLabs v2/turbo support
  (TTS becomes v3-only; `seed_voice_test.py` goes with it). The Qwen local-TTS switches STAY
  (Qwen engine retained on purpose).
- Dependencies (A1, A8) fixed by the owner: `campplus.onnx`, Breeze 8-bit, `mlx-audio` 0.5.1,
  `cmudict`, `pyusb` present.
- Remaining items follow the recommendations below (D4 no-op weight placeholder, D5 delete,
  D7 delete software AEC + radar orient, D9 delete, D10/D11/D12 as recommended).

Recommendations in **bold**.

- **D1 — Retire the `LEAN_BRAIN_ENABLED=False` rollback?** Unlocks Stage 4 (~3.5k lines).
  Lean has been primary since July; with the flag off today you would NOT get a complete
  classic robot anyway (A4, A7). **Recommend: retire.** Keep the flag only as a hard
  "lean_brain import failed" guard, or drop it.
- **D2 — Callback humor (A2): fix or retire?** **Recommend: fix + live test** (it is the
  "remembers + funny" pillar and the code is already written); retire only if it misbehaves.
- **D3 — Action governor.** Verified state: ENFORCE is the real mode;
  `ACTION_GOVERNOR_SHADOW_MODE` is a no-op; presence lines, `do_idle_clip` and the startup
  empty-room joke are still arbitrated by the older `conversation_agenda.claim_proactive_purpose`;
  priority tables disagree (people_roast 46 vs 27, idle_monologue 22 vs 15). Options:
  (a) **delete only the dead parts now** (no-op SHADOW flag, non-ENFORCE branches, dead
  candidate-id/token plumbing, suppressed-purpose table rows — ~150 lines, behavior-neutral)
  — **recommended first**; (b) then finish the cutover (route presence/idle-clip/startup
  through the governor, delete the agenda claim; ~280–320 lines; behavior change — note
  presence `memory_followup`/`celebration_checkin` would become Lean-suppressed and need new
  purposes; needs a live run); (c) remove the governor (~970 prod + ~900 test; not neutral).
- **D4 — Idle micro-behavior weights.** ~16 of 23 weight points pick behaviors the governor
  always rejects; a dead pick still consumes the slot. Deleting them makes idle clips
  (audible mp3s) and neck scans ~4.6× more frequent (every ~1–1.5 min vs ~6 min).
  **Recommend: keep a single no-op entry carrying the dead weight** (behavior-neutral),
  tune later.
- **D5 — Behaviors that went dark with no Lean replacement:** group lull, people roast
  (config comment says owner wants "roast when idle"), appearance riff, ambient
  observation, private thoughts, aspirations. Delete, or re-home as Lean cues?
  **Recommend: delete the old generators; if you want any back, add a Lean cue.**
- **D6 — Real runtime fallbacks** (fire on import/load failure or documented env rollback):
  dlib face backend (+122 MB models; nobody would be recognized under it — DB has only
  512-d ArcFace prints), MediaPipe EfficientDet object fallback (4.4 MB), Resemblyzer voice
  (0 prints in DB → recognizes nobody; frees librosa/webrtcvad), ElevenLabs v2/turbo model
  support (v3 stitching etc.), Qwen take sentence-split / full-buffer switches.
  **Recommend: remove dlib, EfficientDet, Resemblyzer, v2/turbo; keep Qwen switches**
  (Qwen is retained on purpose). Keep ECAPA until CAM++ has run live (A1).
- **D7 — Parked experiments:** radar orient (`MOTION_RADAR_ORIENT_ENABLED`, off since
  09-02 field spins), explore self-trigger, explore head-only fallback,
  `MOTION_ROUTE_ORGANIC_ENABLED`, `MOTION_HEADING_ALTERNATIVES_ENABLED` (owner: keep),
  `WAKE_WORD_ALLOW_DURING_TTS` talk-over (pending experiment: keep), software AEC
  (measured ineffective). **Recommend: delete software AEC and radar orient; keep the rest.**
- **D8 — `rex_pov`** (495 lines + ~135 config) no longer reaches Lean replies; its only
  consumers are web-search answers and classic tool fallbacks. Wire into Lean (like
  `rex_mood.prompt_lines`) or retire? **Recommend: wire in** (not dead code; separate task).
- **D9 — Classic-fallback-only prompt text** (slim contract/stance render, `_slim_*`,
  `comedy_modes.build_directive`/`_SLIM_STANCE`, agenda append; ~250 lines). Keep
  `assemble_system_prompt` itself (11 tool-routed "answer from knowledge" fallbacks + one-voice
  failure + web search). **Recommend: delete** after A4 makes the fallback real, falling back
  to the plain persona prompt. Note the `[agenda]` INFO log line would lose its content.
- **D10 — Tools with no references but some use:** `higgs_clone_test`, `seed_voice_test`
  (work after a v2 revert), `check_breeze_speech`, `lean_replay` (real-model latency, not
  superseded by `production_replay`), `twentyq_eval`, `tof_matrix_monitor`,
  `test_chest_nano`, `wave_back_smoketest`, `audio_test`, `conversation_text_harness`
  (stale since April; bypasses today's router/lean path). **Recommend: delete
  `conversation_text_harness` + `gpt5_ab_test` + `audio_test` + `wave_back_smoketest`;
  keep the rest.**
- **D11 — DJ local library.** `features/dj.scan()` has never had a caller; `assets/music`
  empty; the prompt at `interaction.py:26205` falsely tells users they can play local tracks.
  Delete (~109 lines + `mutagen`) or wire in (1 line)? **Recommend: delete.**
- **D12 — `turn_classifier.py`** (147, never imported) was left "INERT for possible future
  OFF-path use" (rework.md:224). **Recommend: delete** (git history keeps it).

## Part C — Stages

Each stage = one commit to `main` (+ docs in the same commit), executed with the protocol in
Part D. Stages 1–3 need no owner decision beyond this plan; 4–7 are gated as noted.

### Stage 1 — Unreferenced code (SAFE, no flags) — ~2,300 prod lines

> **Landed 2026-09-23** (on top of `36564fb`; about 4,100 lines deleted: ~2,400 prod, ~1,050
> tools, ~460 test, ~130 config). Every item below landed except the ones listed under
> Skipped, and each symbol was re-grepped repo-wide before deletion. Additions beyond the
> bullets: `rex_preferences._YES_WORDS` (the twin of `_NO_WORDS`), `leds_head.charge_status`,
> the `novelty_drive` globals that only `status()` read, and `semantic.invalidate_candidates`
> (only the deleted `facts.delete_facts`/`interests.delete_interest` called it). Other changes:
> - Config: all 15 listed constants were removed, along with `CONVERSATION_TURN_CLASSIFIER_*`,
>   `FACE_DETECTOR_{FORCE_HOG,MODEL}`, `COMMON_FIRST_NAME_LAST_NAME_DISAMBIGUATION_ENABLED`,
>   `COMMON_FIRST_NAMES_REQUIRE_LAST_NAME` and `SPEAKER_ID_SINGLE_VISIBLE_CONTINUITY_FLOOR`.
>   Stale comments were fixed for the flinch corroboration, the unseen-voice challenge,
>   explore legs, the lean brain and the last-name ask.
> - Tests: A6 is fixed. `test_voice_enrollment_requires_longer_sample` is deleted. The offscreen
>   tests now assert that no voice is enrolled, and `..._matching_clip_enrolls` is renamed
>   `..._matching_clip_not_enrolled`. `test_onboarding` builds its answered row with
>   `save_question_asked` + `answer_latest_pending_question`.
> - Behavior fix, not a deletion: `tools/mic_check.py` now reads `AUDIO_DEVICE_NAME` from
>   `utils.config_loader` (the A9 minor bug).
>
> Skipped:
> - The vacuous `test_aec_drain_release` source grep: the plan names no fix, and Stage 3
>   deletes software AEC.
> - Kept because the plan does not list them, although they are now uncalled or dead:
>   - `RexAvatar._is_speaking` (on the KEEP list).
>   - The chest `CHARGE` mirror/render branches (the menubar tools still use them).
>   - `animations.nod`/`headshake`/`thinking`/`surprised`/`dismissal`/`camera_pose`.
>   - The `rex_preferences._TopicOpinion` answer fields.
>   - The `personality._mood_intensity` decay branch.
>   - In `interaction._handle_pending_offscreen_identify_reply`: `enroll_audio`/`enroll_text`,
>     `_offscreen_identity_enrollment_audio` and the stale "CLAIM VERIFICATION" comment.
> - Stale mentions in dated docs are left for Stage 8: `exploration_mode_plan.md`,
>   `comedy_improvements.md` and `active_speaker_detection.md`:155.
> - The local asset `assets/models/face/mmod_human_face_detector.dat` is the owner's to delete.

Nothing calls these in any mode; deletions are behavior-neutral. Tests that only exercise
them are deleted; tests that also cover live code are retargeted.

- **Reply brain:** `intelligence/turn_classifier.py` + config 551-561 + `tests/test_turn_classifier.py` (D12);
  `lean_brain.stream_sentences` (1062-1088) + `_SENTENCE_END` (45); `social_frame._salvage_non_question_lead`
  (1159-1180) + `_QUESTION_CLAUSE_START_PAT` (39-43); `conversation_steering.build_directive` (390-392);
  fix stale `lean_brain` docstring (15-23). Keep `lean_brain.respond` (tool `lean_replay`).
- **GUI 2D avatar painter** (`gui/rex_avatar.py`, ~700): `paintEvent` 193-203, `_draw_*`/`_capsule`/`_joint`
  205-816, `_draw_grid` 873-879, `_value` 887-~935, `servo_to_*` 72-104, colour consts 36-53 (incl.
  unused `_CREAM`/`_GUNMETAL`/`_DARK`), `_CHEST_SQUARES` 896-900, `_CHEST_CHARGE_CHASE` 932, `_last_paint`,
  `_show_grid`, `_mouth_phase`, painter imports 22-23, and the no-op override `rex_avatar_3d.py:227-229`.
  KEEP `set_snapshot`, `_tick_eye_animation`, `_smooth`, `_is_speaking`, `chest_render_state`,
  `_eye_color`, `_chest_gauge_color`, `_prand`, `_pick`, `_boot_norms`, `_neutral_norms`, `_servo_name`,
  `normalize_servo`. Edit `tests/test_chest_led_gui.py::AvatarIngestionTest`.
- **Animations** (`sequences/animations.py`, 138): `_speaking_loop`/`speech_start`/`speech_stop`/`speech_level`
  (1280-1366; keep the `_speaking` event), `visor_flutter`, `look_left/right/center`, `arm_hero_pose`,
  `arm_fidget`, `arm_rhythm_tick`, `excited_burst`, `roast_pose`, `return_to_neutral`; transitively
  `leds_head.set_eye_emotion` (487-490), `servos.get_face_tracking_baseline` (965-968), unused import
  `emotion_orchestrator` (30). 2 tests in `test_log_volume_and_led_footguns`, 1 in `test_gaze_engine`.
- **Vision:** mmod CNN detector (`FACE_DETECTOR_FORCE_HOG` is a constant True): `_cnn_detector` + slow-frame
  counters (~44-60), mmod branch of `_load_dlib` (215-223) and `_detect_rects` (269-289),
  `FACE_DETECTOR_FORCE_HOG`, `FACE_DETECTOR_MODEL`, mmod entry in `setup_assets.DLIB_MODELS`
  (+`mmod_human_face_detector.dat`). `scene.detect_animals` (461-505) and `count_crowd` (852-905).
- **Small unreferenced functions** (verified): `active_speaker.current_speaker` (421-467; 3 tests),
  `servos.idle_animation`, `jeopardy_panel._draw_rex_badge`, dashboard `_format_position`/
  `_format_face_fraction`/`_last_seen_label`, `social.detect_child_present` + `_CHILD_AGE_VALUES` +
  `_ENGAGED_ZONES`, `face.get_face_position`, `motion.wait_ack` + `motion._acks` (edit
  `test_motion.py:185`), `vision_panel._person_details`, `camera.unregister_on_reconnect`,
  `leds_chest.charge_status` (+ other `charge_status`; edit `test_chest_led_gui:69`), `leds_chest.next_pattern`,
  `pose._midpoint`, `pose.head_anchor_px` (retarget `test_pose_face_guard` 29/39 to `head_anchors_px`),
  `increment_interaction_count`, `set_scene_description`, `compass.service_calibrated`,
  `place_recognition.reset_belief`, `main._play_listening_chime_async` (390-413),
  `utils/config_loader._require_int_env`, `jeopardy.format_board`, `games.current_game` (+docstring
  line 21), `trivia.reset_session`, `novelty_drive.status`, `rex_mood._today`,
  `personality.set_param_by_level`/`get_all_params`/`get_emotion`/`set_emotion`/`reset_anger`,
  `database.executemany` (only `tools/backfill_conversation_log.py`, deleted below).
  KEEP (look dead, are not): Qt `mouseReleaseEvent`, Cocoa `drawRect_`/`mouseDown_`, test seams
  (`_inject_for_tests`, `_reset_for_tests`), `place_recognition.score_frame` (offline harness
  `tests/place_recognition_harness.py`, README:509), `servos.arm_gesture_active`/`head_gesture_active`
  (deliberate test accessors), alias-called `ramp_toward`, `hello_info`, place_service accessors,
  `face_expression.reset_*_baselines`.
- **Memory API superseded by live code** (~170): `facts.get_stale_facts`, `interests.delete_interest`,
  `preferences.delete_preference`/`delete_preferences`/`mark_preference_used`,
  `conversations.delete_conversations`, `events.delete_events`, `facts.delete_facts`,
  `relationships.delete_qa`/`save_qa` (edit `test_onboarding::test_answered_question_skipped` fixture),
  `social.delete_for_person`, `disposition.delete_stats`, `emotional_events.get_unacknowledged_since`/
  `mark_all_acknowledged_for_person`, `callbacks.has_topic` + `_SENSITIVITY_RANK`, `database.get_db`,
  `name_validation.is_single_token_name`, `admin.COMMON_FACT_KEYS`. Fix log strings naming `save_qa`
  (interaction 5476, 5606, 21099, 25463).
- **`rex_preferences` answer cluster** (~200): `PreferenceReply` 35-43, `answer_preference_query` 392-461,
  `_answer_yes_no`, `_answer_open`, `_choose_option`, `_yes/_no_for_positive/_negative` (568-589),
  `_strong_no`, `_soften_strong_no`, `_child_detected`, `_is_sensitive_group_topic`, `_NO_WORDS`, unused
  `world_state` import. Delete 5 of 6 `test_rex_preferences`. KEEP `prompt_lines`,
  `extract_preference_query`, `is_group_rating_request`, `_opinion_for_topic`, `_favorite_for_domain`,
  normalizers.
- **Speaker-ID / identity:** `speaker_id.identify_speaker_raw` (414-427), `identify_speaker` (430-465;
  retarget 4 `IdentifySpeakerAcceptanceTest` + 2 partial `ThinChallengerReliefTest` to `rank_speakers`/
  `required_ambiguity_margin`), unused `Tuple` import; `people.find_by_voice` (216-245),
  `count_native_voice_prints` (1216-1234), `latest_biometric_id` (619-628), `_person_score` (401-406);
  `breeze_fast_depth.disable_fast_depth` + `_ORIGINAL`, `breeze_ref_cache.disable_ref_cache`.
  KEEP `people.delete_biometric` (undo tool; patched in `test_intro_misread_guards`).
- **Runtime-dead handler:** `interaction._handle_common_first_name_intro_last_name_reply` (11168-11228)
  — `_pending_common_first_name_introduction` is never set non-None in prod. Also its call/speak block
  (28269-28288), the `is not None` guards (24747, 24819-24820), global decl (12663), resets; and
  transitively `_is_common_first_name_only` (8508), config `COMMON_FIRST_NAME_LAST_NAME_DISAMBIGUATION_ENABLED`,
  `COMMON_FIRST_NAMES_REQUIRE_LAST_NAME` (7232-7259), `SPEAKER_ID_SINGLE_VISIBLE_CONTINUITY_FLOOR`
  (7161-7165), `_single_visible_engaged_continuity_floor` (9668). Sibling
  `_handle_common_first_name_last_name_reply` is LIVE — keep. (MEDIUM: verify with a fresh grep.)
- **Config constants nothing reads** (~60): 2493 `ANIMAL_SPECIES_REMARK_COOLDOWN_SECS`, 3652-3659
  `SPEAKER_GAZE_SEARCH_INTERVAL_SECS`, 7091 `SPEAKER_ID_UNSEEN_GRACE_SECS` (also fix stale docstring
  interaction.py:10671), 7215 `SPEAKER_ID_GRIEF_FLOW_FLOOR`, 8727-8728 `STARTUP_THINKING_LOOP_*`,
  8809 `SHUTDOWN_TTS_EMOTION`, 9137 `PLAN_SUGGESTION_WEB_SEARCH_ENABLED`, 9626
  `MOTION_COME_SIGHT_FRESH_SECS`, 9636 `MOTION_COME_RESIGHT_TURN_DEG` (KEEP its 9-line comment — it
  documents live fused-bearing behavior), 10251 `MOTION_FLINCH_CORROBORATION_MAX_M` (fix contradicting
  comment 10242-10249), 10344-10347 `EXPLORE_LEG_DIST_M`/`EXPLORE_LEG_DIST_JITTER_M`, 10844
  `MOTION_ACK_TIMEOUT_SECS`, 1120 `ACTIVE_SPEAKER_STALE_SECS`. Also remove `user_config.example.py:363`
  (`JEOPARDY_ONLY_CHARGE_THE_ANSWERER`). Verified live despite no literal refs: `THROTTLE_{kind}_{SPEED,ACCEL}`
  (f-string, `throttle_arm.py:225`), `PRIDE_FLOURISH_*` (`animations.py` 2008-2017) — KEEP.
- **Tools, SAFE** (~1,350): `kokoro_voice_test`, `piper_voice_test` (packages not installed; also
  `.gitignore` 297-299), `cleanup_2026_08_29_intro_misread`, `clean_memory`, `memory_cleanup`,
  `backfill_conversation_log`, `remove_phantom_person` (superseded by `cleanup_phantom_people`; fix
  its pointer at `cleanup_phantom_people.py:16`), `gpt5_smoke_test`. Fix pointers in config
  178/248/1532, `memory/database.py:239`, README, `docs/gpt-5_4_mini.md`. KEEP `asr_bench.py`
  (README re-benchmark procedure; fix its bogus pointer to `tests/test_asr_bench.py`).
- **Test hygiene:** fix A6; `tests/test_aec_drain_release.py:120-126` is vacuous (greps the whole module).

### Stage 2 — Retired routing machinery (MEDIUM: env-flippable rollbacks) — ~1,500 prod lines

> **Landed 2026-09-23** (on top of `2d2989c`; about 3,680 lines deleted: ~1,500 prod, ~850
> tools, ~920 test net, ~70 config). All four sub-stages landed. Every symbol was re-grepped
> repo-wide before deletion.
> - `action_router`: `decide()` ends in an unconditional `conversation.reply` (same 0.6
>   confidence) after the shutdown pre-pass and the three explicit classifiers. The LLM tail,
>   `warmup` (+ `main.py` call), `start_shadow_decision`, `_client`, `_SYSTEM_PROMPT`,
>   `_coerce_decision`, `_strip_code_fence`, the catalog sets, the skip machinery, the dead
>   `_apply_context_overrides` branches (the live name_correction→event.cancel override
>   stays), the listed regexes, `_pending_question_context` and the repair evidence branch
>   are gone.
> - `intent_classifier`: only `classify_deterministic` remains (no LLM, no `_log`).
> - `tool_router`: the Phase 0 shadow and `_DEFAULT_LIVE_ACTIONS` are gone.
>   `live_actions()` reads `config.TOOL_ROUTER_LIVE_ACTIONS` directly.
> - `interaction`: the dead takeover arms, the helpers, the repair check (with its
>   `router_action` param), the router `emotional.boundary` block, the event.cancel executor
>   and both shadow call sites are deleted. The `LEGACY_COMMAND_FUZZY_EXECUTE_ENABLED` read
>   is collapsed.
> - Config: `ACTION_ROUTER_{LLM_FALLBACK_ENABLED,MODEL,REASONING_EFFORT,MAX_CONTEXT_CHARS,
>   DETERMINISTIC_SKIP_ENABLED,SELF_QUERY_SKIP_ENABLED,SHADOW_ENABLED}`, `TOOL_ROUTER_SHADOW_*`
>   and `INTENT_CLASSIFIER_{LLM_FALLBACK_ENABLED,LLM_BACKEND,LOCAL_TIMEOUT_SECS,
>   OPENAI_TIMEOUT_SECS}` are removed, along with the stale comments that described them.
>   `user_config.example.py` drops `TOOL_ROUTER_SHADOW_ENABLED` and the action-router
>   mention.
> - Tools/docs: `tools/{conversation_text_harness,gpt5_ab_test,tool_router_report}.py` and
>   `TOOL_ROUTER_TEST_SCRIPT.md` are deleted (D10; the script was pulled forward from Stage 8).
>   CONTEXT.md, README, `docs/tool_router_scope.md` (Phase 4b marked done),
>   `docs/gpt-5_4_mini.md`, `docs/local_tts_impersonation_plan.md` and the CLAUDE.md
>   known-failure list are updated. Stale docstrings in `audio/tts.py` and `motion_route.py`
>   are fixed.
> - Tests: `test_router_downgrades_*`/`allows_*` (15), `test_router_sleep_candidate_must_be_standalone`
>   and `test_actor_harness_strips_speaker_prefixes` are deleted from gating. 14 gating tests
>   moved to `classify_deterministic`. The gating module runs 407 tests with 4 failures, all
>   of them in the `c00eed5` baseline.
>
> Skipped:
> - `test_review_regressions::test_intent_classifier_allows_known_named_person_memory_topic`
>   is RETARGETED to `classify_deterministic` instead of deleted. Its mocked LLM was never
>   reached, and it is the only coverage of live known-person memory routing.
> - `"conversation.repair"` stays in `ACTION_ROUTER_EXECUTE_ACTIONS`. `decide()` can no longer
>   produce it, but dropping it is a policy edit (and `test_action_router_execution_gate`
>   pins it), not a deletion.
> - Fixed in the review pass (comment-only): stale mentions of deleted symbols in
>   `interaction.py` (`_deterministic_self_query_intent`, `_SELF_QUERY_SKIP_INTENTS`),
>   `action_router.py` (`tools/tool_router_report.py` ×4, the "LLM-decided motion branch in
>   `_handle_router_takeover_action`", `_GAME_STOP_REQUEST_RE`) and `tool_router.py` (the
>   "JSON-prose router prompt" / `_handle_router_takeover_action` arg-name lists), plus the
>   CONTEXT.md "action-router guardrails" bullet. `test_action_router_skip` re-pins the 13
>   chat/self-query utterances (and an offline pair) that the deleted skip tests covered as
>   `decide()` → `conversation.reply`.

Owner already scheduled this: `docs/tool_router_scope.md:113-122` "Phase 4b, once the flag has
held off in the field" (off since 2026-08-13). No API call is paid today; removal is
behavior-neutral (only the unlogged `decision.reason` string changes).

- **JSON-prose LLM router** (`ACTION_ROUTER_LLM_FALLBACK_ENABLED=False`, ~1,050):
  - `action_router.py`: make the guard at 4133-4141 an unconditional return and drop comment 4120-4132;
    delete LLM tail 4144-4180, `warmup` 4183-4209 (+ call `main.py:1620`), `_client` + OpenAI/apikeys
    imports (20, 25), `_SYSTEM_PROMPT` 362-446, `_coerce_decision` 3501-3574, `_strip_code_fence`
    1487-1496, `ACTION_CATALOG`/`_VALID_ACTIONS`/`PERFORMANCE_ACTIONS`, `ACTION_CATEGORIES` 349-351;
    skip machinery 3917-4044 + call sites 4082-4118; `_apply_context_overrides` dead branches 3745-3816
    and 3830-3913; regexes `_FORGET_SPECIFIC_REQUEST_RE` 662-672, `_BOUNDARY_REQUEST_RE` 706-716,
    `_REPAIR_REQUEST_RE` 766-774 + `missing_required_evidence_reason` repair branch 3332-3333,
    `_GAME_STOP_REQUEST_RE` 477-482, dead part of 787-843 (787-798, 811-843: `_TOPIC_KNOWLEDGE_QUERY_RE`,
    `_NAMED_DAY_EXPLANATION_RE`, `_EVENT_CONTINUATION_STATUS_RE`, `_PRONOUN_ONLY_INTRO_RE`,
    `_NAMED_PERSON_FACT_STATEMENT_RE`, `_NAMED_RELATION_INTRO_RE`, `_RELATIONSHIP_SCORE_QUERY_RE`),
    `_pending_question_context`, `references_person_memory_target` import (22), `threading`, `_connectivity`.
  - KEEP in action_router: `decide()` shutdown pre-pass + the three explicit classifiers;
    `_apply_context_overrides` 3705-3743, **3817-3828 (live name_correction→event.cancel override)**, 3914;
    `_EVENT_CANCEL_OR_STALE_RE` 799-810; `_text_has_identity_name_correction_content`; `ActionDecision`,
    `log_decision`, `_compact_json`; `ACTION_SPECS`, `EXECUTABLE_ACTIONS`; `tool_router_owns*`;
    `_clean_name_arg`, `_MUSIC_PLAY_REQUEST_RE`, `_HUMAN_VISUAL_PLAN_RE`, `_is_recent_discard_request`.
  - `interaction.py` dead `_handle_router_takeover_action` arms: repair 23463-23471, performance.impersonate
    23494-23500, forget_specific 23502-23523, who_is_speaking..status.battery 23533-23725, system.sleep
    23753-23773, motion.* 23775-23816 (keep 23817 `return None`); helpers `_router_repair_move`,
    `_router_system_command`, `_looks_like_date_query` + `_DATE_QUERY_PAT` 4237-4244,
    `_visible_known_name_for_intent`; the repair check in `_intent_execution_block_reason` 2233-2239 (then
    its `router_action` param is unused); router `emotional.boundary` block 30158-30206 (keep
    `_handle_router_emotional_boundary`, live via tool path); router event.cancel executor 30830-30860
    (event.cancel is not in `ACTION_ROUTER_EXECUTE_ACTIONS`). KEEP arms `conversation.reply`,
    `identity.name_correction`, `_PLAN_ROUTER_ACTIONS` (offline), `memory.recent_discard` (offline),
    `system.shutdown`; `_generate_repair_response`; `_handle_router_impersonation`, `_handle_router_motion_action`,
    `_handle_explore_invite`, `_handle_face_requester`, `_motion_route_from_tool_args`, `_handle_motion_route`.
  - config 6093-6104, 6188-6220 (`ACTION_ROUTER_MODEL` et al.), comment 11093-11094.
  - Tests: delete `test_action_router_skip` Deterministic/SelfQuery/rollback tests,
    `test_regex_routing_guards::JsonProseRouterRetirementTest` + test at 785, `test_action_router_catalog`
    tests at 49/64 and the dead-override tests (**keep 182, 219, 238 — they test the live override**),
    ~16 `test_router_downgrades_*`/`allows_*` in gating (8216-8703) + asserts at 3742,
    `test_conversation_revamp` ~600-618, known failure
    `test_review_regressions::test_router_keeps_known_named_person_topic_as_memory_query`. Edit
    `test_action_router_catalog` 23-47/272, `test_regex_routing_guards:561`, `test_exploration:124-125`,
    `test_field_2026_08_03:119,123`, `test_action_router_replay` fixtures.
- **Intent-classifier LLM fallback** (SAFE, ~190): `classify` 401-479, `_classify_with_llm` 536-556,
  `_llm_label_blocked` 487-511, `_PROMPT_TEMPLATE` 357-393, `_LOCAL_SYSTEM_PROMPT` 395-398, `_VALID_INTENTS`
  27-41, `_BARE_TOPIC_RE` 116, `_client` + imports, `_log` (22), config 2602-2608, stale docstring 1-11.
  KEEP `classify_deterministic`, `_deterministic_label`, `_memory_query_allowed`, all `*_QUERY_RE`
  (`_BATTERY_QUERY_RE` is imported elsewhere), `_TOPIC_KNOWLEDGE_QUERY_RE`, `_MUSIC_PLAY_ACTION_RE`,
  `_NAMED_DAY_EXPLANATION_RE`. ~29 gating references move to `classify_deterministic` (several call the
  LLM path unmocked today). `test_action_router_execution_gate:431` is an EDIT.
- **Tool-router Phase-0 shadow** (~90): `tool_router.py` 604-674 (`_SYSTEM`, `shadow_decide`, `start_shadow`),
  `_client` + imports, `_log`; call site `interaction.py` 29933-29940; config 5927-5935;
  `user_config.example.py:52`; 5 tests in `test_tool_router.py` 60-117; `tools/tool_router_report.py` and
  `TOOL_ROUTER_TEST_SCRIPT.md`. Fix stale module docstring. KEEP `_tool_name`, `_NAME_TO_KEY`,
  `tool_schemas`, `tool_schema_for`, `live_reply_tools`, `resolve_tool_call`.
- **Action-router background shadow** (SAFE, ~20): `start_shadow_decision` 4238-4248,
  `_router_audit_note_execute_disabled` 1811-1813, `ACTION_ROUTER_SHADOW_ENABLED`.
- `tool_router._DEFAULT_LIVE_ACTIONS` (436-484) duplicates `config.TOOL_ROUTER_LIVE_ACTIONS` exactly
  (drift hazard) — delete; edit `test_field_2026_08_03:123`.
- `LEGACY_COMMAND_FUZZY_EXECUTE_ENABLED` is undefined → collapse `interaction.py` 2144-2147. KEEP fuzzy
  MATCHING in `command_parser` (it drives ~14 is-command checks and the game stop-confirmation).
- `tools/conversation_text_harness.py` calls `classify()`: delete it with `gpt5_ab_test` and
  `test_actor_harness_strips_speaker_prefixes` (D10), or switch it to `classify_deterministic`.
- Docs: CONTEXT.md:357 and :3043 (humor/character went live 2026-08-13); stale docstrings.

### Stage 3 — Flag-off paths (MEDIUM: kill switches, most env-flippable) — ~1,400 prod lines

Each is off by a constant or `_env_bool` default; none is a runtime fallback.

- **Software AEC** (D7; `AEC_SOFTWARE_ENABLED=False`, measured ineffective): `audio/aec.py` (277),
  `sd_guard.py` **87-100** (keep `return result` at 101), `wake_word.py` **317-323** (keep line 316),
  config 4425-4448 (10 `AEC_*` knobs), `tests/test_aec.py`. KEEP `echo_cancel.py`, `hardware_aec.py`,
  `AEC_SUPPRESSION_FACTOR`, `AEC_SEQUENCE_IDLE_RELEASE_SECS`, `AEC_RELEASE_ON_QUEUE_DRAIN`.
  Docs CONTEXT.md:1452, rework.md:324.
- **Gap-merge phase 1** (`GAP_MERGE_ENABLED=False`; owner decision "finish pending reply" already made):
  `_GapSpeechDetected` 14421-14431, `_reply_gap_speech_onset` 14516-14559, `_merge_gap_speech`
  14562-14610, merge retry loop ~31282-31325, `gap_check_enabled` plumbing (16292, 16320, 16553, 17073,
  17172-17178, 17384), config 5443-5444, `utils/runtime_report.py:42`. KEEP `_gap_span_audio`,
  `_gap_voiced_runs`, `_gap_recovery_on`, arm/disarm, `_maybe_catch_up_gap_speech`. Tests:
  `test_gap_speech` ReplyGapOnset/MergeGapSpeech/StreamGapCheck; edit `test_knobs_exist`.
- **VAD barge-in + post-speech flush** (`VAD_BARGE_IN_ENABLED`, `POST_*_FLUSH_AUDIO_BUFFER` all False):
  barge branch 32022-32047, `_interrupt_ack` (4734), `_vad_barge_in_enabled`, `INTERRUPT_ACKNOWLEDGMENTS`
  (2288-2297); `_post_tts_flush_needed` state machine (global 509, sets 3488/32045, resets 4335/32139/32248,
  consumers 31708-31710/31946-31947), `flush_buffer` computation 3486-3490. KEEP speech_queue
  `flush_on_playback_stop` plumbing. Tests gating 4450, 10732-11259.
- **Sleep transcribed-wake fallback** (`SLEEP_ONNX_ONLY_WAKE=True`): interaction 31592-31611,
  `_wake_from_sleep_if_transcribed` 15268-15286, `_is_sleep_wake_transcript` 4600-4635, main.py GUI wake
  runner 2209-2215; ~4 gating tests (~1630-1690), 2 in `test_regex_routing_guards` (450-462).
- **Slow-path ack** (`SLOW_PATH_ACK_ENABLED=False`): interaction 4675-4701, **4759-4796 and 4802-4913**
  (keep live `_word_count` 4798-4799), `_simple_question`, `_last_slow_path_ack`, global 428, call sites
  25991-25994, 31276-31280, prefill thread 32186-32190, config 7869-7901. KEEP
  `_mark_first_response_queued/_audio_started`, `_prefill_motion_route_ack_cache`, `_prefill_wake_ack_cache`.
  Tests gating 4555-4730, 8812, 8831, 10255-10285.
- **Latency filler** (`LATENCY_FILLER_ENABLED=False`): `_speak_filler` 4738-4756, timer body 4927-4949,
  `_last_filler`, config 7853-7867. Stub the timer to return a set Event (tools harness and
  `production_replay` patch it).
- **Local-TTS WAV cache** (`LOCAL_TTS_CACHE_ENABLED=False`): tts.py 1466-1498, write-back 1596/1670/1720-1735,
  `_local_cache_wav`, `_ensure_cached_local`, local branch of `is_cached`; `local_tts.cache_identity`
  (121-133). `local_tts.synthesize` becomes dead only if the Qwen full-buffer block also goes (D6: keep).
  ~6 tests in `test_local_tts`. Docs CONTEXT.md:559, `user_config.example.py:94`.
- **Slim-contract rollback** (`TURN_PLANNER_SLIM_CONTRACT=True`): else branch **16503-16508** (keep
  16502), dedent 16479-16502; `social_frame.build_directive` 435-482; `llm._RESPONSE_LENGTH_TOKEN_BUDGET`
  515-521 + `Target:` branch 523-526/533-535. Tests `test_comedy_modes` 88-104,
  `test_conversation_revamp` 257/510, `test_conversational_persona:50`, gating 5912,
  `test_turn_planner_slim_contract:75`.
- **Proactive flag-offs:** onboarding LLM rephrase (`_maybe_rephrase`, `_openai_client`) SAFE;
  scenery-change remark (`_step_scenery_change`, `episodic_hooks._maybe_queue_scenery_remark`/
  `take_scenery_remark`/`_pending_scenery_remark`, `llm.scenery_change_remark` 2293-2326, consciousness
  `_scenery_remark_pending`; keep `_previous_startup_caption`); startup profile question
  (`_pick_startup_profile_question`, `_build_startup_profile_question_prompt`, branch 10716-10734,
  `question_key/question_depth` params — removes known failure
  `test_first_sight_sparse_profile_uses_basic_profile_question`); generic sound-event branch 7320-7326.
- **OpenAI lifeform scan** (`LOCAL_ANIMAL_DETECTION_ENABLED` default True): `detect_lifeforms` 728-792,
  `_scan_loop` **1519-1524 and 1530-1546** (keep 1525-1529), `last_monitor_time`, `monitor_interval`,
  `monitor_elapsed`, config 6368-6381 (KEEP `SCENE_CHANGE_MONITOR_MAX_TOKENS`). 3 tests in
  `test_scene_monitor`.
- **Alternate backends never selected:** `CALLBACK_BANK_BACKEND` openai branch (callback_engine
  378-382, 394-395); `CONVERSATION_ARC_BACKEND` local branch (topic_thread 541-543, 554-560 + `rich=False`
  prompt; tests `test_conversation_arc` 235/262). `GUI_BACKEND` (only "pyside6" exists).
  `MOTION_EAGER_ENDPOINT_DURING_GAMES` knob. `IDLE_LISTEN_DURING_DJ_PLAYBACK`,
  `SOUND_EFFECTS_DRIVE_SUPPRESSES_MIC`. 4 unused `DJR3X_*_TEST_OPT_IN` env hooks. Legacy Jeopardy
  daily-double no-wager branch (`games.py` 2604-2608).
- **DJ local library** (D11): `dj.scan` 79-113, `_index`+lock 43-44, `handle_request` steps 1-2 (129-167),
  local-genre loop 198-211, local playback 375-376, `play_by_vibe`, `now_playing`, mutagen/fuzz_process
  imports, `MUSIC_DIR`, `mutagen` in requirements; fix the false prompt at interaction.py:26205.
- **Radar orient** (D7): `_maybe_radar_orient` (motion_agency 2106-2225), call 3552, `orient_*` state,
  config 9712-9740, 9791. KEEP `_radar_bodies`, `_any_visible_face`, `_voice_bearing_fresh`; rename
  `MOTION_RADAR_ORIENT_VOICE_DEFER_SECS` (now only gates idle wander, 3009). **`tests/test_motion_agency.py`
  setUpModule (32-40) turns radar orient ON for the whole module** — remove it and re-run the whole module;
  ~13 cases + 1 in `test_wake_orient`.

### Stage 4 — Lean-off rollback and governor-rejected generators (needs D1, D4, D5) — ~3,500 prod lines

Two mechanisms make this code dead: `LEAN_BRAIN_ENABLED` checks, and the governor rejecting
every purpose in `LEAN_SUPPRESSED_PROACTIVE_PURPOSES` (config 489) at `action_governor.py`
369-375 before scoring. The generators still run every tick, arm cooldowns and submit lines
that are always rejected. **Presence-path `memory_followup`/`celebration_checkin` lines are
LIVE** (they bypass governor scoring) — do not touch them.

- **Idle-banter trio** (interaction.py, ~875): `_maybe_interest_idle_followup` 5360-5477,
  `_maybe_low_memory_idle_question` 5480-5607, `_maybe_idle_banter` 7624-7956, the Lean-off `else:` at
  31845-31860, exclusive helpers (`_profile_fact_count`, `_next_profile_question`,
  `_format_low_memory_question`, 3 `_IDLE_BANTER_*` banks, `_idle_plans_*` incl. `_mark_idle_plans_asked`
  5682-5705, interaction `_governor_enforcing`, `_idle_has_live_topic`, `_last_user_turn_text`,
  `_last_user_turn_was_low_content`, `_idle_should_volunteer_take`, `_idle_banter_directive`), state
  `_idle_banter_count`/`_threshold` (+ resets 536-537), `_interest_idle_followups_spoken`,
  `_low_memory_idle_questions_spoken`, `_idle_plans_asked` **and their live writers** (13377,
  20518-20520, 20814-20815, 20881, 32131); `profile_questions.next_profile_question` 126-164; config
  `IDLE_BANTER_*` (incl. `IDLE_BANTER_SECS`, `IDLE_BANTER_MAX_SECS`), `INTEREST_IDLE_FOLLOWUP_*`,
  `LOW_MEMORY_IDLE_QUESTION_*`, `IDLE_PLANS_QUESTION_PROBABILITY`, `TOPIC_BAN_PROACTIVE_SUPPRESS`;
  `_OPENER_DIVERSITY_PURPOSES` entries `idle_monologue`/`celebration_checkin`; stale source string in
  `action_governor._ACTIVE_CONVERSATION_ALLOWED_SOURCES`. KEEP `_proactive_opener_repeats` (imported by
  speech_engine), `_floor_held_until`, `_last_proactive_line_at`, `_session_exchange_count`,
  `IDLE_BANTER_RECENT_QUESTION_DEDUP_SECS`, `LOW_MEMORY_PROFILE_MAX_FACTS`, `_maybe_idle_outro`,
  `PRESENT_REENGAGE`. Tests: delete `test_idle_banter_relevance`, `test_idle_banter_low_content`,
  `test_idle_tease_silence`, ~15 gating, 2 `test_proactive_dedupe`; edit `test_onboarding`,
  `test_tier2_question_suppression`, `test_action_governor` 303-325, `test_proactive_discipline:65`.
  Delete `tools/gpt5_ab_test.py`.
- **Consciousness steps skipped under Lean** (~425): `_step_open_thread_followup`, `_step_news_remark`,
  `_step_interest_discovery`, `_step_holiday_plans`, `_engaged_interest_topics`, their state vars
  (`_last_news_remark_at`, `_news_remarks_this_session`, `_open_thread_asked_persons`,
  `_last_holiday_plans_check_at`, `_last_interest_discovery_at`, `_interest_discovery_sessions_asked`) and
  config (`HOLIDAY_PLANS_PROBABILITY`, `_CHECK_INTERVAL_SECS`, `INTEREST_DISCOVERY_*`, `NEWS_REMARK_*`,
  `OPEN_THREAD_PRIORITY`). KEEP `_next_holiday_plan_for_person`.
- **Governor-rejected generators** (~1,300 incl. D5 items):
  consciousness — safe contiguous spans **8215-8446** (small talk, `_voice_pov_as_micro_behavior`, roast
  helpers), **8492-8504**, **8548-8655**, **8658-9002** (visual curiosity, lull callback), plus
  `_step_group_lull` 11343-11413, `_step_weekly_smalltalk`+`_pick_weekly_slot` 11721-11872,
  `_step_disengagement` 7030-7060, `_mood_clause_for` 8039-8081, state vars 114-133/286-287;
  **KEEP 8449-8489 `_visual_curiosity_blocked_by_empathy` and 8507-8545
  `_note_emotional_checkin_fired`/`note_emotional_checkin_boundary`** (live boundary handling at
  interaction 21255/21325). Trigger A2 block 11988-12027 (fixes A5).
  idle_behaviors — `do_private_thought`, `do_aspiration`, `do_memory_musing`, `do_ambient_observation`,
  `do_appearance_riff`, `do_people_roast`, `do_live_vision_comment`, `do_empty_room_joke` (whole function;
  unreachable while `BOREDOM_ENABLED`), banks `PRIVATE_THOUGHTS`/`ASPIRATIONS`/`EMPTY_ROOM_JOKES` +
  `EMPTY_ROOM_JOKE_PROBABILITY`. **Per D4: replace their `_idle_micro_behavior_choices` entries with one
  no-op entry carrying the same weight** (16/23 people-present, 14/19 crowd; ×3 novelty boost applies).
  Transitive: `callback_engine.build_lull_prompt`, `CALLBACK_LULL_*` config, `object_qa.mark_asked_labels`/
  `known_answer`, `room_model.human_label`, `SituationAssessor.recent_speech_turn_count`,
  `rex_pov.active_pov_text`, `VISUAL_CURIOSITY_*` (12), `MOOD_AWARE_SMALLTALK_ENABLED`,
  `MOOD_ANALYSIS_PROBABILITY`, `WEEKLY_SMALLTALK_*`, `ROOM_MODEL_NOVELTY_MAX_SIGHTINGS`, `VENUE_NAME`,
  `_question_key_for_presence_line` small_talk branch. KEEP `_pick_appearance_hint`, `_get_or_detect_mood`,
  `get_cached_mood`, callback_engine `lull_gates_clear`/`pick_lull_premise`/`spend_lull_premise`,
  `do_ambient_scan`, `do_idle_clip`, `do_empty_room_observation`, `do_bored_environment_snark`,
  `STARTUP_EMPTY_ROOM_JOKES`, emotional check-in Triggers A and B.
  Tables: governor priority rows for the 11 suppressed purposes, `_REPEAT_COOLDOWN_EXCLUDED_PURPOSES`,
  dead allowed-source entries; agenda `_PROACTIVE_RULES`/grace rows ONLY for small_talk, visual_curiosity,
  lull_callback, reengagement, ambient_observation, appearance_riff, people_roast. **KEEP agenda rows for
  idle_monologue (do_idle_clip claims it), memory_followup, celebration_checkin, startup_empty_room,
  group_turn_invite, presence_reaction, emotional_checkin.**
  Tests: gating 5357/5369/5390/11202, live-vision cases in `test_idle_behaviors_cooldown`, known failure
  `test_proactive_discipline::test_idle_monologue_is_excluded_from_the_cooldown` (delete); edit
  `test_object_detection:287`, `test_object_qa:78`, `test_room_model:168`, `test_room_reaction:165-189`,
  `test_callback_humor` lull-prompt case, 2 `test_rex_pov` cases. Delete tests BEFORE config keys (many
  `mock.patch` calls lack `create=True`: gating 9193/9244/8008-8010/10560-10578/1390-1540,
  `test_sound_events:256`, `test_room_reaction` 168-189).
- **Lean-off reply branches** (~65): `build_turn_plan` selection **16381-16383**, `else llm.get_response`
  **16581-16586** (keep 16587-16590), classic half of `_prepare_stream_sentence` 16918-16928,
  `social_frame.govern_stream_sentence` 727-757, `comedy_modes.polish_stream_sentence` 365-380. Tests
  `test_streaming_tts` 186-300 (StreamingOrchestrationTest pins Lean off), `test_sharp_roast_tier`
  119-140/209, `test_turn_planner_slim_contract` 113-133. Edit `utils/runtime_report.py:37,71` and
  `tests/test_runtime_report.py` if the flag goes.
- **`build_turn_plan` tail** (~740; every `build_lean_turn_plan` trigger returns by line 940):
  conversation_agenda 942-1175 **replaced by a terminal default return** (a `pending_closure` TTL race can
  fall through today and write interest facts via steering); plans state **510-524**; `_plan_*` **526-636**
  (keep `_NEW_DIRECTION_PAT` comment 639-644); `_next_useful_question`/`_known_fact_keys` 421-450;
  `_friendship_question_allowed` 453-477; `_PLAN_STATEMENT_PAT` 48-56; `reset_plans_state` (+ call
  interaction 32158); `build_turn_directive` 1178-1188; all of `plan_intent.py`;
  `conversation_steering.note_user_turn` 216-284 + its ~65 lines of helpers; config `WHAT_IF_PLANS_*`,
  `PLAN_INTENT_*`, `PLANS_CLARIFY_TTL_SECS`, `REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED`,
  `SUBJECT_CHANGE_ON_CUE_ENABLED`. KEEP head 715-940 (its directive TEXT is load-bearing:
  `select_mode` matches 'grief'/'no roast'), `_finish`, `_populate_signals`, `_wants_new_direction`,
  `_looks_like_*`, `social_scene.unknown_group_context`, `_record_banned_topic`. Tests: delete
  `test_what_if_plans`, `PivotAgendaTest`, `test_friendship_question_blocked_for_minor`,
  `PlanIntentLeanGateTest`, known failure `test_conversation_revamp::test_one_word_passion_answer_drives_engaged_curiosity`;
  edit `test_turn_plan`, `test_conversation_replay`, ~12 `build_turn_directive` tests, gating steering
  block, `evals/run_quality_eval.py` (97-101 swallows errors — make it fail loudly).
  `_arm_visible_unknown_identity_followup` (14169-14226) is only NEAR-dead — leave it.
- **Classic-fallback-only prompt text** (D9, ~250): cancellation 16390-16403 (after A7 re-homes it),
  slim block 16478-16502, social_frame 485-586, comedy_modes `build_directive` 225-263 (**still used at
  16496 for banked callbacks — keep unless that site moves**), `_SLIM_STANCE` 266-280,
  `build_slim_directive` 283-303, `_recent_premise_summary` 505-514, llm agenda append 1510-1513.
  KEEP `recent_openers_to_avoid`, `_news_followup_story`, `assemble_system_prompt`.

### Stage 5 — Governor dead parts (D3a) — ~150 lines

`ACTION_GOVERNOR_SHADOW_MODE` and its `active()` read; `speech_engine` **322-329** only (321 and 330 are
the live `governed=False` path), 655-690, condition simplifications at 547 and `action_governor` 232-234,
263-264, 283; `_do_speak`'s 6 `_mark_governor_candidate(None, ...)` calls; `generate_and_speak._task` token
branches; `candidate_id` plumbing; `CONSCIOUSNESS_ALLOW_PROACTIVE_IN_ACTIVE` collapse makes
`speech_engine` 394-400 and `action_governor` 447-448/523 dead. `test_action_governor` 269-273/331 patch
ENFORCE False — delete those. Then D3b (cutover) as a separate, live-tested change if chosen.

### Stage 6 — Runtime fallbacks (D6) — ~550 prod lines + ~140 MB assets + 2–4 pip packages

- **dlib face backend:** `_load_dlib` 204-245, `_detect_rects` 250-289, `_detect_faces_dlib` 361-384,
  `detect_faces` fallthrough 320, globals 44-47, fallback in `_load_models` 136-147; `people.py` 169-172
  (128-d branch); config `FACE_DETECTOR_UPSAMPLE` 826-831, `FACE_DETECTOR_MIN_CONFIDENCE` 842-847,
  `FACE_RECOGNITION_DISTANCE_THRESHOLD` 3967, `FACE_RECOGNITION_MARGIN` 3972,
  `FACE_IDENTIFY_STRONG_DISTANCE_DLIB` 3998, `FACE_LANDMARK_MODEL`, `FACE_RECOGNITION_MODEL`; setup_assets
  dlib step; `dlib` in requirements/`setup_macos.sh`; models `shape_predictor_68` (99.7 MB),
  `dlib_face_recognition_resnet` (22.5 MB). InsightFace load failure then means no face detection — make it
  fail loudly. **Do NOT remove** the consciousness 396-401 no-confidence clause (tests rely on it; edit
  the comment). KEEP `face.active_backend()`, `face_landmarker.task`, mediapipe. Tests
  `test_face_match_margin`, `tools/test_face_id.py:86`.
- **MediaPipe EfficientDet object fallback:** MP block of `animal_detector._load_model`, MP branches of
  `detect_animals`/`detect_objects`, `_model_path`, `MEDIAPIPE_OBJECT_DETECTOR_MODEL`,
  `LOCAL_ANIMAL_DETECTION_MODEL`, setup step, `efficientdet_lite0.tflite`. KEEP `_rf_detections_to_mp`,
  record builders, mediapipe. Tests `test_rfdetr_backend`, `test_animal_detector`, `test_scene_monitor`
  source assertion.
- **Resemblyzer:** speaker_id 45-54, 84-90, 141-144, 194-198, 256-entry at 468; `voice_score.py:43`;
  setup_assets 917-955, 1566-1567, entry 93; config 761; `tools/voice_recordings.py:66,72`; make unknown
  `VOICE_EMBEDDER` values FAIL LOUDLY instead of falling through (speaker_id 79-91). Tests
  `test_speaker_challenge.py` (known failing), 3-4 `test_voice_backend`, 2 `test_voice_primary_identity`;
  gating 9931/10021 switch to campplus. pip: `resemblyzer` (+`librosa`, `webrtcvad` after confirming the
  ASR stack doesn't need librosa). Update config 4034 rollback comment.
- **ElevenLabs v2/turbo support:** `_stitch_previous_text` (1951-1972), `TTS_V3_STITCH_*`, non-v3 returns in
  `_pin_v3_stability`/`_v3_tags_active`/`_v3_seed`, the `previous_text`/`stream_prev_text` plumbing
  (~40 sites across speech_queue/tts/interaction); update config 1752-1758 comment; tests
  `eleven_multilingual_v2` cases in `test_v3_audio_tags`. `seed_voice_test.py` then goes too.
- **LATER (after CAM++ runs live, A1):** ECAPA rollback (`_load_ecapa`, `_embed_ecapa`, ECAPA offset in
  `voice_score.map_similarity`, `interaction._ecapa_genuine_band` 2463-2479 + uses 10693/27384, config
  7105-7144, speechbrain, 85 MB) and voice-signature→person resolution (interaction 4126-4159,
  `_signature_resolves_to_person` 10574-10610, `voice_signatures.attach_person` 212-232 + named branch of
  `bump`, config 6267-6276, 7146-7151; removes known failure `test_cold_signature_needs_strict_bar`).
  Keep `torchaudio` (CAM++ needs it).

### Stage 7 — Flag collapse (optional, low value) — ~400 lines

~100 always-True kill switches whose False side is a one-line early return. Recommend doing this only
opportunistically when touching a module, never as a sweep. Do NOT collapse: runtime-set flags
(`PLAY_*`, `GUI_ENABLED`, `NO_AUDIO_MODE`, `STARTUP_GROUP_GREETING_ENABLED`,
`MOOD_AWARE_FIRST_SIGHT_ENABLED`), `TOOL_ROUTER_LIVE_ENABLED`, `MOTION_ROUTE_ENABLED`,
`MOTION_ROUTE_REQUIRE_TRUSTED_TRANSCRIPT` (wheel safety), motion safety toggles, vision kill switches,
`WAKE_WORD_ALLOW_DURING_TTS`/`WAKE_WORD_SHUTDOWN_DURING_TTS`, `LLM_STREAMING_TTS_ENABLED`, anything in
`utils/runtime_report._CONFIG_KEYS` without editing it.

### Stage 8 — Docs

- Move stale history to `docs/archive/`: `rework.md`, `docs/junecodereview.md` (after carrying its still-open
  items), `comedy_improvements.md`, `callback_humor_design.md`, `active_speaker_detection.md`,
  `exploration_mode_plan.md`, `gpt-5_4_mini.md`, `local_tts_impersonation_plan.md`,
  `motion_sensing_roadmap.md`, `CONVERSATION_TEST_SCRIPT.md`, the
  `field_2026_09_15_*`/`come_here_2026-09-06`/`jeopardy_live_reliability_review` field notes.
- CONTEXT.md: move the dated changelog (1269-3032, 3055-3819; ~2,500 lines) to
  `docs/archive/context_changelog.md`, keep the architecture reference (1-1268) + "Likely Future Work"
  (refresh it); refile the 2026-09-16 mic item misfiled under `## GUI`.
- CLAUDE.md known-failure list: remove the failures deleted with their dead code
  (`test_idle_monologue_is_excluded_from_the_cooldown`, `test_first_sight_sparse_profile_uses_basic_profile_question`,
  `test_one_word_passion_answer_drives_engaged_curiosity`,
  `test_cold_signature_needs_strict_bar`), add/resolve A6.
- Untracked, unreferenced: `.recovery/` (694 MB, 09-16 snapshot) and `models/dj_r3x/` (421 MB avatar
  build files) — owner to delete or gitignore.

## Part D — Execution protocol (every stage)

1. Re-derive spans at the current HEAD with `ast` (function/class `lineno`..`end_lineno`); for
   branch deletions read the enclosing `if/else` and check negations and `getattr` defaults.
2. Baseline: run each affected test module on the pre-change tree
   (`venv/bin/python -m unittest tests.<module>`, one module per process; `tools/run_lean_checks.py`
   for the Lean suite) and record failures. Tree is clean after commits, so compare against the
   previous commit by swapping single files back (`git show <base>:path > path`), never
   `git stash -u` (no-op on committed work) or a worktree (no `.env`/`apikeys.py`/`assets/`).
3. Delete tests of the dead code FIRST (many patch config keys without `create=True`), then the
   code, then config keys, then docs.
4. After editing: repo-wide grep (prod, tools, evals, tests, firmware/tools, *.md) for every removed
   symbol; `python -m py_compile` all touched files; import-smoke `main.py` in `--noaudio` with
   hardware/network blocked via `tools/run_lean_checks.py` conventions.
5. Re-run the affected modules + `tools/run_lean_checks.py`; failures must match the baseline set.
   Servo-touching test runs end with `tests.test_zzz_servo_park`.
6. Commit to `main` with the docs update, push. One stage (or sub-stage) per commit so any stage can
   be reverted alone.

## Estimated totals

| Stage | Prod lines | Gate |
|---|---|---|
| 1 Unreferenced code + safe tools | ~2,300 + ~1,350 tools | none |
| 2 Retired routing | ~1,500 | none (owner-scheduled Phase 4b) |
| 3 Flag-off paths | ~1,400 | D7, D11 for their items |
| 4 Lean-off + rejected generators | ~3,500 | D1, D4, D5, D9 |
| 5 Governor dead parts | ~150 | D3 |
| 6 Runtime fallbacks | ~550 (+ECAPA/sig ~230 later) | D6 |
| 7 Flag collapse | ~400 | optional |
| **Total** | **~9,800 prod + ~1,350 tools**, plus ~600–800 config lines and several thousand test lines | |
