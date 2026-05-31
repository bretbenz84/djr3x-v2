# DJ-R3X v2 - Concise Project Context

This file is a compact handoff for future AI/development sessions. It explains what the project is, how to run it, the main architecture, and the current design assumptions. It intentionally omits long personality transcripts, old planning notes, and exhaustive behavior specs.

## Project Summary

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It runs on macOS, combines live audio, camera perception, face and voice identity, persistent memory, LLM conversation, TTS, games, music, and physical droid hardware control.

The core loop is:

1. Detect or receive a human utterance.
2. Resolve who likely spoke.
3. Interpret the intent.
4. Choose a local action, routed action, or LLM response.
5. Speak or log the response.
6. Save useful memory only when appropriate.
7. Emit telemetry showing what happened.

Primary workspace:

```bash
/Users/bbenziger/djr3x-v2
```

Use the project virtual environment:

```bash
source venv/bin/activate
venv/bin/python -m unittest discover -s tests
```

`pytest` is not assumed to be installed; use `unittest` unless the repo changes.

## Supported Runtime Modes

DJ-R3X is designed for macOS, preferably Apple Silicon.

| Mode | Command | Behavior |
| --- | --- | --- |
| Full voice mode | `python main.py` | Uses mic, wake word, transcription, speaker ID, TTS, camera, memory, and available hardware. |
| GUI mode | `python main.py --gui` | Adds the PySide6 dashboard while keeping normal audio behavior. |
| Text-only GUI mode | `python main.py --gui --noaudio` | Disables mic/audio/TTS calls. GUI text input is processed like speech and responses appear as text. |
| Jeopardy launch | `python main.py --jeopardy` | Starts directly in Jeopardy mode and skips normal startup introductions. |

Startup flags owned by `main.py`:

| Flag | Purpose |
| --- | --- |
| `-gui`, `--gui` | Open the optional dashboard. |
| `-jeopardy`, `--jeopardy` | Start directly in Jeopardy mode. |
| `-noaudio`, `--noaudio`, `--no-audio` | Disable microphone capture, wake word, audio output, and ElevenLabs calls. |

In no-audio mode, `main.py` sets runtime-only config values:

- `config.NO_AUDIO_MODE = True`
- `config.AUDIO_OUTPUT_SUPPRESSED = True`

It skips Whisper verification, audio stream startup, wake word listening, audio scene analysis, TTS prewarm, speaker-ID preload, startup/shutdown audio, listening chime, ElevenLabs fetches, and direct playback.

## Configuration And Secrets

Tracked configuration:

- `config.py`: tunable defaults, model names, thresholds, feature flags, servo defaults, latency settings.
- `.env.example`: host-specific config template.
- `apikeys.example.py`: API key template.

Untracked local/runtime files:

- `.env`: machine-specific camera, microphone, hardware ports, servo limit overrides.
- `apikeys.py`: OpenAI and ElevenLabs keys.
- `assets/memory/people.db`: local person database.
- `assets/audio/tts_cache/`: generated ElevenLabs cache.
- downloaded model assets.

Never commit real secrets, local databases, generated TTS cache, local music, or downloaded model files.

## Repository Map

Important entry points and modules:

```text
main.py                  Startup, CLI flags, service orchestration, shutdown.
state.py                 Runtime state machine.
world_state.py           Thread-safe shared perception/session state.
config.py                Tunable defaults and feature flags.

audio/
  stream.py              Mic stream and rolling buffer.
  vad.py                 Silero VAD.
  transcription.py       mlx-whisper plus OpenAI fallback.
  speaker_id.py          Resemblyzer voice embeddings and speaker matching.
  wake_word.py           OpenWakeWord loop.
  speech_queue.py        Prioritized response queue and playback/text completion.
  tts.py                 ElevenLabs TTS, cache, no-audio bypass.
  echo_cancel.py         Playback suppression/AEC state.
  scene.py               Background audio scene analysis.

intelligence/
  interaction.py         Main turn pipeline for speech and GUI text input.
  consciousness.py       Proactive loop, greetings, presence, empty-room behavior.
  dialogue_act.py        Cheap conversational-frame gate before executable actions.
  action_router.py       LLM action routing.
  command_parser.py      Fast/local command recognition.
  intent_classifier.py   Intent fallback and deterministic guards.
  llm.py                 Main LLM prompt assembly and response generation.
  local_llm.py           Ollama sidecar for low-latency local calls.
  empathy.py             Affect classification and emotional event handling.
  social_frame.py        Response shape/governance cleanup.

memory/
  database.py            SQLite connection, schema, migrations.
  people.py              People, face/voice biometrics, familiarity.
  facts.py               Person facts and observations.
  events.py              Upcoming/follow-up events.
  emotional_events.py    Sensitive/celebratory emotional memories.
  social.py              Inter-person relationship edges.

vision/
  camera.py              Camera capture.
  face.py                dlib face recognition.
  scene.py               Environment/scene analysis.
  pose.py                Body pose/gesture hooks when available.
  proxemics.py           Distance/space estimation.

gui/
  dashboard.py           PySide6 dashboard and text input surface.

features/
  dj.py                  Music playback.
  games.py               Game orchestration.
  jeopardy.py            Jeopardy mode.
  trivia.py              Trivia.

hardware/
  servos.py              Pololu Maestro and servo behaviors.
  leds_head.py           Head LED Arduino.
  leds_chest.py          Chest LED Arduino.
```

## Runtime Architecture

### Startup

`main.py` owns the process lifecycle:

1. Verify database schema and local configuration.
2. Initialize optional hardware; missing ports disable features gracefully.
3. Start audio, camera, awareness, consciousness, and interaction services.
4. Prewarm output and preload models when enabled.
5. Enter the main state loop until shutdown.

The local Ollama model `qwen2.5:1.5b` is preloaded for quick classifier/shaping tasks when configured.

### Speech And Text Turns

`intelligence/interaction.py` is the main turn pipeline.

Spoken turn:

1. VAD/wake or idle speech activation.
2. Transcribe audio.
3. Run speaker ID.
4. Fuse voice with visible/recent world state.
5. Run dialogue-act/contextual binding before executable routing.
6. Execute local handler or call LLM.
7. Queue speech/text output.
8. Extract memories after the turn unless suppressed.
9. Log `[character_loop]` telemetry.

GUI text turn:

1. `gui/dashboard.py` sends text to `interaction.submit_text(...)`.
2. The normal pipeline runs with transcription and speaker-ID bypassed.
3. Attribution prefers recent engagement, then a single unambiguous visible known person.
4. In `--noaudio`, responses are logged as text and no ElevenLabs call is made.

Use `interaction.submit_text(...)` for programmatic text injection. Do not invent a parallel text-only pipeline.

### Turn Routing Policy

The desired routing hierarchy is now:

1. Dialogue act: decide whether the utterance is answering Rex's last turn.
2. Contextual binding: bind answers to the correct person/frame, including multi-person cases.
3. Executable action gate: only run actions with strong, action-shaped evidence.
4. Conversation fallback: ambiguous turns stay conversational and can be handled by the main LLM.

`intelligence/dialogue_act.py` owns the cheap session-local frame check. Rex turns are recorded as `RexTurnFrame` objects with source, topic, target person, expected reply types, and blocked actions. This prevents normal replies such as "Yeah, that's not happening anymore" from being promoted into identity, memory, weather, game, or music commands.

`interaction.py` now has a central turn-policy gate for old fallback layers:

- Legacy `command_parser` claims go through `_legacy_command_execution_block_reason(...)`.
- Deterministic intent claims go through `_intent_execution_block_reason(...)`.
- Both gates consult the dialogue decision and `action_router.missing_required_evidence_reason(...)`.
- Legacy fuzzy command execution is disabled by default via `LEGACY_COMMAND_FUZZY_EXECUTE_ENABLED` being absent/false.
- The live interaction path uses `intent_classifier.classify_deterministic(...)`, so the intent-classifier LLM fallback is not called during normal turns.

When investigating false positives, search logs for:

- `[dialogue_act]`: whether a turn was bound to Rex's last frame.
- `[turn_policy] blocked ...`: an old command/intent claim was rejected before execution.
- `[action_router_audit]`: final routed action, legacy claim, allowlist/block reason, and executed path.

The failure mode to avoid: a normal contextual response gets a second chance in a later legacy layer and becomes a durable action. Do not add new command/intent bypasses after the dialogue gate unless they pass the same central evidence policy.

### Output

`audio/speech_queue.py` is the central response queue. It handles priority, coalescing, playback start callbacks, and completion.

`audio/tts.py` handles ElevenLabs cache lookup/fetch/playback. In no-audio mode, `speak()` and `ensure_cached()` return before network or playback work. By default `speak()` derives expressive ElevenLabs `voice_settings` from the line's emotion (`emotion_orchestrator.voice_settings_for_emotion`, backed by `config.TTS_VOICE_SETTINGS_*`); an explicit empathy/grief override passed by the caller takes precedence.

## Latency And Telemetry

The project now has explicit latency instrumentation.

Key logs:

- `[latency]`: stage timings inside a turn.
- `[ttfs]`: time to first response queued and first audio/text start.
- `[character_loop]`: full per-turn summary: speaker, interpretation, execution, output, memory suppression, timing.
- `[action_router_audit]`: final action routing result.

Recent latency architecture:

- `audio.speaker_id.preload()` runs at startup when `config.SPEAKER_ID_PRELOAD_ON_STARTUP` is true, removing first-turn Resemblyzer load cost.
- Slow-path acknowledgments (short "One sec." receipts for known-slow `general`/`memory`/`vision` paths) and the delayed latency filler (in-character "One sec, thinking." lines) are now **disabled by default** — `config.SLOW_PATH_ACK_ENABLED = False` and `config.LATENCY_FILLER_ENABLED = False`. They felt out of place, and the streaming answer path now gets Rex's real first sentence out fast, so the latency cover is unnecessary. The machinery and tunables (`SLOW_PATH_ACK_LINES`, `SLOW_PATH_ACK_EXPECTED_SECS`, `LATENCY_FILLER_LINES`, `SLOW_PATH_ACK_IN_TEXT_ONLY`) remain; flip either flag back to True to restore. The slow-path-ack tests enable the flag explicitly to keep covering the firing logic.
- End-of-speech wait `config.SILENCE_TIMEOUT_SECS = 0.6` (was 0.9): how long of sustained silence after the user stops before transcription begins. Lowered for responsiveness on every turn; raise toward 0.8 if slow / pausing speakers get cut off mid-sentence.

When assessing responsiveness, prefer TTFS/audio-start timings over total turn duration. Total duration includes how long Rex speaks.

## Identity And Multiple Speakers

Identity combines:

- Face recognition from `vision.face`.
- Voice embeddings from `audio.speaker_id`.
- Current visible people from `world_state`.
- Recent engaged speaker/session continuity.
- Conservative fallbacks when ambiguity is high.

Important behavior:

- A hard voice threshold prevents casual misidentification.
- A softer session-sticky threshold can keep continuity during an active exchange.
- If only one known person is visibly engaged, world-state continuity may override weak voice scores.
- If multiple known faces are visible and voice confidence is low, the system can create an anonymous voice label like `unknown_voice_1` instead of forcing a person match.
- Unknown voices can be tracked within the session even before a name is known.
- Directional audio intelligence is a future design target, not currently implemented.

Recent introduction repair:

- Relationship-only intros such as "I'd like you to meet my sister" open a pending introduction slot even if no unknown face is visible at that exact instant.
- If the follow-up name matches a known visible/recent person, Rex should welcome/link that existing person rather than treating it as a rename of the current speaker.
- This prevents cases like known face Jennifer being interpreted as "rename Bret to Jennifer."

## Memory Model

Memory is stored in SQLite under `assets/memory/people.db`.

This local DB is **disposable development/test data** — it is untracked and may be
freely reset, cleaned, or fully erased during development (e.g. to clear stale rows
like duplicate test events). It is not precious user data and does **not** need to
be backed up before changes; wipe or edit it directly when iterating.

Main concepts:

- People records with names, biometrics, familiarity, and relationship metadata.
- People records include `lifetime_greeting_count` and `last_greeted_at` for grounded answers to greeting-count questions going forward.
- Face and voice biometric rows for identity matching.
- Person facts, preferences, interests, and events.
- Emotional events for celebrations, grief, wins, worries, and follow-ups.
- Social relationship edges between people.

Memory extraction runs after turns and may call OpenAI. It should be suppressed for commands or corrections where learning would be wrong. Forget/discard commands exist and should be respected.

Do not treat every utterance as a permanent fact. Prefer explicit user statements, repeated stable preferences, and meaningful life events.

## Proactive Behavior

`intelligence/consciousness.py` runs background awareness and proactive behavior.

Important proactive cases:

- Startup greeting: if a known person is in front of the camera, Rex should greet them by name.
- Empty-room startup: if nobody is visible, Rex can make a short snarky empty-room remark.
- First-sight celebration/event check-ins can happen when a remembered relevant event exists.
- Holiday-plan proactivity is major-holiday-only by default; minor public holidays require `HOLIDAY_PLANS_INCLUDE_MINOR = True`.
- The action governor arbitrates proactive candidates so Rex does not stack too many remarks.

If startup greetings feel wrong, inspect face detection timing, world-state updates, and action-governor candidate selection.

## Social Conversation Layers

The project still has several layers that shape final speech, but executable turn ownership should remain centralized:

- Dialogue-act frame gate for contextual replies.
- Action router for executable intents and evidence policy.
- Command parser for legacy exact/pattern commands after turn-policy gating.
- Deterministic intent classifier for local data/tool answers after turn-policy gating.
- Conversation agenda/social frame to keep responses short, relevant, and socially targeted.
- Empathy classifier for emotional mode and event capture.
- Memory injection for known people.

For group settings, the desired future direction is not "ignore all crosstalk." It is social turn triage: decide whether a line is directed at Rex, overheard but relevant, background crosstalk, or a group-addressed turn.

## Hardware

The robot can run without hardware attached.

Optional hardware:

- Pololu Maestro servo controller.
- Head LED Arduino.
- Chest LED Arduino.
- Camera.
- Microphone or ReSpeaker Lite.
- Speakers/audio output.

Missing serial ports are warnings, not fatal errors, unless a feature explicitly requires hardware safety validation. Servo min/max overrides belong in `.env`, using microsecond values from the Maestro Control Center. Do not connect live servos until safe travel limits are configured.

## GUI

The PySide6 dashboard is optional and launched with `--gui`.

Important GUI behavior:

- It mirrors runtime state and conversation logs.
- Its text input can submit turns through `interaction.submit_text(...)`.
- With `--gui --noaudio`, the app becomes a text-only test interface for the full conversation/router/memory pipeline.

## External Services

OpenAI is used for main chat, vision/scene analysis, extraction, and classifiers depending on path.

ElevenLabs is used for TTS in audio mode only. No-audio mode must not call ElevenLabs.

Ollama/local LLM is used as a low-latency sidecar for quick local tasks when configured.

Network calls may dominate response latency. Prefer local fast paths for clear commands, short acks for slow paths, and telemetry-driven optimization.

## Development Rules

- Use existing project patterns before adding new abstractions.
- Keep changes scoped; this repo has many interacting behavior layers.
- Do not revert unrelated dirty worktree changes.
- Preserve privacy and safety gates around memory, vision snapshotting, and identity changes.
- Avoid adding new permanent memory writes unless the user clearly intends Rex to remember something.
- In no-audio mode, avoid any mic, wake word, TTS, or ElevenLabs work.
- For frontend/GUI changes, keep text input routed through the same interaction pipeline as speech.

Useful commands:

```bash
venv/bin/python -m unittest discover -s tests
venv/bin/python main.py --gui --noaudio
venv/bin/python main.py --gui
venv/bin/python main.py
```

## Recent Architecture Changes To Preserve

- Command-line no-audio mode with `--noaudio`, `--no-audio`, and `-noaudio`.
- GUI text input routed through `interaction.submit_text(...)`.
- TTS and speech queue bypass in no-audio mode.
- Speaker-ID encoder preload at startup.
- Exact TTFS logging.
- `[character_loop]` per-turn telemetry.
- Slow-path acknowledgments (general/memory/vision) and the latency filler exist but are now **disabled by default** (`SLOW_PATH_ACK_ENABLED` / `LATENCY_FILLER_ENABLED` = False) — see the Latency And Telemetry section.
- Action-router guardrails downgrade common false positives: ongoing status updates are not event cancellations, pronoun-only fragments are not introductions, named holiday explanations are not date queries, and relationship-score questions outside games route to memory.
- Dialogue-act frame gate (`intelligence/dialogue_act.py`) protects normal replies to Rex's last turn before routers can claim them.
- Central turn-policy gates in `interaction.py` now require legacy command-parser and deterministic intent claims to pass dialogue context plus action-shaped evidence.
- Intent classifier live path is deterministic only; no extra LLM classifier call is added on normal turns.
- Stale event-cancellation acknowledgments are deterministic to avoid weird generated lines and extra LLM latency.
- Regression corpus for misroutes lives under `tests/fixtures/misroute_replays.json`; add false-positive examples there when they appear in live logs.
- Memory-query grounding for self relationship metrics and greeting counts.
- Minor public holiday proactive questions are gated behind `HOLIDAY_PLANS_INCLUDE_MINOR`.
- Introduction handling that links known visible/recent people instead of renaming the current speaker.
- README startup flag documentation.
- Expressive TTS voice: `tts.speak()` derives ElevenLabs `voice_settings` from the turn's emotion frame (`emotion_orchestrator.voice_settings_for_emotion`) whenever the caller passes no explicit override, so normal lines carry an expressive baseline instead of the voice clone's flat (style≈0) defaults. Tunables: `config.TTS_VOICE_SETTINGS_BASELINE`, `TTS_VOICE_SETTINGS_BY_STYLE` (keyed by emotion `voice_style`), and the `TTS_EXPRESSIVE_VOICE_ENABLED` kill switch. `TTS_MODEL_ID` is `eleven_multilingual_v2` because it honors `style` strongly (turbo_v2 applies it only weakly). The resolved voice settings and `model_id` are part of the TTS cache key, and `is_cached()`/`ensure_cached()` take an `emotion` arg so ack/wake prefill keys match live playback. Do not regress `speak()` back to sending `voice_settings=None` on normal turns, and keep empathy/grief overrides taking precedence.
- Streaming answer → TTS: on audio turns the conversational answer streams sentence-by-sentence (`interaction._stream_and_speak_sentences`) — the first sentence is spoken as soon as the LLM produces it; the rest queue behind it through the single one-at-a-time speech queue, so Rex never overlaps himself. Per-sentence safety governance is preserved (`social_frame.govern_stream_sentence`, `comedy_modes.polish_stream_sentence`), the one-question cap holds across the stream, and a fallback speaks a whole-reply govern result if every sentence is filtered out. Flagged by `config.LLM_STREAMING_TTS_ENABLED` (default on); bypassed in no-audio mode. `speech_queue.enqueue`/`tts.speak` take `log_text=False` so a streamed turn is logged once, not per sentence.
- WorldState lost-update race fix: `world_state.mutate(field, fn)` runs the whole read-modify-write under the lock. Every `people` writer (face recognition, pose, expression loops, the identity binders, face-mood) goes through it instead of `get()`+`update()`, so concurrent threads can't silently revert each other (the cause of `person_db_id` flicker / misattribution). Slow work (face-DB lookups) stays outside the lock. Use `mutate()` for any new people / self_state writers.
- OpenAI connection warmup at startup: `llm.warmup()` + `action_router.warmup()` (separate clients) run in a background thread, gated by `config.OPENAI_WARMUP_ON_STARTUP`, so the first turn doesn't pay cold TLS/HTTP setup.
- Stale-event-cancel guard: `memory.events.looks_like_cancellation` now requires a cancellation phrase AND the absence of a false-positive idiom ("not going to lie", "not doing too bad", "on my way"), so a conversational outcome reply can't durably mark a remembered event canceled. This gate protects both the follow-up handler and `_cancel_stale_event_memory`.
- Identity sub-0.75 floors require the top voice candidate to BE the attributed person: the single-visible-continuity and multi-visible-recent floors now check `raw_best_id == person` (the engaged-visible and grief floors already did), so a second speaker in a one-on-one frame is treated as off-camera-unknown instead of pinned on the engaged person.
- Bug fixes to preserve: `SCENE_MUSIC_BAND_ENERGY_MIN = 2e-6` (was `2-6`, i.e. −4, which made every band count as active → music always "detected"); dead `GUI_SHOW_FPS` removed; `social_frame` optional-lookup `except` handlers now log at debug instead of swallowing silently.
- Event follow-up resolution (fixes the "Rex obsessively re-asks how the concert went" loop): a reply that an event never happened ("I never went / didn't go" — `interaction._followup_event_did_not_happen`) resolves a pending follow-up instead of being held open as a repair, and an unanswered follow-up is dropped after `config.FOLLOWUP_MAX_HELD_OPEN_TURNS` (default 1) so it stops being re-injected into the agenda as Rex's "unresolved question". The follow-up handler lives in `interaction.py` (search `_awaiting_followup_event`); the proactive ask comes from `consciousness` Priority 2.5 (`get_pending_followups`).
- The "one sec" fillers (slow-path ack + latency filler) are disabled by default and `SILENCE_TIMEOUT_SECS` is 0.6 — see the Latency And Telemetry section. Do not re-enable the fillers without a reason.
- The local `assets/memory/people.db` is disposable dev/test data — reset, clean, or wipe it freely (no backups needed) when iterating; see the Memory Model section.
- Upstream work merged onto `main` alongside this voice/latency/identity effort (authored separately, around commit `ffa068e`): special per-person greetings / identity intros in `intelligence/person_specials.py`, delayed last-name prompts, sleep wake-word fallback, turn-completion for embedded answer clauses, and head-LED speech-stop stabilization.
- Friendlier, profile-building conversation (intentional — do not regress): R3X now asks about hobbies / interests / music / preferences with pointed follow-ups (not just plans) and adapts to each person's worldview, while keeping his roast-first voice. Most machinery already existed but was gated off; the key levers:
  - `config.REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED = True` (was absent → False) makes `conversation_agenda` weave a `QUESTION_POOL` question into normal reactive turns instead of falling through to "just respond, don't pivot". Plus `TIER_MAX_DEPTH` acquaintance → 2 (unlocks hobbies/obsession) and `LOW_MEMORY_PROFILE_MAX_FACTS` → 12 (keeps asking); the idle question prefix is warmed.
  - A per-person rule in `llm.assemble_system_prompt` (the `if person_id is not None:` block) instructs Rex to be warm/curious, ask pointed follow-ups about people's interests, and **adapt to their job/worldview/interests** (engage a scientist vs a person of faith differently) — explicitly WITHOUT softening his roast-first voice. Per-person roast control (`social_frame._roast_level`: tier + boundaries + preference facts) is untouched.
  - `llm.extract_facts` now captures religious/scientific/values cues as a `worldview` fact; `memory/facts.py` makes `belief`/`worldview` high-importance and `worldview` permanent so it ranks into the prompt and actually drives the adaptation. (Hobbies/music/interests were already extracted to `person_interests`/`person_preferences`.)
  - Proactive small-talk (`consciousness._do_small_talk_question`) now leads with curiosity about the person (hobbies, music, what they're into) instead of defaulting to their schedule.
  - Cold opens stay casual (do not regress): the first-sight/startup greeting must NOT lead with a profile question ("What kind of music are you into?") — that's an awkward intake-form opener. `config.STARTUP_PROFILE_QUESTION_ENABLED = False` gates `consciousness._pick_startup_profile_question` so the opener falls through to the mood check-in or the `FIRST_GREETING_STEERING_PHRASES` greeting ("What are you up to today?" / "how are you?"). Profile-building still happens once the conversation is rolling (reactive friendship questions) and during lulls (`LOW_MEMORY_IDLE_QUESTION_ENABLED`). The downstream profile-question machinery is untouched — flip the flag True to restore the old behavior.
- "Muzzle" means stop/decline the music (do not regress): Rex's music-offer line is "...play some X, or keep the jukebox muzzled?", so a reply like "let's keep it muzzled for now" must read as a decline. `interaction._DECLINE_PAT` now matches `muzzle(d)`, `keep it off/down/quiet/muzzled`, `no music`, `hold off`, `let's not`, etc., and `_classify_consent` checks decline before affirm — otherwise the leading "let's" was matching `_AFFIRM_PAT` and Rex played music ("Yes detected"). `command_parser` also maps "muzzle"/"muzzle the music/jukebox/song" → `dj_stop` for direct commands.
- Wake-word barge-in during DJ playback (do not regress): a wake word while music plays now STOPS the track and listens, instead of being swallowed. Previously `_dj_suppresses_conversation()` made the IDLE/ACTIVE wake handlers log "wake ack suppressed during DJ playback" and `continue`, so the user had no voice way to stop the music (only killing the process). `interaction._stop_dj_for_wake()` stops the DJ and clears the post-playback suppression tail; both wake handlers call it before going ACTIVE / acking. Note: VAD can't hear over playback because `_chunk_for_vad` attenuates the mic by `AEC_SUPPRESSION_FACTOR` (0.05) whenever DJ `_playing` — so the wake word, not VAD, is the intended interrupt during music.
- Wake word over music: loud playback masks the phrase, so `audio.wake_word._threshold(model, dj_playing=True)` drops the bar by `config.WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA` (0.15 → 0.35 effective) floored at `WAKE_WORD_MIN_THRESHOLD` (0.30). Tune the delta down toward 0.0 if music ever false-triggers a wake. The detection loop checks `dj.is_playing()` once per chunk (lazy import, no cycle).
- Post-question handoff stickiness (fixes "I answer right after Rex's question but he doesn't hear me"): a streamed reply fires `_apply_post_tts_handoff` per sentence AND once for the whole reply. If Rex asked a question but his trailing sentence is a statement, that last handoff used to flip back to the long flush window (`POST_SPEECH_LISTEN_DELAY_SECS` 0.35 + buffer flush), deleting the human's immediate answer. Now any question handoff sets `interaction._last_fast_handoff_at`, and a non-question handoff within `config.POST_QUESTION_HANDOFF_STICKY_SECS` (1.5s) is kept responsive (short delay, no flush). NOTE the deeper half-duplex limit remains: `VAD_BARGE_IN_ENABLED=False` + AEC mic suppression mean answers spoken *while Rex is still talking* are still lost — the wake word is the only mid-speech interrupt. Tests reset the module global in `setUp` (it's wall-clock-keyed; a mocked `time.monotonic` otherwise leaks stickiness across tests).
- Crosstalk suppression (best-effort, do not over-tighten): with an always-on mic Rex treats the user talking to a partner / another room as a turn aimed at him. `interaction._looks_like_third_party_crosstalk` is a HIGH-PRECISION/low-recall gate — it suppresses only the clearest third-party lines (partner endearments `babe`/comma-set-off `honey`/`sweetheart`/…, and "love you (too)" at a clause edge) and NEVER when a Rex token (`rex`/`r3x`/`dj`/`droid`/`robot`/`jukebox`) is present, so it can't make Rex ignore real input. Gated by `config.CROSSTALK_SUPPRESSION_ENABLED`; on a false IDLE activation Rex drops straight back to IDLE. It deliberately won't catch ambiguous overheard lines — fuller addressee/social-turn triage (gaze, voice-vs-visible, an addressee classifier) is still future work, and body-pose "facing" cues are unavailable on the current mediapipe build.
- No-response quip rhetorical guard: `_question_sentence_expects_response` now returns False for rhetorical "who doesn't / wouldn't / hasn't … ?" forms, so Rex's own flourishes ("who doesn't appreciate a droid that can sing?") no longer arm the "Guess that landed in the cargo bay" no-response quip (it misfired twice in live logs).

## Likely Future Work

- Decide whether the streaming answer path is sufficient latency cover on its own, or whether to re-enable (and tune) the slow-path ack / latency filler for the slowest paths.
- Add directional audio support for stereo ReSpeaker Lite input.
- Improve group turn triage for crosstalk and ambiguous addressees.
- Continue reducing OpenAI calls on common conversational paths.
- Expand tests around identity introduction, GUI text mode, no-audio mode, and multi-speaker ambiguity.
