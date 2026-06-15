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
  tell_me_about.py       "Tell me about someone" pre-briefing parsing/lines/classifier.
  motion_controller.py   High-level drive-base API: turn/move/come/stop + heartbeat + safety gates.

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
  motion.py              ESP32 motion-base serial transport (handshake, telemetry, send).

firmware/
  djr3x_motion/          ESP32 motion-controller firmware (Arduino sketch, FreeRTOS).
  tools/                 Host-side serial tools (motion protocol smoke test).
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

### Pre-briefing people who aren't here ("tell me about someone")

"I'd like to tell you about my coworker Daniel" opens a short structured flow
(`intelligence/tell_me_about.py` + `interaction._handle_tell_about_turn`):

1. Detect the briefing intent (many phrasings: "let me tell you about…", "we've
   got some tea on…", "fill you in on…", "you should know about…"). Live
   introductions ("I'd like you to meet…") and non-person topics ("my weekend")
   are explicitly excluded.
2. If no name was given, ask for it first (reuses the introduction name parser).
3. Ask "juicy gossip or boring facts?" — skipped when the opener already labeled
   it ("got some tea on Karen"). The label becomes the default classification.
4. Invite details ("spill the beans on X"); if the teller stalls, ask pointed
   questions (gender → what to remember → how they know them → what X does).
   Every detail ack carries a continuation cue ("Logged. Anything else about
   X?") — a bare "Noted." reads as Rex going silent (live-tested).
   If another Rex behavior barges in mid-briefing (smile reaction, greeting,
   idle banter), the flow re-anchors right behind it: "Still telling me about
   X, or have you moved on?" — yes resumes, no closes with "X's details
   logged to my memory banks. I will use them wisely.", a detail-bearing
   reply just keeps collecting, anything else releases the turn. Hook:
   `consciousness.note_rex_utterance` → `interaction.tell_about_on_external_rex_line`
   (the flow's own lines pass source="tell_about" and are exempt).
5. Each volunteered detail is classified (small LLM call, heuristic fallback —
   `TELL_ABOUT_CLASSIFY_LLM_ENABLED`) as gossip/fact with a kindness score
   (-1 mean … +1 kind) and stored as a `person_facts` row on the SUBJECT with
   `source='secondhand'`, `fact_kind`, `kindness`, and `told_by` (teller id).
6. Exits: "that's it / never mind" closes the flow, as do the explicit exits
   ("exit gossip mode", "stop gossiping", "enough about Joe", "change of
   subject", bare "stop" — `tell_me_about.is_exit`). The invite line tells the
   teller the exit phrase. Additionally, a turn shaped like a request TO Rex
   ("can you give me a recipe...", "what's the weather", "play some music") is
   treated as a subject pivot: the flow closes and the turn is RELEASED to
   normal routing instead of being filed (`looks_like_request_to_rex` — any
   third-person pronoun or the subject's name marks it as a detail instead,
   so "can you believe he..." still files). Flow turns are consumed before
   routers, so volunteered details never become facts about the TELLER.

The subject's person row is created up front (`find_or_create_person`) with a
relationship edge from the teller, so when the person later shows up and is
introduced or says their name, the normal name-matching reuses the pre-filled
row and memory injection surfaces the dossier. `format_fact_for_prompt` hedges
secondhand material; mean gossip (kindness ≤ -0.25) is marked NEVER-repeat —
background context only. Secondhand facts never overwrite the person's own
explicit statements (`_SOURCE_RANK`). The introduction welcome also adds one
"so you're the famous X" beat via `interaction._told_about_teller_name` when
the person was pre-briefed and has never visited.

Stability hardening (live-logged failures from 2026-06-10):

- **Proactive speech is fully suppressed while a briefing is open**:
  `speech_engine.can_proactive_speak` consults
  `interaction.tell_about_flow_active()` (before the salient bypasses, so even
  animal arrivals wait), and the interaction loop skips its three idle-filler
  paths while the flow is fresh. Idle banter had won a governor cycle
  mid-briefing and derailed the collection; the re-anchor hook remains as the
  backstop, not the norm.
- **30s spoken inactivity timeout** (`TELL_ABOUT_INACTIVITY_TIMEOUT_SECS`):
  with idle filler suppressed, nothing else breaks a stalled briefing's
  silence — so `interaction._maybe_tell_about_timeout` (called each loop pass)
  closes the flow OUT LOUD ("X's details logged to my memory banks") after 30s
  with no teller input, holding off while Rex himself is speaking. The silent
  240s step TTL remains as the deep fallback.
- **A statement ABOUT Rex is not a pivot**: `looks_like_request_to_rex` only
  releases the turn when the Rex-vocative is followed by request-shaped text —
  "Rex is a very close friend of mine" stays in the file as a detail (it
  previously closed the flow with zero details filed).

Config: `TELL_ABOUT_ENABLED`, `TELL_ABOUT_STEP_TTL_SECS`,
`TELL_ABOUT_INACTIVITY_TIMEOUT_SECS`, `TELL_ABOUT_CLASSIFY_LLM_ENABLED`.
Tests: `tests/test_tell_me_about.py`.

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

## Motion System (drive base)

An optional ESP32-controlled differential-drive base lets Rex physically move around a
room on spoken command while avoiding obstacles with Time-of-Flight sensors. Two brains:
the **ESP32 owns the reflexes** (real-time motor PID, ToF safety stop, heartbeat
watchdog — it can stop the base without the Mac) and the **Mac owns the intent**
(speech → high-level command over USB serial). Full design: `docs/motion_system.md`.

- **Wire contract:** `docs/motion_protocol.md` (v1, locked) — handshake, 20 Hz telemetry,
  commands (`drive`/`turn`/`move`/`come`/`stop`/`estop`/`clear`/`config`), `ack`/`done`/
  `event`, heartbeat watchdog (500 ms) + drive deadman (300 ms), clamping, enums. If code
  and that doc disagree, the doc wins. Sign convention is REP-103: **+linear = forward,
  +angle/+deg = LEFT/CCW** everywhere (`drive.ang` rad/s, `turn.deg` degrees, `odom.theta` rad).
- **Firmware:** `firmware/djr3x_motion/` (ESP32, Arduino/arduino-cli, FreeRTOS). Phase 0
  build runs the full protocol against a **stubbed hardware layer** (`MOTION_HW_PRESENT=0`
  in `hal.h`): a plant model synthesizes odometry and ToF reads clear, so it runs on a
  bare ESP32 with nothing wired. Flashes reliably at **115200** (default 921600 fails on
  the CH340 bridge). Bench test: `firmware/tools/motion_serial_smoketest.py`.
- **Mac transport:** `hardware/motion.py` mirrors `servos.py` — `connect()` runs the
  `hello` handshake (version-gates motion on/off), a background reader keeps a thread-safe
  telemetry/ack/done snapshot. `connected()` is the availability signal.
- **Mac controller:** `intelligence/motion_controller.py` — `turn/move/come/stop/drive` +
  voice verbs, speed-cap clamping, the heartbeat thread, and the **autonomous gate**:
  suppressed while `config.INTERACTION_PAUSED` or while a gamepad owns the base
  (`owner == "manual"` in telemetry); `stop`/`estop` always pass. Sends a `config` command
  with the Mac's caps/zones at connect. **`available()` is False when no base is
  connected, so the conversation pipeline is unchanged unless `MOTION_ESP32_PORT` is set.**
- **Routing:** `action_router` has `motion.turn/move/come/stop` action specs and
  `classify_explicit_motion` (deterministic, high-precision — no LLM). `interaction.py`
  dispatches them via `_handle_router_motion_action` and runs the classifier in the fast
  local takeover (gated on `motion_controller.available()`). Bare "stop" routes to the
  base **only while it is moving**, so it never steals stop-music/stop-game/stop-talking.
- **Config:** `MOTION_*` tunables in `config.py`; `MOTION_ESP32_PORT` in `.env` (loaded as
  `MOTION_PORT_SET`). `main.py` Step-4 connects it and logs `Motion base: enabled/disabled`
  like the other hardware; shutdown stops the heartbeat and leaves the base stopped.
- **Setup:** `setup_macos.sh` (under "physical droid" → "motion base") **auto-detects the
  ESP32 by protocol probe** — it opens each USB-serial port and looks for the firmware's
  `hello` reply. This is the only reliable discriminator because the board's USB bridge is
  a CH340, the same chip as the chest Arduino (chip-ID can't tell them apart). It can also
  install the ESP32 core + ArduinoJson and flash the firmware.
- **Tests:** `tests/test_motion.py` (fake-ESP32 serial; transport, controller gates,
  clamping, classifier).

Manual control (a Bluetooth gamepad paired directly to the ESP32) and the real
motor/encoder/ToF drivers are Phase 1 — see `docs/motion_system.md` §11, §17.

## GUI

The PySide6 dashboard is optional and launched with `--gui`.

Important GUI behavior:

- It mirrors runtime state and conversation logs.
- Its text input can submit turns through `interaction.submit_text(...)`.
- With `--gui --noaudio`, the app becomes a text-only test interface for the full conversation/router/memory pipeline.
- GUI-first startup: the window opens maximized within seconds (Qt owns the main
  thread); controller startup runs on a background thread with cooperative
  abort checkpoints (`main._abort_startup_if_shutdown`). The top bar shows
  Booting…/Connected/Startup failed, text input is gated until services are up,
  and a failed boot keeps the window open so the log panel stays readable.
- A SYSTEM LOG strip at the bottom auto-scrolls the active app log
  (`utils.logging.install_gui_log_handler` mirrors root-logger records into
  `gui_bridge.add_log_line`; `GUI_LOG_PANEL_MAX_LINES`). With `DEBUG_MODE=True`
  the active file is the per-run `logs/djr3x-<stamp>.log`, not `djr3x.log`.
  Lines are colored by level (WARNING amber, ERROR/CRITICAL red).
- Top-bar state badge: the runtime `State` (IDLE/QUIET/ACTIVE/SLEEP/SHUTDOWN),
  overlaid with SPEAKING from `speech_state.speaking` (`_state_badge_spec`).
- The vision panel's camera meta line is truthful: measured FPS from
  `camera.frame_info()` (EMA at capture), real resolution, and a "STALE Xs"
  indicator + dimmed frame on dropout — no more hardcoded "30 FPS".

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

- Expressive TTS voice: `tts.speak()` derives ElevenLabs `voice_settings` from the turn's emotion frame (`emotion_orchestrator.voice_settings_for_emotion`) when the caller passes no override; `TTS_MODEL_ID=eleven_multilingual_v2` (honors `style`); voice settings + model_id are in the TTS cache key and `is_cached()/ensure_cached()` take `emotion`. Don't send `voice_settings=None` on normal turns; empathy/grief overrides win. Knobs: `TTS_VOICE_SETTINGS_*`, `TTS_EXPRESSIVE_VOICE_ENABLED`.
- Streaming answer→TTS: audio turns stream sentence-by-sentence (`interaction._stream_and_speak_sentences`) — first sentence speaks ASAP, the rest queue through the single one-at-a-time speech queue (no overlap).
- WorldState lost-update fix: `world_state.mutate(field, fn)` does the read-modify-write under the lock; every `people` writer uses it (not `get()`+`update()`).
- OpenAI warmup: `llm.warmup()` + `action_router.warmup()` run in a background thread (`OPENAI_WARMUP_ON_STARTUP`) so the first turn skips cold TLS/HTTP.
- Stale-event-cancel guard: `memory.events.looks_like_cancellation` requires a cancellation phrase AND no false-positive idiom ("not going to lie", etc.).
- Identity sub-0.75 floors require `raw_best_id == person` (the continuity/recent floors), so a 2nd speaker in a 1:1 frame is treated as off-camera.
- Bug fixes to keep: `SCENE_MUSIC_BAND_ENERGY_MIN=2e-6` (was a typo making music always "detected"); dead `GUI_SHOW_FPS` removed; `social_frame` optional-lookup excepts log at debug.
- Event follow-up resolution: a reply that an event never happened (`interaction._followup_event_did_not_happen`) resolves a pending follow-up instead of re-asking (kills the "how was the concert?" loop).
- The "one sec" fillers (slow-path ack + latency filler) are disabled by default and `SILENCE_TIMEOUT_SECS=0.6` — see Latency And Telemetry. Don't re-enable without reason.
- The local `assets/memory/people.db` is disposable dev/test data — wipe freely (see Memory Model).
- Upstream merge (~`ffa068e`, authored separately): per-person greetings/intros (`intelligence/person_specials.py`), delayed last-name prompts, sleep wake-word fallback, turn-completion for embedded answers.
- Comedy-forward balance (current intent): Rex leads with a comedic/curious beat, not an every-turn polite interview that opens "Ah," and ends with a profile question. Keep the profile-building machinery, not the interview cadence.
- Friendlier profile-building conversation (machinery, rebalanced behind comedy-forward): asks about hobbies/interests/music/preferences with pointed follow-ups and adapts per person.
- "Muzzle" = decline the music (`interaction._DECLINE_PAT` matches `muzzle(d)`, `keep it off/down/quiet`, `no music`), since the offer line is "…or keep the jukebox muzzled?".
- Wake-word barge-in during DJ playback STOPS the track and listens (was swallowed by `_dj_suppresses_conversation()`).
- Wake over music: `audio.wake_word._threshold(dj_playing=True)` raises the bar by `WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA`, floored at `WAKE_WORD_MIN_THRESHOLD`.
- Post-question handoff stickiness: a question turn keeps the response-wait open even if its trailing sentence is a statement (`interaction._apply_post_tts_handoff`, per-sentence + once per reply).
- Crosstalk suppression: `interaction._looks_like_third_party_crosstalk` is HIGH-precision/low-recall — suppresses only the clearest third-party lines; don't over-tighten.
- No-response-quip rhetorical guard: `_question_sentence_expects_response` returns False for rhetorical "who doesn't/wouldn't…?" forms, so Rex's flourishes don't arm the quip.
- Silence drives conversation, doesn't end it: quiet (no goodbye) → re-engage, not sign off (`end_thread._CLOSURE_PAT` no longer matches bare politeness like "thank you").
- Face-tracking responsiveness: the head-pose loop (`consciousness._step_face_tracking`, ~12.5 Hz) is a closed loop (camera on head); `FACE_TRACKING_CENTERING_GAIN` is matched to the FOV.
- Supervisor wake detection is ONNX-only (`wakeuprex.onnx` openWakeWord on 80 ms mono frames, threshold `REX_SUPERVISOR_WAKE_THRESHOLD`); do NOT regress to VAD/Whisper/RMS.
- Always-on wake-word supervisor + single-instance lock (`docs/supervisor.md`, `rex_supervisor.py` LaunchAgent): listens the whole login session, launches `main.py` on "wake up Rex"; an flock keeps main.py single-instance so the supervisor stays dormant while a controller is alive.
- Same-day repeat-visit banter: a repeat summon within one local day opens the startup greeting with a short repeat-visit roast, then normal conversation (`memory.people`).
- Listening motion: VAD onset (`interaction._begin_user_turn`) → `servos.start_listening_motion()` for the transcription→LLM→TTS wait so Rex isn't frozen while thinking.
- Eye blink fix: the head Arduino blinks only while `eyesActive==true` (cleared by `SPEAK_STOP`/`OFF`); the speech path re-asserts it so Rex keeps blinking after his first line.
- Randomized two-axis room scan: the startup face-search/speaker-gaze no longer cycles a fixed single-axis pose list every boot.
- Startup dead-space fix: the boot line + a head scan kick off BEFORE the heavy preloads (`main._run_controller_startup`) instead of after, so the head isn't frozen+silent during load.
- Whisper repetition-filter false-positive fix: `audio.transcription._is_hallucination` no longer discards natural repetition ("I like Beethoven, I like Bach…").
- Software AEC (`audio/aec.py`) is DISABLED by default (only ~5 dB in-room; ineffective). Hardware AEC (ReSpeaker Lite XU316 on-chip) is the real fix for hearing a wake word over Rex's own speech.
- Wake-word multi-fire + head-thrash fixes: debounce repeated openWakeWord frames from one phrase; calmer head motion.
- Startup vision false-negative → duplicate chime → starved greeting: one root cause fixed (don't regress).
- Empty/blank transcript drops back to IDLE instead of camping in ACTIVE (a false VAD trigger / fully-filtered audio).
- Self-speech suppression no longer eats the front of replies to STATEMENTS (hardware/room-sensitive — validate on the robot).
- Proactive-speech "yield the floor" guard: pre-cache the line's audio, re-check the mic right before playback, and bail if the user already started (`speech_engine.speak_async` + `audio.barge_guard`).
- Curious-conversationalist behavior: Rex shows genuine interest in what the user shared before teasing; don't kill a topic with a reflexive joke (`tests/test_conversation_revamp.py`).
- Turn-taking + routing + dialogue revamp (round 2): don't interrupt unfinished thoughts, don't misroute/over-roast (`tests/test_conversation_revamp.py`).
- Repeat / hallucination / truncation / joke-replay fixes (round 3): `tests/test_conversation_revamp.py` + `test_performance_output.py`.
- Subject pivot: steering can CHANGE the channel when a topic isn't landing, not only deepen it (`intelligence/conversation_steering.py`; `tests/test_conversation_revamp.py::SubjectPivotTest`).
- Conversation arc memory (Bet 1): a cheap-LLM running summary of the live conversation + callbacks fed into the system prompt (`intelligence/topic_thread.py`; `tests/test_conversation_arc.py`).
- TurnPlan (Bet 2): typed `conversation_agenda`→`social_frame` handoff replaces prose-directive regex re-parsing (`tests/test_turn_plan.py`).
- Relationship-tone tracking: warmth/edge tracks the RELATIONSHIP, not per-turn (`llm._relationship_tone_rule` over `warmth/antagonism/trust_score`; `tests/test_relationship_tone.py`).
- Offline conversational-quality replay harness (no robot): replays scenarios through the deterministic stack (`tests/test_conversation_replay.py` + `tests/fixtures/conversation_replays.json`).
- Cold-open ranker: the startup celebration cold-open RANKS gate-passing candidates instead of taking the first (`consciousness._pick_due_celebration_checkin`; `tests/test_celebration_ranker.py`).
- Turn classifier (Bet 3): SHELVED/dormant — do NOT regress to on-path (`intelligence/turn_classifier.py`).
- Rex persistent POV: carries ONE current preoccupation and leads with substance instead of react→roast→question (`intelligence/rex_pov.py`; `tests/test_rex_pov.py`).
- Roast rebalance: curious-first, not roast-first (`config REX_CORE_PROMPT`/`PERSONALITY_DEFAULTS` + the live personality DB).
- Memory-followup cadence clamp: stop the proactive event interrogation (`interaction._post_response`/`_memory_followup_cadence_allows`, `FOLLOWUP_*`; `tests/test_followup_resolution.py`).
- Ellipsis trail-off cut-off + idle-banter volunteers the POV + no invented props (`interaction._tail_is_speakable`/`_maybe_idle_banter`, `social_frame` visual rule; `tests/test_streaming_tts.py`).
- Idle-POV invented-prop guardrail: the no-invented-prop rule travels WITH the POV directive (`rex_pov._DIRECTIVE_TEMPLATE`, `REX_POV_SEEDS`; `tests/test_rex_pov.py`).
- LLM-in-the-loop conversation-quality eval (`evals/run_quality_eval.py`, `evals/checkers.py`, `evals/quality_corpus.json`, `evals/README.md`): the offline gate that replaced "run robot → read one log → patch one line".
- Boundary handling: don't roast a boundary/withdrawal (`social_frame._BOUNDARY_RE`/`_roast_level`, `conversation_agenda` boundary branch; `tests/test_comedy_modes.py`).
- Judge calibration + cantina-origin fix + eval fidelity (`evals/checkers._SINCERE_SYS`, `evals/judge_cases.json`, `--check-judges`).
- Cantina-bleed fixed at the root — comedy mode + session summary (`comedy_modes`, `config.COMEDY_LINE_BANKS`, `llm.generate_session_summary`; `tests/test_comedy_modes.py`).
- Anti-repetition-hack deletion: ONLY the angle rotation was arc-redundant; `social_frame._is_near_repeat` + `comedy_modes.strip_banned_opener` are KEPT (don't re-delete).
- TurnPlan regex patterns are LOAD-BEARING (don't delete without completing Bet 2): `social_frame._purpose_from`/`_ASK_ALLOWED_PAT`/`_HARD_NO_QUESTION_PAT`/`_EXPLICIT_FOLLOWUP_PAT`.
- Live facial expression in the REPLY prompt (`llm._live_expression_prompt_line` + §4b in `assemble_system_prompt`, `LIVE_EXPRESSION_IN_REPLY_ENABLED`; `tests/test_facial_expression_reactions.py`).
- Truncated-tail cut-off via the streaming safety-net fallback (`interaction._complete_sentence_prefix`; `tests/test_streaming_tts.py`).
- Celebration check-in re-lead — cross-process cooldown so it doesn't re-lead every startup (`memory.emotional_events.get_startup_celebrations`, `PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS`).
- Boundary → mute matching event check-ins (`memory.boundaries.apply_detected_boundary` + `_boundary_mutes_events`, `BOUNDARY_MUTES_MATCHING_EVENTS`; `tests/test_boundary_event_mute.py`).
- Proactive-layer consolidation: the action governor is the SINGLE proactive-speech decider under `ACTION_GOVERNOR_ENFORCE` — mechanisms submit a `CandidateMove` with a deferred `speak_fn` and only the winner speaks (`intelligence/action_governor.py`, `speech_engine`/`consciousness` governor cycle; grace + question-budget gates surface as candidate metadata → rejection reasons).
- Eval-loop tightening + QoL: `--gate` CI regression guard on the quality eval, fixed `over_questioning` checker, startup log-clear gated under the test runner, broadened `_BOUNDARY_RE`.
- Rex-POV cross-session persistence (`rex_pov.snapshot_state`/`restore_state`/`persist`/`load_persisted`, `REX_POV_PERSIST_ENABLED`/`REX_POV_STATE_PATH`; gated so the suite writes no real state file — the persist test asserts a no-op, tolerating a real-run artifact).
- Cold-open ranker extended across facts/interests (`consciousness._cold_open_*`, `COLD_OPEN_INTEREST_RANK_ENABLED`).
- Birthday recognition (works; test-covered): `consciousness._pick_birthday_window`/`_build_birthday_prompt` + Priority-1 greeting tier, `awareness/holidays.days_until_birthday`, `BIRTHDAY_REMINDER_WINDOW_DAYS`/`BIRTHDAY_WINS_ON_DAY`; `tests/test_birthday_greeting.py`. The birthday is owned by the structured `birthday` (MM-DD) fact + the window path ONLY — free-text "today is X's birthday" statements are NOT durable facts (see ephemeral-fact filter below); they used to be injected forever and made Rex wish happy birthday for days after.
- Ephemeral-fact filter (`memory.facts._is_ephemeral_statement` + `get_prompt_worthy_facts`): person_facts whose value pins to a relative day ("today/tomorrow/yesterday/tonight/last night") are dropped from prompt injection — a one-day-true line ("today is the speaker's birthday") must not be recited as a standing trait. `get_facts` is unfiltered so the birthday-window path still reads the structured fact. `tests/test_greeting_repeat_fixes.py`.
- Visit-milestone greeting fires ONCE, not every boot (`people.last_milestone_greeted` column + `record_milestone_greeted`; `consciousness._pick_milestone` guard + `milestone_to_mark` commit). visit_count only advances after a real conversation (`interaction._end_session` skips empty transcripts), so a person parked at visit 4 used to hear "your 5th visit" on every startup. `VISIT_MILESTONES=[5,10,25,50,100]`. `tests/test_greeting_repeat_fixes.py`.
- Episodic memory rex.db — PHASE 1 (capture only; nothing reads it back yet): `memory/rex_db.py` (2nd connection layer, path read at call time) + `memory/episodes.py`, gated by `EPISODIC_MEMORY_ENABLED` + a test-runner suppression so the SUITE NEVER creates/writes a real rex.db (reads gated too). Capture hooks + an LLM shutdown session-summary; schema in `setup_assets.py`; `tests/test_episodes.py`.
- Episodic batch-2 capture kinds: enrollment/visit-departure/celebrity/emotional-checkin/celebration/boundary/games + memorable greeting tiers (birthday/milestone/celebration/reunion). Proactive-speech captures are SPOKE-GATED (only when Rex actually spoke); real-world events fire at the event. `memory/episodes.py` + `episodic_hooks` + `interaction`/`boundaries`/`games` hooks.
- Mood-driven body language (`intelligence/body_mood.py`, pure state): a decaying "body mood" set by compliments/insults/amusement shapes posture — head lift/tilt bias on the RESTING pose (never fights the face-centering controller), visor openness, breathing cadence, occasional idle gesture. Visor is hard-clamped to the lens-clear floor (6400) so a mood can't blind the camera. `consciousness._step_mood_expression`/`_mood_rest_bias`; `tests/test_body_mood.py`.
- Calmer head during speech + at the servo rails (`consciousness._step_face_tracking`/`_neck_saturated_at_rail`, `FACE_TRACKING_SPEECH_*`/`FACE_TRACKING_RAIL_DAMP_*`): soften centering while speaking; hold the neck instead of jittering when it's pinned at a limit (`tests/test_face_tracking.py`).
- Compliment detection coverage (`config.COMPLIMENT_KEYWORDS/PHRASES`): broadened so everyday compliments ("nice robot", "good boy", "you're sweet/cool") fire the layer-1 proud beat BEFORE the reply (when the arm servos are free). Phrases, not bare words, to avoid false positives.
- Idle "mind of his own" head wander (`consciousness._idle_wander`/`_step_idle_head_wander`/`_drive_idle_head_wander`, `IDLE_HEAD_WANDER_*`; `tests/test_idle_head_wander.py`): when the conversation lulls with a face locked, look around the room then return gaze and maybe re-greet. The face-loop drives it ABOVE the frame/listening early-returns (self-aborts on speech/listening/resumed talk); a 1Hz backstop ends any stalled wander — `active` can never get stuck.
- Bored environmental snark (`intelligence/idle_behaviors.do_bored_environment_snark`, `BORED_ENV_SNARK_*`; `tests/test_bored_env_snark.py`): an idle riff on the ROOM via `vision.scene.describe_scene_detailed` — complaint / faux-clueless object question / clutter jab / art opinion / take-me-somewhere — grounded in real objects (never invents props), hard-cooldowned.
- Tidy-up — episodic capture hooks → `intelligence/episodic_hooks.py` (leaf module; consciousness calls `episodic_hooks.<name>`).
- Tidy-up — idle micro-behaviours → `intelligence/idle_behaviors.py` (dispatcher stays in consciousness and calls `idle_behaviors.do_<name>`; the behaviours reach consciousness's speak engine via a lazy `_c` proxy; `_do_small_talk_question` stayed, being mood-detection-coupled).
- Tidy-up — proactive-speech ENGINE → `intelligence/speech_engine.py` (15 functions; consciousness re-exports each as a `_name` shim so call sites + test patches are unchanged; intra-engine calls route through the `_c` shims for full patch-transparency; `note_rex_utterance` + shared speech state stayed in consciousness; `tests/test_speech_engine.py`). The governor metadata key MUST stay `"can_proactive_speak"` (action_governor reads it).
- GUI-first startup (`main._run_gui_mode`): dashboard shows immediately (maximized), controller startup runs on the `controller-startup` thread with `_StartupAborted` checkpoints so window-close/Ctrl+C stops the boot at the next phase boundary; a second Ctrl+C during teardown is absorbed (SIGINT re-pointed in the finally) so `_shutdown()` always runs; fatal startup paths keep the window open on "failed" status and exit non-zero after teardown. System-log panel mirrors the root logger via `utils.logging.install_gui_log_handler` → `gui_bridge.add_log_line`.
- "Tell me about someone" pre-briefing (`intelligence/tell_me_about.py`, `interaction._handle_tell_about_turn`/`_pending_tell_about`): pre-populates the person DB for someone who is NOT here — name → gossip-or-facts → details, stored as `secondhand` person_facts with `fact_kind`/`kindness`/`told_by` (new columns in `setup_assets.py` + `database._run_migrations`). Mean gossip is never recited to the subject (prompt hedging in `facts.format_fact_for_prompt`); secondhand never overwrites the person's own explicit facts. Escapable by design (live-tested): explicit exits (`is_exit` — "exit gossip mode"/"enough about X"/"stop") AND a subject-pivot guard (`looks_like_request_to_rex`) that releases requests aimed at Rex ("can you give me a recipe...") back to normal routing instead of filing them as gossip — the chicken-soup regression in `tests/test_tell_me_about.py`. Proactive barge-ins (smile reaction etc.) trigger a re-anchor question via the `note_rex_utterance` hook → `tell_about_on_external_rex_line`, and every ack invites more (no bare "Logged."). See the Memory Model section.

- Named vision descriptions (fold dlib identity into GPT-4o vision): `vision.face.visible_known_names()` resolves currently-visible recognized people (world_state.people `person_db_id` → name) and is woven into the GPT-4o prompts so a known person is named ("Bret is at his desk") instead of anonymized ("a man at a desk"). `vision.scene.analyze_environment(..., known_names=None)` (the GUI's visual description + LLM world-context, auto-resolves) and `analyze_directed_attention(..., known_names=None)` (the "what do you see?" path — its blanket "do not identify anyone" rule is lifted ONLY for people Rex already recognizes; all other identity/age/health guessing stays banned). The naming directive rides in the directed-attention prompt BODY (not just the safety footer) so `target_summary` itself is named. No extra vision spend — same image, slightly longer text prompt. `intelligence/episodic_hooks._known_visible_names` now delegates to the same resolver. The GUI's visual-description panel reads `world_state.environment["description"]`, which previously only refreshed on the slow periodic scan — so a "what do you see?" query looked frozen; `interaction._update_scene_description` now writes the fresh directed-look summary back into that field so the panel updates on demand. Tests: `tests/test_vision_named_people.py`.
- Callback humor (design: `docs/callback_humor_design.md`): Rex banks durable, light, SELF-volunteered "fun facts" per person (`person_callback_material` in people.db via `memory/callbacks.py`; banker = local qwen labelled-lines + heuristic fallback in `intelligence/callback_engine.py`, run from `_post_response`'s background thread) and resurfaces ONE later — reactively when the background relevance judge connects a stored premise to the live topic (claim seam in `interaction._stream_llm_response` between `select_mode` and the directive join; the claim rides as the `callback_banked` comedy mode and `llm._build_person_context`'s hook chain stands down via `callback_engine.turn_claim_active`), or in a mid-conversation lull (`consciousness._step_lull_callback`, governor purpose `lull_callback` priority 58). Sensitivity is classified at CAPTURE with a deterministic protected-category wall (health/grief/body/orientation/finances/religion-politics/family-conflict/addiction-legal) the model can only move material TOWARD 'excluded' on, never toward 'safe'; only `sensitivity='safe'` rows can ever fire and `active_pool` hard-filters. Spend-at-SPEAK (settle echo-check on the reply path, `on_spoke` on the lull path), per-premise reuse cooldown + use-count decay, per-session no-repeat + volume ledger (cleared for real in `_end_session`), 30-min sober-room window after any heavy-sensitivity turn (`note_heavy_moment` from the sensitive prepass), boundary→retire hook in `boundaries.apply_detected_boundary`, forget-flow deletion in `forgetting.py`, crowd/tier/`callback_style`-restraint gates. Flags: `CALLBACK_BANK_ENABLED` / `CALLBACK_HUMOR_ENABLED` (env-overridable A/B pair) + `CALLBACK_*` tunables. Tests: `tests/test_callback_humor.py`.

- Motion system (drive base): wire contract `docs/motion_protocol.md` (v1, locked) + Phase 0 ESP32 firmware (`firmware/djr3x_motion`, full protocol over a stubbed HAL; flashed + 27/27 smoke test) + Mac side (`hardware/motion.py` transport, `intelligence/motion_controller.py` controller, `action_router` motion.* specs + `classify_explicit_motion`, `interaction` dispatch/fast-path, `MOTION_*` config, `MOTION_ESP32_PORT`, `main.py` Step-4 wiring). Gated on `motion_controller.available()` so it's a NO-OP for the whole pipeline unless a base is connected; `stop`/`estop` always pass and bare "stop" only routes to the base while moving. Flash at 115200 (921600 fails on the CH340 bridge); `setup_macos.sh` auto-detects the ESP32 by protocol probe (chip-ID can't — it shares a CH340 with the chest Arduino). Sign convention REP-103. Tests: `tests/test_motion.py`. See the Motion System section above.

## Likely Future Work

- Motion Phase 1: wire the real drive base (BTS7960 motor driver + Hall encoders + per-wheel PID + 5× VL53L0X ToF) and fill the `hal.cpp` `MOTION_HW_PRESENT` driver sections; add the Bluetooth-gamepad manual override (`docs/motion_system.md` §11, §17). Known Phase-1 fidelity gaps: a pure `turn` (spin) is not yet ToF-gated (no side sensors), and the stub plant carries residual velocity from a finished finite command into the next one.
- Decide whether the streaming answer path is sufficient latency cover on its own, or whether to re-enable (and tune) the slow-path ack / latency filler for the slowest paths.
- Deeper conversation steering: detect a topic shift semantically (not just explicit "I like / I'm building X") and update/expire the active interest accordingly; today a new subject the user is clearly engaged in but doesn't name in an interest form is not picked up.
- Add directional audio support for stereo ReSpeaker Lite input.
- Improve group turn triage for crosstalk and ambiguous addressees.
- Continue reducing OpenAI calls on common conversational paths.
- Expand tests around identity introduction, GUI text mode, no-audio mode, and multi-speaker ambiguity.
