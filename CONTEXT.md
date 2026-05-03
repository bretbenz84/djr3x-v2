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
5. Run command parser/action router/intent classifier.
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

### Output

`audio/speech_queue.py` is the central response queue. It handles priority, coalescing, playback start callbacks, and completion.

`audio/tts.py` handles ElevenLabs cache lookup/fetch/playback. In no-audio mode, `speak()` and `ensure_cached()` return before network or playback work.

## Latency And Telemetry

The project now has explicit latency instrumentation.

Key logs:

- `[latency]`: stage timings inside a turn.
- `[ttfs]`: time to first response queued and first audio/text start.
- `[character_loop]`: full per-turn summary: speaker, interpretation, execution, output, memory suppression, timing.
- `[action_router_audit]`: final action routing result.

Recent latency architecture:

- `audio.speaker_id.preload()` runs at startup when `config.SPEAKER_ID_PRELOAD_ON_STARTUP` is true, removing first-turn Resemblyzer load cost.
- Slow-path acknowledgments are short cached responses for known-slow paths: `general`, `memory`, and `vision`.
- `config.SLOW_PATH_ACK_LINES` controls acknowledgment text.
- `config.SLOW_PATH_ACK_EXPECTED_SECS` controls when a path is expected to be slow enough to acknowledge.
- Slow-path acks are different from `LATENCY_FILLER_LINES`: acks are immediate receipts; latency fillers are delayed in-character thinking lines.
- In audio mode, slow-path acks should already be cached so they never trigger an ElevenLabs round trip.
- In no-audio mode, cache checks are skipped because the output is text only.

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

Main concepts:

- People records with names, biometrics, familiarity, and relationship metadata.
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
- The action governor arbitrates proactive candidates so Rex does not stack too many remarks.

If startup greetings feel wrong, inspect face detection timing, world-state updates, and action-governor candidate selection.

## Social Conversation Layers

The project has several layers that shape final speech:

- Command parser for known local commands.
- Action router for executable intents.
- Intent classifier fallback.
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
- Slow-path acknowledgments for general, memory, and vision paths.
- Introduction handling that links known visible/recent people instead of renaming the current speaker.
- README startup flag documentation.

## Likely Future Work

- Tune slow-path acknowledgment thresholds so Rex acknowledges only when useful.
- Add directional audio support for stereo ReSpeaker Lite input.
- Improve group turn triage for crosstalk and ambiguous addressees.
- Continue reducing OpenAI calls on common conversational paths.
- Expand tests around identity introduction, GUI text mode, no-audio mode, and multi-speaker ambiguity.
