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
| `-noservos`, `--noservos`, `--no-servos` | Disable the Pololu Maestro entirely for the run, even with `MAESTRO_PORT` configured. Seeded as `DJR3X_NO_SERVOS` before the config imports (mirrors `--noaudio`'s env-seed mechanism) so `config_loader.SERVOS_ENABLED` computes False; every servo call is already a no-op when servos are disabled (`hardware/servos.py`). |
| `-local-tts`, `--local-tts` | Use the on-device Qwen3-TTS voice clone instead of ElevenLabs for the run. Seeded as `DJR3X_LOCAL_TTS` before config imports (mirrors `--noaudio`) so `config.LOCAL_TTS_MODE` computes True; `main.py` preloads the model (hard-fail if missing) and skips the ElevenLabs warmup. See "Conversation Voice / TTS backends" below. |

In no-audio mode, `main.py` sets runtime-only config values:

- `config.NO_AUDIO_MODE = True`
- `config.AUDIO_OUTPUT_SUPPRESSED = True`

It skips Whisper verification, audio stream startup, wake word listening, audio scene analysis, TTS prewarm, speaker-ID preload, startup/shutdown audio, listening chime, ElevenLabs fetches, and direct playback.

## Configuration And Secrets

Tracked configuration:

- `config.py`: tunable defaults, model names, thresholds, feature flags, servo defaults, latency settings. Source of truth for all defaults.
- `user_config.example.py`: heavily commented template of the ~45 user-facing overrides (AI models, personality dials + base prompt, location/venue, feature toggles, key timeouts), grouped by topic. Every setting shown commented-out at its current default.
- `.env.example`: host-specific config template.
- `apikeys.example.py`: API key template.

Untracked local/runtime files:

- `.env`: machine-specific camera, microphone, hardware ports, servo limit overrides.
- `apikeys.py`: OpenAI and ElevenLabs keys.
- `user_config.py`: user-facing overrides, copied from `user_config.example.py` by `setup_macos.sh`. `config.py` imports it LAST (`from user_config import *`, wrapped in try/except ImportError) so its values win over defaults; a missing file is harmless. Uncomment a line to override, re-comment to revert.
- `assets/memory/people.db`: local person database.
- `assets/audio/tts_cache/`: generated ElevenLabs cache.
- downloaded model assets.

Never commit real secrets, local databases, generated TTS cache, local music, or downloaded model files.

`config.py` ends with the `from user_config import *` override import followed by a re-derive tail that recomputes values built from a base the user may have overridden (`ACTION_ROUTER_MODEL = LLM_MODEL`, `STARTUP_BOOT_TTS_LINE`), so overriding the base propagates. Computed `Path(__file__)` state-paths and `.env`-only serial ports are deliberately NOT exposed in `user_config`. Deeper internal tuning (CV/audio thresholds, cooldowns, scoring weights, prompt fragments) stays in `config.py`.

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
  speaker_id.py          Voice embeddings + speaker matching (ECAPA-TDNN primary,
                         Resemblyzer fallback; scores mapped via voice_score.py).
  voice_score.py         Embedder-backend score mapping (ECAPA cosine -> the
                         Resemblyzer-calibrated scale all thresholds are tuned on).
  wake_word.py           OpenWakeWord loop.
  speech_queue.py        Prioritized response queue and playback/text completion.
  tts.py                 TTS backends + cache + no-audio bypass. Dispatches per
                         line to ElevenLabs (default) or the local engine
                         (_speak_local) for --local-tts mode / an impersonation
                         voice_ref / when the ElevenLabs fallback breaker is open.
  local_tts.py           On-device Qwen3-TTS voice clone (mlx-audio): model
                         lifecycle + raw streaming synthesis only. Loads offline
                         from assets/models/qwen_tts/. VoiceRef(wav, ref_text, label).
  echo_cancel.py         Playback suppression/AEC state.
  scene.py               Background audio scene analysis.

intelligence/
  interaction.py         Main turn pipeline for speech and GUI text input.
  consciousness.py       Proactive loop, greetings, presence, empty-room behavior.
  dialogue_act.py        Cheap conversational-frame gate before executable actions.
  action_router.py       LLM action routing.
  tool_router.py         Native tool-calling router. LIVE: the live-action tool
                         schemas ride the lean REPLY call, so a routed turn costs
                         zero extra LLM round-trips; a tool choice unwinds as
                         ToolCallRequested before any speech. Also holds the
                         off-by-default shadow collector.
  command_parser.py      Fast/local command recognition.
  intent_classifier.py   Intent fallback and deterministic guards.
  llm.py                 Main LLM prompt assembly and response generation.
  local_llm.py           Ollama sidecar for low-latency local calls.
  empathy.py             Affect classification and emotional event handling.
  social_frame.py        Response shape/governance cleanup.
  bit_ledger.py          Persistent per-person comedy-bit cooldown (rex.db).
                         Records SPOKEN lean impulses by topic signature; blocks a
                         re-run within BIT_LEDGER_COOLDOWN_DAYS and feeds recent
                         angles to the lean prompt as an exclusion list.
  tell_me_about.py       "Tell me about someone" pre-briefing parsing/lines/classifier.
  motion_controller.py   High-level drive-base API: turn/move/come/stop + heartbeat + safety gates.
  motion_agency.py       Autonomous motion decisions: requested COME search/align/approach,
                         FLINCH back-off when someone crowds
                         the front (front-matrix ToF intrusion, rear-ToF-limited retreat —
                         backs up only to a point, holds when cornered), turn base to face
                         the tracked person (neck-offset signal), approach a far person
                         (`come`, ToF-guarded). AUTONOMOUS_MOTION_ENABLED + per-behavior
                         flags; one maneuver/tick, idle-state only; flinch is a reflex
                         (no tracked person needed, may fire mid-sentence) while
                         realign/approach defer mid-sentence; gamepad owner always wins.
                         Explicit "come here/over here/to me" rotates in bounded 45° steps
                         until a person is tracked, aligns from the neck offset, then sends
                         an obstacle-gated approach that stops 1 m from the nearest front return.

memory/
  database.py            SQLite connection, schema, migrations.
  people.py              People, face/voice biometrics, familiarity.
  facts.py               Person facts and observations.
  events.py              Upcoming/follow-up events + session-opener continuity threads
                         (get_recent_open_threads: undated open plans from a previous
                         session, greeting Priority 2.6 — "last night you never told me
                         how the soup turned out"; SESSION_OPENER_CONTINUITY_ENABLED).
  emotional_events.py    Sensitive/celebratory emotional memories.
  social.py              Inter-person relationship edges.

vision/
  camera.py              Camera capture.
  face.py                Face detection + recognition (InsightFace SCRFD+ArcFace; dlib fallback).
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
  impersonation.py       "Do an impersonation of me/<person>": target resolution
                         (me/known/famous), live voice-capture persistence, the
                         boundary-excluding parody-script prompt, and the spoken
                         performance (clones a voice via audio/local_tts). The
                         interaction.py glue is the router branch + a pending
                         voice-capture slot.

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

The local Ollama model `qwen2.5:1.5b` (config.OLLAMA_MODEL) is a REQUIRED boot dependency by default (`LOCAL_LLM_ENABLED=True`, `OLLAMA_PRELOAD_REQUIRED=True`): `main.py` preloads it and aborts with `sys.exit(1)` if the Ollama server is unreachable. To degrade gracefully, set `OLLAMA_PRELOAD_REQUIRED=False` (boots with a warning, no sidecar) or disable it entirely with `LOCAL_LLM_ENABLED=False`.

### Speech And Text Turns

`intelligence/interaction.py` is the main turn pipeline.

Spoken turn:

1. VAD/wake or idle speech activation.
2. Transcribe audio.
3. Run speaker ID.
4. Fuse voice with visible/recent world state.
5. Run dialogue-act/contextual binding before executable routing.
6. Execute local handler, or generate the reply via the LEAN BRAIN (primary voice; see
   "Conversation Voice" below), falling back to the classic assembled prompt on lean errors.
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

### Tool-Calling Router (LIVE — routing rides the reply call)

`intelligence/tool_router.py` is the fifth and LAST routing layer, and it is **live**
(`TOOL_ROUTER_LIVE_ENABLED = True`, Phase 1 cutover 2026-08-01, Phase 2 batch
2026-08-02). Full plan + cutover evidence: `docs/tool_router_scope.md`. NOTE the
module docstring still says "Phase 0 (SHADOW ONLY)" — it is stale; the shadow
collector is only the second half of the file.

How it works: `live_reply_tools()` attaches the LIVE subset of tool schemas to the
lean **reply** call, so the model either answers in prose or calls a tool. Routing
therefore folds INTO a call that already happens — it *removes* an LLM round-trip on
routed turns rather than adding one. A tool choice raises `ToolCallRequested` out of
the stream **before any text is spoken** (`lean_brain.py` ~line 795), and
`interaction.py` catches it and dispatches the SAME `_handle_classified_intent`
executor the intent classifier uses, stamping
`final_executed_path="tool_router.<action>"`.

Rules to preserve:

- **It runs LAST, not first.** Every deterministic layer still runs ahead of it, so
  the tool router only catches what used to fall through to conversation — the
  off-pattern phrasings ("how's the weather looking tomorrow?", "kill the music",
  "please forget who I am"). It is not a replacement for the regex fast lanes; the
  measured shipped misses all lived in that fall-through.
- **Never execute a non-live tool.** `resolve_tool_call` returns None for any action
  outside `live_actions()`, so widening the blast radius requires editing
  `TOOL_ROUTER_LIVE_ACTIONS` (config.py, each addition carries its field-log
  justification in a comment). `vision.snapshot` was deliberately held back until it
  had an executor.
- **Write-path and safety guards stay downstream.** `event.cancel` keeps
  `looks_like_cancellation`; `system.sleep`/`system.shutdown` are re-verified with
  `command_parser.is_shutdown_request`/`is_sleep_request`; `memory.forget_person` may
  only ever target the CURRENT speaker (a named third party is never deleted off an
  LLM tool call) and still routes through the wipe-confirmation flow. The tool call
  selects the action; it does not earn a bypass.
- **`ToolCallRequested` stores args on `tool_args`, NEVER on `args`.** `args` is
  `BaseException`'s reserved attribute and silently coerces a dict to a tuple of its
  KEYS — this deafened the robot in the field (see the 2026-08-03 section). A
  source-scan test pins it.
- Humor / character / motion actions are deliberately NOT live: their fast lanes work.
- Kill switch `TOOL_ROUTER_LIVE_ENABLED` reverts to pre-cutover behavior instantly.
  `TOOL_ROUTER_SHADOW_ENABLED` (default False) is the separate off-path collector —
  one extra small call per routed turn, for a collection week; report via
  `tools/tool_router_report.py`.

Tests: `tests/test_tool_router.py` (contracts + coverage-enforcement: a new
`ActionSpec` without a tool definition fails CI).

### Conversation Voice (lean brain primary, classic prompt as fallback)

`intelligence/lean_brain.py` is Rex's PRIMARY conversational voice (`LEAN_BRAIN_ENABLED`).
One streaming model call per turn: the coherent persona (`config.REX_CORE_PROMPT`, via the
`LEAN_BRAIN_PERSONA` override hook) as the system message + a small live situation block
(who he's with, what he sees/hears right now, recent-topic bans, quiet time, mood) + the
recent transcript. It also owns the LULL BREAKERS (`consider_initiating`): quick impulse
and patient re-engage instructions with a rotating, session-deduped "fresh angles" menu and
an anti-music-reflex rule, returning PASS when watching is the better move. Phase 4
("one voice"): proactive/greeting/reaction/onboarding text generated through
`llm.stream_response` routes through `lean_brain.stream_directive` when
`LEAN_ONE_VOICE_ENABLED`, so every spoken line shares the same voice.

The classic layered prompt (`llm.assemble_system_prompt`: REX_CORE_PROMPT + personality
params + emotion/empathy + world summary + person context + transcript + arc + rules) is
NOT dead — it is retained as (a) the reply-path fallback whenever the lean brain errors or
yields nothing (`classic=True` call sites in `interaction.py`), and (b) the base prompt for
web-search replies (`web_search.py`). Do not delete it; do keep new persona/taste rules in
`REX_CORE_PROMPT` (shared by both voices) rather than in classic-only sections.
`REX_CORE_PROMPT` lives in `config.py` and is actively iterated — user_config overrides of
it freeze a stale copy and are deliberately discouraged (both user-config files carry only
a pointer note, not a copy of the text).

### Output

`audio/speech_queue.py` is the central response queue. It handles priority, coalescing, playback start callbacks, and completion.

`audio/tts.py` handles ElevenLabs cache lookup/fetch/playback. In no-audio mode, `speak()` and `ensure_cached()` return before network or playback work. By default `speak()` derives expressive ElevenLabs `voice_settings` from the line's emotion (`emotion_orchestrator.voice_settings_for_emotion`, backed by `config.TTS_VOICE_SETTINGS_*`); an explicit empathy/grief override passed by the caller takes precedence.

**TTS backends.** ElevenLabs is Rex's TRUE voice and the default. `audio/local_tts.py`
is a second, on-device backend (mlx-audio Qwen3-TTS voice clone) that `tts.speak()`
dispatches to per line when: (a) `--local-tts` mode is on (`config.LOCAL_TTS_MODE`),
(b) an explicit `voice_ref` was passed (impersonation — an arbitrary cloned voice),
or (c) the **ElevenLabs fallback circuit breaker** is open. That breaker
(`_note_api_failure`/`_note_api_success`/`_api_circuit_open`) opens on any ElevenLabs
failure — network down, quota exhausted, API error — so Rex finishes the reply in his
local voice instead of dropping the line, and holds the fallback for
`LOCAL_TTS_FALLBACK_HOLD_SECS` (default 120) so the rest of the reply doesn't pay the
API timeout per sentence; the hold expires and the next line re-probes ElevenLabs (a
success clears it early). Both streamed paths share one playback-parity implementation
(`_begin_speech`/`_drive_mouth_chunk`/`_end_speech`: output gate, AEC, mouth LEDs,
servo speech motion, barge-in). Local synthesis is tag-free (Qwen would read `[audio
tags]` aloud) and, for Rex's own voice, cached as WAV under a backend-distinct key
only when `LOCAL_TTS_CACHE_ENABLED` is on — **off by default** so `--local-tts`
testing always hears freshly synthesized audio (the ElevenLabs cache is separate and
unaffected); impersonation takes are never cached. `speech_queue.enqueue(..., voice_ref=...)` threads
the cloned voice to the worker's `tts.speak()` call.

### Web Search (current-info replies)

`intelligence/web_search.py` lets Rex answer questions that need CURRENT information
via OpenAI's hosted `web_search` tool on the **Responses API** (the rest of the app
uses Chat Completions; this branch talks to Responses directly — `llm_compat` is
Chat-Completions-shaped and does not apply). It reuses the existing `OPENAI_API_KEY`
— no new provider, dependency, or secret.

It is a self-contained BRANCH off the normal reply: `interaction._maybe_web_search_reply`
runs at the TOP of `_stream_llm_response` (so the action router and local handlers still
win first), and the tuned streaming reply is untouched. Flow: speak a stall line
immediately (non-blocking, so the multi-second search overlaps with playback and TTFS is
credited to the stall line), run the search, then speak the answer through the normal
`speech_queue`. The answer is voiced through Rex's full persona prompt
(`llm.assemble_system_prompt`) plus `WEB_SEARCH_PERSONA_ADDENDUM`, so it stays in
character. The addendum deliberately OVERRIDES the core prompt's "default to ONE short
sentence" hard limit (the searched call passes no per-turn agenda contract and bypasses
the streaming sentence-governor) so a lookup has room to actually answer — bounded to
~2–4 spoken sentences, no padding. URLs/links/bare domains/"(source: …)" citations are
stripped from the spoken answer (`web_search.strip_links`, `WEB_SEARCH_STRIP_LINKS`) —
Rex reads replies aloud, so a web address is noise; the prompt also forbids speaking
links. Everything is failure-safe — any no-trigger /
no-result / error returns None and Rex falls through to a normal from-knowledge reply
(never silent).

Two triggers:
- **Explicit** — any phrase in `WEB_SEARCH_TRIGGER_PHRASES` (substring, case-insensitive)
  forces a search (`tool_choice="required"`).
- **Autonomous** — Rex decides on his own: a cheap keyword prefilter
  (`WEB_SEARCH_AUTONOMOUS_KEYWORDS`, gated to question-shaped turns) narrows to plausibly
  time-sensitive questions, then a small `gpt-4o-mini` classifier confirms
  (`WEB_SEARCH_AUTONOMOUS_GATE_ENABLED`; off → keyword-only). This keeps the gate off the
  path for ordinary chatter.

`WEB_SEARCH_MODEL` defaults to the conversation model (so the answer is in-voice) but is
independently overridable — point it at a search-capable model if the conversation model
can't host the tool. Search runs at `WEB_SEARCH_REASONING_EFFORT` (off the realtime
first-token path, so a little reasoning is fine). User-tunable knobs (enable, trigger
phrases, stall lines, model, autonomous gate) live in `user_config.example.py`; the rest
of the machinery is in `config.py`. **Enabled by default** with `WEB_SEARCH_ENABLED` as
the kill switch.

**Post-search inquisitiveness:** after a search, if the person goes quiet the proactive
loop would otherwise keep COMMENTING on the searched topic (re-summarizing it, piling on
opinions). A successful search arms a short-lived marker (`web_search.note_search` →
`recent_search`, window `WEB_SEARCH_FOLLOWUP_WINDOW_SECS`, cleared the moment the person
speaks again via `topic_thread.note_user_turn`). While armed, `conversation_agenda`
`.with_proactive_directive` — the single choke point EVERY proactive LLM line passes
through (small-talk questions, visual curiosity, lull callbacks, POV mutterings, env
snark, chime-ins) — appends a directive flipping the lull line to be INQUISITIVE about
the topic ("what got you asking about X?", "are you into it?") instead of repeating the
answer or lecturing. He can still offer an opinion, but attached to a question.
`WEB_SEARCH_FOLLOWUP_INQUISITIVE_ENABLED` toggles it. Tests: `tests/test_web_search.py`.

## Latency And Telemetry

The project now has explicit latency instrumentation.

Key logs:

- `[latency]`: stage timings inside a turn.
- `[ttfs]`: time to first response queued and first audio/text start.
- `[character_loop]`: full per-turn summary: speaker, interpretation, execution, output, memory suppression, timing.
- `[action_router_audit]`: final action routing result.

Recent latency architecture:

- `audio.speaker_id.preload()` runs at startup when `config.SPEAKER_ID_PRELOAD_ON_STARTUP` is true, removing first-turn encoder load cost (ECAPA ~1.3s, Resemblyzer ~0.6s).
- Slow-path acknowledgments (short "One sec." receipts for known-slow `general`/`memory`/`vision` paths) and the delayed latency filler (in-character "One sec, thinking." lines) are now **disabled by default** — `config.SLOW_PATH_ACK_ENABLED = False` and `config.LATENCY_FILLER_ENABLED = False`. They felt out of place, and the streaming answer path now gets Rex's real first sentence out fast, so the latency cover is unnecessary. The machinery and tunables (`SLOW_PATH_ACK_LINES`, `SLOW_PATH_ACK_EXPECTED_SECS`, `LATENCY_FILLER_LINES`, `SLOW_PATH_ACK_IN_TEXT_ONLY`) remain; flip either flag back to True to restore. The slow-path-ack tests enable the flag explicitly to keep covering the firing logic.
- End-of-speech wait `config.SILENCE_TIMEOUT_SECS = 0.65`: how long of sustained silence after the user stops before transcription begins. This is the largest "I stopped talking, why is Rex waiting?" knob. History: 0.6 → 0.85 (2026-07, owner was getting cut off mid-thought) → **0.65** (2026-08-02 latency batch: with every other stage tuned, the hold was the single largest fixed cost left). If mid-sentence cutoffs return, **0.85 is the known-good fallback** and the turn-completion repair prompt is the backstop. Explicit motion commands bypass this entirely via eager endpointing (see the 2026-08-02 latency batch).

When assessing responsiveness, prefer TTFS/audio-start timings over total turn duration. Total duration includes how long Rex speaks.

## Identity And Multiple Speakers

**Voice is the primary signal for WHO is speaking** (`VOICE_PRIMARY_IDENTITY_ENABLED`).
Rex must know who is talking to him even when he can't see them — off-camera, in a
group, in a crowded room. So identity resolution is voice-led; the camera no longer
decides who spoke.

Identity combines:

- Voice embeddings from `audio.speaker_id` (per-person centroid, cosine) —
  `config.VOICE_EMBEDDER` selects the backend: `ecapa` (default; ECAPA-TDNN via
  SpeechBrain, 192-dim, ~20ms/embedding CPU, model under `assets/models/ecapa/`
  downloaded by `setup_assets.py`, auto-falls-back to Resemblyzer if it fails to
  load) or `resemblyzer` (legacy 256-dim). All matchers skip stored prints of the
  other dimension, so the two enrollment generations coexist but never cross-match —
  re-enroll voices after switching (`tools/test_voice_id.py --enroll NAME --replace`).
  THRESHOLD SCALE: every SPEAKER_ID_*/VOICE_SIGNATURE_* threshold stays on the
  Resemblyzer-calibrated scale; `audio/voice_score.map_similarity` shifts ECAPA
  cosines onto it (+`VOICE_SCORE_OFFSET_ECAPA`=0.25, constant so margin knobs keep
  their meaning). ECAPA's genuine/impostor separation is far wider (impostors land
  ~0.25-0.45 mapped vs genuine ~0.55-0.95), which is what retires the
  ambiguous-between-knowns incidents.
- Face recognition from `vision.face` — `config.FACE_BACKEND` selects the backend:
  `insightface` (default; SCRFD detector + ArcFace 512-dim L2-normalized embeddings via
  ONNX Runtime, models under `assets/models/insightface/` downloaded by `setup_assets.py`,
  auto-falls-back to dlib if they fail to load) or `dlib` (legacy HOG/mmod + 128-dim
  descriptor). `memory/people.find_by_face` is dimension-aware: Euclidean thresholds are
  picked by the query's dim (dlib 0.6/margin 0.06; ArcFace 1.10 ≈ cos 0.40/margin 0.08 —
  live-measured same-person d≈0.24-0.37 vs different-person d≈1.37) and stored rows of the
  other dim are skipped, so the two enrollment generations coexist but never cross-match —
  a person enrolled under dlib must re-enroll their FACE under insightface (voice is
  unaffected).
- Current visible people from `world_state`.
- Recent engaged speaker/session continuity.
- Conservative fallbacks when ambiguity is high.

Resolution hierarchy (`intelligence/interaction.py` `_handle_speech_segment`; the
single-visible-face decision is the pure, unit-tested `_voice_primary_face_decision`):

1. **Confident voice wins outright** — a margin-guarded voice match at/above
   `SPEAKER_ID_CONFIDENT_THRESHOLD` (0.70) is trusted regardless of whose face is on
   camera. This is what lets Rex name an off-camera or group speaker.
2. **Accepted-but-not-confident voice does NOT override a visible known face** — the
   accept tiers (hard 0.50 and known-floor 0.45 are margin-guarded; the
   session-sticky 0.60 tier drops the margin guard and instead requires the candidate
   to match the recently-engaged person) resolve a person, but a match *below* the confident threshold (0.45–0.70) that
   points at someone OTHER than the single visible known face does **not** override
   that face: the present, clearly-visible known person anchors identity
   (`voice_weak_face_wins`). A sub-confident match is exactly where an absent/poor
   voiceprint lands a voice on its nearest neighbor (the logged Bret→Wade failure),
   so it must not beat a face the camera shows is the one talking. The only exception
   is when the visual active-speaker latch positively names a *different* on-camera
   talker than the visible face — then the camera agrees the face is not the source,
   and the off-camera voice is kept. Confident matches (≥0.70, tier 1) still win
   outright, which is what preserves genuine off-camera/group recognition.
3. **Weak/absent voice → the face only CORROBORATES** — if the voice did not reach an
   accept tier, Rex attributes the turn to the single visible person **only when the
   voice still leans toward them** (`raw_best_id == that person`), or when there is no
   voice candidate at all in a clean 1:1 with the engaged person on camera. If the
   voice leans toward someone *else*, or the scene is ambiguous, the speaker is treated
   as **off-screen / unknown** — never pinned on whoever happens to be in frame.
4. **Voiceprint refresh is voice- AND vision-gated** — `_maybe_auto_refresh_voice`
   appends a face-confirmed sample only when (guard 1) the voice's own best candidate
   already IS that person AND (guard 2, `AUTO_VOICE_REFRESH_REQUIRE_VISUAL_SPEAKER`,
   default on) the visual active-speaker latch confirms that same person is the on-camera
   talker — so a 3rd-party/TV/AI voice that merely *scores* onto a visible person's print
   can't corrupt it. Refresh is opportunistic and retried per turn, so a turn the camera
   can't confirm is simply skipped (a missed refresh is harmless).

Important behavior:

- The margin guard (`SPEAKER_ID_KNOWN_MARGIN`) is the real protection against a stranger
  who merely *sounds* like a known person: a confident-looking score with no clear lead
  over the runner-up is rejected as ambiguous. Voice scoring is purely relative (an
  unenrolled voice lands on the nearest print at ~0.55–0.65), so there is no absolute
  "is this really X" confidence — margin + threshold together stand in for it.
- An unrecognized voice is tracked within the session as an anonymous label
  (`unknown_voice_1`, …) with its own embedding, in groups and crowds too — it is not
  forced onto a visible person and not dropped. Distinct unknown voices get distinct
  slots, so per-turn attribution works across multiple unnamed speakers.
- **Cross-session voice memory** (`memory/voice_signatures.py`, `voice_signatures` table,
  `VOICE_SIGNATURE_PERSIST_ENABLED`): once an anonymous voice recurs within a session
  (`VOICE_SIGNATURE_PERSIST_MIN_TURNS`) its embedding is persisted, so Rex recognizes it
  in a LATER session ("I've heard your voice before") — without ever creating a nameless
  person row. The moment that voice is finally named (off-screen identify / self-intro),
  `_retire_anonymous_speaker_slot` links the signature to the new person via
  `voice_signatures.attach_person` (the WRITE side). The READ side is wired too:
  when an unrecognized voice confidently matches a signature already linked to a known
  person, `_resolve_anonymous_speaker_slot` resolves the turn STRAIGHT to that person
  (returns them instead of minting a fresh `unknown_voice_N`), so a named-then-departed
  speaker is recognized in a LATER session even without a face/biometric match. Gated by
  `VOICE_SIGNATURE_RESOLVE_PERSON_ENABLED` (default on) above
  `VOICE_SIGNATURE_RESOLVE_PERSON_MIN_SCORE` (0.80 — above the 0.74 match threshold, so
  naming someone needs a confident print). Only fires when there is NO live face/voice
  person match. The "who's speaking?" handler also acknowledges a recurring (this session)
  or previously-heard (prior session) voice instead of a flat "no idea." Writes are
  suppressed under the test runner on the default DB path (temp-DB fixtures opt back in).
- Legacy behavior — "a single visible face wins regardless of voice" — is retained only
  behind `VOICE_PRIMARY_IDENTITY_ENABLED=False` (the `_single_visible_face_voice_override`
  path) as a rollback.
- **Visual active-speaker corroboration** (`vision/active_speaker.py`,
  `docs/active_speaker_detection.md`): when 2+ people are in frame, per-face lip-motion
  energy (jawOpen variance, gated on head yaw + the live VAD flag) decides WHICH visible
  mouth is moving and publishes `is_speaking`/`speaking_confidence`/`speaking_updated_at`
  on `world_state.people`. It composes with voice resolution as a TIE-BREAKER only: in the
  multi-visible branch, a weak voice that leans toward a visible person is accepted at the
  lower `SPEAKER_ID_MULTI_VISIBLE_SPEAKING_FLOOR` when the camera saw exactly that person
  speaking. Vision only CONFIRMS the voice's lean (`_visual_corroborated_speaker`), never
  overrides a confident voice nor invents an off-screen speaker. Because voice resolution
  runs AFTER a turn ends (the live `is_speaking` has cleared by then), it reads the decaying
  latch `active_speaker.recent_visual_speaker()`, not the instantaneous field. Lip energy
  alone can't tell speech from chewing/yawning (chewing reads higher) — the VAD gate makes
  it speech-conditional. Thresholds calibrated on-device (`tools/test_active_speaker.py`,
  `ACTIVE_SPEAKER_LOG_SCOREBOARD`). Tests: `tests/test_active_speaker.py`,
  `tests/test_voice_primary_identity.py`.
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
- Cross-session relationship trends (`memory/trends.py`): visit cadence (streaks,
  visits-per-week, medium gaps), and topics recurring across 3+ distinct days —
  computed from the per-session `conversations` rows, zero LLM calls, cached per day.
  Feeds (a) one compact "relationship trend" line into person context and (b) the
  greeting cadence hook ("third day in a row" / "4 visits this week" / "first time
  in about 2 weeks" — the 2–60-day gap band no other hook covered). Cadence remarks
  only fire for established relationships (sparse profiles get onboarding instead)
  and at most once per day.

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

### First-meeting onboarding (gathering a baseline on NEW people)

When Rex meets someone brand new, a scoped, stranger-only "onboarding" burst
gathers a useful baseline of facts before free-form conversation — a short
ladder of research-backed first-meeting questions, NOT an interview.
`intelligence/onboarding.py` owns the pure pieces; the multi-turn flow STATE
lives in `interaction.py` (`_pending_onboarding`), mirroring the introduction /
"tell me about someone" flows. **Enabled** (`ONBOARDING_ENABLED=True`); set it
False to fully disable the burst.

- **Why it exists:** three gates clamped strangers hardest — the first-meeting
  path ended at the enrollment ack with no question stage; `TIER_MAX_DEPTH
  ["stranger"]=1` locked them to depth-1 questions; and the global question
  budget (`QUESTION_BUDGET_MAX_QUESTIONS=5`/90s) is tier-blind. The fix does NOT
  loosen the deliberately-tight friend-protecting budget — it carves new people
  out of it.
- **Trigger:** armed at the `_enroll_new_person` choke point via
  `_maybe_begin_onboarding` for an eligible newcomer (`onboarding.eligible`:
  `visit_count <= ONBOARDING_MAX_VISITS`, `profile_fact_count <=
  ONBOARDING_FACT_FLOOR`, never a minor — shares `profile_questions.person_is_minor`).
  (`person_specials` VIPs/the creator are INCLUDED by default —
  `ONBOARDING_INCLUDE_VIPS=True`, owner call 2026-07-07: a wiped/fresh VIP row is a
  data-blank like any newcomer, and the visit/fact gates already spare established
  VIPs, so the old default-False skip only ever bit exactly where the burst was
  wanted — live-logged: the creator wiped his row to test and got zero
  getting-to-know-you questions. Set it False to restore the "never interrogate
  the maker" exemption.)
- **No pile-on:** on close, the person is added to
  `_low_memory_idle_questions_spoken` so the *separate* low-memory idle profile
  question doesn't immediately re-fire (onboarding's facts live under categories
  `profile_fact_count` excludes, so that path would otherwise over-question —
  live-logged: "What's your favorite movie?" right after an 8-question burst).
- **Loop:** the opener fires a beat after the ack from the idle loop
  (`_maybe_onboarding_question`); each subsequent answer is consumed before
  routers (`_handle_onboarding_turn`) and answered with a short, warm retort
  (2-5 words, no "?" so it never costs a budget slot) + the next tier-appropriate
  question, with a Rex self-reveal woven in ~every `ONBOARDING_REVEAL_EVERY`
  questions (reciprocity — keeps it an exchange, not an intake form).
- **Question ladder:** `config.ONBOARDING_QUESTION_POOL`, Tier A (essential
  baseline facts) → B (interests/energy) → C (earned depth, only with momentum).
  Ignores `TIER_MAX_DEPTH` (the whole point) but reuses `QUESTION_POOL` keys so
  the asked/answered de-dup (`memory.relationships`) and `QUESTION_BOUNDARY_TOPICS`
  apply for free. Tier-C `origin_followup` is LLM-generated against the live
  answer via `llm.generate_curiosity_question` (the main OpenAI model
  `config.LLM_MODEL` — same brain as the rest of the conversation, with built-in
  grief/heavy-topic restraint; a validated "how'd you get into X?" template
  covers the offline/disabled case); the rest are authored. Optional
  `ONBOARDING_LLM_REPHRASE_ENABLED` re-voices authored questions (also OpenAI).
- **Budget:** rides the `newcomer_baseline` urgent bypass in
  `intelligence/question_budget._URGENT_KINDS`, so it is never blocked by the
  global cap; bounded instead by its own `ONBOARDING_MIN/MAX_QUESTIONS` (3/5 —
  pulled back from 4/8 after a live run felt like an interview).
  The burst's questions still register in the global window, so right AFTER it
  Rex naturally backs off normal questioning.
- **Exits (what keeps it from being an interview):** hard decline / boundary →
  back off out loud (`ONBOARDING_BACKOFF_LINES`); a request or question aimed at
  Rex (pivot) → close and RELEASE the turn to normal routing; soft/flat answers
  do NOT abort before the `ONBOARDING_MIN_QUESTIONS` floor but wind down after it
  (`ONBOARDING_SOFT_DISENGAGE_LIMIT` in a row); reaching MAX closes with a
  closer; sustained silence closes via `_maybe_onboarding_timeout`. Cleared on
  session reset/end.
- **Memory:** each answer calls `rel_memory.answer_latest_pending_question`
  (familiarity bump via `qa_depth_N`) plus a heuristically-tidied
  `facts.add_fact` (`store="fact"`) / `interests.upsert_interest`
  (`store="interest"`) so the baseline feeds prompt injection for the rest of the
  session.
- Proactive speech is suppressed while the burst is open
  (`speech_engine.can_proactive_speak` consults `onboarding_flow_active()`), like
  a tell-about briefing.

Config: `ONBOARDING_ENABLED`, `ONBOARDING_MIN/MAX_QUESTIONS`,
`ONBOARDING_MAX_VISITS`, `ONBOARDING_FACT_FLOOR`, `ONBOARDING_KICKOFF_SECS`,
`ONBOARDING_INACTIVITY_TIMEOUT_SECS`, `ONBOARDING_STEP_TTL_SECS`,
`ONBOARDING_SOFT_DISENGAGE_LIMIT`, `ONBOARDING_REVEAL_EVERY`,
`ONBOARDING_LLM_FOLLOWUP_ENABLED`, `ONBOARDING_LLM_REPHRASE_ENABLED`,
`ONBOARDING_QUESTION_POOL`, `ONBOARDING_REVEAL_LINES`, `ONBOARDING_CLOSERS`,
`ONBOARDING_BACKOFF_LINES`, `COMEDY_LINE_BANKS["onboarding_retort_*"]`.
Tests: `tests/test_onboarding.py`.

## Proactive Behavior

`intelligence/consciousness.py` runs background awareness and proactive behavior.

Important proactive cases:

- Startup greeting: if a known person is in front of the camera, Rex should greet them by name.
- Empty-room startup: if nobody is visible, Rex can make a short snarky empty-room remark.
- First-sight celebration/event check-ins can happen when a remembered relevant event exists.
- Holiday-plan proactivity includes minor public holidays by default — the shipped config sets `HOLIDAY_PLANS_INCLUDE_MINOR = True`. Set it to `False` to restrict proactivity to major holidays only.
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
  with the Mac's caps/zones and linear/angular acceleration limits at connect. Firmware
  slews manual and finite autonomous commands through the same ramp. **`available()` is False when no base is
  connected, so the conversation pipeline is unchanged unless `MOTION_ESP32_PORT` is set.**
- **Routing:** `action_router` has `motion.turn/move/come/stop` action specs and
  `classify_explicit_motion` (deterministic, high-precision — no LLM). `interaction.py`
  dispatches them via `_handle_router_motion_action` and runs the classifier in the fast
  local takeover (gated on `motion_controller.available()`). Bare "stop" routes to the
  base **only while it is moving**, so it never steals stop-music/stop-game/stop-talking.
  Unitless numeric turns such as `turn 180` are local commands; diagnostic questions
  (`why/how come ... move`) and negated motion phrases are explicitly non-executable.
  A successfully issued turn/move/arc seeds one adjacency-sensitive continuation:
  `more` repeats it, `a little more` preserves direction with a small increment, and
  `keep turning` requires a prior turn, `keep moving` requires a prior move/arc, and
  generic `keep going` repeats any of them. Stop, come,
  exploration, an intervening non-motion turn, or the 45-second TTL clears the context.
  Explicit clauses can be chained with `then`, punctuation, or `and` plus another motion
  verb (up to eight steps). Each finite step waits for the ESP32 completion event and an
  idle base before the next; blocked, failed, timed-out, stopped, or superseded routes
  abort their remaining steps rather than partially continuing.
- **Social autonomy:** `motion_agency.py` resolves recognized face locks (`db:<id>`) against
  `person_db_id`, so an already-visible speaker can be approached without a blind scan.
  The flinch reflex requires temporal approach evidence even when firmware reports
  `blocked`; a persistent close ToF return is never, by itself, permission to reverse.
- **Invited exploration:** `exploration.py` scores the eight radial ToF headings (with
  the floor-rejected front 8x8 matrix overlaid in `fl/fr`), turns toward live open space,
  then derives each forward leg from post-turn clearance minus a body margin. Repeated
  legs continue through the opening until clearance shrinks; distance is not randomized.
- **Config:** `MOTION_*` tunables in `config.py`; `MOTION_ESP32_PORT` in `.env` (loaded as
  `MOTION_PORT_SET`). `main.py` Step-4 connects it and logs `Motion base: enabled/disabled`
  like the other hardware; shutdown stops the heartbeat and leaves the base stopped.
- **Runtime tuning (Phase 1):** PID gains + calibration geometry (`counts_per_meter`,
  `track_width_m`, `kp/ki/kd`) are runtime-tunable via the `config` command — calibrate +
  tune the real base live without reflashing each iteration (`firmware/tools/motion_bench.py
  set/show/straight/turn`). `calib.h` holds the firmware boot defaults; the Mac pushes
  overrides only when the matching `MOTION_WHEEL_*`/`MOTION_COUNTS_PER_METER`/
  `MOTION_TRACK_WIDTH_M` key is set (opt-in, so a connect never clobbers a bench-tuned
  value). Wire contract: `docs/motion_protocol.md` §10. ToF is still a stub (avoidance
  inactive); the right motor is hardware-confirmed, left + calibration are next.
  Current-build effective turn track is `0.446 m` (2026-07-21 field correction from
  command 180° → observed ~270° at the former 0.297 m); `config.py` pushes it on connect.
- **Setup:** `setup_macos.sh` (under "physical droid" → "motion base") **auto-detects the
  ESP32 by protocol probe** — it opens each USB-serial port and looks for the firmware's
  `hello` reply. This is the only reliable discriminator because the board's USB bridge is
  a CH340, the same chip as the chest Arduino (chip-ID can't tell them apart). It can also
  install the ESP32 core + ArduinoJson and flash the firmware.
- **Full live ESP32 flash runbook (important):** the battery menu bar LaunchAgent owns
  `/dev/cu.usbserial-110` while `main.py` is down, so FIRST run
  `launchctl bootout gui/$(id -u)/com.djr3x.battery`; otherwise `esptool` can fail
  mid-write with `Invalid head of packet` / `chip stopped responding`. Then run
  `arduino-cli compile --fqbn esp32-bluepad32:esp32:esp32:UploadSpeed=115200
  --build-property "compiler.cpp.extra_flags=-DMOTION_HW_PRESENT=1
  -DMOTION_GAMEPAD_PRESENT=1 -DMOTION_TOF_PRESENT=1
  -DMOTION_TOF_MATRIX_PRESENT=1" --upload -p /dev/cu.usbserial-110
  firmware/djr3x_motion`. LSM6DS3 IMU and QMC5883L/P compass need NO flags; they
  always compile/probe. Verify `charging:true` (~14.2 V), `imu.ok:true`,
  `mag.ok:true`, and live `tof_mm`, then restore the meter with
  `launchctl bootstrap gui/$(id -u)
  "$HOME/Library/LaunchAgents/com.djr3x.battery.plist"`. Full procedure and failure
  recovery: `firmware/djr3x_motion/README.md` § “Safe full-robot flash runbook”.
- **Tests:** `tests/test_motion.py` (fake-ESP32 serial; transport, controller gates,
  clamping, classifier).

Manual control (a Bluetooth gamepad paired directly to the ESP32) and the real
motor/encoder/ToF drivers are Phase 1 — see `docs/motion_system.md` §11, §17.

- **Gamepad soundboard / animation buttons** (8BitDo Pro 2): the firmware
  (`gamepad.cpp` `poll_action_buttons`, behind `-DMOTION_GAMEPAD_PRESENT=1`) forwards the
  buttons motion doesn't use (A/X/Y, Select, Home, L3/R3) to the Mac as
  `event:"button"` — fired whenever the pad is connected, independent of drive owner, so
  they never grab the wheel. `intelligence/motion_controller._on_motion_event` dispatches
  each via `config.MOTION_GAMEPAD_BUTTON_ACTIONS` to a sound clip
  (`audio/soundboard.py` plays an MP3 from `SOUNDBOARD_CLIPS_DIR=assets/audio/clips/`,
  no-audio-safe, mic-suppressed, output-gated so it never talks over a reply) and/or a
  servo animation (`sequences.animations.play_body_beat`). Data-driven map; clips are
  gitignored local audio. Tests: `tests/test_gamepad_actions.py`.
- **D-pad → absolute-heading turn (encoder validation):** the four arrows are NOT
  soundboard buttons — `gamepad_tick` repurposes them to spin the base to absolute headings
  (Up=0°, Left=+90° CCW, Down=180°, Right=−90° CW) for checking the wheel encoders. Each
  rising-edge press issues a MANUAL finite turn (`ctl_manual_turn`) BY the shortest-path
  delta from the live encoder heading (`g_ctx.odom.theta`) — the same encoder-closed-loop
  spin as `turn`, so a correct base lands square at 90° steps; a flipped `ENC_SIGN_*` runs
  away and a wrong `counts_per_meter`/`track_width_m` over/under-rotates. MANUAL so the
  heartbeat watchdog can't abort it (works with the USB link down) and the Mac can't compete;
  a stick push cancels it, B e-stops, Start returns to AUTO. A spin isn't ToF-gated (no
  linear travel) — bench/clear-floor use. Needs `-DMOTION_HW_PRESENT=1` for real encoders.
- **Live gamepad mirror in the GUI**: telemetry carries a `gp` object
  (`{connected, lx, ly, btn}` — left stick + pressed-button bitmask, built in
  `gamepad_tick` / `emit_telemetry`). The GUI's "Motivator Control" window shows a
  read-only **PHYSICAL CONTROLLER** panel (`GamepadMirrorWidget`, `gui/dashboard.py`):
  the dot tracks the real stick and held buttons light up, fed from the existing 150 ms
  telemetry tick. Bit order (`GP_BTN_*` in `gamepad.cpp`) is mirrored by `_GP_BTN_LABELS`.

## GUI

The PySide6 dashboard is optional and launched with `--gui`.

Important GUI behavior:

- It mirrors runtime state and conversation logs.
- **Read-along reply streaming:** a spoken reply fills the conversation panel
  sentence-by-sentence AS Rex generates it (the text leads / reads along with the
  TTS), instead of the whole reply appearing as one block after playback. The
  streaming path (`interaction._stream_and_speak_sentences._consume`) calls
  `conv_log.log_rex_stream(sentence)` the moment each sentence is governed/polished;
  `gui_bridge.append_rex_stream` GROWS a single Rex bubble in place (stable seq), and
  `conversation_panel.set_snapshot` re-renders on a render key of `(last seq, last
  text)` so a growing line under one seq still repaints. The on-disk transcript is
  still written ONCE at the end via `conv_log.log_rex(full_text, to_gui=False)` +
  `conv_log.finish_rex_stream(full_text)` (file-only, so the bubble isn't duplicated).
  Root cause it fixed: the bubble used to be logged once at generation-completion,
  which lands at/after the audio when early (cache-hit) sentences play while the cloud
  model is still streaming. GUI-only + gated on `GUI_ENABLED`; no-audio/non-GUI paths
  unchanged. Tests: `tests/test_conversation_streaming.py`.
  **Performance actions** (roast / joke / free-bit / dj-bit) take a SEPARATE path —
  they speak via `_speak_blocking(log_text=False)` and the caller logs the returned
  text only AFTER the blocking playback, so the line used to appear after TTS. Fixed
  with an `on_text` hook on `performance_output.execute_plan` that logs the line
  (`conv_log.log_rex`) the instant it's generated, BEFORE the speak; the caller's later
  log of the same text dedupes within the 30s window. Tests:
  `tests/test_performance_output.py::test_on_text_fires_with_generated_line_before_speaking`.
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

Ollama/local LLM (`qwen2.5:1.5b`, config.OLLAMA_MODEL) is the low-latency sidecar for quick local tasks. It is ENABLED by default and is a hard boot dependency (`OLLAMA_PRELOAD_REQUIRED=True`): `main.py` aborts (`sys.exit(1)`) if the Ollama server is unreachable. Set `OLLAMA_PRELOAD_REQUIRED=False` to boot without it (warning only, no sidecar), or `LOCAL_LLM_ENABLED=False` to disable it entirely.

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
- Minor public holiday proactive questions are gated behind `HOLIDAY_PLANS_INCLUDE_MINOR`, which ships **True** (minor holidays are ON by default; set it `False` to restrict to major holidays).
- Holiday-plan questions are Lean-owned when `LEAN_BRAIN_ENABLED`: the next eligible
  lull receives a one-shot calendar cue and asks the known person about that upcoming
  holiday. De-dupe remains per `(person_id, holiday_date)` in session and in
  `proactive_topics_asked` across sessions, so the same holiday remains fair game for
  a new person. The old consciousness step is retained only for non-Lean fallback.
  `awareness/holidays.py` uses a local US federal-holiday fallback when its hosted
  calendar cannot be reached; non-US failures retry after `HOLIDAY_FETCH_RETRY_SECS`.
- Remembered-event follow-ups are Lean-owned (`LEAN_EVENT_FOLLOWUP_ENABLED`, default on):
  the old silence-fill `memory_followup`/`small_talk` behavior ("how did the interview
  go?") rode purposes the lean brain suppresses and went dark, so a due plan now enters
  the single lull speaker as one cue instead of a competing proactive line.
  `interaction._lean_event_followup_cue` reads `events.get_pending_followups` (dated plans
  whose date has passed, or undated ones older than `FOLLOWUP_UNDATED_DAYS`) NON-destructively
  — deliberately NOT `consciousness.get_pending_followup`, which POPS the reactive queue —
  and `lean_brain.consider_initiating` gets an `event_followup` cue slot ranked holiday >
  **event** > callback > visual-riff. UPCOMING events are intentionally excluded: anticipation
  ("big day tomorrow — ready?") is still owned by the greeting-time `_pick_anticipated_event`
  path. Coordination with the reactive `_post_response` follow-up and the startup greeting
  follow-up is via shared session state — the moderate `_memory_followup_cadence_allows`
  clamp (gap/cooldown/flat-room) plus the `_fired_followup_event_ids` anti-repeat set every
  path honors — so one event is never asked twice. On a SPOKEN cue (never on PASS/dropped),
  it registers the line under the `memory_followup` RexTurnFrame (so the next reply binds as
  status-update/cancel/dismissal, not an identity command), arms the normal
  `set_awaiting_followup_event` → `_resolve_awaiting_followup` loop that closes the event in
  memory, and purges the in-memory queue via `_pending_followups_lock_remove`. Two hardening
  details from adversarial review: (1) the cue is ranked BELOW celebration (see next bullet)
  — the full order is celebration > holiday > event > callback > visual; (2) the clause is
  DATED-aware — a dated plan whose date passed is asserted to have happened ("how did it
  go?"), but a dateless aspiration surfaced only because it's stale ("redo the kitchen
  sometime") must NOT assert completion, so it asks "did you ever get to it?"; and (3) when a
  LATER lull line opens a *different* thread, the impulse now clears any stale
  `_awaiting_followup_event` so the reply to that new line can't mis-close the old event with
  an unrelated outcome (the single global `_awaiting_followup_event` is frame-independent).
  Fail-safe to no-cue on any DB/lookup error. Tests: `tests/test_lean_event_followup.py`.
- Remembered good-news / CELEBRATION check-ins are Lean-owned (`LEAN_CELEBRATION_CHECKIN_ENABLED`,
  default on): the lean rework left an asymmetry — the HARD-event / negative-affect check-ins
  (purpose `emotional_checkin`, NOT suppressed) kept firing via the legacy
  `consciousness._step_emotional_checkin`, but its POSITIVE branch (Trigger A2, purpose
  `celebration_checkin`) was suppressed, so Rex would console bad news yet silently drop good
  news. `interaction._lean_celebration_cue` restores it as the TOP lull cue (good news is the
  most meaningful open), reading `emotional_events.get_due_celebrations` (valence>0, unmuted,
  not decayed, not acknowledged within the ack-gap). Faithful to the legacy Trigger A2: gated
  by the same masters (`EMPATHY_ENABLED`/`EMPATHY_PROACTIVE_CHECKIN_ENABLED`). Three guards from
  adversarial review: (1) CROWD DISCRETION — good news can be private (a pregnancy, an
  engagement), so the cue honors `EMPATHY_DISCRETION_IN_CROWD` and stays silent when
  `_current_crowd_count() > 1`, exactly like the bad-news console path (the legacy A2 lacked
  this; reviving it without the guard would make good news LESS discreet than bad news). (2)
  DIRECTIONAL session gate — it CHECKS but does not SET `consciousness._emotional_checkin_fired`,
  so a console that already fired blocks a later celebration (don't pile good news on someone
  just consoled), but a celebration doesn't block a later console about a DIFFERENT event
  (matching legacy). (3) STARVATION backstop — since celebration is the TOP cue, a due one the
  model keeps declining to voice (PASS) would starve the lower cues; `_celebration_unvoiced_attempts`
  caps un-voiced offers per silent stretch (`LEAN_CELEBRATION_MAX_UNVOICED_ATTEMPTS`, cleared on
  the next user turn) and then steps aside. On a SPOKEN cue it `mark_acknowledged`s the event
  (per-event 7-day dedup, shared with the legacy path so neither re-celebrates it), logs the
  same rex.db "I celebrated their good news" episode via `episodic_hooks.celebration`, and
  registers under the `celebration_checkin` RexTurnFrame. The legacy A2 governor candidate stays
  suppressed, so there's never a second celebration speaker. Fail-safe to no-cue on any
  DB/lookup error. Tests: `tests/test_lean_celebration.py`.
- "Who's this?" relationship inquiry is a lean-owned PERCEPTION reactor now
  (`consciousness._step_relationship_inquiry`, dispatched in the tick loop). When an unknown
  face lingers alongside the known engaged person for `UNKNOWN_WITH_ENGAGED_CONFIRM_SECS`, Rex
  asks who they are and arms `_pending_relationship_prompt` so the next utterance is parsed as
  the {name, relationship} answer (answer side in `interaction._handle_relationship_reply`).
  It was mis-filed in `LEAN_SUPPRESSED_PROACTIVE_PURPOSES` (it's a perception ask like
  `presence_reaction`, not silence-fill) and went dark under lean; removed from that set, it
  now fires — and its "who's this, {name}?" line generates through the lean one-voice path
  (`generate_and_speak` → `get_response` → `lean_brain.stream_directive`), so Rex asks in the
  lean voice. It stays priority 95, grace-whitelisted (`end_thread`), and question-budget-exempt
  (`question_budget._URGENT_KINDS`). CRITICAL arming fix (also a latent pre-existing bug):
  under `ACTION_GOVERNOR_ENFORCE` `_generate_and_speak` returns True at governor SUBMISSION, so
  the old pre-speak arming + `if not _generate_and_speak(): clear` self-heal was dead code and
  left the reply window armed on candidates the governor then REJECTED (a higher-priority
  reactor wins the tick) — the next user statement got mis-parsed as an answer to a question
  Rex never asked. Arming now lives in an `on_spoke` callback (fires only after the line
  enqueues) plus a `_relationship_prompt_in_flight` latch with a
  `RELATIONSHIP_PROMPT_INFLIGHT_STALE_SECS` auto-clear (so a governor-rejected candidate can't
  wedge the reactor) — mirrors the `identity_prompt` reactor. NOTE the stale window is 40s, NOT
  identity_prompt's 10s: identity speaks a FIXED string (in-flight ≈ enqueue latency) whereas
  this line runs the LLM inside the in-flight window, so the window must exceed
  `LLM_REQUEST_TIMEOUT_SECS` (30s) or a slow-but-legitimate generation gets judged stale and
  re-asked twice. Tests: `tests/test_relationship_inquiry_arming.py`, `tests/test_lean_agency.py`.
- Weather proactive comment is lean-owned now (`consciousness._step_proactive_reactions`, purpose
  `weather.proactive_comment`, priority 42). It reacts to a NOTABLE weather-feed change (a
  rain/snow/thunder/fog onset, a >=10F shift, or a swing into very-hot/cold) — a real-event
  reaction the model can't invent because it isn't told the weather. It was mis-filed in
  `LEAN_SUPPRESSED_PROACTIVE_PURPOSES` and went dark under lean; removed from that set it fires
  again, in the lean voice via one-voice (`generate_and_speak` -> `get_response` ->
  `lean_brain.stream_directive`). No arming/dedup change was needed: weather is fire-and-forget
  (no pending-answer flow like relationship_inquiry) and now behaves exactly like its sibling
  world-change triggers in the same step — date + time-of-day rollover — which already fire under
  lean via the un-suppressed `world_reaction` purpose with the same consume-on-select dedup
  (`tests/test_world_reaction_dedupe.py`). It's gated to a genuine lull by the
  `_ACTIVE_CONVERSATION_LOW_PRIORITY` -35 penalty (so it never interrupts an active conversation),
  a 30-min `WEATHER_PROACTIVE_REACTION_COOLDOWN_SECS`, and per-signature dedup
  (`_acknowledged_weather_signatures`). Kill switch `WEATHER_PROACTIVE_REACTIONS_ENABLED`. Tests:
  `tests/test_lean_agency.py`, `tests/test_weather_awareness.py`, `tests/test_action_governor.py`.
- Episodic "memory musing" is a Lean cue now (`interaction._lean_memory_musing_cue`,
  `LEAN_MEMORY_MUSING_ENABLED`): in a quiet moment Rex occasionally reminisces aloud about
  something from his rex.db diary ("since I was last on" — `episodic_recall.session_recap`). The
  old idle `do_memory_musing` behavior (purpose `memory_musing`) went dark under lean (suppressed
  governor candidate); rather than un-suppress a competing idle speaker, it's fed to the single
  lull impulse as the LOWEST-priority cue (only when no celebration/holiday/event/callback/
  visual-riff fires), so a low-stakes nostalgia beat never crowds out the richer cues. It's
  probability-gated (`EPISODIC_RECALL_SESSION_RECAP_PROBABILITY`) and capped at one per session
  (`_lean_memory_mused_this_session`, reset in `_end_session`) because the recap is stable within
  a session. Data-driven — the model can't invent a memory it wasn't given — so it's a genuine
  cue, not a generic-impulse angle. Generates in the lean voice through the normal
  `consider_initiating` path. See the Memory Model episodic-recall note for the (still-classic-only)
  reply-callback surface. Tests: `tests/test_lean_memory_musing.py`.
- Introduction handling that links known visible/recent people instead of renaming the current speaker.
- README startup flag documentation.
- User-facing override layer: `config.py` ends with `from user_config import *` (try/except ImportError) so `user_config.py` — gitignored, copied from the committed `user_config.example.py` template by `setup_macos.sh` — overrides defaults without editing `config.py`. Defaults stay in `config.py` (source of truth); `from config import X` is unaffected since the change is purely an additive tail. A re-derive tail after the import recomputes `ACTION_ROUTER_MODEL` (= `LLM_MODEL`) and `STARTUP_BOOT_TTS_LINE` so overriding their base propagates. Scope is ~45 essentials (models, personality dials + base prompt, location, feature toggles, timeouts); each ships commented-out at its current default. See the Configuration And Secrets section.

- Expressive TTS voice: `tts.speak()` derives ElevenLabs `voice_settings` from the turn's emotion frame (`emotion_orchestrator.voice_settings_for_emotion`) when the caller passes no override; `TTS_MODEL_ID=eleven_multilingual_v2` (honors `style`); voice settings + model_id are in the TTS cache key and `is_cached()/ensure_cached()` take `emotion`. Don't send `voice_settings=None` on normal turns; empathy/grief overrides win. Knobs: `TTS_VOICE_SETTINGS_*`, `TTS_EXPRESSIVE_VOICE_ENABLED`.
- **Eleven v3 audio tags — leading AND inline/mid-sentence** (`TTS_V3_AUDIO_TAGS_ENABLED`): the affect-mapped LEADING tag ([sarcastic]/[laughs]/… from comedy_mode+emotion, `tts.resolve_audio_tag`) rides chunk 1 of a reply only; INLINE tags mid-reply come from (a) authored canned seam lines — e.g. `repair_moves`' "[excited] I'm sure we'll have better luck next time!" appended after a correction reply — and (b) the lean brain, whose system prompt gains a one-tag-max rule from `tts.llm_inline_tag_rule()` (`TTS_V3_LLM_INLINE_TAGS_ENABLED`). Every synthesis path sanitizes inline tags (`tts._sanitize_inline_tags`: whitelist + `TTS_V3_INLINE_TAG_CAP` on v3; STRIPPED entirely on v2/turbo or kill-switch-off — brackets must never be read aloud); `suppress_audio_tag` only skips the leading prepend, never the sanitize. Tags reach ElevenLabs ONLY: the shared strip helper is `utils/audio_tags.py` (re-exported as `tts.strip_audio_tags`), applied centrally in `conv_log` (transcript+GUI) and to interaction's canonical `spoken`/full_text (memory, handoff, dialogue frames). `repair_moves` matching is tag-insensitive (`_contains_recovery_line`, `note_assistant_turn`). Tests: `tests/test_v3_audio_tags.py`, `tests/test_repair_variation.py`.
- Streaming answer→TTS: audio turns stream sentence-by-sentence (`interaction._stream_and_speak_sentences`) — first sentence speaks ASAP, the rest queue through the single one-at-a-time speech queue (no overlap).
- WorldState lost-update fix: `world_state.mutate(field, fn)` does the read-modify-write under the lock; every `people` writer uses it (not `get()`+`update()`).
- OpenAI warmup: `llm.warmup()` + `action_router.warmup()` run in a background thread (`OPENAI_WARMUP_ON_STARTUP`) so the first turn skips cold TLS/HTTP.
- Stale-event-cancel guard: `memory.events.looks_like_cancellation` requires a cancellation phrase AND no false-positive idiom ("not going to lie", etc.).
- **Voice-primary identity** (`VOICE_PRIMARY_IDENTITY_ENABLED`, default on): WHO is speaking is decided by the VOICE, not the visible face — see the "Identity And Multiple Speakers" section. A *confident* voice match (≥`SPEAKER_ID_CONFIDENT_THRESHOLD` 0.70) wins regardless of who's on camera, but an *accepted-but-not-confident* match (0.45–0.70) pointing at someone OTHER than the single visible known face does NOT override that face — the present known person anchors identity (`voice_weak_face_wins`), since a sub-confident score is exactly where an absent/poor print lands a voice on its nearest neighbor (the Bret→Wade failure); the off-camera voice is kept only if the active-speaker latch names a *different* on-camera talker. A weak/absent match lets the visible face only CORROBORATE (when `raw_best_id == that person`) and otherwise resolves off-screen/unknown; voiceprint auto-refresh is gated on `raw_best_id == person_id` so a different voice can't pollute a print. The old "single visible face wins regardless of voice" rule is retained only behind the flag (`_single_visible_face_voice_override`). Decision logic is the pure, unit-tested `_voice_primary_face_decision`; `tests/test_voice_primary_identity.py`. (Earlier note, now superseded: "sub-0.75 floors require raw_best_id == person, so a 2nd speaker in a 1:1 is treated as off-camera" — the corroboration rule generalizes this to all frames.)
- Bug fixes to keep: `SCENE_MUSIC_BAND_ENERGY_MIN=2e-6` (was a typo making music always "detected"); dead `GUI_SHOW_FPS` removed; `social_frame` optional-lookup excepts log at debug.
- Event follow-up resolution: a reply that an event never happened (`interaction._followup_event_did_not_happen`) resolves a pending follow-up instead of re-asking (kills the "how was the concert?" loop).
- The "one sec" fillers (slow-path ack + latency filler) are disabled by default and `SILENCE_TIMEOUT_SECS=0.65` — see Latency And Telemetry. Don't re-enable without reason.
- The local `assets/memory/people.db` is disposable dev/test data — wipe freely (see Memory Model).
- Upstream merge (~`ffa068e`, authored separately): per-person greetings/intros (`intelligence/person_specials.py`), delayed last-name prompts, sleep wake-word fallback, turn-completion for embedded answers.
- Comedy-forward balance (current intent): Rex leads with a comedic/curious beat, not an every-turn polite interview that opens "Ah," and ends with a profile question. Keep the profile-building machinery, not the interview cadence.
- Friendlier profile-building conversation (machinery, rebalanced behind comedy-forward): asks about hobbies/interests/music/preferences with pointed follow-ups and adapts per person.
- "Muzzle" = decline the music (`interaction._DECLINE_PAT` matches `muzzle(d)`, `keep it off/down/quiet`, `no music`), since the offer line is "…or keep the jukebox muzzled?".
- Wake-word barge-in during DJ playback STOPS the track and listens (was swallowed by `_dj_suppresses_conversation()`).
- Wake over music: `audio.wake_word._threshold(dj_playing=True)` LOWERS the bar — it drops the threshold by `WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA` (floored at `WAKE_WORD_MIN_THRESHOLD`) so a music-masked "hey Rex" can still fire. The TTS-playback path (`tts_playing=True`) lowers it the same way via `WAKE_WORD_TTS_PLAYBACK_THRESHOLD_DELTA`.
- Post-question handoff stickiness: a question turn keeps the response-wait open even if its trailing sentence is a statement (`interaction._apply_post_tts_handoff`, per-sentence + once per reply).
- Crosstalk suppression: `interaction._looks_like_third_party_crosstalk` is HIGH-precision/low-recall — suppresses only the clearest third-party lines; don't over-tighten.
- No-response-quip rhetorical guard: `_question_sentence_expects_response` returns False for rhetorical "who doesn't/wouldn't…?" forms, so Rex's flourishes don't arm the quip.
- Silence drives conversation, doesn't end it: quiet (no goodbye) → re-engage, not sign off (`end_thread._CLOSURE_PAT` no longer matches bare politeness like "thank you").
- Face-tracking responsiveness: the head-pose loop (`consciousness._step_face_tracking`, ~12.5 Hz) is a closed loop (camera on head); `FACE_TRACKING_CENTERING_GAIN` is matched to the FOV.
- Supervisor wake detection is ONNX-only (`wakeuprex.onnx` openWakeWord on 80 ms mono frames, threshold `REX_SUPERVISOR_WAKE_THRESHOLD`); do NOT regress to VAD/Whisper/RMS.
- Always-on wake-word supervisor + single-instance lock (`docs/supervisor.md`, `rex_supervisor.py` LaunchAgent): listens the whole login session, launches `main.py` on "wake up Rex"; an flock keeps main.py single-instance so the supervisor stays dormant while a controller is alive.
- Menu bar utilities (LaunchAgents installed by `scripts/install_supervisor.sh`, both
  dormant while main.py holds the single-instance flock, reclaiming their serial port
  when Rex exits. ESP32-port opens must be plain DEFAULT pyserial opens — on macOS the
  kernel asserts DTR+RTS at open, which is benign, but the Linux "no-reset"
  pre-drop-DTR/RTS trick makes pyserial pass through the EN-low reset state and
  REBOOTS the board on every open (measured 2026-07-13; dropped the gamepad each
  time). Probing the flock uses LOCK_SH so concurrent pollers never false-positive
  each other into port flaps.):
  `com.djr3x.battery` (`tools/rex_battery_menubar.py`, needs MOTION_ESP32_PORT) — pack
  SOC/voltage/current from the ESP32's always-on telemetry, an estimated runtime /
  time-to-full (coulomb-based: remaining_mah ÷ EMA-smoothed current, `_BATT_CAPACITY_MAH`
  must track calib.h), "Set Battery to 100%" gauge sync, "Restart ESP32" DTR reset
  pulse; `com.djr3x.servo` (`tools/rex_servo_menubar.py`,
  needs MAESTRO_PORT) — "Servo Control": live sliders for all 8 Maestro channels
  (Pololu compact protocol direct on the wire, positions read back at connect) +
  "Restart Pololu" (go-home). Servo table mirrors `config.SERVO_CHANNELS` with the same
  `.env` µs overrides — keep in sync if the robot gains a servo.
- Stateless supervisor auto-update (`utils/repo_updater.py`): checks `origin/main`
  at supervisor startup, every four hours, and before controller launch. It only
  fast-forwards a clean local `main`; periodic checks fetch-only while the
  controller lock is held, network/Git failures run installed code, and an
  updated supervisor replaces itself. No updater state files are written.
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
- Conversation arc memory (Bet 1): a running summary of the live conversation + callbacks fed into the system prompt (`intelligence/topic_thread.py`; `tests/test_conversation_arc.py`). Runs off the speech path; the default backend is OpenAI `gpt-4o-mini` (`CONVERSATION_ARC_BACKEND="openai"`) — set `CONVERSATION_ARC_BACKEND="local"` to use the Ollama sidecar instead.
- TurnPlan (Bet 2): typed `conversation_agenda`→`social_frame` handoff replaces prose-directive regex re-parsing (`tests/test_turn_plan.py`).
- Relationship-tone tracking: warmth/edge tracks the RELATIONSHIP, not per-turn (`llm._relationship_tone_rule` over `warmth/antagonism/trust_score`; `tests/test_relationship_tone.py`).
- Offline conversational-quality replay harness (no robot): replays scenarios through the deterministic stack (`tests/test_conversation_replay.py` + `tests/fixtures/conversation_replays.json`).
- Cold-open ranker: the startup celebration cold-open RANKS gate-passing candidates instead of taking the first (`consciousness._pick_due_celebration_checkin`; `tests/test_celebration_ranker.py`).
- Turn classifier (Bet 3): SHELVED/dormant — do NOT regress to on-path (`intelligence/turn_classifier.py`).
- Rex persistent POV: carries ONE current preoccupation and leads with substance instead of react→roast→question (`intelligence/rex_pov.py`; `tests/test_rex_pov.py`).
- Roast rebalance: curious-first, not roast-first (`config REX_CORE_PROMPT`/`PERSONALITY_DEFAULTS` + the live personality DB).
- Memory-followup cadence clamp: stop the proactive event interrogation (`interaction._post_response`/`_memory_followup_cadence_allows`, `FOLLOWUP_*`; `tests/test_followup_resolution.py`).
- Ellipsis trail-off cut-off + idle-banter volunteers the POV + no invented props (`interaction._tail_is_speakable`/`_maybe_idle_banter`, `social_frame` visual rule; `tests/test_streaming_tts.py`).
- Topic-aware idle-banter ask slot: `_idle_has_live_topic()` is now consulted for the ask-user slot too (computed independently of `ask_user`), so once the user has opened a topic the "spotlight on the user" slot DEEPENS that thread (`_IDLE_BANTER_LIVE_TOPIC_ASK`) instead of falling back to the generic "what are you up to?" interview. Fixes the field-logged blunder where Rex asked "What's the latest project you're diving into?" ~50s after the user said they were working on his ToF sensors ("I just told you"). `interaction._idle_banter_directive`/`_maybe_idle_banter` + `_deliver` agenda; `tests/test_idle_banter_relevance.py`.
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
- Episodic memory rex.db — CAPTURE + RECALL (recall IS wired in, contrary to older notes): `memory/rex_db.py` (2nd connection layer, path read at call time) + `memory/episodes.py` capture, gated by `EPISODIC_MEMORY_ENABLED` + a test-runner suppression so the SUITE NEVER creates/writes a real rex.db (reads gated too). Capture hooks + an LLM shutdown session-summary; schema in `setup_assets.py`; `tests/test_episodes.py`. **Recall (Phase 2) is implemented + enabled** via `memory/episodic_recall.py` under the SEPARATE `EPISODIC_RECALL_ENABLED` switch (default on), with two surfaces. (1) The idle **"memory musing" beat** (`session_recap` — a "since I was last on" scene vibe + a couple of experiential highlights) is now a **Lean cue**: `interaction._lean_memory_musing_cue` feeds it into `lean_brain.consider_initiating` at the LOWEST priority (only when no celebration/holiday/event/callback/visual-riff fires), gated by `LEAN_MEMORY_MUSING_ENABLED` + `EPISODIC_RECALL_SESSION_RECAP_PROBABILITY` and capped at ONE per session (`_lean_memory_mused_this_session`, reset in `_end_session`) since the recap is stable within a session. The legacy `idle_behaviors.do_memory_musing` (purpose `memory_musing`, still in `LEAN_SUPPRESSED_PROACTIVE_PURPOSES`) is the pre-lean path and stays governor-suppressed — the cue replaced it. Tests: `tests/test_lean_memory_musing.py`. (2) The per-person **shared-memory reply callback** (`llm._pick_episodic_callback`, prob `EPISODIC_RECALL_PERSON_CALLBACK_PROBABILITY`) is injected into the CLASSIC `assemble_system_prompt` AND — since 2026-08-08 — into the lean reply path too (`lean_brain._person_lines` calls the same picker; shared roll + session dedup, kill switch `LEAN_EPISODIC_CALLBACK_ENABLED`; see the 2026-08-08 entry). The former lean-path recall gap is CLOSED. Two switches kept independent so the diary builds silently during A/B. Do NOT delete `episodic_recall.py` as dead scaffolding.
- Episodic batch-2 capture kinds: enrollment/visit-departure/celebrity/emotional-checkin/celebration/boundary/games + memorable greeting tiers (birthday/milestone/celebration/reunion). Proactive-speech captures are SPOKE-GATED (only when Rex actually spoke); real-world events fire at the event. `memory/episodes.py` + `episodic_hooks` + `interaction`/`boundaries`/`games` hooks.
- Mood-driven body language (`intelligence/body_mood.py`, pure state): a decaying "body mood" set by compliments/insults/amusement shapes posture — head lift/tilt bias on the RESTING pose (never fights the face-centering controller), visor openness, breathing cadence, occasional idle gesture. Visor is hard-clamped to the lens-clear floor (6400) so a mood can't blind the camera. `consciousness._step_mood_expression`/`_mood_rest_bias`; `tests/test_body_mood.py`.
- Calmer head during speech + at the servo rails (`consciousness._step_face_tracking`/`_neck_saturated_at_rail`, `FACE_TRACKING_SPEECH_*`/`FACE_TRACKING_RAIL_DAMP_*`): soften centering while speaking; hold the neck instead of jittering when it's pinned at a limit (`tests/test_face_tracking.py`).
- Compliment detection coverage (`config.COMPLIMENT_KEYWORDS/PHRASES`): broadened so everyday compliments ("nice robot", "good boy", "you're sweet/cool") fire the layer-1 proud beat BEFORE the reply (when the arm servos are free). Phrases, not bare words, to avoid false positives.
- Idle "mind of his own" head wander (`consciousness._idle_wander`/`_step_idle_head_wander`/`_drive_idle_head_wander`, `IDLE_HEAD_WANDER_*`; `tests/test_idle_head_wander.py`): when the conversation lulls with a face locked, look around the room then return gaze and maybe re-greet. The face-loop drives it ABOVE the frame/listening early-returns (self-aborts on speech/listening/resumed talk); a 1Hz backstop ends any stalled wander — `active` can never get stuck.
- Bored environmental snark (`intelligence/idle_behaviors.do_bored_environment_snark`, `BORED_ENV_SNARK_*`; `tests/test_bored_env_snark.py`): an idle riff on the ROOM via `vision.scene.describe_scene_detailed` — complaint / faux-clueless object question / clutter jab / art opinion / take-me-somewhere — grounded in real objects (never invents props), hard-cooldowned.
- Empty-room arc (`consciousness._step_boredom_escalation`): when `BOREDOM_ENABLED`, one owner replaces the old random empty-room micro-behaviors and advances through four paced phases: a fresh camera-grounded look/comment after `EMPTY_ROOM_OBSERVATION_ONSET_SECS`; bored grumbles after `BOREDOM_ONSET_SECS`; "someone left me activated" snark after `BOREDOM_LEFT_ON_PHASE_FRACTION` of the remaining window; then a resignation line and SLEEP after `BOREDOM_SLEEP_AFTER_SECS` of boredom. Any visible person or human engagement resets it. All phases use the Lean-exempt `boredom` purpose. In SLEEP, only the dedicated `wakeuprex` ONNX model returns Rex to interaction; general wake models, Whisper fallback, GUI wake, and text input do not (the `shut_down` ONNX kill-switch may still shut the process down).
- Wave back (`consciousness._step_wave_reaction`, `WAVE_BACK_*`; `tests/test_wave_back.py`): pose detection runs in `vision.pose`'s OWN background loop (`vision.pose._loop`, a `pose-detection` daemon thread at `POSE_ANALYSIS_INTERVAL_SECS`=0.2s), classifying `gesture` onto `world_state.people` — NOT driven by `_step_body_social_analysis`. This reaction step (dispatched right before `_step_smile_reaction`) watches for a visible person's `gesture=="waving"`, LATCHES it, and returns the wave via `animations.wave_back_gesture(half_period=…)` — the half-period MIRRORS the user's measured wave speed (`vision.pose.recent_wave_speed`) — plus one short warm line (`WAVE_BACK_LINES`/`_NO_NAME`, `_speak_async(purpose="wave_back")`). FIRE-WHEN-FREE: if the speech gates are blocked the latch is HELD and retried so the greeting isn't lost. Repeat waves drive an ESCALATING comedy bit (`_wave_escalation`, per person: warm greeting → progressively terser → a joke about the repetition → eventually just ignore it). Debounced per person (`WAVE_BACK_PER_PERSON_COOLDOWN_SECS`=6s) + globally (`WAVE_BACK_MIN_GAP_SECS`=4s); gated by `_can_proactive_speak` (DJ/game/awaiting-reply/give-space) + `suppress_proactive`. Needs MediaPipe pose installed to SEE the wave (graceful no-op otherwise). NOTE: the gesture-DETECTION heuristic is still single-frame (hand at face/shoulder height, arm extended laterally — a held-up hand, not the back-and-forth motion); multi-frame motion is used ONLY to mirror the wave SPEED. The cooldown absorbs the occasional false positive.
- Multi-person vision (`POSE_MAX_PEOPLE`=3): MediaPipe Pose tracks up to N bodies (was 1).
  `vision.pose.detect_pose` extracts ALL poses and `_update_world_state` binds each pose to
  the NEAREST face slot by position (normalized face-box center), so in a group the right
  person gets the right skeleton/gesture (not always slot 0). The phantom-face guard
  (`consciousness._reject_faces_off_body`, `POSE_FACE_GUARD_ENABLED`) now keeps a face near
  ANY tracked head (`vision.pose.head_anchors_px`) and rejects only faces far from EVERY
  body — so a second real person is no longer dropped as a phantom (the earlier single-pose
  guard cost them both a bounding box AND a skeleton). `POSE_FACE_MATCH_MAX_DIST` tunes the
  pose↔face binding distance. Each pose adds inference cost, so keep `POSE_MAX_PEOPLE` small.
  - **Phantom-pose filter** (`vision.pose._is_plausible_pose`, `POSE_PHANTOM_FILTER_ENABLED`):
    at `num_poses>1` MediaPipe hallucinates weak skeletons onto bright blobs (ceiling lights,
    reflections). `detect_pose` drops any pose without a confidently-visible shoulder girdle
    (`POSE_MIN_TORSO_VISIBILITY`, `POSE_MIN_SHOULDER_WIDTH`) BEFORE it reaches world_state, so
    the GUI never draws a light as a body. Keeps frontal / upper-body-only / side-on bodies.
    Detection confidence is also raised + configurable (`POSE_MIN_DETECTION_CONFIDENCE`=0.6,
    was a hardcoded 0.5). Tests: `tests/test_pose_phantom_filter.py`.
  - **Two-person hardening** (JT-intro failures): pose↔face binding in `_update_world_state`
    is now MUTUAL-NEAREST (a pose binds a face slot only if each is the other's nearest
    within `POSE_FACE_MATCH_MAX_DIST`) so a pose can't cross-bind to a neighbour's face;
    `_is_plausible_pose` also rejects frame-spanning blobs (`POSE_MAX_SHOULDER_WIDTH`). The
    GUI only draws a skeleton coherent with a visible face (`GUI_POSE_REQUIRE_FACE`,
    `GUI_POSE_FACE_COHERENCE_DIST`) — kills phantom wireframes "above" people and mis-bound
    ones. Fact/interest/preference extraction filters to HUMAN turns only
    (`llm._human_turns_only`) so Rex's own bits (e.g. "JT volleyball celebrity") can't be
    stored as a person's facts (the name-keyed celebrity bits in `person_specials`, e.g. the
    JT volleyball easter-egg, are INTENTIONAL and still fire on the name — including a fresh
    introduction; the human-only filter just stops the bit's content from being persisted as
    a real interest). The "who's the mystery guest?" agenda stands down for
    `UNKNOWN_GUEST_AGENDA_SUPPRESS_AFTER_INTRO_SECS` after an intro
    (`introductions.intro_recent`). Test: `tests/test_two_person_handling.py`.
- Tidy-up — episodic capture hooks → `intelligence/episodic_hooks.py` (leaf module; consciousness calls `episodic_hooks.<name>`).
- Tidy-up — idle micro-behaviours → `intelligence/idle_behaviors.py` (dispatcher stays in consciousness and calls `idle_behaviors.do_<name>`; the behaviours reach consciousness's speak engine via a lazy `_c` proxy; `_do_small_talk_question` stayed, being mood-detection-coupled).
- Tidy-up — proactive-speech ENGINE → `intelligence/speech_engine.py` (15 functions; consciousness re-exports each as a `_name` shim so call sites + test patches are unchanged; intra-engine calls route through the `_c` shims for full patch-transparency; `note_rex_utterance` + shared speech state stayed in consciousness; `tests/test_speech_engine.py`). The governor metadata key MUST stay `"can_proactive_speak"` (action_governor reads it).
- GUI-first startup (`main._run_gui_mode`): dashboard shows immediately (maximized), controller startup runs on the `controller-startup` thread with `_StartupAborted` checkpoints so window-close/Ctrl+C stops the boot at the next phase boundary; a second Ctrl+C during teardown is absorbed (SIGINT re-pointed in the finally) so `_shutdown()` always runs; fatal startup paths keep the window open on "failed" status and exit non-zero after teardown. System-log panel mirrors the root logger via `utils.logging.install_gui_log_handler` → `gui_bridge.add_log_line`.
- "Tell me about someone" pre-briefing (`intelligence/tell_me_about.py`, `interaction._handle_tell_about_turn`/`_pending_tell_about`): pre-populates the person DB for someone who is NOT here — name → gossip-or-facts → details, stored as `secondhand` person_facts with `fact_kind`/`kindness`/`told_by` (new columns in `setup_assets.py` + `database._run_migrations`). Mean gossip is never recited to the subject (prompt hedging in `facts.format_fact_for_prompt`); secondhand never overwrites the person's own explicit facts. Escapable by design (live-tested): explicit exits (`is_exit` — "exit gossip mode"/"enough about X"/"stop") AND a subject-pivot guard (`looks_like_request_to_rex`) that releases requests aimed at Rex ("can you give me a recipe...") back to normal routing instead of filing them as gossip — the chicken-soup regression in `tests/test_tell_me_about.py`. Proactive barge-ins (smile reaction etc.) trigger a re-anchor question via the `note_rex_utterance` hook → `tell_about_on_external_rex_line`, and every ack invites more (no bare "Logged."). See the Memory Model section.

- Named vision descriptions (fold dlib identity into GPT-4o vision): `vision.face.visible_known_names()` resolves currently-visible recognized people (world_state.people `person_db_id` → name) and is woven into the GPT-4o prompts so a known person is named ("Bret is at his desk") instead of anonymized ("a man at a desk"). `vision.scene.analyze_environment(..., known_names=None)` (the GUI's visual description + LLM world-context, auto-resolves) and `analyze_directed_attention(..., known_names=None)` (the "what do you see?" path — its blanket "do not identify anyone" rule is lifted ONLY for people Rex already recognizes; all other identity/age/health guessing stays banned). The naming directive rides in the directed-attention prompt BODY (not just the safety footer) so `target_summary` itself is named. No extra vision spend — same image, slightly longer text prompt. `intelligence/episodic_hooks._known_visible_names` now delegates to the same resolver. The GUI's visual-description panel reads `world_state.environment["description"]`, which previously only refreshed on the slow periodic scan — so a "what do you see?" query looked frozen; `interaction._update_scene_description` now writes the fresh directed-look summary back into that field so the panel updates on demand. Tests: `tests/test_vision_named_people.py`.
- Callback humor (design: `docs/callback_humor_design.md`): Rex banks durable, light, SELF-volunteered "fun facts" per person (`person_callback_material` in people.db via `memory/callbacks.py`; banker = local qwen labelled-lines + heuristic fallback in `intelligence/callback_engine.py`, run from `_post_response`'s background thread) and resurfaces ONE later — reactively when the background relevance judge connects a stored premise to the live topic, or in a natural short lull. **Lean integration:** a reactive claim now enters `lean_brain.stream_reply` as one narrow system-side `turn_directive` (ordinary Lean turns still receive no agenda stack); a lull premise is selected by `_lean_callback_lull_cue` and passed into the existing single `consider_initiating` speaker, ahead of visual riffs but behind one-shot holiday cues. The old standalone `lull_callback` governor candidate remains suppressed in Lean mode, so there is never a second callback speaker. Long-silence re-engagement never dredges up a callback, and the Lean prompt requires transformation rather than recap (no "you told me/I remember/earlier you said"). Sensitivity is classified at CAPTURE with a deterministic protected-category wall (health/grief/body/orientation/finances/religion-politics/family-conflict/addiction-legal) the model can only move material TOWARD 'excluded' on, never toward 'safe'; only `sensitivity='safe'` rows can ever fire and `active_pool` hard-filters. Spend-at-SPEAK (settle echo-check on the reply path; completed-speech check on the Lean lull path), per-premise reuse cooldown + use-count decay, per-session no-repeat + volume ledger (cleared for real in `_end_session`), 30-min sober-room window after any heavy-sensitivity turn (`note_heavy_moment` from the sensitive prepass), boundary→retire hook in `boundaries.apply_detected_boundary`, forget-flow deletion in `forgetting.py`, crowd/tier/`callback_style`-restraint gates. Flags: `CALLBACK_BANK_ENABLED` / `CALLBACK_HUMOR_ENABLED` (env-overridable A/B pair) + `CALLBACK_*` tunables. Tests: `tests/test_callback_humor.py`.

- Motion system (drive base): wire contract `docs/motion_protocol.md` (v1, locked) + Phase 0 ESP32 firmware (`firmware/djr3x_motion`, full protocol over a stubbed HAL; flashed + 27/27 smoke test) + Mac side (`hardware/motion.py` transport, `intelligence/motion_controller.py` controller, `action_router` motion.* specs + `classify_explicit_motion`, `interaction` dispatch/fast-path, `MOTION_*` config, `MOTION_ESP32_PORT`, `main.py` Step-4 wiring). Gated on `motion_controller.available()` so it's a NO-OP for the whole pipeline unless a base is connected; `stop`/`estop` always pass and bare "stop" only routes to the base while moving. Flash at 115200 (921600 fails on the CH340 bridge); `setup_macos.sh` auto-detects the ESP32 by protocol probe (chip-ID can't — it shares a CH340 with the chest Arduino). Sign convention REP-103. Tests: `tests/test_motion.py`. See the Motion System section above.

- Comedic delivery profiles (`comedy_modes.voice_settings_for_mode` + `config.COMEDY_DELIVERY_PROFILES`/`COMEDY_MODE_DELIVERY_PROFILE`/`COMEDY_DELIVERY_PROFILES_ENABLED`): each comedy STANCE (`comedy_modes.select_mode`) now reaches ElevenLabs with its own timbre instead of the same neutral voice a condolence uses — `dry_ack`/`self_own`/`callback`/`callback_banked` → **deadpan** (flat, dry, deliberate), `friendly_roast` → **smug** (controlled swagger). Mirrors the empathy delivery layer (`empathy._MODE_VOICE_SETTINGS`) but keyed on the turn's comedy mode. Injected at BOTH reply seams in `interaction.py` — non-stream (`_stream_llm_response`) and streaming (`_stream_and_speak_sentences`) — layered UNDER empathy via `if delivery_voice_settings is None`, so **empathy/grief delivery always wins** and `straight`/care turns (no profile) are never comedically shaped. Rides under the global `TTS_EXPRESSIVE_VOICE_ENABLED` (off → comedy off, flat clone + pre-existing cache). Shipped **voice-only** — timing is already covered by `POST_PUNCHLINE_BEAT_MS` (800–1500ms on non-question streaming replies), so no comedy beat map was needed. Started small (deadpan + smug) to bound TTS cache regen. Tests: `tests/test_comedy_modes.py::ComedyDeliveryProfileTests`. See `docs/comedy_improvements.md` (do-first #1, shipped).
- Comedic personas (`comedy_modes._MODES`: `smug_superiority` / `appliance_conspiracy` / `dramatic_narrator`): three recurring-character comedy STANCES added to `select_mode`'s rotation as drop-in `ComedyMode` entries (also in `_SLIM_STANCE`, `_premise_for` anti-repeat tags, and the pool branches). The two SELF-ABSORBED bits (`dramatic_narrator`, `appliance_conspiracy`) are deliberately kept OUT of the interest and engaged-1:1 pools so they don't talk over a sincere share — they fire on explicit-humor / system-word / general-idle turns; `smug_superiority` (a topic-engaging condescension) is allowed on engaged turns. Sensitive/`straight` turns never reach the pools (upstream gate). Each persona pairs with a delivery profile via `config.COMEDY_MODE_DELIVERY_PROFILE` — `smug_superiority`→smug, `appliance_conspiracy`→deadpan, `dramatic_narrator`→a new **theatrical** profile (low stability / high style movie-trailer voice). Completes the comedy plan's Batch 1. Tests: `tests/test_comedy_modes.py::ComedicPersonaTests`.
- Vision-oriented "roast me" (`config.ROAST_VISION_ENABLED` + `VISION_DETAIL["roast"]`, `vision.scene.describe_for_roast`, `performance_plan.plan_for_action(..., visual_material=...)`, `interaction._roast_targets_speaker`/`_roast_visual_material`): when someone explicitly asks to be roasted, Rex roasts **what he SEES** (their look/build/posture/outfit, the mess behind them) instead of riffing on what they last said. The `humor.roast` dispatch (`_handle_router_performance_action`) takes a cheap **gpt-4o-mini** look (`config.VISION_MODEL`, reuses `_call_gpt4o`/`camera.get_frame`) and threads a short visible-detail description into a LOOSENED roast `prompt_contract`; empty description → the original verbal, vibe-based roast (byte-for-byte). **Safety scoping (all fail SAFE to the verbal roast):** only a SELF/room roast (target ∈ {speaker, room} via `_clean_roast_target`, or the speaker's own name) — a named third party ("roast Dave") never gets the vision read; an **unidentified speaker** (`person_id is None`, could be a camera-visible minor) → no vision roast; a known **minor** (`profile_questions.person_is_minor`) → no; an **error resolving age** → no (never assume adult); a **tender/empathy/boundary** moment (non-default `empathy.get_delivery_overrides` mode) → no. Even the loosened contract AND the vision prompt hard-exclude race/ethnicity/religion/disability/medical-conditions/hateful/slurs (the performance path skips the social-frame governor, so the prompt + these gates ARE the floor). One cheap vision call per explicit roast request, never per turn; kill switch `ROAST_VISION_ENABLED`. The prompt + the `describe_for_roast` prompt both put **THE PERSON FIRST** — go after their appearance/build/posture/outfit, explicitly NOT the room/desk/clutter unless there's nothing about the person (a first pass roasted the messy desk instead of the user). Hardened against a multi-agent adversarial safety review (2 minor-safety blockers + a tender-context gap, all fixed). Tests: `tests/test_humor_actions.py::RoastVisionTests`.
- Local object detection → `world_state.objects` (`vision/animal_detector.detect_objects` + `scene.detect_objects_local`, `config.OBJECT_DETECTION_*`, `GUI_OBJECT_BOXES_ENABLED`). BACKEND (`config.OBJECT_DETECTOR_BACKEND`, 2026-07-06): `rfdetr` (default — RF-DETR nano, Apache 2.0, ~40ms/frame CPU, weights in `assets/models/rfdetr/` via setup_assets, auto-falls-back to mediapipe if it fails to load) or `mediapipe` (legacy EfficientDet-Lite0). RF-DETR outputs are adapted to the MediaPipe detection duck-type in `_rf_detections_to_mp`, so BOTH backends feed the same record builders — species lists, per-species thresholds, position phrasing, and the exclusions below are backend-independent. The original stream design: the animal detector already ran the full COCO model and discarded every non-animal box; this stream KEEPS the rest — the room's furniture/items — as `world_state.objects` (each `{id, label, position, last_seen, confidence, source, box?}`), the substrate for §2 object-grounded curiosity / "wait, that's new" change detection / the room model. Reuses the SAME loaded MediaPipe detector as animals (one model; a separate inference pass on its own `OBJECT_DETECTION_INTERVAL_SECS` scan in `scene._scan_loop`), with a `_confirm_persistent_objects` consecutive-scan debounce mirroring animals. **"Rich" privacy posture — open vocabulary MINUS screens/devices** (`OBJECT_DETECTION_BANNED_CLASSES`: laptop/tv/monitor/cell phone/keyboard/mouse/remote, dropped AT detection time so a screen never reaches world_state / GUI / prompt), MINUS `person` (→ world_state.people) and MINUS the animal species (→ world_state.animals). The shared detector's `_load_model` loads when EITHER animals or objects are enabled and uses `max_results=max(animal,object)`; `_mp.Image`+`detect()` run under `_model_lock` (use-after-close hardened on both animal and object paths). GUI: `gui/vision_panel._draw_objects` overlays a violet bbox+label per object (pixel-box scaling identical to `_draw_animals`), gated by `GUI_OBJECT_BOXES_ENABLED`. Animal detection behavior is unchanged. Tests: `tests/test_object_detection.py`. See `docs/comedy_improvements.md` (do-first #2, shipped).
- Land-the-laugh / take-a-bow (`consciousness._step_room_reaction`, `config.ROOM_REACTION_*`; `audio.speech_queue.seconds_since_last_speech`): Rex reacts to the ROOM responding to HIS material — applause → a take-a-bow (`animations.play_body_beat("proud_dj_pose")` + a `ROOM_APPLAUSE_REACTION_LINES` line), laughter → a dry `ROOM_LAUGHTER_REACTION_LINES` follow-through. Reads the otherwise-unread `world_state.audio_scene.applause_detected`/`laughter_detected` (momentary roomwide booleans; applause wins if both fire in a cycle). Mirrors the wave-back step but with NO latch (fire-when-free else skip). Gated on a **recent-Rex-utterance window** so ambient noise/music/TV never sets him off — via the NEW universal `speech_queue.seconds_since_last_speech()` (the worker records `_last_speech_end_at` when ANY line finishes, so it covers replies+roasts+proactive, where the proactive-only `_last_proactive_speech_at` would miss a roast's laugh) ≤ `ROOM_REACTION_AFTER_REX_SECS`; plus a global cooldown (`ROOM_REACTION_MIN_GAP_SECS`, which also collapses one multi-cycle applause burst into ONE reaction) and a LOW per-session cap (`ROOM_REACTION_SESSION_CAP`, reset on session reset). Yields to live speech/games/music via `_can_proactive_speak(reactive=True)` + `profile.user_mid_sentence`. Audio analysis is suppressed during Rex's own TTS, so the laugh/applause lands just AFTER he stops. Tests: `tests/test_room_reaction.py`.
- Object-grounded curiosity (`consciousness._visual_curiosity_objects_line` + injection into `_step_visual_curiosity`, `config.VISUAL_CURIOSITY_USE_OBJECTS`/`_OBJECTS_MAX`/`_OBJECTS_MIN_CONFIDENCE`): the visual-curiosity question is now grounded in a detector-CONFIRMED object from the live `world_state.objects` COCO stream (label + coarse position, confidence-filtered, sorted by confidence, capped). The prompt gets a "Confirmed objects in view (detector-verified — safe to name)" block + a "PREFER grounding it in a confirmed object" instruction, so Rex names a REAL object instead of a possibly-hallucinated GPT detail — while the "never invent an object" guardrail and the non-sensitive bans are preserved and the elaborate gating chain is UNCHANGED. Empty/low-confidence objects degrade to the prior GPT-summary-only behavior. The "most novel / longest-present" prioritization + interest-cross still want the persistent room_model (a later §2 item). Tests: `tests/test_room_reaction.py`.
- Relationship-toned arrivals + departures (`consciousness._presence_relationship_tone`, `config.PRESENCE_RELATIONSHIP_TONE_ENABLED`): Rex already greets known people BY NAME on arrival (`_step_presence_reactions`/`_step_first_sight_presence`, a birthday→emotional→milestone→warm-greeting cascade), welcomes them back (return reactions), and sends them off by name (departure quip) — arrivals already scaled tone via `_greeting_profile`, but the RETURN and DEPARTURE reactions were tone-FLAT ("warm but dry" / "playful and dry" for everyone). `_presence_relationship_tone(person_db_id)` now scales those two — it reuses the reply path's `llm._relationship_tone_rule` over the person's warmth/antagonism/tier to inject a sharper rib for a friend who needles Rex, a warmer one for a close friend, and "" (plain) for a near-stranger/neutral. Fail-safe (returns "" on any error / unknown / disabled). The physical turn-toward (bbox→direction → `directed_look_pose`) is DEFERRED — `directed_look_pose` sets no face-tracking hold (a one-shot glance is instantly re-centered by `_step_face_tracking`) and blocks ~0.65s; build that primitive once when name-and-point call-outs need it too. Tests: `tests/test_presence_tone.py`. See `docs/comedy_improvements.md` (§3, shipped).
- Open plans in the live reply (`llm._open_plans_prompt_line` + `_open_plan_anticipated`, `consciousness.event_recently_anticipated`, `config.OPEN_PLANS_*`): `_build_person_context` read `emotional_events` but never the calendar, so mid-conversation Rex didn't know you have a thing tomorrow. It now appends a short **"Open plans they mentioned: X (tomorrow / on DATE)"** block (added LAST = lowest priority) from `memory.events.get_upcoming_events` (dated, today-or-future, not-followed-up, planned), capped to `OPEN_PLANS_MAX` (2) within `OPEN_PLANS_WITHIN_DAYS` (14), DATED only (undated nags, excluded). It carries a **restraint rule** ("background awareness, do NOT lead with it / force it / nag") and SKIPS any event the proactive ANTICIPATION path already raised this session — via the new `consciousness.event_recently_anticipated(person_id, event_id)` accessor over `_anticipated_events`, reached by a LAZY `from intelligence import consciousness` (consciousness imports llm, so it can't be a module-level dep). Fail-safe (returns "" on any error / disabled / nothing near-term). `OPEN_PLANS_IN_REPLY_ENABLED`. Tests: `tests/test_open_plans.py`. See `docs/comedy_improvements.md` (§8, shipped).
- Room model — object permanence (`memory/room_model.py` + `room_objects` rex.db table, `config.ROOM_MODEL_*`/`ROOM_CHANGE_*`): a persistent per-object ledger so the live COCO stream (`world_state.objects`) gains memory across time/sessions. `record_objects` upserts ONE row PER LABEL (lowercased; `label PRIMARY KEY`; bump `last_seen`+`sighting_count` on conflict) — keyed on label NOT (label,bucket) because the head moves an object's coarse position frame-to-frame, so a chair is one chair wherever it lands; fed from `vision.scene.detect_objects_local` (the COCO scan thread). `label_sightings`/`established_count` query the baseline. Schema lives in BOTH `rex_db.SCHEMA` (runtime) and `setup_assets.REX_DB_SCHEMA` (fresh install) — idempotent `CREATE TABLE IF NOT EXISTS`, so an existing rex.db gets the table via `ensure_schema` (rex.db has NO migration system). Gated + test-suppressed exactly like `memory.episodes` (rides `EPISODIC_MEMORY_ENABLED` + its own `ROOM_MODEL_ENABLED`; the suite never writes a real rex.db — point `REX_DB_PATH` at a temp file to opt in). Screens/devices/people/animals are already filtered out upstream and never reach the table. Two payoffs: (1) **novelty-aware curiosity** — `_visual_curiosity_objects_line` floats objects with `label_sightings < ROOM_MODEL_NOVELTY_MAX_SIGHTINGS` to the front + a "X is NEW" note (degrades to confidence order when empty); (2) **"wait, that's new"** — `consciousness._step_room_change` fires once per new label per session when `established_count(ROOM_MODEL_ESTABLISHED_SIGHTINGS) >= ROOM_CHANGE_MIN_BASELINE` AND a current object's sighting count is in `[ROOM_CHANGE_MIN_SIGHTINGS, ROOM_CHANGE_MAX_SIGHTINGS]` (confirmed-but-recent), heavily gated (baseline kills the fresh-install flood; per-label de-dup marked BEFORE the enqueue so a speech race can't re-fire; 120s cooldown; session cap; lull-only `_can_proactive_speak`). Thread-safe (rex_db `_lock` serializes the vision-thread writes vs consciousness-thread reads). Adversarially reviewed (0 blocker/major; 1 de-dup minor fixed). Tests: `tests/test_room_model.py`. See `docs/comedy_improvements.md` (§2, shipped).
- First-meeting onboarding (`intelligence/onboarding.py` + `interaction._pending_onboarding`/`_handle_onboarding_turn`/`_maybe_begin_onboarding`/`_maybe_onboarding_question`/`_maybe_onboarding_timeout`): a scoped, stranger-only baseline-gathering burst armed at `_enroll_new_person` for a brand-new, non-minor, near-empty profile. Asks a research-backed Tier A→B→C ladder (`config.ONBOARDING_QUESTION_POOL`, ignores `TIER_MAX_DEPTH`, reuses `QUESTION_POOL` keys for de-dup/boundaries), leads each answer with a warm 2-5 word retort (no "?", `COMEDY_LINE_BANKS["onboarding_retort_*"]`) + a periodic self-reveal, writes a tidied baseline (`answer_latest_pending_question` familiarity bump + `add_fact`/`upsert_interest`), and exits on hard-decline/pivot/wind-down-after-MIN/MAX/silence. Rides the `newcomer_baseline` question-budget urgent bypass (does NOT loosen the friend cap); bounded by `ONBOARDING_MIN/MAX_QUESTIONS` (3/5). On close, adds the person to `_low_memory_idle_questions_spoken` so the separate low-memory profile question doesn't pile on. Tier-C `origin_followup` is LLM-generated via `llm.generate_curiosity_question` (main OpenAI model, validated template fallback). Suppresses proactive speech while open (`speech_engine.can_proactive_speak` → `onboarding_flow_active()`). **Master flag `ONBOARDING_ENABLED` is ON** (set False to disable). Related fix: a name-only "this is X" arriving while Rex awaits an answer to his own question (no visible newcomer) is treated as the ANSWER, not an introduction (`_intro_is_answer_to_rex_question`) — the Doubtfire-as-favorite-movie misfire. See the "First-meeting onboarding" subsection above. Tests: `tests/test_onboarding.py`.

- Who's-that voice challenge + enrollment-seeded continuity (2026-07-05/07): a marginal
  (<0.70) voice match on the visible face is CHALLENGED ("who's speaking?") instead of
  silently credited, unless the person's own voice matched confidently within the
  continuity window (`_voice_primary_face_decision` → `challenge_identity`,
  `_voice_continuity_active`, `SPEAKER_ID_CONTINUITY_WINDOW_SECS`) — the camera never
  upgrades a marginal voice. A successful voice ENROLLMENT stamps that continuity anchor
  (`_safe_enroll_voice`): the saved sample IS the person's voice, so a fresh one-sample
  print scoring ~0.5 on the very next turn attributes (`voice_agrees_no_refresh`) instead
  of Rex asking "who's talking?" seconds after being told (live-logged 2026-07-07).
  Tests: `tests/test_voice_primary_identity.py::EnrollmentSeedsVoiceContinuityTest`.
- Onboarding includes VIPs/creator by default (`ONBOARDING_INCLUDE_VIPS=True`,
  owner call 2026-07-07): a wiped/fresh VIP person row is a data-blank like any
  newcomer; established VIPs are already spared by the visit-count/fact-floor gates, so
  the old default-False skip only ever fired on empty profiles — exactly where the
  getting-to-know-you burst was wanted (live-logged: creator wiped his row, got zero
  onboarding questions, silently — the skip logs at DEBUG only). Set False to restore
  the exemption. `tests/test_onboarding.py`.
- 20 Questions guesser rework (2026-07-07, after a live loss to "a rubber ducky"): the
  spine (`features/twentyq_kb.py`) gained authored TIER-2 branch questions (toy/kitchen/
  bathroom/sports/decorative for objects; pet/four-legs/flies for animals; real/famous/
  performer for people; landmark/building for places; drink/sweet/hot for food) gated by
  `requires`/`not_true`, plus smarter pruning ("is it a person?" deferred until man-made=no;
  holdable⇒never a place; edible/place/person prune the material-and-category probes) and a
  size-before-place ask order; `_20Q_SPINE_TURNS` 5→12. The LLM endgame (`games._20q_decide`,
  now medium reasoning effort) gets an established-facts digest, a verbatim never-re-ask
  list, a vetted-splitter menu, and a discriminate-within-the-shortlist rule; a deterministic
  guess gate (`_20q_guess_gate_ok`: shortlist ≤2, or late-game ≤3, or ≤2 questions left)
  converts premature stabs into questions, near-duplicate questions fall back to a proven
  splitter, and after any YES from Q7 on Rex checks for a confident early strike between
  spine questions. Verified 4/4 wins in live-API offline sims (rubber ducky, pizza, stapler,
  Eiffel Tower — the first two shapes were prior losses). `tests/test_twentyquestions_guesser.py`.

- Post-question retro scan (2026-07-07): between the end of a spoken QUESTION and the
  loop's first live mic read there are ~0.3-0.7s (echo-cancel tail + listen-resume delay +
  the synchronous turn unwind) during which NO mic audio is examined — the loop only ever
  VADs the latest 32ms chunk, and while `speech_queue.is_speaking()` it skips chunks
  entirely (`VAD_BARGE_IN_ENABLED=False`). A clipped one-word answer ("no") spoken in that
  dead window lands in the rolling buffer but never triggers live VAD, so it was silently
  lost (live-logged during 20 Questions; normal conversation never showed it because longer
  replies still reach live VAD and the preroll-to-capture-floor recovery grabs their front).
  Fix: `interaction._maybe_recover_post_question_answer` — a question handoff arms a
  ONE-SHOT retrospective VAD scan (`_post_question_retro_scan_at`) of the buffered
  dead-window span, run when the loop resumes listening; a hit feeds the normal
  preroll/floor capture path. The first `POST_QUESTION_RETRO_SCAN_SKIP_SECS` (0.15) after
  the handoff are excluded so Rex's decaying room echo can't fake a hit; raw (unfiltered)
  frames on purpose since `echo_cancel.filter` keys off the CURRENT tail state. Knobs:
  `POST_QUESTION_RETRO_SCAN_*`. An answer that fully OVERLAPS Rex's own playback is still
  unrecoverable without hardware AEC (by design). Tests:
  `tests/test_audio_and_conversation_gating.py::PostQuestionRetroScanTest`.

- ECAPA genuine-band trust floors (2026-07-07, first live ECAPA session): the who's-that
  challenges (2026-07-05) were calibrated on RESEMBLYZER scores, where an impostor
  cross-match lands 0.55-0.66 — indistinguishable from a genuine short turn. Under ECAPA
  an impostor maps to ~0.25-0.45 (below the 0.50 accept bar) while genuine SHORT
  utterances land ~0.55-0.65 mapped — structurally below the 0.75 confident bar — so the
  FIRST short turn of every session was challenged ("who's speaking?") even with the
  right face on camera (no continuity anchor exists at session start; live-logged:
  "yup, I'm back" at 0.597). Fix: `interaction._ecapa_genuine_band` — when the ACTIVE
  embedder is ecapa, a score at/above the trust floor is credible without continuity:
  `_voice_primary_face_decision` gets `score_genuine_band` (accepted agreeing match on
  the visible face → `voice_agrees_no_refresh`, floor `SPEAKER_ID_ECAPA_TRUST_FLOOR_FACE`
  = the 0.50 accept bar) and `_voice_only_attribution_suspect` stands down at/above
  `SPEAKER_ID_ECAPA_TRUST_FLOOR_VOICE_ONLY` (0.55 — no visual prior, higher bar). The
  Resemblyzer fallback keeps the strict guards untouched (its tests pin the backend).
  Kill switch `SPEAKER_ID_ECAPA_TRUST_ENABLED`. Never refreshes the print from these
  turns. Tests: `tests/test_voice_primary_identity.py::EcapaGenuineBandTest`.

- Jeopardy spoken-answer judging rework (2026-07-07): players answer by VOICE, and the
  deterministic matcher missed what STT actually produces. `features/jeopardy.py`:
  `normalize_answer` strips stacked spoken lead-ins ("um, I think it's…", the
  contraction forms "what's/who's" the old prefix regex missed) with an empty-result
  fallback (an answer literally named "Maybe" survives); numbers are canonicalized
  (ordinal words / regnal roman numerals / "8th" → digits, tens+units merged) so
  "Henry the eighth" matches "Henry VIII"; `_spoken_number_string` matches spoken
  years/numbers ("fourteen ninety two"→1492, "nineteen oh five"→1905, "two thousand
  one"→2001) against digit answers; `_phonetic_match` (full-length soundex, prefix
  tolerance 1, length-ratio ≥0.6, fuzz ≥50 co-signal) accepts whisper garbles
  ("day cart"→Descartes, "shack"→Shaq) without re-opening multi-part-answer holes;
  `_surname_match` accepts the surname alone ("Poe" for Edgar Allan Poe — real
  Jeopardy rules); the permissive partial_ratio path now requires user len ≥4 ("ed"
  no longer credits "Edgar Allan Poe"); pass detection covers "no clue"/"beats me"/
  "I give up"/"dunno". games.py adds `_jeopardy_llm_judge` — a strict yes/no
  gpt-4o-mini rescue consulted ONLY when the deterministic matcher says wrong and
  the turn isn't a pass (`JEOPARDY_LLM_JUDGE_ENABLED`, fail-safe to wrong, never
  re-litigates an accept). Validated at scale on the real clue bank: 2000/2000
  "what is X" self-matches, 0/1500 false accepts on random wrong pairs; live-API
  judge spot-check 5/5. Tests: `tests/test_jeopardy_answers.py`.
  With the GUI up, the per-turn spoken "Remaining categories: …" reminder is
  SKIPPED (`_jeopardy_categories_reminder` returns "" under `GUI_ENABLED` — the
  JeopardyPanel shows the live board; the read-out was tiresome); voice-only play
  keeps it, `JEOPARDY_READ_CATEGORIES_WITH_GUI=True` restores it with the GUI, and
  the once-per-round fresh-board announcement is unchanged.

- I Spy look-around (2026-07-07): Rex physically SCANS THE ROOM before picking the
  secret object (`games._ispy_scan_room`: left → center → right via
  `animations.directed_look_pose` under `consciousness.hold_directed_gaze` so the
  face-tracking loop doesn't fight the sweep, `camera.capture_current_gaze` frame at
  each pose, recenter + release after) — the showmanship the physical droid was
  always supposed to have, and a 3× wider object pool. `_ispy_pick_target` sends the
  labeled views in ONE GPT-4o call returning `{object, clue, view}`; the view is
  stored and Rex GLANCES back toward the object at the reveal (correct guess /
  out of guesses / stop) — never during play. A canned `ISPY_SCAN_LINES` stall line
  plays non-blocking UNDER the sweep+vision so it isn't dead air. Servo-less
  machines degrade to the old single-frame pick automatically (`servos.connected()`
  gate); `ISPY_SCAN_ENABLED` kill switch, `ISPY_SCAN_SETTLE_SECS` per-pose settle.
  Live-verified: synthetic 3-view frames → picked the correct object AND view.
  Tests: `tests/test_ispy.py`.

- Person-oriented object salience (2026-07-08, live-logged: Bret held a cup for minutes
  while the lean impulse riffed on a background chair): `vision.scene.
  tag_person_adjacent_objects` runs at the object-stream publish point
  (`detect_objects_local`) and tags small objects whose box center falls inside a
  visible person's body zone (face box widened ×2, extended down 6 face-heights;
  object height ≤2.5 face-heights so a chair can't qualify by overlap; furniture
  labels excluded outright) with `near_person`/`near_person_name`. Both curiosity
  consumers put held items FIRST with an explicit "this beats the furniture" note:
  `lean_brain._scene_summary` ("IN THEIR HANDS … what Bret is drinking/eating/
  fiddling with beats ANY furniture") and `consciousness._visual_curiosity_objects_line`
  (held outranks room-model novelty; items render as "cup (in their hands)").
  Kill switch `OBJECT_NEAR_PERSON_ENABLED`. Live-verified: replaying the failing
  scene through `consider_initiating` produced cup questions ("what's in it?") 3/3.
  Tests: `tests/test_object_detection.py::PersonAdjacentObjectTests` /
  `PersonOrientedCuriosityTests`.
- Held-object remark + adaptive re-engage (2026-07-08 round 2, live-logged: the
  salience above still didn't fire — the lean impulse was blocked by a flat 14s
  flow-quiet gate and the session ended in dead air after "good"→quip): TWO fixes.
  (1) `consciousness._step_held_object_remark` — an EVENT-DRIVEN "what's that you're
  drinking?" that fires once a `near_person` object PERSISTS in-hand for
  `HELD_OBJECT_REMARK_MIN_HOLD_SECS` (first-seen tracking absorbs one-frame
  flicker), yields to live talk via `_can_proactive_speak`, bounded by per-label
  session de-dup + `HELD_OBJECT_REMARK_COOLDOWN_SECS` + `_SESSION_CAP`. Needs NO
  room-model baseline (unlike `_step_room_change`) — a held object is salient on a
  fresh install. Governor purpose `held_object_remark` priority 63 (above
  visual_curiosity 55 / lull_callback 58, below sincerity flows); NOT in
  `LEAN_SUPPRESSED_PROACTIVE_PURPOSES`, so it fires under the lean brain. (2) ADAPTIVE
  re-engage wait in `_maybe_lean_impulse`: the flow-quiet gate SHORTENS to
  `LEAN_IMPULSE_FLOW_QUIET_AFTER_STATEMENT_SECS` (7s) when Rex's last line was a
  CLOSED statement (`_last_rex_line_was_question` False, set in
  `_register_rex_utterance`) — the exchange stalled on him, so he bridges the awkward
  silence sooner; after a QUESTION the floor-hold already governs the wait so the full
  14s stands. Clothing/appearance curiosity ("where'd you get that shirt?") already
  reaches the impulse via the periodic vision scan's `visible_clothing` →
  `_summarize_world_state` env description. Live-verified: held-object prompt →
  "Bret, what's in the cup—coffee, or are you fueling the chaos more creatively?".
  Kill switch `HELD_OBJECT_REMARK_ENABLED`. Tests: `tests/test_held_object_curiosity.py`.
- Personal small-talk impulse register (2026-07-08 round 3, owner: "we're missing
  proactive sentences like 'so, got any plans for the weekend?'"): the held-object /
  scenery emphasis made EVERY lull line anchor on a visible object (logged: cup, chair,
  chair). `lean_brain._choose_impulse_intent` now alternates each impulse between
  `scene` (anchored to what Rex sees) and `personal` (an open life question), with two
  anti-monotony rails — never `personal` twice running, and never a THIRD `scene` in a
  row (after two scene lulls the next is forced personal), so a visible object can't own
  a quiet stretch. `personal` fills the instruction's `{angles}` slot with
  `_personal_steer_clause` ("set the objects and the room ASIDE … ask ONE open, warm
  personal question — 'got any plans this weekend?' energy") drawn from a deduped
  `_PERSONAL_DIRECTIONS` menu (plans / what they're working on / how the week's been /
  what they're into lately). `scene` keeps the existing fresh-angles path. Odds of the
  non-forced turn going personal = `LEAN_IMPULSE_PERSONAL_PROB` (0.4). Live-verified: 6
  consecutive impulses on the logged cup scene yielded 2 personal questions ("what's
  been stealing your attention lately?", "what are you looking forward to this week?")
  and never 3 cup/chair lines in a row. State cleared by `reset_offered_angles`. Tests:
  `tests/test_lean_agency.py::PersonalSmallTalkIntentTest`.

- Wave-back too-close guard relaxed (2026-07-08, live-logged: a genuine wave at 44% face
  height was ignored, no wave-back — "There is code for waving but it appears to be not
  firing"): `WAVE_BACK_MAX_FACE_FRACTION` shipped at 0.30, which rejected the PRIMARY use
  case — someone seated at a desk webcam waves with their face at ~40-50% of frame height
  (the guard's whole point was to drop a face pressed to the lens, but 0.30 caught normal
  desk distance). Raised the default to **0.72** (only a face filling ~¾+ of the frame is
  rejected now). The real anti-phantom protection is elsewhere and untouched — the
  plausible-pose shoulder-girdle filter + `WAVE_BACK_CONFIRM_FRAMES=2` streak — so this
  guard is just a backstop for the on-the-lens degenerate case. The rest of the pipeline
  was already correct (pose→'waving'→face-slot binding→`_wave_face_too_close` all fired in
  the log); only the threshold blocked it. Wave-back speaks its line with or without servos
  (the physical arm gesture no-ops gracefully when no Maestro is connected). Tests:
  `tests/test_wave_back.py` (`test_desk_wave_passes_at_default_threshold`,
  `test_desk_webcam_wave_fires`, retuned close-gate cases).

### Lean-brain cue integration + impulse discipline (2026-07-18)

The lean brain OWNS silence-filling (consciousness proactive candidates are
suppressed under it: `lean_brain_silence_fill_suppressed`). Field session
2026-07-18 showed the new memory features never fired (wrong channel) while
lean improvised six generic questions in three minutes at a tired user. Fixes:

- NEW LEAN CUES (interaction cue ladder): open_thread (after event_followup),
  room_question (before visual riff; skipped when low-energy), news_story
  (before memory musing; one/session, spent once ever). Consciousness-side
  steps for these remain but are dead code under lean.
- IMPULSE DISCIPLINE: rolling rate cap (LEAN_IMPULSE_MAX_PER_WINDOW per
  RATE_WINDOW — does NOT reset on user replies), low-energy read from
  user_energy (statement-or-PASS, longer gaps, no reengage), question_budget
  consulted (exhausted -> no-question addendum + post-check drop).
- FOLLOW-UP HYGIENE: dated events expire FOLLOWUP_DATED_MAX_AGE_DAYS past
  their date (lazily marked followed_up at the source — all consumers);
  wave-backs during recent conversation are GESTURE-ONLY (no spoken
  re-greeting).
- DETECTOR HUMILITY: room-change remarks need a min first/last-seen SPAN
  (ROOM_CHANGE_MIN_SPAN_SECS) and never fire on soft/carriable labels near a
  person (ROOM_CHANGE_SOFT_LABELS); novelty-drive resets at CONFIRM time not
  first sight; "actually that's a X" after a room remark renames the object in
  the room model (room_questions.note_room_remark latch) and answers in
  character instead of the memory_correct_fact canned failure.

### Curiosity plan Phases 2-5 (2026-07-17)

- OPEN THREADS (`intelligence/open_threads.py` + `consciousness.
  _step_open_thread_followup`): the diary's open_threads surface as lull
  follow-ups ("did the thing happen?") at priority 62 — above lull callbacks
  (58) and news (54). Freshness window 6h-21d; each thread asked at most once
  EVER (spent flag persisted in the episode detail JSON).
- NOVELTY DRIVE (`awareness/novelty_drive.py`): time-since-anything-new, fed
  from capture points (new room object, learned name, person enrolled,
  animal). Stale (30 min) tilts the idle micro-behavior mix toward looking;
  very stale + empty room can self-trigger exploration — but ONLY behind
  `EXPLORE_SELF_TRIGGER_ENABLED` (default OFF: it moves the robot unprompted).
- LEARNED NAMES: visual-curiosity prompts speak the human-given object name
  ("they call it 'the sourdough starter'", hedged when single-source).
- RETENTION SWEEP (`memory/consolidation.py`, shutdown): person_seen deduped
  per person per day + 30d age-out, visits 90d, stale pending room questions
  auto-dismiss at 7d. Pure SQL — LLM distillation stays a future upgrade.
- COMPASS SERVICE scaffold (`hardware/compass.start_service`, gated by
  `COMPASS_ENABLED`, default OFF until the QMC5883L is wired + calibrated).
  Spatial anchoring (landmarks at headings) is the remaining deferred piece;
  exploration's open-vocab labels already feed the room model / question
  queue via record_objects.

### Learn-by-asking room questions (2026-07-17, curiosity Phase 1)

The room model now runs a durable ask-about-this queue on `room_objects`
(`ask_status` + `human_name`/`name_confidence` columns, in-place ALTER
migration in rex_db). Rules to preserve:

- QUEUEING is rarity-gated (label never logged before) AND baseline-gated AND
  age-gated (`ROOM_QUESTION_MIN_ROOM_AGE_DAYS`, default 1 — a fresh install's
  furniture trickle must not become an interview).
- STARVATION RULE in `interaction._maybe_ask_low_memory_idle_question`: a
  pending room question outranks the personal profile-question pool; when the
  room is stale, personal curiosity resumes. Shares question_budget pacing.
- ANSWER CAPTURE is passive (`room_questions.maybe_capture_answer`, called on
  every human turn before exploration handling): regex identity extraction,
  never consumes the turn, latch expires after 2 turns / 90s (question then
  auto-dismissed so it can't re-ask forever).
- CORROBORATION (memory-poisoning defense): first answer = confidence 1; a
  matching repeat bumps it; a contradicting claim only replaces a
  single-source name — twice-confirmed names resist one joker.
- `room_model.human_label(label)` exposes the learned name + confidence for
  future curiosity/description prompts ("the sourdough starter", hedged when
  confidence is 1).

### Current-events knowledge (2026-07-17)

`awareness/current_events.py` fetches the day's ~5 notable/viral stories ONCE
per day (date-gated JSON cache at `CURRENT_EVENTS_PATH`) via the hosted
web_search Responses call, kicked as a background thread from main.py at the
start of model preloads (logged, never spoken at startup; a failed fetch keeps
yesterday's cache). Consumer: `consciousness._step_news_remark` offers ONE
unmentioned story per session in a conversation lull as an invitation ("did
you hear about ...?"), priority `NEWS_REMARK_PRIORITY` (54 — deliberately
below lull callbacks: news is B-material and must lose ties to personal
memory). Stories are spent (persisted) only after the line actually plays.
The fetch prompt demands CONCRETE events — it explicitly forbids meta-stories
about news outlets/homepages, which the first live test produced.

### Rex diary quality rework (2026-07-17)

The rex.db first-person diary capture was reworked after the store filled with
third-person null reports ("The person did not share...") at a hardcoded salience
0.8 and forty near-identical scene rows. Rules to preserve:

- `intelligence/llm.py generate_diary_entry` is the DIARY extractor (structured
  JSON: remember/note/salience/open_threads, first-person Rex voice, concrete
  anchors required, permission to stay silent). `generate_session_summary`
  remains for the people.db conversations-table consumer — do not merge them.
- `main.py _episodic_shutdown_summary` gates: `EPISODIC_SUMMARY_MIN_HUMAN_TURNS`
  human turns before the LLM even runs; `remember=false` or salience below
  `EPISODIC_SUMMARY_MIN_SALIENCE` writes NOTHING. No row is the correct output
  for an unmemorable session.
- `detail.open_threads` on conversation_summary rows is the seed for future
  next-visit callbacks ("did you fix the thing?") — planned consumer, keep it.
- Ambient scenes: `SCENE_EPISODE_MIN_GAP_SECS` + token-overlap material-difference
  gate in `episodic_hooks.scene_changed` (near-rewordings of the same room must
  not create new rows).

### Motion sensor verification + autonomous hallway steering (2026-07-22)

- Firmware hallway assist is active during normal forward gamepad `drive`, finite
  Python `move`, and the forward phase of `come`. The paired side ToF sensors center
  between walls; split front ToF adds look-ahead when approaching a wall obliquely.
  Correction is bounded and additive. Reverse, pure turns, and intentional arcs are
  not auto-centered; the ordinary stop/slow reflex still has final authority.
- Finite `turn`, gamepad D-pad turn, and the turn phase of `come` use signed LSM6DS3
  gyro yaw as their completion signal whenever the IMU is healthy at command start.
  Encoder angle is retained as the no-IMU fallback. Wrong-direction motion cannot
  complete the command, and failure to achieve physical yaw eventually emits
  `done:aborted` rather than allowing encoder slip or a stalled pivot to grind forever.
- `intelligence/motion_controller.py` optionally performs a second, absolute-heading
  check after current settles and may issue one bounded corrective turn. It is
  deliberately inert until the QMC5883L is calibrated in place with
  `venv/bin/python tools/compass_calibrate.py` and `COMPASS_ENABLED=1`; raw or
  uncalibrated magnetic readings must never steer the base. A newer motion command,
  stop, e-stop, or disconnect invalidates any delayed correction.

### Latency batch (2026-08-02)

A measured pass over every fixed cost on a turn, after a field session averaged
3.9s perceived reply on simple conversation. Each item was A/B'd, not guessed.

- **Router skips.** Two mirror-image short-circuits around the blocking routing
  call. `ACTION_ROUTER_DETERMINISTIC_SKIP_ENABLED` skips it on
  deterministically-conversational turns (~0.8s). `ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED`
  skips it when the deterministic intent classifier ALREADY claims the turn as a
  self-knowledge query answered from local data (time/date/weather/uptime/
  capabilities/games/who-is-speaking) — the LLM router could only agree (~0.9s). The
  claim must still pass the router's OWN evidence regexes (the classifier's patterns
  are looser), active games keep full routing, and **music/memory/vision are
  deliberately excluded** — the router owns their args and disambiguation.
- **Action router on `gpt-5.4-nano`** via `llm_compat` (0.78s → 0.68s warm, cheaper
  per token). `ACTION_ROUTER_MODEL` is now DECOUPLED from `LLM_MODEL` — the
  user_config re-derive alias was removed, so override `ACTION_ROUTER_MODEL` directly
  to roll back. (The general/utility model stays `gpt-4o-mini`; the conversation path
  is `LLM_CONVERSATION_MODEL = gpt-5.4-mini`.)
- **Warmup 400s fixed.** `llm.warmup()`'s `max_tokens=1` ping had failed on every
  boot since the gpt-5.4-mini flip: GPT-5-family models **400 on a cap they cannot
  finish within** instead of truncating. This was the mystery startup "400 Bad
  Request" in field logs. Cap raised to 16 tokens (same fix in the action_router
  warmup). Keep this in mind for any new GPT-5-class warmup ping.
- **Endpoint hold `SILENCE_TIMEOUT_SECS` 0.85 → 0.65s** — the largest fixed cost
  left once everything else was tuned. 0.85 is the known-good fallback if
  mid-sentence cutoffs return; the turn-completion repair prompt is the backstop.
- **Eager endpointing for motion commands** (`MOTION_EAGER_ENDPOINT_ENABLED`): at
  0.35s of silence a background probe transcribes the segment-so-far, and if it
  decodes to a COMPLETE explicit drive command the turn ends immediately and the
  probe transcript is REUSED (never decoded twice) — wheels move 0.6-0.9s sooner.
  Ordinary speech probes, misses the motion regexes, and waits out the normal hold.
  A trailing connective ("turn left and…") never cuts. Robot-only by default
  (`MOTION_EAGER_ENDPOINT_REQUIRE_AEC` gates on `hardware_aec.is_active()`).
- **ASR context prompt trimmed** (measured ~0.5ms/char): Rex lines join NEWEST
  first (the user re-uses entities from the line Rex JUST spoke — oldest-first meant
  the cap truncated the freshest line's tail, backwards) and
  `QWEN_ASR_CONTEXT_MAX_CHARS` 600 → 400. Removing context entirely would save the
  full ~0.18s but forfeits the field-documented bias fixes ("Lake Folsom" → "like
  falsum"); `QWEN_ASR_CONTEXT_BIAS_ENABLED` remains for that experiment.
- **Reaction-delay pause absorbed** into transcription time: the randomized 0-80ms
  "don't feel robotic" pause used to sleep serially BEFORE transcription. The
  deadline is now set before processing and only the REMAINDER is slept — 0.000s on
  audio turns, while the GUI text path (no ASR) keeps its full pacing pause.
- **Vision call hygiene** (a "what do you see?" turn took 17.5s to first audio):
  frames downscale to `VISION_UPLOAD_MAX_DIM`=1024 before every upload EXCEPT face
  enrollment (which needs the detail); room-level directed looks use detail `"low"`
  (held-object queries keep `"auto"` — small objects need the pixels);
  `VISION_REQUEST_TIMEOUT_SECS`=12 on every `_call_gpt4o` (callers already handle
  None); and the 180s periodic scan DEFERS while a user-initiated directed look is
  in flight. Tests: `tests/test_vision_call_hygiene.py`.

### Conversation quality: plan lifecycle, repairs, low-trust reprompt, bit ledger (2026-08-02)

Four fixes from a Jul 31 - Aug 2 conversation-log analysis.

- **A stored plan is a BELIEF, not a fact.** `extract_events` preserves hedges
  ("might") into a `person_events.hedged` column; hedged plans are ASKED about
  ("still the plan?"), never asserted as scheduled. Every follow-up prompt (startup
  2.5/2.6, lean cue, reactive, Monday-weekend, engagement plan clause) now asks
  whether the plan ended up HAPPENING instead of presupposing it did. Continuity
  greetings stop conflating mention-time with event-time ("How'd Lake Folsom go
  earlier today?" for a trip planned tomorrow). Cancellations with garbled event
  names ("like falsum") fall back to the event Rex raised seconds earlier via the
  memory hint. Diary open threads drop bookkeeping shapes (name corrections,
  mishears, forget requests).
- **Repair moves must actually repair.** The deflection lines were DELETED ("We'll
  get there — recalibrating", "Noted. I'll route around that one", "Consider it
  logged. Onward"): they sounded like acknowledgment while refusing the repair, and
  one ate a direct question. Clarify repairs now explain or honestly admit they
  can't; recovery tags are only injected for the kinds that use them; an embedded
  question inside a correction gets answered.
- **Low-trust reprompt:** a garbled decode carrying real content gets a human
  "Sorry — what was that?" instead of a bluffed reply ("I'm not a cat." at logprob
  -1.88 earned a quip about mystery voices). Once per exchange; short backchannels,
  games, and motion/stop commands exempt.
- **Bit ledger** (`intelligence/bit_ledger.py`, `BIT_LEDGER_*`): session-level
  anti-repeat couldn't see YESTERDAY — the haircut observation ran Jul 31 AND Aug 2,
  "I made you" was re-roasted the next day. Spoken lean impulses are recorded in
  rex.db by topic signature (quoted phrases + content words); a regenerated bit
  inside `BIT_LEDGER_COOLDOWN_DAYS` (5) is dropped, and recent angles feed the lean
  prompt as an EXCLUSION list so generation steers away instead of being vetoed
  after the fact. Repeat = a quoted phrase matches, OR `BIT_LEDGER_MIN_OVERLAP` (2)
  content words are shared with one prior bit, OR a single DISTINCTIVE word
  (`BIT_LEDGER_DISTINCTIVE_LEN`, 7+ chars) recurs. Follow-up-shaped cues (event
  follow-ups, open threads, celebrations, workday check-in) are exempted BY THE
  CALLER — a "how did the interview go?" is attentiveness, not a bit. Fail-safe
  throughout: any error reads as "not a repeat". Tests: `tests/test_bit_ledger.py`.

Also landed this window: **evening workday check-in** (`_lean_workday_checkin_cue`
— Mon-Fri 17:00-23:00, ONE per person per day, durable across restarts via
`mark_proactive_asked`; a single memoized probability roll per (person, day) decides
whether today is a check-in day at all, so it never becomes a ritual; profession read
from `person_facts` so the question can nod to what they actually do; ranked below
real remembered threads, above environment cues) and **time-aware lull nudges**
(`_day_shape_line` — weekend midday / Friday evening / Sunday evening / weekday
evening get an actionable nudge, not just a clock reading).

### Group-room behavior + animal presence ledger (2026-08-02 → 08-03)

From a 3-person session with heavy cross-talk, plus an owner request.

- **Pet-directed speech guard:** "Come here, Max." fired `motion_agency.request_come`
  and the robot DROVE at the speaker; "Lay down." was answered as if aimed at Rex.
  Utterances carrying a known pet name (`config.PET_NAMES`) + an imperative, or bare
  pet-only command shapes (lay down / sit / stay / fetch — things a droid can't do
  anyway), are ignored before any reply/command/motion path. Bare "come here" and
  "turn around" stay Rex commands.
- **Known speakers are gated during group chatter.** The existing ignore path only
  covered `person_id is None`, so the moment a second person was ENROLLED Rex
  answered every human-to-human exchange and literal self-talk. With 2+ humans
  trading turns, a recognized speaker now earns a reply only on DIRECTED evidence
  (name mention, parsed command, awaited answer, weather/vision/look query shapes,
  drive commands, second-person asks); otherwise Rex listens. The lean impulse still
  interjects on its own governed cadence — participation by choice, not reflex.
- **Own-echo coverage window widened to 45s** (`OWN_ECHO_COVERAGE_WINDOW_SECS`): a
  spliced echo joined a 20s-old line (outside the 12s ratio window) to a fresh one.
- **Mouth-still veto recalibrated** (same-day regression): the veto challenged
  genuine Bret twice in one session (0.660 and 0.742, squarely in the ECAPA genuine
  band) because the active-speaker detector misses real jaw motion on short
  utterances at room distance. An empty ASD latch is WEAKER evidence than a
  band-level voice score, so the veto now only overrules SUB-genuine-band scores —
  exactly where impostor cross-matches live. Also removed from `face_only_continuity`
  (an ASD miss there would resurrect the 2026-07-06 "Guest 1 all session" bug).
- **Animal presence ledger** replaces the flat cooldown (`tests/test_animal_returns.py`):
  the old `ANIMAL_ARRIVAL_COOLDOWN_SECS` made the dog's comings and goings invisible
  — first sighting spoke, then silence. Now a per-species ledger in consciousness:
  first sighting = the existing arrival reaction; out of frame under
  `ANIMAL_DEPARTURE_GRACE_SECS` (30s) = frame flicker, NOT a departure (the
  floor-level wide-angle loses the dog constantly — this is what the old cooldown was
  really protecting); a real departure then return = an ESCALATING return joke
  ("womp rat energy" → "doing laps" → "standing docking clearance"), on a happy
  frame not surprise, because the joke is that Rex clocked the pattern. Anti-annoyance:
  ≥120s between SPOKEN remarks per species, ≤4 per species per run (unspoken stagings
  that lose the governor race don't burn the cap), absence ≥30min resets to a fresh
  arrival. Pendings are keyed by species. This rework FIXED one of the three
  documented pre-existing gating failures — the old species cooldown was the
  cross-test leak.

### Field fixes, 2026-08-03 (deafness root cause, known-context recall, rename sweep)

- **THE DEAFNESS BUG — `BaseException.args` clobbered live tool arguments.** The
  first live `web.search` call ever fired was also the last thing the speech loop
  did. `ToolCallRequested` stored its arguments on `self.args` — BaseException's
  RESERVED attribute, which silently coerces a dict to a **tuple of its keys**. So
  `{"query": …}` reached the executor as `('query',)`, `.get` raised AttributeError,
  and the exception killed the listening-loop thread. Wake word kept detecting and
  consciousness kept speaking, so the failure looked like *selective hearing*; only a
  manual GUI shutdown ended it. Every argument-less live tool (time/weather/battery)
  had MASKED the bug — `()` is falsy, so `args or {}` papered over it. Fixes:
  arguments renamed to `tool_args` (+ a source-scan test so no consumer can read the
  clobbered attribute again), and **containment** — `_handle_speech_segment` is now
  wrapped at BOTH loop call sites, so a turn-handler bug costs one turn, never the
  ears. Preserve that wrapper.
- **Known-context recall — memories about the topic at hand now shape the reply.**
  Reply-time recall fired only on memory QUESTIONS, so a STATEMENT touching stored
  memory retrieved nothing ("I got all the new interns set up" → "How many interns
  were there?", a stranger's question, with the plan sitting in `person_events` and
  two diary episodes). `recall.known_context_lines` does statement-time recall over
  person_events + rex_episodes + prior-session summaries, matched with
  `text_match.strong_overlap` (2 shared stems, or one distinctive ≥6-char stem —
  deliberately conservative, since a wrong "you already know this" is worse than a
  missed connection), injected into `lean_brain._person_lines` with a
  connect-don't-re-learn instruction. `events.complete_matching_events` closes a plan
  on a spontaneous outcome report.
- **Embedded shutdown in a compound farewell:** "I will talk to you later, and I
  would like you to shut down." scored conversation (0.20) and the reply model
  generated "Powering down." as a farewell QUIP without powering down. Three layers:
  `command_parser` polite-leader regex accepts desire-form directives;
  `action_router.decide()` pre-routes verified shutdown requests deterministically
  (no LLM, no reply-model tool-call mood); and `system.shutdown` was added to the
  execute allowlist — it was MISSING, so even correct routing died silently.
- **Facing beats mirror-silence.** Rex mirrored a minute of silence at someone
  deliberately waiting for him. Two-step misread: a terse but DIRECTLY responsive
  answer ("It's a Delorean.") scored as a "short reply" and flipped the energy read
  to quiet, and lean's low-energy mode then suppressed every impulse AND the
  re-engage path. Now: a short answer to a question Rex JUST asked never counts
  against engagement (only unprompted terseness reads as low energy), and the new
  `consciousness.person_visibly_facing()` overrides the low-energy read entirely —
  someone facing you during a lull is waiting, not withdrawing
  (`LEAN_LOW_ENERGY_FACING_OVERRIDE_ENABLED`). Disengagement is still judged from
  ACTUAL ignores by the unanswered-run discipline.
- **Rename propagation completed.** The earlier fix only rewrote the RENAMED
  person's own rex.db episodes — but the field mentions live in the OWNER's rows (the
  diary files a session under the primary person present). `rename_person` now sweeps
  all rex_episodes mentioning the old first name AND people.db speakable free text
  (conversation summaries/topics, event names/notes, fact values, interest
  names/notes/stories, Q&A text, relationship labels). Two guards: the sweep stays
  scoped to the renamed person's own rows when ANOTHER person still carries the old
  first name, and the pattern skips "`<OldFirst> <Capitalized>`" so "Brad Pitt" in a
  fact value survives untouched.
- **Phantom wall faces** die on pose-miss ticks; **boot filler stutter** fixed with
  `AUDIO_PLAYBACK_BOOT_LATENCY_SECS` 1.0 → 2.5 (boot-only — the RF-DETR torch load +
  first inference is the boot's heaviest sustained GIL/Metal burst and landed under
  the filler's tail).

### Impersonation hardening + famous voices (2026-08-03 → 08-04)

The feature shipped 2026-07-19; this window made it hold up live.

- **Voice drifted mid-bit** because each pipeline unit is a SEPARATE conditioning
  pass on the reference clip and the passes don't land identically — a
  multi-sentence parody started as the man and finished as someone else doing him.
  Takes now render as ONE unit (`LOCAL_TTS_TAKE_WHOLE_CLIP`, default on). The cost is
  latency (the room waits on the whole bit, not its first sentence), covered by the
  thinking loop and bounded by capping scripts near 45 words. Pipelining is intact
  behind the flag and its tests pin it explicitly.
- **Every bit was about a party.** The prompt said "a room of friends" and the angles
  said "party", and the model read that as a standing fact about the world —
  partygoers, snacks, the dance floor — in a quiet room with one person in it. The
  script may now assume only that a droid is doing an impression and somebody can
  hear it; inventing an occasion is called out as a failed take.
- **Famous mode = half the president, half the droid who took his voice.** The prompt
  requires BOTH halves: anchor on something unmistakably him (the line every
  impressionist does, a fixation, a verbal tic), then collide it with a droid at a
  house party having borrowed his voice. A generic dignified statesman is called out
  as a failed take, because that is what it kept producing. Direction had to be
  spelled out: early takes drifted into the president calling HIMSELF a droid, which
  inverts the joke — he may mock one or deny being one, never claim to be one.
- Intro/bow lines CYCLE rather than picking at random (the repetition was a real bug,
  not model temperature); no emotion chirp fires before an impersonation; head and
  mouth animate through a cloned-voice line; "Impersonate." with nobody named ASKS
  who instead of refusing.
- **Voice reference clips are TRACKED in git** (14 deceased-president refs, ~13 MB,
  trimmed to ~20s of whole sentences and transcribed with whisper-large-v3-turbo) so
  a fresh checkout gets impressions. See `docs/presidential_voice_refs.md`.

### Phantom-wave + phantom-stranger vetoes (2026-08-05)

Live-logged (session 18-31-55): a busy workshop defeated both clutter guards at
once, from one root — MediaPipe fits phantom/mis-fitted POSES across furniture,
and those poses then vouch for other phantoms.

- **Static-wrist veto** (`WAVE_BACK_MIN_SPEED`, default 0.15): chair ARMRESTS at
  camera eye level got "wrist" keypoints planted on them at face height, which the
  single-frame posture check reads as 'waving' — the seated user never raised a
  hand, yet ten wave-backs fired in 90s, every one measuring 0.05–0.09
  normalized-x/s wrist speed (a real wave sweeps 0.25+, per
  `WAVE_SPEED_MIRROR_SLOW`). `_step_wave_reaction` phase (A) now reads
  `recent_wave_speed()` BEFORE latching and rejects a measured speed below the
  floor ("wave ignored … below WAVE_BACK_MIN_SPEED"); None (no motion history
  yet) passes for back-compat, since the confirm streak already spans 1-2s of
  pose ticks. This kills static-wrist false waves regardless of whether the bad
  wrist comes from a fully phantom pose or a mis-fitted pose on a real person —
  the confirm-streak and shoulder-girdle filters can't (a persistent clutter pose
  is stable across frames, exactly what they trust). Tune off the `speed=` value
  now on the wave-detected log line.
- **Unknown-face detector-confidence floor** (`FACE_UNKNOWN_MIN_CONFIDENCE`,
  default 0.62, `consciousness._unknown_face_conf_ok`): a shelf minted a
  PERSISTENT face at ~(545,519) all session — `FACE_UNKNOWN_CONFIRM_FRAMES`
  can't filter what doesn't flicker. The pose-face guard dropped it while the
  user's real pose was the only head anchor, but the moment the user left, the
  only anchor left was a phantom pose on the SAME clutter — which vouched for
  the phantom face (`_reject_faces_off_body` keeps any face near ANY pose head),
  and Rex asked the shelf "what name should I save for you?", then bade it
  farewell when the head panned away. An unidentified face must now also clear
  the SCRFD det-score floor to count as an unknown person or feed the
  persistence streak (real faces score ~0.7–0.95; clutter FPs hug the 0.5
  accept threshold). Known-face tracking is embedding-vouched and unaffected;
  dlib (no det score) passes unchanged. Calibration logging: rejected faces log
  `[face_conf_gate]` with score+box; accepted unknowns log `det_scores=[...]`
  on the "unknown face detected" line — raise the floor from live data if the
  shelf still gets through.
- The deeper root (phantom poses coasting on MediaPipe VIDEO-mode tracking for
  minutes — "1 real pose(s)" cycling neutral/raising_hand/leaning_in in an
  EMPTY room, even "2 real pose(s)") is deliberately NOT touched yet: tightening
  `_is_plausible_pose`'s side-on fallback or the pose confidences risks
  re-breaking real two-person scenes; revisit with live det-score/pose data.
- Tests: `tests/test_wave_back.py::WaveSpeedGateTest`,
  `tests/test_unknown_face_persistence.py::UnknownFaceConfidenceGateTest`.

### Rex's day mood + greeting cadence (2026-08-05)

Owner gripe, two halves with one root — Rex had no persistent state about HIMSELF or
about what he had already said to you:

1. **"How are you?" always got "operating within normal parameters."** Nothing
   special-cases a wellbeing question about Rex; it falls through every classifier to
   the generic LLM reply, and that prompt carried ZERO information about his own
   state. `REX_CORE_PROMPT` then hands the model "systems nominal" as a canonical
   droid tic, so a status report was the only attractor. The reciprocal case ("I'm
   good, how about you?") was worse: `intent_classifier._CONTEXTUAL_FOLLOWUP_RE`
   forces it to `general`, and nothing anywhere knows the human is bouncing Rex's own
   question back at him.
2. **Repeat visits re-ran the full hello.** Every anti-repeat guard was IN-MEMORY —
   `_greeted_this_session` is a set wiped at process start, `_should_fire_presence`
   uses `time.monotonic()` cooldowns that reset with the process — so a RESTART, the
   single most common way to "arrive" twice in an hour, defeated all of them.
   `PRESENCE_STARTUP_RECENT_RETURN_GRACE_SECS` only ever changed the P4 wording.

- **`intelligence/rex_mood.py`** — ONE mood per LOCAL day, minted from what the day
  handed him (holiday → news → weather → hardware → clock, first firing signal owns
  the stated reason, every firing signal contributes tags to the roll), DRIFTING
  during the day (`note("good_conversation"/"long_quiet"/"insulted"/…)`, clamped to
  `REX_MOOD_DRIFT_LIMIT` so the day keeps its character), tapering energy in late
  hours at READ time only, and persisted to `assets/memory/rex_mood_state.json` so a
  relaunch at 4pm resumes the morning's mood plus its drift. Anti-repeat is an
  EXCLUSION, not a de-weighting: the RNG is seeded on the date, so shrinking a weight
  just slides the cumulative bands under a fixed cursor and lands back on the same
  seed (measured: three identical moods in a row). Do not "simplify" that back.
- **Do NOT write the day mood into `world_state.self_state["emotion"]`.** The three
  existing affect layers are all short and reactive — emotion_orchestrator frame ~8s,
  `body_mood` ~45s, `personality._mood_intensity` ~10min — and
  `personality.apply_mood_decay` would stomp a day mood to `neutral` within ten
  minutes. The day mood is the BASELINE those ride on, held in its own state.
- **Injection goes in BOTH voices.** `lean_brain._system_prompt` (the live path —
  covers replies AND greeting/proactive directives under ONE VOICE) *and*
  `llm.assemble_system_prompt` (classic fallback + web search). This is the rex_pov
  trap, avoided deliberately: rex_pov injects only into `assemble_system_prompt`, so
  under `LEAN_BRAIN_ENABLED` its preoccupation never reaches a direct reply at all.
  A new self-state that lands in only one builder is invisible in production.
- **`REX_CORE_PROMPT` gained a wellbeing rule** (shared by both voices, per the
  "keep persona rules in REX_CORE_PROMPT" rule above): answer a question about your
  own state as someone who HAS a day; never with "systems nominal" / "normal
  parameters" / an uptime figure; never the same answer twice in one day. The authored
  mood line is explicitly an EXAMPLE, not a script — otherwise the fix just relocates
  the repetition one level down.
- **`intelligence/greeting_cadence.py`** — stateless; every answer derives from
  PERSISTED per-person timestamps, which is exactly what a reboot cannot reset.
  `recency()` buckets `people.last_greeted_at` into SNAP (<20 min → a half-sentence
  acknowledgment, asks nothing) / RECENT (<3 h → warm return beat, no wellbeing
  question) / None (normal ladder). New `people.last_wellbeing_ask_at` column tracks
  the ask on its own clock (`WELLBEING_ASK_COOLDOWN_SECS`, 4 h) because a return hello
  and "how are you?" decay differently. Consumed as greeting-ladder **Priority 3.4**
  (above the calendar-day-coarse P3.5 same-day beat) plus a suppression clause
  appended to whichever branch won.
- **The ask is detected from Rex's FINAL TEXT, not from which prompt-builder ran** —
  the builder only says what he was told to do. Recorded at both seams:
  `consciousness.note_rex_utterance` (greetings/proactive) and
  `interaction._register_rex_utterance` (replies), falling back to
  `get_recent_engagement()` since only ~7 of ~25 callers pass `target_person_id`.
- **Volunteering it unprompted** (owner follow-up, same day: "real people do that").
  A mood only ever revealed under interrogation is still a lookup table, so
  `interaction._lean_mood_share_cue` offers it in a lull as ONE dry aside via
  `lean_brain._MOOD_SHARE_INSTRUCTION`. It sits BELOW every cue about them (asking
  after someone's weekend beats talking about yourself) and ABOVE generic news.
  Four gates keep it from becoming a new daily ritual — which would just be the
  original complaint wearing a different hat: notable days only
  (`rex_mood.is_notable`, ~12 of 18 shipped moods; nobody announces feeling exactly
  average), a random roll (`REX_MOOD_SHARE_PROBABILITY`), friend tier or better,
  and once per DAY via the persisted `spoken` flag — so answering "how are you?"
  in the morning also spends the unprompted share, and a reboot doesn't re-arm it.
  `rex_mood.share_cue()` deliberately owns only "is there something worth saying";
  the roll and the social fit live in the caller, like every other lull cue.
- **Also in the HELLO** (owner follow-up: "can his opening line ever offer up his
  current emotional state, or does it only fire during lulls?"). It was lull-only —
  the greeting directive already received the mood, but the day-mood bullet ends with
  "don't announce your mood unprompted", so the hello only TINTED his tone.
  `consciousness._greeting_mood_aside` now appends an aside to plain-hello branches
  (`_GREETING_MOOD_ASIDE_LABELS`), appended LAST so it explicitly overrides that
  standing rule for the one line. Excluded: anything about THEM (birthday,
  celebration, emotional check-in, milestone, follow-up, anticipation) and the
  <20-min "snap" quick-return (whose contract is four words). Shares the same
  once-per-DAY `spoken` spend as the lull cue, so it decides WHICH of the two gets
  it, never both — spent on dispatch inside the `if queued:` block, with a
  text-match belt in `note_rex_utterance`. Measured over 80 simulated days: ~31% of
  days at the hello, ~20% in a lull, ~49% not at all.
- Tests: `tests/test_rex_mood.py`, `tests/test_greeting_cadence.py`,
  `tests/test_self_state_injection.py`, `tests/test_mood_share_cue.py`.

### Lull quietness telemetry + tuning (2026-08-05)

Owner: "during lulls he sits there quiet so much." The lull gauntlet in
`interaction._maybe_lean_impulse` is ~18 gates deep, each added for a documented
over-talking gripe — individually defensible, multiplicatively harsh, and with no
visibility into which gate was doing the silencing.

- **Telemetry:** every consult now records exactly one outcome (`spoken`,
  `watched_pass`, `dropped_*`, or a gate name) via `_impulse_blocked` /
  `_impulse_outcome`. Gate TRANSITIONS log at INFO (the consult loop ticks ~1/s, so
  repeats are counted silently); a per-session rollup logs at session end — grep
  `[lean] impulse session summary` to see exactly where the speech went. A
  source-level completeness test (`tests/test_lean_quietness_tuning.py`) fails if a
  new gate is added without instrumentation — it caught one on day one.
- **PASS re-arms only half the window** (`LEAN_IMPULSE_PASS_REARM_FRACTION`, 0.5).
  The anchor arms on every consult so the model isn't hammered, but the instructions
  praise PASSing, so each polite shrug bought a FULL window of guaranteed silence —
  chained PASSes on the 40s re-engage path meant minutes of dead air. 1.0 restores
  old behavior.
- **+1 unanswered-line allowance when visibly on camera**
  (`LEAN_IMPULSE_MAX_UNANSWERED_VISIBLE_BONUS`): sitting in plain view is soft
  permission for one more try; the base cap of 2 stands for voice-only, where
  silence more plausibly means they left. `_person_visible_now` fails CLOSED — it
  only grants a bonus.
- **Probe snooze 600s → 240s** (`ENGAGEMENT_PROBE_NO_ANSWER_SNOOZE_SECS`): ten
  minutes of total silence was a harsh sentence for missing one 30s answer window.
- Tune from live data now: run a session, grep the summary, and adjust the gate the
  numbers actually blame — don't blanket-loosen (the gates each trace to a real
  dated over-talking field report).

### Reaction awareness + spoken news digest (2026-08-05 evening session)

Two fixes from the 20:54 field log:

- **First-person reaction awareness** (`intelligence/reaction_awareness.py`). The
  smile reaction used to speak a canned interjection ("Oh look, I made the lifeform
  smile") — a sensor report wearing a joke, and it over-triggered. The DETECTION
  pipeline (quip arms a watch, adaptive-baseline smile confirm, cooldown, diary
  hook, giddy body mood) is unchanged; the confirm now records awareness instead of
  speaking. That awareness rides in the live prompt (lean `_reaction_lines` +
  classic section) so Rex's NEXT line can own that the joke landed in first person
  — or not mention it and just let it color his tone. One-shot spend on his next
  finalized line after injection (both seams: `_register_rex_utterance` and
  `consciousness.note_rex_utterance`), TTL 90s, per-person isolation (Bret's smile
  never colors a reply to JT), newest reaction replaces the old. Legacy canned path
  is behind `SMILE_REACTION_CANNED_LINES_ENABLED` (ships False).
  Field-fixed same night (e942cb7): the awareness fired but never reached a prompt —
  (a) `consider_initiating` built its system prompt from the BARE persona, so lull
  lines were generated blind to ALL self-state; it now carries day mood + reaction
  awareness + the spent how-are-you ask (person/scene context deliberately stays
  out — the situation block owns that). (b) The SECOND smile system
  (`_step_facial_expression_reactions`, spontaneous sustained smiles) still emitted
  canned interjection candidates; smiles there now feed `reaction_awareness` with a
  "you just caught them smiling" framing (no trigger line exists).
  Surprise/brow-furrow keep their spoken path.
- **Spoken news digest contract.** "Tell me more" about an offered story produced a
  ~150-word press release read aloud (platform roll-calls, "if you want, I can also
  pull the submission rules…"). `_compose_news_search_input` now demands three
  short sentences max, friend-relaying-news register, no platform lists /
  submission mechanics / marketing phrasing, and no closing fetch-menu.
- Tests: `tests/test_reaction_awareness.py`; `tests/test_smile_reaction.py` updated
  to the new default contract (+ a legacy-flag test).

### Field fixes, 2026-08-05 21:18 session (five issues, one run)

All in `tests/test_field_2026_08_05_night.py`:

- **Phantom wave-backs at a motionless arm** — every firing logged `speed=n/a`.
  Two fixes: (a) wrist speed is now measured RELATIVE to the wrist's own shoulder
  (`vision/pose._raised_wrist_x`) so Rex's neck pans (camera egomotion) cancel out
  of the measurement; (b) UNMEASURED speed now fails the veto — flickering clutter
  keypoints wipe the motion history, so the phantom is precisely the unmeasured
  case (`WAVE_BACK_UNMEASURED_IS_WAVE=False`; True restores the old pass).
- **Own-echo rejector ate a human echo-question** ("An AWS outage?" repeated back
  as a follow-up, voice-scored 0.868). A voiceprint match to a known human at
  `OWN_ECHO_VOICE_OVERRIDE_SCORE` (0.80)+ now overrides the text match — Rex's AEC
  residual comes back as `unknown_voice_N`, never as a confident human match.
- **Triumphant "proud" chirp minutes after he said he felt sluggish** — celebratory
  chirps (`proud`, `laughing`) now defer to the day mood
  (`REX_MOOD_GATES_CELEBRATORY_CHIRPS`; suppress at valence ≤ −0.2 or effective
  energy ≤ 0.25). Posture still shifts; only the loud fanfare is gated.
- **"Why thank you!" → "Why? Thank you."** — ASR idiom correction in
  `WHISPER_CORRECTIONS` (the LLM was answering the "Why?").
- **"I don't know, … can you shut down?" → "I couldn't safely parse that whole
  route."** — `classify_explicit_motion_sequence`'s negation/explanation guard
  returned None (spoken route-rejection) on ANY comma-containing utterance with a
  "don't", before classifying a single clause. The guard now applies AFTER clause
  classification: negation over real motion clauses still refuses whole (None);
  negation over zero motion clauses is conversation ([]).

### "He stops hearing me right after he speaks" (2026-08-05 → 08-06)

Owner symptom: *"I pause a second after he speaks and he doesn't hear my first
line — so I repeat myself."* Four commits, and the shape of the investigation is
worth preserving because a **negative** telemetry result is what cracked it.

- **Capture-drop telemetry first** (`interaction._capture_outcome` /
  `_capture_dropped` / `_log_capture_session_summary`, same shape as the lull-impulse
  telemetry): count every outcome, log TRANSITIONS at WARNING/INFO, dump one line per
  session — grep **`[capture] session summary`**. It reported `captured=6 / dropped=0`
  for the very run that lost three utterances. That negative result **proved the loss
  was upstream of capture entirely**: the audio never became a segment, because the
  mic was muted — not because the segment died.
- **ROOT CAUSE — the mic was released by bookkeeping, not by silence.**
  `echo_cancel.start_sequence()` defers every per-segment `set_playing(False)` until
  `end_sequence()`, and that release sat at the END of the reply path, behind the
  post-greet relationship ask, the curiosity routine, and pool-topic recording. The
  mic stayed attenuated for **1-5s after Rex's last audio** (measured medians 1.0-2.0s,
  max 8.0s; the outliers line up exactly with the owner-labelled repeats).
  `_chunk_for_vad` flattened a reply spoken into that window to ~5%, VAD never fired,
  and the turn left NO trace anywhere. Release now happens from the speech-queue
  done-callback the moment the queue is **DRAINED**. Gated on the new
  `speech_queue.is_drained()` and NOT on `not is_speaking()` — the latter is briefly
  true BETWEEN the sentences of one streamed reply, and releasing there would re-open
  the mic into Rex's own next sentence. A later line re-suppresses on its own
  playback, so releasing early costs nothing. `AEC_RELEASE_ON_QUEUE_DRAIN=False`
  restores the old behavior. Tests: `tests/test_aec_drain_release.py`.
- **The capture floor anchored on the wrong clock** (front of fast replies clipped:
  "I know, am I right?" heard as "Am I right?"). `_apply_post_tts_handoff` runs from
  the done-callback, which fires 0.5-1.5s AFTER audio actually stopped (streamed-take
  cache save, sequence bookkeeping), and it set the floor to `now - grace` — i.e.
  AFTER words the human had already spoken into the clean post-playback buffer.
  `echo_cancel` now stamps when sound REALLY stopped (`last_playback_ended_at()`,
  including for the final segment, whose `set_playing(False)` the sequence hold
  swallows) and the floor anchors there. On hardware AEC the grace still reaches back
  PAST that end (residual is ~17dB down); on software suppression the floor sits AT
  the end and reaches no further, because that direction is Rex at full volume.
  Bounded by `CAPTURE_FLOOR_PLAYBACK_END_MAX_LAG_SECS` (3s) — an OLD stamp from a
  previous line would otherwise drag the floor seconds into the past and let capture
  reach back over unrelated audio (the suite caught this in the first cut).
- **Software suppression flattens a fast reply's ONSET**, so VAD triggers a beat late
  and a preroll-sized reach-back still misses the first word. When VAD fires within
  `CAPTURE_FROM_FLOOR_NEAR_SECS` of the floor, capture now starts **AT the floor** —
  everything after it is post-playback and clean, worst case a second of leading room
  tone that ASR shrugs off. Tests: `tests/test_front_clip_fast_reply.py`.
- **Bit ledger read delivery tags as content.** Five good lull lines were refused in a
  row as "repeats a recent bit", on five unrelated subjects. `content_tokens()`
  tokenized the inline `[curious]` DELIVERY tag as a content word — at 7 characters
  `curious` clears `BIT_LEDGER_DISTINCTIVE_LEN`, and that rule short-circuits on a
  single shared distinctive token, so one stored bit containing the word "curious"
  silently blocked EVERY future line tagged `[curious]`, whatever it was about. Tags
  say how a line is SPOKEN, never what it is about, so they never belonged in the
  comparison; `content_tokens` now strips them via `utils.audio_tags.strip_audio_tags`
  first. Costlier than one line each: every drop also benches its cue for
  `LEAN_CUE_DROP_COOLDOWN_SECS`, so five false positives silenced five cues for ten
  minutes on top of the lost lines. Tests: `tests/test_bit_ledger_tags.py`.
- **Remaining instrumented gap:** the ACTIVE loop can `continue` without polling the
  mic (post-TTS resume window, output busy) and those ticks were unlogged. They are
  now counted and SPLIT — `mic_skip_output_busy_sfx` vs `mic_skip_rex_speaking` — so a
  servo chirp landing in the human's reply window is distinguishable from Rex actually
  talking. A sound effect is audio output, so the mic is not read while one plays, and
  effect rate was ~1.8× higher in the worst runs than in the good one. **Suggestive,
  not proven** — read the next session's `[capture] session summary` before acting.
- **Methodology note worth keeping:** an earlier "not a regression" call in this
  thread was WRONG because it compared dev-Mac runs against ROBOT runs. Those are not
  comparable — do not mix them when bisecting.
- `docs/revert-2026-08-05-session.env` turns off every behavior change from the
  2026-08-05 session in one paste (eight env-overridable flags), so the whole session
  can be A/B'd in a single run instead of one hypothesis at a time.

### News timing + digest length, and the invisible searched answer (2026-08-06)

- **"Did you hear about the eclipse TODAY" — the event was six days out.** The stored
  story was correctly dated in BOTH the headline and the summary and the model still
  said "today": **a news frame implies immediacy, so a headline handed over bare gets
  announced as now.** This is the same failure `_build_anticipation_prompt` already
  fixed for remembered events, and it gets the same cure — compute the delta IN CODE
  and STATE it. Both news paths (the lull offer and the "tell me more" wrapper) now
  carry a TIMING clause. When a story mentions several dates ("premiered July 23 …
  finale airs August 9") it LISTS them and makes the model choose: asserting the wrong
  one confidently is worse than the vagueness this set out to remove.
  Date-parsing rules that a 30-agent adversarial review had to beat into the first
  cut, all worth preserving: **exact month names only** (`(jan|…|dec)[a-z]*` made any
  month-prefixed word a month — "Officials DEClared 6 counties" parsed as December 6,
  replacing a safe "you don't know when" hedge with a confident falsehood); a bare
  number before a month is NOT a day ("Pixel 9 August feature drop" → August 9) —
  day-first needs an ordinal, an "of", or a year; and the scan must not return on the
  first regex hit, or one bogus match suppresses a real date.
- **Digest length: contradictory rules in one prompt.** The system prompt said "give
  the COMPLETE answer … two to four sentences" while the user message said "three
  max", and "tell me more" ran ~90 words and read "(esa.int)" aloud. News follow-ups
  now carry their OWN system contract (three sentences, ~45 words, no platform lists,
  no closing fetch-menu); every other search keeps the complete-answer one. An
  overshoot is SHORTENED by a second cheap call rather than truncated — per the owner,
  "don't just cut it off."
- **The searched answer was invisible in the GUI for ~30s.** It rode the caller's
  post-return log in `_handle_speech_segment`, but `_maybe_web_search_reply` speaks
  through `_speak_blocking`, which returns only after PLAYBACK finishes — so the
  longer the answer, the longer it stayed invisible, and the news digest is the
  longest thing Rex says. (Ordinary replies look instant because the streaming path
  logs as soon as the text exists.) The search path now writes transcript + GUI
  **before the first word is spoken** and parks the text in a one-shot marker so the
  caller skips its duplicate write. Deliberately NOT `conv_log.claim_rex_line()`,
  which exists for exactly this shape — its dedupe is time-bounded at
  `_REX_DEDUPE_WINDOW_SECS` = 30s, and a long news answer takes about that long to
  speak, so the caller's later write would land on the boundary and duplicate roughly
  half the time. The marker is time-independent, also covers `add_to_transcript`
  (which `claim_rex_line` does not touch), and is cleared in `_begin_user_turn` so a
  path that errors after logging can't suppress a legitimate write on a later turn.
- Tests: `tests/test_news_timing_and_digest.py`, `tests/test_websearch_gui_immediacy.py`.

### A reply to Rex's NEXT topic filed as the room's name (2026-08-06)

Field log 10-24-04: Rex asked "Which room is this, Bret?", Bret answered "This is the
workshop", 24s later Rex offered a news story, Bret said "Tell me more." — and that
became a room. `[place_questions] learned room 'tell me more' (place_id=4)`, ack'd with
"Got it — the tell me more. I'll remember this place.", and the observe loop enrolled
**8 embeddings of the actual workshop** under it before shutdown. Two independent
defects, both fixed; either alone would have prevented it.

- **The latch only ever counted HUMAN turns, so it survived REX changing the subject.**
  `note_asked()` armed a 3-turn/120s window and nothing closed it when Rex opened a
  different thread — so the news reply landed in a window belonging to a question he had
  already talked past. `place_questions.note_rex_line(text, source)` now disarms it,
  called from `consciousness.note_rex_utterance` (the seam EVERY Rex line passes through,
  including `interaction._register_rex_utterance`, which delegates to it). Lines in
  `_PLACE_FLOW_SOURCES` keep the latch, and `note_asked(text)` records the ask's own text
  so the ask can never cancel its own latch whatever order the caller uses (today
  registration runs first, but the text exemption makes that ordering non-load-bearing).
  This is the `_awaiting_followup_event` rule — a later line opening a different thread
  clears stale awaiting-state — applied to the place flow.
- **The latched bare-answer path accepted any 1–4 word phrase** that didn't start with a
  filler, with no notion of what a room name looks like. `_bare_answer` now vetoes
  request shapes (`_IMPERATIVE_OPENERS` on the FIRST WORD + a `_NOT_ROOM_PHRASES` set):
  "tell me more", "go on", "shut down", "play music", "turn left" are things said TO Rex,
  never what a room is called. **Precision guard:** a phrase carrying a known room word,
  or simply ending in a room head noun (`_room_head_nouns()` — derived from
  `PLACE_ROOM_WORDS`, so user_config additions extend it free), bypasses the veto — which
  is what keeps the compound "play room" / "show room" / "the back nook" working. The
  first draft vetoed those and was caught by its own test.
- Cleanup: the poisoned `places.db` row + its 8 embeddings and 1 observation were
  deleted (backup at `data/places.db.bak-*`); the gallery is back to dining room /
  workshop / living room at 15 embeddings each.
- **Not fixed, separate root cause:** Bret's real answer never reached the pipeline.
  At 10:26:56 the segment decoded as Rex's OWN two prior lines and was correctly
  rejected — `[transcription] context-echo hallucination rejected (ratio 0.87 …)` →
  `EMPTY result — segment dropped`. The answer was spoken into the post-TTS capture seam
  and lost to AEC residual, which is why Rex moved on at all. See the capture-seam notes;
  the guards above mean a lost answer now costs a missed room, not a fabricated one.
- Tests: `tests/test_place_questions.py` (request-veto, head-noun bypass, disarm,
  ask-exemption, place-flow-source cases).

### A servo whir was eating the answer to Rex's own question (2026-08-06)

The capture-drop telemetry added the night before paid for itself on its first field
log (13-04-31): `[capture] session summary: mic_skip_output_busy_sfx=148,
mic_skip_rex_speaking=71, mic_skip_listen_resume=11, captured=6`. The dominant loss
channel was not Rex speaking — it was his own **decorative sound effects**.

- **Root cause: the capture loop never consulted `sound_effects._suppresses_mic`.**
  That function already decided (field 2026-07-25 — "the move sound effects and motor
  whine are cutting me off from being heard") that `motion`/`servo`/`headlift` whirs are
  Rex's own machinery and must NOT mute the mic; a motor whine transcribes to junk the
  hallucination filters drop anyway. But the decision only reached `echo_cancel`. The
  mic loop skipped the read for **any** holder of the shared output gate, and
  `_play_gated` — the one effects path that takes that gate — is used by exactly those
  exempt families. So the 2026-07-25 fix never actually reached capture, and a ~1.5 s
  whir went on deafening him for its full length.
- **Why it hits the reply window every time:** Rex re-centres his head when a turn ends,
  which fires a servo accent ~1 s after he stops talking — precisely when the person
  starts answering. All three post-speech gaps in the 13-04-31 session had one: 13:05:27,
  13:06:22, and 13:07:10. The last one swallowed "This is the workshop room" whole — no
  VAD, no segment, no transcript, a 16-second silence, and then the repeat landed.
- **Fix:** `sound_effects.gated_effect_mutes_mic()` publishes the family decision for the
  currently-playing gated effect, and `interaction._effect_allows_listening()` lets the
  loop keep reading when the only gate holder is a non-muting effect. Speech-family
  chirps are voice-like and still mute; `speech_queue.is_speaking()` still always wins;
  the helper fails CLOSED to the old skip on any error.
  `SOUND_EFFECTS_DRIVE_SUPPRESSES_MIC=True` restores the old deafening behavior for the
  whole class. This makes capture CONSISTENT with an already-shipped decision rather
  than making a new one — the tradeoff (a whir now reaches VAD unfiltered, since these
  families deliberately don't set `echo_cancel.set_playing`) is the same one accepted in
  July, and wants a live listen to confirm no junk segments appear.
- Tests: `tests/test_sound_effects.py` (flag set/cleared per family, cleared on
  exception, and the interaction-side gate for whir / chirp / TTS / idle).

### No-effects reply window + one-word own-echo hole (2026-08-06, session 13-20-57)

Rex played a triumphant chirp two seconds after finishing a line, the chirp was captured
and transcribed as "Naturally." — the first word of the line he had just spoken — and he
answered his own echo as a stranger ("who are you?"). Owner's call, and the better fix
than the mic-side gate above: **don't play effects while a reply is expected.**

- **`sound_effects._in_reply_window(family)`** (`SOUND_EFFECTS_REPLY_WINDOW_SECS`, 3.0)
  holds effects once the speech queue is DRAINED and Rex stopped talking within the
  window. This attacks the CAUSE that both 2026-08-06 failures share — an effect firing
  into the gap where the person is answering — rather than the symptom. Scoping matters:
  `is_drained()` is False between the sentences of one reply and while anything is
  queued, so chirps riding his own speech (including `play_for_speech`'s synthesis-gap
  chirp) and reactions to a HUMAN turn are untouched. `motion` is exempt
  (`SOUND_EFFECTS_REPLY_WINDOW_EXEMPT_FAMILIES`) — motor sound is feedback for a move
  the person just asked for, and muting it re-opens the 2026-07-24 complaint. Fails OPEN
  and `force=True` still bypasses, so nothing can be silenced by a bookkeeping error.
- **`OWN_ECHO_MIN_WORDS` (3) let a one-word echo straight through.** The floor exists
  because "yeah"/"okay" are likelier to be the human — true for backchannels, false for
  a distinctive word that is verbatim how Rex JUST opened a line.
  `_looks_like_short_own_echo` now rejects a 1–2 word transcript that whole-word-prefixes
  a line spoken inside the capture seam, excluding `_ECHO_SHORT_COMMON_WORDS`. The
  confident-voiceprint override (`OWN_ECHO_VOICE_OVERRIDE_SCORE`, 0.80) still wins, so a
  human genuinely saying it is safe — today's echo scored 0.407.
- **This is also the voice-signature poisoning fix.** Echo rejection returns UPSTREAM of
  person resolution and `_resolve_anonymous_speaker_slot`, so a rejected echo never
  reaches `voice_signatures`. It had been reaching it: signature id=15 matched the
  phantom at 0.990 and had grown to **14 turns across sessions since 2026-08-02** — Rex
  learning his own AEC residual as a recurring person. A pinned regression test asserts
  the ordering. **Row id=15 was dropped on owner approval 2026-08-06** after it went on
  to match a real human's laugh at 0.863 (next entry); backup at
  `assets/memory/people.db.bak-*`. The other 12 anonymous rows were left alone and all
  sit at 1–2 turns — the runaway growth was unique to the poisoned one, which is the
  cheapest signal to watch for a recurrence: `select id, turns from voice_signatures`,
  anything climbing past a handful of turns with `person_id IS NULL` is suspect.
- **Relationship to the mic-gate change (a09960d):** verified inert here — the `proud`
  chirp is speech-family with `_suppresses_mic=True`, so the mic was skipped during it
  exactly as before; the capture had already begun in the post-TTS seam BEFORE the chirp
  and pulled the chirp in via the rolling buffer. a09960d is kept because it still covers
  the case this window does not: a LOOPING motion whir during a long manoeuvre
  (2026-07-25), which is not a reply window at all.
- **Caveat on that earlier diagnosis:** `mic_skip_output_busy_sfx` counts any non-speech
  gate holder, and the counter name is picked by `speech_queue.is_speaking()` — the
  13-04-31 session logged 0 `mic_skip_rex_speaking` despite ~11.5 s of TTS, so the
  148 figure likely folds in TTS playback and overstates the effects share. The specific
  13:07:10 servo incident stands on its own timeline; the aggregate does not.
- Tests: `tests/test_field_2026_08_06_sfx_echo.py`.

### A laugh is not a stranger (2026-08-06, session 16-23-25)

Bret laughed at Rex's own joke — face recognized, on camera, confident voice 16 s
earlier — and got "Nice laugh, mystery voice—who are you, exactly?".

- **Root cause: a laugh defeats BOTH identity signals, for the same reason.** ECAPA
  embeds SPEECH, so laughter lands ~0.44 on the laugher's own print (below the 0.50
  genuine band); and `vision/active_speaker.py` measures VAD-gated jaw ARTICULATION, so
  it reports `visual_mouth_still` straight through a laugh. Neither is a bug on its own —
  both are speech models being handed something that isn't speech.
- **They compound because of decision ORDER.** In `_voice_primary_face_decision`, the
  mouth-still veto sits ABOVE the `voice_continuity` check, so the confident anchor from
  his turn 16 s earlier was never consulted. (Proof it was live: the very next two turns,
  0.730 and 0.538, both resolved "accepted via voice continuity".) The `short_utterance`
  branch — which already encodes exactly the right rule, "an unscoreable clip is not
  evidence of a stranger", and deliberately carries no mouth-still veto — sits lower
  still and was never reached.
- **Fix:** `_is_non_speech_vocalization(text)` flags a laughter-ONLY transcript
  (whole-transcript token match, ≤6 tokens, so "Haha, that is funny" and "Hannah" are
  normal turns). It (a) exempts the turn from the mouth-still veto, which lets continuity
  resolve it, and (b) adds a last-resort branch for a laugh with NO continuity — e.g. the
  first thing someone does is laugh at his greeting — gated on the voice's own best
  candidate BEING the visible face and the camera not naming anyone else. Always
  `voice_agrees_no_refresh`: folding laughter into a speech voiceprint would corrupt it.
  Kill switch `LAUGH_NOT_A_STRANGER_ENABLED`.
- **The JT hole stays closed.** The veto still fires for short SPOKEN turns — only
  laughter is exempt — so the 2026-08-02 case (JT at ~20 ft cross-matching Bret's print
  at 0.455 while Bret sat silently on camera) is unchanged. Pinned by
  `test_a_short_spoken_turn_is_still_vetoed`.
- **The poisoned signature made it worse:** the laugh matched `voice_signatures` id=15 at
  0.863 — the same row that matched Rex's own echo at 0.990 in session 13-20-57. A row
  that confidently matches BOTH Rex's AEC residual and a human's laughter is not anyone's
  voice; it is a junk non-speech cluster. Dropped on owner approval (previous entry).
- Tests: `tests/test_voice_primary_identity.py::LaughIsNotAStrangerTest` /
  `LaughterDetectionTest`.

### Episodic shared-memory callback reaches the LEAN path (2026-08-08)

The recall Phase 2 "we have history" hook ("I made you laugh", "we played trivia")
was injected only by `llm.assemble_system_prompt` — the CLASSIC prompt. With the
lean brain live as the primary voice, the diary reached replies only on the classic
FALLBACK path, so the flagship episodic-recall beat almost never fired in the field.

- **Fix:** `lean_brain._person_lines` now calls `llm._pick_episodic_callback` and
  renders the same SHARED-MEMORY HOOK line. Deliberately the SAME function, not a
  copy: the probability roll (`EPISODIC_RECALL_PERSON_CALLBACK_PROBABILITY`) and the
  once-per-session dedup set (`llm._episodic_callbacks_used_this_session`) are shared,
  so a memory surfaced on either path can never repeat on the other, and the classic
  path's session-reset (`llm` reset hook) covers both.
- **Reply turns only.** The hook is gated on `user_text` being non-empty —
  `_system_prompt` passes `""` on the directive path (`stream_directive`), whose
  proactive cue owns that turn and must not compete with a memory hook. The idle
  "memory musing" beat already covers proactive episodic surfacing.
- **Ordering note:** the classic path treats this as the lowest-priority hook under
  the one-callback-per-turn budget (fact-confirmation / nostalgia / next-question
  first). The lean path has no such hook stack, so no budget arbitration is needed;
  topic tokens from the live utterance still rank the FITTING memory first.
- Kill switch `LEAN_EPISODIC_CALLBACK_ENABLED` (lean injection only; the master
  `EPISODIC_RECALL_ENABLED` still gates everything inside the picker).
- Tests: `tests/test_lean_episodic_callback.py` (hook renders, directive path never
  rolls, kill switch, exception-safe, shared dedup with the classic path).

### Sound-event awareness — a real classifier for non-speech hearing (2026-08-08)

Rex's non-speech hearing was four energy heuristics (laughter bursts, applause
flatness, scream centroid, sudden-loud spikes). New: `audio/sound_events.py` runs
YAMNet (Google's AudioSet classifier; Apache-2.0 waveform-in ONNX export, ~16MB,
521 classes, ~3-4ms per window on the onnxruntime the face stack already ships)
inside the scene loop and maps classes onto behavior FAMILIES: scream,
glass_break, bang, alarm, siren, baby_cry, doorbell, knock, dog_bark, cat,
laughter, applause.

- **Placement:** called from `scene._analyze_cycle`, so it sits behind the
  existing self-noise gate (`_should_skip_cycle`) — it never hears Rex's own
  TTS/music. Per-class MAX over the window's frames (a 0.5s bang must not be
  diluted by a 2s mean). Per-family thresholds (`SOUND_EVENT_FAMILY_THRESHOLDS`
  over `SOUND_EVENT_DEFAULT_THRESHOLD`) + a per-family cooldown
  (`SOUND_EVENT_FAMILY_COOLDOWN_SECS`, 30s) so a barking dog is ONE event.
- **Publication:** classifier events land in `audio_scene.sound_events` and the
  highest-priority reactable family becomes `last_sound_event`, bumping
  `last_sound_event_seq` — the seq is what lets a REPEAT of the same family
  (dog barks again a minute later) re-fire downstream, where value-change
  detection alone stayed silent forever. Legacy heuristic events deliberately do
  NOT bump seq (they'd re-fire every cycle of a sustained scream); their
  original value-change semantics are preserved, and they remain the working
  fallback whenever the model is missing/broken (module disables itself with
  one warning; nothing else changes). Confident classifier laughter/applause CORROBORATE
  the burst/flatness heuristics (set `laughter_detected`/`applause_detected`)
  rather than adding second paths.
- **Reactions** (`consciousness._step_proactive_reactions`): scream/glass_break/
  bang joined `STARTLE_SOUND_EVENTS` (yelp path, surprise frame —
  `emotion_orchestrator.frame_for_event` learned the two new keys). The other
  families get flavored prompts (`SOUND_EVENT_REACTION_PROMPTS`: doorman bit for
  the doorbell, opinions about organic alarm systems for dog_bark, genuine
  no-bit concern for alarm/baby_cry → "concerned") behind
  `SOUND_AWARENESS_REACTIONS_ENABLED` with a shared 90s cooldown
  (`_last_notable_sound_reaction_at`, consumed only when the trigger is chosen —
  same ack pattern as startle). Laughter deliberately has NO reaction prompt:
  the existing laughter/bow path owns that.
- **Assets:** `setup_assets.py` step 9 downloads `yamnet.onnx` + the official
  class map into `assets/models/yamnet/` (gitignored). No new pip dependencies.
  Class names in `SOUND_EVENT_FAMILY_CLASSES` must match the class map exactly —
  pinned by a config-consistency test; unknown names are skipped with a debug log.
- Kill switches: `SOUND_AWARENESS_ENABLED` (classifier),
  `SOUND_AWARENESS_REACTIONS_ENABLED` (speech). Tests:
  `tests/test_sound_events.py` (24: detector, scene publication, reaction
  branch, config consistency, real-model smoke).
- **Live-tuning note:** thresholds shipped conservative and UNVALIDATED against
  real room audio — the first field runs should watch `[sound_event]` log lines
  for false fires (dishes → bang, TV → everything) before trusting reactions in
  DJ-adjacent rooms.

### Unprompted impressions — famous mentions + self-mock (2026-08-18)

`features/organic_impersonation.py`. Owner idea: Rex had a Jimmy Carter voice on
file the whole time Bret was saying he's going to Plains, and nothing could reach
for it — every performance capability was request-only. Now a reply turn can
CLAIM an impression the way it claims banked callback humor:

- **Trigger A — famous mention.** Deterministic roster scan of the utterance
  against `assets/voices/famous/`: full name, alias slug (fdr/jfk/lbj/ike), or a
  title + surname. A bare surname never fires (Ford, Bush, Carter, Johnson are
  all people you might know). Explicit "impersonate …" phrasings are left to the
  explicit flow (`_EXPLICIT_RE`).
- **Trigger B — self-mock.** The speaker has a captured voice ref (from a past
  "impersonate me"), the social frame allows a roast (`allow_roast` normal/sharp),
  a probability dial passes, and ONE small LLM call both judges the line
  mock-worthy and writes the ≤18-word playback (or answers NONE). Sad /
  vulnerable / health / money / family / work-stress lines are NONE by prompt,
  and the person's boundary terms ride along as hard NONE triggers.
- **It must not SOUND like the requested flow** (owner note): no stall line, no
  thinking chirp. `maybe_claim` (called in `interaction._stream_llm_response`,
  right after the callback claim) starts the script + `local_tts.start_take` in
  the background and returns only a directive telling the reply model NOT to do
  the impression in prose or via the tool. Rex's ordinary ElevenLabs reply covers
  the render. A player thread then waits for, in order: prep done → the reply
  spoken (`note_reply_done()` from the main turn handler) → the clone rendered →
  the floor free (`speech_queue.is_drained()` + `can_proactive_speak(reactive=True)`,
  so heavy-moment / game / DJ / live-speech gates all apply). Only then: bridge
  line in Rex's voice ("Oh — hang on. Jimmy Carter, everybody:") → the take →
  the bow. If the moment never comes inside `IMPERSONATION_ORGANIC_MAX_WAIT_SECS`
  (60; self-mock 35) the bit is dropped SILENTLY and the take released. In
  `--local-tts` the take waits for the reply first (engine is serialized).
- **Discipline:** one bit in flight; `IMPERSONATION_ORGANIC_MIN_GAP_SECS` (600)
  between any two, `..._VOICE_MIN_GAP_SECS` (3600) per famous voice (alias
  symlinks resolve to one key), `..._MAX_PER_SESSION` (4); self-mock
  `IMPERSONATION_SELF_MOCK_MIN_GAP_SECS` (900) per person and
  `..._CONSIDER_PROB` (0.5) before the judge call is spent. An explicit request
  cancels a pending organic bit (`impersonation.perform` → `cancel`), and
  `start_take` would evict its parked take anyway (`Take.is_closed` lets the
  player notice). Kill switches `IMPERSONATION_ORGANIC_ENABLED`,
  `IMPERSONATION_SELF_MOCK_ENABLED`.
- Every fire records a `rex_episodes` row with `detail.trigger`
  (`mention:famous:jimmy-carter` / `self_mock:judged`) and the utterance — the
  first programmatically-true "why did you do that" record, ahead of the general
  decision ledger discussed the same day. Tests: `tests/test_organic_impersonation.py`.
- **Field-tuned 2026-08-19** (first live run, log `djr3x-2026-08-19-19-59-18`,
  Carter fired on the Plains trip — worked, two fixes):
  1. **The script is now a conversation CAMEO, not the requested-flow act.** The
     bit had come back generic ("peanut butter sandwich... shared by a droid!")
     because the standard famous block (droid-borrowed-my-voice framing + random
     angle) dominated. With `context` set, `impersonation._script_prompt` swaps
     that block for a cameo one — the figure BUTTS INTO the live conversation,
     reacting to its specific details, no droid/impression references, no angle —
     and `organic._convo_excerpt` feeds it the last ~6 transcript lines plus the
     triggering utterance instead of the bare utterance.
  2. **Deep playback buffer while the clone engine works.** The ElevenLabs reply
     stuttered audibly through the 16.7 s render (model load + warmup + take are
     Metal+GIL bursts). `local_tts` now keeps an engine-busy counter (load,
     warmup, any generation) and `tts.playback_stream_kwargs()` — the shared
     kwargs for every playback stream — applies an explicit host buffer
     (`AUDIO_PLAYBACK_CLONE_LATENCY_SECS`=1.2, `_CLONE_BLOCKSIZE`=8192, kill
     switch `_CLONE_DEEP_BUFFER_ENABLED`) whenever the engine is busy OR an
     organic impression is pending — the same mechanism as the boot deep buffer,
     which still takes precedence in its window. Tests:
     `tests/test_clone_deep_buffer.py`.
  Still owed live: a self-mock on Bret's captured ref, and confirming the
  stutter is gone on the next organic fire.

### "Is that Max?" — pet-name guess on furry arrivals (2026-08-18)

Owner note: "small furry lifeform" is dumb when Rex knows I have a dog named Max.
`consciousness._pet_name_guess_line(species)` now runs first in the furry-arrival
branch of `_animal_reaction_frame_and_line`: it walks `_pet_owner_candidates`
(visible known people → `get_recent_engagement` → `people.recently_seen_people`,
DB `last_seen` inside `ANIMAL_PET_NAME_GUESS_RECENT_SECS`=900, restart-proof) and
the first owner with pets wins. `memory.facts.get_pets(person_id)` reads the pet
facts back as `[{name, species}]` regardless of the key the extractor minted
(`dog`, `dog_name_2`, `pet_name`, "a cat named Pixel"; ages/conditions are not
pets). Same-species pets → `ANIMAL_PET_GUESS_LINES` / `_TWO_LINES` ("Bret, is
that Max or Toby?"); species mismatch (detector flip-flops dog/cat) still asks
but says so (`_MISMATCH_LINES`); nobody recent with a named pet → the old pool.
The guess is remembered per species (`_animal_guessed_pet`, cleared with the
other animal state) so return remarks say "Hi again, Max. Probably." Kill switch
`ANIMAL_PET_NAME_GUESS_ENABLED`. Tests: `tests/test_animal_pet_name_guess.py`
(17); the two furry-pool tests in `test_animal_returns` /
`test_audio_and_conversation_gating` hold the guess off because it reads the
REAL people DB (on the robot Mac it finds Bret's dogs and swaps the line —
which is how the wiring was first confirmed).

### Decision ledger — "why did you do that?" answered from the record (2026-08-18)

`intelligence/decision_ledger.py`. Owner note: the model invented reasons that
were plausible and false. Now the sites that already KNOW their reason write a
plain-words, first-person `why` into a session ring (`record(kind, why, said=,
detail=)`), and when the utterance looks like a why-question
(`looks_like_why_question`: "why did/'d you…", "how come you…", "what made
you…", "what was that about", "who were you looking for"…) the reply gets a
directive with the newest entries and their ages, plus the order: **answer from
this record; past its edge say honestly you're not sure — never invent a
mechanism**; opinion/preference questions answer normally. Deliberately
minimal (owner: only what's feasible and easy):

- Instrumented: every accepted governor line (`speech_engine.speak_async` /
  `generate_and_speak` → `why_for_purpose(purpose, label)`, a purpose→phrase
  table with a humanized fallback); every interaction-side proactive line
  (`_speak_proactive(..., why=)` on completion, label→phrase table); the lean
  impulse (`_lean_impulse_why(kind, quiet, long_silence)` — which cue won and
  how long the room was quiet); the reply frame each turn (`_reply_frame_why`:
  purpose / comedy stance / roast level / banked callback — recorded AFTER the
  why-directive is built so the current turn's frame isn't in its own answer;
  capped to the two newest in the directive so it can't crowd out a turn or a
  bit); the pet-name guess; an unprompted impression; an idle head wander; a
  speaker gaze search STARTING (`_record_face_tracking_state` edge); a flinch
  retreat. Not instrumented: micro gaze aversions, come-here/explore internals,
  emotion orchestrator, mood — those get "honestly, I'm not sure" (mood is
  already in the prompt via `rex_mood.prompt_lines`).
- Kill switch `DECISION_LEDGER_ENABLED`; `DECISION_LEDGER_WHY_WINDOW_SECS` (240)
  / `_WHY_LIMIT` (6). Grep `[decision_ledger]` in a session log to see what he
  had on record when asked. Tests: `tests/test_decision_ledger.py` (20).

### Jeopardy live-game batch — five fixes from one session log (2026-08-25)

Session `logs/djr3x-2026-08-25-18-41-13.log` (Bret + PJ playing Jeopardy)
surfaced five distinct failures; each is fixed at its own layer:

- **Theme front-clipped answers.** With `JEOPARDY_PLAY_THINKING_THEME` on, the
  theme starts right after the clue — exactly when a fast player answers. The
  game-audio barge-in path stopped the theme but RESET `speech_start` to now
  and the theme's queue done-callback restamped the capture floor at the
  theme's end, so words spoken under it were discarded ("What is a moon" →
  HEARD "As a moon"; "hydrogen and helium" → "Lithium" — the tail of *helium*).
  On hardware AEC the buffer under the theme is clean (instrumental residual
  can't transcribe): the interrupt now keeps the VAD onset and pins the floor
  via the one-shot `interaction._game_barge_floor_at`, honored in
  `_speech_capture_secs`. Software-suppression machines keep the old reset.
  Kill switch `GAME_BARGE_KEEP_ONSET_ENABLED`. Tests:
  `tests/test_front_clip_fast_reply.py::GameBargeFloorOverrideTests`.
- **Timeout stole the turn mid-answer (twice).** The 12s answer timer fired
  while the player was speaking ("Floral" at the very second of the beeper);
  the rebound advanced `current_player_idx`, so the in-flight answer graded
  for the NEXT contestant ("$1000 to Bret", "$400 to Bret" off "What is
  Nike"). Two guards in `features/games.py`: `_jeopardy_timeout_fired` defers
  (`JEOPARDY_TIMEOUT_SPEECH_GRACE_SECS`, same token re-arm) while
  `situation.assessor` reports speech/turn in flight; and a timeout rebound
  records `timeout_rebound` — an answer that lands while the rebound
  announcement is still being delivered (`awaiting_prompt_delivery`) is graded
  for the timed-out player, the queued announcement is dropped by tag, and
  the post-timeout waiter thread stands down via the `rebound_at` token.
  Grace closes when `_jeopardy_arm_timeout` pops the delivery flag. Tests:
  `tests/test_jeopardy_answers.py::TimeoutGraceTest`.
- **Base wandered/turned away mid-game.** `motion_agency` social lanes (radar
  orient, comfort realign, idle wander) read seated players + think-silence as
  "nobody here, go look": radar orient chased a rear return (+57°), realign
  turned +45° under a parked neck. New gate in `_step_inner` holds ALL social
  lanes while `features.games.is_active()` (`MOTION_HOLD_DURING_GAMES`); the
  flinch reflex and explicit come-here still run. Tests:
  `tests/test_motion_agency.py::GameHoldTest`.
- **"COMBINED STATE ABBREV." read as "abreev".** `jeopardy.speak_category`
  expands a conservative map of unambiguous dataset abbreviations
  (ABBREV./MISC./GOVT./NATL./…) and drops the trailing period — SPEECH ONLY:
  the GUI board, `snapshot()`, and the selection fuzzy-matcher keep raw names.
  Applied in `format_categories`, `format_board_readout`, and the games.py
  clue/rebound/repeat announce sites. Ambiguous tokens (LIT, PRES) stay raw on
  purpose. Tests: `tests/test_jeopardy_answers.py::SpeakCategoryTest`.
- **Category reminder fatigue curve (owner call, same session).** The per-turn
  "Remaining categories: …" read-back is great early game, tiresome once
  everyone knows the board. `_jeopardy_categories_reminder` now counts scoring
  turns (`categories_reminder_reads`, reset each round by
  `_jeopardy_load_round`): the first `JEOPARDY_CATEGORIES_REMINDER_FULL_READS`
  (4) speak every time, then only every
  `JEOPARDY_CATEGORIES_REMINDER_EVERY`-th (3; ≤0 = never again that round).
  The GUI mute path does not consume reads, and an explicit "what are the
  categories?" is always answered in full via `_jeopardy_board_text`. Tests:
  `tests/test_jeopardy_answers.py::CategoriesReminderCadenceTest`.
- **The 19:08 "crash": a CoreAudio device wedge, and a 64s escalation.** At
  19:07:58 the theme barge-in `sd.stop()` landed while two `motion_turning`
  sound effects were in flight (the base was wandering mid-game — now
  prevented by the motion hold above); CoreAudio wedged, mic callbacks died,
  and the output gate sat held by 'sound-effects' for 34s+. The feffb8c
  supervisor-restart escalation DID fire — but only after
  `AUDIO_STALL_FATAL_SECS` (60s) across 11 reopen attempts that ALL wedged:
  that was the felt "minute of nothingness" where even "shut down" was
  ignored. New fast path: `stream._wedged_reopen_streak` counts CONSECUTIVE
  reopen attempts ending in a wedge signature (worker stuck past its budget,
  or lock still held by a stuck predecessor);
  `AUDIO_STALL_FATAL_WEDGED_REOPENS` (4) such attempts escalate immediately
  (~25s). A plain reopen failure (mic unplugged, enumerating) resets the
  streak and keeps the patient clock, so a replugged mic still recovers
  in-process. Tests: `tests/test_stream_watchdog.py` (wedge-streak +
  plain-failure cases).
- **Mid-game board Q&A + LLM fallback (owner ask, same session).** New
  deterministic lanes in `features/jeopardy.py`, answered in BOTH phases via
  `games._jeopardy_answer_board_question` (selecting: answer + re-prompt;
  during a clue: only after is_correct AND the judge say no — answer, then
  `_jeopardy_repeat_clue_reply`): `category_board_query` ("what's left /
  what squares are free / what values are open / how much is left / anything
  left in X", "what does X have left" — 14 phrasings pinned in tests; empty
  category → "cleaned out", unknown → full readout), `value_availability_query`
  ("is the $400 still there in history?" — checked BEFORE parse_selection
  because the value-wins pick rule used to CONSUME the square; answers
  yes/gone, and with no category names where that value is still live),
  `is_score_request` / `is_turn_request` ("what's the score", "who's winning",
  "whose turn/pick") — a score question during a live clue used to grade as a
  wrong answer and deduct. Anything question-shaped
  (`jeopardy.looks_like_question`) the lanes miss, carrying NO dollar value,
  gets `_jeopardy_board_question_llm`: one `_rex_respond` call with the real
  remaining board + scores + whose turn in context ("answer from THIS data
  only"), selecting phase only (`JEOPARDY_BOARD_QA_LLM_FALLBACK_ENABLED`).
  Also `interaction._GAME_STOP_INTENT_RE`: start-anchored natural stop shapes
  ("let's stop playing this game", "no more games", "we're done playing") now
  escape to stop_game — both failed live at 19:04 and the game played on.
  Tests: `tests/test_jeopardy_answers.py::BoardQuestionLanesTest` /
  `BoardQuestionHandlerTest`,
  `tests/test_regex_routing_guards.py::GameStopIntentTest`.
- **Stop-confirmation guard (owner ask, same session).** "Stop playing"
  mid-game now asks "But we're having so much fun, are you sure you want to
  end the game?" and only an affirmative (or repeating the stop demand)
  actually ends it. `games.request_stop_confirmation()` arms a one-shot
  `stop_confirm_at` in `_game_state` and FREEZES a live Jeopardy answer clock
  (a "Time's up" over the exchange would steal the paused turn);
  `games.resolve_stop_confirmation(text, pid, stop_shaped=)` returns
  ("stop", closing_line) / ("resume", line — a live clue is re-read with a
  fresh window, selecting re-prompts the picker) / ("pass", None — any other
  reply drops the ask and grades normally). Wired via
  `interaction._game_stop_confirmation_response` at BOTH mid-game claim sites
  (the pair that drifted before), path label `game.stop_confirmation`.
  "Shut down"/sleep escapes are never gated. Kill switch
  `GAME_STOP_CONFIRM_ENABLED`, TTL `GAME_STOP_CONFIRM_WINDOW_SECS` (45).
  Tests: `tests/test_jeopardy_answers.py::StopConfirmationTest`,
  `tests/test_regex_routing_guards.py::GameStopConfirmationGuardTest`.
- **Theme/Daily-Double clip caps (owner questions, same session).** The
  thinking theme died at 6s of the 12s answer window: the clip on disk is
  ~31s and playback truncates at `JEOPARDY_THEME_MAX_SECS` — raised to 12.0
  to match `JEOPARDY_ANSWER_TIMEOUT_SECS` so the music runs to the beeper.
  The Daily Double sting (queued AHEAD of the announcement) is 12.6s on disk
  — new `JEOPARDY_DAILY_DOUBLE_MAX_SECS` (6.0) caps it via the same
  speech_queue clip-cap table. DD squares themselves: predetermined at board
  build (real air-date positions from the dataset; if the sampled six
  categories carry none, one random higher-value square is marked).
- **Round/wager batch (owner: "fix the rest", same session).** Six changes:
  (1) *Pending category*: a FAILED pick that named a category stashes it
  (`pending_category`, consumed by the next parse) so the bare value that
  follows completes THAT category — 18:50 field case: "Pop culture for 300"
  rejected, then bare "400" picked the last PLAYED category instead.
  (2) *Score cadence*: `_jeopardy_score_announcement` — the answerer's new
  total normally ("That puts Bret at $1200"), full scoreboard every
  `JEOPARDY_SCOREBOARD_EVERY` (4) scoring events; counter resets per round;
  round transitions / finish / "what's the score?" always full. One 22s
  response in the log was mostly scoreboard.
  (3) *Daily Double wagers* (`JEOPARDY_DD_WAGER_ENABLED`): a DD square goes to
  phase `awaiting_wager` — sting, "you're at $X, wager $5 to $MAX" (max =
  max(score, 1000×round)); `jeopardy.parse_wager` handles digits, number
  words ("fifteen hundred", "a thousand"), "everything"/"true daily double",
  "minimum"; out-of-range re-asks with the rails; then the clue reads for the
  wager. NO rebound on a DD (show rules — gated at all three rebound sites).
  False restores flat auto-double.
  (4) *Final Jeopardy* (`JEOPARDY_FINAL_ENABLED`): auto after Double Jeopardy
  completes, or by voice. `jeopardy.load_final_clues()` — the ~364 real
  round-3 clues the loader used to filter out. Phases `final_wager` (lowest
  score first; ≤$0 players "ride along" at $0 but still answer) →
  `final_answer` (clue + the real 30.5s think music via after-response clip
  `final_theme`, capped `JEOPARDY_FINAL_THINK_MAX_SECS`; answers collected in
  the same order, graded silently by the shared `_jeopardy_grade` ladder) →
  reveal: wagers settled, winner crowned (ties handled), outro, game ends.
  (5) *Round jumping*: `jeopardy.round_jump_request` — "next round"/"double
  jeopardy"/"new categories" → deal round 2 (scores kept); from round 2, or
  "final jeopardy" from anywhere → begin Final. A dollar value in the
  utterance is a pick, never a jump. Once per round at ≤
  `JEOPARDY_ROUND_JUMP_OFFER_REMAINING` (15) squares, the pick prompt
  mentions the jump. (6) `_clear_game` cancels a live answer timer.
  `_jeopardy_grade(text, clue)` now holds the single grading ladder
  (matcher → hedge-residual promote → LLM judge) used by the live board and
  Final. Stop-guard resume lines cover the new phases. Tests:
  `tests/test_jeopardy_rounds.py` (32).
- **Passive voiceprint growth + impersonation-capture rework (owner spec
  2026-08-26).** Forensics first: Bret's stored clone ref was LITERALLY the
  words "impersonate me" (5.18s buffer, ~1.5s speech — the padded segment
  defeated the 4s minimum, and a repeated request while the slot was open
  became the take); PJ's "Mary" ref was verified PJ by the owner's ear yet
  embeds 0.795 vs Bret (true acoustic twins through this mic chain — JT's ref
  scores Bret only 0.371, so no systematic capture bias); a controlled clone
  test (same sentence from JT/PJ/rex refs) proved the cloner tracks its ref —
  three mutually distinct outputs (cosines 0.04-0.18) — but people-ref
  fidelity is weak (~0.35-0.43 self-similarity): quality tracks the REF
  (rex = 19.5s clean 44.1kHz studio; people = ~8s padded far-field 16kHz).
  Changes: (1) `interaction._maybe_passive_voice_enroll` — no line-reading:
  when exactly one known face is visible, nobody unknown, nobody else
  seen/heard within `PASSIVE_VOICE_ENROLL_SOLO_WINDOW_SECS`, and the turn has
  ≥2s VOICED audio / ≥4 words, the turn silently becomes a voiceprint for the
  visible person — voiceless people enroll despite cross-matches up to the
  0.75 confident bar (twin band is 0.55-0.80), thin prints grow to
  `PASSIVE_VOICE_PRINT_TARGET` (4) while foreign <0.60 and self <0.80;
  session cap 3, spacing 90s, logs `[passive_enroll]`. (2) Capture guards:
  `_voiced_duration_secs` (speech frames, not buffer length) now backs BOTH
  the voice-sample and impersonation minimums
  (`IMPERSONATION_CAPTURE_MIN_VOICED_SECS` 6.0); a repeated impersonation
  REQUEST is never a take; an off-script take from the right person gets one
  nudge back to the line; `impersonation._trim_silence` strips pad/room-tone
  before a ref is saved; capture lines lengthened to ~15s spoken. (3)
  "Impersonate ME" retargets to the solo visible face when the voice guess
  disagrees (PJ asked, attribution said Bret, Bret's ref performed). Cleanup
  done in place: Bret's junk ref deleted (backups + forensic clone tests in
  `assets/voices/backups/`), PJ's print rebuilt from the verified clip
  (people.db person 7, 1 fresh ECAPA row). Remaining known gap: PJ↔Bret
  genuinely overlap at the embedding level; depth (passive growth) is the
  mitigation, pair-aware margins the possible next step. Tests:
  `tests/test_passive_voice_enroll.py` (28).
- **PJ's voiceprint AND impersonation ref were junk/contaminated.** His print
  enrolled from a ~1s "Hey Rex." (worthless under ECAPA short-turn behavior →
  the whole game heard PJ as "Bret Benziger" or unknown_voice_N), and his
  impersonation ref (`assets/voices/people/7.wav`, "Mary had a little lamb")
  voice-scored **Bret 0.784** at capture — the "PJ" clone sounded like Bret.
  Guards: voice-sample enrollment now requires `VOICE_SAMPLE_MIN_SECS` /
  `VOICE_SAMPLE_MIN_WORDS` (re-asks for a full sentence; the ask itself now
  says "full sentence"), and an impersonation capture take that voice-matches
  a DIFFERENT enrolled, recently-VISIBLE person ≥
  `IMPERSONATION_CAPTURE_FOREIGN_VOICE_BAR` (0.75) gets ONE solo-retake ask
  (visibility requirement keeps the 2026-07-23 junk-twin capture shape
  working; the second take always saves). OWNER ACTIONS still owed: delete
  `assets/voices/people/7.{wav,txt,json}` and re-capture PJ's impersonation
  ref; re-enroll PJ's voiceprint with a long solo line
  (`tools/test_voice_id.py --enroll "PJ Thomas" --replace`). Tests:
  `tests/test_voiceless_face_wins.py` (min-length),
  `tests/test_impersonation.py::CaptureConsumerTest` (foreign-voice retake).

## Likely Future Work

- **OPEN (instrumented, awaiting data): do sound effects mute the mic mid-reply?** The
  2026-08-06 work fixed the AEC hold and the capture floor, but the ACTIVE-loop skip
  counters were added specifically to test the remaining hypothesis — a servo chirp is
  audio output, so the mic is not read while one plays, and effect rate was ~1.8×
  higher in the runs that lost utterances. Read `[capture] session summary` from the
  next field session (`mic_skip_output_busy_sfx` vs `mic_skip_rex_speaking`) BEFORE
  changing anything. If sfx are confirmed, the fix is ducking them out of the human's
  reply window, not disabling them.
- **Tool router Phases 3-4** (`docs/tool_router_scope.md`): humor/character actions are
  still shadow-only (their fast lanes work); `motion` is unchanged. Phase 4 cleanup
  merges the per-action schemas in `tool_router._TOOL_DEFS` into `ActionSpec` itself.
  The module docstring still describes Phase 0 and should be rewritten at that point.
- Motion Phase 1: wire the real drive base (BTS7960 motor driver + Hall encoders + per-wheel PID + 5× VL53L0X ToF) and fill the `hal.cpp` `MOTION_HW_PRESENT` driver sections; add the Bluetooth-gamepad manual override (`docs/motion_system.md` §11, §17). Known Phase-1 fidelity gaps: a pure `turn` (spin) is not yet ToF-gated (no side sensors), and the stub plant carries residual velocity from a finished finite command into the next one.
- Decide whether the streaming answer path is sufficient latency cover on its own, or whether to re-enable (and tune) the slow-path ack / latency filler for the slowest paths.
- Deeper conversation steering: detect a topic shift semantically (not just explicit "I like / I'm building X") and update/expire the active interest accordingly; today a new subject the user is clearly engaged in but doesn't name in an interest form is not picked up.
- Add directional audio support for stereo ReSpeaker Lite input.
- Improve group turn triage for crosstalk and ambiguous addressees.
- Continue reducing OpenAI calls on common conversational paths.
- Expand tests around identity introduction, GUI text mode, no-audio mode, and multi-speaker ambiguity.
