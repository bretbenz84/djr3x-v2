# DJ-R3X v2

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It combines speech recognition, text-to-speech, camera awareness, face and voice memory, conversational social behaviors, hardware control, music playback, and verbal games into one interactive companion system.

The project is built for live, in-room use: Rex can recognize people, remember details across sessions, react to arrivals, handle introductions, play games like Jeopardy, answer visual prompts, and drive connected LEDs and servos when the hardware is attached.

## Features

- Wake-word and always-listening conversation flow
- Local Whisper transcription with OpenAI fallback support
- ElevenLabs TTS with cached speech output — ElevenLabs is Rex's "true voice" and the default
- On-device TTS voice clone (mlx-audio Qwen3-TTS) as a second backend: run Rex's whole voice offline with the `--local-tts` flag, and — always on — **automatic fallback** to the local voice whenever ElevenLabs is unreachable, errors, or runs out of credits, so Rex never goes silent (toggle with `LOCAL_TTS_FALLBACK_ENABLED`). The ~2.9 GB model is downloaded by `setup_assets.py`
- Voice impersonations for fun — "do an impersonation of me / of Jimmy Carter": Rex clones a voice and delivers a short, affectionate parody in it. For someone he knows, he captures a quick voice sample (asks you to repeat a line) and mines that person's memory for the material — while hard-excluding any boundaries or sensitive topics; for a famous person, drop a clip + transcript in `assets/voices/famous/` (toggle with `IMPERSONATION_ENABLED`; needs the local TTS model)
- Camera-based scene, face, appearance, and animal awareness — face detection/recognition runs on InsightFace (SCRFD detector + ArcFace 512-dim embeddings via ONNX Runtime; `FACE_BACKEND=dlib` restores the legacy stack); local animal/object detection runs on RF-DETR nano (`OBJECT_DETECTOR_BACKEND=mediapipe` restores EfficientDet)
- Voice and face enrollment for known people — speaker ID runs on ECAPA-TDNN embeddings (SpeechBrain, 192-dim; `VOICE_EMBEDDER=resemblyzer` restores the legacy stack)
- Persistent memory database for people, relationships, preferences, and events (`people.db`)
- Rex's own first-person episodic memory (`rex.db`) — a timestamped log of his experiences (people seen, scenes observed, things he did, per-session conversation summaries)
- Social intelligence layers for repairs, boundaries, grief, celebrations, callbacks, and group discretion
- First-meeting curiosity — when Rex meets someone brand new he runs a short, in-character "getting to know you" burst: research-backed baseline questions with quick witty reactions between them and the occasional self-reveal, building a useful profile before settling into free conversation. It's bounded and backs off the moment you're not into it (toggle with `ONBOARDING_ENABLED`)
- Mood-driven body language — Rex's posture (head lift/tilt, visor openness, breathing, idle gestures) reflects a sustained "body mood" that shifts when he's complimented, insulted, or amused, riding on top of face-tracking
- A wandering attention of his own — when the conversation lulls he'll stop staring, glance around the room, then look back and sometimes re-greet, so he doesn't feel locked to a fixed stare
- Bored environmental snark — left idle, he looks around and invents in-character jabs about the room he actually sees: complaints about how dull it is, faux-clueless questions about objects ("what's that black chair for?"), digs at the clutter, snobby art opinions, or pleas to be taken somewhere with more life forms
- Visual curiosity — when an engaged conversation goes quiet, Rex takes a fresh look and asks one grounded question about something he can actually see right now, instead of generic small talk (toggle with `VISUAL_CURIOSITY_ENABLED`)
- Web search for current info — when a question needs live data (you ask him to "look it up", or he decides on his own that it needs the latest), Rex says a quick stall line, searches the web via OpenAI's hosted `web_search` tool, and answers in character; trigger phrases and stall lines are editable in your user config (toggle with `WEB_SEARCH_ENABLED`)
- Waves back — when the camera sees someone wave at him (MediaPipe pose gesture), Rex returns the wave with his arm and a short warm line, mirroring the speed of your wave the way you'd wave back across a room. Keep waving and it turns into a bit — his responses escalate (warm greeting → progressively terser → a crack about the repetition → eventually he just ignores you). Debounced so a single wave gets a single wave-back (toggle with `WAVE_BACK_ENABLED`)
- Live pose wireframe — the GUI dashboard's camera preview overlays each detected body as a real-time skeleton (MediaPipe pose landmarks, up to `POSE_MAX_PEOPLE` people) on top of the per-person face boxes, so you can see what Rex's body-tracking sees (toggle with `GUI_POSE_WIREFRAME_ENABLED`)
- A sense of place — visual place recognition (MobileCLIP-S2) that recognizes which enrolled room Rex is in and publishes a debounced belief to `world_state.current_place`. Rooms are taught by name, recognition is stable (temporal hysteresis plus a motion gate so he can't "change rooms" without moving), and the whole feature no-ops cleanly if the encoder isn't available (toggle with `PLACE_RECOGNITION_ENABLED`)
- Servo and LED hardware hooks for a physical droid body
- Voice-driven motion — an optional ESP32 drive base lets Rex physically roll around the room on command ("turn left", "back up", "come here", "halt"), avoiding obstacles and people with onboard sensors. An explicit "come here" / "come over here" / "come to me" starts a bounded person search: Rex rotates until face tracking finds someone, turns his chassis toward them, then approaches until the front ToF reaches 1 metre; furniture or another obstacle stops him first. The ESP32 owns the real-time, fail-safe motor loop while the Mac sends high-level commands
- A back-off reflex — when Rex is parked and someone steps right up into his face, the front 8×8 ToF sensor feels the approach and he reflexively edges backward, the way an animal gives itself room. He only retreats as far as the rear ToF sensors say is safe — stopping short of the wall and simply holding his ground when he's cornered (toggle with `MOTION_FLINCH_ENABLED`; needs the drive base)
- Music controls and verbal games: I Spy, 20 Questions, themed five-question Trivia rounds, Jeopardy, and Word Association

See [CONTEXT.md](CONTEXT.md) for more detailed project features, architecture notes, hardware mappings, and behavior design.

## Requirements

- macOS, preferably Apple Silicon
- Terminal access
- Git
- Internet access for setup and model downloads
- OpenAI API key
- ElevenLabs API key
- Optional hardware:
  - Pololu Maestro servo controller
  - Head and chest LED controllers using Arduino Nano or Arduino Uno variants
  - ESP32 motion controller (drive base) with motor driver, encoders, and Time-of-Flight sensors
  - Camera and microphone

The macOS setup script installs Homebrew dependencies, Ollama, pyenv, Python 3.11.9, the virtual environment, Python packages, config templates, assets, models, and database setup.

## How To Install

Clone the repository:

```bash
git clone https://github.com/bretbenz84/djr3x-v2.git
cd djr3x-v2
```

Make the macOS setup script executable:

```bash
chmod +x setup_macos.sh
```

Run the setup script:

```bash
./setup_macos.sh
```

The setup script creates local config files from templates and prompts for local setup choices:

- `apikeys.py` for OpenAI and ElevenLabs credentials
- `.env` for machine-specific camera, audio, and hardware device paths
- `user_config.py` for user-facing overrides (AI models, personality, location, feature toggles, timeouts), copied from `user_config.example.py`
- Optional replacement of `ELEVENLABS_VOICE_ID` in `config.py`
- Optional guided droid hardware setup for the chest Arduino, head LED Arduino, Pololu Maestro, and ESP32 motion base
- Arduino CLI, Arduino AVR core, and FastLED setup for uploading the included LED firmware
- For the motion base: auto-detects the ESP32 (by talking to its firmware over USB), installs the ESP32 core + ArduinoJson on demand, and can flash the motion firmware for you
- Ollama plus `qwen2.5:1.5b` for local low-latency classifier/shaping work

You can leave a prompt blank to keep the current value, or edit the generated files manually later.

## How To Run

Activate the virtual environment:

```bash
source venv/bin/activate
```

Start DJ-R3X:

```bash
python main.py
```

Startup flags:

| Flag | Purpose |
| --- | --- |
| `-gui`, `--gui` | Open the optional PySide6 GUI dashboard for this run. |
| `-jeopardy`, `--jeopardy` | Start directly in Jeopardy mode and skip startup introductions. |
| `-noaudio`, `--noaudio`, `--no-audio` | Disable microphone capture, wake word listening, audio output, and ElevenLabs TTS calls. Responses are written as text to the conversation log and GUI. |
| `-noservos`, `--noservos`, `--no-servos` | Disable the Pololu Maestro servo controller entirely for this run, even when `MAESTRO_PORT` is configured. All servo motion (head tracking, gestures, animations) is skipped; everything else runs normally. |
| `-local-tts`, `--local-tts` | Use the on-device Qwen3-TTS voice clone instead of ElevenLabs for this run (no ElevenLabs calls at all). Runs fully offline; the model is preloaded at startup. See the on-device-TTS feature note below. |

Open the optional GUI dashboard:

```bash
python main.py --gui
```

Run the GUI as a text-only input/output interface:

```bash
python main.py --gui --noaudio
```

Flags can be combined:

```bash
python main.py --gui --noaudio --jeopardy
```

At startup, DJ-R3X preloads the local Ollama `qwen2.5:1.5b` model before accepting input and keeps it loaded for the run.

You need to activate the virtual environment in every new terminal session before running project commands.

### Always-on "wake up Rex" launcher (optional)

Instead of starting `main.py` by hand, you can have macOS stay quietly ready and launch the robot by voice. A tiny LaunchAgent (`rex_supervisor.py`) listens only for **"wake up Rex"** and launches the full controller on demand (it starts `main.py --gui`, so the dashboard opens on every wake); **"shut down"** powers it back down while the listener keeps running. Install with `scripts/install_supervisor.sh` (the setup script also offers this). See **[docs/supervisor.md](docs/supervisor.md)** for how it works and how the single-instance lock prevents a double-launch (including when Rex is asleep).

The same installer adds a **menu bar battery meter** (`tools/rex_battery_menubar.py`) when `MOTION_ESP32_PORT` is set: the drive base's charge, voltage, and current stay visible in the macOS menu bar even while the robot is off, by passively reading the ESP32's always-on telemetry stream. It releases the serial port automatically whenever `main.py` is running (same flock the supervisor uses for the mic) and reclaims it when Rex shuts down. A **"Set Battery to 100%"** menu item lets you sync the firmware's charge gauge the moment your charger's taper current says the pack is full.

It also adds a **"Servo Control" menu bar console** (`tools/rex_servo_menubar.py`) when `MAESTRO_PORT` is set: a dropdown with live sliders for all 8 Maestro servo channels (labelled with the current position in microseconds, initialized from the board's actual positions) plus a **"Restart Pololu"** action that sends the Maestro's go-home command. Sliders command the servos directly over the same wire protocol the robot uses. Like the battery meter, it releases the serial port automatically whenever `main.py` is running and reclaims it when Rex shuts down.

On the physical robot, the supervisor also keeps a clean `main` checkout current
with `origin/main`: it checks at supervisor startup, every four hours, and again
immediately before launching `main.py`. A running controller is never updated
underneath itself, failures fall back to the installed version, and no updater
state files are created. See [docs/supervisor.md](docs/supervisor.md) for the
safety rules and configuration switches.

## Configuration

User-tunable defaults live in [config.py](config.py). API keys should stay in `apikeys.py`, and host-specific hardware paths plus build-specific servo limit overrides should stay in `.env`; both are intentionally excluded from git.

The settings most people actually want to change — AI model selection, Rex's personality dials and base prompt, location/venue, feature on/off switches, and key timeouts — are surfaced in [user_config.example.py](user_config.example.py), a heavily commented template grouped by topic. The setup script copies it to `user_config.py` (gitignored), which `config.py` imports last so its values win over the defaults. (The script only creates `user_config.py` when it doesn't already exist, so when new settings are added to the template — e.g. the web-search options — copy those sections over manually to expose them.) Every setting ships commented out at its current default: uncomment a line to override it, or re-comment/delete it to fall back to the `config.py` default. A missing `user_config.py` is harmless — `config.py`'s defaults are used unchanged, so `from config import X` keeps working everywhere. Values computed from a base (e.g. `ACTION_ROUTER_MODEL` follows `LLM_MODEL`) are re-derived after the override so changing the base propagates. Deeper internal tuning (CV thresholds, cooldowns, scoring) intentionally stays in `config.py`.

Useful setup checks:

```bash
ls /dev/tty.usb*
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

The setup script can walk you through serial device detection for Maestro and Arduino hardware. For microphone setup, prefer `AUDIO_DEVICE_NAME` in `.env` so CoreAudio index changes do not break listening after reboot or replug. For camera setup, update `CAMERA_INDEX` or `CAMERA_DEVICE_NAME` in `.env`.

## Servo Safety

Do not connect a Pololu Maestro to live servos until the servo limits have been configured in the Maestro Control Center app. Set safe minimum and maximum values there first, write those values down, and then store the matching servo limit overrides in `.env` using the setup script or the `SERVO_<NAME>_MIN_US` / `SERVO_<NAME>_MAX_US` keys from [.env.example](.env.example).

Servo limits in `.env` use the Maestro app's microsecond values, such as `496 - 2496`; `config.py` converts them to Pololu quarter-microseconds at runtime. Non-numeric values, values outside `300 - 3000`, or a min without a matching max stop startup rather than silently falling back. For servo safety keys, the project `.env` file takes priority over inherited shell environment variables.

Connecting the Maestro before limits are programmed can drive a servo past its safe travel range and damage the mechanism.

## Motion Base (optional)

An optional ESP32-controlled drive base lets Rex physically move around a room on spoken command while avoiding obstacles. The ESP32 runs a real-time, fail-safe motor loop (PID speed control, Time-of-Flight obstacle/cliff stop, heartbeat watchdog) and the Mac sends high-level commands (`turn`, `move`, `come`, `stop`) over USB serial. Spoken intents like "turn left", "back up", "come here", and "halt" route through the normal conversation pipeline to the base; "stop" only steers the base while it is actually moving, so it never hijacks stop-music/stop-game.

- **Wire contract:** [docs/motion_protocol.md](docs/motion_protocol.md). **Feature spec & wiring:** [docs/motion_system.md](docs/motion_system.md).
- **Firmware:** [firmware/djr3x_motion](firmware/djr3x_motion/) (Arduino sketch for the ESP32). The current Phase 0 build runs the full protocol against a stubbed hardware layer, so it works on a bare ESP32 with nothing wired — flip `MOTION_HW_PRESENT` in `hal.h` to `1` as motors, encoders, and sensors are added.
- **Enable it:** set `MOTION_ESP32_PORT` in `.env` (the setup script can auto-detect and set this). Motion is fully disabled — with zero change to the rest of Rex's behavior — until that port is set.
- **No base attached?** If you give Rex a drive command ("turn left", "move forward", "come here") while the ESP32 isn't connected, he refuses out loud with a pre-canned in-character quip instead of silently ignoring it — there are no wheels to move, so he says so (toggle/edit with `MOTION_NO_BASE_DENIAL_ENABLED` / `MOTION_NO_BASE_DENIAL_LINES`). A bare "stop"/"halt" is unaffected.
- **Bring-up test:** `venv/bin/python firmware/tools/motion_serial_smoketest.py` exercises the whole protocol against a connected board.

> Safety: the ESP32 stops the base on its own (obstacle/cliff/lost-comms) independent of the Mac. Do not attach motor power until the base has been bench-tested with wheels off the ground.

## Project Layout

```text
audio/          Speech input, VAD, transcription, TTS, playback, and audio scene logic
awareness/      Time, holidays, interoception, and background awareness systems
features/       Games, music, commandable behaviors, and interactive features
hardware/       Servo, LED, and motion-base serial integrations
intelligence/   Conversation, memory, LLM prompting, empathy, social behavior, and motion control
vision/         Camera, face recognition (InsightFace SCRFD+ArcFace), scene analysis, and image utilities
firmware/       ESP32 motion-controller firmware (Arduino sketch) + host serial tools
assets/         Models, audio, game assets, memory database, and cached generated assets
logs/           Runtime logs
```

## Notes

- The program can run with missing droid hardware, but servo and LED features will be disabled until the configured devices are connected.
- Face recognition uses InsightFace by default (`config.FACE_BACKEND`). Its models (~190MB) are downloaded by `setup_assets.py` and are gitignored — run the script once on each machine. If they fail to load, the module falls back to the legacy dlib backend automatically. InsightFace (512-dim) and dlib (128-dim) face embeddings are incompatible: people enrolled under one backend must have their face re-enrolled after switching (voice ID is unaffected). Note the InsightFace pretrained weights are licensed for non-commercial use only, consistent with this project's license.
- Speaker ID uses ECAPA-TDNN by default (`config.VOICE_EMBEDDER`; SpeechBrain model ~80MB, downloaded by `setup_assets.py`, gitignored). If it fails to load, the legacy Resemblyzer embedder is used automatically. ECAPA (192-dim) and Resemblyzer (256-dim) voice prints are incompatible: re-enroll voices after switching (`venv/bin/python tools/test_voice_id.py --enroll "Name" --replace`, then `--calibrate "Name"` to verify your score band). All speaker-ID thresholds stay on the original calibrated scale — ECAPA scores are mapped onto it by `audio/voice_score.py`.
- Local animal/object detection uses RF-DETR nano by default (`config.OBJECT_DETECTOR_BACKEND`; Apache 2.0, ~350MB weights downloaded by `setup_assets.py`, gitignored, ~40ms/frame CPU). If it fails to load, the legacy MediaPipe EfficientDet-Lite0 detector is used automatically. No re-enrollment involved — species lists, thresholds, and the no-screens exclusion rule are backend-independent.
- On-device TTS uses mlx-community Qwen3-TTS (`config.LOCAL_TTS_MODEL_VARIANT`, default `1.7B-Base-8bit`; ~2.9GB weights downloaded by `setup_assets.py` into `assets/models/qwen_tts/`, gitignored). It is Rex's voice only when `--local-tts` is set or when ElevenLabs fails; otherwise it is not loaded. Voice reference clips live under `assets/voices/` (also gitignored): `rex/` (Rex's own reference), `people/` (live-captured for impersonations), and `famous/` (user-supplied `<name>.wav` + `<name>.txt` for famous-person impressions). The Qwen pretrained weights are licensed for non-commercial use, consistent with this project's license.
- Visual place recognition (`config.PLACE_RECOGNITION_ENABLED`, default on) gives Rex a sense of *which room he is in*. It embeds the camera frame with MobileCLIP-S2 (open_clip; ~0.4GB weights downloaded by `setup_assets.py` into `assets/models/mobileclip/`, gitignored, ~40ms/frame CPU) and matches it against a small per-room gallery in `data/places.db` (gitignored; created on first run — override the location with `PLACE_DB_PATH`). The debounced belief is published to `world_state.current_place` for the rest of the system to read; a room is taught by name through the enrollment API in `perception/place_service.py`. If the encoder fails to load, the feature disables itself and nothing else changes. The MobileCLIP pretrained weights are licensed for non-commercial use, consistent with this project's license. Offline threshold tuning against your own room photos: `venv/bin/python tests/place_recognition_harness.py`.
- Logs are written to `logs/djr3x.log` and `logs/conversation.log`.
- Real API keys should never be committed.
- Two SQLite databases under `assets/memory/` (both gitignored, both created by `setup_assets.py`):
  - `people.db` — what Rex knows **about people** (faces, voices, facts, interests, events, conversation summaries per person).
  - `rex.db` — Rex's own **episodic memory** (his "diary"): a timestamped, first-person log of experiences. He records people seen, scenes observed ("the room was cluttered"), things he did ("I made Bret laugh", "I saw a dog"), people he **met** ("I met Bret."), **visits** ("I spent about 40 minutes with Bret."), **games** ("I played Trivia with Bret — scored 4 out of 5."), **boundaries** people set ("Bret asked me not to ask about his ex."), **emotional check-ins**, **celebrity** sightings (the Jeff/JT easter eggs), and memorable greeting moments — **birthdays, milestones, celebrations, and long-absence reunions** — plus an LLM session summary saved on shutdown. Capture is toggled with `config.EPISODIC_MEMORY_ENABLED`. **Recall (Phase 2) is implemented and enabled by default** (separate `config.EPISODIC_RECALL_ENABLED` switch): `memory/episodic_recall.py` surfaces these memories back into conversation — a per-person "shared memory" callback in the reply prompt ("I made you laugh", "we played Trivia") and an idle "memory musing" beat. The two switches are independent so the diary can build silently while recall is A/B-tested.

## License

Except where otherwise noted, this project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE). You may share and modify it with attribution for noncommercial purposes, but commercial use is not permitted without prior written permission.

DJ-R3X v2 is an unofficial fan project. It is not affiliated with, endorsed by, or sponsored by Disney, Lucasfilm, OpenAI, ElevenLabs, Jeopardy Productions, or any other referenced rights holder. Third-party names, trademarks, sound clips, clue data, models, libraries, and other materials remain the property of their respective owners and may be subject to separate terms.

## Authors

- Bret Benziger
- OpenAI Codex
- Claude Code
