# DJ-R3X v2

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It combines speech recognition, text-to-speech, camera awareness, face and voice memory, conversational social behaviors, hardware control, music playback, and verbal games into one interactive companion system.

The project is built for live, in-room use: Rex can recognize people, remember details across sessions, react to arrivals, handle introductions, play games like Jeopardy, answer visual prompts, and drive connected LEDs and servos when the hardware is attached.

## Features

- Wake-word and always-listening conversation flow
- On-device speech recognition — **Qwen3-ASR** (`mlx-community/Qwen3-ASR-1.7B-8bit`, via MLX) is the primary backend, with a three-step fallback chain: Qwen3 → local MLX Whisper (`whisper-large-v3-turbo`) → the OpenAI Whisper API. Switched from Whisper in July 2026 after `tools/asr_bench.py` measured identical word accuracy on real room recordings at roughly twice the speed (0.57 s vs 1.02 s median). Qwen3 also reports a much cleaner confidence signal, which is what gates whether a turn is trusted enough to learn from (`QWEN_ASR_TRUST_MIN_AVG_LOGPROB`). Set `TRANSCRIPTION_BACKEND = "whisper"` to go back
- ElevenLabs TTS with cached speech output — ElevenLabs is Rex's "true voice" and the default
- On-device TTS voice clone (mlx-audio Qwen3-TTS) as a second backend: run Rex's whole voice offline with the `--local-tts` flag, and — always on — **automatic fallback** to the local voice whenever ElevenLabs is unreachable, errors, or runs out of credits, so Rex never goes silent (toggle with `LOCAL_TTS_FALLBACK_ENABLED`). The ~2.9 GB model is downloaded by `setup_assets.py`
- Voice impersonations for fun — "do an impersonation of me / of Jimmy Carter": Rex clones a voice and delivers a short, affectionate parody in it. For someone he knows, he captures a quick voice sample (asks you to repeat a line) and mines that person's memory for the material — while hard-excluding any boundaries or sensitive topics; famous people work out of the box — ~30 references ship with the repo (Jimmy Carter, Obama, JFK, Oprah, the Star Trek computer…); add your own by dropping a clip + transcript in `assets/voices/famous/` (toggle with `IMPERSONATION_ENABLED`; needs the local TTS model)
- Camera-based scene, face, appearance, and animal awareness — face detection/recognition runs on InsightFace (SCRFD detector + ArcFace 512-dim embeddings via ONNX Runtime; `FACE_BACKEND=dlib` restores the legacy stack); local animal/object detection runs on RF-DETR nano (`OBJECT_DETECTOR_BACKEND=mediapipe` restores EfficientDet)
- Voice and face enrollment for known people — speaker ID runs on ECAPA-TDNN embeddings (SpeechBrain, 192-dim; `VOICE_EMBEDDER=resemblyzer` restores the legacy stack)
- Persistent memory database for people, relationships, preferences, and events (`people.db`)
- Rex's own first-person episodic memory (`rex.db`) — a timestamped log of his experiences (people seen, scenes observed, things he did, per-session conversation summaries)
- Social intelligence layers for repairs, boundaries, grief, celebrations, callbacks, and group discretion
- First-meeting curiosity — when Rex meets someone brand new he runs a short, in-character "getting to know you" burst: research-backed baseline questions with quick witty reactions between them and the occasional self-reveal, building a useful profile before settling into free conversation. It's bounded and backs off the moment you're not into it (toggle with `ONBOARDING_ENABLED`)
- Mood-driven body language — Rex's posture (head lift/tilt, visor openness, breathing, idle gestures) reflects a sustained "body mood" that shifts when he's complimented, insulted, or amused, riding on top of face-tracking
- Droid sound effects — short chirps and servo whirs (`assets/audio/sound_effects/`) that color his reactions: an emotion-matched chirp fires the instant a reply's TTS starts generating (filling the synthesis gap, never delaying the voice — effects yield the speaker to speech within ~50 ms), drive-base commands get motor-whir/turning clips plus an arrival chirp and a "whoa, blocked" accent, and body gestures get servo-whir accents. Multi-variant clips are picked at random; cooldowns keep it an accent, not a tic (toggle with `SOUND_EFFECTS_ENABLED`, per-family switches and volume in `config.py`)
- Sound-event awareness — a local AudioSet classifier (YAMNet ONNX, ~16MB, milliseconds per window on CPU) gives Rex real non-speech hearing: dog barks, doorbells, knocks, laughter, screams, breaking glass, bangs, sirens, and smoke alarms are recognized as named events. Urgent sounds (scream, glass, bang) ride the existing startle reflex; notable ones get a short in-character reaction — the doorbell gets a droid-doorman announcement, a smoke alarm gets genuine concern — with cooldowns so a barking dog is one remark, not a running commentary. Runs behind the same self-noise gate as the rest of scene analysis (never reacts to his own voice or music), and degrades cleanly to the legacy energy heuristics if the model is missing (toggle with `SOUND_AWARENESS_ENABLED` / `SOUND_AWARENESS_REACTIONS_ENABLED`)
- A wandering attention of his own — when the conversation lulls he'll stop staring, glance around the room, then look back and sometimes re-greet, so he doesn't feel locked to a fixed stare
- Bored environmental snark — left idle, he looks around and invents in-character jabs about the room he actually sees: complaints about how dull it is, faux-clueless questions about objects ("what's that black chair for?"), digs at the clutter, snobby art opinions, or pleas to be taken somewhere with more life forms
- Visual curiosity — when an engaged conversation goes quiet, Rex takes a fresh look and asks one grounded question about something he can actually see right now, instead of generic small talk (toggle with `VISUAL_CURIOSITY_ENABLED`)
- Web search for current info — when a question needs live data (you ask him to "look it up", or he decides on his own that it needs the latest), Rex says a quick stall line, searches the web via OpenAI's hosted `web_search` tool, and answers in character; trigger phrases and stall lines are editable in your user config (toggle with `WEB_SEARCH_ENABLED`)
- Reads the news, and brings it up — one web-search call per day fetches a handful of notable stories (plus stories tailored to what he knows you're into), and in a conversational lull Rex offers ONE of them the way a person does: "hey, did you hear about…?". Ask for more and he looks it up and gives you a short spoken digest, not a press release. Each story is offered at most once, ever (toggle with `CURRENT_EVENTS_ENABLED` / `INTEREST_NEWS_ENABLED`)
- A mood of his own — Rex mints ONE mood per day from what the day actually contains (the weather, whatever news he's chewing on, a holiday, his own hardware, plain chance), drifts it as the day goes, and persists it so relaunching at 4pm resumes the mood he woke up with. Ask "how are you?" — directly, or by bouncing his own question back — and you get a real answer instead of "systems nominal". On a notable day he may mention it unprompted, in the hello or in a lull, at most once a day (toggle with `REX_MOOD_ENABLED`)
- Notices what changed — a persistent per-room object ledger means Rex can tell a genuinely NEW thing from the furniture he's seen a hundred times, so "what's that?" fires on the box that appeared today, not the couch (toggle with `ROOM_MODEL_ENABLED` / `ROOM_CHANGE_REMARK_ENABLED`)
- Keeps working offline — when the Mac loses internet, Rex fails over to a local Ollama reply brain and his on-device voice, and the paths that need the network (weather, news, web search) fast-skip instead of paying timeouts. He tells you his "galactic internet link is out" rather than going mute, and recovers on his own. Pull the offline model with `ollama pull qwen3.5:2b` — `setup_assets.py` does not fetch it yet (toggle with `OFFLINE_MODE_ENABLED`)
- Waves back — when the camera sees someone wave at him (MediaPipe pose gesture), Rex returns the wave with his arm and a short warm line, mirroring the speed of your wave the way you'd wave back across a room. Keep waving and it turns into a bit — his responses escalate (warm greeting → progressively terser → a crack about the repetition → eventually he just ignores you). Debounced so a single wave gets a single wave-back (toggle with `WAVE_BACK_ENABLED`)
- Live pose wireframe — the GUI dashboard's camera preview overlays detected bodies as real-time skeletons (MediaPipe pose landmarks, up to `POSE_MAX_PEOPLE` people) on top of the per-person face boxes, so you can see what Rex's body-tracking sees. By default only poses whose head lines up with a visible face box are drawn (`GUI_POSE_REQUIRE_FACE`), which hides the phantom poses MediaPipe fits onto furniture; set it False to draw every detected pose (toggle the overlay with `GUI_POSE_WIREFRAME_ENABLED`)
- A sense of place — visual place recognition (MobileCLIP-S2) that recognizes which enrolled room Rex is in and publishes a debounced belief to `world_state.current_place`. Rooms are taught by voice ("this is the living room" — or just answer when he asks), recognition is stable (temporal hysteresis plus a motion gate so he can't "change rooms" without moving — with escape hatches for being picked up and carried: sustained visual evidence flips the belief, and sustained unfamiliarity makes him admit he's lost instead of insisting on a stale room), and when he genuinely doesn't recognize where he is he'll ask what room it is during a lull and remember your answer. The whole feature no-ops cleanly if the encoder isn't available (toggle with `PLACE_RECOGNITION_ENABLED`; the talking layer with `PLACE_QUESTIONS_ENABLED`)
- Servo and LED hardware hooks for a physical droid body
- Voice-driven motion — an optional ESP32 drive base lets Rex physically roll around the room on command ("turn left", "back up", "come here", "halt"), avoiding obstacles and people with onboard sensors. An explicit "come here" / "come over here" / "come to me" starts a bounded person search: Rex rotates until face tracking finds someone, turns his chassis toward them, then approaches until the front ToF reaches 1 metre; furniture or another obstacle stops him first. The ESP32 owns the real-time, fail-safe motor loop while the Mac sends high-level commands
- A back-off reflex — when Rex is parked and someone steps right up into his face, the front 8×8 ToF sensor feels the approach and he reflexively edges backward, the way an animal gives itself room. He only retreats as far as the rear ToF sensors say is safe — stopping short of the wall and simply holding his ground when he's cornered (toggle with `MOTION_FLINCH_ENABLED`; needs the drive base)
- Music controls and verbal games: I Spy, 20 Questions, themed five-question Trivia rounds, Jeopardy, and Word Association

See [CONTEXT.md](CONTEXT.md) for more detailed project features, architecture notes, hardware mappings, and behavior design.

The restructuring status and remaining live validation are tracked in
[the Lean Brain plan](docs/lean_brain_restructuring_plan.md). Speech recognition
remains batch-based. Rex finishes a pending reply before handling later captured
speech; mixed-speaker detection can abstain when it cannot safely assign a name.
Run `venv/bin/python tools/run_lean_checks.py` for isolated checks with real I/O blocked.

## Requirements

- macOS on **Apple Silicon** (required, not just preferred — `mlx`, `mlx-whisper`, and `mlx-audio` have no x86-64 wheels, and the default ASR and the local TTS voice are both MLX-only)
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
- Ollama plus two local models pulled by `setup_assets.py`: `qwen2.5:1.5b` for low-latency classifier/shaping work, and `nomic-embed-text` (~270MB) for embedding-based semantic memory recall (`MEMORY_SEMANTIC_RECALL_ENABLED`, on by default — without it recall degrades to keyword matching)

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

Instead of starting `main.py` by hand, you can have macOS stay quietly ready and launch the robot by voice. A tiny LaunchAgent (`rex_supervisor.py`) listens only for **"wake up Rex"** and launches the full controller on demand (it starts `main.py` headless — no dashboard on wake, since 2026-08-08); **"shut down"** powers it back down while the listener keeps running. Install with `scripts/install_supervisor.sh` (the setup script also offers this). See **[docs/supervisor.md](docs/supervisor.md)** for how it works and how the single-instance lock prevents a double-launch (including when Rex is asleep).

The same installer adds a **menu bar battery meter** (`tools/rex_battery_menubar.py`) when `MOTION_ESP32_PORT` is set: the drive base's charge, voltage, and current stay visible in the macOS menu bar even while the robot is off, by passively reading the ESP32's always-on telemetry stream. It releases the serial port automatically whenever `main.py` is running (same flock the supervisor uses for the mic) and reclaims it when Rex shuts down. A **"Set Battery to 100%"** menu item lets you sync the firmware's charge gauge the moment your charger's taper current says the pack is full. The same dropdown also carries a **drive joystick** — dragging it sends live drive setpoints straight to the ESP32, so you can move the robot without starting `main.py` (it holds the port only while Rex is off, same as the meter).

It also adds a **"Servo Control" menu bar console** (`tools/rex_servo_menubar.py`) when `MAESTRO_PORT` is set: a dropdown with live sliders for all 8 Maestro servo channels (labelled with the current position in microseconds, initialized from the board's actual positions) plus a **"Restart Pololu"** action that sends the Maestro's go-home command. Sliders command the servos directly over the same wire protocol the robot uses. Like the battery meter, it releases the serial port automatically whenever `main.py` is running and reclaims it when Rex shuts down.

And an **"LED Control" menu bar console** (`tools/rex_led_menubar.py`) when `ARDUINO_HEAD_PORT` or `ARDUINO_CHEST_PORT` is set: a dropdown with one button per animation the head and chest firmware support, so you can audition any LED pattern while the robot is off. Head "speak" animations are an equalizer that normally rides a `SPEAK_LEVEL` stream derived from live TTS audio, so clicking one also starts a synthetic level wave to make the mouth actually dance; chest speak patterns animate on their own. It starts in **Battery Meter Mode** (both ports released, buttons inert) because the battery meter needs those same exclusive-open ports to paint the chest charge gauge and the mouth's state-of-charge breathing while Rex is off — holding them permanently left the robot sitting dark on the charger. Toggle the top menu item, or just click any animation, to take the ports; the choice persists across relaunches. It reads ports straight from `.env` and never imports `config.py`, so it runs without API keys configured.

On the physical robot, the supervisor also keeps a clean `main` checkout current
with `origin/main`: it checks at supervisor startup, every four hours, and again
immediately before launching `main.py`. A running controller is never updated
underneath itself, failures fall back to the installed version, and no updater
state files are created. See [docs/supervisor.md](docs/supervisor.md) for the
safety rules and configuration switches.

## Configuration

User-tunable defaults live in [config.py](config.py). API keys should stay in `apikeys.py`, and host-specific hardware paths plus build-specific servo limit overrides should stay in `.env`; both are intentionally excluded from git.

The settings most people actually want to change — AI model selection, Rex's personality dials and base prompt, location/venue, feature on/off switches, and key timeouts — are surfaced in [user_config.example.py](user_config.example.py), a heavily commented template grouped by topic. The setup script copies it to `user_config.py` (gitignored), which `config.py` imports last so its values win over the defaults. (The script only creates `user_config.py` when it doesn't already exist, so when new settings are added to the template — e.g. the web-search options — copy those sections over manually to expose them.) Every setting ships commented out at its current default: uncomment a line to override it, or re-comment/delete it to fall back to the `config.py` default. A missing `user_config.py` is harmless — `config.py`'s defaults are used unchanged, so `from config import X` keeps working everywhere. A few values computed from a base are re-derived after the override so changing the base propagates. Note `ACTION_ROUTER_MODEL` is **no longer** one of them — it was decoupled from `LLM_MODEL` on 2026-08-02 and is a literal (`gpt-5.4-nano`); override it directly if you want to change it. Deeper internal tuning (CV thresholds, cooldowns, scoring) intentionally stays in `config.py`.

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

### Direction of travel

**`headtilt` is the only inverted channel.** Every other channel correlates: a higher quarter-microsecond value moves that joint in the direction named below, and a lower value moves it back.

| Ch | Channel | Higher value → | Lower value → |
| --- | --- | --- | --- |
| 0 | `neck` | head turns right | head turns left |
| 1 | `headlift` | head physically higher | head lower |
| 2 | `headtilt` | **inverted** — head tilted *down* | head tilted *up* |
| 3 | `visor` | visor more open (lens clear) | visor closed over the lens |
| 4 | `elbow` | arm lifted up | arm hanging down — where it falls unpowered |
| 5 | `hand` | wrist rotates one way | wrist rotates the other |
| 6 | `pokerarm` | — | — |
| 7 | `heroarm` | arm raised toward horizontal | arm hanging down the torso |

The elbow's low end is also its **unpowered rest**: with the robot off the servos go limp and the arm falls there, so `config.SERVO_CHANNELS["elbow"]["rest"]` parks and starts it at that value. See "unpowered rest" in [config.py](config.py).

## Motion Base (optional)

An optional ESP32-controlled drive base lets Rex physically move around a room on spoken command while avoiding obstacles. The ESP32 runs a real-time, fail-safe motor loop (PID speed control, Time-of-Flight obstacle stop, heartbeat watchdog) and the Mac sends high-level commands (`turn`, `move`, `come`, `stop`) over USB serial. Spoken intents like "turn left", "back up", "come here", and "halt" route through the normal conversation pipeline to the base; "stop" only steers the base while it is actually moving, so it never hijacks stop-music/stop-game.

- **Wire contract:** [docs/motion_protocol.md](docs/motion_protocol.md). **Feature spec & wiring:** [docs/motion_system.md](docs/motion_system.md).
- **Firmware:** [firmware/djr3x_motion](firmware/djr3x_motion/) (Arduino sketch for the ESP32). The live drive stack has shipped — BTS7960 motor drivers, Hall quadrature encoders, per-wheel PID, an LSM6DS3 IMU, a QMC5883P magnetometer, and an 8×8 matrix ToF. The repo default still *builds* against the stubbed hardware layer so a bare ESP32 (and the smoke test) keeps compiling; the real robot is flashed by passing the hardware flags at build time (`-DMOTION_HW_PRESENT=1` and friends — see [firmware/djr3x_motion/README.md](firmware/djr3x_motion/README.md)), **not** by editing `hal.h`.
- **Enable it:** set `MOTION_ESP32_PORT` in `.env` (the setup script can auto-detect and set this). Motion is fully disabled — with zero change to the rest of Rex's behavior — until that port is set.
- **No base attached?** If you give Rex a drive command ("turn left", "move forward", "come here") while the ESP32 isn't connected, he refuses out loud with a pre-canned in-character quip instead of silently ignoring it — there are no wheels to move, so he says so (toggle/edit with `MOTION_NO_BASE_DENIAL_ENABLED` / `MOTION_NO_BASE_DENIAL_LINES`). A bare "stop"/"halt" is unaffected.
- **Manual driving:** a Bluetooth gamepad (8BitDo Pro 2) pairs straight to the ESP32 via the Bluepad32 core and drives the base directly, with teleop owned by the firmware so it keeps working even if the Mac is busy — plus D-pad absolute-heading turns and buttons wired to soundboard clips and body animations. Built in with `-DMOTION_GAMEPAD_PRESENT=1`.
- **Cardinal directions:** with a calibrated QMC5883P magnetometer, Rex understands headings — "turn north", "go east two feet" — rotating to a true bearing rather than a relative angle. Needs `COMPASS_ENABLED` plus an in-situ calibration (`venv/bin/python tools/compass_calibrate.py`); without both, cardinal commands are declined rather than guessed.
- **Bring-up test:** `venv/bin/python firmware/tools/motion_serial_smoketest.py` exercises the whole protocol against a connected board.

> **Safety:** the ESP32 stops the base on its own (obstacle / lost-comms) independent of the Mac. Do not attach motor power until the base has been bench-tested with wheels off the ground.
>
> **There is no cliff detection.** The sensor layout is 8 *horizontal* radial ToF sensors with no down-facing sensor, so a stair edge or table edge is invisible to the base and it **will drive off a drop-off**. The `cliff` zone exists in the wire protocol but nothing can currently raise it. Do not run the base near stairs or on a table.

## Project Layout

```text
audio/          Speech input, VAD, transcription, TTS, playback, and audio scene logic
awareness/      Time, weather, holidays, current events, interoception, background awareness
features/       Games, music, commandable behaviors, and interactive features
gui/            PySide dashboard — live camera preview, transcript, memory banks, controls
hardware/       Servo, LED, and motion-base serial integrations
intelligence/   Conversation, LLM prompting, empathy, social behavior, and motion control
memory/         Person/fact/event stores (people.db), Rex's episodic diary (rex.db), recall
perception/     Visual place recognition (MobileCLIP-S2) — which room Rex is in
vision/         Camera, face recognition (InsightFace SCRFD+ArcFace), pose, scene analysis
utils/          Shared helpers (logging, config loading, audio tags, locks)
sequences/      Scripted servo/LED animations and body beats
firmware/       ESP32 motion-controller firmware (Arduino sketch) + host serial tools
arduino/        Head and chest LED firmware (Arduino Nano sketches)
tools/          Menu bar apps, benchmarks, and hardware/voice test utilities
scripts/        Installers (LaunchAgent supervisor + menu bar consoles)
launchd/        macOS LaunchAgent plist templates
docs/           Protocol specs, design notes, and feature plans
tests/          Per-module unittest suite (see CLAUDE.md for how to run it)
evals/          Conversation-quality eval harness
assets/         Models, audio, game assets, memory database, cached generated assets
data/           Place-recognition gallery (places.db)
logs/           Runtime logs
```

## Notes

- The program can run with missing droid hardware, but servo and LED features will be disabled until the configured devices are connected.
- Face recognition uses InsightFace by default (`config.FACE_BACKEND`). Its models (~190MB) are downloaded by `setup_assets.py` and are gitignored — run the script once on each machine. If they fail to load, the module falls back to the legacy dlib backend automatically. InsightFace (512-dim) and dlib (128-dim) face embeddings are incompatible: people enrolled under one backend must have their face re-enrolled after switching (voice ID is unaffected). Note the InsightFace pretrained weights are licensed for non-commercial use only, consistent with this project's license.
- Transcription uses Qwen3-ASR by default (`config.TRANSCRIPTION_BACKEND`, default `"qwen3"`; `mlx-community/Qwen3-ASR-1.7B-8bit`, ~2.3GB weights downloaded by `setup_assets.py` into `assets/models/qwen_asr/`, gitignored). Local MLX Whisper (`whisper-large-v3-turbo`, ~1.5GB, same script) stays installed as the second link in the chain, and the OpenAI Whisper API is the last resort — set `TRANSCRIPTION_BACKEND = "whisper"` to make Whisper primary again. The two backends report confidence differently, so the learn-from-this-turn trust floors are separate (`QWEN_ASR_TRUST_MIN_AVG_LOGPROB` vs `WHISPER_TRUST_*`); re-benchmark with `venv/bin/python tools/asr_bench.py` if you switch. The Qwen pretrained weights are licensed for non-commercial use, consistent with this project's license.
- Speaker ID uses ECAPA-TDNN by default (`config.VOICE_EMBEDDER`; SpeechBrain model ~80MB, downloaded by `setup_assets.py`, gitignored). If it fails to load, the legacy Resemblyzer embedder is used automatically. ECAPA (192-dim) and Resemblyzer (256-dim) voice prints are incompatible: re-enroll voices after switching (`venv/bin/python tools/test_voice_id.py --enroll "Name" --replace`, then `--calibrate "Name"` to verify your score band). All speaker-ID thresholds stay on the original calibrated scale — ECAPA scores are mapped onto it by `audio/voice_score.py`.
- Local animal/object detection uses RF-DETR nano by default (`config.OBJECT_DETECTOR_BACKEND`; Apache 2.0, ~350MB weights downloaded by `setup_assets.py`, gitignored, ~40ms/frame CPU). If it fails to load, the legacy MediaPipe EfficientDet-Lite0 detector is used automatically. No re-enrollment involved — species lists, thresholds, and the no-screens exclusion rule are backend-independent.
- On-device TTS uses mlx-community Qwen3-TTS (`config.LOCAL_TTS_MODEL_VARIANT`, default `1.7B-Base-8bit`; ~2.9GB weights downloaded by `setup_assets.py` into `assets/models/qwen_tts/`, gitignored). It is Rex's voice only when `--local-tts` is set or when ElevenLabs fails; otherwise it is not loaded. Voice reference clips live under `assets/voices/`: `rex/` (Rex's own reference) and `famous/` (`<name>.wav` + `<name>.txt`) are **tracked in git**, so a fresh checkout has a working local voice and ~30 ready-made impressions; only `people/` (live-captured for impersonating someone Rex knows) is gitignored. The Qwen pretrained weights are licensed for non-commercial use, consistent with this project's license.
- Visual place recognition (`config.PLACE_RECOGNITION_ENABLED`, default on) gives Rex a sense of *which room he is in*. It embeds the camera frame with MobileCLIP-S2 (open_clip; ~0.4GB weights downloaded by `setup_assets.py` into `assets/models/mobileclip/`, gitignored, ~40ms/frame CPU) and matches it against a small per-room gallery in `data/places.db` (gitignored; created on first run — override the location with `PLACE_DB_PATH`). The debounced belief is published to `world_state.current_place` for the rest of the system to read. Rooms are taught by voice — say "this is the living room" (or answer when he asks what room he's in) and `intelligence/place_questions.py` names + enrolls it; the proactive "what room is this?" ask rides the same lull-speaker path as the object-curiosity questions and is gated by the shared question budget (toggle with `PLACE_QUESTIONS_ENABLED`). If the encoder fails to load, the feature disables itself and nothing else changes. The MobileCLIP pretrained weights are licensed for non-commercial use, consistent with this project's license. Offline threshold tuning against your own room photos: `venv/bin/python tests/place_recognition_harness.py`.
- Logs are written to `logs/`. With the shipped `DEBUG_MODE = True`, each run gets its own timestamped pair — `logs/djr3x-<YYYY-MM-DD-HH-MM-SS>.log` and `logs/conversation-<same-stamp>.log`. Set `DEBUG_MODE = False` for the single rolling `logs/djr3x.log` / `logs/conversation.log` instead.
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
