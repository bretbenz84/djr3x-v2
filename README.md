# DJ-R3X v2

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It combines speech recognition, text-to-speech, camera awareness, face and voice memory, conversational social behaviors, hardware control, music playback, and verbal games into one interactive companion system.

The project is built for live, in-room use: Rex can recognize people, remember details across sessions, react to arrivals, handle introductions, play games like Jeopardy, answer visual prompts, and drive connected LEDs and servos when the hardware is attached.

## Features

- Wake-word and always-listening conversation flow
- On-device speech recognition — **Qwen3-ASR** (`mlx-community/Qwen3-ASR-1.7B-8bit`, via MLX) is the primary backend, with a three-step fallback chain: Qwen3 → local MLX Whisper (`whisper-large-v3-turbo`) → the OpenAI Whisper API. Switched from Whisper in July 2026 after `tools/asr_bench.py` measured identical word accuracy on real room recordings at roughly twice the speed (0.57 s vs 1.02 s median). Qwen3 also reports a much cleaner confidence signal, which is what gates whether a turn is trusted enough to learn from (`QWEN_ASR_TRUST_MIN_AVG_LOGPROB`). Set `TRANSCRIPTION_BACKEND = "whisper"` to go back
- ElevenLabs TTS with cached speech output — v3 Conversational is the default, using the existing StarTours voice clone and expressive audio tags. Set `TTS_MODEL_ID = "eleven_v3"` in `user_config.py` to restore the previous engine.
- On-device TTS voice clone (**Breeze TTS 2, 8-bit**, via mlx-audio; Qwen3-TTS optional): run Rex's whole voice offline with the `--local-tts` flag, and — always on — **automatic fallback** to the local voice whenever ElevenLabs is unreachable, errors, or runs out of credits, so Rex never goes silent (toggle with `LOCAL_TTS_FALLBACK_ENABLED`). The selected model is downloaded by `setup_assets.py` (~4.6 GB for Breeze). Breeze streams ordinary speech after buffering 1.5 seconds of audio and prepares complete impersonations before playback; select with `LOCAL_TTS_BACKEND = "breeze"` or `"qwen"` in `user_config.py`
- Voice impersonations for fun — "do an impersonation of me / of Jimmy Carter": Rex clones a voice and delivers a short, affectionate parody in it. Known people can be requested by name or a unique stored nickname. He reuses accepted enrollment recordings (or collects ordinary speech when needed) and builds one coherent joke from supported work, interests or preferences — excluding passing remarks, uncertain claims, boundaries and sensitive profile details; famous people work out of the box — ~30 references ship with the repo (Jimmy Carter, Obama, JFK, Oprah, the Star Trek computer…); add your own by dropping a clip + transcript in `assets/voices/famous/` (toggle with `IMPERSONATION_ENABLED`; needs the local TTS model)
- Camera-based scene, face, appearance, and animal awareness — face detection/recognition runs on InsightFace (SCRFD detector + ArcFace 512-dim embeddings via ONNX Runtime; `FACE_BACKEND=dlib` restores the legacy stack); local animal/object detection runs on RF-DETR nano (`OBJECT_DETECTOR_BACKEND=mediapipe` restores EfficientDet)
- Voice and face enrollment for known people — speaker ID uses CAM++ on CPU and learns from confirmed introductions and consistent ordinary conversation. No voice-ID recitation or mouth tracking is required; accepted original audio is kept locally for future model migrations. See [voice learning](docs/conversational_voice_learning.md).
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
- Voice-driven motion — an optional ESP32 drive base lets Rex physically roll around the room on command ("turn left", "back up", "come here", "halt"), avoiding obstacles and people with onboard sensors. An explicit "come here" / "come over here" / "come to me" uses the caller's visible face first: Rex aligns his head and chassis, then approaches with a configured front clearance (currently 1.3 metres). Voice direction and radar guide a bounded search when the caller is off camera; furniture or another obstacle stops him first. The ESP32 owns the real-time, fail-safe motor loop while the Mac sends high-level commands
- A back-off reflex — when Rex is parked and someone steps right up into his face, the front 8×8 ToF sensor feels the approach and he reflexively edges backward, the way an animal gives itself room. He only retreats as far as the rear ToF sensors say is safe — stopping short of the wall and simply holding his ground when he's cornered (toggle with `MOTION_FLINCH_ENABLED`; needs the drive base)
- Music controls and verbal games: I Spy, 20 Questions, themed five-question Trivia rounds, Jeopardy, and Word Association

Jeopardy scores every answer for the player whose turn is open, regardless of the
voice-ID label. No voiceprint setup is required. Spoken-answer matching includes
an offline pronunciation dictionary for homophones and spelling variants; uncertain
near-matches get a repeat request without a deduction. The dictionary is installed
with `requirements.txt` by `setup_assets.py`.

The thinking music follows the regular/Daily Double/rebound answer clock,
including speech grace, and stops for feedback or the time-up chime. Hardware
AEC lets it continue during player speech; without it, music pauses for capture
and resumes if the turn remains open. Other proposed game reliability fixes are
tracked in [the live reliability review](docs/jeopardy_live_reliability_review.md).

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
| `-local-tts`, `--local-tts` | Use the configured local voice (Breeze 8-bit by default). Preloads at startup; synthesis needs no network. If local assets cannot load, logs an error and can fall back to ElevenLabs. This flag changes TTS, not the reply brain. |

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

It also adds a **"Servo Control" menu bar console** (`tools/rex_servo_menubar.py`) when `MAESTRO_PORT` is set: a dropdown with live sliders for all 11 Maestro servo channels (labelled with the current position in microseconds, initialized from the board's actual positions) plus a **"Restart Pololu"** action that sends the Maestro's go-home command. Sliders command the servos directly over the same wire protocol the robot uses. Like the battery meter, it releases the serial port automatically whenever `main.py` is running and reclaims it when Rex shuts down.

And an **"LED Control" menu bar console** (`tools/rex_led_menubar.py`) when `ARDUINO_HEAD_PORT` or `ARDUINO_CHEST_PORT` is set: a dropdown with one button per animation the head and chest firmware support, so you can audition any LED pattern while the robot is off. Head "speak" animations are an equalizer that normally rides a `SPEAK_LEVEL` stream derived from live TTS audio, so clicking one also starts a synthetic level wave to make the mouth actually dance; chest speak patterns animate on their own. It starts in **Battery Meter Mode** (both ports released, buttons inert) because the battery meter needs those same exclusive-open ports to paint the chest charge gauge and the mouth's state-of-charge breathing while Rex is off — holding them permanently left the robot sitting dark on the charger. Toggle the top menu item, or just click any animation, to take the ports; the choice persists across relaunches. It reads ports straight from `.env` and never imports `config.py`, so it runs without API keys configured.

On the physical robot, the supervisor also keeps a clean `main` checkout current
with `origin/main`: it checks at supervisor startup, every four hours, and again
immediately before launching `main.py`. A running controller is never updated
underneath itself, failures fall back to the installed version, and no updater
state files are created. See [docs/supervisor.md](docs/supervisor.md) for the
safety rules and configuration switches.

## Configuration

User-tunable defaults live in [config.py](config.py). API keys should stay in `apikeys.py`, and host-specific hardware paths plus build-specific servo limit overrides should stay in `.env`; both are intentionally excluded from git.

The settings most people actually want to change — AI model selection, Rex's personality dials and base prompt, location/venue, feature on/off switches, and key timeouts — are surfaced in [user_config.example.py](user_config.example.py), a heavily commented template grouped by topic. The setup script copies it to `user_config.py` (gitignored), which `config.py` imports last so its values win over the defaults. (The script only creates `user_config.py` when it doesn't already exist, so when new settings are added to the template — e.g. the web-search options — copy those sections over manually to expose them.) Every setting ships commented out at its current default: uncomment a line to override it, or re-comment/delete it to fall back to the `config.py` default. A missing `user_config.py` is harmless — `config.py`'s defaults are used unchanged, so `from config import X` keeps working everywhere. A few values computed from a base are re-derived after the override so changing the base propagates. Deeper internal tuning (CV thresholds, cooldowns, scoring) intentionally stays in `config.py`.

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

**Among the original eight channels, `headtilt` is the only inverted channel.** Every other channel correlates: a higher quarter-microsecond value moves that joint in the direction named below, and a lower value moves it back.

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

### Throttle arm commissioning (channels 8–10)

The throttle arm has coupled clearance constraints: a low shoulder restricts elbow
extension, and wrist clearance depends on both upstream joints. Individual Maestro
limits do not prevent chassis collisions. The dedicated runtime controller uses a
limited upward repertoire and the empirically tested coupled clearance boxes.

`config.THROTTLE_SERVO_CHANNELS` defines the linkages separately from the independent
`SERVO_CHANNELS` table, so generic independent-joint commands cannot move them.
`sequences/throttle_arm.py` owns their startup, idle, speech, and parking motion.
The separate menu-bar console provides
manual commissioning sliders for all three throttle joints, regardless of the main
program's animation enable flag. These sliders enforce individual limits and apply
the profiles below before each target, but do not enforce coupled clearance.
Opening/reloading the helper only reads positions; movement requires a slider action. Generic target/profile APIs reject throttle channels.

| Channel | Name | Measured limits (µs) | Down (µs) | Up (µs) | Speed / acceleration |
| --- | --- | --- | --- | --- | --- |
| 8 | `throttle_shoulder` | 535–2280 | 2280 | 535 | 30 / 6 |
| 9 | `throttle_elbow` | 500–2500 | 500 | 2500 | 70 / 12 |
| 10 | `throttle_wrist` | 500–2500 | 2500 | 500 | 70 / 12 |

For collecting measurements, turn on **Measurement mode (slow throttle)** in the
Servo Control menu. Subsequent throttle slider moves and nudges use shoulder speed /
acceleration 10 / 2 and elbow/wrist 20 / 3. Toggling the mode does not move anything
or change an already-running move. Each throttle joint has a **Nudge** submenu with
−10, −5, −1, +1, +5, +10 µs actions, starting from a fresh board pulse readback.
Nudges require a stationary controller and refuse off/unreadable channels.

Choose **Record this pose…**, type a clearance or obstacle note in the text field,
and click Record. Recording works in either normal or measurement mode.
The helper reads all three throttle pulses and appends a timestamped row to
`data/throttle_measurements.csv`. Recording rejects pending/moving/off/unreadable
poses and never sends a movement command. Values are controller pulses, not physical
shaft feedback; check actual clearance before saving. Measurement status and saved
values appear in the menu. Go-home is hidden while measuring. Turn measurement
mode off to restore normal profiles on subsequent moves. Coupled clearance remains
manual during measurement; this mode does not automatically approve recorded poses.

#### Recording a throttle movement sequence

Use **Throttle sequences → Start recording…**, name the demonstration, and wait
for the menu to say **Recording**. Start recording requires all three throttle
channels to be on and stationary; it reads the starting pulses without moving them.
Move the throttle sliders/nudges along a path you observe to be clear. The helper
records successful throttle target writes, their speed/acceleration settings, their
order, and elapsed time, including pauses. Other robot channels are not recorded.
Wait for the arm to settle, then choose **Stop and save recording**. Files are saved
under `data/throttle_sequences/` as uniquely named JSON; existing files are not
overwritten. **Discard recording** drops the unsaved demonstration without moving.
Connection loss or Rex taking over cancels the active session; it never auto-resumes.

**Play saved sequence** is a separate, explicit movement action. Manually return
along a safe path to the sequence's starting pose first. Playback requires fresh,
stationary pulse readings within 1 µs of all three recorded starting values and
refuses to reposition automatically. It reuses the demonstrated timing and profiles,
even if measurement mode is currently different. Sliders/nudges are blocked during
playback; go-home is blocked during recording and playback. **Stop playback — hold
position** cancels future targets and sends the current pulse readbacks as hold
targets (it does not switch servo torque off). A missed timing deadline aborts playback
instead of bursting overdue commands. Completion checks the final pulse readbacks.

A demonstrated command sequence is not a measured physical trajectory: the Maestro
has no shaft feedback, and mechanical lag/obstacles can change. This recorder does
not certify collision safety or connect sequences to Rex's autonomous behaviors.
Saved pose notes and sequence recordings remain separate datasets.

The measured shoulder-dependent elbow limits are:

| Shoulder pulse (µs) | Minimum elbow pulse (µs) | Maximum elbow pulse (µs) |
| --- | --- | --- |
| 2280 (shoulder fully down) | 1546 | 2500 |
| 1702 | 636 | 2500 |
| 1636 | 500 | 2500 |

`config.throttle_elbow_limits(shoulder_qus)` returns this static envelope in
quarter-microseconds, intersected with the individual elbow limits. Between measured
points it uses the more restrictive neighboring minimum: 1546 µs above a shoulder
pulse of 1702, 636 µs above 1636 through 1702, and 500 µs at/below 1636.
The last range assumes clearance does not decrease as the shoulder rises further.
No linear clearance interpolation is assumed. This records a pose constraint, not
a verified movement path; the elbow may need to retract before the shoulder lowers.
These coupled limits are not enforced by the manual menu-bar sliders. Runtime
animation additionally checks wrist clearance and every transition's full joint
progress box in `hardware/throttle_motion.py`.

The owner reconfirmed **1546 µs** as the lowered-shoulder elbow minimum after
reviewing the saved poses. Measurements 3 and 4 (shoulder 2272 µs, elbow 1397.25 µs)
therefore conflict with that limit and are excluded from the supervised pose tour;
the original CSV is retained as raw observations, not a list of approved poses.

On 2026-09-23 the owner subsequently confirmed recorded pose 3 (2272 / 1397.25 /
2377.75 µs) and entry/exit clearance. Runtime “arm down” now uses that exact pose.
Its exception permits shoulder travel only with the recorded elbow and wrist
fixed. On entry the worker sets the recorded elbow/wrist at the current shoulder
height when the full travel box is clear, then lowers the shoulder. Normal idle,
speech, and command poses therefore avoid a preliminary shoulder lift or wrist-up
detour. From configurations outside that clearance region (such as park), the
verified raised entry remains the fallback. Exit
raises the shoulder, then uses a raised tuck configuration for retraction.
This also supports shutdown and base retraction, including interrupted shoulder
travel. General elbow/wrist limits remain unchanged; pose 4 is not approved by
this exception. The standalone tour's established-limit filtering is unchanged.

`tools/throttle_pose_tour.py` defaults to printing a plan without opening hardware.
Its explicit `--action park` and `--action run` commands operate only throttle
channels, acquire Rex's single-instance lock to make the helper release the port,
and check pulse readback after each move. The supervised tour raises the shoulder
before adjusting elbow/wrist together, then lowers only after those joints settle.
It skips the two conflicting poses, holds each included pose for three seconds,
and finishes parked. This route assumes the owner's confirmed clear setup; it is
not an autonomous geometric collision planner. Touching `data/throttle-tour.stop`
or interrupting the process aborts remaining targets and attempts to hold current
pulses. Remove the stop file deliberately before a future run.

For the owner's explicitly requested all-recorded-poses test, the supervised runner
also accepts `--limits recorded --dwell 1`. This includes all saved tuples, including
poses 3 and 4, as exact exceptions; it does not lower the general 1546 µs elbow bound
or approve unmeasured wrist positions. The default remains `--limits established`
with a three-second dwell. Recorded mode is for the owner's observed, cleared setup,
not general autonomous operation. `--dwell` controls the hold after reaching a pose,
not travel time or the separate brief settling check.

`tools/throttle_coordinated_tour.py` adds supervised shoulder/downstream overlap.
Without `--run` it only prints a plan. With `--run` it requires the parked pose,
uses capped proportional speed/acceleration and a single Mini Maestro multi-target
packet for each movement, and runs all 11 recorded poses without intentional holds.
Six transitions combine shoulder with elbow/wrist. These use conservative clearance
boxes so safety does not depend on identical joint progress: the lowered-shoulder
box keeps elbow at least 1546 µs and wrist at 512 µs; raised/intermediate boxes use
the owner's elbow/wrist boundary observations and saved intermediate pose. This
assumes clearance improves with raising the shoulder/curling the elbow. The two
low-elbow recorded exceptions retain staged transitions. This remains a supervised,
empirical route, not a geometry-verified autonomous motion planner.

The **reach study** (`tools/throttle_coordinated_tour.py --routine reach-study`)
plans a short expressive routine: ready → curious reach → forward reach → draw back
→ offer → extend offer → gather → higher reach → relax → tuck → park. Every segment
moves shoulder and elbow together. Reaching lifts the shoulder while extending the
elbow and uncurling the wrist toward 1500 µs; withdrawal reverses that relationship.
There are no standalone wrist moves or scheduled holds. New intermediate poses are
checked against the existing clearance boxes, including independent joint progress;
they do not use the low-elbow recorded exceptions. These are gesture intentions,
not a calibrated Cartesian hand trajectory. Append `--run` only for an explicitly
requested supervised execution; default invocation just prints the plan. The start
must be parked. Plan output is saved at `data/throttle_reach_study_plan.json`.

The **reach-and-up** variation (`--routine reach-and-up`) continues the forward
reaches into three upward → forward → draw-in gestures before a single return to
park. Shoulder, elbow, and wrist move together through the upward section, with
different wrist curls and extensions on each pass. Each upward reach sweeps
directly into a forward reach before withdrawing; no scheduled holds are added.
It uses the same clearance checks and capped profiles. Its plan is saved at
`data/throttle_reach_and_up_plan.json`; append `--run` for supervised execution.

The owner verified the intended startup/shutdown destination: shoulder **2280 µs**,
elbow **2500 µs**, wrist **500 µs**. These values are recorded as `park` in the
throttle configuration. The joints retain their position through friction when
unpowered; they do not fall to a gravity-rest pose. The board's actual quantized
park is **2272 / 2496 / 512 µs**, which the runtime uses for arrival checks.

#### Main-program throttle animation

`SERVO_THROTTLE_ARM_ENABLED=true` in this robot's `.env` enables the dedicated
worker on startup and wake. It shares the main program's serial lock; it does not
open a competing serial connection. Ordinary idle/speech remains upward; explicit
person introductions briefly use the measured level forward extension.

- **Startup:** park → shoulder-clearance tuck → gently unfold upward → comfortable
  bent-elbow rest (**1050 / 1930 / 1280 µs**). Targets roughly 1.25 seconds per pose
  with a dedicated startup profile, alongside the head startup.
- **Quiet operation:** small coordinated variations around that upward rest,
  moving over roughly 2.3–3.3 seconds with 7–12 seconds between movements.
- **Speech:** four varied upward gestures, with shoulder, elbow, and wrist moving
  together over roughly 1.35–2 seconds per pose. Wrist poses span 850–2110 µs,
  with about 1000 µs of wrist travel within each paired gesture. Audio pauses of at least 0.35 seconds
  are phrase proxies; a sustained opening and sparse fallback cover longer lines.
  A 3.5–5.5 second minimum interval survives TTS sentence boundaries, so short
  acknowledgments and individual words do not each cause a gesture. These are
  acoustic timing cues, not semantic concept analysis. At speech end, the current
  pose finishes, followed by a rest return after a 1.4-second grace period. During
  uninterrupted speech, fallback gestures are eligible 6.5 seconds after the last
  gesture, without adding another full cooldown to that delay.
- **Emotion:** the existing decaying body mood blends idle and speech poses toward
  a lowered pose for sad/bored/resigned moods and a raised pose for excited/giddy,
  happy, or proud moods. Mood expiry returns the normal repertoire. Lowering stops
  at the established intermediate clearance boundary, rather than using the two
  excluded fully-down measurements. The lowered anchor is **1636 / 1550 / 1500 µs**;
  a nonlinear blend makes moderate sadness visibly lower the shoulder and forearm
  even during speech. The explicit playback emotion takes priority over body mood
  during a reply and its settling grace, matching the frame used by the LEDs.
  Expression changes log the mood, strength, and target pulses for field diagnosis.
  Pride mode curls the wrist downward to
  2254 µs, within the clearance box; it releases when that mode expires.
- **Introductions:** an accepted person introduction requests an eight-second
  level extension (**544 / 650.25 / 1484.5 µs**, from the labeled measurement).
  This takes priority over mood, Pride, and speech poses, then returns to the
  current mood. A bent-elbow rest bridge is used when a direct transition fails
  the independent-joint clearance check. Callbacks never start/recover a worker.
- **Base movement interlock:** host-issued drive, turn, move, approach and wheel
  commands wait for the arm to retract through tuck to park. Retraction uses its
  own brisk profile (`THROTTLE_RETRACT_*`): 0.9 seconds requested per segment,
  speed caps 30/70/70 and acceleration caps 6/12/12, followed by 0.5 seconds settling.
  The Maestro reports output pulses, **not actual joint positions**; a stalled
  servo cannot be detected by this guard. Missing readback, faults or a missing
  worker block movement when the throttle arm is enabled. Speech and mood cannot
  extend the arm during travel. Fresh post-command idle telemetry releases the
  hold; stale telemetry or a lost base link keeps it parked. Stop/estop and zero
  velocity commands bypass the wait and cancel pending movement. ESP32-local
  gamepad commands bypass the Mac and are outside this host-side interlock.
- **Sleep/shutdown:** head and throttle arm park concurrently. The head latches as
  soon as its rest pose is commanded, while the throttle worker may finish only
  the coupled tuck (**1636 / 2100 / 512 µs**) and park (roughly 1.5 seconds per pose).
  Serial teardown waits for completion. Late speech callbacks cannot restart the arm.

Timing/profile settings are `THROTTLE_STARTUP_*`, `THROTTLE_IDLE_*`, `THROTTLE_SPEECH_*`, and
`THROTTLE_PARK_*` in `config.py`, overridable in `user_config.py`. Integer Maestro
profiles/acceleration may lengthen the requested travel duration. All targets use
one three-channel packet and fresh pulse readback; the original hero-arm speech
behavior remains independent.

Startup accepts parked pulse readbacks or a recognized startup waypoint (within
the same 0.5 µs arrival tolerance), resuming the remaining verified startup path.
An interrupted position can also recover through tuck when every pulse is within
configured/board limits and the entire joint travel box to tuck passes clearance.
Clearance boundaries use that same tolerance to avoid accepting arrival and then
rejecting the next movement over a fraction of a microsecond. If all outputs are off, it can reassert
park only after a previously completed park recorded in the local, ignored
`data/throttle_arm_parked.json`. A missing marker, partially disabled channels, or
a pose without a verified path to tuck prevents automatic repositioning. Runtime, tour, and menu-bar
writes invalidate that marker before moving; successful runtime/tour parking (or
an explicit `throttle_pose_tour.py --action inspect` at stationary park) restores it.
Do not rely on the marker after physically repositioning an unpowered joint or
using an external controller: inspect/reestablish park first. The Maestro reports
output pulses, not shaft feedback. Manual override, an unreadable/changed serial
connection, or a movement timeout stops the worker and holds current pulses where
possible; it never replays movement after reconnect. Recovery requires a new
program run from park. Setting `SERVO_THROTTLE_ARM_ENABLED=false` disables the worker.

Profiles use Maestro units and are provisional commissioning values. The menu-bar
helper applies each joint's profile before a manual target. Runtime uses slower
idle/speech profiles, capped by each joint's commissioning profile.
`up`/`down` describe measured endpoints; `park` records the chosen destination.
Optional paired `SERVO_THROTTLE_<JOINT>_MIN_US` / `_MAX_US` overrides may narrow
these bounds to match the actual stored board limits; they cannot widen them.

For legacy shoulder-only commissioning, first disable `SERVO_THROTTLE_ARM_ENABLED`,
keep the elbow and wrist servos disconnected and
ensure their linkages are supported in a pose that clears the shoulder's movement.
Set `SERVO_THROTTLE_SHOULDER_STARTUP_US` in `.env` to the deliberately selected
startup pulse, then set `SERVO_THROTTLE_SHOULDER_ENABLED=true`. A blank or out-of-range
target stops startup when enabled. The default is disabled with no assumed pose.
On servo connect the program sends acceleration, speed, then this target on channel 8;
it sends no throttle elbow/wrist targets. The shoulder holds that target through
normal neutral/shutdown routines; there is no automatic throttle parking move.

A first pulse can cause a jump from the actual unpowered position despite speed
limits; the Maestro reports commanded pulse position, not physical joint feedback.
The software gate does not change board startup/error/home behavior or constrain
external controllers, including the menu-bar **Restart Pololu (all home)** command.
Configure board behavior accordingly before applying servo power. The runtime
repertoire stays within the tested envelope; new reach triggers will need their
own explicit choreography and clearance checks.

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
- Speaker ID uses CAM++ by default (`config.VOICE_EMBEDDER`, ONNX Runtime CPU). CAM++, ECAPA, and Resemblyzer prints are stored separately; existing prints remain intact. [Conversational enrollment](docs/conversational_voice_learning.md) learns missing CAM++ voices through normal conversation, including multiple visible people without directional audio. The local recording archive can be exported or re-embedded with `tools/voice_recordings.py`. Model details: [CAM++](docs/campplus_voice_id.md).
- Local animal/object detection uses RF-DETR nano by default (`config.OBJECT_DETECTOR_BACKEND`; Apache 2.0, ~350MB weights downloaded by `setup_assets.py`, gitignored, ~40ms/frame CPU). If it fails to load, the legacy MediaPipe EfficientDet-Lite0 detector is used automatically. No re-enrollment involved — species lists, thresholds, and the no-screens exclusion rule are backend-independent.
- On-device TTS defaults to **Breeze TTS 2 8-bit** (`mlx-community/Breeze-TTS-2-mlx-8bit`, ~4.6 GB). `setup_assets.py` downloads the selected backend into `assets/models/breeze_tts/8bit/` or `assets/models/qwen_tts/<variant>/`; weights are gitignored. Set `LOCAL_TTS_BACKEND = "qwen"` to retain the previous Qwen voice engine, then rerun setup if its weights are missing and restart Rex. ElevenLabs remains the normal online voice; the selected local engine handles `--local-tts`, offline/API fallback, and impersonations. Breeze streams ordinary Rex speech after 1.5 seconds of preroll and prepares complete impersonations before playback. The model is loaded lazily unless `--local-tts` or `LOCAL_TTS_WARM_ON_BOOT` requests preloading. Reference clips and exact transcripts under `assets/voices/rex/` and `famous/` are tracked; captured `people/` voices stay gitignored. The Breeze reference is the test bench's matched 24 kHz `RX24-pure-24k` pair. Breeze weights are licensed for research and non-commercial use; see the [model card](https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-8bit). Details, tuning, and offline validation: [local TTS backends](docs/local_tts_backends.md).
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

#### Pride rainbow LEDs

While Pride mode is active, the head mouth displays a rotating rainbow swirl
across its serpentine 10×8 PCB. Speaking retains the audio-reactive equalizer
shape and brightness caps; quiet operation retains the dim breathing glow.
The eyes keep their normal colors. The chest covers all 98 pixels in a moving
rainbow, slower/dimmer at idle and faster/brighter during speech.

Both sketches accept `PRIDE:1` / `PRIDE:0` as color overlays. The host refreshes
the active overlay every 1.5 seconds, including during speech; firmware drops
it after ten seconds without refresh. Sleep, off, charging, startup (chest),
and shutdown fade retain their own behavior. The host uses the existing
`intelligence.pride` mode/expiry; no separate activation flag is required.

Firmware: `arduino/head_nano` (this head is detected as **Uno**, build target
`arduino:avr:uno`) and `arduino/chest_nano` (`arduino:avr:nano:cpu=atmega328`, newer bootloader).
The mouth rainbow swaps R/G for its GRB PCB on the shared RGB eye data line;
the chest uses its existing FastLED GRB configuration.

#### Spoken throttle-arm poses

Runtime motion uses per-joint pace multipliers: shoulder 1.375×, elbow 1.76×,
wrist 2.2× the original profiles (10% above the previous tuning), including startup, idle, speech, poses, parking,
and base retraction. Speed and acceleration caps scale with the pace; joints
can finish at different times within the existing independent-progress clearance
checks. Holds and pauses are unchanged. These are commanded profile increases,
not measured physical speed guarantees.

- “Put your arm down” / “lower your arm”: recorded pose 3, with shoulder,
  forearm, and wrist fully down (2272 / 1397.25 / 2377.75 µs).
- “Hold out your hand” / “extend your arm” / “outstretch your arm” /
  “stretch your arm out” / “reach out your hand” / “outstretch your hand” /
  “stretch out your hand” / “stretch your hand out” / “put your hand straight out” /
  “hold your hand straight out”: the measured level forward reach.
- “Give me a high five” / “raise your hand” / “raise your arm” / “lift your arm” /
  “put your arm up”: the measured raised-hand pose.
- “Relax your arm” / “pull your arm back” / “return your arm to neutral”: the
  normal bent-elbow rest pose.

High five coordinates all three joints with the same proportional pace (1.375×).
The owner confirmed simultaneous movement from full-down: this directed recorded
transition now uses one atomic target batch, without a preliminary shoulder lift
or tuck detour. The reverse lowering transition retains its staged route.

Commands accept polite prefixes and hold for ten seconds **after arrival**, then
return to emotional animation. These are poses, not contact-detecting gestures.
Accepted arm commands use a brief “OK,” “Alright,” or “Sounds good” acknowledgment
instead of repeating the requested movement. Unavailable motion is still reported.
The worker owns every target; sleep/shutdown, manual overrides, and the base
retraction interlock take priority. A new pose replaces the pending hold. If the
worker is unavailable or the base owns the arm, Rex says he cannot move it.

“Stand down pride mode”, “turn off pride mode”, “stop pride mode”, and “end pride
mode” immediately clear the voice/body overlay and send `PRIDE:0` to both LED
boards. No firmware change is needed for these new spoken commands.
“Standdown pride mode”, “standown pride mode”, and the observed ASR mishearing
“scan down pride mode” also execute that same exit command, as do “disable pride
mode” and “deactivate pride mode”.

Additional arm phrasing includes “reach for the sky,” “hand up,” “raise your
throttle arm,” “put your hand out,” “reach forward,” “hold your arm out in
front,” “arm back down,” “let your arm hang,” “lower your throttle arm,”
“bring your arm in,” “pull your hand in,” and “back to your resting pose.”

“Hold that pose” / “keep that pose” holds the pose after the current movement
finishes without a timer. “You can relax now” / “resume arm animation” releases
it. “A little higher/lower” requests a 60 µs shoulder adjustment, limited to
the normal raised/intermediate range and checked against the entire travel
clearance box; at a limit it stays in place. Adjusted poses hold ten seconds.
Base movement and shutdown still override all holds.
