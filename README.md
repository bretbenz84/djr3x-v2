# DJ-R3X v2

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It combines speech recognition, text-to-speech, camera awareness, face and voice memory, conversational social behaviors, hardware control, music playback, and verbal games into one interactive companion system.

The project is built for live, in-room use: Rex can recognize people, remember details across sessions, react to arrivals, handle introductions, play games like Jeopardy, answer visual prompts, and drive connected LEDs and servos when the hardware is attached.

## Features

- Wake-word and always-listening conversation flow
- Local Whisper transcription with OpenAI fallback support
- ElevenLabs TTS with cached speech output
- Camera-based scene, face, appearance, and animal awareness
- Voice and face enrollment for known people
- Persistent memory database for people, relationships, preferences, and events (`people.db`)
- Rex's own first-person episodic memory (`rex.db`) — a timestamped log of his experiences (people seen, scenes observed, things he did, per-session conversation summaries)
- Social intelligence layers for repairs, boundaries, grief, celebrations, callbacks, and group discretion
- Mood-driven body language — Rex's posture (head lift/tilt, visor openness, breathing, idle gestures) reflects a sustained "body mood" that shifts when he's complimented, insulted, or amused, riding on top of face-tracking
- A wandering attention of his own — when the conversation lulls he'll stop staring, glance around the room, then look back and sometimes re-greet, so he doesn't feel locked to a fixed stare
- Bored environmental snark — left idle, he looks around and invents in-character jabs about the room he actually sees: complaints about how dull it is, faux-clueless questions about objects ("what's that black chair for?"), digs at the clutter, snobby art opinions, or pleas to be taken somewhere with more life forms
- Servo and LED hardware hooks for a physical droid body
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
- Optional replacement of `ELEVENLABS_VOICE_ID` in `config.py`
- Optional guided droid hardware setup for the chest Arduino, head LED Arduino, and Pololu Maestro
- Arduino CLI, Arduino AVR core, and FastLED setup for uploading the included LED firmware
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

Instead of starting `main.py` by hand, you can have macOS stay quietly ready and launch the robot by voice. A tiny LaunchAgent (`rex_supervisor.py`) listens only for **"wake up Rex"** and starts the full controller on demand; **"shut down"** powers it back down while the listener keeps running. Install with `scripts/install_supervisor.sh`. See **[docs/supervisor.md](docs/supervisor.md)** for how it works and how the single-instance lock prevents a double-launch (including when Rex is asleep).

## Configuration

User-tunable defaults live in [config.py](config.py). API keys should stay in `apikeys.py`, and host-specific hardware paths plus build-specific servo limit overrides should stay in `.env`; both are intentionally excluded from git.

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

## Project Layout

```text
audio/          Speech input, VAD, transcription, TTS, playback, and audio scene logic
awareness/      Time, holidays, interoception, and background awareness systems
features/       Games, music, commandable behaviors, and interactive features
hardware/       Servo and LED integrations
intelligence/   Conversation, memory, LLM prompting, empathy, and social behavior layers
vision/         Camera, face recognition, scene analysis, and image utilities
assets/         Models, audio, game assets, memory database, and cached generated assets
logs/           Runtime logs
```

## Notes

- The program can run with missing droid hardware, but servo and LED features will be disabled until the configured devices are connected.
- Logs are written to `logs/djr3x.log` and `logs/conversation.log`.
- Real API keys should never be committed.
- Two SQLite databases under `assets/memory/` (both gitignored, both created by `setup_assets.py`):
  - `people.db` — what Rex knows **about people** (faces, voices, facts, interests, events, conversation summaries per person).
  - `rex.db` — Rex's own **episodic memory** (his "diary"): a timestamped, first-person log of experiences. He records people seen, scenes observed ("the room was cluttered"), things he did ("I made Bret laugh", "I saw a dog"), people he **met** ("I met Bret."), **visits** ("I spent about 40 minutes with Bret."), **games** ("I played Trivia with Bret — scored 4 out of 5."), **boundaries** people set ("Bret asked me not to ask about his ex."), **emotional check-ins**, **celebrity** sightings (the Jeff/JT easter eggs), and memorable greeting moments — **birthdays, milestones, celebrations, and long-absence reunions** — plus an LLM session summary saved on shutdown. **Phase 1 is capture-only** — these are logged for later use but nothing reads them back into Rex's behavior yet. Toggle with `config.EPISODIC_MEMORY_ENABLED`.

## License

Except where otherwise noted, this project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE). You may share and modify it with attribution for noncommercial purposes, but commercial use is not permitted without prior written permission.

DJ-R3X v2 is an unofficial fan project. It is not affiliated with, endorsed by, or sponsored by Disney, Lucasfilm, OpenAI, ElevenLabs, Jeopardy Productions, or any other referenced rights holder. Third-party names, trademarks, sound clips, clue data, models, libraries, and other materials remain the property of their respective owners and may be subject to separate terms.

## Authors

- Bret Benziger
- OpenAI Codex
- Claude Code
