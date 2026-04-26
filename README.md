# DJ-R3X v2

DJ-R3X v2 is a local, voice-first droid brain inspired by Rex from Star Tours and Oga's Cantina. It combines speech recognition, text-to-speech, camera awareness, face and voice memory, conversational social behaviors, hardware control, music playback, and verbal games into one interactive companion system.

The project is built for live, in-room use: Rex can recognize people, remember details across sessions, react to arrivals, handle introductions, play games like Jeopardy, answer visual prompts, and drive connected LEDs and servos when the hardware is attached.

## Features

- Wake-word and always-listening conversation flow
- Local Whisper transcription with OpenAI fallback support
- ElevenLabs TTS with cached speech output
- Camera-based scene, face, appearance, and animal awareness
- Voice and face enrollment for known people
- Persistent memory database for people, relationships, preferences, and events
- Social intelligence layers for repairs, boundaries, grief, celebrations, callbacks, and group discretion
- Servo and LED hardware hooks for a physical droid body
- Music controls and verbal games, including Jeopardy-style play

## Requirements

- macOS, preferably Apple Silicon
- Terminal access
- Git
- Internet access for setup and model downloads
- OpenAI API key
- ElevenLabs API key
- Optional hardware:
  - Pololu Maestro servo controller
  - Head and chest Arduino controllers
  - Camera and microphone

The macOS setup script installs Homebrew dependencies, pyenv, Python 3.11.9, the virtual environment, Python packages, config templates, assets, models, and database setup.

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

You need to activate the virtual environment in every new terminal session before running project commands.

## Configuration

User-tunable settings live in [config.py](config.py). API keys should stay in `apikeys.py`, and host-specific hardware paths should stay in `.env`; both are intentionally excluded from git.

Useful setup checks:

```bash
ls /dev/tty.usb*
```

Use that to find connected serial devices for Maestro and Arduino hardware. For camera setup, update `CAMERA_INDEX` or `CAMERA_DEVICE_NAME` in `.env`.

## Servo Safety

Do not connect a Pololu Maestro to live servos until the servo limits have been configured in the Maestro Control Center app. Set safe minimum, maximum, neutral, speed, and acceleration values there first, write those values down, and then update the matching servo limits in [config.py](config.py).

Connecting the Maestro before limits are programmed can drive a servo past its safe travel range and damage the mechanism.

## Project Layout

```text
audio/          Speech input, VAD, transcription, TTS, playback, and audio scene logic
awareness/      Time, holidays, interoception, and background awareness systems
features/       Games, music, commandable behaviors, and interactive features
hardware/       Servo and LED integrations
intelligence/   Conversation, memory, LLM prompting, empathy, and social behavior layers
vision/         Camera, face recognition, scene analysis, and image utilities
assets/         Models, audio, Jeopardy data, memory database, and cached generated assets
logs/           Runtime logs
```

## Notes

- The program can run with missing droid hardware, but servo and LED features will be disabled until the configured devices are connected.
- Logs are written to `logs/djr3x.log` and `logs/conversation.log`.
- Jeopardy data lives in `assets/jeopardy`.
- Real API keys should never be committed.

## Authors

- Bret Benziger
- OpenAI Codex
