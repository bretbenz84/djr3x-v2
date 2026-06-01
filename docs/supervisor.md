# Always-on wake-word supervisor

This lets your Mac stay quietly ready to hear **"wake up Rex"** without the full
robot running. Saying the phrase launches the controller; saying **"shut down"**
powers it back down — while the listener keeps running.

## The two processes

| Process | What it is | Lifetime |
| --- | --- | --- |
| `rex_supervisor.py` | Tiny always-on listener. Loads ONLY `wakeuprex.onnx`. | Whole login session (LaunchAgent) |
| `main.py` | The full DJ-R3X controller (audio, vision, servos, LLM…). | From "wake up Rex" until "shut down" |

The supervisor is intentionally minimal: it does **not** import the project
config or need API keys, so it starts cleanly at login and just waits.

## How they coordinate (no double-launch)

`main.py` holds a single-instance lock (`utils/single_instance.py`, an `flock`)
for its **entire** lifetime — including while it is merely **asleep**. The
supervisor checks that lock and stays **dormant** whenever a controller is alive:

```
main.py awake   → lock held → supervisor dormant (main.py owns the mic)
main.py asleep  → lock held → supervisor dormant (main.py's OWN "wake up rex"
                              detector wakes it — we must NOT spawn a 2nd one)
no main.py      → lock free → supervisor listens for "wake up rex"
```

Only one process opens the microphone at a time, so there's no contention. If
`main.py` ever crashes, the OS frees the `flock` automatically and the supervisor
resumes listening on its own.

## Voice commands

| You say | Result |
| --- | --- |
| **"wake up Rex"** (while off) | Supervisor launches `main.py` |
| **"go to sleep"** / "good night" | `main.py` stays running, asleep; only its own "wake up Rex" wakes it |
| **"wake up Rex"** (while asleep) | `main.py` wakes itself (handled internally, not by the supervisor) |
| **"shut down"** / "shut down Rex" / "power down" | `main.py` exits cleanly; supervisor resumes listening |

"Shut down" only triggers on a short, direct phrase — narration like *"I had to
shut down my old server"* or *"shut down the music"* will **not** power off the
droid.

## Install

```bash
scripts/install_supervisor.sh           # render plist for this repo + load it
scripts/install_supervisor.sh status    # check it's running
scripts/install_supervisor.sh uninstall # stop + remove
```

The installer substitutes this repo's absolute path into
`launchd/com.djr3x.supervisor.plist.template` and installs the result to
`~/Library/LaunchAgents/com.djr3x.supervisor.plist`.

**First run:** macOS will prompt for **Microphone** permission for the venv
Python. Grant it (System Settings → Privacy & Security → Microphone). The robot's
camera/automation prompts still appear the first time `main.py` itself runs.

Logs: `logs/supervisor.out.log` and `logs/supervisor.err.log`.

## Tunables (environment variables)

| Var | Default | Meaning |
| --- | --- | --- |
| `REX_SUPERVISOR_WAKE_MODE` | `both` | How wake is detected: `transcribe` (VAD + local Whisper, reliable), `onnx` (wakeuprex.onnx score only), or `both` |
| `REX_SUPERVISOR_WAKE_THRESHOLD` | `0.5` | onnx confidence to trigger (only used by the `onnx`/`both` paths) |
| `REX_SUPERVISOR_DEBUG` | unset | Set to `1` for verbose per-frame logging |
| `DJR3X_LOCK_PATH` | `<tmpdir>/djr3x-main.lock` | Single-instance lock location (must match between supervisor and `main.py`) |
| `DJR3X_SKIP_SINGLE_INSTANCE` | unset | Set to `1` to let `main.py` skip the lock (manual dev runs) |
| `AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX` | from `.env` | Mic the supervisor listens on (same keys `main.py` uses) |

## How wake detection works (and why)

The custom `wakeuprex.onnx` model is unreliable on its own — it often never
crosses the score threshold. The full robot doesn't trust it either: when it
wakes from the "go to sleep" state it uses **VAD + the local Whisper model** and
matches the transcribed phrase, not the ONNX score. The supervisor mirrors that.

By default (`both`) it runs two detectors in parallel and fires on either:

1. **transcribe** — Silero VAD detects a spoken phrase, the local `mlx-whisper`
   model transcribes it, and it fires if the text is "wake up Rex" (and close
   variants: "rex wake up", "wake up r3x", …). This is the reliable path.
2. **onnx** — the `wakeuprex.onnx` confidence score crossing the threshold.

If you want the lighter, lower-CPU behavior and your ONNX model works well for
you, set `REX_SUPERVISOR_WAKE_MODE=transcribe` to skip the (idle) Whisper work,
or `=onnx` to skip transcription.

## Checking the microphone

First confirm which input the supervisor will use and that audio is arriving —
no launchd, no robot launch:

```bash
venv/bin/python rex_supervisor.py --list-devices   # which mic is selected
venv/bin/python rex_supervisor.py --meter          # live input-level bar; speak
```

`--list-devices` prints every input device and marks the one the supervisor
resolved from `.env`. `--meter` opens that mic and shows a live RMS bar — speak
and it should jump; if it stays flat/zero, the supervisor isn't getting audio
(permission or wrong device), not a detection problem.

Notes on device selection (these were real "no trigger" causes):
- `AUDIO_DEVICE_NAME` in `.env` may be quoted (`"MacBook Pro Microphone"`); the
  supervisor strips the quotes and matches case-insensitively / by substring,
  the same as the main app. If the name doesn't match any input it logs the
  available list and falls back to `AUDIO_DEVICE_INDEX`, then the system default.
- Multi-channel mics (the **ReSpeaker Lite** is 2-in) are opened with their real
  channel count and mixed to mono. Forcing such a device to 1 channel can yield
  silence — which is why a "correct" ReSpeaker produced no wake trigger.

## Troubleshooting: "I said 'wake up Rex' and nothing happened"

The supervisor logs to `logs/supervisor.out.log` / `logs/supervisor.err.log`. It
prints a `[diag]` line every 5 seconds while listening, showing the peak ONNX
score and the mic RMS. Run it by hand to watch live:

```bash
venv/bin/python rex_supervisor.py
# then say "wake up rex" and watch the output
```

- **`mic rms` stays ~0.0000 even while you talk** → it's not hearing the mic.
  Run `--meter` to confirm. Check Microphone permission for the venv Python
  (System Settings → Privacy & Security → Microphone) and that
  `AUDIO_DEVICE_NAME`/`INDEX` in `.env` point at the right input (`--list-devices`).
  Under launchd the permission prompt may not surface; running by hand once
  forces it.
- **`mic rms` moves but `peak onnx score` stays low** → the ONNX model isn't
  firing (expected; that's why `transcribe` is the default). You should see a
  `Heard … → 'wake up rex' (wake=True)` line from the transcription path; if you
  don't, confirm the local Whisper model exists (`assets/models/whisper/config.json`).
- **Nothing logs at all / it says "dormant"** → a controller (`main.py`) is
  already running or asleep and holding the lock, so the supervisor is
  intentionally silent. Shut down `main.py` first.

## Running the supervisor manually (debug)

```bash
venv/bin/python rex_supervisor.py            # default mode=both
REX_SUPERVISOR_DEBUG=1 venv/bin/python rex_supervisor.py   # verbose
```
