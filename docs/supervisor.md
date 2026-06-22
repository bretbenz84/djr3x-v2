# Always-on wake-word supervisor

This lets your Mac stay quietly ready to hear **"wake up Rex"** without the full
robot running. Saying the phrase plays a startup chime (instant "I heard you"
feedback) and launches the controller (`main.py --gui`, so the dashboard opens on every
wake); saying **"shut down"** powers it back
down — while the listener keeps running. Test the chime any time with
`venv/bin/python rex_supervisor.py --test-chime`.

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
| **"wake up Rex"** (while off) | Supervisor launches `main.py --gui` (dashboard opens on every wake) |
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

**Logs — kept separate from the controller's:**
- `logs/supervisor.out.log` / `logs/supervisor.err.log` — the supervisor's OWN output
  (launchd-captured): wake-word diagnostics, launch/dormant transitions. These contain
  *only* supervisor activity.
- `logs/controller.console.log` — the launched controller's (`main.py`) raw stdout/stderr,
  redirected here per launch (truncated each time) so it never floods the supervisor logs.
  Mostly a duplicate of the controller's own structured log plus any pre-logging boot/crash
  output.
- `logs/djr3x.log` — the controller's full, rotated structured log (written by `main.py`).

## Tunables (environment variables)

| Var | Default | Meaning |
| --- | --- | --- |
| `REX_SUPERVISOR_WAKE_THRESHOLD` | `0.7` | `wakeuprex.onnx` confidence required to trigger. A clean "wake up rex" scores ~0.99 and background TV/ambient tops out around 0.12, so 0.7 has wide margin both ways. Lower it only if your real phrase won't cross it (`--meter` shows your live score); raise it if anything still false-triggers. |
| `REX_SUPERVISOR_WAKE_CONSECUTIVE` | `3` | How many consecutive 80 ms frames must clear the threshold before firing. A real phrase holds the score near 1.0 for ~10 frames in a row; a TV phonetic near-miss is a 1-2 frame spike, so this rejects background-audio false triggers. Raise toward 4-5 if a noisy room still trips it; 1 disables the debounce. |
| `REX_SUPERVISOR_DEBUG` | unset | Set to `1` for verbose per-frame logging |
| `REX_SUPERVISOR_CHIME` | `1` | Play `startup_chime.mp3` the instant a wake is accepted (instant feedback before the robot boots). Set `0` to disable. |
| `DJR3X_LOCK_PATH` | `<tmpdir>/djr3x-main.lock` | Single-instance lock location (must match between supervisor and `main.py`) |
| `DJR3X_SKIP_SINGLE_INSTANCE` | unset | Set to `1` to let `main.py` skip the lock (manual dev runs) |
| `AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX` | from `.env` | Mic the supervisor listens on (same keys `main.py` uses) |

## How wake detection works (and why)

It's deliberately simple: the supervisor reads 80 ms mic frames, mixes them to
mono, and runs the `wakeuprex.onnx` openWakeWord model. It fires (chime + launch)
when the score clears `REX_SUPERVISOR_WAKE_THRESHOLD` for `REX_SUPERVISOR_WAKE_CONSECUTIVE`
frames in a row. No VAD, no Whisper, no transcription — the robot is OFF while the
supervisor listens, so the only job is to spot one wake word and launch `main.py`.

**Avoiding false triggers (e.g. the TV):** a real "wake up rex" pegs the model
near 1.0 for ~10 consecutive frames; background TV/ambient rarely exceeds ~0.12,
and the occasional phonetic near-miss is just a 1-2 frame blip. Two cheap guards
exploit that gap: the threshold default is **0.7** (well above ambient, well
below a real phrase), and firing requires a **sustained run** of frames over the
bar (default **3**), not a single spike. If the TV still trips it, raise either
knob; if a real phrase stops triggering, lower the threshold.

**The bug that made it look like the ONNX model "didn't work":** openWakeWord's
melspectrogram front-end is trained on **16-bit PCM** (range ±32767), but
sounddevice hands us **float32 in [-1, 1]**. Feeding that raw float makes the
model see near-silence, so every score pins at ~0.001 and nothing ever fires.
The supervisor now rescales each frame to int16 (`_to_oww_input`) before
`predict()`, and a clean "wake up rex" scores ~0.99. If you ever see scores
stuck near zero while the mic clearly has audio, that scaling is the first thing
to check (the main app's `audio/wake_word.py` has the same latent issue).

## Checking the microphone

First confirm which input the supervisor will use and that audio is arriving —
no launchd, no robot launch:

```bash
venv/bin/python rex_supervisor.py --list-devices   # which mic is selected
venv/bin/python rex_supervisor.py --meter          # live input-level bar; speak
```

`--list-devices` prints every input device and marks the one the supervisor
resolved from `.env`. `--meter` opens that mic and shows a live RMS bar **and the
live wakeuprex score** — speak and the bar should jump; say "wake up rex" and the
score should spike toward 1.0. If the bar stays flat/zero the supervisor isn't
getting audio (permission or wrong device); if the bar moves but the score never
rises, it's detection (threshold or the int16 scaling described below).

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
- **`mic rms` moves but `peak onnx score` stays near 0.001** → audio is arriving
  but the model sees near-silence. This is the int16-scaling bug (see "How wake
  detection works"); the supervisor rescales with `_to_oww_input`, so if you see
  this, that step is missing/broken. Confirm with `--meter` (it shows the live
  score) and that `wakeuprex.onnx` exists under `assets/models/wake_word/`.
- **`peak onnx score` rises but never crosses the threshold** → say "wake up rex"
  more clearly/closer, or lower `REX_SUPERVISOR_WAKE_THRESHOLD` toward the peak
  you see in `--meter`.
- **Nothing logs at all / it says "dormant"** → a controller (`main.py`) is
  already running or asleep and holding the lock, so the supervisor is
  intentionally silent. Shut down `main.py` first.

## Running the supervisor manually (debug)

```bash
venv/bin/python rex_supervisor.py                          # listen + launch on wake
REX_SUPERVISOR_DEBUG=1 venv/bin/python rex_supervisor.py   # verbose
```
