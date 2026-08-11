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

## Automatic Git updates

The supervisor keeps the physical robot's `main` checkout current with
`origin/main` without writing any updater state files:

1. **Supervisor startup:** before loading the wake model or opening the mic, it
   fetches and fast-forwards, then replaces itself if code changed so the newly
   pulled supervisor is what stays resident.
2. **Every four hours:** it fetches `origin/main`. While `main.py` is running or
   asleep this is check-only—the working tree is never changed beneath a live
   controller. When Rex is off, it may fast-forward and restart itself.
3. **Every controller launch:** after the wake chime and before `main.py --gui`,
   it performs one final fetch/fast-forward so the controller starts directly
   from the newest code.

Updates are deliberately conservative: the checkout must be on `main`, the
worktree must be clean, and local `HEAD` must be an ancestor of `origin/main`.
The updater uses a fast-forward-only merge and never stashes, resets, discards
changes, or creates merge commits. Network/Git failures only produce a warning;
Rex launches the installed version instead of becoming unavailable offline.
Machine-local `.env`, `apikeys.py`, `user_config.py`, databases, models, caches,
and logs remain untouched through the project's existing ignore rules.

No periodic timestamp or "update pending" marker is stored. The four-hour timer
lives in supervisor memory, and the fetched `origin/main` ref is sufficient to
tell whether an update is waiting.

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

## Launching `main.py` by hand (the serial-port handoff)

```bash
venv/bin/python main.py
```

You do **not** have to quit or unload the battery / servo / LED menu bar apps
first. It is worth being precise about why, because the obvious mental model is
wrong: **the supervisor never touches a serial port.** It only spawns `main.py`.
The thing every menu bar app actually watches is the single-instance flock — each
one polls it ~1×/s and closes its port the moment a controller holds it. A
hand-typed launch takes the same lock, so it gets the same handoff.

What a manual launch used to lack was the *pause*. `main.py` took the lock and
opened the ports inside the same second, ahead of that 1 Hz poll.
`hardware/servos.py` and `hardware/motion.py` absorbed it with their 3×1 s
connect retries; `leds_head`/`leds_chest` opened exactly once, so a launch that
overlapped the LED console holding the boards lost the head and chest LEDs for
the whole session while the servos and base came up fine. Worse, the failure was
usually silent: pyserial only takes an advisory `flock` when `exclusive=True`
(which the menu bar apps pass and the robot's drivers do not), so a `/dev/cu.*`
port can genuinely be open in two processes at once, interleaving bytes rather
than raising.

So `main.py` now performs the handoff itself, at `=== Initializing hardware ===`
and before the first `connect()` (`utils/port_handoff.py`):

1. Ask `lsof` who holds `MAESTRO_PORT`, `ARDUINO_HEAD_PORT`,
   `ARDUINO_CHEST_PORT`, `MOTION_ESP32_PORT` — only the ones this run will
   actually open (`--noservos`, `MOTION_ENABLED=False` and unset ports are
   skipped).
2. Free? Return immediately. One `lsof`, ~40 ms, nothing logged. That is every
   supervisor-launched wake.
3. Held? Log who holds what by script name and pid, then poll until they let go
   — typically ~0.3 s.
4. Still held at `SERIAL_HANDOFF_TIMEOUT_SECS`? Log a warning naming the stuck
   port and connect anyway. The handoff never blocks startup.

Detection is `lsof` on purpose: it never opens the device (an open reboots both
Arduinos, so probe-by-open would add a board reset to every startup) and it sees
*any* holder — an Arduino IDE serial monitor or a stray `tools/` script counts
too. If `lsof` can't be run at all, it pauses 1.5 s (one companion poll) and
continues rather than assuming the ports are free.

The head and chest Arduino opens now retry 3× like the Maestro and motion base
(`HEAD_/CHEST_ARDUINO_CONNECT_RETRY_ATTEMPTS`), as a backstop behind the wait.

`main.py` logs which path started it — `Single-instance lock acquired (…) —
manual launch` vs `supervisor launch`. That is diagnostics only; both paths do
the identical handoff.

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

The installer substitutes this repo's absolute path into the templates under
`launchd/` and installs the results to `~/Library/LaunchAgents/`. It installs
**four** agents: `com.djr3x.supervisor` (the wake-word listener),
`com.djr3x.battery` (the menu bar battery meter, below — skipped when
`MOTION_ESP32_PORT` isn't set in `.env`), `com.djr3x.servo` (the Servo Control
console — skipped when `MAESTRO_PORT` isn't set), and `com.djr3x.led` (the LED
Control console, below — skipped when neither `ARDUINO_HEAD_PORT` nor
`ARDUINO_CHEST_PORT` is set). `uninstall` removes all of them.

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

## Menu bar battery meter

`tools/rex_battery_menubar.py` (LaunchAgent `com.djr3x.battery`) keeps the ESP32
drive base's **charge / voltage / current** visible in the macOS menu bar even
while the robot is off — `🔋 78%` normally, `🪫` at ≤20%, `⚡` while charging,
with voltage, current, power, base state/fault, and reading age in the dropdown.

It needs no firmware or protocol changes because the motion firmware streams a
telemetry frame at 10 Hz from the moment it boots — before any handshake — and
every frame already carries `batt_mv` / `batt_ma` / `batt_soc`
(see docs/motion_protocol.md §6.1). The app opens `MOTION_ESP32_PORT` and is
**passive with one exception**: it never sends motion commands, so it can't
claim ownership or interfere with the base. The exception is the **"Set Battery
to 100%"** menu item (shown only while the meter owns the port): when you watch
your charger's taper current hit cutoff — the definitive "pack is full" signal,
which the firmware can't see on its own because a charging pack is never at
rest — one click sends the `batt_full` command (protocol §5.11) and the ESP32
sets its coulomb ledger to 100% and persists it to NVS. The SOC in the dropdown
snaps to 100% within a second or two as confirmation. Terminal equivalent:
`venv/bin/python tools/rex_battery_menubar.py --mark-full`.

**Port sharing with the robot** uses the same flock the supervisor uses for the
microphone. Serial ports are exclusive-open, so the app polls the single-instance
lock about once a second:

- `main.py` alive (awake **or** asleep) → the app closes the port and shows the
  last reading under a `🤖` icon ("Rex is running — port handed over").
- no `main.py` → it reopens the port and resumes the live meter.

`main.py` takes the lock at startup, well before motion connects, and
`hardware/motion.py` opens with retries — so the ≤1 s release lag is absorbed.
The flock frees itself if `main.py` crashes, so the meter recovers on its own.

Charger transitions have matching audio in both ownership modes. While `main.py`
is running, its battery-awareness loop plays `droid_gaining_electric.mp3` on
plug-in and `droid_losing_electric.mp3` on unplug. While Rex is off, the battery
companion already owns the only serial connection, so it edge-detects the same
debounced charging state and calls the supervisor's output-routed audio helper.
The first reading after either process starts or reacquires the port is only a
silent baseline; stable telemetry never repeats a cue. A 14.0 V fallback covers
a full attached pack after charging current has tapered to zero.

The same off-state worker drives the chest Nano's `CHARGE:<soc>:<attached>` mode
only while `main.py` is not running. The first eight LEDs of each of panels A/B/C
form ONE contiguous 24-LED meter split into thirds: panel A holds the red/orange
low end, panel B the yellow/green middle, panel C the green/blue top, each bar
filling bottom-up. Every pixel has a fixed colour tied to its position; the
charge only decides how many are lit (pixel k lights above (k-1)*100/24 %), so
as the pack drains, pixels go dark one by one from the blue end down. The
topmost lit pixel blinks (~1 Hz) until it goes off. When attached, a cyan-white
packet also climbs from the fill boundary to the top of the whole meter. When
R3X starts, the battery companion releases ownership and normal startup/active
chest animations replace the gauge; it is never shown while `main.py` is alive,
including the controller's sleep state.

While off and attached to the charger, the mouth PCB also receives
`CHARGE:<soc>` and keeps its eyes dark while the mouth breathes slowly in an SOC
color: red at 0–25%, orange at 26–50%, yellow at 51–75%, green at 76–90%, and
blue at 91–100%. Unplugging sends `OFF`. As with the chest gauge, this never runs
while `main.py` is alive.

Both of those displays need a *brief* open of an Arduino that no one else holds,
so a second app sitting on either port makes them silently impossible. If an open
fails, the meter logs one WARNING per outage (naming the LED console as the usual
culprit), forgets its dedup baseline, and retries every 5 s — so the display
repaints within seconds of the port coming free instead of waiting for the SOC to
move or for the 30-minute refresh.

Bring-up check without the GUI (prints raw battery frames):

```bash
venv/bin/python tools/rex_battery_menubar.py --probe
```

Its logs are `logs/battery_menubar.out.log` / `.err.log`.

## Menu bar LED console

`tools/rex_led_menubar.py` (LaunchAgent `com.djr3x.led`) puts an **"LED
Control"** item in the menu bar. The dropdown has two areas — **HEAD** and
**CHEST** — each with a live status row and one clickable button per animation
that board's firmware supports (the same newline commands `hardware/leds_head.py`
and `hardware/leds_chest.py` send): idle, active, every speak emotion, speak
stop, sleep, fade-off, off, plus chest-only startup, compliment flash,
next-pattern, and a 50 % battery-meter demo. Use it to audition any animation
while the robot is off.

Head speak animations are normally driven by a `SPEAK_LEVEL` audio stream from
`main.py`; the app substitutes a synthetic level wave while a head Speak button
is active so the mouth equalizer actually dances. Clicking any other head
button stops the wave.

**Battery Meter Mode** is the checkbox at the top of the dropdown, and it is
**on by default**. Checked, the console releases both Arduinos and shows
"🔋 battery meter owns this board" in each status row, which is what lets the
battery companion above paint the chest charge gauge and the mouth's SOC glow
while Rex is off. Unchecked, this console holds both ports and the animation
buttons are live — but nothing is drawing the charge display. Clicking any
animation button unchecks it for you (a dead button would be worse than an
implied takeover); the click itself is held in the queue and fires once the
board finishes its ~2 s reboot. The choice persists in
`assets/state/led_console_mode.json`, so if you leave the console in control
mode, it is still in control mode after the next login — the chest stays dark on
the charger until you check the box again. That is exactly how the gauge went
missing for two days (owner 2026-08-04).

**Port sharing with the robot** follows the battery/servo pattern: each zone
polls the single-instance flock ~1×/s, releases its serial port while `main.py`
is alive (buttons go inert, status shows "Rex is running"), and reopens it when
Rex shuts down. `main.py` outranks the mode checkbox — while Rex is alive the
ports are his either way. Opening a port reboots that Arduino, so the app waits
~2 s after each (re)open before sending. Its logs are `logs/led_menubar.out.log`
/ `.err.log`.

## Tunables (environment variables)

| Var | Default | Meaning |
| --- | --- | --- |
| `REX_SUPERVISOR_WAKE_THRESHOLD` | `0.7` | `wakeuprex.onnx` confidence required to trigger. A clean "wake up rex" scores ~0.99 and background TV/ambient tops out around 0.12, so 0.7 has wide margin both ways. Lower it only if your real phrase won't cross it (`--meter` shows your live score); raise it if anything still false-triggers. |
| `REX_SUPERVISOR_WAKE_CONSECUTIVE` | `3` | How many consecutive 80 ms frames must clear the threshold before firing. A real phrase holds the score near 1.0 for ~10 frames in a row; a TV phonetic near-miss is a 1-2 frame spike, so this rejects background-audio false triggers. Raise toward 4-5 if a noisy room still trips it; 1 disables the debounce. |
| `REX_SUPERVISOR_DEBUG` | unset | Set to `1` for verbose per-frame logging |
| `REX_SUPERVISOR_CHIME` | `1` | Play `startup_chime.mp3` the instant a wake is accepted (instant feedback before the robot boots). Set `0` to disable. |
| `REX_AUTO_UPDATE_ENABLED` | `1` | Fetch and fast-forward `origin/main` at supervisor startup, periodically, and before controller launch. Set `0` to disable. |
| `REX_AUTO_UPDATE_INTERVAL_SECS` | `14400` | Periodic update-check interval (four hours; minimum 60 seconds). |
| `REX_AUTO_UPDATE_TIMEOUT_SECS` | `45` | Maximum time allowed for each individual Git operation before startup falls back to installed code. |
| `DJR3X_LOCK_PATH` | `<tmpdir>/djr3x-main.lock` | Single-instance lock location (must match between supervisor and `main.py`) |
| `config.SERIAL_HANDOFF_TIMEOUT_SECS` | `5.0` | Ceiling on the serial-port handoff wait at hardware init (see above). A ceiling, not a delay: the wait ends the instant the ports are free. Set `0` to skip waiting and rely on the drivers' connect retries alone. |
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
