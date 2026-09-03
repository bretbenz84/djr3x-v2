# reSpeaker Flex XVF3800 Circular-4 — mic array setup, verification, tuning

Installed 2026-09-02, replacing the ReSpeaker Lite. Same job: it is the robot's
only microphone AND the playback path (USB → core board → amp), so the XMOS
XVF3800's echo canceller sees exactly what the speaker plays and removes it
from the mic. What changed: four capsules with beamforming, DoA, on-chip AGC and
noise suppression, and six USB input channels instead of two.

Owner rules (2026-09-02): **no sound, mic recording, servo motion, firmware
flash, or DSP parameter write without Bret's explicit go.** Read-only queries
(`tools/flex_ctl.py` with no `--write`, `sd.query_devices`) are fine.

## Hardware as found

| item | value |
|---|---|
| CoreAudio name | `reSpeaker Flex XVF3800 C16K6Ch` (6 in / 2 out, default 16 kHz; accepts 16/24/44.1/48 k opens) |
| USB | VID `0x2886` PID `0x001e`, "Seeed Studio" |
| firmware | `VERSION 1.0.0`, `BLD_MSG ua-io16-6ch-sqr` — the stock **6-channel 16 kHz USB** build; USB bit depth 16/16 |
| array | `AEC_MIC_ARRAY_TYPE 2` (square/circular), 4 mics, 1 far-end |
| speaker | amp on the core board's own output (JST / 3.5 mm) |

### USB input channel map (verified by capture + parameter read)

| ch | source (`AUDIO_MGR_OP_*`) | what it is | room floor vs raw |
|---|---|---|---|
| 0 | `OP_L = (8,0)` user-chosen → auto-select beam, post-processed | **Conference** output: AEC + beamform + NS + residual-echo suppression + **AGC** (max gain 32) + limiter | **~+26 dB** — and it ducks ~20 dB while Rex plays |
| 1 | `OP_R = (7,3)` | **ASR** output of the auto-selected beam: AEC + beamform, `AEC_ASROUTONOFF=1`, gain 1.0, **no AGC** | ~+5 dB |
| 2–5 | fixed by the 6ch build (`OP_CH3..6` answer status 65) | raw capsules mic0..mic3, no processing, **no AEC** | 0 (reference) |

Correlation on a passive room capture: ch0↔ch1 0.96, ch2..5 mutually 0.95–0.98,
the two groups ≈0 — the processed pair and the raw quartet are independent
signals. Averaging all six (what the old blank `AUDIO_AEC_INPUT_CHANNEL` did,
and what the supervisor did before this change) would be dominated by the AGC
channel's hot floor.

**Pipeline channel: 1.** ASR-tuned, echo-cancelled, beamformed, no AGC pumping
the noise floor into the scene analyzer's "sudden loud sound" detector.

### DSP parameters as shipped (read 2026-09-02, `tools/flex_ctl.py`)

```
AEC_HPFONOFF 2 (125 Hz)   AEC_AECEMPHASISONOFF 2   AEC_FAR_EXTGAIN 0.0
AEC_FIXEDBEAMSONOFF 0     AEC_ASROUTONOFF 1        AEC_ASROUTGAIN 1.0
AUDIO_MGR_MIC_GAIN 10.0   AUDIO_MGR_REF_GAIN 1.9   AUDIO_MGR_SYS_DELAY 12
AUDIO_MGR_SELECTED_CHANNELS (3,3)   AUDIO_MGR_FAR_END_DSP_ENABLE 0
PP_AGCONOFF 1  PP_AGCMAXGAIN 32  PP_AGCGAIN 32 (pinned at max in a quiet room)
PP_AGCDESIREDLEVEL 0.0045  PP_ECHOONOFF 1  PP_NLAEC_MODE 0  PP_DTSENSITIVE 12
PP_MIN_NS 0.15  PP_MIN_NN 0.51  PP_LIMITONOFF 1  PP_ATTNS_MODE 0
```

Nothing has been written to the board. All of these are volatile-writable via
`tools/flex_ctl.py --write NAME VALUE` (asks for confirmation) and persist only
after `SAVE_CONFIGURATION`.

## .env (robot)

```
AUDIO_DEVICE_NAME=ReSpeaker          # substring-matches the Flex's name
AUDIO_OUTPUT_DEVICE_NAME=ReSpeaker   # playback THROUGH the Flex = AEC reference
AUDIO_AEC_INPUT_CHANNEL=1            # ASR beam (was blank on the Lite)
AUDIO_INPUT_GAIN=4.0                 # measured 2026-09-02; matches the Lite's absolute levels
WAKE_WORD_ALLOW_DURING_TTS=false     # unchanged; a separate talk-over decision
```

`audio/hardware_aec.is_active()` matches the hint `respeaker` on both devices,
so every AEC-gated behavior (short deaf window, VAD 0.4, 1.0 s pre-roll, eager
motion endpoint, during-DJ command listener) stays on exactly as with the Lite.
`config.AUDIO_INPUT_CHANNELS=2` means the main app opens channels 0–1 only and
reads column 1; the supervisor opens all six and reads column 1
(`rex_supervisor._frames_to_mono`).

## What was measured on 2026-09-02 (and what it does not prove)

Conditions: TV on in the room, Rex not running, supervisor idle-listening.
Playback through the Flex output at 16 kHz.

**1 kHz tone, 4 s, 0.25 FS** — narrowband level at 1 kHz:

| | raw capsules (avg) | ch0 Conference | ch1 ASR |
|---|---|---|---|
| tone level | −27.8 dB | −78.2 dB | −66.2 dB |
| cancellation | — | **50 dB** | **38 dB** |

The broadband residual left on ch1 was 84 % in 150–300 Hz: room rumble, not the
tone. The tone itself is gone. Lite comparison (2026-06-08): −68.6 dBFS in-mic,
≈41 dB — the Flex is at least as good on a tone.

**Rex speech clip, 3.4 s, peak −6 dBFS** — broadband ERLE per 0.5 s window on
ch1: −0.8, −2.6, 0.0, 0.8, **9.5**, 3.0 dB. The adaptive filter had not converged
(a fresh far-end path, only ~3 s of non-stationary signal); ch0 showed a steady
13–15 dB from its residual-echo suppressor + AGC ducking, which is not a real
ERLE. **This number is not the Flex's speech ERLE — it is an unconverged first
look.** The Lite's remembered 17 dB broadband figure was a 20-plus-second,
listened-to session, not a 3 s clip.

Raw-capsule peak during the clip: −18.5 dBFS at −6 dBFS playback — headroom is
fine, the amp is not driving the capsules into clipping.

## Re-test results, 2026-09-02 evening (TV off, Rex stopped, gain 1.5x)

`noise`: floor **−58.7 dBFS** post-gain (Lite: −49.2 to −54.0).

`channels` at 3 ft: layout confirmed — ch0/ch1 corr 0.91, ch2-5 mutually
0.89–0.96, groups independent; ch1 −50.2 dBFS RMS while talking, ch0 −26.8.

`speech` (pipeline math, ch1 × 1.5):

| distance | speech RMS | between-word floor | SNR | peak |
|---|---|---|---|---|
| 3 ft | −44.7 dBFS | −60.6 | 15.9 dB | −23.8 |
| 6 ft (usual spot) | −45.7 dBFS | −69.0 | **23.3 dB** | −26.6 |
| 9 ft | −47.4 dBFS | −60.9 | **13.5 dB** | −27.9 |

Lite at 5–7 ft was 12.7–15.1 dB. So: +8–10 dB SNR at the usual spot, and 9 ft
on the Flex ≈ 6 ft on the Lite. Speech level falls only ~3 dB from 3 to 9 ft
(the beam output holds level); the whole curve is ~10 dB quieter than the
Lite's, so `AUDIO_INPUT_GAIN` went 1.5 → **4.0**, which reproduces the Lite's
absolute speech/floor levels (~−34 / ~−49) that the VAD, wake word and startle
detector were tuned on.

`aec` (20 s of Rex's voice at −12 dBFS peak): converged at ~10 s, chip flag
`AEC_AECCONVERGED=1`. Last 5 s: raw echo −44.7 dBFS; ch1 residual −59.9 (floor
−62.2) = **15.2 dB, floor-limited** — the residual sits at room noise, so the
true cancellation is higher and would need louder playback to measure; ch0
residual −74.1 = 29.4 dB. Logged in `logs/mic_check/aec_history.jsonl`.

Still owed: `score` from 6 ft (Lite: 0.964–0.971), and a live session with a
talk-over attempt and a "come here" from 9 ft.

## Re-test plan (needs Bret at the keyboard, TV off, Rex stopped)

Run in this order; every command states what it will do and `aec`/`listen` wait
for Enter before making sound.

1. `venv/bin/python tools/flex_ctl.py` — confirm the parameter dump above
   (nothing drifted; `AEC_AECPATHCHANGE`, `PP_AGCGAIN` are the live ones).
2. `venv/bin/python tools/mic_check.py channels` — talk from the usual spot;
   expect "known layout … set AUDIO_AEC_INPUT_CHANNEL=1" and ch1 speech level.
3. `venv/bin/python tools/mic_check.py noise` then `speech` — floor and SNR
   through the live pipeline math (channel 1 × `AUDIO_INPUT_GAIN`). Decide the
   gain here: target speech ≈ −30 dBFS RMS, floor as low as it goes.
4. `venv/bin/python tools/mic_check.py aec` — 20 s of Rex's cached voice at
   −12 dBFS peak through the Flex; ERLE per second on ch0/ch1 vs the raw
   capsules, the last-5 s average, and the chip's own `AEC_AECCONVERGED` flag.
   Logged to `logs/mic_check/aec_history.jsonl`.
5. `venv/bin/python tools/mic_check.py score` — the scripted 8-sentence
   accuracy benchmark, logged to `logs/mic_check/history.jsonl`.
6. `venv/bin/python tools/mic_check.py distance` — 3 / 6 / 9 ft sweep. This is
   the "does it hear people farther away" question directly.
7. Then a normal Rex session with a talk-over attempt ("hey Rex" while he is
   mid-sentence) and a "come here" from across the room, watching
   `[hardware_aec] active=True` and `[identity_decision]` scores — voiceprints
   were enrolled through the Lite's processing and may score lower at first.

### ReSpeaker Lite baselines to beat (same tool, same room, same spot)

| date | test | result |
|---|---|---|
| 2026-06-08 | 1 kHz loopback | −68.6 dBFS in-mic (≈41 dB cancelled) |
| 2026-06-18 | broadband speech, by ear | ≈17 dB |
| 2026-07-24 | `speech` at 5–7 ft | speech −35…−37.6 dBFS, floor −48…−50, **SNR 12.7–15.1 dB** |
| 2026-07-31 20:47 | `score` (gain 1.5) | mean accuracy 0.80 (one 0 % take: started before the prompt), floor −54.0 |
| 2026-07-31 21:12 | `score` | **0.971**, floor −49.2, speech −31…−38 dBFS |
| 2026-07-31 21:26 | `score` (8 sentences) | **0.964**, floor −52.6, speech −32…−39 dBFS |

Whisper's usable floor is ~12 dB SNR; the Lite sat right at it from 6 ft. The
Flex's beamformer is the first lever that can actually raise SNR (rather than
level) at distance — the `speech`/`distance` numbers are the ones to compare.

## Tuning levers (all `flex_ctl.py --write`, all reversible, none applied yet)

- `AUDIO_MGR_MIC_GAIN` (10.0): pre-AEC capsule gain. Raise if ch1 speech is
  quiet at 9 ft; lower if the raw peak in `aec` approaches −3 dBFS.
- `AEC_FIXEDBEAMSONOFF` + `AEC_FIXEDBEAMSAZIMUTH_VALUES`: lock beams to where
  people sit instead of free-running. Worth trying if the auto beam wanders
  onto the TV.
- `AEC_HPFONOFF` (2 = 125 Hz): 3 or 4 trims more rumble; the residual on ch1
  was mostly 150–300 Hz, so this is cheap SNR if speech survives it.
- `PP_*` only shape ch0 (Conference). Irrelevant while the pipeline reads ch1 —
  unless `AUDIO_MGR_OP_R` is re-pointed, which is the one routing change that
  would matter and needs a deliberate decision.
- `AUDIO_MGR_SYS_DELAY` (12 samples): reference alignment. Only if `aec`
  shows the filter never converging.
- Firmware: the 2-channel build (`respeaker_flex_usb_c16k2ch`) drops the raw
  channels and nothing else; there is no reason to flash it. The 48 kHz builds
  would change `AUDIO_SAMPLE_RATE` assumptions everywhere — do not.

## Tools

- `tools/flex_ctl.py` — read/dump parameters (default read-only; `--write`
  confirms interactively). Own implementation of Seeed's control protocol over
  pyusb + the wheel-bundled libusb (no Homebrew libusb).
- `tools/mic_check.py aec` — the echo-cancellation measurement, per channel,
  with history.
- `tools/mic_check.py channels` — now N-channel aware; prints the correlation
  matrix and the known Flex layout.
- Vendor references: Seeed `respeaker/reSpeaker_Flex` (firmware bins,
  `python_control/xvf_host.py`), `respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY`
  (host_control binaries incl. `mac_arm64`, DFU guide), XMOS XVF3800 user guide
  v3.2.1 for the parameter semantics.
