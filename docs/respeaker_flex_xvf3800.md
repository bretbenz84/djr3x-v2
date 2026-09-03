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
AUDIO_INPUT_GAIN=1.5                 # owner: no host gain — trial the board's AGC instead
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
Lite's. Host gain 4.0 would reproduce the Lite's absolute levels, but the owner
call is to use the board's own AGC instead — see "AGC trial" below. Host gain
stays 1.5 meanwhile.

`aec` (20 s of Rex's voice at −12 dBFS peak): converged at ~10 s, chip flag
`AEC_AECCONVERGED=1`. Last 5 s: raw echo −44.7 dBFS; ch1 residual −59.9 (floor
−62.2) = **15.2 dB, floor-limited** — the residual sits at room noise, so the
true cancellation is higher and would need louder playback to measure; ch0
residual −74.1 = 29.4 dB. Logged in `logs/mic_check/aec_history.jsonl`.

`score` (19:45, ch1, gain 1.5, floor −59.2): **8/8 sentences word-perfect,
mean accuracy 1.00** at segment levels of −47 to −52 dBFS — including the
1–2 word lines ("Halt.", "Come here.") that were the Lite's weak spot. Lite
best was 0.971 / 0.964 at −31 to −39 dBFS. Transcription does not need the
level lifted; the only open level question is whether VAD / wake word trigger
reliably at −45 dBFS speech, which the live session answers.

First live session (20:45–20:58, `logs/djr3x-2026-09-02-20-45-24.log`): boot
lines confirm ch1 / 1.5x / `[hardware_aec] active=True`. 23 heard turns, every
one transcribed cleanly, no watchdog stalls. Bret's voice score p50 0.696 (p25
0.60), runner-up p50 0.58 — no drop from the Lite-era numbers. Two off-camera
short commands after Rex turned away went to 'Jeremy Thomas' (0.58 vs Bret
0.44) and 'unknown_voice_1' (0.56 vs 0.53) — the pre-existing off-camera
misattribution pattern, not the mic; nothing was enrolled. **Watch item:** 25
context-echo hallucination rejections in 13 min (a Lite session the night
before had 4 in 8 min), all 3–14 s AFTER playback ended, so not AEC residual
— VAD firing on room sounds (dogs, motor/sfx during the drive) and the decoder
echoing Rex's last line via context bias. The 4-layer guard dropped every one;
the cost is a ~2 s decode each. Talk-over was untestable: the config leaves
`WAKE_WORD_ALLOW_DURING_TTS=false`, so the wake word cannot fire during speech
regardless of AEC; the three "VAD barge-in suppressed" lines show Bret's
talk-over was heard and captured after Rex finished. "Come here" turned to the
radar body, found no face, then the swing check refused every escape turn —
motion, not audio.

Still owed: a talk-over session with `WAKE_WORD_ALLOW_DURING_TTS=true`.

### AGC trial (pending owner go for two volatile writes)

The XVF3800's AGC (`PP_AGC*`) lives in the post-processing block, which only
feeds the **Conference** output (USB ch0). The ASR output (ch1) has a fixed
`AEC_ASROUTGAIN` and no AGC — our captures confirm it: with `PP_AGCONOFF=1`
and the gain pinned at its 32× max, ch0 sat at −26 dBFS in silence AND while
talking (the AGC hits its target both ways), while ch1 tracked the raw
capsules +5 dB. So "turn AGC on" means "read ch0", and as shipped ch0 pumps
the room floor up to the speech target whenever nobody talks
(`PP_ATTNS_MODE=0` = no extra attenuation during non-speech).

Proposed trial, both writes volatile (a power cycle restores defaults; nothing
persists without `SAVE_CONFIGURATION`):

    tools/flex_ctl.py --write PP_ATTNS_MODE 1      # pull gain down when no speech
    tools/flex_ctl.py --write PP_AGCMAXGAIN 8      # cap at +18 dB instead of +30

then measure ch0 without touching .env:

    tools/mic_check.py noise  --channel 0 --gain 1.0
    tools/mic_check.py speech --channel 0 --gain 1.0     # at 3 / 6 / 9 ft

and compare speech level, floor and SNR against the ch1 table above. Switching
the pipeline to ch0 also means the residual-echo suppressor and NS gain floors
(`PP_ECHOONOFF`, `PP_MIN_NS/NN`) are in the ASR path — the artifacts those add
are why vendors ship a separate ASR output, so `score` has to be run on both
channels before the decision.

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

## Direction of arrival → voice bearing (shipped 2026-09-02, live test owed)

The XVF3800 tracks where speech comes from and publishes `DOA_VALUE` (0–359°
plus a speech flag) on the USB control endpoint; `AEC_AZIMUTH_VALUES[3]` is
the auto-selected beam's azimuth and agrees with it. Reads cost ~10 ms.

**Convention, measured with `tools/flex_doa.py`** (ring mounted with its
printed 0° edge forward, Bret ~4 ft away, 20 s each):

| Bret stood | DoA median | spread | notes |
|---|---|---|---|
| front | 359° | 16° | |
| left | 90° | 4° | |
| right | 291° (mode 264–277°) | 82° | a third of samples snapped to ~86° between words |
| back | 171° (beam 172°) | 28° | competing cluster at ~55° |

So chip 0 = ahead, 90 = Rex's **left**, 180 = behind, 270 = his **right**:
base bearing (+ = left/CCW) = `wrap180(DoA)`, no sign flip
(`FLEX_DOA_SIGN=1`, `FLEX_DOA_FORWARD_OFFSET_DEG=0`). Note this is the
MIRROR of what Seeed's drawing suggests for a mic-side-up ring; the
measurement wins and the knobs exist for a re-mount. **Between a talker's
words the register falls back to another source in the room** (86° in two
runs, 55° in another) — the fusion therefore takes the largest cluster of
mutually-agreeing samples over the segment, not a median.

**Mount (owner, 2026-09-02): the ring sits on top of the torso**, on the
section that carries the hero arm, so it is body-frame — `FLEX_DOA_MOUNT=base`
(the default) is correct and the neck does not enter the conversion. The hero
arm servo swings that section a little; at the servo's midpoint the ring's 0°
is straight ahead, and the arm's travel is too small to throw off the opening
bearing of a "come here" (owner's judgement). If a hero-arm pose ever shows up
as a consistent DoA offset, `FLEX_DOA_FORWARD_OFFSET_DEG` is the knob, or a
per-sample correction from the heroarm servo position would be the fix.
`FLEX_DOA_MOUNT=head` stays available for a ring that turns with the neck.

**Pipeline** (`hardware/flex_doa.py` → `interaction._note_voice_bearing` →
`motion_agency.request_come_here(voice_bearing_deg=…)` /
`consciousness.note_speaker_gaze_intent(bearing_deg=…)`):

1. A daemon polls DoA at `FLEX_DOA_POLL_HZ` (8) and keeps 20 s of samples.
2. Every captured speech segment gets `[voice_doa] bearing ±N° (chip …, k/n
   samples agree, spread …)` — the dominant cluster over the segment window.
3. `come here`: evidence order is camera sighting → explicit words ("I'm
   behind you", "to your left") → radar body agreeing with the voice within
   `MOTION_COME_VOICE_RADAR_MATCH_DEG` (30°) → the voice bearing alone as the
   opening turn (off-axis by more than `MOTION_COME_VOICE_TURN_MIN_DEG`, 15°;
   a fruitless dwell there marks the spot visited like a radar body) → radar
   alone → blind sweep. Log lines: `voice came from ±N° — leading with a turn
   toward it`, `… the spoken direction takes precedence`, `… agrees with the
   voice`, `(no body near the voice at ±N°)`.
4. Off-camera speaker gaze: the search's first waypoint points the neck at
   the voice (`_voice_bearing_waypoint`), then the usual scan continues; the
   hint is spent after one pass.

Kill switches: `FLEX_DOA_ENABLED`, `MOTION_COME_VOICE_BEARING_ENABLED`.
Tests: `tests/test_flex_doa.py` (convention, clustering, precedence, the sign
trap, gaze plan). **Live test owed:** a session with "come here" from behind
and from off-camera left/right, watching the `[voice_doa]` and
`[motion_agency] requested come:` lines — and the audio stream's health
(`sounddevice status` / `stream_watchdog`), since the poller shares the USB
device with the mic stream.

## Voice bearing ↔ face attribution (shipped 2026-09-02, bench test owed)

Owner spec: "a face to the right of the robot would be connected to the voice
on the right". `perception/voice_bearing_match.py` puts a visible face into the
base frame — face bearing = −(neck yaw + face offset × `MOTION_COME_CAM_HALF_FOV_DEG`),
the ONE negation between the camera's +right and the DoA's +left — and ranks
faces by angular distance from the voice. Three outputs feed
`interaction._handle_speech_segment` (`_voice_bearing_face_match`, one
`[voice_doa] voice … vs faces …` line per turn):

- **selected** (nearest face within `VOICE_BEARING_FACE_TOLERANCE_DEG` 20°,
  next face ≥ `VOICE_BEARING_FACE_MARGIN_DEG` 10° farther): with two or more
  known faces on camera this becomes the "visible face" for the voice-primary
  decision (`single_visible` / `other_known_recently` relaxed for it) — before
  the array that case fell to "no single visible face". In the multi-visible
  weak-voice block it corroborates like the lip detector does
  (`voice_corroborated_by_bearing`).
- **confirm_pid** == the visible face → folded into the visual-speaker witness
  (`visual_speaker_pid`), so a marginal voice on that face is `voice_agrees`.
- **contradicts** (nearest face > `VOICE_BEARING_CONTRADICTION_DEG` 45° from
  the voice) → folded into `visual_mouth_still`: the calibrated veto that keeps
  a silently-on-camera face from absorbing an off-camera voice
  (`voice_weak_face_wins` → `off_screen_unknown`, sub-genuine-band marginal
  → `challenge_identity`). This is the misattribution shape from the 20:45
  session (off-camera "Turn around" credited to Jeremy Thomas).

Requires a fresh bearing (`FLEX_DOA_MAX_AGE_SECS`) with cluster share ≥
`VOICE_BEARING_MIN_SHARE` (0.5). Kill switch `VOICE_BEARING_ATTRIBUTION_ENABLED`.
Tests: `tests/test_voice_bearing_match.py`; the decision-function suites are
unchanged and green.

**Bench tool:** `tools/voice_face_test.py` — records a take through the robot
mic path, polls DoA for the same window, grabs a frame mid-take, then prints
the voice bearing, every face's identity + bearing, the voice-ID scoreboard and
the matcher's verdict; saves an annotated frame + `logs/mic_check/voice_face.jsonl`.
Run with Rex stopped, head centred (`--neck-deg` otherwise). Expected for
Bret ~20° right of the nose: voice ≈ −20°, face box ≈ 80 % across to the
right ⇒ face ≈ −20°, verdict "consistent".

### Bench results 2026-09-02 21:41–21:44 and the lens calibration they forced

Three takes, Bret at roughly −20/−30/+30° by his own estimate ("off by a 10°
margin"), head reported centred, `tools/voice_face_test.py`:

| take | voice bearing (cluster) | face px off centre | face @25° half-FOV | voice ID |
|---|---|---|---|---|
| right | −33.1° (48/55, spread 1.9°) | +254 | −6.6° | Bret 0.753 |
| right, farther | −39.7° (39/46, spread 3.3°) | +378 | −9.9° | Bret 0.810 |
| left | +33.0° (51/51) | −698 | +18.2° | Bret 0.794 (face unidentified — looking down) |

The voice clusters were tight and the voice ID right every time; the FACE
bearings were 2–5× too small. The camera is a very wide fisheye (the visor
edges fill the bottom corners of every frame), so the 25° half-FOV inherited
from the come-here calibration is wrong for angular work. Fitting the lens
against the voice (`--fit`, `perception.voice_bearing_match.fit_camera_model`):
**14.6 px/deg (half-frame ≈ 66°), yaw offset +14.8°, rms 0.8°** — i.e. a
face's angle off the camera axis is nearly linear in pixels (equidistant
fisheye), and there is a constant ~15° between the camera axis and the mic's
0°. Adopted: `VOICE_BEARING_CAM_PX_PER_DEG = 14.6`. The +14.8° is left at
`VOICE_BEARING_CAM_YAW_OFFSET_DEG = 0` until the owner confirms whether the
head was centred during the takes — if it was, that constant is a camera/mic
mount offset and belongs in the knob; if the head was turned toward him, the
live app already adds the neck readback and the knob must stay 0.

**Follow-up flag:** come-here alignment still uses `MOTION_COME_CAM_HALF_FOV_DEG
= 25` (≈38 px/deg) for the fused neck+face bearing; the voice-referenced fit
says the lens is ~14.6 px/deg. If that is right, alignment under-turns toward
an off-centre face by ~2.5× — worth a field check before touching it, since
that value was set from a real turn (2026-08-11) and the loop is tuned around it.

More takes sharpen the fit: one at the frame CENTRE (pins the offset alone),
one near each EDGE. Placement accuracy does not matter — the voice is the
reference, the face box is the measurement.

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
