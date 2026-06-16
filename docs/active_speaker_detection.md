# Active-Speaker Detection (Visual) — Feature Spec

**Status:** Draft / ready to build
**Target module:** `vision/active_speaker.py` (new), with hooks in `vision/face_expression.py`
**Owner:** Bret
**Depends on:** existing MediaPipe Face Landmarker pipeline, `world_state`, dlib face identity

---

## 1. Problem statement

When two or more people are in frame, Rex needs to know **which visible person is currently speaking**, so conversation, gaze/face-tracking, and memory attribution target the right individual. Audio direction-of-arrival is unavailable: the ReSpeaker Lite (XMOS XU316) delivers a single AEC-processed stream, so there is no usable inter-channel cue for sound localization.

This feature solves active-speaker detection **visually**, by reading lip motion per face and gating it on head orientation, then exposing a per-person `is_speaking` signal on `world_state.people`.

### Non-goals

- No audio localization (hardware can't support it; do not add mic-array code).
- No lip *reading* / phoneme recognition. We only detect *that* a mouth is articulating, not *what* it says.
- No new identity system. Identity stays with dlib; this feature only attaches a speaking signal to slots dlib/face pipeline already own.
- No change to VAD or the speech-recognition path. This consumes the existing "is speech happening right now" signal; it does not produce transcription.

---

## 2. Background: what already exists (build on this, don't duplicate)

The current `vision/face_expression.py` already does the hard association work, and this feature must reuse it rather than re-detect faces:

- Runs MediaPipe **Face Landmarker** with `num_faces=2` on the shared camera frame every ~0.25s (`FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS`).
- Computes **blendshapes per face**, including `jawOpen` (already in `_BLENDSHAPE_KEYS`), plus the mouth shapes.
- Derives a `face_box` per face from landmarks (`_face_box_from_landmarks`).
- **Associates each face to the correct `world_state.people` slot** via IoU between the expression `face_box` and the dlib-owned `face_box` (`_match_expression_to_people`), writing under the `world_state.mutate("people", ...)` lock.

That IoU-to-identity association is the critical reuse point: it means a per-face mouth signal can already be tied to a specific person slot. **Do not introduce a second, parallel face-association scheme.**

Two facts about the wider system that constrain the design:

- `vision/pose.py` (MediaPipe legacy `solutions.pose`) is **single-person only** — it cannot tell person A from person B. **Do not use body pose for any part of this feature.** All signals here come from the multi-face Face Landmarker path.
- The blendshapes are already computed every cycle, so reading `jawOpen` for this feature is **zero additional model cost** — we are adding buffering and math, not inference.

---

## 3. Design overview

Three cooperating layers, in dependency order. Layers 1–2 are the feature; Layer 3 is a thin consumer hook.

| Layer | Signal | Cost | Role |
|---|---|---|---|
| 1 | **Head yaw** per face (from existing landmarks) | ~free | Gate: discount faces turned away from Rex |
| 2 | **Lip-motion energy** per face (rolling `jawOpen`/mouth variance) | cheap | Primary active-speaker signal |
| 3 | **VAD gate + arbitration** | trivial | Only attribute a speaker while speech is actually happening; pick one winner with hysteresis |

Core idea: **lip-motion energy says who is articulating; head yaw says who is plausibly addressing Rex; VAD says when any of it counts; hysteresis keeps the choice stable.**

---

## 4. Layer 1 — Head pose (orientation gate)

### Purpose
A mouth can move while a person talks to someone *other* than Rex, or while chewing. Head orientation toward the camera is the cheapest disambiguator. We need only **yaw** (left/right turn); pitch/roll are not required for v1.

### Method
Use the per-face landmarks already produced by Face Landmarker. Two acceptable implementations — pick the simpler that passes the calibration test:

- **A (preferred, no new deps):** estimate yaw from horizontal landmark asymmetry — the signed ratio of (nose-tip-to-left-eye-corner) vs (nose-tip-to-right-eye-corner) horizontal distances, normalized by inter-ocular distance. Symmetric ⇒ facing forward; skewed ⇒ turned. Output a normalized `yaw` in roughly [-1, +1] and/or a boolean `facing_camera`.
- **B (fallback if A is too noisy):** solve PnP against a canonical 3D face model using a fixed subset of stable landmarks (nose tip, eye corners, mouth corners, chin). Use a small numpy implementation; **do not import cv2** (project deliberately avoids it — see `image_utils.py` rationale). Only adopt B if A fails calibration.

### Output
Per face: `facing_camera: bool` (derived from `abs(yaw) <= FACING_YAW_MAX`) and raw `yaw: float` for tuning/logging.

### Acceptance
- A person looking at the camera reads `facing_camera = True` across a normal range of small head movement.
- A person turned ~45°+ toward a side conversation reads `facing_camera = False`.

---

## 5. Layer 2 — Lip-motion energy (primary signal)

### Purpose
Distinguish a moving (talking) mouth from a still (listening) one, per face, every cycle.

### Method
Maintain a **per-person rolling buffer** of mouth-openness samples and compute a motion-energy score from it.

1. **Per-cycle sample.** From the per-face blendshapes already computed, take `jawOpen` as the primary scalar. Optionally combine with mouth open/close shapes if calibration shows it sharpens the signal; keep it to scalars already present in the blendshape output. Call this sample `m_t ∈ [0,1]`.
2. **Rolling buffer.** Keep the last `LIPSYNC_WINDOW_SECS` (default **1.0s**) of `(timestamp, m_t)` per person. At ~0.25s cadence that's ~4 samples; the design must tolerate variable cadence (don't assume fixed N — store timestamps and trim by age).
3. **Motion energy.** Compute **variance** of `m_t` over the window (primary). Optionally also compute a zero-/mean-crossing rate of `m_t` as a secondary robustness signal — talking oscillates across its mean; a held-open or held-closed mouth does not. Combine into a single `lip_energy` score.
4. **Threshold.** `lip_active = lip_energy >= LIPSYNC_ENERGY_THRESHOLD`.

### Buffer keying — important
Buffers are **per person identity**, keyed off the same slot the expression pipeline writes to (resolve via `person_db_id` when present, else the stable in-frame slot index used by `_match_expression_to_people`). When a face leaves frame, age its buffer out (drop after `LIPSYNC_STALE_SECS`) so a returning person doesn't inherit stale motion. Do **not** key buffers off raw landmark order — it is not stable across frames.

### Output
Per face: `lip_energy: float` and `lip_active: bool`.

### Acceptance
- A talking person sustains `lip_active = True`; a silent listener stays `False`.
- Brief closed-mouth gaps mid-sentence do not immediately flip `lip_active` to False (the window smooths this; hysteresis in Layer 3 finishes the job).
- Chewing/yawning may trip `lip_active` — this is expected and handled by the Layer 1 gate plus Layer 3 VAD gate, not here.

---

## 6. Layer 3 — Arbitration (VAD gate + winner selection + hysteresis)

### Purpose
Turn per-face booleans into a single, stable answer: *who is the current speaker.*

### Inputs
- Per face: `lip_energy`, `lip_active`, `facing_camera` (Layers 1–2).
- System: **current VAD state** — "is speech happening right now" — from the existing audio pipeline. (Spec assumes a readable VAD/'speech active' flag exists; if it must be surfaced, note it as a small dependency, do not rebuild VAD.)

### Algorithm
1. **VAD gate.** If VAD reports no speech, **no one** is the active speaker. Decay/clear the current speaker after `SPEAKER_RELEASE_SECS` of no speech. This single gate removes most chewing/yawning false positives for free.
2. **Candidate set.** Among visible faces, keep those with `facing_camera = True`. If that empties the set (everyone slightly turned), fall back to all visible faces so Rex still attributes *someone* during active speech.
3. **Winner.** Pick the candidate with the highest `lip_energy`. Require it to clear `lip_active` AND beat the runner-up by `SPEAKER_MARGIN` (mirrors the margin-guard pattern already used in `speaker_id.identify_speaker` — reuse that mental model for consistency).
4. **Hysteresis.** Once a person is the active speaker, keep them until a *different* candidate out-scores them by `SPEAKER_SWITCH_MARGIN` for at least `SPEAKER_SWITCH_SECS`. This prevents flicker between two animated people and rides over mid-sentence mouth closes.
5. **Single-face shortcut.** Exactly one visible face during active speech ⇒ that person is the speaker (skip margin checks). Mirrors the `len(available) == 1` shortcut already in `_match_expression_to_people`.

### Output
Write to each `world_state.people` slot:
- `is_speaking: bool`
- `speaking_confidence: float` (normalized `lip_energy`, or the margin over runner-up)
- `speaking_updated_at: float`

Exactly **zero or one** slot should have `is_speaking = True` at a time.

---

## 7. Module / integration plan

- **New:** `vision/active_speaker.py`
  - Owns the per-person rolling buffers, Layer 2 energy math, Layer 1 yaw (or imports a small head-pose helper), and Layer 3 arbitration.
  - Exposes `update(face_signals, vad_active) -> None` that writes the `is_speaking` fields via `world_state.mutate("people", ...)`, following the read-modify-write-under-lock pattern in `face_expression.merge_expressions_into_world_state` so concurrent identity/pose writes aren't clobbered.
  - Lazy-load nothing heavy; it has no model of its own. Keep module-level state (buffers) guarded by a lock like the other vision modules.
- **Hook:** in `vision/face_expression.py`, where per-face blendshapes + `face_box` + matched person index already exist, pass that per-face data (jawOpen scalar, landmarks or precomputed yaw, matched slot key) into `active_speaker.update(...)`. **Reuse the existing IoU match — do not re-associate faces.**
- **VAD:** read the existing speech-active flag. If it isn't currently exposed to the vision side, surface it via `world_state` or a small accessor; document this as the one cross-cutting dependency.
- **Config:** add keys to `config.py` (Section 9). Follow the existing `getattr(config, KEY, default)` access style so missing keys degrade gracefully.
- **Lifecycle:** if `active_speaker` keeps a thread, mirror `face_expression`'s `start()/stop()` + `_stop_event` pattern. Preference: run it **inline** inside the existing face-expression cycle (no new thread) since it piggybacks on that cadence and data.

### Feature flag
Gate the whole feature behind `ACTIVE_SPEAKER_ENABLED` (default True). When False, never write `is_speaking` and leave consumers to their current behavior.

---

## 8. World-state contract

Per entry in `world_state.people` (additive; do not remove existing fields):

```
is_speaking:          bool   # exactly one True at most, system-wide
speaking_confidence:  float  # 0..1
speaking_updated_at:  float  # epoch seconds
```

Consumers (conversation pipeline, face-tracking) read `is_speaking` the same way they read identity today. A helper mirroring `face.visible_known_names()` — e.g. `active_speaker.current_speaker(snapshot=None) -> Optional[person]` returning the speaking slot (with `person_db_id`/name resolved) — is recommended so callers don't re-scan the list.

---

## 9. Config keys (new, with proposed defaults)

```python
ACTIVE_SPEAKER_ENABLED            = True

# Layer 1 — head pose gate
FACING_YAW_MAX                    = 0.45   # |yaw| above this ⇒ not facing camera (tune)

# Layer 2 — lip motion
LIPSYNC_WINDOW_SECS               = 1.0
LIPSYNC_ENERGY_THRESHOLD          = 0.0025 # variance of jawOpen; calibrate on-device
LIPSYNC_STALE_SECS                = 2.0    # drop a person's buffer after this long unseen

# Layer 3 — arbitration
SPEAKER_MARGIN                    = 0.0015 # winner must beat runner-up by this in lip_energy
SPEAKER_SWITCH_MARGIN             = 0.0030 # higher bar to STEAL active-speaker from current
SPEAKER_SWITCH_SECS               = 0.4    # ...sustained for this long before switching
SPEAKER_RELEASE_SECS              = 0.6    # clear speaker this long after speech stops
```

All thresholds are **starting estimates** and must be tuned on-device (see calibration). Energy thresholds especially depend on camera resolution, distance, and blendshape scaling — treat the numbers above as placeholders, not truth.

---

## 10. Calibration & test plan

Provide a small dev script (mirror the spirit of `tooling/test_voice_id.py`) that, given the live camera + VAD:

1. **Single talker, facing camera.** Log per-face `lip_energy` while one person reads aloud, then stays silent. Pick `LIPSYNC_ENERGY_THRESHOLD` to cleanly separate the two regimes.
2. **Two people, alternating.** A and B take turns speaking. Verify `is_speaking` follows the talker and that exactly one (or zero) slot is True. Tune `SPEAKER_MARGIN` / switch params to kill flicker.
3. **Distractors.** Chewing, yawning, laughing without speech — confirm the VAD gate keeps `is_speaking` False when no speech is present.
4. **Turn-away.** Talker faces a side conversation — confirm `facing_camera` gate suppresses attribution to them, and that the fallback still names someone when *everyone* is slightly turned during real speech.
5. **Leave/return.** A person exits and re-enters frame — confirm buffers age out and don't carry stale motion.

Log format suggestion (reuse the `Name#id=score` style from `speaker_id._log_scoreboard` for familiarity):
`[active_speaker] vad=on facing={A,B} energy: A#3=0.0041 B#7=0.0006 → speaking=A#3`

---

## 11. Known limitations (state these honestly in code comments)

- Two people talking **simultaneously** ⇒ only the higher-energy mouth wins; no multi-speaker output in v1.
- Heavy occlusion of the mouth (hand, mug, mask) ⇒ no signal; person can't be selected.
- Profile/extreme angles degrade both yaw estimate and jaw visibility; the facing gate intentionally drops these.
- This is articulation detection, not voice attribution — it cannot tell that an *off-screen* person is the one talking. If VAD fires and no visible mouth is moving, `is_speaking` stays empty (correct behavior).
- Cadence-bound: resolution is the face-expression interval (~0.25s). Fast back-and-forth shorter than the switch window won't be tracked turn-by-turn.

---

## 12. Build order (suggested commits)

1. `vision/active_speaker.py` skeleton + config keys + world-state fields, feature-flagged, no-op when disabled.
2. Layer 2 buffers + energy math; log-only (don't write `is_speaking` yet). Calibrate threshold.
3. Layer 1 yaw helper + `facing_camera`; log-only.
4. Layer 3 arbitration + hysteresis; begin writing `is_speaking`. Calibrate margins.
5. `current_speaker()` helper + wire one consumer (e.g. face-tracking targets the speaker).
6. Dev/calibration script + on-device tuning pass; finalize defaults.

Each step is independently testable and leaves the system working if the next isn't built yet — consistent with the capture-only / phased approach already used for episodic memory and the motion base.
