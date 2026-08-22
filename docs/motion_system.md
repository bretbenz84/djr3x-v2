# DJ-R3X Motion System — Feature Spec

Status: **Draft / proposed** · Owner: Bret · Last updated: 2026-06-14

A self-contained drive base that lets DJ-R3X move around a room on spoken command
("turn left", "come here", "move back") while autonomously avoiding obstacles and
people with Time-of-Flight sensors. An ESP32 owns the real-time motor + sensor loop;
the Mac (the existing DJ-R3X brain) sends high-level commands over USB serial.

---

## 1. Goals & non-goals

**Goals**
- Drive the robot with a 2-wheel **differential drive** (2 powered wheels + 2 omni
  caster wheels for support).
- Execute high-level spoken intents autonomously: `turn left/right`, `move
  forward/back`, `come here`, `stop`.
- **Never drive into an obstacle or person** in the direction of travel — ToF sensors
  gate all motion; the base slows in a near zone and hard-stops in a danger zone.
- Closed-loop motion (Hall encoders) so "turn 90°" and "back up 30 cm" are repeatable.
- Fail safe by default: lose comms, lose a heartbeat, or hit a fault → motors stop.
- **Manual override** — a Bluetooth gamepad paired directly to the ESP32 can take over
  and drive the robot by hand at any time, overriding autonomous/voice motion (see §11).
- Plug into the existing hardware pattern (config-gated, disabled cleanly when the
  ESP32 isn't connected — exactly like servos/LEDs today).

**Non-goals (this version)**
- **No camera-based navigation.** The USB camera is *not* part of the motion control
  loop. Obstacle/person avoidance is ToF-only. (The camera may later supply a *heading
  hint* for "come here" — see §9 — but never drives obstacle avoidance.)
- No SLAM, mapping, or waypoint navigation. This is reactive local motion, not global
  path planning.
- No autonomous roaming/patrol. The robot moves only in response to a command.

---

## 2. Bill of materials

| Qty | Part | Role |
| --- | --- | --- |
| 2 | JGB37-520 12V gear motor, 176:1, ~25 kg·cm, **Hall quadrature encoder** | Powered drive wheels |
| 2 | Omni / caster wheels | Passive support (front+back or side balance) |
| 2 | BTS7960 43A motor driver module (full H-bridge) | One per drive motor |
| 4 | VL53L0X ToF sensor (short range ~1.2 m, I²C, 940 nm) | Lateral wall clearance — left/right pairs |
| 4 | VL53L1X ToF sensor (long range ~4 m, I²C) | Room-scale sensing + stop reflex — front/rear pairs (±22.5°) |
| 1 | TCA9548A I²C multiplexer | Puts all 8 ToF sensors (each at 0x29) on one I²C bus |
| 1 | ESP32 dev board | Real-time motion controller ("the base brain") |
| 1 | 12 V battery / supply + 5 V buck converter | Motor power + ESP32/sensor logic power |
| — | USB cable ESP32 → Mac | Command + telemetry link |
| 1 | **Bluetooth gamepad — TBD** (Bluepad32-compatible: Xbox / PS4 / PS5 / 8BitDo / Switch Pro) | Manual override input, paired to the ESP32 (§11) |

---

## 3. System architecture

```
  ┌─────────────────────────────┐         USB serial         ┌────────────────────────────┐
  │  Mac  (DJ-R3X brain)         │  commands  ───────────────▶│  ESP32  (motion controller) │
  │  - speech → action_router    │  telemetry ◀───────────────│  - command parser           │
  │  - hardware/motion.py        │   (JSON lines, 115200+)    │  - PID speed loop (50–100 Hz)│
  │  - heartbeat + safety caps   │                            │  - 8× ToF read loop (mux)       │
  └─────────────────────────────┘   BT gamepad ──(BLE)──────▶ │  - control arbiter + safety │
                                     (manual override, §11)    └───────┬──────────┬──────────┘
                                                            PWM/EN    │          │  I²C
                                                          ┌───────────▼──┐   ┌───▼──────────┐
                                                          │ 2× BTS7960   │   │ 8× ToF (mux) │
                                                          └──┬────────┬──┘   └──────────────┘
                                                          M1 │        │ M2
                                                        ┌────▼──┐  ┌──▼────┐
                                                        │wheel L│  │wheel R│  (+ encoders A/B)
                                                        └───────┘  └───────┘
```

**Division of responsibility**
- **ESP32 = the reflexes.** Hard real-time: motor PWM, encoder PID, ToF reads, and the
  safety stop. It must be able to stop the robot *without the Mac* (its own ToF gate +
  watchdog). The Mac going away never leaves the robot driving.
- **Mac = the intent.** Turns speech/decisions into high-level commands, enforces
  policy (speed caps, "is motion allowed right now"), logs, and shows telemetry. It
  does **not** do real-time control.

---

## 4. Drive geometry (differential drive)

- Two powered wheels on a shared axle; two omni wheels for balance. Steering is by
  **wheel-speed difference** (skid/diff drive): both forward = straight; equal and
  opposite = spin in place; one faster = arc.
- Kinematics (ESP32):
  - `v_left  = v_linear − ω · (track_width / 2)`
  - `v_right = v_linear + ω · (track_width / 2)`
  - where `v_linear` is m/s and `ω` is rad/s.
- Odometry (per control tick) from encoder deltas:
  - `d_left/right = counts_delta / COUNTS_PER_METER`
  - `d_center = (d_left + d_right)/2`; `dθ = (d_right − d_left)/track_width`
  - integrate `x, y, θ` for distance-/angle-terminated commands.
- **Calibration constants** (measured, stored in firmware/config):
  `WHEEL_DIAMETER_MM`, `TRACK_WIDTH_MM`, `COUNTS_PER_REV`, `COUNTS_PER_METER`.
  - Encoder math: Hall ≈ 11 pulses/motor-rev/channel × **176:1** gear × 4 (quadrature)
    ≈ **7744 counts/output-rev**; verify empirically (drive 1 m / spin 360° and read).

---

## 5. Motor subsystem (BTS7960 ×2)

Each BTS7960 is a full H-bridge driving **one** motor bidirectionally.

| BTS7960 pin | Connect to | Notes |
| --- | --- | --- |
| RPWM | ESP32 PWM (LEDC) | forward duty |
| LPWM | ESP32 PWM (LEDC) | reverse duty |
| R_EN / L_EN | ESP32 GPIO (may tie together per driver) | enable; pull low to coast/disable |
| VCC | 3.3 V (logic) | from ESP32/buck — **common ground required** |
| B+ / B− | 12 V battery | motor power rail |
| M+ / M− | motor leads | — |
| R_IS / L_IS | (optional) ESP32 ADC | current sense for stall/overcurrent detection |

- Drive a wheel by PWM-ing **one** of RPWM/LPWM (the other at 0). Never both high.
- Per-wheel **PID** on encoder-measured speed → PWM duty (closed loop), so both wheels
  track the commanded speed despite friction/load differences (keeps "straight" straight).
- 43 A capacity is far above the JGB37-520's running draw; size the 12 V supply/fuse to
  the motors, not the driver max. Add a per-motor fuse.

---

## 6. Sensor subsystem (8 radial ToF: 4× VL53L0X + 4× VL53L1X)

> **Front 8x8 Matrix ToF (added 2026-07-16):** a DFRobot SEN0628 (VL53L7CX 64-zone
> matrix + onboard RP2040, I²C addr 0x33 on the same trunk) can be mounted at the
> DIRECT FRONT, pointed level, and built in with `-DMOTION_TOF_MATRIX_PRESENT=1` —
> independent of this radial array. Its floor-rejected left/right-half nearest-obstacle
> distances override (or min-combine with) the front pair `fl`/`fr`, so the front
> stop/slow reflex works with just this one sensor: full contour view of chairs and
> low clutter across a 45° FOV. The lower rows see the FLOOR at short range by
> geometry (the sensor sits ~0.15 m up); a per-row expected-floor rejection keeps the
> empty floor CLEAR while anything standing proud of it registers. Driver, math, and
> calibration: `firmware/djr3x_motion/tof_matrix.cpp` + the firmware README's
> "Front 8x8 Matrix ToF" section. Rear/sides (and reversing protection) still need
> the radial array below.

### 6.1 The I²C addressing gotcha (must-handle)
**Every VL53L0X *and* VL53L1X powers up at the same I²C address (0x29),** so 8 on one
bus collide. This base uses a **TCA9548A I²C multiplexer**: all 8 keep 0x29 and the mux
selects one channel at a time — 0 addressing logic, 0 XSHUT GPIOs. (XSHUT sequencing,
one GPIO per sensor, is not viable here — 8 sensors exceed the ESP32's free GPIOs, so
the firmware `#error`s on the XSHUT build for this layout.) Mux channel map: **ch 0-3 =
short VL53L0X, ch 4-7 = long VL53L1X.**

### 6.2 Placement (8 sensors, radial) — rev 2, 2026-07-04
8 sensors at the **540 mm base-ring surface**, every 45° starting **22.5° off the
forward axis** (nothing on the cardinals themselves). The two long-range pairs straddle
the travel axes (stop reflex fore/aft + room sense); the two short-range pairs read the
lateral wall clearance for the hallway steering assist (§6.4). Bearings are robot-frame
(REP-103: front 0°, +left/CCW):

| Mux ch | Sensor | Bearing | `tof_mm` field | Role |
| --- | --- | --- | --- | --- |
| 4 | VL53L1X (long ~4 m) | front-left +22.5° | `fl` | stop reflex + wall ahead |
| 5 | VL53L1X (long ~4 m) | front-right −22.5° | `fr` | stop reflex + wall ahead |
| 6 | VL53L1X (long ~4 m) | rear-left +157.5° | `rl` | reversing reflex |
| 7 | VL53L1X (long ~4 m) | rear-right −157.5° | `rr` | reversing reflex |
| 0 | VL53L0X (short ~1.2 m) | left-front +67.5° | `lf` | lateral clearance (assist) |
| 1 | VL53L0X (short ~1.2 m) | left-back +112.5° | `lb` | lateral clearance (assist) |
| 2 | VL53L0X (short ~1.2 m) | right-front −67.5° | `rf` | lateral clearance (assist) |
| 3 | VL53L0X (short ~1.2 m) | right-back −112.5° | `rb` | lateral clearance (assist) |

> **No down-facing cliff sensor** in this layout — so there is **no cliff / stair-drop
> protection** (a deliberate trade for all-around spatial awareness; revisit if indoor
> drop-offs are a risk). The reflex/zone logic is obstacle-only.

### 6.3 Coverage limits (call out honestly)
- **Blind spots between cones** — thin/low/narrow obstacles (chair legs, pet, cable)
  can pass between sensor cones and be missed. Keep speeds low; don't claim full coverage.
- **Range ~1.2 m** caps safe lookahead → **cap max speed** so stopping distance < min
  reliable range.
- **ToF can't classify** — it reports distance, not "person vs wall." "Avoid people" is
  really "avoid anything in the path." A person stepping in front = an obstacle that
  triggers slow/stop. That is sufficient for the goal, but state it plainly.

### 6.4 Hallway steering assist (manual forward drive)
While the gamepad commands **forward**, the firmware steers the base away from nearby
walls and centers it between two (a typical US hallway, ~915–1220 mm, leaves only
~190–340 mm per side around the 540 mm ring). Mechanism (`control.cpp`): per side,
take the nearest valid reading of that side's short pair, **capped at
`assist_engage_mm`** (default 450 — walls beyond it are ignored, so open rooms drive
untouched); steer toward the more open side at `assist_gain` rad/s per metre of
left-right imbalance, with the front long pair adding an anticipatory term (approaching
a wall at an angle steers toward the open side). The correction **adds to the
operator's stick** (capped at 60 % of `max_ang` so the human always wins), rides the
normal slew/blend path, and the Z_STOP reflex still hard-blocks head-on regardless.
Inactive when: disabled (`assist_enabled`), reversing, spinning, FULL-OVERRIDE held,
or autonomous (owner AUTO). Runtime-tunable: `assist_enabled` / `assist_engage_mm` /
`assist_gain` via `config` (bench: `set --assist 0/1 --assist-engage-mm --assist-gain`).

---

## 7. ESP32 firmware

FreeRTOS tasks (suggested):

| Task | Rate | Job |
| --- | --- | --- |
| **Control loop** | 50–100 Hz | encoder read → PID per wheel → PWM; integrate odometry; enforce distance/angle targets |
| **Sensor loop** | 20–50 Hz | read 5 ToF (continuous mode); maintain latest distances + zone flags |
| **Safety supervisor** | every control tick | gate motion on ToF zones + watchdog + faults; can force STOP independent of the Mac |
| **Serial RX** | event | parse incoming commands; update setpoints/targets; reset watchdog |
| **Serial TX (telemetry)** | 10–20 Hz | stream odometry, ToF distances, status/faults, heartbeat |

**Sample ESP32 pin budget** (illustrative; finalize on the board):
- 4 × LEDC PWM (RPWM/LPWM ×2 motors), 2–4 × GPIO enables
- 4 × encoder inputs (A/B ×2) on interrupt-capable pins (external pull-ups if using
  input-only 34–39)
- I²C SDA/SCL (2) + 5 × XSHUT (or 1 × TCA9548A on the same I²C bus)
- Optional: 2 × ADC for BTS7960 current sense; 1 × GPIO for a hardware e-stop input
ESP32 has enough usable GPIO for this; lay out PWM and interrupt pins first.

---

## 8. Communication protocol (Mac ↔ ESP32, USB serial)

> **The authoritative wire contract is [motion_protocol.md](motion_protocol.md)** (v1,
> locked). This section is a summary; if the two disagree, the protocol doc wins. It
> resolves the open conventions (sign/units, heartbeat/deadman timing, ack/`done`/`event`
> semantics, error/clamp behavior) into concrete decisions both sides build against.

- **Transport:** USB serial, **115200** baud (bump to 921600 if telemetry needs it),
  newline-delimited **JSON** objects (one per line), UTF-8. JSON keeps it debuggable
  and trivial to parse in Python.
- **Versioned:** every message carries `"v": 1`.

### 8.1 Mac → ESP32 (commands)
| Command | Example | Meaning |
| --- | --- | --- |
| heartbeat | `{"v":1,"cmd":"ping","seq":42}` | keep-alive (≥ 2× per watchdog window) |
| drive | `{"v":1,"cmd":"drive","lin":0.15,"ang":0.0}` | continuous velocity (m/s, rad/s) until changed/expired |
| turn | `{"v":1,"cmd":"turn","deg":-90,"rate":40}` | rotate in place N° (closed loop), ± = L/R |
| move | `{"v":1,"cmd":"move","dist":0.3,"speed":0.15}` | drive a fixed distance (m), ± = fwd/back |
| come | `{"v":1,"cmd":"come","heading":0,"stop_at":0.6}` | advance toward heading, stop `stop_at` m from nearest obstacle |
| stop | `{"v":1,"cmd":"stop"}` | immediate controlled stop |
| estop | `{"v":1,"cmd":"estop"}` | hard disable until explicit `clear` |
| config | `{"v":1,"cmd":"config","max_lin":0.25,...}` | set caps/zones at runtime |

- **Every motion command carries an implicit deadman:** `drive` setpoints **expire**
  (e.g. 300 ms) unless refreshed; `turn`/`move`/`come` run to their target then stop.
  The robot is never "left driving."

### 8.2 ESP32 → Mac (telemetry, 10–20 Hz)
```json
{"v":1,"t":12834,"state":"moving","fault":null,
 "owner":"auto","gamepad":"none",
 "odom":{"x":0.42,"y":0.01,"theta":-1.57,"lin":0.15,"ang":0.0},
 "tof_mm":{"fl":820,"fc":410,"fr":900,"rear":1100,"down":60},
 "zone":"slow","blocked_dir":"front","cmd_seq":42,"batt_mv":11820}
```
- `state`: `idle|moving|blocked|estop|fault`
- `owner`: `auto|manual` — who is currently driving (manual = gamepad override, §11).
- `gamepad`: `none|connected` — paired-controller link status.
- `fault`: `null | encoder_stall | overcurrent | tof_error | low_batt`
- `zone`: aggregate of the worst sensor in the direction of travel.

---

## 9. Command semantics & behaviors

| Spoken intent | Base behavior |
| --- | --- |
| **"turn left/right"** | Spin in place a default step (e.g. 45–90°, configurable) or a stated angle; closed-loop on encoders. The HOST swing check (`intelligence/motion_swing.py`) shrinks or refuses the angle when the ring/arms would sweep into a ToF return — the firmware itself does not gate spins. |
| **"move back"** | Reverse a default/stated distance; **gated by the rear ToF** — slow then stop if something's behind. |
| **"move forward"/"go"** | Drive forward a default/stated distance; front ToF gated. |
| **"stop"** | Immediate controlled stop (always honored, highest priority). |
| **"come here" / "come over here" / "come to me"** | Rotate in bounded search steps until face tracking acquires a person, align the chassis from the tracked neck offset, then advance until the nearest forward obstacle is 1.0 m away. Furniture or a wall stops the approach before the person if it is closer. |

**"Come here" person acquisition.** `intelligence/motion_agency.py` owns the deliberate
sequence. It rotates the base by `MOTION_COME_SEARCH_TURN_DEG` after each settled turn,
up to `MOTION_COME_SEARCH_MAX_TURNS` or `MOTION_COME_SEARCH_TIMEOUT_SECS`. Once normal
face tracking locks onto a visible person, the neck offset supplies the chassis bearing;
Rex makes a proportional alignment turn and then sends firmware `come` with
`MOTION_COME_REQUEST_STOP_AT_M` (default 1.0 m). The camera selects and orients toward a
person but does not navigate the path. The ESP32's forward ToF remains authoritative, so
any nearer obstacle ends the approach.

---

## 10. Obstacle & person avoidance

Direction-aware **safety zones**, evaluated on the ESP32 every control tick using the
sensors facing the travel direction (front sensors when moving forward; rear when
reversing; the swing side when spinning):

| Zone | Distance (tunable) | Action |
| --- | --- | --- |
| CLEAR | > 0.6 m | full commanded speed |
| SLOW | 0.25–0.6 m | scale speed down toward the limit |
| STOP | < 0.25 m | halt; report `blocked`; refuse further motion *in that direction* |
| CLIFF | down sensor > floor + margin | halt immediately (drop-off ahead) |

- A STOP/CLIFF condition **overrides any command** — the Mac cannot override the ESP32's
  reflex stop (only `stop`/`estop`/re-clear, never "drive into the wall anyway").
- **Phantom-tolerant, in two layers (2026-08-01** — a parked base flapped BLOCKED ~600
  times in 7 min with >1 m genuinely clear and refused a spoken "move forward"**):**
  (1) each matrix-ToF half publishes its *second*-nearest qualifying zone, so a lone
  speckle zone reads as clear (`TOF_MATRIX_MIN_OBSTACLE_ZONES`, calib.h — a real
  obstacle near enough to matter subtends multiple zones); (2) a finite move/come
  survives a block that clears within `FINITE_BLOCK_GRACE_MS` (900 ms) — velocity is
  still cut instantly, but the command pauses and resumes instead of dying on a
  transient. Only a persistent block (real wall/person) ends it with `done:blocked`.
- The robot may still move *away* from a blockage (e.g. blocked in front → reverse/turn
  still allowed if those directions are clear).
- Zone thresholds and max speed are co-tuned so **stopping distance < min reliable ToF
  range** at full speed.

**Flinch reflex (Mac-side, layered on the firmware stop).** Separate from the firmware
zones above, `intelligence/motion_agency.py` gives a parked Rex a lifelike *back-off*.
Each front matrix ToF half (`fl`/`fr`) is tracked on its own *adaptive open-distance
baseline* — it drifts toward the reading while the front is clear (capped per tick, so a
single spurious far frame can't fake an approach) and *freezes* the instant something
enters personal space, so the "where they came from" reference survives a slow approach
or a long gated stretch. A flinch fires when a side is inside `MOTION_FLINCH_TRIGGER_M`
*and* has closed by `MOTION_FLINCH_APPROACH_DROP_M` off that baseline for
`MOTION_FLINCH_CONFIRM_TICKS` consecutive ticks (a real intrusion — fast or slow, either
side — not static clutter or one noisy frame). A firmware `BLOCKED`-on-the-front state (a
crowder too close/fast for the ~1 Hz sampler) triggers the same back-off immediately. The
retreat is *soft-capped* by the rear ToF (`rl`/`rr`) so he keeps
`MOTION_FLINCH_REAR_MARGIN_M` of clearance and stops short of the wall; cornered — or
*blind* behind (both rear sensors dead, where the firmware stop also fails open, per §safety)
— he holds. This is a decision layer only — it issues a normal `move` (−distance), so the
firmware's always-on rear-ToF STOP reflex remains the hard backstop that makes wall
contact impossible whenever the rear sensors report. It needs no tracked/known person and,
being a reflex, may fire mid-sentence (`MOTION_FLINCH_ALLOW_MID_SENTENCE`).

---

### 10.1 Autonomous liveliness batch (2026-08-19)

`intelligence/motion_agency.py` gained four sibling behaviors alongside
flinch/realign/approach — all decision-layer only, all through the ToF-gated
closed-loop verbs, each behind its own kill switch (config clusters named):

- **IDLE WANDER** (`MOTION_IDLE_WANDER_*`) — occasional paired weight-shift
  maneuvers (slight turn+inverse, or short fore/aft shuffle+inverse; zero net
  pose drift). Clearance-gated per axis (fails closed on unknown sensors),
  roominess-scaled, silenced by no-drive rooms, user holds, mid-sentence, and
  the traction stand-down; an aborted wander turn feeds the traction detector.
- **RADAR ORIENT** (`MOTION_RADAR_ORIENT_*`) — nobody on camera but the LD2450
  ring shows a persistent body → neck glance within ~40°, base turn beyond it.
  (Ring targets are seam-deduped host-side: `hardware/radar.py::_seam_merge`,
  `RADAR_SEAM_MERGE_*`.)
- **EDGE-IN** (`MOTION_EDGE_IN_*`) — mid-conversation at social distance, one
  short slow step closer (front-ToF-checked, keeps 1 m clearance, minutes-long
  cooldown).
- **OBJECT STEP** (`MOTION_OBJECT_STEP_*`) — an object he just asked about and
  that sits roughly ahead pulls the body one small step toward it; armed at ask
  time, executed after the human's answer moment.

The come-here errand also gained a **drive gaze** (`MOTION_COME_GAZE_COMP_*`):
while the approach drives, the neck counter-pans the IMU yaw deviation so his
gaze holds the travel heading while the firmware assist arcs around obstacles,
and the camera dips slightly (`MOTION_COME_DRIVE_PITCH`) to see floor clutter.
Speed variability: `cmd:come` accepts an optional `speed` (protocol §5.5),
spontaneous approaches saunter (`MOTION_APPROACH_SPEED_JITTER*`), and
exploration legs jitter per leg (`EXPLORE_LEG_SPEED_JITTER_*`).

## 11. Manual control & Bluetooth gamepad override

A Bluetooth gamepad **paired directly to the ESP32** (not the Mac) lets you grab the
wheel and drive the robot by hand, overriding autonomous/voice motion. Pairing at the
ESP32 layer keeps manual control the lowest-latency, most authoritative input — it works
even if the Mac or the USB link is down, which is exactly what you want from an override
and for recovering the robot.

### 11.1 Controller support (none chosen yet)
Use **Bluepad32** on the ESP32 (Arduino/ESP-IDF). It normalizes a wide range of
controllers — Xbox Wireless (BLE), PS4 DualShock4, PS5 DualSense, Switch Pro, 8BitDo,
many generic HID pads — into one API, so the firmware's input layer is
**controller-agnostic** and the final mapping can be decided once a pad is in hand.

- **Recommended when you buy one** (good BLE support): Xbox Wireless Controller (recent
  BLE models), PS4/PS5, 8BitDo Pro 2 / SN30 Pro, Switch Pro. Avoid no-name clones
  (flaky pairing/latency).
- **Pairing UX:** ESP32 enters pairing/scan on boot or via a button; put the pad in
  pairing mode; Bluepad32 stores the bond and auto-reconnects next time. Connection
  status is reported in telemetry (`gamepad`).

### 11.2 Control mapping (abstract — finalize with the chosen pad)
| Input | Action |
| --- | --- |
| Left stick Y | linear velocity (forward/back) |
| Left stick X *or* right stick X | angular velocity (turn) — single- or twin-stick, configurable |
| Trigger / shoulder | speed scale ("boost"/"creep") |
| Face button (e.g. B) | **E-stop** (immediate stop) |
| Start / Menu | clear e-stop / return to AUTO |
| Toggle button | latch MANUAL on/off |
| Held "full-override" button | temporarily bypass ToF gating (see §11.4) |

Sticks have a deadzone so resting drift doesn't count as input.

### 11.3 Mode arbitration (the override)
Control `owner` is **AUTO** (Mac/voice) or **MANUAL** (gamepad):
- **Any meaningful gamepad input** (stick past deadzone, drive button) → switch to
  **MANUAL**. While MANUAL, the ESP32 **ignores** Mac drive/turn/move/come commands (it
  still accepts `stop`, `estop`, `config`, `ping`).
- **Return to AUTO:** explicit toggle button, **or** after a manual-idle timeout
  (`MOTION_MANUAL_IDLE_RETURN_SECS`, e.g. 3–5 s) if `MOTION_MANUAL_AUTORETURN` is on.
  *Open choice:* some operators prefer **explicit-only** return so the robot never
  "resumes itself" — default to explicit toggle, make auto-return opt-in.
- The ESP32 reports `owner` in telemetry so the Mac/GUI shows who's driving and the Mac
  **yields** (pauses sending autonomous commands; a voice "come here" is ignored or
  queued while MANUAL).

### 11.4 Safety interaction (key decision)
- **Default = MANUAL-ASSISTED:** the gamepad drives, but the ToF safety zones + cliff
  stop (§10) **still apply** — you can't manually drive into a wall, person, or edge.
  Recommended default.
- **Optional FULL-OVERRIDE (held button):** while a dedicated button is held, ToF gating
  is bypassed for nudging through tight spots / recovery. Re-enables on release. The
  operator takes responsibility; use sparingly.
- **E-stop is always honored**, in any mode.
- **Disconnect failsafe:** if the paired pad drops while MANUAL, motors **stop
  immediately** (never hold the last stick value) and the base goes to safe idle; AUTO
  resumes only on an explicit command. Mirrors the Mac heartbeat watchdog.

### 11.5 Precedence (highest first)
1. **E-stop** (button or `estop` command)
2. **ESP32 reflex safety stop** (ToF STOP / CLIFF) — *except* while FULL-OVERRIDE is held
3. **MANUAL gamepad input**
4. **AUTO commands** from the Mac (voice / autonomous)
5. Idle (stopped)

So the controller is a true override of autonomous motion, while the non-negotiable
safety stop stays above it (unless the operator deliberately takes full control).

### 11.6 Notes for the Mac side
The gamepad is entirely an ESP32-side concern — the Mac needs **no driver**, just
*awareness*: read `owner`/`gamepad` from telemetry, stop issuing autonomous commands
while `owner == manual`, and surface "MANUAL (gamepad)" + controller status in the GUI.
Config knobs: `MOTION_MANUAL_IDLE_RETURN_SECS`, `MOTION_MANUAL_AUTORETURN`; deadzone and
speed-scale live in ESP32 firmware config.

---

## 12. Safety model

Layered, defense-in-depth:
1. **ESP32 reflex stop** — ToF zones + cliff, independent of the Mac.
2. **Heartbeat watchdog** — no `ping` from the Mac within the window (e.g. 500 ms) →
   stop motors. Covers Mac crash, USB unplug, app exit.
3. **Command deadman** — `drive` setpoints expire; finite commands self-terminate.
4. **Speed caps** — hard max linear/angular in firmware (not just policy).
5. **Fault stop** — encoder stall (commanded but not moving → likely jammed/lifted),
   overcurrent (BTS7960 IS), low battery → stop + report.
6. **E-stop** — software `estop` command and (recommended) a **physical button** that
   cuts the 12 V motor rail. Recovery requires an explicit clear.
7. **Mac-side policy gates** — autonomous motion is **suppressed** when the conversation
   is paused (`INTERACTION_PAUSED`, e.g. Memory Banks open), during family-safe/sensitive
   moments, or when the operator disables it. (Mirror the existing audio-suppression
   pattern.) Note: the **manual gamepad override is not** subject to these Mac-side gates
   — it lives on the ESP32 and stays available even when autonomous motion is paused.
8. **Gamepad disconnect failsafe** — if the paired controller drops while driving
   MANUAL, motors stop immediately (see §11.4).
9. **Manual override precedence** — manual driving overrides AUTO commands, but the
   reflex/e-stop layers stay above it (full precedence list in §11.5).
10. **Start-up safe** — boots to `idle`, motors disabled, owner = AUTO, until an
    explicit command or gamepad input.

---

## 13. Integration with the existing codebase

Follow the established hardware pattern (servos/LEDs are config-gated and degrade
cleanly when unplugged):

- **Config (`config.py`)** — add `MOTION_ESP32_PORT` (serial device path). Motion is
  **disabled unless the port is set**, exactly like `MAESTRO_PORT` / `ARDUINO_HEAD_PORT`.
  Plus tunables: `MOTION_MAX_LINEAR_MS`, `MOTION_MAX_ANGULAR_DEG_S`, `MOTION_STOP_ZONE_M`,
  `MOTION_SLOW_ZONE_M`, `MOTION_DEFAULT_TURN_DEG`, `MOTION_COME_STOP_AT_M`,
  `MOTION_HEARTBEAT_MS`, `MOTION_ENABLED` master switch.
- **`hardware/motion.py`** — mirror `hardware/servos.py`: `connect()` (open serial,
  handshake/version-check), `send(cmd)`, a background reader thread for telemetry, and a
  thread-safe latest-telemetry snapshot. Returns connected/not so `main.py` can log
  `Motion base: enabled/disabled` in its Step-4 hardware block.
- **`intelligence/motion_controller.py`** (new) — the high-level API the rest of the app
  calls: `turn(deg)`, `move(dist)`, `come_here(...)`, `stop()`, plus Mac-side safety
  (heartbeat thread, speed caps, the `INTERACTION_PAUSED`/family-safe gate, debouncing).
- **`intelligence/action_router.py`** — add motion intents so spoken commands route to
  the controller: `motion.turn`, `motion.move`, `motion.come`, `motion.stop`. Classify
  "turn left/right", "move/back up/come forward", "come here", "stop"/"halt"/"freeze".
  `stop` must be high-priority and always executable.
- **GUI (`gui/dashboard.py`)** — optional MOTION panel: live ToF distances (a small
  radar/bar view), odometry/heading, current state, and a prominent **E-STOP button**.
- **Telemetry → memory/log** — motion events ("I rolled over to Bret", "backed away
  from an obstacle") can optionally feed the existing episodic memory for flavor.

---

## 14. Calibration & tuning

1. **Encoder counts/rev** — spin one output rev by hand / under power; confirm count.
2. **Counts per meter & track width** — drive a measured 1 m straight; spin a measured
   360°; solve for `COUNTS_PER_METER` and `TRACK_WIDTH_MM` from odometry error.
3. **Per-wheel PID** — tune so a "straight" command tracks straight and speed is steady
   under load; minimize wheel mismatch.
4. **ToF zones** — measure actual stopping distance at max speed; set STOP zone above
   it with margin; set SLOW so deceleration is smooth.
5. **Turn accuracy** — command 90°, measure actual; trim `TRACK_WIDTH_MM`.

---

## 15. Test plan & acceptance criteria

**Bench (wheels off the ground)**
- [ ] Each motor spins both directions under PWM; enables gate correctly.
- [ ] Both encoders count, correct sign; PID holds a commanded speed.
- [ ] All 5 ToF return distinct, sane distances (addressing works).
- [ ] Serial command/telemetry round-trips; version handshake works.
- [ ] Watchdog: stop sending `ping` → motors stop within the window.

**Floor**
- [ ] `move 1 m` lands within ±5 cm; `turn 90°` within ±5°.
- [ ] Straight command drives straight (no consistent drift).
- [ ] Forward into a wall: SLOW then STOP before contact; reports `blocked`.
- [ ] Person steps in front while moving → stops before contact.
- [ ] `move back` stops for an obstacle behind.
- [ ] Cliff sensor halts at a table edge / top of a step.
- [ ] USB unplug mid-move → robot stops (watchdog).
- [ ] `stop`/e-stop halts immediately from any state.
- [ ] Spoken "turn left", "move back", "stop" produce the right motion via the router.

**Acceptance:** all floor items pass; no contact with a wall/person in 20 consecutive
mixed commands; clean enable/disable when the ESP32 is unplugged (no crashes, logged
like servos).

---

## 16. Risks, limitations & open questions

- **ToF blind spots / range** — narrow cones miss thin/low obstacles; ~1.2 m range caps
  speed. Mitigation: low speed, conservative zones, more/angled sensors later. *Accept
  that coverage is not complete and tune speed accordingly.*
- **No classification** — can't tell a person from furniture; "avoid people" = avoid
  obstacles. Acceptable for the goal; documented.
- **Odometry drift** — wheel slip / carpet vs hard floor degrades distance/angle
  accuracy over time. Fine for short reactive moves; not for long navigation.
- **"Come here" target** — needs a heading source without camera navigation (see §9).
- **Cliff/stairs** — only covered if the 5th sensor is down-facing; otherwise a real
  fall risk. *Recommend the downward sensor.*
- **Power/EMI** — motor noise on the logic rail; keep ESP32 on a clean 5 V buck (or USB)
  with common ground and decoupling; fuse the motor rail.
- **Tip-over** — a tall robot accelerating/stopping hard on 2 driven + 2 omni wheels can
  rock; keep accel limits gentle and the center of mass low.
- **Two-way authority** — confirm the ESP32 reflex stop can *always* override a Mac
  command (it must).
- **Gamepad selection (none chosen yet)** — pick a Bluepad32-supported BLE pad and
  verify pairing reliability + latency before committing; avoid no-name clones.
- **Does manual override bypass ToF safety?** Default is MANUAL-ASSISTED (ToF still
  protects); FULL-OVERRIDE is behind a held button. Confirm this is the behavior you
  want (§11.4).
- **Auto-return to AUTO vs explicit toggle** — decide whether the robot resumes
  autonomous mode after a manual-idle timeout or only on an explicit toggle (§11.3).

---

## 17. Phased roadmap

- **Phase 0 — bring-up:** wiring, ESP32 firmware skeleton, motors spin, encoders read,
  5 ToF addressed & reading, serial echo + version handshake. (Bench only.)
- **Phase 1 — closed-loop base:** PID speed control, odometry, `move`/`turn`/`stop`
  with ToF STOP/SLOW gating + watchdog. Drive from a laptop terminal.
- **Phase 1.5 — manual override (great test tool):** pair a BT gamepad via Bluepad32,
  MANUAL-ASSISTED driving with ToF gating + disconnect failsafe + e-stop button.
  Invaluable for shaking down the mechanics by hand before voice autonomy is solid.
- **Phase 2 — Mac integration:** `MOTION_ESP32_PORT` config, `hardware/motion.py`,
  `motion_controller.py`, `action_router` intents → spoken "turn/move/back/stop" work.
  Heartbeat + pause/family-safe gating.
- **Phase 3 — "come here" + polish:** heading hint, smoother avoidance, GUI motion panel
  with ToF view + E-STOP, optional episodic flavor lines.

---

## 18. Glossary

- **Differential drive** — steering by left/right wheel-speed difference.
- **ToF** — Time-of-Flight distance sensor (VL53L0X), reports range in mm.
- **Deadman / watchdog** — auto-stop when commands/heartbeats stop arriving.
- **Zone** — distance band (CLEAR/SLOW/STOP/CLIFF) that gates speed in the travel
  direction.
- **Odometry** — position/heading estimate integrated from encoder counts.
