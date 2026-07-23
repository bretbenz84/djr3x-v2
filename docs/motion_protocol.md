# DJ-R3X Motion Protocol — Mac ↔ ESP32 wire contract

Status: **v1 — locked for implementation** · Owner: Bret · Last updated: 2026-06-14

This is the **authoritative contract** for the serial link between the Mac (DJ-R3X
brain) and the ESP32 (motion controller). Both sides — `hardware/motion.py` /
`intelligence/motion_controller.py` on the Mac, and the ESP32 firmware — implement
*this* document. If the feature spec ([motion_system.md](motion_system.md) §8) and this
file ever disagree, **this file wins**; update the spec to match.

The goal of locking this first is so the two tracks can be built in parallel without
drifting. Anything still genuinely open is collected in [§14](#14-open-points). Everything
else is **decided** — pick the value here, don't re-litigate it in code.

---

## 0. Scope & responsibilities

| Side | Implements | Module(s) |
| --- | --- | --- |
| **Mac** | Command sender, telemetry consumer, heartbeat, handshake, Mac-side policy gates | `hardware/motion.py`, `intelligence/motion_controller.py` |
| **ESP32** | Command parser, telemetry emitter, real-time control, **reflex safety stop**, deadman/watchdog | firmware |

The ESP32 is the source of truth for *physical state*; the Mac is the source of truth for
*intent*. The Mac can **request** motion; the ESP32 can always **refuse or stop** it
(reflex safety, §[motion_system.md](motion_system.md) §10). No protocol message lets the
Mac override the ESP32 reflex stop — that is by design.

---

## 1. Transport & framing

- **Physical:** USB serial (CDC-ACM). Default **115200** baud, 8N1, no flow control.
  Bump to **921600** only if telemetry saturates the link; baud is a build/config
  constant on both sides, not negotiated at runtime.
- **Framing:** **NDJSON** — one UTF-8 JSON object per line, terminated by a single
  `\n` (0x0A). A leading/trailing `\r` (0x0D) is tolerated and stripped. No other
  framing, no binary, no nested newlines (JSON must be emitted compact / single-line).
- **Max line length: 512 bytes.** A line that exceeds this before a `\n` is **discarded
  up to and including the next `\n`**, a parse-error counter is incremented, and (ESP32
  side) an optional `log` message may be emitted. Never block, never crash on a long or
  garbled line.
- **Robustness rule:** any line that is not valid JSON, is missing `v`, or is missing
  `cmd`/`type` is **dropped silently** (counter++). A malformed line never aborts the
  reader and never partially applies.
- Both sides MUST tolerate interleaving: a telemetry frame may arrive between a command
  and its `ack`. Match by `seq`, never by arrival order.

---

## 2. Message envelope & versioning

Every message — both directions — is a flat JSON object carrying:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `v` | int | **yes** | Protocol version. **`1`** for this document. |
| `cmd` | string | Mac→ESP32 | Command name (see §5). Present only on commands. |
| `type` | string | ESP32→Mac | Message kind (see §6). Present only on ESP32 messages. |

A message has **exactly one** of `cmd` (Mac→ESP32) or `type` (ESP32→Mac). All other
fields are message-specific. **Unknown extra fields are ignored**, not rejected — this is
the forward-compat seam (§13).

**Version mismatch:** if either side sees `v` it does not support, it MUST NOT act on the
message. The Mac treats an unsupported ESP32 `v` as "incompatible firmware → motion
disabled" and logs it (handshake, §3). The ESP32 acks an unsupported command `v` with
`reason:"bad_version"` and takes no action.

---

## 3. Handshake (connection bring-up)

The link is **session-oriented**: the Mac re-establishes it every time it opens the port.

```
Mac opens serial
Mac  → {"v":1,"cmd":"hello","seq":1,"host":"djr3x","proto":1}
ESP32→ {"v":1,"type":"hello","proto":1,"fw":"0.3.1","caps":["drive","turn","move","come","gamepad"],"boot_id":7741}
        (… then telemetry begins streaming …)
```

- The Mac sends `hello` first and waits up to **`MOTION_HANDSHAKE_TIMEOUT_MS` (default
  1500 ms)** for the ESP32 `hello`.
- **No `hello` reply, or `proto` the Mac can't speak → motion is DISABLED** for the
  session (logged like an unplugged servo bus; `connect()` returns `False`). The robot
  still runs; it just won't move.
- `caps` lets the Mac feature-gate optional commands (e.g. don't expose "come here" if
  `"come"` isn't advertised). The minimal viable `caps` is `["drive","stop"]` (Phase 0/1).
- `boot_id` is a random/counter value that changes on every ESP32 reset. If the Mac sees
  `boot_id` change mid-session (in telemetry or a new `hello`), the ESP32 rebooted → the
  Mac re-runs handshake and treats prior odometry as invalid.
- After a successful handshake the Mac begins the heartbeat (§7) immediately.

The ESP32 boots to a **safe idle** (motors disabled, `owner:"auto"`, `state:"idle"`)
*before* any handshake — it never requires the Mac to be safe.

---

## 4. Conventions (units, signs, frames, time)

Decided once, here, so firmware and Python never disagree.

- **Frame:** ROS REP-103 right-handed. **+x = forward, +y = left, +z = up.**
- **Linear velocity / distance:** metres, m/s. **`+` = forward, `−` = reverse.**
- **Angular velocity / angle:** **`+` = LEFT / counter-clockwise (CCW), `−` = RIGHT /
  clockwise (CW)** (right-hand rule about +z). This is the *one* convention used
  everywhere: `drive.ang`, `turn.deg`, `come.heading`, and `odom.theta`.
  - So `turn deg:+90` rotates **90° to the LEFT**; `turn deg:-90` rotates 90° right.
  - `drive ang:+0.5` curves left.
- **Angle units:** `drive.ang` is **rad/s**; `turn.deg`, `turn.rate`, `come.heading`,
  `default_turn_deg` are **degrees / deg·s⁻¹**. `odom.theta` is **radians**, wrapped to
  `(−π, π]`. (Rationale: continuous control is SI; human-facing discrete commands are
  degrees.)
- **`come.heading` frame:** degrees **relative to the robot's current heading at the
  moment the command is received** (0 = straight ahead, + = left). It is a one-shot
  point-then-go hint, not a world bearing.
- **Time `t`:** ESP32 **milliseconds since boot**, `uint32`. Wraps at ~49.7 days; both
  sides treat `t` as monotonic-with-wrap (use unsigned subtraction for deltas, never
  assume it only grows). Telemetry carries `t`; the Mac never sends wall-clock time.
- **`seq`:** see §7.1.
- **Booleans are JSON `true`/`false`**, never `0`/`1`. **`null`** is used only where a
  field's table says so (e.g. `fault`).

---

## 5. Mac → ESP32 commands

Every command carries `v`, `cmd`, and a **`seq`** (§7.1). Fields below are *in addition*
to those. Ranges out of bounds are **clamped to the active caps and accepted** with
`ack.reason:"clamped"` — motion is never rejected merely for being too fast/far (see §11).

### 5.1 `ping` — heartbeat
```json
{"v":1,"cmd":"ping","seq":42}
```
Keep-alive only. **Not individually acked** (its effect is the watchdog reset; liveness is
visible via `cmd_seq` in telemetry). Sent at the heartbeat rate (§7.2). A `ping` does **not**
refresh a `drive` setpoint (§7.3) — only a new `drive` does.

### 5.2 `drive` — continuous velocity (teleop-style)
```json
{"v":1,"cmd":"drive","seq":43,"lin":0.15,"ang":0.0}
```
| Field | Type | Units | Required | Default |
| --- | --- | --- | --- | --- |
| `lin` | float | m/s, +fwd | yes | — |
| `ang` | float | rad/s, +left | yes | — |

Sets the target velocity **until changed or expired**. Subject to the **300 ms drive
deadman** (§7.3): without a refreshing `drive`, the base ramps to a stop. Used by the
heading-less teleop path and (conceptually) mirrors gamepad sticks — though the gamepad
feeds the ESP32 *locally*, not via this command (§12).

### 5.3 `turn` — closed-loop rotate in place
```json
{"v":1,"cmd":"turn","seq":44,"deg":-90,"rate":40}
```
| Field | Type | Units | Required | Default |
| --- | --- | --- | --- | --- |
| `deg` | float | degrees, +left | yes | — |
| `rate` | float | deg/s, magnitude | no | `MOTION_DEFAULT_TURN_RATE` |

Spins in place by `deg`, then stops. When the IMU is healthy at command start, completion
is closed on signed integrated gyro yaw (physical chassis rotation); encoder odometry is
the fallback when the IMU is unavailable. An IMU-verified turn that does not reach its
target before the bounded verification timeout aborts instead of accepting wheel slip as
rotation. **Finite** → emits a `done` (§6.3) when complete/aborted. Aborts if the
swing-side ToF enters STOP (`done result:"blocked"`).

### 5.4 `move` — closed-loop straight distance
```json
{"v":1,"cmd":"move","seq":45,"dist":0.30,"speed":0.15}
```
| Field | Type | Units | Required | Default |
| --- | --- | --- | --- | --- |
| `dist` | float | m, +fwd / −back | yes | — |
| `speed` | float | m/s, magnitude | no | derived from caps |

Drives `dist` metres (sign = direction), ToF-gated in the travel direction, then stops.
During forward travel, the side pairs and split front view add a bounded hallway-centering
correction; reverse travel is not auto-centered. **Finite** → `done`.

### 5.5 `come` — advance toward a heading, stop at social distance
```json
{"v":1,"cmd":"come","seq":46,"heading":0,"stop_at":0.6}
```
| Field | Type | Units | Required | Default |
| --- | --- | --- | --- | --- |
| `heading` | float | degrees, +left, robot-relative | no | `0` (straight ahead) |
| `stop_at` | float | m from nearest fwd obstacle | no | `MOTION_COME_STOP_AT_M` |

Turn toward `heading` (one-shot), then advance until the nearest forward ToF reads
`stop_at`, then stop. **Finite** → `done` (`completed` on reaching `stop_at`, `blocked`
if it can't proceed). Optional capability (`caps` must include `"come"`).

### 5.6 `stop` — controlled stop
```json
{"v":1,"cmd":"stop","seq":47}
```
Immediate controlled deceleration to idle. **Always honored**, in any mode (AUTO or
MANUAL), highest command priority. Cancels any in-flight finite command (that command
emits `done result:"superseded"`). Acked.

### 5.7 `estop` — hard disable
```json
{"v":1,"cmd":"estop","seq":48}
```
Cuts motor drive immediately and latches `state:"estop"`. **No further motion** (drive/
turn/move/come are acked-rejected `reason:"estop"`) until a `clear` (§5.8). Mirrors the
physical e-stop. Acked.

### 5.8 `clear` — clear estop / latched fault
```json
{"v":1,"cmd":"clear","seq":49}
```
Clears a latched `estop` or a clearable `fault` and returns to `state:"idle"`, `owner`
unchanged. Has no effect on an active reflex STOP/CLIFF zone (that clears only when the
obstacle/edge is gone or the robot moves away). Acked (`accepted:false reason:"nothing_to_clear"`
if already idle).

### 5.9 `config` — runtime tunables
```json
{"v":1,"cmd":"config","seq":50,"max_lin":0.25,"max_ang":1.2,"slow_zone_m":0.6,"stop_zone_m":0.25}
```
Sets any subset of the runtime parameters in §10. Omitted keys are unchanged. Values are
**clamped to firmware hard caps** (a `config` can lower a cap, never raise it above the
compiled-in limit). Acked; the ack echoes the *effective* (post-clamp) values.

### 5.10 `wheel` — single-wheel bring-up jog (diagnostic)
```json
{"v":1,"cmd":"wheel","seq":51,"side":"left","frac":0.35,"ms":1500}
```
| Field | Type | Meaning | Required | Default |
| --- | --- | --- | --- | --- |
| `side` | string | `"left"`/`"l"` or `"right"`/`"r"` | yes | — |
| `frac` | float | signed drive fraction −1..1 of full duty; **+ = that wheel forward** per its `MOTOR_SIGN_*` | yes | — |
| `ms` | int | run time, then auto-stop | no | 1500 (hard cap 3000) |

Powers **exactly one** wheel's H-bridge at a fixed duty for `ms`, **bypassing the
differential kinematics AND the velocity PID** — the open-loop bring-up test for "is this
motor wired to the right side, and does it spin the right way?". Deliberately open-loop so
a mis-wired/unread encoder can't fight the test (unlike `drive`, whose PID would). The
magnitude is floored at the stiction breakaway (`min_duty`) and clamped to full duty; the
other wheel is held off. Runs as an AUTO finite command: it completes with
`done result:"completed"` after `ms`, and the heartbeat watchdog (§7.2), `stop`, `estop`,
or a gamepad takeover cut it like any finite command. **NOT** obstacle-gated (a spinning
wheel on a stand isn't translating) — stand only, wheels off the ground. Acked (`clamped`
if `frac`/`ms` were clamped). Diagnostic only: **not advertised in `hello.caps`** and not
issued by the normal Mac controller — it is driven by `firmware/tools/motion_bench.py wheel`.

### 5.11 `batt_full` — sync the SOC gauge to 100%
```json
{"v":1,"cmd":"batt_full","seq":52}
```
No fields. The operator observed the **charger's taper current reach cutoff** — direct
evidence the pack is full that the firmware itself can't see (mid-absorption the pack is
never "at rest", so the §6.1 rest-voltage full anchor can't fire until after a
power-cycle). Sets the coulomb ledger to 100% (`batt_soc:100`), arms the same
once-per-charge latch as the rest anchor, and **persists to NVS immediately**. Applied on
the next 1 Hz battery tick, so telemetry reflects it within ~1 s. Never gated (it cannot
move the base): accepted under estop, faults, and manual ownership, like `config`/`ping`.
Acked `ok`; on a build/board without the coulomb gauge (no motor-ranged shunt, or no
INA226) → `ack accepted:false reason:"unsupported_cap"`. Advertised in `hello.caps` as
`"batt_full"` when available. Sender in practice is the Mac **menu bar battery meter**
(`tools/rex_battery_menubar.py`, "Set Battery to 100%") — the meter is otherwise a
purely passive telemetry listener.

---

## 6. ESP32 → Mac messages

Distinguished by `type`. `telemetry` is periodic; the rest are event-driven.

### 6.1 `telemetry` — periodic state (default 10 Hz)

> Rate history: launched at 20 Hz; halved 2026-07-11 when the frame grew to ~480 B
> (imu/gp/wheels/battery fields) — 20 Hz was ~84% of the 115200-baud line and
> pad-driving load backed frames up (stale GUI). Every consumer reads a
> latest-snapshot, so 10 Hz stays fresher than anyone consumes at ~42% line util.
> Numeric fields are quantized at emit (mm-scale odometry, 0.1° attitude).
```json
{"v":1,"type":"telemetry","t":12834,"state":"moving","owner":"auto","gamepad":"none",
 "fault":null,"zone":"slow","blocked_dir":"front","cmd_seq":42,
 "odom":{"x":0.42,"y":0.01,"theta":-1.57,"lin":0.15,"ang":0.0},
 "tof_mm":{"fl":1100,"fr":1400,"rl":2600,"rr":2400,"lf":240,"lb":260,"rf":480,"rb":500},
 "batt_mv":11820,"errs":0}
```
| Field | Type | Meaning |
| --- | --- | --- |
| `t` | uint32 | ESP32 ms since boot (§4). |
| `state` | enum | `idle` \| `moving` \| `blocked` \| `estop` \| `fault` \| `comms_lost` (§8). |
| `owner` | enum | `auto` \| `manual` (who's driving; §12). |
| `gamepad` | enum | `none` \| `connected`. |
| `fault` | enum\|null | `null` \| `encoder_stall` \| `overcurrent` \| `tof_error` \| `low_batt` \| `comms_lost` (§9). |
| `zone` | enum | Worst zone in the travel direction: `clear` \| `slow` \| `stop` \| `cliff` (§9). |
| `blocked_dir` | enum | `none` \| `front` \| `rear` \| `left` \| `right`. |
| `cmd_seq` | int | `seq` of the most recently **applied** command (heartbeat liveness + ack matching aid). |
| `odom` | object | `{x,y}` m, `theta` rad (−π,π], `lin` m/s, `ang` rad/s. Reset to 0 on `boot_id` change / explicit `config` reset. |
| `tof_mm` | object | Per-sensor distance in mm — **8 radial sensors** (§6), every 45° starting 22.5° off the forward axis: long-range front pair `fl,fr` + rear pair `rl,rr` (VL53L1X, ±22.5° off the axis) and short-range left pair `lf,lb` + right pair `rf,rb` (VL53L0X). A sensor in error reports `-1`; a large value (per-type out-of-range cap) means nothing in range = clear. No down/cliff sensor in this layout. |
| `batt_mv` | int | Pack voltage, millivolts (INA226 VBUS; `-1` = no sensor / VBUS unwired). |
| `batt_ma` | int | Pack current, milliamps (INA226 shunt, signed, + = discharging; `0` until a shunt is configured — `BATT_SHUNT_MICROOHM` in calib.h). Independent of `batt_mv`: either can report without the other. |
| `batt_soc` | int | Coulomb-counted state of charge, 0–100% (`-1` = unknown). Ledger persists in ESP32 NVS across power-off; reconciled at boot against LiFePO4 rest-voltage anchors (≥ full-anchor at rest → 100%; below a knee → clamped down; on the flat plateau → the saved ledger is trusted). Charging happens while the ESP32 is dark, so the full anchor is what re-syncs the gauge after a charge; the host can also sync it explicitly with `batt_full` (§5.11) when it watches a charge finish live. |
| `imu` | object | MPU-6050 attitude: `{ok}` always; when `ok:true` also `{pitch,roll,yaw}` in degrees. `pitch`/`roll` are gravity-referenced (complementary filter); `yaw` is bias-corrected gyro integration **relative to boot heading** (drifts slowly; no indoor magnetometer by design). `ok:false` = no sensor answered the boot probe. |
| `errs` | int | Cumulative parse/framing error count (for link-health monitoring). |

The Mac keeps the **latest** telemetry as a thread-safe snapshot (mirror of the servo
pattern). `boot_id` is *not* in every telemetry frame to save bytes — it lives in `hello`
and in a `boot` event (§6.4); the Mac caches it from there.

### 6.2 `ack` — command acknowledgement (every non-`ping` command)
```json
{"v":1,"type":"ack","seq":44,"accepted":true,"reason":null}
```
| Field | Type | Meaning |
| --- | --- | --- |
| `seq` | int | Echoes the command's `seq`. |
| `accepted` | bool | Whether the command was admitted (a finite command being *admitted* is not the same as *completed* — see `done`). |
| `reason` | enum\|null | `null` when accepted; otherwise / informationally: `clamped`, `manual_override`, `estop`, `fault`, `unknown_cmd`, `bad_field`, `bad_version`, `nothing_to_clear`, `unsupported_cap`. `clamped` may appear with `accepted:true`. |

The Mac SHOULD wait for `ack` (typ. < 50 ms) before considering a command in flight, but
MUST tolerate a missing ack (treat as not-applied after a short timeout; safety never
depends on an ack arriving).

### 6.3 `done` — finite-command completion (`turn`/`move`/`come`)
```json
{"v":1,"type":"done","seq":44,"result":"completed","odom":{"x":0.0,"y":0.0,"theta":-1.57}}
```
| Field | Type | Meaning |
| --- | --- | --- |
| `seq` | int | The finite command's `seq`. |
| `result` | enum | `completed` \| `blocked` \| `aborted` \| `superseded` \| `estopped`. |
| `odom` | object | Odometry snapshot at termination. |

Lets the Mac controller know "the turn finished" vs "it stopped early because blocked",
without polling odometry. `drive`/`stop`/`estop`/`clear`/`config`/`ping` never emit `done`.

### 6.4 `event` — discrete notable transitions
```json
{"v":1,"type":"event","t":13002,"event":"owner_change","owner":"manual"}
```
`event` ∈ `boot` (carries `boot_id`,`fw`), `owner_change` (carries `owner`),
`gamepad` (carries `gamepad`: connected/none), `zone_block` (carries `blocked_dir`),
`cliff`, `fault` (carries `fault`), `fault_clear`, `estop`, `estop_clear`,
`comms_lost`, `comms_restored`. Events are **edge-triggered** (sent once on transition),
unlike telemetry which is level/periodic. They are advisory — telemetry remains the
authoritative level state — but they're the clean hook for GUI updates and episodic-memory
flavor lines ("backed away from an obstacle"). The Mac must not *depend* on receiving an
event (it could be dropped); reconcile against telemetry.

### 6.5 `log` — optional debug text
```json
{"v":1,"type":"log","t":13050,"lvl":"warn","msg":"tof[3] init retry 2"}
```
Free-text firmware diagnostics, `lvl` ∈ `debug|info|warn|error`. The Mac routes these into
its logger (e.g. `[motion_fw]`). Never load-bearing.

---

## 7. Timing, heartbeat & deadman

Three independent safety timers. **All defaults are firmware constants; the Mac mirrors
them in `config.py` so policy and firmware agree.**

### 7.1 `seq` rules
- The Mac assigns a **monotonically increasing** `seq` (uint32, wraps) to **every**
  command, including `ping`. It is the correlation key for `ack`/`done` and the heartbeat
  freshness indicator (`cmd_seq` in telemetry).
- The ESP32 never invents a `seq`; it only echoes the relevant command's `seq`.
- On `boot_id` change the Mac MAY reset its `seq` to 1; the ESP32 must not assume `seq`
  only increases across a reboot.

### 7.2 Heartbeat watchdog (covers Mac crash / USB unplug / app exit)
- Mac sends `ping` (or any command — any valid Mac→ESP32 line resets the watchdog) at
  **≤ `MOTION_HEARTBEAT_MS` (default 150 ms)**.
- ESP32 watchdog window **`MOTION_WATCHDOG_MS` (default 500 ms)** — comfortably ≥ 3×
  heartbeat. No valid Mac line within the window → ESP32 **stops motors**, enters
  `state:"comms_lost"` / `fault:"comms_lost"`, emits a `comms_lost` event.
- Recovery: the next valid Mac line clears `comms_lost` → `idle` (emits `comms_restored`).
  AUTO motion resumes only on an explicit new command, never by resuming a stale setpoint.
- **The watchdog applies in MANUAL too** for *Mac* liveness, but losing the Mac does NOT
  end MANUAL gamepad control — the gamepad is local to the ESP32 (§12). Losing the Mac in
  MANUAL stops *autonomous* authority only; the gamepad keeps driving.

### 7.3 `drive` deadman (covers a stuck/forgotten continuous setpoint)
- A `drive` setpoint **expires `MOTION_DRIVE_EXPIRY_MS` (default 300 ms)** after receipt.
  Without a refreshing `drive`, the base ramps to a controlled stop. `ping` does **not**
  refresh it.
- Finite commands (`turn`/`move`/`come`) are **not** subject to drive-expiry — they run to
  their target — but they **are** subject to the heartbeat watchdog (lose comms mid-turn →
  abort + stop, `done result:"aborted"`).

### 7.4 Finite-command lifecycle
```
Mac → turn(seq=N)        ESP32 → ack(seq=N, accepted)
                         … executes, telemetry state:"moving" …
                         ESP32 → done(seq=N, result:"completed")
```
- Only **one** finite command runs at a time. A new motion command **supersedes** the
  current one (`done result:"superseded"` for the old, `ack` for the new). `stop`/`estop`
  also supersede.
- A reflex STOP/CLIFF mid-finite → `done result:"blocked"`, `state:"blocked"`. Motion in
  that direction is refused until the zone clears; motion *away* is still allowed.

---

## 8. State machine

```
            ┌──────── clear / zone clears ────────┐
            ▼                                      │
  comms_lost ──Mac line──► idle ──motion cmd──► moving ──zone STOP/CLIFF──► blocked
     ▲  (watchdog)          ▲ ▲                   │  │                         │
     │                      │ └──── stop/done ────┘  └── estop ──► estop ◄── estop
  (lose Mac)                │                                         │
                            └────────── clear ◄── (fault) ◄──fault────┘
```
| `state` | Meaning | Accepts motion cmds? |
| --- | --- | --- |
| `idle` | Stopped, ready. | yes |
| `moving` | Executing drive/turn/move/come. | yes (supersedes) |
| `blocked` | Reflex STOP/CLIFF active in travel dir. | only motion *away* from block |
| `estop` | Latched hard stop. | no (until `clear`) |
| `fault` | Latched fault (§9). | no (until resolved + `clear`) |
| `comms_lost` | Watchdog tripped. | resumes to `idle` on next Mac line |

`owner` (`auto`/`manual`) is **orthogonal** to `state` — manual override changes who
commands motion, not the safety state machine (§12).

---

## 9. Enumerations (single source of truth)

| Enum | Values |
| --- | --- |
| `state` | `idle` `moving` `blocked` `estop` `fault` `comms_lost` |
| `owner` | `auto` `manual` |
| `gamepad` | `none` `connected` |
| `fault` | `null` `encoder_stall` `overcurrent` `tof_error` `low_batt` `comms_lost` |
| `zone` | `clear` `slow` `stop` `cliff` |
| `blocked_dir` | `none` `front` `rear` `left` `right` |
| `ack.reason` | `null` `clamped` `manual_override` `estop` `fault` `unknown_cmd` `bad_field` `bad_version` `nothing_to_clear` `unsupported_cap` |
| `done.result` | `completed` `blocked` `aborted` `superseded` `estopped` |
| `event.event` | `boot` `owner_change` `gamepad` `zone_block` `cliff` `fault` `fault_clear` `estop` `estop_clear` `comms_lost` `comms_restored` |
| `tof_mm` keys | `fl` `fr` `rl` `rr` (long VL53L1X front/rear pairs) · `lf` `lb` `rf` `rb` (short VL53L0X left/right pairs) — sensor in error → value `-1` |
| `log.lvl` | `debug` `info` `warn` `error` |

**Zone semantics** (firmware evaluates per control tick on the sensors facing travel):
`clear` > slow_zone · `slow` = stop_zone…slow_zone (scale speed) · `stop` < stop_zone
(halt + refuse that direction) · `cliff` = down-sensor drop-off (the zone enum remains, but
the current 8-sensor radial layout has NO down sensor, so `cliff` is never produced).

---

## 10. Runtime parameters (the `config` command)

These map 1:1 to `config.py` keys on the Mac. **Firmware holds a compiled hard cap for
each; `config` can tighten but never exceed it.** The ack echoes effective values.

| `config` field | `config.py` key | Units | Default | Hard cap (fw) |
| --- | --- | --- | --- | --- |
| `max_lin` | `MOTION_MAX_LINEAR_MS` | m/s | 0.40 | board limit |
| `max_ang` | `MOTION_MAX_ANGULAR_DEG_S` → rad/s | (key is deg/s) | 85 | board limit |

`max_lin`/`max_ang` cap **autonomous** motion only (Mac drive/turn/move/come). MANUAL
gamepad teleop clamps to the firmware's own ceilings (`calib.h GAMEPAD_MAX_LIN_MS` /
`GAMEPAD_MAX_ANG_RADS`, bounded by the hard caps) — so the Mac pushing conservative
autonomous caps no longer slows the human operator.
| `slow_zone_m` | `MOTION_SLOW_ZONE_M` | m | 0.60 | — |
| `stop_zone_m` | `MOTION_STOP_ZONE_M` | m | 0.15 | — |
| `come_stop_at_m` | `MOTION_COME_STOP_AT_M` | m | 0.6 | — |

The host may override `stop_at` per command. The explicit person-seeking voice sequence
uses `MOTION_COME_REQUEST_STOP_AT_M` (1.0 m by default), while spontaneous social
approach retains the shorter `MOTION_COME_STOP_AT_M` default.
| `default_turn_deg` | `MOTION_DEFAULT_TURN_DEG` | deg | 90 | — |
| `default_turn_rate` | `MOTION_DEFAULT_TURN_RATE` | deg/s | 40 | board limit |
| `heartbeat_ms` | `MOTION_HEARTBEAT_MS` | ms | 150 | — (Mac-side send rate) |
| `watchdog_ms` | `MOTION_WATCHDOG_MS` | ms | 500 | firmware-owned |
| `drive_expiry_ms` | `MOTION_DRIVE_EXPIRY_MS` | ms | 300 | firmware-owned |
| `manual_idle_return_secs` | `MOTION_MANUAL_IDLE_RETURN_SECS` | s | 4 | — |
| `manual_autoreturn` | `MOTION_MANUAL_AUTORETURN` | bool | false | — |
| `kp` | `MOTION_WHEEL_KP` | duty per m/s | calib.h | 1e5 |
| `ki` | `MOTION_WHEEL_KI` | duty·s per m/s | calib.h | 1e5 |
| `kd` | `MOTION_WHEEL_KD` | duty·s² per m/s | calib.h | 1e5 |
| `kff` | `MOTION_WHEEL_KFF` | duty per m/s of command | calib.h (~640) | 1e5 |
| `min_duty` | `MOTION_WHEEL_MIN_DUTY` | duty (running floor while rolling) | calib.h (120) | 1023 |
| `breakaway_duty` | `MOTION_WHEEL_BREAKAWAY_DUTY` | duty (stall-gated dead-stop punch, straight drive) | calib.h (358 ≈ 35%) | 1023 |
| `counts_per_meter` | `MOTION_COUNTS_PER_METER` | counts/m | calib.h | 1e3–1e6 |
| `track_width_m` | `MOTION_TRACK_WIDTH_M` | m | calib.h | 0.05–2.0 |

`slow_zone_m` / `stop_zone_m` are the **full-speed envelope**: the firmware scales the
effective zones with measured linear speed, from hard floors at rest (0.10 m stop /
0.18 m slow, `calib.h ZONE_*`) up to the configured values at ~full teleop speed — so a
fast approach brakes early while slow positioning can get close; the stop floor keeps
contact impossible. (Rationale: the ±22.5° pairs see off-path clutter at range.)

Mac-only keys (never sent over the wire): `MOTION_ENABLED` (master switch),
`MOTION_ESP32_PORT` (serial device path — **motion is disabled unless set**, mirroring
`MAESTRO_PORT`), `MOTION_HANDSHAKE_TIMEOUT_MS` (default 1500).

The **drive-tuning keys** (`kp`/`ki`/`kd`, `counts_per_meter`, `track_width_m`) are
runtime-tunable so the base can be calibrated + PID-tuned **live, without a reflash per
iteration** (real-HW build only). The firmware's `calib.h` holds the cold-boot defaults;
the Mac pushes an override **only when the matching `config.py` key is set** (else the
firmware default stands, so a connect never clobbers a bench-tuned value). The config
ack echoes the effective (post-clamp) values. Firmware-only calibration that is *not*
tuned over the wire: `WHEEL_DIAMETER_MM`, `COUNTS_PER_REV`, and the per-wheel
`ENC_SIGN_*` (a one-time wiring fact).

> **Tune geometry at idle.** `kp`/`ki`/`kd` are safe to change while moving (live PID
> tuning). The geometry keys (`counts_per_meter`, `track_width_m`) re-scale odometry
> immediately, so changing them mid-`move`/`turn` re-scales that command's progress —
> change them only when the base is idle. The bench tool enforces this client-side.

---

## 11. Error handling & clamping (decided)

| Situation | Behavior |
| --- | --- |
| Unparseable / non-JSON / missing `v` | Drop silently, `errs`++. |
| Unknown `cmd` | `ack accepted:false reason:"unknown_cmd"`. |
| Missing/wrong-typed required field | `ack accepted:false reason:"bad_field"`. |
| Value over an active cap | **Clamp to cap, `ack accepted:true reason:"clamped"`, execute.** Never reject for magnitude. |
| Unsupported `v` | `ack accepted:false reason:"bad_version"`, no action. |
| Command needs an unadvertised cap | `ack accepted:false reason:"unsupported_cap"`. |
| Motion cmd while `estop`/`fault` | `ack accepted:false reason:"estop"`/`"fault"`. |
| drive/turn/move/come while `owner:"manual"` | `ack accepted:false reason:"manual_override"` (but `stop`/`estop`/`config`/`ping`/`clear`/`batt_full` still accepted). |
| Line > 512 B | Discard through next `\n`, `errs`++. |

**Principle:** the protocol degrades safe and quiet. The only thing that should ever make
the robot *move unexpectedly* is a valid, accepted, in-cap command — everything else
no-ops toward stop.

---

## 12. Manual gamepad override & the protocol

The BT gamepad is paired to the **ESP32**, not the Mac (motion_system.md §11). It produces
**no wire traffic** — the Mac learns about it only by reading telemetry/events:

- Meaningful gamepad input → ESP32 sets `owner:"manual"`, emits `owner_change`, and
  **ignores** Mac drive/turn/move/come (acked `reason:"manual_override"`). It still honors
  `stop`/`estop`/`config`/`ping`/`clear`/`batt_full`.
- The Mac's obligation: when `owner == "manual"`, **stop issuing autonomous motion
  commands** (a voice "come here" is dropped or queued, the controller's choice) and
  surface "MANUAL (gamepad)" + `gamepad` status in the GUI. Keep sending `ping` (heartbeat
  is still required) and `config`.
- Return to AUTO: ESP32-side (explicit toggle, or idle-timeout if `manual_autoreturn`).
  The Mac just observes the `owner_change` back to `auto` and resumes.
- Disconnect failsafe and FULL-OVERRIDE (ToF bypass on a held button) are **entirely
  ESP32-side** and not expressible in this protocol — the Mac only sees their *effects* in
  telemetry (`gamepad:"none"`, motion stopping). The Mac must never assume it can re-enable
  motion the firmware has locked out.

---

## 13. Versioning & forward-compat policy

- **Additive changes stay `v:1`:** new optional fields, new `event`/`fault`/`reason`/`cap`
  enum values, new `tof_mm` keys. Both sides MUST ignore unknown fields and tolerate
  unknown enum values (treat an unknown `state`/`zone` as its safest neighbor — e.g.
  unknown `zone` → behave as `stop`; unknown `fault` → treat as a fault).
- **Breaking changes bump `v`:** removing/renaming a field, changing units or sign
  conventions, changing a required field's type. A `v` bump means coordinated
  firmware+Mac release; the handshake (`proto`) is the gate.
- The Mac advertises the `proto` it speaks in `hello`; the ESP32 advertises its `proto` +
  `caps`. Mismatch → motion disabled, logged, robot otherwise unaffected.

---

## 14. Open points (the only things NOT locked)

These don't block building against v1 — they're behavioral defaults, tunable later — but
flag them as decisions still owed:

1. **`come here` heading source** (motion_system.md §9): face-bearing hint vs straight-
   ahead vs defer. The protocol already carries `heading`; this is a Mac-side policy choice
   about *what to put in it*, not a wire change.
2. **Manual auto-return** default (`MOTION_MANUAL_AUTORETURN`): shipped `false` (explicit
   toggle only) per the spec's "robot never resumes itself" preference. Flip if you prefer
   idle-timeout resume.
3. **Telemetry rate** (10 vs 20 Hz) and **baud** (115200 vs 921600) — start at 20 Hz /
   115200; raise only if the GUI radar view or link health needs it.
4. **`tof_mm` layout** — RESOLVED (rev 2, 2026-07-04) to an 8-sensor radial array,
   every 45° starting 22.5° off the forward axis: 2 long-range VL53L1X FRONT pair
   (`fl,fr`) + 2 REAR pair (`rl,rr`) at ±22.5° off each axis, and 2 short-range VL53L0X
   LEFT pair (`lf,lb`) + 2 RIGHT pair (`rf,rb`), all on a TCA9548A mux (ch 0-3 short,
   4-7 long). The side pairs feed forward hallway steering for manual gamepad drive,
   finite `move`, and the forward phase of `come`.
   This dropped the down-facing cliff sensor — **cliff/drop-off detection is no longer
   available** (the `cliff` zone/event remain in the enum but are never produced).

---

## 15. Implementation checklist (per side)

**Phase 0/1 minimal viable subset** (bench + first floor tests) — implement these first:
- Commands: `hello`, `ping`, `drive`, `stop`, `estop`, `clear`.
- Messages: `hello`, `telemetry`, `ack`, `event`(boot/comms_lost/estop).
- Timers: heartbeat watchdog (§7.2), drive deadman (§7.3).
- `caps:["drive","stop"]`.

**Mac side** (`hardware/motion.py` / `motion_controller.py`):
- [ ] `connect()` → open serial, send `hello`, await `hello`/timeout → bool (logs
      `Motion base: enabled/disabled` in main.py Step-4).
- [ ] Background reader thread: parse NDJSON, drop-on-error, keep latest-telemetry
      snapshot (thread-safe, servo-pattern), route `ack`/`done`/`event`/`log`.
- [ ] Heartbeat thread: `ping` every `MOTION_HEARTBEAT_MS`.
- [ ] `seq` allocator; `ack`/`done` correlation with timeouts.
- [ ] Send helpers: `drive/turn/move/come/stop/estop/clear/config` with clamping mirror.
- [ ] Policy gates: suppress AUTO motion while `owner=="manual"`, while
      `INTERACTION_PAUSED`/family-safe, or `MOTION_ENABLED` off.
- [ ] Unit tests against a **mocked serial** (no hardware), mirroring the servo tests.

**ESP32 side** (firmware):
- [ ] NDJSON line reader (≤512 B, drop-on-overflow, `errs` counter).
- [ ] Command dispatch + `ack` per §11 table.
- [ ] Telemetry emitter at the configured rate with the full §6.1 schema.
- [ ] Heartbeat watchdog + drive deadman + finite-command lifecycle (`done`).
- [ ] Boots to safe idle before handshake; reflex STOP/CLIFF independent of the Mac.
- [ ] `boot_id` on every boot; emit `boot` event.

---

## Appendix A — `protocol.h` sketch (firmware) ↔ Python constants

Keep these two in lockstep with §9. Suggested shared definitions:

```c
// protocol.h  (ESP32)
#define MOTION_PROTO_VERSION 1
// states
enum MotionState { ST_IDLE, ST_MOVING, ST_BLOCKED, ST_ESTOP, ST_FAULT, ST_COMMS_LOST };
// faults
enum MotionFault { F_NONE, F_ENCODER_STALL, F_OVERCURRENT, F_TOF_ERROR, F_LOW_BATT, F_COMMS_LOST };
// zones
enum MotionZone  { Z_CLEAR, Z_SLOW, Z_STOP, Z_CLIFF };
// timers (ms) — defaults; config may tighten
#define MOTION_WATCHDOG_MS    500
#define MOTION_DRIVE_EXPIRY_MS 300
```

```python
# (Mac) e.g. intelligence/motion_controller.py constants — must match protocol.h / §9
MOTION_PROTO_VERSION = 1
MOTION_STATES  = ("idle","moving","blocked","estop","fault","comms_lost")
MOTION_FAULTS  = (None,"encoder_stall","overcurrent","tof_error","low_batt","comms_lost")
MOTION_ZONES   = ("clear","slow","stop","cliff")
ACK_REASONS    = ("clamped","manual_override","estop","fault","unknown_cmd",
                  "bad_field","bad_version","nothing_to_clear","unsupported_cap")
DONE_RESULTS   = ("completed","blocked","aborted","superseded","estopped")
```

## Appendix B — Worked example session

```
# bring-up
Mac  → {"v":1,"cmd":"hello","seq":1,"host":"djr3x","proto":1}
ESP32→ {"v":1,"type":"hello","proto":1,"fw":"0.3.1","caps":["drive","turn","move","stop"],"boot_id":7741}
ESP32→ {"v":1,"type":"event","t":12,"event":"boot","boot_id":7741,"fw":"0.3.1"}
ESP32→ {"v":1,"type":"telemetry","t":50,"state":"idle","owner":"auto","gamepad":"none","fault":null,"zone":"clear","blocked_dir":"none","cmd_seq":1,"odom":{"x":0,"y":0,"theta":0,"lin":0,"ang":0},"tof_mm":{"fl":1200,"fr":1150,"rl":1800,"rr":1600,"lf":900,"lb":950,"rf":1500,"rb":1450},"batt_mv":12010,"errs":0}

# spoken "turn left" → 90° CCW
Mac  → {"v":1,"cmd":"turn","seq":2,"deg":90,"rate":40}
ESP32→ {"v":1,"type":"ack","seq":2,"accepted":true,"reason":null}
       … telemetry state:"moving" …
ESP32→ {"v":1,"type":"done","seq":2,"result":"completed","odom":{"x":0,"y":0,"theta":1.571}}

# heartbeat throughout
Mac  → {"v":1,"cmd":"ping","seq":3}        (every 150 ms; not acked)

# spoken "come here" but someone grabs the gamepad mid-move
Mac  → {"v":1,"cmd":"come","seq":4,"heading":0,"stop_at":0.6}
ESP32→ {"v":1,"type":"ack","seq":4,"accepted":true,"reason":null}
ESP32→ {"v":1,"type":"event","t":18420,"event":"owner_change","owner":"manual"}
ESP32→ {"v":1,"type":"done","seq":4,"result":"superseded","odom":{...}}   # gamepad took over
Mac  → {"v":1,"cmd":"drive","seq":5,"lin":0.1,"ang":0}
ESP32→ {"v":1,"type":"ack","seq":5,"accepted":false,"reason":"manual_override"}
       … Mac stops sending AUTO motion, keeps pinging, shows "MANUAL (gamepad)" …
```
