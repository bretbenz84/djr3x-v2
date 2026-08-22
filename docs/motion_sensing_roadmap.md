# Motion Sensing Roadmap — room scan, IMU fusion, outdoor heading/GPS

**Status: DESIGN / BRAINSTORM (2026-07-11). Nothing in this document is built.**
It captures the feasibility analysis and phasing for three related ideas so they
survive across sessions: (1) a ToF **room boundary scan** ("poor-man's LiDAR"),
(2) **IMU fusion** for heading truth + interoception, and (3) **outdoor
direction sense** ("go north") from a compass + GPS. Read alongside
[motion_system.md](motion_system.md) (the built drive base) and
[motion_protocol.md](motion_protocol.md) (the wire contract).

---

## 1. What exists today (the substrate)

- ESP32 drive base: encoder-closed velocity PID, odometry `(x, y, theta)`
  integrated from wheel encoders, finite `turn`/`move`/`come`, gamepad manual
  override, ToF stop reflex + hallway assist
  ([firmware/djr3x_motion](../firmware/djr3x_motion/)).
- 8 radial ToF sensors on a TCA9548A mux: 4× VL53L1X long (~4 m) straddling the
  fore/aft axes (`fl fr rl rr`, ±22.5°), 4× VL53L0X short (~1.2 m) on the sides
  (`lf lb rf rb`). Each sensor refreshes every ~80 ms (round-robin 2/tick at the
  50 Hz sensor task).
- Telemetry at 20 Hz already carries `odom` (incl. `theta`) + `tof_mm` + battery
  to the Mac over USB serial.
- I2C trunk on GPIO 21/22 (mux 0x70, sensors 0x29 behind it, INA226 0x40).

### Calibration debt (blocks everything below)

`calib.h` geometry is still **placeholder**: `TRACK_WIDTH_M 0.200` (real base
ring is 540 mm), `WHEEL_DIAMETER_M 0.080`, `COUNTS_PER_REV_OUTPUT` unverified.
Until the docs §14 pass is done (drive a measured 1 m; spin a measured 360°),
encoder headings and distances are systematically wrong — every feature in this
roadmap inherits that error. **Phase 0 = calibrate.**

---

## 2. Spin-geometry constraint (measure `d` first)

> **Spinning in place is NOT footprint-safe on this base.** The drive axle sits
> **aft of the ring's geometric center** (drive wheels rear, omniwheels front),
> so a pure spin rotates about the axle midpoint and sweeps the FRONT of the
> ring out wider than the parked footprint.

With `d` = axle-to-ring-center offset (⚠ **not yet measured**):

- Swept envelope = circle of radius `(270 mm + d)` about the axle midpoint.
- Worst case is a **rear** obstacle: rear edge parks at `270 − d` from the axle
  but the front swings out to `270 + d` — so rear clearance must exceed **~2d**
  (+ margin) or the front clips it coming around.

Consequences:

1. `d` becomes a calibration constant (belongs in `calib.h` next to
   `track_width_m`), used by both the scan pre-flight and the point-cloud math.
2. Any large ungated spin (D-pad turns, voice "turn left", realign-to-face,
   scan sweeps) in tight quarters can clip obstacles. **Swing check — SHIPPED
   2026-08-22** (`intelligence/motion_swing.py`, wired into
   `motion_controller.turn()/come()/arc()`): after Rex twice lost his left hand
   turning left a foot from a bookshelf. Every radial return is re-expressed about
   the axle, every body extent (ring sampled at 45° plus the arm tips from
   `MOTION_BODY_EXTENTS`) is swept through the requested angle, and the turn is
   shrunk to the largest clear angle or refused under `MOTION_SWING_MIN_TURN_DEG`.
   `d` is `MOTION_AXLE_AFT_OF_CENTER_M` (0.23 m by eye — **still unmeasured**, as
   are the arm reaches). D-pad turns remain ungated (manual).
3. Sensor mount transforms must be expressed relative to the **axle midpoint**
   (that is what odometry tracks), with the ring center at `+d` forward in the
   body frame.

---

## 3. Room boundary scan (ToF + rotation → point map)

**Idea:** spin the base slowly through 360° while logging
`(theta, sensor bearing, distance)`; convert polar→cartesian to get a
point-ring of the room's motion boundary at ring height.

**Feasibility verdict: YES, as a coarse boundary map** — ±few cm range, ±2–5°
bearing after calibration + loop closure, corners rounded by sensor FOV. Not a
floor-plan-grade scan and never will be. Genuinely useful anyway (see §3.5).

### 3.1 Why the hardware suits it

- A 360° spin gives **4 redundant full sweeps** (each long sensor traces the
  whole room) → median-of-4 per bearing rejects glitches for free. Even a 180°
  spin covers all bearings via the front+rear pairs combined.
- Sampling density at a slow ~15°/s scan spin: one sample per ~1.2° per sensor,
  ≳1,000 unique points per sweep, ~24 s per scan.
- The side L0X sensors ride along free — a denser, more accurate short-range
  (<1.2 m) ring in the same sweep.

### 3.2 The three hard problems (priority order)

1. **Heading truth** — the map's bearing accuracy IS odometry accuracy.
   Uncalibrated `track_width_m` + wheel scrub on carpet (5–15% heading error
   over a spin) dominate. Fixes: Phase 0 calibration; **gyro fusion** (§4 —
   measures true rotation independent of wheel slip); **loop closure** — a 360°
   profile must match itself at 0°/360°, so cross-correlate the first/last ~20°
   and distribute the residual. Loop closure also doubles as a calibration
   quality metric.
2. **Timestamp alignment** — telemetry `theta` is fresh but each `tof_mm` value
   is up to ~80+ ms stale (round-robin), an unknown-per-sensor smear of
   ~1.2–2.5° at 15°/s. Acceptable for v1; the real fix is firmware stamping
   `odom.theta` at the moment of each ToF read (`scan_pt` records).
3. **FOV cone blur** — the L1X reports the NEAREST return in a ~27° cone
   (~1.9 m wide at 4 m): corners round off, near furniture "pulls walls in",
   and **doorways vanish beyond ~1.5–2 m** (the cone sees the frame on both
   sides). The Pololu driver supports ROI narrowing to ~15–20° — enable in a
   scan-mode config. Map quality: good inside ~2 m, coarse beyond.

### 3.3 Geometry (do not skip)

`point = R(θ_odom + bearing) · mount_offset + distance · ray(θ_odom + bearing)`
with `mount_offset` relative to the **axle midpoint** (ring center at `+d`
forward, sensors on the 270 mm ring). Irrelevant at 3 m, a 60%+ error at
0.4 m if ignored. During the sweep the sensors *orbit* the axle — the transform
handles it exactly because rotation about the axle is what the encoders
measure.

### 3.4 Scan phases

| Phase | What | Where | Notes |
| --- | --- | --- | --- |
| **0** | Calibrate `counts_per_meter`, `track_width_m`; **measure `d`** | bench | prerequisite for everything |
| **1** | Proof of concept: Mac commands a slow 360 (existing `turn`), logs existing 20 Hz telemetry, builds the polar plot (GUI or matplotlib). **Zero firmware change.** Includes the **scan pre-flight**: all 8 sensors must read ≳ 2d + margin (rear pair worst-cased) before spinning; on failure refuse, partial-sweep, or nudge forward first (the forward nudge IS ToF-gated) | host only | accept ~2° smear |
| **2** | Accuracy: firmware `scan_pt` records (theta stamped at read time); loop-closure correction; median-of-4 fusion; optional L1X ROI narrowing in scan mode | fw + host | |
| **3** | Product: map persisted per scan pose; GUI room-map panel (extends the ToF radar); motion planner clamps `move`/`come` against known boundaries (covers the blind wedges between cones); Rex commentary + episodic memory ("scanned the room — about 4 by 5 meters, mostly couch") | both | |
| **4** | Stretch: re-scan correlation to re-localize after driving; multi-pose stitching | host | ambitious; drift-limited |

**Design position:** each map is a **boundary snapshot anchored to the scan
pose**, not a persistent world model. Once the base translates, encoder drift
corrupts registration; re-scanning (~25 s) is cheaper than maintaining a
drifting map.

### 3.5 What it buys

Memory of walls the reflex can't currently see (blind wedges between cones);
path sanity for `come`/approach; room-size awareness grounded in real
measurement (persona gold); an episodic memory entry per scan.

---

## 4. Phase A — MPU-6050 IMU (GY-521) — **indoor, build first**

Owned hardware. 6-axis (3-axis gyro + 3-axis accel), I2C addr **0x68**, on the
existing trunk (GPIO 21/22) — no address conflicts (see §7). Mount near the
axle midpoint, away from motor cables. Old part; fine. Needs a rest-time gyro
bias calibration at boot (the base boots idle, so this is natural).

**The gyro is the single most valuable sensor in this roadmap**: it measures
true rotation independent of wheel slip, directly fixing the scan's #1 problem
and improving every turn (D-pad nudges, voice turns, realign-to-face).
Fusion: complementary filter — gyro yaw for short-term truth, encoders for
translation, loop closure (scans) for absolute correction. Lives on the ESP32
next to the 100 Hz control loop.

**Do NOT dead-reckon position from the accelerometer** — double-integrated
consumer MEMS drifts meters within seconds. The accel's actual jobs:

- **Tilt e-stop** — tall top-heavy droid; cut motors past a tilt threshold.
  Belongs beside the watchdog in `safety.cpp` (fail-safe with USB unplugged).
- **Bump/collision reflex** — impact spikes catch low obstacles in the ToF
  blind wedges.
- **Stuck/slip detection** — encoders say "moving", IMU says "we're not" →
  carpet-stuck fault instead of confidently wrong odometry.
- **Interoception/persona** — Rex feels being picked up, carried, shaken,
  tilted, and reacts in character (fits the `awareness/` interoception layer).

---

## 5. Phase B — QMC5883L compass (GY-271) — outdoor heading

> ⚠ **The GY-271 is a QMC5883L, NOT an HMC5883L.** Different register map,
> different I2C address (**0x0D**, not the HMC's 0x1E), different libraries.
> Every HMC tutorial will fail on this module — use QMC-specific code.

Purpose: absolute heading for **outdoor** use ("face north", "go south").
Indoors, magnetometers are unusable-to-misleading here (rebar, steel furniture,
and the BTS7960s pumping tens of amps of *dynamic* field that hard/soft-iron
calibration cannot remove) — **indoor yaw stays gyro+encoder+loop-closure; the
mag is ignored indoors.**

Build constraints:

- **Mount as far from motors/H-bridges/battery leads as possible** — up in the
  body or head; distance is the only real fix for dynamic fields.
- Hard-iron calibration routine (figure-8 / in-place spin), plus a motors-on
  vs motors-off offset check to quantify residual motor interference.
- Tilt compensation uses the MPU-6050 accel (owned) → full tilt-compensated
  heading. Outdoor accuracy ±2–5° is realistic after calibration.

Command layer: "go north" = the L1+D-pad absolute-heading turn generalized to
the **world frame** — turn-to-heading anchored by mag-fused yaw instead of the
boot frame, then a ToF-gated `move`. Conceptually small once the heading stack
exists.

---

## 6. Phase C — NEO-6M GPS (GY-NEO6MV2) — outdoor waypoints

> ⚠ **The NEO-6M is UART, not I2C** — NMEA sentences at 9600 baud default. It
> needs a spare ESP32 UART routed to free GPIOs (4/5/15 are free per `pins.h`;
> avoid 16/17 — the pins that degraded the left PWM). RX-only suffices to read
> fixes; TX optional for module config. Ceramic antenna needs sky view — mount
> high.

Reality: ~2.5 m CEP on a good day, 1–5 Hz. Useless indoors (no fix) and useless
for precise paths. Right scale split for "go north 10 meters":

- **Compass** picks the direction,
- **encoders + gyro** measure the distance,
- **GPS** only anchors long traverses (>5–10 m), waypoint-ish navigation,
  "return to where we started", and episodic memory of outdoor excursions.

### Outdoor safety caveat (applies to B and C)

**Obstacle avoidance degrades exactly when terrain risk rises**: VL53L1X range
collapses in direct sunlight (ambient IR floods the SPADs — 4 m indoors can
become <1 m in sun), there is no cliff sensing (porch steps, curbs), and the
omniwheels are untested on grass/gravel. Outdoor mode launches as **slow,
supervised, open-terrain** driving.

---

## 7. I2C / pin plumbing (combined)

All I2C devices share the existing trunk (GPIO 21 SDA / 22 SCL). Address map —
no conflicts:

| Addr | Device | Notes |
| --- | --- | --- |
| 0x0D | QMC5883L compass | Phase B; trunk (NOT behind the mux) |
| 0x29 | VL53L0X/L1X ×8 | behind the mux only — never seen on the trunk |
| 0x40 | INA226 battery | existing |
| 0x68 | MPU-6050 IMU | Phase A; trunk |
| 0x70 | TCA9548A mux | existing |

GPS: UART (see §6), not on this bus.

---

## 8. Build order (each step independently useful)

1. **Phase 0** — calibrate `counts_per_meter` + `track_width_m`; **measure `d`**
   (axle line → ring center) and add it to `calib.h`.
2. **Phase A** — MPU-6050 on the trunk: gyro-fused heading, tilt e-stop,
   bump/pickup detection. Pays off indoors immediately.
3. **Scan Phase 1** — host-only proof-of-concept sweep + pre-flight clearance
   check. Validates the whole scan idea in an afternoon.
4. **Scan Phase 2–3** — `scan_pt` stamping, loop closure, GUI map, planner
   integration.
5. **Phase B** — compass, calibration routine, "face north".
6. **Phase C** — GPS waypoints, return-home.

Related memory notes: `drive-base-spin-geometry`, `motion-firmware-phase0`,
`autonomous-motion-v1`.
